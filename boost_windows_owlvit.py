#!/usr/bin/env python3
"""
Boost window detection using OWL-ViT zero-shot object detector.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from tqdm import tqdm


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Apply NMS to remove overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        ious = [compute_iou(boxes[current], boxes[i]) for i in indices[1:]]
        indices = indices[1:][np.array(ious) < iou_threshold]
    
    return keep


def box_to_mask(box, image_shape):
    """Convert bounding box to binary mask."""
    mask = np.zeros(image_shape[:2], dtype=bool)
    x1, y1, x2, y2 = map(int, box)
    mask[y1:y2, x1:x2] = True
    return mask


def mask_to_rgba(mask, original_image):
    """Convert binary mask to RGBA PNG."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, :3] = original_image[mask, :3]
    rgba[mask, 3] = 255
    return rgba


def boost_windows(photos_dir, output_dir, parts_index_path, device='cuda', 
                 max_side=1024, score_thr=0.25, iou_wall_thr=0.5, 
                 nms_thr=0.5, skip_if_has_windows=True):
    """
    Detect windows using OWL-ViT and add to parts index.
    
    Args:
        photos_dir: Directory with original photos
        output_dir: Directory with existing parts (will add new windows)
        parts_index_path: Path to parts_index.csv
        device: 'cpu' or 'cuda'
        max_side: Resize images to this max dimension
        score_thr: Minimum confidence score
        iou_wall_thr: Minimum IoU with wall mask
        nms_thr: NMS IoU threshold
        skip_if_has_windows: Skip images that already have windows
    """
    
    print(f"Loading OWL-ViT model...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model = model.to(device)
    model.eval()
    
    # Load existing parts index
    parts_df = pd.read_csv(parts_index_path)
    
    # Get unique images
    images_with_walls = parts_df[parts_df['part'] == 'wall']['image_id'].unique()
    images_with_windows = parts_df[parts_df['part'] == 'window']['image_id'].unique()
    
    # Filter images to process
    if skip_if_has_windows:
        images_to_process = [img for img in images_with_walls if img not in images_with_windows]
        print(f"Processing {len(images_to_process)} images without windows (skipping {len(images_with_windows)} with windows)")
    else:
        images_to_process = list(images_with_walls)
        print(f"Processing {len(images_to_process)} images with wall masks")
    
    new_windows = []
    
    for image_id in tqdm(images_to_process):
        # Load original image
        photo_files = list(Path(photos_dir).glob(f"{image_id}.*"))
        if len(photo_files) == 0:
            continue
        
        photo_path = photo_files[0]
        image = Image.open(photo_path).convert('RGB')
        
        # Resize for faster processing
        orig_w, orig_h = image.size
        scale = max_side / max(orig_w, orig_h)
        if scale < 1:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image_resized = image.resize((new_w, new_h))
        else:
            image_resized = image
            scale = 1.0
        
        # Load wall mask for ROI
        wall_mask_path = parts_df[(parts_df['image_id'] == image_id) & 
                                   (parts_df['part'] == 'wall')]['path'].values[0]
        wall_mask = np.array(Image.open(wall_mask_path))[:, :, 3] > 0  # Use alpha channel
        
        # Detect windows
        text_queries = ["a window", "a glass window", "building window"]
        inputs = processor(text=text_queries, images=image_resized, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        target_sizes = torch.tensor([image_resized.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=score_thr, target_sizes=target_sizes
        )[0]
        
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        
        if len(boxes) == 0:
            continue
        
        # Scale boxes back to original size
        boxes = boxes / scale
        
        # Filter boxes that overlap with wall mask
        filtered_boxes = []
        filtered_scores = []
        
        for box, score in zip(boxes, scores):
            box_mask = box_to_mask(box, wall_mask.shape)
            iou_with_wall = (box_mask & wall_mask).sum() / box_mask.sum()
            
            if iou_with_wall >= iou_wall_thr:
                filtered_boxes.append(box)
                filtered_scores.append(score)
        
        if len(filtered_boxes) == 0:
            continue
        
        # Apply NMS
        keep_indices = non_max_suppression(filtered_boxes, filtered_scores, nms_thr)
        
        # Save detected windows
        image_np = np.array(image)
        total_pixels = image_np.shape[0] * image_np.shape[1]
        
        for idx, box_idx in enumerate(keep_indices):
            box = filtered_boxes[box_idx]
            score = filtered_scores[box_idx]
            
            # Create mask from box
            window_mask = box_to_mask(box, image_np.shape)
            area_ratio = window_mask.sum() / total_pixels
            
            # Save as RGBA
            rgba_mask = mask_to_rgba(window_mask, image_np)
            mask_path = os.path.join(output_dir, f"{image_id}_window_det_{idx:02d}.png")
            Image.fromarray(rgba_mask).save(mask_path)
            
            # Add to new windows list
            new_windows.append({
                'image_id': image_id,
                'part': 'window',
                'path': mask_path,
                'area_ratio': area_ratio,
                'detection_score': score
            })
    
    # Append to parts index
    if len(new_windows) > 0:
        new_df = pd.DataFrame(new_windows)
        updated_df = pd.concat([parts_df, new_df], ignore_index=True)
        updated_df.to_csv(parts_index_path, index=False)
        
        print(f"\nWindow detection complete:")
        print(f"  New windows detected: {len(new_windows)}")
        print(f"  Updated parts index: {parts_index_path}")
    else:
        print("\nNo new windows detected")


def main():
    parser = argparse.ArgumentParser(
        description='Boost window detection using OWL-ViT'
    )
    parser.add_argument('--photos', required=True, help='Photos directory')
    parser.add_argument('--outdir', required=True, help='Parts output directory')
    parser.add_argument('--parts_index', required=True, help='Parts index CSV')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--max_side', type=int, default=1024)
    parser.add_argument('--score_thr', type=float, default=0.25)
    parser.add_argument('--iou_wall_thr', type=float, default=0.5)
    parser.add_argument('--nms_thr', type=float, default=0.5)
    parser.add_argument('--skip_if_has_windows', action='store_true', default=True)
    
    args = parser.parse_args()
    
    boost_windows(args.photos, args.outdir, args.parts_index, args.device,
                 args.max_side, args.score_thr, args.iou_wall_thr, 
                 args.nms_thr, args.skip_if_has_windows)


if __name__ == '__main__':
    main()
