#!/usr/bin/env python3
"""
Segment building parts (wall, door, window) using SegFormer from HuggingFace.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm


# ADE20K class indices (SegFormer trained on ADE20K)
ADE20K_CLASSES = {
    'building': 1,
    'door': 14,
    'window': 8,
    'roof': 5,
    'wall': 0
}


def extract_part_mask(seg_mask, part_type, image_shape):
    """Extract binary mask for specific part type."""
    
    if part_type == 'wall_derived':
        # Wall = building - (door + window + roof)
        building_mask = (seg_mask == ADE20K_CLASSES['building'])
        door_mask = (seg_mask == ADE20K_CLASSES['door'])
        window_mask = (seg_mask == ADE20K_CLASSES['window'])
        roof_mask = (seg_mask == ADE20K_CLASSES['roof'])
        
        wall_mask = building_mask & ~(door_mask | window_mask | roof_mask)
        return wall_mask
    
    elif part_type in ADE20K_CLASSES:
        return seg_mask == ADE20K_CLASSES[part_type]
    
    return np.zeros(image_shape[:2], dtype=bool)


def mask_to_rgba(mask, original_image):
    """Convert binary mask to RGBA PNG with original image colors."""
    
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Copy RGB from original where mask is True
    rgba[mask, :3] = original_image[mask, :3]
    # Set alpha channel
    rgba[mask, 3] = 255
    
    return rgba


def segment_parts(photos_dir, output_dir, device='cpu', window_min=0.001, door_min=0.002):
    """
    Segment building parts from photos using SegFormer.
    
    Args:
        photos_dir: Directory containing input photos
        output_dir: Directory for output masks
        device: 'cpu' or 'cuda'
        window_min: Minimum area ratio for window detection
        door_min: Minimum area ratio for door detection
    """
    
    print(f"Loading SegFormer model...")
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    photo_files = list(Path(photos_dir).glob('*.jpg')) + list(Path(photos_dir).glob('*.png'))
    
    if len(photo_files) == 0:
        print(f"No images found in {photos_dir}")
        return
    
    print(f"Processing {len(photo_files)} images...")
    
    parts_index = []
    
    for photo_path in tqdm(photo_files):
        image_id = photo_path.stem
        
        # Load image
        image = Image.open(photo_path).convert('RGB')
        image_np = np.array(image)
        
        # Prepare for model
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run segmentation
        with torch.no_grad():
            outputs = model(**inputs)
            seg_mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()
        
        # Resize mask to original image size
        seg_mask = np.array(Image.fromarray(seg_mask.astype(np.uint8)).resize(
            image.size, Image.NEAREST
        ))
        
        # Extract parts
        parts_to_extract = [
            ('wall', extract_part_mask(seg_mask, 'wall_derived', image_np.shape)),
            ('door', extract_part_mask(seg_mask, 'door', image_np.shape)),
            ('window', extract_part_mask(seg_mask, 'window', image_np.shape)),
        ]
        
        total_pixels = image_np.shape[0] * image_np.shape[1]
        
        for part_name, mask in parts_to_extract:
            area_ratio = mask.sum() / total_pixels
            
            # Skip if too small
            if part_name == 'window' and area_ratio < window_min:
                continue
            if part_name == 'door' and area_ratio < door_min:
                continue
            if part_name == 'wall' and area_ratio < 0.01:  # Wall must be at least 1%
                continue
            
            # Create RGBA mask
            rgba_mask = mask_to_rgba(mask, image_np)
            
            # Save mask
            mask_path = os.path.join(output_dir, f"{image_id}_{part_name}.png")
            Image.fromarray(rgba_mask).save(mask_path)
            
            # Add to index
            parts_index.append({
                'image_id': image_id,
                'part': part_name,
                'path': mask_path,
                'area_ratio': area_ratio
            })
    
    # Save parts index
    index_df = pd.DataFrame(parts_index)
    index_path = os.path.join(output_dir, 'parts_index.csv')
    index_df.to_csv(index_path, index=False)
    
    print(f"\nSegmentation complete:")
    print(f"  Total parts extracted: {len(parts_index)}")
    print(f"  Parts index saved to: {index_path}")
    
    # Summary by part type
    if len(parts_index) > 0:
        print("\nParts summary:")
        for part in index_df['part'].unique():
            count = len(index_df[index_df['part'] == part])
            print(f"  {part}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Segment building parts using SegFormer'
    )
    parser.add_argument('--photos', required=True, help='Photos directory')
    parser.add_argument('--outdir', required=True, help='Output directory for masks')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to use')
    parser.add_argument('--window_min', type=float, default=0.001,
                       help='Minimum area ratio for window (default: 0.001)')
    parser.add_argument('--door_min', type=float, default=0.002,
                       help='Minimum area ratio for door (default: 0.002)')
    
    args = parser.parse_args()
    
    segment_parts(args.photos, args.outdir, args.device, args.window_min, args.door_min)


if __name__ == '__main__':
    main()
