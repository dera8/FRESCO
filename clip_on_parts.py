#!/usr/bin/env python3
"""
Classify material for each building part using CLIP zero-shot classification.
"""

import argparse
import pandas as pd
from PIL import Image
import numpy as np
import torch
import open_clip
from tqdm import tqdm


# Material prompts for different parts
MATERIAL_PROMPTS = {
    'wall': [
        'a wooden wall',
        'a brick wall',
        'a plaster wall',
        'a concrete wall',
        'a stone wall',
        'a metal wall',
        'a glass wall',
    ],
    'window': [
        'a glass window',
        'a wooden window frame',
        'a metal window frame',
        'a plastic window',
    ],
    'door': [
        'a wooden door',
        'a metal door',
        'a glass door',
        'a plastic door',
    ]
}

# Simplified material labels (remove "a" and "wall"/"window"/"door")
def simplify_label(prompt):
    """Extract material name from prompt."""
    prompt = prompt.replace('a ', '').replace('an ', '')
    for part in ['wall', 'window', 'door', 'frame']:
        prompt = prompt.replace(f' {part}', '')
    return prompt.strip()


def classify_parts(parts_index_path, output_path, device='cpu', 
                   conf_threshold=0.3, margin_threshold=0.1):
    """
    Classify materials for all parts using CLIP.
    
    Args:
        parts_index_path: Path to parts_index.csv
        output_path: Output CSV for predictions
        device: 'cpu' or 'cuda'
        conf_threshold: Minimum confidence for non-unknown classification
        margin_threshold: Minimum margin between top-2 predictions
    """
    
    print(f"Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # Load parts index
    parts_df = pd.read_csv(parts_index_path)
    
    print(f"Classifying {len(parts_df)} parts...")
    
    predictions = []
    
    for idx, row in tqdm(parts_df.iterrows(), total=len(parts_df)):
        image_id = row['image_id']
        part = row['part']
        img_path = row['path']
        
        # Get prompts for this part type
        prompts = MATERIAL_PROMPTS.get(part, MATERIAL_PROMPTS['wall'])
        
        # Load and preprocess image
        try:
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            predictions.append({
                'image_id': image_id,
                'part': part,
                'label': 'unknown',
                'confidence': 0.0,
                'margin': 0.0,
                'path': img_path
            })
            continue
        
        # Tokenize text prompts
        text_inputs = tokenizer(prompts).to(device)
        
        # Get CLIP embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(2)
        
        # Get top prediction
        top_conf = values[0].item()
        top_idx = indices[0].item()
        top_label = simplify_label(prompts[top_idx])
        
        # Calculate margin (difference between top-2)
        if len(values) > 1:
            margin = values[0].item() - values[1].item()
        else:
            margin = top_conf
        
        # Apply thresholds
        if top_conf < conf_threshold or margin < margin_threshold:
            final_label = 'unknown'
        else:
            final_label = top_label
        
        predictions.append({
            'image_id': image_id,
            'part': part,
            'label': final_label,
            'confidence': top_conf,
            'margin': margin,
            'path': img_path
        })
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    
    print(f"\nClassification complete:")
    print(f"  Total predictions: {len(pred_df)}")
    print(f"  Unknown: {len(pred_df[pred_df['label'] == 'unknown'])}")
    print(f"  Output saved to: {output_path}")
    
    # Show material distribution
    print("\nMaterial distribution:")
    for material in pred_df['label'].value_counts().head(10).items():
        print(f"  {material[0]}: {material[1]}")


def main():
    parser = argparse.ArgumentParser(
        description='Classify materials using CLIP zero-shot'
    )
    parser.add_argument('--parts_index', required=True, help='Parts index CSV')
    parser.add_argument('--out', required=True, help='Output predictions CSV')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                       help='Minimum confidence threshold (default: 0.3)')
    parser.add_argument('--margin_threshold', type=float, default=0.1,
                       help='Minimum margin between top-2 (default: 0.1)')
    
    args = parser.parse_args()
    
    classify_parts(args.parts_index, args.out, args.device, 
                  args.conf_threshold, args.margin_threshold)


if __name__ == '__main__':
    main()
