#!/usr/bin/env python3
"""
Aggregate part-level material predictions to building level with multi-view fusion.
"""

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


def weighted_vote(labels, confidences, distances=None):
    """
    Aggregate predictions using weighted voting.
    
    Args:
        labels: List of material labels
        confidences: List of confidence scores
        distances: Optional list of camera-to-building distances
    
    Returns:
        (best_label, avg_confidence, num_votes)
    """
    
    if len(labels) == 0:
        return 'unknown', 0.0, 0
    
    # Calculate weights
    weights = np.array(confidences)
    
    if distances is not None:
        # Inverse distance weighting (closer views are more reliable)
        dist_weights = 1.0 / (np.array(distances) + 1.0)
        weights = weights * dist_weights
    
    # Aggregate by label
    label_scores = defaultdict(float)
    for label, weight in zip(labels, weights):
        label_scores[label] += weight
    
    # Get best label
    best_label = max(label_scores.items(), key=lambda x: x[1])[0]
    
    # Calculate average confidence for best label
    best_confidences = [conf for lab, conf in zip(labels, confidences) if lab == best_label]
    avg_confidence = np.mean(best_confidences) if best_confidences else 0.0
    
    return best_label, avg_confidence, len(labels)


def aggregate_to_buildings(preds_path, links_path, output_path):
    """
    Aggregate part predictions to building level.
    
    Args:
        preds_path: Path to material predictions CSV
        links_path: Path to image-to-building links CSV
        output_path: Path for aggregated output CSV
    """
    
    # Load data
    print(f"Loading predictions from {preds_path}")
    preds_df = pd.read_csv(preds_path)
    
    print(f"Loading image-building links from {links_path}")
    links_df = pd.read_csv(links_path)
    
    # Merge predictions with building links
    merged_df = preds_df.merge(links_df, on='image_id', how='left')
    
    # Filter out images without building assignment
    merged_df = merged_df[merged_df['building_id'].notna()]
    
    print(f"Aggregating {len(merged_df)} predictions for buildings...")
    
    # Group by building and part type
    building_materials = []
    
    for building_id in merged_df['building_id'].unique():
        building_data = merged_df[merged_df['building_id'] == building_id]
        
        result = {
            'building_id': building_id,
            'building_type': building_data['building_type'].iloc[0] if 'building_type' in building_data.columns else 'unknown',
            'n_images': len(building_data['image_id'].unique())
        }
        
        # Aggregate each part type
        for part in ['wall', 'window', 'door']:
            part_data = building_data[building_data['part'] == part]
            
            if len(part_data) > 0:
                labels = part_data['label'].tolist()
                confidences = part_data['confidence'].tolist()
                distances = part_data['distance_m'].tolist() if 'distance_m' in part_data.columns else None
                
                best_label, avg_conf, n_votes = weighted_vote(labels, confidences, distances)
                
                result[f'{part}_material'] = best_label
                result[f'{part}_conf'] = avg_conf
                result[f'n_votes_{part}'] = n_votes
            else:
                result[f'{part}_material'] = 'unknown'
                result[f'{part}_conf'] = 0.0
                result[f'n_votes_{part}'] = 0
        
        building_materials.append(result)
    
    # Create output dataframe
    output_df = pd.DataFrame(building_materials)
    
    # Sort by building_id
    output_df = output_df.sort_values('building_id')
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"\nAggregation complete:")
    print(f"  Total buildings: {len(output_df)}")
    print(f"  Buildings with wall material: {(output_df['wall_material'] != 'unknown').sum()}")
    print(f"  Buildings with window material: {(output_df['window_material'] != 'unknown').sum()}")
    print(f"  Buildings with door material: {(output_df['door_material'] != 'unknown').sum()}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate part predictions to building level'
    )
    parser.add_argument('--preds', required=True, help='Material predictions CSV')
    parser.add_argument('--links', required=True, help='Image-to-building links CSV')
    parser.add_argument('--out', required=True, help='Output aggregated CSV')
    
    args = parser.parse_args()
    
    aggregate_to_buildings(args.preds, args.links, args.out)


if __name__ == '__main__':
    main()
