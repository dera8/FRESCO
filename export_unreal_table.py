#!/usr/bin/env python3
"""
Export material assignments in Unreal Engine compatible format.
"""

import argparse
import pandas as pd


# Mapping from material names to Unreal Material Instance names
MATERIAL_MAPPING = {
    'wood': 'MI_Wood_01',
    'wooden': 'MI_Wood_01',
    'brick': 'MI_Brick_Red',
    'plaster': 'MI_Plaster_White',
    'concrete': 'MI_Concrete_Gray',
    'stone': 'MI_Stone_Gray',
    'metal': 'MI_Metal_Brushed',
    'glass': 'MI_Glass_Clear',
    'plastic': 'MI_Plastic_White',
    'unknown': 'MI_Default_Gray'
}


def export_for_unreal(input_path, output_path, mapping=None):
    """
    Convert material assignments to Unreal-compatible format.
    
    Args:
        input_path: Input materials CSV
        output_path: Output Unreal assignments CSV
        mapping: Optional custom material mapping dict
    """
    
    if mapping is None:
        mapping = MATERIAL_MAPPING
    
    # Load data
    df = pd.read_csv(input_path)
    
    unreal_assignments = []
    
    for _, row in df.iterrows():
        building_id = row['building_id']
        
        assignment = {'building_id': building_id}
        
        # Map each part to Unreal material instance
        for part in ['wall', 'window', 'door']:
            col = f'{part}_material'
            if col in df.columns:
                material = row[col]
                # Map to Unreal MI name
                mi_name = mapping.get(material, mapping.get('unknown', 'MI_Default_Gray'))
                assignment[part] = mi_name
        
        unreal_assignments.append(assignment)
    
    # Create output dataframe
    output_df = pd.DataFrame(unreal_assignments)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"Unreal export complete:")
    print(f"  Total buildings: {len(output_df)}")
    print(f"  Output saved to: {output_path}")
    print(f"\nSample assignments:")
    print(output_df.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Export materials for Unreal Engine'
    )
    parser.add_argument('--in', dest='input', required=True, help='Input materials CSV')
    parser.add_argument('--out', required=True, help='Output Unreal assignments CSV')
    
    args = parser.parse_args()
    
    export_for_unreal(args.input, args.out)


if __name__ == '__main__':
    main()
