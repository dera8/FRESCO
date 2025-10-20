#!/usr/bin/env python3
"""
Export material metadata for UR-MAT simulation system with EM properties.
"""

import argparse
import json
import pandas as pd


# Material electromagnetic properties for UR-MAT
EM_PROPERTIES = {
    'wood': {
        'relative_permittivity': 2.0,
        'conductivity': 0.0001,
        'attenuation_db_per_m': 0.5
    },
    'brick': {
        'relative_permittivity': 4.5,
        'conductivity': 0.01,
        'attenuation_db_per_m': 3.0
    },
    'plaster': {
        'relative_permittivity': 2.5,
        'conductivity': 0.001,
        'attenuation_db_per_m': 1.0
    },
    'concrete': {
        'relative_permittivity': 6.0,
        'conductivity': 0.02,
        'attenuation_db_per_m': 6.0
    },
    'stone': {
        'relative_permittivity': 7.0,
        'conductivity': 0.01,
        'attenuation_db_per_m': 5.0
    },
    'metal': {
        'relative_permittivity': 1.0,
        'conductivity': 10000000,
        'attenuation_db_per_m': 100.0
    },
    'glass': {
        'relative_permittivity': 6.5,
        'conductivity': 0.0001,
        'attenuation_db_per_m': 2.0
    },
    'plastic': {
        'relative_permittivity': 2.8,
        'conductivity': 0.0001,
        'attenuation_db_per_m': 0.8
    },
    'unknown': {
        'relative_permittivity': 3.0,
        'conductivity': 0.01,
        'attenuation_db_per_m': 2.0
    }
}


def export_urmat(input_path, output_path):
    """
    Export materials with EM properties for UR-MAT system.
    
    Args:
        input_path: Input materials CSV
        output_path: Output JSON file
    """
    
    # Load data
    df = pd.read_csv(input_path)
    
    urmat_data = []
    
    for _, row in df.iterrows():
        building_id = row['building_id']
        
        building_materials = {
            'building_id': building_id,
            'parts': []
        }
        
        # Add each part with EM properties
        for part in ['wall', 'window', 'door']:
            col = f'{part}_material'
            conf_col = f'{part}_conf'
            
            if col in df.columns:
                material = row[col]
                confidence = row.get(conf_col, 0.0)
                
                em_props = EM_PROPERTIES.get(material, EM_PROPERTIES['unknown'])
                
                part_data = {
                    'part_type': part,
                    'material': material,
                    'confidence': float(confidence),
                    'em_properties': em_props
                }
                
                building_materials['parts'].append(part_data)
        
        urmat_data.append(building_materials)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(urmat_data, f, indent=2)
    
    print(f"UR-MAT export complete:")
    print(f"  Total buildings: {len(urmat_data)}")
    print(f"  Output saved to: {output_path}")
    print(f"\nSample entry:")
    if urmat_data:
        print(json.dumps(urmat_data[0], indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Export materials for UR-MAT simulation'
    )
    parser.add_argument('--in', dest='input', required=True, help='Input materials CSV')
    parser.add_argument('--out', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    export_urmat(args.input, args.out)


if __name__ == '__main__':
    main()
