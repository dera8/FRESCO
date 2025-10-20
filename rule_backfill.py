#!/usr/bin/env python3
"""
Apply rule-based fallbacks for unknown materials.
"""

import argparse
import pandas as pd
import yaml


DEFAULT_RULES = {
    'window': {
        'unknown': 'glass',
        'priority': ['glass', 'metal', 'wood']
    },
    'door': {
        'unknown': 'wood',
        'priority': ['wood', 'metal', 'glass']
    },
    'wall': {
        'unknown': 'plaster',
        'priority': ['wood', 'brick', 'plaster', 'concrete', 'stone']
    }
}


def apply_fallback_rules(input_path, output_path, rules_path=None):
    """
    Apply fallback rules to fill unknown materials.
    
    Args:
        input_path: Input CSV with materials
        output_path: Output CSV with filled materials
        rules_path: Optional YAML file with custom rules
    """
    
    # Load rules
    if rules_path:
        print(f"Loading rules from {rules_path}")
        with open(rules_path, 'r') as f:
            rules = yaml.safe_load(f)
    else:
        print("Using default fallback rules")
        rules = DEFAULT_RULES
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Add source column to track where material came from
    for part in ['wall', 'window', 'door']:
        col = f'{part}_material'
        source_col = f'{part}_source'
        
        if col in df.columns:
            # Initialize source as 'clip' for non-unknown values
            df[source_col] = df[col].apply(lambda x: 'clip' if x != 'unknown' else 'unknown')
            
            # Apply fallback rules
            if part in rules:
                fallback = rules[part].get('unknown', 'unknown')
                mask = df[col] == 'unknown'
                df.loc[mask, col] = fallback
                df.loc[mask, source_col] = 'rule'
    
    # Save output
    df.to_csv(output_path, index=False)
    
    print(f"\nFallback application complete:")
    print(f"  Output saved to: {output_path}")
    
    # Show statistics
    for part in ['wall', 'window', 'door']:
        col = f'{part}_material'
        source_col = f'{part}_source'
        
        if col in df.columns and source_col in df.columns:
            clip_count = (df[source_col] == 'clip').sum()
            rule_count = (df[source_col] == 'rule').sum()
            unknown_count = (df[col] == 'unknown').sum()
            
            print(f"\n{part.capitalize()}:")
            print(f"  From CLIP: {clip_count}")
            print(f"  From rules: {rule_count}")
            print(f"  Still unknown: {unknown_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Apply fallback rules for unknown materials'
    )
    parser.add_argument('--in', dest='input', required=True, help='Input materials CSV')
    parser.add_argument('--out', required=True, help='Output CSV')
    parser.add_argument('--rules', help='Optional rules YAML file')
    
    args = parser.parse_args()
    
    apply_fallback_rules(args.input, args.out, args.rules)


if __name__ == '__main__':
    main()
