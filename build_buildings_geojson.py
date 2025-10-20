#!/usr/bin/env python3
"""
Extract building footprints from OSM data and export to GeoJSON.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
import json


def parse_osm_to_geojson(osm_path, output_path):
    """Parse OSM XML and extract building polygons to GeoJSON."""
    
    print(f"Parsing OSM file: {osm_path}")
    tree = ET.parse(osm_path)
    root = tree.getroot()
    
    # First pass: collect all nodes with their coordinates
    nodes = {}
    for node in root.findall('node'):
        node_id = node.get('id')
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        nodes[node_id] = (lon, lat)  # GeoJSON uses lon, lat order
    
    # Second pass: collect ways that are buildings
    buildings = []
    building_id = 1
    
    for way in root.findall('way'):
        tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
        
        # Check if this is a building
        if 'building' in tags:
            # Get node references
            nd_refs = [nd.get('ref') for nd in way.findall('nd')]
            
            # Build coordinate list
            coords = []
            for ref in nd_refs:
                if ref in nodes:
                    coords.append(list(nodes[ref]))
            
            # Valid polygon needs at least 4 points (closing point is duplicate)
            if len(coords) >= 4:
                building_type = tags.get('building', 'yes')
                roof_material = tags.get('roof:material', None)
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "building_id": f"building_{building_id:05d}",
                        "building_type": building_type,
                        "osm_id": way.get('id')
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                if roof_material:
                    feature["properties"]["roof_material"] = roof_material
                
                buildings.append(feature)
                building_id += 1
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": buildings
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Extracted {len(buildings)} buildings to {output_path}")
    return len(buildings)


def main():
    parser = argparse.ArgumentParser(
        description='Extract building footprints from OSM data'
    )
    parser.add_argument('--osm', required=True, help='Input OSM XML file')
    parser.add_argument('--out', required=True, help='Output GeoJSON file')
    
    args = parser.parse_args()
    
    parse_osm_to_geojson(args.osm, args.out)


if __name__ == '__main__':
    main()
