#!/usr/bin/env python3
"""
Assign images to buildings based on camera position and viewing direction.
"""

import argparse
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import numpy as np


def compute_view_ray(lat, lon, heading, distance=50):
    """
    Compute a view ray from camera position and heading.
    
    Args:
        lat, lon: Camera position
        heading: Compass heading in degrees (0 = North, 90 = East)
        distance: Ray length in meters
    
    Returns:
        LineString representing the view ray
    """
    # Approximate conversion: 1 degree lat ~ 111km, lon varies by latitude
    lat_per_m = 1.0 / 111000.0
    lon_per_m = 1.0 / (111000.0 * np.cos(np.radians(lat)))
    
    # Convert heading to radians (0° = North, clockwise)
    heading_rad = np.radians(heading)
    
    # Calculate endpoint (North is up in our coordinate system)
    dx = distance * np.sin(heading_rad)  # East component
    dy = distance * np.cos(heading_rad)  # North component
    
    end_lon = lon + dx * lon_per_m
    end_lat = lat + dy * lat_per_m
    
    return LineString([(lon, lat), (end_lon, end_lat)])


def assign_images_to_buildings(buildings_path, manifest_path, output_path, max_distance=100):
    """
    Assign each image to the nearest building in its viewing direction.
    
    Args:
        buildings_path: Path to buildings GeoJSON
        manifest_path: Path to photos manifest CSV (image_id, lat, lon, heading)
        output_path: Path for output CSV
        max_distance: Maximum distance in meters to consider
    """
    
    print(f"Loading buildings from {buildings_path}")
    buildings_gdf = gpd.read_file(buildings_path)
    
    print(f"Loading photo manifest from {manifest_path}")
    photos_df = pd.read_csv(manifest_path)
    
    # Validate required columns
    required_cols = ['image_id', 'lat', 'lon', 'heading']
    missing = [col for col in required_cols if col not in photos_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")
    
    results = []
    
    for idx, row in photos_df.iterrows():
        image_id = row['image_id']
        lat = row['lat']
        lon = row['lon']
        heading = row['heading']
        
        # Create view ray
        view_ray = compute_view_ray(lat, lon, heading, distance=max_distance)
        
        # Find buildings that intersect with view ray
        intersecting = buildings_gdf[buildings_gdf.intersects(view_ray)]
        
        if len(intersecting) > 0:
            # Calculate distance to camera for each intersecting building
            camera_point = Point(lon, lat)
            
            distances = []
            for _, building in intersecting.iterrows():
                nearest_pt = nearest_points(camera_point, building.geometry)[1]
                dist = camera_point.distance(nearest_pt) * 111000  # Rough conversion to meters
                distances.append(dist)
            
            # Get closest building
            closest_idx = np.argmin(distances)
            closest_building = intersecting.iloc[closest_idx]
            
            results.append({
                'image_id': image_id,
                'building_id': closest_building['building_id'],
                'distance_m': distances[closest_idx],
                'building_type': closest_building.get('building_type', 'unknown')
            })
        else:
            # No intersection - try nearest building within max_distance
            camera_point = Point(lon, lat)
            buildings_gdf['distance'] = buildings_gdf.geometry.apply(
                lambda geom: camera_point.distance(geom) * 111000
            )
            
            nearest = buildings_gdf[buildings_gdf['distance'] < max_distance].sort_values('distance')
            
            if len(nearest) > 0:
                closest = nearest.iloc[0]
                results.append({
                    'image_id': image_id,
                    'building_id': closest['building_id'],
                    'distance_m': closest['distance'],
                    'building_type': closest.get('building_type', 'unknown')
                })
            else:
                results.append({
                    'image_id': image_id,
                    'building_id': None,
                    'distance_m': None,
                    'building_type': None
                })
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    assigned = output_df['building_id'].notna().sum()
    print(f"\nAssignment complete:")
    print(f"  Total images: {len(output_df)}")
    print(f"  Assigned to buildings: {assigned}")
    print(f"  Unassigned: {len(output_df) - assigned}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Assign images to buildings based on camera position and view direction'
    )
    parser.add_argument('--geo', required=True, help='Buildings GeoJSON file')
    parser.add_argument('--photos_manifest', required=True, 
                       help='Photos manifest CSV (image_id, lat, lon, heading)')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--max_distance', type=float, default=100,
                       help='Maximum distance in meters (default: 100)')
    
    args = parser.parse_args()
    
    assign_images_to_buildings(args.geo, args.photos_manifest, args.out, args.max_distance)


if __name__ == '__main__':
    main()
