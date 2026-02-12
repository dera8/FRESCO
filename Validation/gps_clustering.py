#!/usr/bin/env python3
"""
GPS-Based Photo Clustering for Cross-View Stability
====================================================

Given a set of Mapillary photos with GPS metadata, clusters them into
"pseudo-buildings" using spatial proximity. This enables cross-view
stability evaluation even without building-level annotations.

Input:
  - Mapillary photos with GPS metadata (lat/lon)
  
Output:
  - Building clusters: building_id → [list of image_ids]
  - Enables stability validation across viewpoints

Usage:
  python gps_clustering.py \
    --photos data_input/bryggen/photos_manifest.json \
    --output data_output/bryggen/building_clusters.json \
    --radius 20  # meters
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN

@dataclass
class Photo:
    image_id: str
    lat: float
    lon: float
    compass_angle: float = 0.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance in meters between two GPS coordinates.
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def degrees_to_meters_approx(delta_lat: float, delta_lon: float, avg_lat: float) -> Tuple[float, float]:
    """
    Approximate conversion from degrees to meters at given latitude.
    
    Returns:
        (delta_lat_meters, delta_lon_meters)
    """
    # 1 degree latitude ≈ 111,000 meters everywhere
    lat_meters = delta_lat * 111000
    
    # 1 degree longitude varies with latitude
    lon_meters = delta_lon * 111000 * np.cos(np.radians(avg_lat))
    
    return lat_meters, lon_meters

def load_mapillary_manifest(manifest_path: Path) -> List[Photo]:
    """
    Load photo metadata from Mapillary manifest.
    
    Expected structure:
    {
      "features": [
        {
          "id": "966039691808614",
          "geometry": {
            "coordinates": [lon, lat]
          },
          "properties": {
            "compass_angle": 123.4
          }
        }
      ]
    }
    
    Alternative structure (if manifest is list of dicts):
    [
      {
        "id": "966039691808614",
        "lat": 60.3967,
        "lon": 5.3245,
        "compass_angle": 123.4
      }
    ]
    """
    with open(manifest_path) as f:
        data = json.load(f)
    
    photos = []
    
    # Handle GeoJSON format
    if "features" in data:
        for feat in data["features"]:
            image_id = feat["id"]
            lon, lat = feat["geometry"]["coordinates"]
            compass_angle = feat.get("properties", {}).get("compass_angle", 0.0)
            
            photos.append(Photo(image_id, lat, lon, compass_angle))
    
    # Handle simple list format
    elif isinstance(data, list):
        for item in data:
            image_id = item.get("photo_id") or item.get("id")

            lat = item["lat"]
            lon = item["lon"]
            compass_angle = item.get("compass_angle", 0.0)
            
            photos.append(Photo(image_id, lat, lon, compass_angle))
    
    # Handle dict format (image_id → metadata)
    elif isinstance(data, dict):
        for image_id, meta in data.items():
            lat = meta["lat"]
            lon = meta["lon"]
            compass_angle = meta.get("compass_angle", 0.0)
            
            photos.append(Photo(image_id, lat, lon, compass_angle))
    
    else:
        raise ValueError(f"Unknown manifest format: {type(data)}")
    
    return photos

def cluster_photos_gps(photos: List[Photo], radius_meters: float = 20.0, min_samples: int = 3) -> Dict[int, List[str]]:
    """
    Cluster photos by GPS proximity using DBSCAN.
    
    Args:
        photos: List of photos with GPS coords
        radius_meters: Clustering radius in meters
        min_samples: Minimum photos per cluster
    
    Returns:
        {cluster_id: [image_id, image_id, ...]}
    """
    if not photos:
        return {}
    
    # Extract coordinates
    coords = np.array([[p.lat, p.lon] for p in photos])
    
    # Compute average latitude for degree→meter conversion
    avg_lat = np.mean(coords[:, 0])
    
    # Convert radius in meters to degrees (approximate)
    # At latitude, 1 degree ≈ 111km, but longitude varies
    # Use conservative estimate
    radius_deg_lat = radius_meters / 111000
    radius_deg_lon = radius_meters / (111000 * np.cos(np.radians(avg_lat)))
    
    # Use average for eps (DBSCAN expects single value)
    eps = (radius_deg_lat + radius_deg_lon) / 2
    
    print(f"Clustering with eps={eps:.6f} degrees (~{radius_meters}m at lat {avg_lat:.2f}°)")
    
    # Cluster
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(coords)
    
    # Group by cluster
    clusters = {}
    noise_count = 0
    
    for i, label in enumerate(clustering.labels_):
        if label == -1:  # noise
            noise_count += 1
            continue
        
        if label not in clusters:
            clusters[label] = []
        
        clusters[label].append(photos[i].image_id)
    
    print(f"Found {len(clusters)} clusters, {noise_count} noise photos")
    
    # Statistics
    cluster_sizes = [len(imgs) for imgs in clusters.values()]
    if cluster_sizes:
        print(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
    
    return clusters

def filter_small_clusters(clusters: Dict[int, List[str]], min_views: int = 3) -> Dict[int, List[str]]:
    """
    Remove clusters with fewer than min_views photos.
    
    For stability validation, we need at least 2 views (preferably 3+).
    """
    filtered = {
        cid: imgs for cid, imgs in clusters.items()
        if len(imgs) >= min_views
    }
    
    removed = len(clusters) - len(filtered)
    print(f"Filtered {removed} clusters with <{min_views} views")
    
    return filtered

def save_clusters(clusters: Dict[int, List[str]], output_path: Path):
    """
    Save clusters to JSON.
    
    Output format:
    {
      "n_clusters": 342,
      "avg_views_per_cluster": 3.7,
      "clusters": {
        "building_0": ["img1", "img2", "img3"],
        "building_1": ["img4", "img5"],
        ...
      }
    }
    """
    # Convert cluster IDs to building_N format
    building_clusters = {
        f"building_{cid}": imgs
        for cid, imgs in clusters.items()
    }
    
    # Statistics
    cluster_sizes = [len(imgs) for imgs in building_clusters.values()]
    
    output = {
        "n_clusters": len(building_clusters),
        "avg_views_per_cluster": np.mean(cluster_sizes) if cluster_sizes else 0.0,
        "median_views_per_cluster": np.median(cluster_sizes) if cluster_sizes else 0.0,
        "min_views": min(cluster_sizes) if cluster_sizes else 0,
        "max_views": max(cluster_sizes) if cluster_sizes else 0,
        "clusters": building_clusters
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved {len(building_clusters)} clusters to {output_path}")
    print(f"  Average views per cluster: {output['avg_views_per_cluster']:.1f}")
    print(f"  Median views per cluster: {output['median_views_per_cluster']:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Cluster photos by GPS proximity")
    parser.add_argument("--photos", required=True, help="Path to Mapillary photos manifest (JSON)")
    parser.add_argument("--output", required=True, help="Output path for clusters (JSON)")
    parser.add_argument("--radius", type=float, default=20.0, help="Clustering radius in meters (default: 20)")
    parser.add_argument("--min-samples", type=int, default=3, help="Min photos per cluster (default: 3)")
    parser.add_argument("--min-views", type=int, default=3, help="Min views to keep cluster (default: 3)")
    
    args = parser.parse_args()
    
    # Load photos
    print(f"Loading photos from {args.photos}...")
    photos = load_mapillary_manifest(Path(args.photos))
    print(f"Loaded {len(photos)} photos")
    
    if len(photos) < args.min_samples:
        print(f"Not enough photos for clustering (need >={args.min_samples})")
        return
    
    # Cluster
    print(f"\nClustering with radius={args.radius}m, min_samples={args.min_samples}...")
    clusters = cluster_photos_gps(photos, radius_meters=args.radius, min_samples=args.min_samples)
    
    # Filter small clusters
    clusters = filter_small_clusters(clusters, min_views=args.min_views)
    
    if not clusters:
        print("No valid clusters found. Try increasing --radius or decreasing --min-views")
        return
    
    # Save
    save_clusters(clusters, Path(args.output))

if __name__ == "__main__":
    main()
