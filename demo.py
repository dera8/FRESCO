#!/usr/bin/env python3
"""
AutoPBR Pipeline Demo - Validates installation and shows pipeline structure
"""

import sys
import os

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("AutoPBR Pipeline - Installation Verification")
    print("=" * 60)
    print()
    
    print("Checking Python version...")
    print(f"  Python {sys.version}")
    print()
    
    print("Checking core dependencies...")
    core_deps = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
    ]
    
    all_ok = True
    for module, name in core_deps:
        if not check_import(module, name):
            all_ok = False
    
    print()
    print("Checking deep learning frameworks...")
    dl_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("timm", "TIMM"),
        ("open_clip", "OpenCLIP"),
    ]
    
    for module, name in dl_deps:
        if not check_import(module, name):
            all_ok = False
    
    print()
    print("Checking geospatial libraries...")
    geo_deps = [
        ("shapely", "Shapely"),
        ("geopandas", "GeoPandas"),
        ("pyproj", "PyProj"),
    ]
    
    for module, name in geo_deps:
        if not check_import(module, name):
            all_ok = False
    
    print()
    print("Checking utilities...")
    util_deps = [
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
        ("requests", "Requests"),
    ]
    
    for module, name in util_deps:
        if not check_import(module, name):
            all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ All dependencies installed successfully!")
        print()
        print("Pipeline Structure:")
        print("  1. build_buildings_geojson.py    - Extract buildings from OSM")
        print("  2. assign_images_to_buildings.py - Associate images to buildings")
        print("  3. seg_parts_hf.py               - Segment parts (SegFormer)")
        print("  4. boost_windows_owlvit.py       - Boost windows (OWL-ViT)")
        print("  5. clip_on_parts.py              - Classify materials (CLIP)")
        print("  6. aggregate_parts.py            - Aggregate to buildings")
        print("  7. rule_backfill.py              - Apply fallback rules")
        print("  8. export_unreal_table.py        - Export for Unreal Engine")
        print("  9. export_urmat_metadata.py      - Export for UR-MAT")
        print()
        print("Ready to process street imagery! See README.md for usage.")
    else:
        print("✗ Some dependencies are missing!")
        print()
        print("Install missing packages with:")
        print("  pip install -r requirements.txt")
        print()
        print("For PyTorch with CUDA support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
