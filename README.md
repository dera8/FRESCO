# AutoPBR Pipeline

**Automatic PBR Material Assignment for Buildings from Street-Level Imagery**

A research pipeline for extracting building parts (walls, windows, doors) from street photos and automatically assigning PBR (Physically Based Rendering) materials using deep learning. Designed for 3D reconstruction, urban simulation, and game engine integration.

## Overview

AutoPBR processes street-level photographs (e.g., from Mapillary) through a multi-stage pipeline:

1. **Geo-processing**: Associate images with building footprints using OSM data
2. **Segmentation**: Extract building parts using SegFormer (walls, doors, windows)
3. **Detection Boost**: Enhance window detection with OWL-ViT zero-shot detector
4. **Classification**: Assign materials using CLIP zero-shot classification
5. **Aggregation**: Fuse multi-view predictions per building with confidence weighting
6. **Fallback**: Apply rule-based defaults for unknown materials
7. **Export**: Generate assignments for Unreal Engine, UR-MAT simulations, etc.

## Pipeline Architecture

```
Street Photos → Segmentation → Detection → Classification → Aggregation → Export
                 (SegFormer)   (OWL-ViT)     (CLIP)       (Multi-view)   (UE/JSON)
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd bryggen_autopbr
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
# PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# All other dependencies
pip install -r requirements.txt
```

## Quick Start

### Minimal Example (Photo → Materials)

```bash
# 1. Place your photos in the photos/ directory
cp /path/to/photos/*.jpg photos/

# 2. Run segmentation
python seg_parts_hf.py --photos photos --outdir parts --device cpu

# 3. Boost window detection (optional, requires GPU for speed)
python boost_windows_owlvit.py --photos photos --outdir parts \
  --parts_index parts/parts_index.csv --device cpu

# 4. Classify materials
python clip_on_parts.py --parts_index parts/parts_index.csv \
  --out mat_preds_parts.csv --device cpu

# 5. Export for Unreal Engine
python export_unreal_table.py --in mat_preds_parts.csv \
  --out outputs/autopbr_assignments.csv
```

### Full Pipeline with Building Association

```bash
# 1. Extract buildings from OSM
python build_buildings_geojson.py --osm map.osm --out buildings.geojson

# 2. Associate images with buildings
python assign_images_to_buildings.py \
  --geo buildings.geojson \
  --photos_manifest photos_manifest.csv \
  --out image_to_building.csv

# 3. Segment parts
python seg_parts_hf.py --photos photos --outdir parts --device cpu

# 4. Boost windows
python boost_windows_owlvit.py --photos photos --outdir parts \
  --parts_index parts/parts_index.csv --device cuda

# 5. Classify materials
python clip_on_parts.py --parts_index parts/parts_index.csv \
  --out mat_preds_parts.csv --device cpu

# 6. Aggregate to building level
python aggregate_parts.py \
  --preds mat_preds_parts.csv \
  --links image_to_building.csv \
  --out outputs/materials_by_building.csv

# 7. Apply fallback rules
python rule_backfill.py \
  --in outputs/materials_by_building.csv \
  --out outputs/materials_final.csv \
  --rules configs/rules.yaml

# 8. Export for Unreal Engine
python export_unreal_table.py \
  --in outputs/materials_final.csv \
  --out outputs/autopbr_assignments.csv

# 9. Export for UR-MAT simulation
python export_urmat_metadata.py \
  --in outputs/materials_final.csv \
  --out outputs/urmat_materials.json
```

## Input Data Formats

### Photos Manifest CSV
Required columns: `image_id`, `lat`, `lon`, `heading`

```csv
image_id,lat,lon,heading
img_001,60.3975,5.3237,90.5
img_002,60.3976,5.3238,85.2
```

### Buildings GeoJSON
Standard GeoJSON with building footprints and `building_id` property.

## Output Formats

### Unreal Engine CSV
```csv
building_id,wall,window,door
building_00001,MI_Wood_01,MI_Glass_Clear,MI_Wood_01
building_00002,MI_Brick_Red,MI_Glass_Clear,MI_Metal_Brushed
```

### UR-MAT JSON
```json
{
  "building_id": "building_00001",
  "parts": [
    {
      "part_type": "wall",
      "material": "wood",
      "confidence": 0.85,
      "em_properties": {
        "relative_permittivity": 2.0,
        "conductivity": 0.0001,
        "attenuation_db_per_m": 0.5
      }
    }
  ]
}
```

## Configuration

Edit `configs/autopbr_config.yaml` to customize:
- Model devices (CPU/GPU)
- Detection thresholds
- Confidence/margin thresholds
- Export formats

Edit `configs/rules.yaml` to customize fallback rules for unknown materials.

## Troubleshooting

### Missing Windows
- Lower `--score_thr` in `boost_windows_owlvit.py` (try 0.20-0.25)
- Adjust `--iou_wall_thr` to reduce false positives

### Confused Materials (plaster vs stone)
- Use supervised probe instead of zero-shot CLIP (future feature)
- Adjust confidence/margin thresholds

### GPU Memory Issues
- Reduce `--max_side` in window detection (try 768 or 512)
- Use `--device cpu` for models

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{autopbr2025,
  title={AutoPBR: Automatic PBR Material Assignment from Street Imagery},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/autopbr}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- SegFormer models from NVIDIA/HuggingFace
- OWL-ViT from Google Research
- OpenCLIP from LAION
- Built for Bryggen digital twin project
