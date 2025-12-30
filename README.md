# AutoPBR v2 Pipeline
**Automatic PBR Material Assignment for Buildings from Street-Level Imagery**

A research pipeline for extracting building parts (walls, roofs, roads) from street photos and automatically assigning PBR (Physically Based Rendering) materials using deep learning with hierarchical segmentation. Designed for 3D reconstruction, urban simulation, and game engine integration.

---

## Overview

AutoPBR v2 processes street-level photographs through an intelligent multi-stage pipeline:

```
Street Photos → SAM3 Segmentation → Instance Detection → Material Analysis → Export
                (Semantic Classes)   (Per Building)      (VLM + Context)    (UE/JSON)
```

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Semantic Segmentation (SAM3)                          │
│ RGB → wall.png, roof.png, road.png                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 2: Instance Segmentation (Per Building)                  │
│ RGB + Semantic Masks → building00_wall, building01_wall, ...   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 3: Hierarchical Composites                               │
│ building00_composite.png (road=1, wall=2, roof=3)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ MATERIAL ANALYSIS: Intelligent VLM Analysis                     │
│ • 1 building  → Simple (RGB ⊙ mask_class)                      │
│ • N buildings → Per-building (RGB ⊙ composite)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ EXPORT: Unreal Engine, UR-MAT, PBR Pipelines                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dera8/AutoPBR.git
cd AutoPBR
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Transformers from GitHub (required for SAM3)
pip install git+https://github.com/huggingface/transformers.git

# Other dependencies
pip install -r requirements.txt
```

### 4. Setup API Keys
```bash
# For NVIDIA Nemotron VLM
export NVIDIA_API_KEY="your_nvidia_api_key"

# For Mapillary (optional, if downloading new photos)
export MAPILLARY_TOKEN="your_mapillary_token"
```

### 5. HuggingFace Login (for SAM3 access)
```bash
huggingface-cli login
# Request access at: https://huggingface.co/facebook/sam3
```

---

## Quick Start

### Minimal Example (Photo → Materials)

```bash
# 1. Place photos in photos2/ directory
mkdir -p photos2
cp /path/to/photos/*.jpg photos2/

# 2. Run semantic segmentation (Level 1)
python 2_sam3hi.py

# 3. Run instance segmentation (Level 2 + 3)
python sam3hi.py --all

# 4. Analyze materials with VLM
python nemotron_hierarchical.py

# Output: materials_hybrid.json with per-building materials
```

### Output Structure
```
sam3_semantic2/              # Level 1: Semantic masks
├── photo_001_wall.png
├── photo_001_roof.png
└── photo_001_road.png

sam3_instance_per_building/  # Level 2: Instance masks
├── photo_001_wall_building00.png
├── photo_001_wall_building01.png
├── photo_001_roof_building00.png
└── photo_001_roof_building01.png

sam3_hierarchical_final/     # Level 3: Composites
├── photo_001_building00_composite.png
├── photo_001_building01_composite.png
└── manifest_hierarchical.json

materials_hybrid.json        # Final materials per building
```

---

## Full Pipeline with Geographic Matching

For street-view photos with GPS coordinates (e.g., Mapillary):

```bash
# 1. Download OSM data and Mapillary photos
python step1_download_data.py --bbox "5.32,60.397,5.326,60.400"

# 2. Match photos to buildings geometrically
python 4_step2_perception.py

# 3. Semantic segmentation
python 2_sam3hi.py

# 4. Instance segmentation
python sam3hi.py --all

# 5. Material analysis (uses matches.json)
python nemotron_hierarchical.py

# 6. Export for Unreal Engine
python export_unreal_table.py \
  --in materials_hybrid.json \
  --out outputs/autopbr_assignments.csv
```

---

## Configuration Files

### Script Parameters

**`2_sam3hi.py` (Level 1 Semantic)**
```python
IMAGES_DIR = Path("photos2")
OUT_DIR = Path("sam3_semantic2")
PROMPTS = {
    "road": "road",
    "roof": "roof", 
    "wall": "wall"
}
SCORE_THR = 0.5
MASK_THR = 0.5
```

**`sam3hi.py` (Level 2+3 Instance)**
```python
MIN_BUILDING_AREA = 100  # Minimum pixels for a building
IOU_THRESHOLD = 0.3      # For matching wall-roof pairs
```

**`nemotron_hierarchical.py` (Material Analysis)**
```python
MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
```

---

## Output Formats

### Materials JSON (`materials_hybrid.json`)
```json
{
  "results": [
    {
      "photo_id": "101989905547842",
      "num_buildings": 5,
      "approach": "building_specific",
      "buildings": [
        {
          "building_idx": 0,
          "classes": {
            "wall": {
              "success": true,
              "data": {
                "material": ["painted wooden planks", "wood siding"],
                "color": ["red ochre", "dark red"],
                "pattern": ["vertical planks"],
                "condition": "weathered",
                "confidence": "high",
                "distinguishing_features": "darker than neighbors"
              },
              "coverage": 35.2
            },
            "roof": {
              "success": true,
              "data": {
                "material": ["clay tiles", "ceramic tiles"],
                "color": ["terracotta", "red-brown"],
                "pattern": ["overlapping tiles"],
                "confidence": "high"
              },
              "coverage": 18.7
            }
          }
        }
      ]
    }
  ],
  "statistics": {
    "total_images": 150,
    "total_buildings": 342,
    "per_class": {
      "wall": {"total": 342, "successful": 318, "rate": "93.0%"},
      "roof": {"total": 342, "successful": 325, "rate": "95.0%"},
      "road": {"total": 150, "successful": 145, "rate": "96.7%"}
    }
  }
}
```

### Unreal Engine CSV
```csv
building_id,wall_material,wall_color,roof_material,roof_color
building_00001,MI_Wood_Painted,Red_Ochre,MI_Tile_Clay,Terracotta
building_00002,MI_Wood_Painted,White,MI_Slate,Dark_Grey
building_00003,MI_Brick_Old,Red_Brown,MI_Metal_Sheet,Grey
```


---

## Advanced Usage

### Custom Class Prompts
Edit `2_sam3hi.py` to add new classes:
```python
PROMPTS = {
    "road": "road pavement ground",
    "wall": "building wall facade",
    "roof": "building roof top",
    "window": "window glass opening",  # New class
    "door": "door entrance opening"     # New class
}
```

### Adjust Detection Sensitivity
```python
# In sam3hi.py
MIN_BUILDING_AREA = 50   # Lower = detect smaller buildings
SCORE_THR = 0.4          # Lower = more detections (less precise)
MASK_THR = 0.4           # Lower = larger masks
```

### Custom Material Prompts
Edit prompts in `nemotron_hierarchical.py`:
```python
CLASS_CONTEXTS = {
    "wall": """
    Your custom context here...
    Expected materials: brick, stone, wood, concrete
    Common patterns: textured, smooth, painted
    """,
}
```

---

## Troubleshooting

### ❌ SAM3 Import Error
```bash
pip install git+https://github.com/huggingface/transformers.git
huggingface-cli login
# Request access: https://huggingface.co/facebook/sam3
```

### ⚠️ Missing Buildings in Instance Segmentation
- Check `sam3_semantic2/` has masks for wall and roof
- Lower `MIN_BUILDING_AREA` in `sam3hi.py`
- Verify semantic masks have sufficient coverage (>0.1%)

### 🐌 Slow Processing
```python
# Use CPU if GPU memory limited
device = "cpu"

# Process subset first
image_paths = sorted(IMAGES_DIR.glob("*.jpg"))[:50]
```

### 🎨 Incorrect Materials
- Check `masked_rgb_hybrid/` visualizations
- Adjust VLM temperature (higher = more creative)
- Provide more context in prompts for your region

### 🏗️ Multiple Buildings Not Separated
- Run Level 2 with `--level 2` separately
- Check `sam3_instance_per_building/` for outputs
- Verify instance prompts are specific enough

---

## Project Structure

```
AutoPBR/
├── 2_sam3hi.py                    # Level 1: Semantic segmentation
├── sam3hi.py                      # Level 2+3: Instance + Composites
├── nemotron_hierarchical.py       # Material analysis
├── step1_download_data.py         # OSM + Mapillary downloader
├── 4_step2_perception.py          # Geometric photo-building matching
├── 5_nemotron_building_priors.py  # VLM analysis
├── photos2/                       # Input photos
├── sam3_semantic2/                # Level 1 outputs
├── sam3_instance_per_building/    # Level 2 outputs
├── sam3_hierarchical_final/       # Level 3 outputs
├── masked_rgb_hybrid/             # Visualization outputs
└── materials_hybrid.json          # Final materials
```

---

**Use Cases**:
- 3D reconstruction for cultural heritage preservation
- Urban planning and simulation (UR-MAT)
- Game engine asset generation (Unreal Engine)
- Material degradation studies
- Tourism and education (VR/AR)

---

## Citation

If you use AutoPBR v2 in your research, please cite:

```bibtex
@software{autopbr_v2_2025,
  title={AutoPBR v2: Hierarchical Material Assignment from Street Imagery},
  author={Your Name},
  year={2025},
  url={https://github.com/dera8/AutoPBR},
  note={SAM3-based semantic and instance segmentation with VLM material analysis}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- **SAM3**: Meta AI / Facebook Research
- **Transformers**: HuggingFace
- **Nemotron VLM**: NVIDIA
- **OSMnx**: Geoff Boeing
- **Mapillary**: Meta AI street imagery

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## Contact

**Issues**: https://github.com/dera8/AutoPBR/issues
**Discussions**: https://github.com/dera8/AutoPBR/discussions

---

**Last Updated**: December 2025
**Version**: 2.0.0
