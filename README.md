# AutoPBR v2

**Structured Material Recognition in Urban Scenes from Street-Level Imagery**

AutoPBR v2 is a research-oriented pipeline for **structured material recognition** in complex urban scenes using street-level photographs.

The core contribution is a **hierarchical decomposition of urban imagery** (road / wall / roof, optionally per-building) that enables **stable, interpretable, and physically meaningful material descriptors**, addressing ambiguity in material recognition from unconstrained images.

While AutoPBR can be connected to PBR and 3D pipelines, **its primary focus is perception, not rendering**.

---

## 🔍 Motivation

Material recognition in urban scenes is inherently ambiguous:

* multiple materials coexist in the same image,
* appearance varies due to lighting, weather, and aging,
* large structures (buildings) mix heterogeneous surfaces.

AutoPBR v2 is built around the hypothesis that:

> **Material recognition becomes ill-posed without explicit structural decomposition.**

By explicitly separating **road, wall, and roof regions**, and optionally **per-building instances**, AutoPBR produces material descriptors that are:

* more stable,
* more interpretable,
* easier to evaluate quantitatively.

---

## 🧠 Core Idea

Instead of predicting a single “material label” per image, AutoPBR outputs:

> **Structured material descriptors conditioned on semantic and structural context.**

Example:

```json
{
  "class": "wall",
  "material": ["brick", "painted_surface"],
  "color": ["red", "white"],
  "surface": "rough",
  "aging": "weathered",
  "confidence": 0.72
}
```

This representation is:

* **semantic** (interpretable),
* **physical** (compatible with simulation),
* **measurable** (coverage-weighted, confidence-aware).

---

## Pipeline Overview

```
Street-Level Photos
        ↓
[Level 1] Semantic Segmentation (SAM3)
        ↓
[Level 2] Structural Decomposition
        ↓
[Level 3] Masked Visual Analysis (VLM)
        ↓
Structured Material Descriptors (JSON)
```

### Level 1 — Semantic Segmentation

Each image is decomposed into:

* `road`
* `wall`
* `roof`

using SAM3-based segmentation.

### Level 2 — Structural Decomposition (Optional)

If multiple buildings are present:

* wall and roof regions are split into **per-building instances**
* otherwise, a **global per-class representation** is used

This allows a controlled trade-off between simplicity and precision.

### Level 3 — Material Analysis

For each valid region:

* masked RGB is generated,
* a Vision-Language Model (VLM) analyzes **only the target surface**,
* outputs structured descriptors (material, color, surface, aging, confidence).

---

## Installation

```bash
git clone https://github.com/dera8/AutoPBR.git
cd AutoPBR

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install torch torchvision
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

### API Keys

```bash
export NVIDIA_API_KEY="your_nvidia_api_key"
```

---

## Quick Start (ECCV-relevant path)

```bash
# 1. Put images here
photos2/

# 2. Semantic segmentation
python 2_sam3hi.py

# 3. Build semantic composites
python sam3hi.py --all

# 4. Material recognition
python nemotron_hierarchical.py
```

Output:

```
materials_hybrid.json
```

---

## Output Format (Core Result)

```json
{
  "photo_id": "101989905547842",
  "classes": {
    "wall": {
      "coverage": 35.2,
      "material": ["painted wood", "wood siding"],
      "color": ["red", "dark red"],
      "surface": "rough",
      "aging": "weathered",
      "confidence": 0.81
    },
    "roof": {
      "coverage": 18.7,
      "material": ["clay tiles"],
      "color": ["terracotta"],
      "surface": "tiled",
      "confidence": 0.88
    }
  }
}
```

---

## What AutoPBR v2 **Is**

✅ A framework for **structured material recognition**
✅ Focused on **urban scenes**
✅ Designed for **evaluation and ablation**
✅ Compatible with **simulation and rendering pipelines**

## What AutoPBR v2 **Is Not**

❌ Not inverse rendering
❌ Not texture reconstruction
❌ Not photorealistic PBR synthesis
❌ Not instance-level material cloning

---

## Optional Downstream Use (Not Core Contribution)

AutoPBR descriptors can be mapped to:

* PBR libraries,
* Unreal Engine materials,
* urban digital twins.

These steps are **explicitly downstream** and **not required** to use or evaluate the method.

---

## Evaluation Use Cases

* Stability under viewpoint changes
* Ambiguity reduction vs. unstructured baselines
* Coverage-weighted material consistency
* Structured vs. unstructured material prediction

---

## Project Structure

```
AutoPBR/
├── 2_sam3hi.py              # Semantic segmentation
├── sam3hi.py                # Structural decomposition
├── nemotron_hierarchical.py # Material analysis
├── photos2/
├── sam3_semantic2/
├── sam3_hierarchical_final/
└── materials_hybrid.json
```

---

## Citation

```bibtex
@software{autopbr_v2_2025,
  title={AutoPBR v2: Structured Material Recognition in Urban Scenes},
  author={Your Name},
  year={2025},
  url={https://github.com/dera8/AutoPBR}
}
```
