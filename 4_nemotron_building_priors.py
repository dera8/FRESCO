#!/usr/bin/env python3
"""
Hybrid Material Analysis: Per-Building + Per-Class
===================================================
INTELLIGENTE: Sceglie automaticamente l'approccio migliore

Se 1 building:  RGB ⊙ mask_class → Nemotron (semplice)
Se N buildings: RGB ⊙ composite_building → Nemotron (preciso)

Input: sam3_hierarchical_final/manifest_hierarchical.json
Output: materials_hybrid.json
"""

import base64
import json
from pathlib import Path
from openai import OpenAI
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

NVIDIA_API_KEY = "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"
MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

# Directories
ROOT = Path(__file__).parent
PHOTOS_DIR = ROOT / "photos2"
SEMANTIC_DIR = ROOT / "sam3_semantic2"
HIERARCHICAL_DIR = ROOT / "sam3_hierarchical_final"
MASKED_DIR = ROOT / "masked_rgb_hybrid"
OUTPUT_FILE = ROOT / "materials_hybrid.json"

MASKED_DIR.mkdir(parents=True, exist_ok=True)

# Classes
CLASSES = ["wall", "roof", "road"]

# Setup OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# ============================================================================
# HELPER: CREATE MASKED RGB
# ============================================================================

def create_class_masked_rgb(rgb_path: Path, class_name: str) -> Path:
    """
    Simple approach: RGB ⊙ mask_class
    Used when only 1 building or for road (always global)
    """
    mask_path = SEMANTIC_DIR / f"{rgb_path.stem}_{class_name}.png"
    
    if not mask_path.exists():
        return None
    
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    mask_3ch = np.stack([mask > 0] * 3, axis=-1)
    masked_rgb = rgb * mask_3ch
    
    output_path = MASKED_DIR / f"{rgb_path.stem}_{class_name}_simple.jpg"
    Image.fromarray(masked_rgb.astype(np.uint8)).save(output_path, quality=95)
    
    return output_path


def create_building_masked_rgb(rgb_path: Path, composite_path: Path, 
                               building_idx: int, class_name: str) -> Path:
    """
    Building-specific approach: RGB ⊙ (composite == class_id)
    Used when multiple buildings in same photo
    """
    if not composite_path.exists():
        return None
    
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    composite = np.array(Image.open(composite_path).convert("L"))
    
    # Map class to ID
    class_ids = {"road": 1, "wall": 2, "roof": 3}
    class_id = class_ids.get(class_name, 0)
    
    # Create mask for this class in this building
    mask = (composite == class_id)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    masked_rgb = rgb * mask_3ch
    
    photo_id = rgb_path.stem
    output_path = MASKED_DIR / f"{photo_id}_building{building_idx:02d}_{class_name}.jpg"
    Image.fromarray(masked_rgb.astype(np.uint8)).save(output_path, quality=95)
    
    return output_path


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_SIMPLE = """
You are analyzing {class_name} materials in Bryggen, Bergen (Norway).

The image shows ONLY the {class_name} (rest is black/masked out).

TASK: Analyze the visible region and identify materials, colors, and patterns.

OUTPUT (JSON only):
{{
  "material": ["option1", "option2"],
  "color": ["color1", "color2"],
  "pattern": ["pattern1"],
  "condition": "description",
  "confidence": "high/medium/low"
}}

CONTEXT:
{context}
"""

PROMPT_BUILDING = """
You are analyzing a SPECIFIC BUILDING's {class_name} in Bryggen, Bergen (Norway).

The image shows ONLY this building's {class_name} (rest is black).
This is building #{building_idx} out of {total_buildings} visible in the photo.

TASK: Identify materials, colors, and patterns for THIS SPECIFIC building's {class_name}.

OUTPUT (JSON only):
{{
  "material": ["specific_material_1", "material_2"],
  "color": ["specific_color_1", "color_2"],
  "pattern": ["pattern_1"],
  "condition": "condition_description",
  "confidence": "high/medium/low",
  "distinguishing_features": "what makes this building unique"
}}

CONTEXT:
{context}
Building coverage: {coverage}%
"""

CLASS_CONTEXTS = {
    "wall": "Walls: painted wooden planks (red ochre, yellow, white, brown), brick, stone",
    "roof": "Roofs: clay tiles, slate, wooden shingles, metal (terracotta, grey, brown)",
    "road": "Roads: cobblestone, stone pavement, wooden docks (grey, brown, black)"
}


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def encode_image_base64(path: Path) -> str:
    """Encode image to base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_simple(masked_rgb_path: Path, class_name: str) -> dict:
    """
    Simple analysis: one class across entire photo
    """
    try:
        img_b64 = encode_image_base64(masked_rgb_path)
        
        prompt = PROMPT_SIMPLE.format(
            class_name=class_name,
            context=CLASS_CONTEXTS[class_name]
        )
        
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.6,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
        )
        
        raw = response.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        
        return {
            "success": True,
            "approach": "simple",
            "data": parsed,
            "masked_rgb": str(masked_rgb_path)
        }
        
    except Exception as e:
        logger.error(f"      ❌ Analysis failed: {e}")
        return {"success": False, "error": str(e)}


def analyze_building_specific(masked_rgb_path: Path, class_name: str,
                              building_idx: int, total_buildings: int, 
                              coverage: float) -> dict:
    """
    Building-specific analysis: one class of one building
    """
    try:
        img_b64 = encode_image_base64(masked_rgb_path)
        
        prompt = PROMPT_BUILDING.format(
            class_name=class_name,
            building_idx=building_idx,
            total_buildings=total_buildings,
            context=CLASS_CONTEXTS[class_name],
            coverage=coverage
        )
        
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.6,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
        )
        
        raw = response.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        
        return {
            "success": True,
            "approach": "building_specific",
            "data": parsed,
            "masked_rgb": str(masked_rgb_path)
        }
        
    except Exception as e:
        logger.error(f"      ❌ Analysis failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# INTELLIGENT PROCESSING
# ============================================================================

def process_image_intelligent(photo_path: Path, buildings_data: list) -> dict:
    """
    INTELLIGENT: Choose best approach based on number of buildings
    
    Strategy:
    - 0-1 buildings: Simple approach (RGB ⊙ mask_class)
    - 2+ buildings: Building-specific approach (RGB ⊙ composite per building)
    - Road: Always simple (global, not building-specific)
    """
    photo_id = photo_path.stem
    num_buildings = len(buildings_data)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📸 {photo_id}")
    logger.info(f"{'='*60}")
    logger.info(f"Buildings detected: {num_buildings}")
    
    result = {
        "photo_id": photo_id,
        "photo_path": str(photo_path),
        "num_buildings": num_buildings,
        "approach": "building_specific" if num_buildings > 1 else "simple",
        "buildings": []
    }
    
    # CASE 1: No buildings or 1 building → Simple approach
    if num_buildings <= 1:
        logger.info("📌 Using SIMPLE approach (1 or 0 buildings)")
        
        building_result = {"building_idx": 0, "classes": {}}
        
        for class_name in CLASSES:
            logger.info(f"  🔍 {class_name.upper()}")
            
            masked_rgb = create_class_masked_rgb(photo_path, class_name)
            
            if not masked_rgb:
                logger.info(f"    ⚠️  No mask available")
                continue
            
            # Calculate coverage
            mask_path = SEMANTIC_DIR / f"{photo_id}_{class_name}.png"
            mask = np.array(Image.open(mask_path).convert("L"))
            coverage = (mask > 0).sum() / mask.size * 100
            
            logger.info(f"    Coverage: {coverage:.1f}%")
            
            if coverage < 0.1:
                logger.info(f"    ⚠️  Too small, skipping")
                continue
            
            analysis = analyze_simple(masked_rgb, class_name)
            
            if analysis["success"]:
                mat = analysis['data'].get('material', ['N/A'])[0]
                col = analysis['data'].get('color', ['N/A'])[0]
                logger.info(f"    ✓ {mat}, {col}")
            
            building_result["classes"][class_name] = analysis
        
        result["buildings"].append(building_result)
    
    # CASE 2: Multiple buildings → Building-specific approach
    else:
        logger.info(f"📌 Using BUILDING-SPECIFIC approach ({num_buildings} buildings)")
        
        for building in buildings_data:
            building_idx = building['building_idx']
            composite_path = Path(building['composite_mask'])
            coverage_data = building['coverage']
            
            logger.info(f"\n  🏢 Building {building_idx:02d}")
            
            building_result = {
                "building_idx": building_idx,
                "composite_mask": str(composite_path),
                "classes": {}
            }
            
            for class_name in CLASSES:
                logger.info(f"    🔍 {class_name}")
                
                # Road is always global (not building-specific)
                if class_name == "road":
                    masked_rgb = create_class_masked_rgb(photo_path, class_name)
                    if masked_rgb:
                        analysis = analyze_simple(masked_rgb, class_name)
                    else:
                        analysis = {"success": False, "error": "no_mask"}
                else:
                    # Wall and roof are building-specific
                    masked_rgb = create_building_masked_rgb(
                        photo_path, composite_path, building_idx, class_name
                    )
                    
                    if not masked_rgb:
                        logger.info(f"      ⚠️  Failed to create masked RGB")
                        continue
                    
                    coverage = coverage_data.get(class_name, 0)
                    
                    if coverage < 0.1:
                        logger.info(f"      ⚠️  Too small ({coverage:.1f}%)")
                        continue
                    
                    analysis = analyze_building_specific(
                        masked_rgb, class_name, building_idx, 
                        num_buildings, coverage
                    )
                
                if analysis.get("success"):
                    mat = analysis['data'].get('material', ['N/A'])[0]
                    col = analysis['data'].get('color', ['N/A'])[0]
                    logger.info(f"      ✓ {mat}, {col}")
                
                building_result["classes"][class_name] = analysis
            
            result["buildings"].append(building_result)
    
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("="*60)
    logger.info("HYBRID MATERIAL ANALYSIS (Intelligent)")
    logger.info("="*60)
    logger.info("Strategy:")
    logger.info("  • 0-1 buildings → Simple (RGB ⊙ mask_class)")
    logger.info("  • 2+ buildings  → Per-building (RGB ⊙ composite)")
    logger.info("  • Road          → Always global")
    logger.info("="*60)
    
    # Load hierarchical manifest
    manifest_path = HIERARCHICAL_DIR / "manifest_hierarchical.json"
    
    if not manifest_path.exists():
        logger.error(f"❌ Manifest not found: {manifest_path}")
        logger.error("   Run: python sam3hi.py --all")
        return
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    logger.info(f"\n✅ Loaded {len(manifest)} images")
    
    # Process each image
    results = []
    total_buildings = 0
    success_by_class = {"wall": 0, "roof": 0, "road": 0}
    total_by_class = {"wall": 0, "roof": 0, "road": 0}
    
    for idx, entry in enumerate(manifest, 1):
        photo_path = Path(entry['image'])
        buildings_data = entry.get('buildings', [])
        
        logger.info(f"\n[{idx}/{len(manifest)}]")
        
        result = process_image_intelligent(photo_path, buildings_data)
        results.append(result)
        
        total_buildings += len(buildings_data) if buildings_data else 1
        
        # Update stats
        for building in result['buildings']:
            for class_name, analysis in building['classes'].items():
                total_by_class[class_name] += 1
                if analysis.get('success'):
                    success_by_class[class_name] += 1
    
    # Save results
    output_data = {
        "results": results,
        "statistics": {
            "total_images": len(results),
            "total_buildings": total_buildings,
            "per_class": {
                cls: {
                    "total": total_by_class[cls],
                    "successful": success_by_class[cls],
                    "rate": f"{success_by_class[cls]/total_by_class[cls]*100:.1f}%"
                        if total_by_class[cls] > 0 else "0%"
                }
                for cls in CLASSES
            }
        }
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 ANALYSIS COMPLETE!")
    logger.info("="*60)
    logger.info(f"\n📊 Statistics:")
    logger.info(f"  Images:    {len(results)}")
    logger.info(f"  Buildings: {total_buildings}")
    for cls in CLASSES:
        logger.info(f"  {cls.capitalize():5s}: {success_by_class[cls]}/{total_by_class[cls]} "
                   f"({success_by_class[cls]/total_by_class[cls]*100:.1f}%)")
    logger.info(f"\n💾 Output:")
    logger.info(f"  {OUTPUT_FILE}")
    logger.info(f"  {MASKED_DIR}/")


if __name__ == "__main__":
    main()
