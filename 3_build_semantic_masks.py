#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# Class IDs nella semantic mask finale
CLASS_TO_ID = {"road": 1, "wall": 2, "roof": 3}

# Ordine "low -> high priority" (perché facciamo overwrite progressivo)
# quindi: road scritto, poi wall lo può sovrascrivere, poi roof sovrascrive tutto
PRIORITY_LOW_TO_HIGH = ["road", "wall", "roof"]


def load_mask(path: Path, size_hw) -> np.ndarray:
    """
    Carica una mask binaria (PNG) e la converte in 0/1.
    Se manca, restituisce zeri.
    """
    H, W = size_hw
    if not path.exists():
        return np.zeros((H, W), dtype=np.uint8)

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Se size diversa, resize (nearest per non introdurre valori intermedi)
    if arr.shape != (H, W):
        img = img.resize((W, H), resample=Image.NEAREST)
        arr = np.array(img, dtype=np.uint8)

    return (arr > 0).astype(np.uint8)


def cov(pixels: int, total: int) -> float:
    return (pixels / total) * 100.0 if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="photos2",
                        help="Cartella con le immagini .jpg (serve per ricavare size e lista foto)")
    parser.add_argument("--sam3_dir", type=str, default="sam3_semantic2",
                        help="Cartella dove stanno le mask di SAM3: <photoid>_road.png, _wall.png, _roof.png")
    parser.add_argument("--out_dir", type=str, default="semantic_masks",
                        help="Output folder per <photoid>_semantic.png + stats_per_image.json")
    parser.add_argument("--use_manifest", action="store_true",
                        help="Se presente sam3_dir/manifest.json, usa quello per la lista immagini")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    sam3_dir = Path(args.sam3_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Lista immagini
    manifest_path = sam3_dir / "manifest.json"
    image_paths = []

    if args.use_manifest and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for e in manifest:
            p = Path(e["image"])
            # se nel manifest c'è path relativo, rendilo relativo a project root
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            image_paths.append(p)
        print(f"📄 Using manifest: {manifest_path} | images={len(image_paths)}")
    else:
        image_paths = sorted(images_dir.glob("*.jpg"))
        print(f"📁 Using images_dir: {images_dir} | images={len(image_paths)}")

    if not image_paths:
        print("❌ Nessuna immagine trovata. Controlla --images_dir o --use_manifest.")
        return

    stats = {}

    # 2) Per ogni immagine: fondi road/wall/roof in una sola semantic mask
    for idx, img_path in enumerate(image_paths, 1):
        photo_id = img_path.stem
        print(f"\n[{idx}/{len(image_paths)}] {photo_id}")

        if not img_path.exists():
            print(f"  ⚠️ missing image: {img_path}")
            continue

        rgb = Image.open(img_path).convert("RGB")
        W, H = rgb.size
        size_hw = (H, W)
        total_px = H * W

        # semantic finale (0..3)
        semantic = np.zeros(size_hw, dtype=np.uint8)

        # carica le 3 mask binarie (0/1)
        masks = {}
        raw_counts = {}
        for cls in PRIORITY_LOW_TO_HIGH:
            mask_path = sam3_dir / f"{photo_id}_{cls}.png"
            m = load_mask(mask_path, size_hw)
            masks[cls] = m
            raw_counts[f"{cls}_px_raw"] = int(m.sum())

        # OVERWRITE progressivo: road -> wall -> roof
        # roof vince su tutto
        for cls in PRIORITY_LOW_TO_HIGH:
            cls_id = CLASS_TO_ID[cls]
            m = masks[cls]  # 0/1
            semantic[m == 1] = cls_id

        # conteggi finali sulla semantic
        road_px = int((semantic == CLASS_TO_ID["road"]).sum())
        wall_px = int((semantic == CLASS_TO_ID["wall"]).sum())
        roof_px = int((semantic == CLASS_TO_ID["roof"]).sum())
        bg_px = int((semantic == 0).sum())

        class_covs = {
            "road": cov(road_px, total_px),
            "wall": cov(wall_px, total_px),
            "roof": cov(roof_px, total_px),
        }
        dominant_class = max(class_covs.items(), key=lambda x: x[1])[0]

        stats[photo_id] = {
            **raw_counts,
            "road_px": road_px,
            "wall_px": wall_px,
            "roof_px": roof_px,
            "bg_px": bg_px,
            "road_cov": round(class_covs["road"], 2),
            "wall_cov": round(class_covs["wall"], 2),
            "roof_cov": round(class_covs["roof"], 2),
            "dominant_class": dominant_class,
            "image": str(img_path),
            "size": {"W": W, "H": H},
        }

        out_sem = out_dir / f"{photo_id}_semantic.png"
        Image.fromarray(semantic, mode="L").save(out_sem)
        # --- VERSIONE COLORATA PER VISUAL DEBUG ---
        color = np.zeros((H, W, 3), dtype=np.uint8)

        # road = grigio
        color[semantic == CLASS_TO_ID["road"]] = (160, 160, 160)

        # wall = blu
        color[semantic == CLASS_TO_ID["wall"]] = (0, 0, 255)

        # roof = rosso
        color[semantic == CLASS_TO_ID["roof"]] = (255, 0, 0)

        out_color = out_dir / f"{photo_id}_semantic_color.png"
        Image.fromarray(color, mode="RGB").save(out_color)


        print(f"  💾 {out_sem.name} | road={class_covs['road']:.1f}% wall={class_covs['wall']:.1f}% roof={class_covs['roof']:.1f}% | dom={dominant_class}")

    # 3) salva stats globali
    stats_path = out_dir / "stats_per_image.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
