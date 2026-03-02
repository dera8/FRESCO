#!/usr/bin/env python3
"""
STEP 1 — Download OSM + Buildings GeoJSON + Mapillary photos + manifest
======================================================================

Modifiche principali (ECCV-oriented):
- Niente token hardcoded nel codice: usa --token o env var MAPILLARY_TOKEN
- Nel manifest salva metadati "camera_prior" e "projection" (pinhole vs equirectangular)
- Salva width/height e un flag is_equirectangular basato su aspect ratio ~2:1
- Mantiene download OSM robusto + conversione GeoJSON

Uso:
  python step1_download_data.py ^
    --bbox "5.32045,60.39670,5.32650,60.39990" ^
    --token "%MAPILLARY_TOKEN%" ^
    --profile autopbr

Oppure:
  set MAPILLARY_TOKEN=MLY|...
  python step1_download_data.py --bbox "..." --profile autopbr
"""

import os
import time
import json
import argparse
import math
import requests
import osmnx as ox
import geopandas as gpd
from PIL import Image

# -----------------------------
# Default params (safe)
# -----------------------------
DEFAULT_BBOX = "5.32045,60.39670,5.32650,60.39990"

# Output structure
DIRS = {"input": "data_input", "photos": "data_input/photos"}
FILES = {
    "osm": os.path.join(DIRS["input"], "map.osm"),
    "buildings_geo": os.path.join(DIRS["input"], "buildings.geojson"),
    "photos_manifest": os.path.join(DIRS["input"], "photos_manifest.json"),
}

OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://api.openstreetmap.fr/oapi/interpreter",
    "http://overpass-api.de/api/interpreter",
]

# Camera prior (per paper + sim matching)
DEFAULT_CAMERA_PRIOR = {
    "type": "street_level",
    "height_m": 1.6,
    "fov_deg": 90,
    "pitch_deg": 0,
    "roll_deg": 0,
    # yaw_deg lo prendiamo da heading
}

# -----------------------------
# Utils
# -----------------------------
def setup_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

def build_overpass_query(bbox_str: str, profile: str = "autopbr") -> str:
    coords = [float(x) for x in bbox_str.split(",")]
    overpass_bbox = f"{coords[1]},{coords[0]},{coords[3]},{coords[2]}"  # S,W,N,E

    if profile == "autopbr":
        core = """
        (
          way["building"];
          way["building:part"];
          relation["building"];
          relation["type"="multipolygon"]["building"];

          way["highway"];
          way["railway"];

          way["landuse"];
          way["leisure"];
          way["natural"="water"];
          way["natural"="coastline"];
          way["man_made"="pier"];
          way["waterway"];

          relation["type"="multipolygon"]["landuse"];
          relation["type"="multipolygon"]["natural"];
        );
        (._;>;);
        out meta;
        """
    else:
        core = "(node; way; rel;); (._;>;); out meta;"

    return f"[out:xml][timeout:180][bbox:{overpass_bbox}];\n{core}"

def download_osm_robust(bbox_str: str, profile: str = "autopbr") -> bool:
    print(f"🌍 1) Download OSM (profile={profile})...")

    # Skip if existing and looks valid
    if os.path.exists(FILES["osm"]) and os.path.getsize(FILES["osm"]) > 1_000_000:
        print("   ✅ OSM già presente (>1MB). Salto download.")
        return True

    query = build_overpass_query(bbox_str, profile)

    for url in OVERPASS_SERVERS:
        print(f"   👉 Tentativo: {url}")
        try:
            resp = requests.get(url, params={"data": query}, timeout=180, stream=True)
            if resp.status_code == 200:
                with open(FILES["osm"], "wb") as f:
                    for chunk in resp.iter_content(chunk_size=4096):
                        f.write(chunk)

                # basic integrity check
                with open(FILES["osm"], "rb") as f:
                    f.seek(-200, 2)
                    tail = f.read()
                if b"</osm>" not in tail:
                    print("      ⚠️ File incompleto (manca </osm>). Riprovo con altro server...")
                    continue

                size_mb = os.path.getsize(FILES["osm"]) / (1024 * 1024)
                print(f"   ✅ SUCCESSO: {FILES['osm']} ({size_mb:.2f} MB)")
                return True

            elif resp.status_code == 429:
                print("      ⚠️ Rate limit 429, attendo 5s...")
                time.sleep(5)
            else:
                print(f"      ❌ HTTP {resp.status_code}")

        except Exception as e:
            print(f"      ❌ Errore connessione: {e}")

    print("❌ ERRORE: impossibile scaricare OSM da Overpass.")
    return False

def convert_to_geojson():
    print("🏗️  2) Estrazione edifici → GeoJSON (per matching)...")
    if not os.path.exists(FILES["osm"]):
        print("   ⚠️ Manca map.osm, salto.")
        return

    try:
        gdf = ox.features_from_xml(FILES["osm"], tags={"building": True})
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

        cols = [
            "geometry",
            "building",
            "building:levels",
            "building:material",
            "building:colour",
            "roof:material",
            "roof:colour",
        ]
        keep_cols = [c for c in cols if c in gdf.columns]
        gdf = gdf[keep_cols]

        gdf = gdf.reset_index()
        if "osmid" in gdf.columns:
            gdf.rename(columns={"osmid": "building_id"}, inplace=True)
        elif "element_id" in gdf.columns:
            gdf.rename(columns={"element_id": "building_id"}, inplace=True)
        else:
            gdf["building_id"] = gdf.index.astype(str)

        gdf = gdf.to_crs(epsg=3857)
        gdf.to_file(FILES["buildings_geo"], driver="GeoJSON")
        print(f"   ✅ Edifici estratti: {len(gdf)} → {FILES['buildings_geo']}")

    except Exception as e:
        print(f"   ⚠️ Warning GeoJSON (non bloccante): {e}")

def _infer_projection_and_size(local_path: str):
    """Return (W, H, is_equirectangular_approx) using aspect ratio ~2:1."""
    try:
        with Image.open(local_path) as im:
            W, H = im.size
        ratio = W / max(H, 1)
        is_equi = abs(ratio - 2.0) < 0.15  # tolleranza
        return W, H, is_equi
    except Exception:
        return None, None, False

def download_mapillary(bbox_str: str, token: str, limit: int = 2000, sleep_s: float = 0.1):
    print("📸 3) Download foto Mapillary + manifest...")

    if not token:
        print("   ⚠️ Token mancante. Passa --token oppure setta env MAPILLARY_TOKEN. Salto.")
        return

    coords = bbox_str.split(",")
    bbox_map = f"{coords[0]},{coords[1]},{coords[2]},{coords[3]}"

    # NB: lasciamo thumb_1024_url per stare leggeri e riproducibili.
    # Se vuoi più qualità: usa thumb_2048_url (se disponibile) nei fields.
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "fields": "id,thumb_1024_url,geometry,computed_compass_angle",
        "bbox": bbox_map,
        "limit": min(limit, 2000),
    }

    photos = []
    try:
        while True:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code != 200:
                print(f"   ❌ Mapillary HTTP {resp.status_code}")
                break

            data = resp.json()
            if "data" in data:
                photos.extend(data["data"])

            if "paging" in data and "next" in data["paging"]:
                url = data["paging"]["next"]
                params = {}  # next is already a full URL with token
                time.sleep(sleep_s)
            else:
                break

        print(f"   -> Trovate {len(photos)} foto candidate.")

        manifest = []
        count_dl = 0

        for p in photos:
            if "geometry" not in p or "coordinates" not in p["geometry"]:
                continue

            img_id = p["id"]
            local_path = os.path.join(DIRS["photos"], f"{img_id}.jpg")

            # download only if missing
            if not os.path.exists(local_path):
                try:
                    img_url = p.get("thumb_1024_url")
                    if img_url:
                        r = requests.get(img_url, timeout=15)
                        if r.status_code == 200:
                            with open(local_path, "wb") as f:
                                f.write(r.content)
                            count_dl += 1
                            time.sleep(sleep_s)
                except Exception:
                    pass

            if os.path.exists(local_path):
                lon, lat = p["geometry"]["coordinates"][0], p["geometry"]["coordinates"][1]
                heading = p.get("computed_compass_angle", 0) or 0

                W, H, is_equi = _infer_projection_and_size(local_path)

                # camera prior to support pose/distribution matching (NOT pixel matching)
                camera_prior = dict(DEFAULT_CAMERA_PRIOR)
                camera_prior["yaw_deg"] = float(heading)

                manifest.append(
                    {
                        "photo_id": str(img_id),
                        "path": local_path,
                        "url": p.get("thumb_1024_url"),
                        "lon": float(lon),
                        "lat": float(lat),
                        "heading": float(heading),
                        "width": W,
                        "height": H,
                        "projection": "equirectangular" if is_equi else "pinhole_or_unknown",
                        "camera_prior": camera_prior,
                        "notes": "Use for pose/distribution matching between real and sim; not pixel-aligned ground truth.",
                    }
                )

        with open(FILES["photos_manifest"], "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Foto scaricate: {count_dl}")
        print(f"   ✅ Manifest: {FILES['photos_manifest']} ({len(manifest)} entries)")

    except Exception as e:
        print(f"   ❌ Errore Mapillary: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=str, default=DEFAULT_BBOX)
    parser.add_argument("--token", type=str, default=os.environ.get("MAPILLARY_TOKEN", ""))
    parser.add_argument("--profile", type=str, default="autopbr")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    setup_dirs()

    ok = download_osm_robust(args.bbox, args.profile)
    if ok:
        convert_to_geojson()
        download_mapillary(args.bbox, args.token, limit=args.limit)

    print("\n🎉 Step 1 completato: OSM + buildings.geojson + photos + photos_manifest.json")

if __name__ == "__main__":
    main()
