import os
import csv
import json
import argparse
import requests
from urllib.parse import urlencode

def save_image(url, path):
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"[warn] skip image {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Mapillary thumbnails + manifest (Windows-friendly).")
    parser.add_argument("--token", type=str, default=None, help="Mapillary access token (MLY|...)")
    parser.add_argument("--bbox", type=str, default="5.32045,60.39670,5.32650,60.39990", help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--out", type=str, default="photos", help="Output folder for images")
    parser.add_argument("--max", type=int, default=1000, help="Max images to fetch")
    args = parser.parse_args()

    token = (args.token or os.getenv("MAPILLARY_TOKEN") or "").strip().strip('"').strip("'")
    if not token:
        raise SystemExit("ERROR: provide --token or set MAPILLARY_TOKEN")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    fields = "id,thumb_1024_url,geometry,computed_compass_angle,captured_at"
    base = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "fields": fields,
        "bbox": args.bbox,
        "limit": 500
    }

    total = 0
    manifest = []

    url = f"{base}?{urlencode(params)}"
    while url and total < args.max:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text)
            raise SystemExit("Mapillary API error. Check token/bbox.")
        j = r.json()

        for img in j.get("data", []):
            img_id = img.get("id")
            geom = img.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            lon, lat = coords[0], coords[1]
            heading = img.get("computed_compass_angle", 0.0)
            url_img = img.get("thumb_1024_url")
            out_path = os.path.join(out_dir, f"{img_id}.jpg")

            if url_img:
                ok = save_image(url_img, out_path)
            else:
                ok = False

            manifest.append({
                "photo_id": img_id,
                "path": out_path if ok else "",
                "lat": lat, "lon": lon,
                "heading": heading,
                "fov": 70.0,
                "timestamp": img.get("captured_at")
            })
            total += 1
            if total >= args.max:
                break

        url = j.get("paging", {}).get("next")

    # Write manifests in current working directory
    json_path = os.path.abspath("photos_manifest.json")
    csv_path = os.path.abspath("photos_manifest.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if manifest:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)

    print(f"Downloaded {total} images.")
    print(f"Manifest JSON: {json_path}")
    print(f"Manifest CSV : {csv_path}")
    print(f"Images folder : {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
