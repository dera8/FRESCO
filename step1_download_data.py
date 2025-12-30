import os
import requests
import argparse
import time
import json
import math
import osmnx as ox
import geopandas as gpd

# --- CONFIGURAZIONE ---
# IMPORTANTE: Rimetti il tuo token qui se non lo passi da riga di comando
DEFAULT_MAPILLARY_TOKEN = "MLY|24242882775395484|17ef5836fb262177a2100c45742efc12"
DEFAULT_BBOX = "5.32045,60.39670,5.32650,60.39990"

DIRS = { "input": "data_input", "photos": "data_input/photos" }
FILES = {
    "osm": os.path.join(DIRS["input"], "map.osm"),
    "buildings_geo": os.path.join(DIRS["input"], "buildings.geojson"),
    "photos_manifest": os.path.join(DIRS["input"], "photos_manifest.json")
}

OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://api.openstreetmap.fr/oapi/interpreter",
    "http://overpass-api.de/api/interpreter",
]

def setup_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

# --- 1. OSM DOWNLOAD (Il tuo codice corretto + Robustezza) ---
def build_overpass_query(bbox_str: str, profile: str = "autopbr") -> str:
    coords = [float(x) for x in bbox_str.split(",")]
    overpass_bbox = f"{coords[1]},{coords[0]},{coords[3]},{coords[2]}"  # S,W,N,E

    if profile == "autopbr":
        # La query PERFETTA: Ricca come Blosm, ma sicura.
        core = """
        (
          // EDIFICI E PARTI (Per la chiesa e i tetti)
          way["building"];
          way["building:part"];
          relation["building"];
          relation["type"="multipolygon"]["building"];

          // TRASPORTI (Per il contesto stradale)
          way["highway"];
          way["railway"];

          // ACQUA E MOLI (Per Bryggen e il porto)
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
        # Fallback semplice
        core = "(node; way; rel;); (._;>;); out meta;"

    return f"[out:xml][timeout:180][bbox:{overpass_bbox}];\n{core}"

def download_osm_robust(bbox_str: str, profile: str = "autopbr") -> bool:
    print(f"🌍 1. Download Mappa (Profile: {profile})...")
    
    # Se il file esiste ed è buono (>1MB), non rischiamo il 504
    if os.path.exists(FILES["osm"]) and os.path.getsize(FILES["osm"]) > 1000000:
        print("   ✅ Mappa già presente e valida. Salto download.")
        return True

    query = build_overpass_query(bbox_str, profile)

    for url in OVERPASS_SERVERS:
        print(f"   👉 Tentativo con: {url} ...")
        try:
            resp = requests.get(url, params={"data": query}, timeout=180, stream=True)

            if resp.status_code == 200:
                with open(FILES["osm"], "wb") as f:
                    for chunk in resp.iter_content(chunk_size=4096):
                        f.write(chunk)

                # Controllo integrità XML
                with open(FILES["osm"], "rb") as f:
                    f.seek(-50, 2)
                    if b"</osm>" not in f.read():
                        print("      ⚠️ File incompleto. Riprovo...")
                        continue

                size_mb = os.path.getsize(FILES["osm"]) / (1024 * 1024)
                print(f"   ✅ SUCCESSO! Mappa salvata: {size_mb:.2f} MB")
                return True

            elif resp.status_code == 429:
                print("      ⚠️ Rate Limit (429). Attendo 5s...")
                time.sleep(5)
            else:
                print(f"      ❌ Errore HTTP {resp.status_code}")

        except Exception as e:
            print(f"      ❌ Errore connessione: {e}")

    print("\n❌ ERRORE CRITICO: Impossibile scaricare la mappa.")
    return False

# --- 2. GEOJSON CONVERSION (Necessario per Step 2) ---
def convert_to_geojson():
    print("🏗️  2. Preparazione Edifici per AI (GeoJSON)...")
    if not os.path.exists(FILES["osm"]): return

    try:
        # Estraiamo SOLO gli edifici per il matching delle foto
        gdf = ox.features_from_xml(FILES["osm"], tags={'building': True})
        
        # Filtro: Solo poligoni validi
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        # Colonne utili per l'AI
        cols = ['geometry', 'building', 'building:levels', 'building:material', 'building:colour', 'roof:material', 'roof:colour']
        keep_cols = [c for c in cols if c in gdf.columns]
        gdf = gdf[keep_cols]
        
        # ID univoco
        gdf = gdf.reset_index()
        if 'osmid' in gdf.columns: gdf.rename(columns={'osmid': 'building_id'}, inplace=True)
        elif 'element_id' in gdf.columns: gdf.rename(columns={'element_id': 'building_id'}, inplace=True)
        
        # Proiezione metrica
        gdf = gdf.to_crs(epsg=3857)
        
        gdf.to_file(FILES["buildings_geo"], driver="GeoJSON")
        print(f"   ✅ Estratti {len(gdf)} edifici per analisi.")
    except Exception as e:
        print(f"   ⚠️ Warning GeoJSON (non bloccante): {e}")

# --- 3. MAPILLARY DOWNLOAD (Foto + Manifest) ---
def download_mapillary(bbox_str, token):
    print("📸 3. Scaricamento Foto Mapillary...")
    
    if not token or "INSERISCI" in token:
        print("⚠️ Token mancante. Salto foto.")
        return

    coords = bbox_str.split(',')
    bbox_map = f"{coords[0]},{coords[1]},{coords[2]},{coords[3]}"
    
    url = "https://graph.mapillary.com/images"
    params = {
        'access_token': token,
        'fields': 'id,thumb_1024_url,geometry,computed_compass_angle',
        'bbox': bbox_map,
        'limit': 2000
    }
    
    photos = []
    try:
        while True:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code != 200: break
            data = resp.json()
            if 'data' in data: photos.extend(data['data'])
            if 'paging' in data and 'next' in data['paging']:
                url = data['paging']['next']
                params = {}
            else: break
            
        print(f"   -> Trovate {len(photos)} possibili foto.")
        
        manifest = []
        count_dl = 0
        
        for p in photos:
            if 'geometry' not in p: continue
            
            img_id = p['id']
            local_path = os.path.join(DIRS["photos"], f"{img_id}.jpg")
            
            # Scarica solo se non c'è
            if not os.path.exists(local_path):
                try:
                    r = requests.get(p.get('thumb_1024_url'), timeout=10)
                    if r.status_code == 200:
                        with open(local_path, 'wb') as f:
                            f.write(r.content)
                        count_dl += 1
                except: pass
            
            if os.path.exists(local_path):
                manifest.append({
                    "photo_id": img_id,
                    "path": local_path,
                    "url": p.get('thumb_1024_url'),
                    "lon": p['geometry']['coordinates'][0],
                    "lat": p['geometry']['coordinates'][1],
                    "heading": p.get('computed_compass_angle', 0)
                })
        
        with open(FILES["photos_manifest"], 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"   ✅ Foto scaricate: {count_dl}. Manifest aggiornato con {len(manifest)} immagini.")
        
    except Exception as e:
        print(f"   ❌ Errore Mapillary: {e}")

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=str, default=DEFAULT_BBOX)
    parser.add_argument("--token", type=str, default=DEFAULT_MAPILLARY_TOKEN)
    parser.add_argument("--profile", type=str, default="autopbr")
    args = parser.parse_args()

    setup_dirs()
    
    # Pipeline Completa
    if download_osm_robust(args.bbox, args.profile):
        convert_to_geojson()
        download_mapillary(args.bbox, args.token)
        
    print("\n🎉 Step 1 Completato! Tutti i dati sono pronti.")