import json
import os
import math
import warnings
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from tqdm import tqdm

# Ignora warning fastidiosi
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
FILES = {
    "buildings": "data_input/buildings.geojson",
    "photos": "data_input/photos_manifest.json",
    "matches": "data_input/matches.json"
}

# PARAMETRI DI TUNING
RAY_LENGTH = 100.0   # Aumentato a 100m per sicurezza
RAY_WIDTH = 30.0     # Cono largo 30m

def create_thick_rays(photos_path):
    print("📐 Calcolo Coni Visuali (Raggi Spessi)...")
    if not os.path.exists(photos_path):
        return None

    with open(photos_path, 'r') as f:
        data = json.load(f)
    
    if not data: return None

    df = pd.DataFrame(data).dropna(subset=['lat', 'lon'])
    
    # Crea GeoDataFrame dai punti GPS
    gdf_pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    ).to_crs(epsg=3857) # Proiezione metrica (metri)

    cones = []
    ids = []
    paths = []

    for idx, row in gdf_pts.iterrows():
        heading = row.get('heading', 0) or 0
        angle_rad = math.radians(90 - heading)
        
        x1, y1 = row.geometry.x, row.geometry.y
        x2 = x1 + math.cos(angle_rad) * RAY_LENGTH
        y2 = y1 + math.sin(angle_rad) * RAY_LENGTH
        
        line = LineString([(x1, y1), (x2, y2)])
        cone = line.buffer(RAY_WIDTH / 2, cap_style=2)
        
        cones.append(cone)
        ids.append(row['photo_id'])
        paths.append(row.get('path', ''))

    return gpd.GeoDataFrame({'photo_id': ids, 'photo_path': paths}, geometry=cones, crs="EPSG:3857")

def main():
    print("⚙️  Avvio Step 2: Geometric Matching...")

    # 1. Carica Edifici
    if not os.path.exists(FILES['buildings']):
        print(f"❌ Manca {FILES['buildings']}. Esegui prima lo Step 1!")
        return
    
    buildings = gpd.read_file(FILES['buildings'])
    
    # --- FIX AUTO-REPAIR ID (Il pezzo che mancava!) ---
    # Cerca colonne candidate per essere l'ID
    id_candidates = ['building_id', 'osmid', 'id', 'element_id']
    found_id = None
    for col in id_candidates:
        if col in buildings.columns:
            found_id = col
            break
    
    if found_id:
        print(f"   -> ID trovato nella colonna: '{found_id}'. Normalizzo...")
        buildings = buildings.rename(columns={found_id: 'building_id'})
    else:
        print("   ⚠️ Nessuna colonna ID trovata. Uso l'indice del file.")
        buildings['building_id'] = buildings.index.astype(str)
    
    # Assicuriamoci che sia stringa per evitare errori dopo
    buildings['building_id'] = buildings['building_id'].astype(str)
    # --------------------------------------------------

    print(f"   -> Caricati {len(buildings)} edifici.")

    # 2. Carica Foto e crea Raggi
    rays = create_thick_rays(FILES['photos'])
    if rays is None or rays.empty:
        print("❌ Nessuna foto trovata nel manifest. Esegui Step 1 con un token valido!")
        return
    print(f"   -> Generati {len(rays)} coni visuali.")

    # 3. Collisione Spaziale
    print("⚔️  Calcolo Intersezioni...")
    intersections = gpd.sjoin(buildings, rays, how="inner", predicate="intersects")
    print(f"   -> {len(intersections)} incroci rilevati.")

    if len(intersections) == 0:
        print("⚠️ Nessuna intersezione trovata. Verifica che le foto siano nella stessa area degli edifici.")
        return

    # 4. Selezione Match Migliore
    matches = {}
    # Ora siamo sicuri che 'building_id' esiste
    grouped = intersections.groupby('building_id')
    
    for b_id, group in tqdm(grouped, desc="Associando foto"):
        best_match = group.iloc[0]
        
        exclude = ['geometry', 'index_right', 'photo_id', 'photo_path', 'building_id']
        tags = {k: v for k, v in best_match.items() if k not in exclude and pd.notnull(v)}
        
        matches[str(b_id)] = {
            "photo_path": best_match['photo_path'],
            "osm_tags": tags
        }

    # 5. Fallback
    all_ids = set(buildings['building_id'].astype(str))
    matched_ids = set(matches.keys())
    missing = all_ids - matched_ids
    
    print(f"   -> Aggiungo fallback logico per {len(missing)} edifici senza foto...")
    for m_id in missing:
        # Trova la riga corretta
        row = buildings[buildings['building_id'] == m_id].iloc[0]
        tags = {k: v for k, v in row.items() if k not in ['geometry', 'building_id'] and pd.notnull(v)}
        matches[m_id] = {
            "photo_path": None,
            "osm_tags": tags
        }

    # Salva
    with open(FILES['matches'], 'w') as f:
        json.dump(matches, f, indent=2)
        
    print(f"\n✅ Step 2 Completato! Risultato in: {FILES['matches']}")
    print(f"   Copertura Visiva: {len(matched_ids)} edifici su {len(all_ids)}.")

if __name__ == "__main__":
    main()