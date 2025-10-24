"""
prepare_morocco_admins.py
-------------------------
Download and merge Morocco + Western Sahara GADM shapefiles
(Level 1 = Regions, Level 2 = Provinces/Prefectures)
into unified layers including Western Sahara.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import requests, zipfile, io


def download_and_extract_gadm(iso, level, out_dir="data/external"):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso}_shp.zip"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Downloading {iso} Level-{level} shapefile...")
    r = requests.get(url)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out_dir / iso)
    # Check if desired level exists
    level_file = list((out_dir / iso).glob(f"gadm41_{iso}_{level}.shp"))
    if not level_file:
        print(f"⚠️ No Level-{level} shapefile found for {iso}. Skipping.")
        return None
    shp_path = level_file[0]
    print(f"✅ Extracted {iso} Level-{level} to {shp_path}")
    return gpd.read_file(shp_path)



def merge_and_save(level, out_dir="data/external"):
    """
    Merge Morocco (MAR) + Western Sahara (ESH) shapefiles
    for a given administrative level, reproject, clean, and save.
    """
    gdf_mar = download_and_extract_gadm("MAR", level, out_dir)
    gdf_esh = download_and_extract_gadm("ESH", level, out_dir)

    # If Western Sahara has no data at this level, just use Morocco
    if gdf_esh is None:
        print(f"⚠️ Western Sahara has no Level-{level} data. Using Morocco only.")
        gdf = gdf_mar.copy()
    else:
        gdf_mar["country"] = "Morocco"
        gdf_esh["country"] = "Western Sahara"
        gdf = pd.concat([gdf_mar, gdf_esh], ignore_index=True)


    # Harmonize coordinate system
    gdf = gdf.to_crs(epsg=4326)

    # ---- Fix missing columns gracefully ----
    for col in ["NAME_0", "NAME_1", "NAME_2"]:
        if col not in gdf.columns:
            gdf[col] = None

    # Fill in reasonable defaults
    gdf["NAME_0"] = gdf["NAME_0"].fillna("Morocco (Unified)")
    gdf["NAME_1"] = gdf["NAME_1"].fillna("Unknown Region")
    if level == 2:
        gdf["NAME_2"] = gdf["NAME_2"].fillna("Unknown Province")

    # Simplify geometry for faster plotting (optional)
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.01)

    # Save shapefile
    level_name = "regions" if level == 1 else "provinces"
    out_path = Path(out_dir) / f"morocco_{level_name}_full.shp"
    gdf.to_file(out_path)

    print(f"✅ Saved unified Level-{level} shapefile → {out_path}")
    print(f"   Number of polygons: {len(gdf)}")
    preview_cols = ["NAME_1"] + (["NAME_2"] if level == 2 else [])
    print(f"   Example entries:\n{gdf[preview_cols].head()}")


def prepare_all_levels():
    """
    Run both levels (regions and provinces).
    """
    print("🔹 Preparing Morocco administrative boundaries (L1 + L2)...")
    merge_and_save(level=1)  # Regions
    merge_and_save(level=2)  # Provinces
    print("🏁 All shapefiles ready for analysis.")


if __name__ == "__main__":
    prepare_all_levels()
