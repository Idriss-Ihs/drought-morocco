"""
prepare_morocco_shapefile.py
----------------------------
Download and merge GADM Level-0 shapefiles for Morocco (MAR)
and Western Sahara (ESH) into a single unified polygon:
'morocco_full.shp'
"""

import geopandas as gpd
from pathlib import Path
import requests, zipfile, io

def download_and_extract_gadm(iso, out_dir="data/external"):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso}_shp.zip"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"gadm41_{iso}_shp.zip"

    print(f"Downloading {iso} shapefile...")
    r = requests.get(url)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out_dir / iso)
    print(f"Extracted {iso} shapefile to {out_dir / iso}")

    shp_path = next((out_dir / iso).glob("*.shp"))
    return gpd.read_file(shp_path)


def merge_morocco_and_ws():
    gdf_mar = download_and_extract_gadm("MAR")
    gdf_esh = download_and_extract_gadm("ESH")

    # Merge the two territories
    gdf_combined = gpd.GeoDataFrame(pd.concat([gdf_mar, gdf_esh], ignore_index=True))
    gdf_combined["country"] = "Morocco (Unified)"
    gdf_combined = gdf_combined.dissolve(by="country")  # one unified polygon

    # Save unified shapefile
    out_path = Path("data/external/morocco_full.shp")
    gdf_combined.to_file(out_path)
    print(f"✅ Saved unified shapefile: {out_path}")

if __name__ == "__main__":
    import pandas as pd
    merge_morocco_and_ws()
