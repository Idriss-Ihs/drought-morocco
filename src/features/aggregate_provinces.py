"""
aggregate_provinces.py
----------------------
Compute mean SPI per province (ADM2) and clean province names.
"""

import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import unicodedata
import re
from pathlib import Path
from tqdm import tqdm
from src.utils.logger import setup_logger


import unicodedata
import re

def clean_province_name(name: str) -> str:
    """Comprehensively clean and normalize province names from shapefile."""
    if not isinstance(name, str) or not name.strip():
        return "Unknown"

    # --- Try to fix encoding issues ---
    try:
        name = name.encode("latin1").decode("utf-8")
    except Exception:
        pass

    # --- Normalize and remove diacritics ---
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w\s'’\-]", " ", name)  # remove stray symbols

    # --- Remove French/English prefixes and suffixes ---
    # e.g. "Province de", "Préfecture de", "Province d’", "Province", etc.
    name = re.sub(r"(?i)\b(province|prefecture|préfecture|region|région)\b", "", name)
    name = re.sub(r"(?i)\b(de|du|d’|des)\b", "", name)

    # --- Remove Arabic words (non-Latin script) ---
    # Any Arabic letters (range \u0600-\u06FF)
    name = re.sub(r"[\u0600-\u06FF]+", "", name)

    # --- Remove multiple spaces and random leftovers ---
    name = re.sub(r"\s{2,}", " ", name).strip()

    # --- Title case ---
    name = name.title()

    # --- Manual corrections for known cases ---
    replacements = {
        "Laayoune": "Laâyoune",
        "Oued Ed Dahab": "Oued Ed-Dahab",
        "Dakhla Oued Ed Dahab": "Dakhla-Oued Ed-Dahab",
    }
    if name in replacements:
        name = replacements[name]

    return name



def aggregate_spi_by_province():
    logger = setup_logger("PROVINCE_AGGREGATION")
    logger.info("Starting provincial SPI aggregation")

    base_path = Path("data")
    shp_path = base_path / "external" / "geoBoundaries-MAR-ADM2.shp"
    spi_dir = base_path / "processed"
    out_dir = spi_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    # --- Load shapefile ---
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        logger.warning("Province shapefile has no CRS, assuming EPSG:4326")
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Determine name column (some shapefiles use shapeName, ADM2_EN, NAME_2, etc.)
    name_col = None
    for col in ["shapeName", "ADM2_EN", "ADM2_FR", "NAME_2"]:
        if col in gdf.columns:
            name_col = col
            break
    if not name_col:
        logger.error("Could not find a suitable name column in shapefile.")
        return

    gdf["province_clean"] = gdf[name_col].apply(clean_province_name)
    logger.info(f"Using column '{name_col}' for province names.")

    # --- SPI files ---
    spi_files = sorted(spi_dir.glob("spi_*.nc"))
    all_results = []

    for f in tqdm(spi_files, desc="Aggregating SPI files"):
        scale = f.stem.split("_")[1]
        logger.info(f"Processing {f.name} ({scale}-month SPI)")

        ds = xr.open_dataset(f)
        var = list(ds.data_vars)[0]
        da = ds[var].rio.write_crs("EPSG:4326")

        for idx, row in gdf.iterrows():
            province = row["province_clean"]
            geom = row.geometry

            try:
                clipped = da.rio.clip([geom], gdf.crs)
                mean_spi = clipped.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()
                mean_spi["province"] = province
                mean_spi["scale"] = scale
                all_results.append(mean_spi)
            except Exception as e:
                logger.warning(f"Failed for {province}: {e}")
                continue

    df = pd.concat(all_results, ignore_index=True)
    df_pivot = df.pivot_table(index=["time", "province"], columns="scale", values=var).reset_index()

    out_csv = out_dir / "provincial_spi.csv"
    df_pivot.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"✅ Saved cleaned provincial SPI data: {out_csv}")

    print(df_pivot.head())


if __name__ == "__main__":
    aggregate_spi_by_province()
