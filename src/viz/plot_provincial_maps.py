"""
plot_provincial_maps.py
-----------------------
Choropleth maps of provincial drought metrics.

Inputs:
  - data/external/geoBoundaries-MAR-ADM2.shp  (provinces shapefile)
  - data/processed/provincial_drought_stats.csv  (yearly metrics per province & scale)

Usage:
  python -m src.viz.plot_provincial_maps --year 2020 --scale 3 --metric mean_spi
  python -m src.viz.plot_provincial_maps --year 2016 --scale 3 --metric drought_months_moderate
"""

from pathlib import Path
import argparse
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import re

from src.utils.logger import setup_logger


# ---------- name cleaning (same logic you used before) ----------
def clean_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    try:
        name = name.encode("latin1").decode("utf-8")
    except Exception:
        pass
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w\s'’\-]", " ", name)
    name = re.sub(r"(?i)\b(province|prefecture|préfecture|region|région)\b", "", name)
    name = re.sub(r"(?i)\b(de|du|d’|des)\b", "", name)
    name = re.sub(r"[\u0600-\u06FF]+", "", name)  # arabic range
    name = re.sub(r"\s{2,}", " ", name).strip()
    name = name.title()
    replacements = {
        "Laayoune": "Laâyoune",
        "Oued Ed Dahab": "Oued Ed-Dahab",
        "Dakhla Oued Ed Dahab": "Dakhla-Oued Ed-Dahab",
    }
    return replacements.get(name, name)


def load_provinces(shp_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    # pick a plausible name column
    name_col = None
    for c in ["shapeName", "ADM2_EN", "ADM2_FR", "NAME_2", "NAME_1"]:
        if c in gdf.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError("Could not find province name column in shapefile.")
    gdf["province"] = gdf[name_col].apply(clean_name)
    return gdf[["province", "geometry"]]


def make_choropleth(year: int, scale: int, metric: str,
                    shp_path=Path("data/external/geoBoundaries-MAR-ADM2.shp"),
                    stats_csv=Path("data/processed/provincial_drought_stats.csv"),
                    cmap="RdBu", vcenter=0.0, save_path: Path | None = None):
    """
    Plot a choropleth for a given year, scale, metric.

    Common metrics: 'mean_spi', 'drought_months_moderate','drought_months_severe',
                    'drought_months_extreme','wet_months_moderate','wet_months_very',
                    'wet_months_extreme','max_drought_spell_moderate'
    """
    logger = setup_logger("PROV_MAP")
    logger.info(f"Loading shapefile: {shp_path}")
    gdf = load_provinces(shp_path)

    logger.info(f"Loading stats: {stats_csv}")
    df = pd.read_csv(stats_csv)
    # normalize
    df["province"] = df["province"].apply(clean_name)
    df["year"] = df["year"].astype(int)
    df["scale"] = df["scale"].astype(str)

    # filter
    sub = df[(df["year"] == year) & (df["scale"] == str(scale))].copy()
    if sub.empty:
        raise ValueError(f"No rows for year={year}, scale={scale} in {stats_csv}")
    if metric not in sub.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(sub.columns)}")

    # join
    m = gdf.merge(sub[["province", metric]], on="province", how="left")

    # plot
    fig, ax = plt.subplots(figsize=(9, 9))
    if metric == "mean_spi":
        # divergent around 0 for SPI
        vmin, vmax = -2.5, 2.5
    else:
        # counts of months: 0..12
        vmin, vmax = 0, 12

    m.plot(column=metric, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="0.2", linewidth=0.5, legend=True)
    ax.set_title(f"{metric.replace('_',' ').title()} — SPI-{scale} — {year}", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()

    if save_path is None:
        save_dir = Path("docs/figures")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"map_{metric}_spi{scale}_{year}.png"

    plt.savefig(save_path, dpi=180)
    logger.info(f"Saved figure: {save_path}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True, help="Year to map, e.g., 2020")
    p.add_argument("--scale", type=int, default=3, help="SPI window in months (1,3,6,12)")
    p.add_argument("--metric", type=str, default="mean_spi", help="Metric column to map")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_choropleth(year=args.year, scale=args.scale, metric=args.metric)
