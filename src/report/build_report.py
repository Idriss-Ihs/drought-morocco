"""
build_report.py
---------------
Generate a data-driven drought report (HTML) for Moroccan provinces.

Inputs (produced earlier):
  - data/processed/provincial_drought_stats.csv
  - data/processed/provincial_spi_classes.csv
  - data/external/geoBoundaries-MAR-ADM2.shp

Outputs:
  - docs/figures/... (PNG charts)
  - docs/report/index.html

Usage:
  python -m src.report.build_report
"""

from __future__ import annotations

from pathlib import Path
import io
import json
import re
import unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from jinja2 import Template

from src.utils.logger import setup_logger


# ---------- Paths ----------
STATS_CSV = Path("data/processed/provincial_drought_stats.csv")
CLASSES_CSV = Path("data/processed/provincial_spi_classes.csv")
ADM2_SHP   = Path("data/external/geoBoundaries-MAR-ADM2.shp")

FIG_DIR = Path("docs/figures")
REPORT_DIR = Path("docs/report")
REPORT_HTML = REPORT_DIR / "index.html"


# ---------- Helpers ----------
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
    name = re.sub(r"[\u0600-\u06FF]+", "", name)  # remove Arabic letters
    name = re.sub(r"\s{2,}", " ", name).strip()
    name = name.title()
    replacements = {
        "Laayoune": "Laâyoune",
        "Oued Ed Dahab": "Oued Ed-Dahab",
        "Dakhla Oued Ed Dahab": "Dakhla-Oued Ed-Dahab",
    }
    return replacements.get(name, name)


def ensure_sources():
    if not STATS_CSV.exists():
        raise FileNotFoundError(f"Missing {STATS_CSV}")
    if not CLASSES_CSV.exists():
        raise FileNotFoundError(f"Missing {CLASSES_CSV}")
    if not ADM2_SHP.exists():
        raise FileNotFoundError(f"Missing {ADM2_SHP}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    stats = pd.read_csv(STATS_CSV)
    stats["province"] = stats["province"].apply(clean_name)
    stats["year"] = stats["year"].astype(int)
    stats["scale"] = stats["scale"].astype(str)

    classes = pd.read_csv(CLASSES_CSV)
    classes["province"] = classes["province"].apply(clean_name)
    classes["time"] = pd.to_datetime(classes["time"])
    classes["year"] = classes["time"].dt.year
    classes["scale"] = classes["scale"].astype(str)

    gdf = gpd.read_file(ADM2_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # figure out province name column & clean
    name_col = None
    for c in ["shapeName", "ADM2_EN", "ADM2_FR", "NAME_2", "NAME_1"]:
        if c in gdf.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError("Cannot find province name column in shapefile.")
    gdf["province"] = gdf[name_col].apply(clean_name)
    gdf = gdf[["province", "geometry"]]

    return stats, classes, gdf


# ---------- Analytics ----------
def national_summary(stats: pd.DataFrame, scale="12") -> dict:
    sub = stats[stats["scale"] == str(scale)]
    # national mean across provinces (weighted equally)
    annual = sub.groupby("year")["mean_spi"].mean().reset_index()

    # worst (driest) years by mean SPI
    worst = annual.nsmallest(5, "mean_spi").to_dict(orient="records")
    best  = annual.nlargest(5, "mean_spi").to_dict(orient="records")

    # trend (simple slope)
    x = annual["year"].values
    y = annual["mean_spi"].values
    if len(x) >= 2:
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = np.nan

    return {"annual": annual, "worst": worst, "best": best, "slope": slope}


def top_dry_provinces_last_n(stats: pd.DataFrame, n_years=10, scale="12") -> pd.DataFrame:
    max_year = stats["year"].max()
    min_year = max_year - n_years + 1
    sub = stats[(stats["scale"] == str(scale)) & (stats["year"].between(min_year, max_year))]
    agg = (
        sub.groupby("province")
           .agg(mean_spi=("mean_spi", "mean"),
                severe_months=("drought_months_severe", "sum"),
                extreme_months=("drought_months_extreme", "sum"))
           .reset_index()
           .sort_values(["mean_spi", "severe_months", "extreme_months"])
    )
    return agg.head(15)


# ---------- Figures ----------
def fig_national_ts(annual_df: pd.DataFrame, scale="12") -> Path:
    out = FIG_DIR / f"national_spi_mean_spi{scale}.png"
    plt.figure(figsize=(10, 4))
    plt.plot(annual_df["year"], annual_df["mean_spi"], marker="o", lw=1)
    plt.axhline(0, color="0.3", lw=1)
    plt.title(f"Morocco — National Mean SPI-{scale} (Annual)", fontweight="bold")
    plt.xlabel("Year"); plt.ylabel("Mean SPI")
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()
    return out


def fig_map_year(stats: pd.DataFrame, gdf: gpd.GeoDataFrame,
                 year: int, scale="12", metric="mean_spi") -> Path:
    sub = stats[(stats["year"] == year) & (stats["scale"] == str(scale))]
    m = gdf.merge(sub[["province", metric]], on="province", how="left")

    out = FIG_DIR / f"map_{metric}_spi{scale}_{year}.png"
    plt.figure(figsize=(9, 9))
    if metric == "mean_spi":
        norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)
        m.plot(column=metric, cmap="RdBu", norm=norm, edgecolor="0.2", linewidth=0.5, legend=True)
    else:
        m.plot(column=metric, cmap="OrRd", vmin=0, vmax=12, edgecolor="0.2", linewidth=0.5, legend=True)
    ttl = f"{metric.replace('_',' ').title()} — SPI-{scale} — {year}"
    plt.title(ttl, fontweight="bold"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()
    return out


def fig_leaderboard_table(df: pd.DataFrame, title: str, fname: str) -> Path:
    out = FIG_DIR / fname
    plt.figure(figsize=(8, 6))
    plt.axis("off")
    shown = df.copy()
    shown["mean_spi"] = shown["mean_spi"].round(3)
    shown["severe_months"] = shown["severe_months"].astype(int)
    shown["extreme_months"] = shown["extreme_months"].astype(int)
    table = plt.table(cellText=shown.values,
                      colLabels=shown.columns.tolist(),
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    plt.title(title, fontweight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()
    return out


# ---------- HTML template ----------
HTML_TEMPLATE = Template(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Drought Monitoring — Morocco (SPI)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; color: #111; }
  header { padding: 24px 28px; background: #0f172a; color: #fff; }
  h1 { margin: 0 0 6px 0; font-size: 26px; }
  h2 { margin: 18px 0 8px 0; }
  .container { padding: 22px 28px; max-width: 1100px; margin: auto; }
  .grid { display: grid; gap: 18px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
  figure { margin: 0; background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  figcaption { margin-top: 8px; font-size: 13px; color:#475569; }
  .pill { display:inline-block; padding: 4px 10px; background:#e2e8f0; border-radius:999px; font-size: 12px; margin-right:8px; }
  .row { margin: 16px 0; }
  .code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#f1f5f9; padding:4px 8px; border-radius:6px; }
</style>
</head>
<body>
<header>
  <h1>Drought Monitoring — Morocco (SPI)</h1>
  <div>
    <span class="pill">SPI scales: 1 / 3 / 6 / 12</span>
    <span class="pill">Admin level: Provinces (ADM2)</span>
  </div>
</header>

<div class="container">
  <h2>Executive Summary (SPI-12)</h2>
  <div class="row">
    <div class="pill">Trend slope: <strong>{{ slope|round(4) }}</strong> SPI units per year (national mean)</div>
  </div>
  <div class="grid">
    <figure>
      <img src="../figures/{{ nat_ts_img }}" style="width:100%; border-radius:8px"/>
      <figcaption>National mean SPI-12 (annual). Dashed line at 0 (near normal).</figcaption>
    </figure>
    <figure>
      <img src="../figures/{{ map_img }}" style="width:100%; border-radius:8px"/>
      <figcaption>Provincial map — {{ map_metric }} (SPI-{{ scale }}) for {{ year }}.</figcaption>
    </figure>
    <figure style="grid-column: 1 / -1;">
      <img src="../figures/{{ top_table_img }}" style="width:100%; border-radius:8px"/>
      <figcaption>Top driest provinces in the last {{ lookback }} years — lower mean SPI and higher severe/extreme counts indicate worse drought conditions.</figcaption>
    </figure>
  </div>

  <h2>Key observations</h2>
  <ul>
    <li>Worst (driest) national years (SPI-{{ scale }}): 
      {% for row in worst_years %}
        <span class="pill">{{ row.year }} ({{ "%.2f"|format(row.mean_spi) }})</span>
      {% endfor %}
    </li>
    <li>Best (wettest) national years (SPI-{{ scale }}): 
      {% for row in best_years %}
        <span class="pill">{{ row.year }} ({{ "%.2f"|format(row.mean_spi) }})</span>
      {% endfor %}
    </li>
  </ul>

  <p>All figures and tables are generated programmatically from the processed SPI datasets. See repository for methodology and code.</p>
  <p class="code">Report built from: {{ stats_csv.name }}, {{ classes_csv.name }}</p>
</div>

</body>
</html>
""")


# ---------- Build pipeline ----------
def build_report():
    logger = setup_logger("REPORT")
    ensure_sources()
    logger.info("Loading data…")
    stats, classes, gdf = load_data()

    # Parameters
    scale = "12"             # focus of executive summary
    year_for_map = int(stats["year"].max())
    lookback = 10
    map_metric = "mean_spi"  # or 'drought_months_moderate'

    logger.info("Computing national summary…")
    summary = national_summary(stats, scale=scale)

    logger.info("Computing top dry provinces…")
    top_dry = top_dry_provinces_last_n(stats, n_years=lookback, scale=scale)
    top_table_img = fig_leaderboard_table(
        top_dry[["province", "mean_spi", "severe_months", "extreme_months"]],
        title=f"Driest provinces — last {lookback} years (SPI-{scale})",
        fname=f"top_driest_provinces_spi{scale}_last{lookback}.png",
    )

    logger.info("Rendering figures…")
    nat_ts_img = fig_national_ts(summary["annual"], scale=scale).name
    map_img = fig_map_year(stats, gdf, year=year_for_map, scale=scale, metric=map_metric).name

    logger.info("Writing HTML report…")
    html = HTML_TEMPLATE.render(
        slope=summary["slope"],
        worst_years=summary["worst"],
        best_years=summary["best"],
        nat_ts_img=Path(nat_ts_img).name,
        map_img=Path(map_img).name,
        map_metric=map_metric.replace("_", " "),
        scale=scale,
        year=year_for_map,
        top_table_img=Path(top_table_img).name,
        lookback=lookback,
        stats_csv=STATS_CSV,
        classes_csv=CLASSES_CSV,
    )
    REPORT_HTML.write_text(html, encoding="utf-8")
    logger.info(f"✅ Report generated: {REPORT_HTML}")


if __name__ == "__main__":
    build_report()
