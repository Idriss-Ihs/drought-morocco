"""
Streamlit dashboard for provincial drought analytics (SPI)
Run:
  streamlit run src/app/app.py
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import streamlit as st
import plotly.express as px


# ---------- Paths ----------
STATS_CSV = Path("data/processed/provincial_drought_stats.csv")
CLASSES_CSV = Path("data/processed/provincial_spi_classes.csv")
ADM2_SHP   = Path("data/external/geoBoundaries-MAR-ADM2.shp")


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
    name = re.sub(r"[\u0600-\u06FF]+", "", name)
    name = re.sub(r"\s{2,}", " ", name).strip()
    name = name.title()
    replacements = {
        "Laayoune": "Laâyoune",
        "Oued Ed Dahab": "Oued Ed-Dahab",
        "Dakhla Oued Ed Dahab": "Dakhla-Oued Ed-Dahab",
    }
    return replacements.get(name, name)


@st.cache_data
def load_tables():
    stats = pd.read_csv(STATS_CSV)
    stats["province"] = stats["province"].apply(clean_name)
    stats["year"] = stats["year"].astype(int)
    stats["scale"] = stats["scale"].astype(str)

    classes = pd.read_csv(CLASSES_CSV)
    classes["province"] = classes["province"].apply(clean_name)
    classes["time"] = pd.to_datetime(classes["time"])
    classes["year"] = classes["time"].dt.year
    classes["scale"] = classes["scale"].astype(str)

    return stats, classes


@st.cache_resource
def load_geojson():
    gdf = gpd.read_file(ADM2_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    name_col = None
    for c in ["shapeName", "ADM2_EN", "ADM2_FR", "NAME_2", "NAME_1"]:
        if c in gdf.columns:
            name_col = c
            break
    gdf["province"] = gdf[name_col].apply(clean_name)
    # Convert to geojson
    gj = json.loads(gdf.to_json())
    return gj, gdf[["province"]]


# ---------- UI ----------
st.set_page_config(page_title="Morocco Drought (SPI) — Provinces", layout="wide")
st.title("🇲🇦 Morocco Drought Monitoring — Provincial SPI Dashboard")

stats, classes = load_tables()
geojson, gdf_names = load_geojson()

# Sidebar filters
scale = st.sidebar.selectbox("SPI scale (months)", ["1", "3", "6", "12"], index=3)
years = sorted(stats["year"].unique())
year = st.sidebar.slider("Year", min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)))
metric = st.sidebar.selectbox(
    "Metric (yearly, per province)",
    ["mean_spi", "drought_months_moderate", "drought_months_severe",
     "drought_months_extreme", "wet_months_moderate", "wet_months_very",
     "wet_months_extreme", "max_drought_spell_moderate"],
    index=0
)
sel_provinces = st.sidebar.multiselect(
    "Provinces (for time series)",
    options=sorted(gdf_names["province"].unique()),
    default=["Safi", "Marrakech", "Agadir Ida-Outanane"]
)

st.sidebar.markdown("---")
st.sidebar.download_button(
    "Download provincial drought stats (CSV)",
    data=STATS_CSV.read_bytes(),
    file_name="provincial_drought_stats.csv",
    mime="text/csv",
)

# Choropleth map
st.subheader(f"Choropleth — {metric.replace('_',' ').title()} (SPI-{scale}) in {year}")
sub = stats[(stats["year"] == year) & (stats["scale"] == str(scale))].copy()
m = sub[["province", metric]].copy()

# Merge to ensure alignment with geojson names (cleaned)
m["province"] = m["province"].apply(clean_name)

# Color range
if metric == "mean_spi":
    range_color = (-2.5, 2.5)
    color_cont = "RdBu"
else:
    range_color = (0, 12)
    color_cont = "OrRd"

fig_map = px.choropleth(
    m, geojson=geojson, locations="province", featureidkey="properties.province",
    color=metric, range_color=range_color, color_continuous_scale=color_cont,
    projection="mercator", scope="africa"
)
fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(height=650, coloraxis_colorbar=dict(title=metric.replace("_"," ").title()))
st.plotly_chart(fig_map, use_container_width=True)

# Time-series for selected provinces
st.subheader(f"Time series — Mean SPI-{scale} (annual)")
ts = stats[stats["scale"] == str(scale)]
ts_fig = px.line(ts[ts["province"].isin(sel_provinces)], x="year", y="mean_spi",
                 color="province", markers=True)
ts_fig.add_hline(y=0, line_color="gray")
ts_fig.update_layout(height=420, yaxis_title="Mean SPI", xaxis_title="Year")
st.plotly_chart(ts_fig, use_container_width=True)

# Heatmap (all provinces)
st.subheader(f"Heatmap — Mean SPI-{scale} by Province × Year")
mat = (ts.pivot(index="province", columns="year", values="mean_spi")
         .reindex(sorted(ts["province"].unique())))
hm_fig = px.imshow(mat, aspect="auto", color_continuous_scale="RdBu", zmin=-2.0, zmax=2.0,
                   labels=dict(color="Mean SPI"))
hm_fig.update_layout(height=max(500, 18 * len(mat)))
st.plotly_chart(hm_fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "All visuals are derived from CHIRPS-based SPI (1/3/6/12 months). "
    "Methods: rolling precipitation sums → Gamma (MoM) fit per calendar month → zero-inflated CDF → Φ⁻¹ → SPI."
)
