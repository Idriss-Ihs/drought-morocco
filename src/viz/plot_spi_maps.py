"""
plot_spi_maps.py
----------------
Visualize SPI maps (monthly or seasonal averages) with color-coded drought intensity.
"""

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

DATA_PATH = Path("data/processed/spi_3.nc")  # SPI scale
BORDER_PATH = Path("data/external/morocco_full.shp")



def plot_spi_map(date="2020-01-01", vmin=-2.5, vmax=2.5):
    ds = xr.open_dataset(DATA_PATH)
    spi = ds["spi"].sel(time=date)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = spi.plot(
        ax=ax,
        cmap="RdBu",      # Red=dry, Blue=wet
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    gdf = gpd.read_file(BORDER_PATH)
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("SPI value", fontsize=11)

    ax.set_title(f"SPI-3 Drought Map — {date[:7]}", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_spi_map("2020-01-01")
