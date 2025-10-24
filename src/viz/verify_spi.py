"""
verify_spi.py
--------------
Visualize SPI results (e.g. SPI-3) as:
1. Color-coded heat maps (drought/wetness)
2. National-average SPI time series

Red = dry (drought), Blue = wet.
"""

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("data/processed/spi_1.nc")  # choose SPI scale to visualize
assert DATA_PATH.exists(), f"File not found: {DATA_PATH}"

# Load data
ds = xr.open_dataset(DATA_PATH)
spi = ds["spi"]

print(f"Dataset loaded: {spi.shape}")
print(f"Time range: {spi.time.values[0]} → {spi.time.values[-1]}")
print(f"SPI range: {float(spi.min())} to {float(spi.max())}")

# ---------------------------------------------------------------------
# 1️⃣ Plot several monthly maps as heat maps
# ---------------------------------------------------------------------

months_to_plot = [
    "2010-11-01",
    "2014-11-01",
    "2018-11-01",
    "2023-11-01",
    "1988-11-01"
]

vmin, vmax = -2.5, 2.5  # consistent color scale

n = len(months_to_plot)
fig, axes = plt.subplots(1, n, figsize=(4*n, 5), constrained_layout=True)

for i, month in enumerate(months_to_plot):
    ax = axes[i]
    data = spi.sel(time=month)
    im = ax.pcolormesh(
        data["longitude"],
        data["latitude"],
        data,
        cmap="RdBu",  # red = dry, blue = wet
        shading="auto",
        vmin=vmin, vmax=vmax
    )
    ax.set_title(f"{month[:7]}", fontsize=12)
    ax.set_xlabel("Longitude")
    if i == 0:
        ax.set_ylabel("Latitude")

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
cbar.set_label("SPI (Standardized Precipitation Index)", fontsize=11)

plt.suptitle("SPI-3 Monthly Drought Maps (Morocco)", fontsize=15)
plt.show()

# ---------------------------------------------------------------------
# 2️⃣ Plot national average SPI time series
# ---------------------------------------------------------------------

spi_mean = spi.mean(dim=["latitude", "longitude"], skipna=True)

plt.figure(figsize=(12, 4))
spi_mean.plot(color="black", linewidth=1.2, label="National Mean SPI-3")

# Shade drought categories
plt.axhspan(-1, -1.5, color="orange", alpha=0.15)
plt.axhspan(-1.5, -2, color="red", alpha=0.15)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)

plt.title("National Average SPI-3 (Drought Intensity Over Time)")
plt.ylabel("SPI (Standardized)")
plt.xlabel("Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
