import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# DATA_PATH = Path("data/interim/chirps_morocco.nc")
# assert DATA_PATH.exists(), f"File not found: {DATA_PATH}"

# ds = xr.open_dataset(DATA_PATH)
# precip = ds["precip"]


# print(f"Dataset dimensions: {dict(ds.sizes)}")
# print(f"Time range: {str(precip.time.values[0])[:10]} → {str(precip.time.values[-1])[:10]}")
# print(f"Mean precip: {float(precip.mean().values):.2f} mm/month")
# print(f"Max precip: {float(precip.max().values):.2f} mm/month")


# precip_yearly = precip.groupby("time.year").sum("time")


# year_to_plot = 1996
# data = precip_yearly.sel(year=year_to_plot)


# plt.figure(figsize=(8, 6))
# im = plt.pcolormesh(
#     data["longitude"],
#     data["latitude"],
#     data,
#     cmap="Blues",
#     shading="auto"
# )
# plt.title(f"CHIRPS Annual Precipitation – Morocco ({year_to_plot})", fontsize=14)
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.colorbar(im, label="Precipitation (mm/year)")
# plt.tight_layout()
# plt.show()
"""
verify.py
---------
Visual verification of CHIRPS Morocco dataset.
Plots annual precipitation maps with consistent color scale
so that interannual differences are visible and comparable.
"""



DATA_PATH = Path("data/interim/chirps_morocco.nc")
assert DATA_PATH.exists(), f"File not found: {DATA_PATH}"

ds = xr.open_dataset(DATA_PATH)
precip = ds["precip"]

precip_yearly = precip.groupby("time.year").sum("time")

vmin = 0
vmax = 700

years_to_plot = [1996, 2010, 2014, 2018, 2023]

n = len(years_to_plot)
fig, axes = plt.subplots(1, n, figsize=(4*n, 5), constrained_layout=True)

for i, year in enumerate(years_to_plot):
    ax = axes[i]
    data = precip_yearly.sel(year=year)
    im = ax.pcolormesh(
        data["longitude"], data["latitude"], data,
        cmap="Blues", shading="auto", vmin=vmin, vmax=vmax
    )
    ax.set_title(f"{year}", fontsize=13)
    ax.set_xlabel("Lon")
    if i == 0:
        ax.set_ylabel("Lat")
    ax.set_aspect("auto")

cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
cbar.set_label("Annual Precipitation (mm)", fontsize=12)

plt.suptitle(f"CHIRPS Annual Precipitation – Morocco {years_to_plot}", fontsize=16)
plt.show()
