"""
verify.py
---------
Quick visual verification of the CHIRPS Morocco dataset.
Plots annual precipitation maps to confirm spatial coverage and values.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1️⃣ Load dataset
DATA_PATH = Path("data/interim/chirps_morocco.nc")
assert DATA_PATH.exists(), f"File not found: {DATA_PATH}"

ds = xr.open_dataset(DATA_PATH)
precip = ds["precip"]

# 2️⃣ Sanity checks
print(f"Dataset dimensions: {dict(ds.sizes)}")
print(f"Time range: {str(precip.time.values[0])[:10]} → {str(precip.time.values[-1])[:10]}")
print(f"Mean precip: {float(precip.mean().values):.2f} mm/month")
print(f"Max precip: {float(precip.max().values):.2f} mm/month")

# 3️⃣ Compute yearly sum (mm/year)
precip_yearly = precip.groupby("time.year").sum("time")

# 4️⃣ Select one year to visualize
year_to_plot = 2020  # change freely
data = precip_yearly.sel(year=year_to_plot)

# 5️⃣ Plot
plt.figure(figsize=(8, 6))
im = plt.pcolormesh(
    data["longitude"],
    data["latitude"],
    data,
    cmap="Blues",
    shading="auto"
)
plt.title(f"CHIRPS Annual Precipitation – Morocco ({year_to_plot})", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(im, label="Precipitation (mm/year)")
plt.tight_layout()
plt.show()
