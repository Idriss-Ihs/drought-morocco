# import xarray as xr, glob

# for f in glob.glob("data/raw/chirps_*.nc"):
#     try:
#         xr.open_dataset(f).close()
#     except Exception as e:
#         print(f"❌ {f} - {e}")

import xarray as xr
from pathlib import Path

path = Path("data/interim/chirps_morocco.nc")

ds = xr.open_dataset(path)
print(ds)
