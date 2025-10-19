import os
import xarray as xr
import rioxarray
import requests
from pathlib import Path
import yaml
from tqdm import tqdm
from src.utils.logger import setup_logger
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_config(path="src/config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def download_chirps(years, out_dir):
    """Download CHIRPS monthly .nc files for given years."""
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/byYear/"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for y in tqdm(years, desc="Downloading CHIRPS"):
        url = f"{base_url}chirps-v2.0.{y}.monthly.nc"
        out_file = out_dir / f"chirps_{y}.nc"
        if out_file.exists():
            continue
        try:
            r = requests.get(url, stream=True, timeout=60, verify=False)
            r.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            logger.warning(f"Failed to download {y}: {e}")

def subset_morocco(cfg):
    """Clip CHIRPS global to Morocco bbox."""
    raw_path = Path(cfg["paths"]["raw"])
    interim_path = Path(cfg["paths"]["interim"])
    interim_path.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_path.glob("chirps_*.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sel(
        latitude=slice(cfg["project"]["bbox"]["lat_max"], cfg["project"]["bbox"]["lat_min"]),
        longitude=slice(cfg["project"]["bbox"]["lon_min"], cfg["project"]["bbox"]["lon_max"]),
    )
    ds.to_netcdf(interim_path / "chirps_morocco.nc")
    logger.info("Saved subset → data/interim/chirps_morocco.nc")


if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger("CHIRPS")
    years = range(2020, 2025)
    logger.info("Starting CHIRPS data ingestion pipeline")
    download_chirps(years, cfg["paths"]["raw"])
    subset_morocco(cfg)
    logger.info("CHIRPS ingestion pipeline completed successfully")
