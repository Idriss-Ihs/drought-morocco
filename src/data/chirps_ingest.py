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

def download_chirps(years, out_dir, logger):
    """Download CHIRPS monthly .nc files for given years."""
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/byYear/"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting CHIRPS downloads ({years.start}–{years.stop - 1})")

    for y in tqdm(years, desc="Downloading CHIRPS"):
        url = f"{base_url}chirps-v2.0.{y}.monthly.nc"
        out_file = out_dir / f"chirps_{y}.nc"
        if out_file.exists():
            logger.info(f"{y}: already exists, skipping.")
            continue
        try:
            r = requests.get(url, stream=True, timeout=60, verify=False)
            r.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"{y}: downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download {y}: {e}")
    logger.info("CHIRPS download stage completed.")


def subset_morocco(cfg, logger):
    """
    Merge all downloaded CHIRPS files and clip them to Morocco's bounding box.
    Output: data/interim/chirps_morocco.nc
    """
    raw_path = Path(cfg["paths"]["raw"])
    interim_path = Path(cfg["paths"]["interim"])
    interim_path.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_path.glob("chirps_*.nc"))
    if not files:
        logger.error("No CHIRPS files found to merge — aborting.")
        return

    logger.info(f"Merging {len(files)} CHIRPS files...")

    # safer multi-file open
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12})

    # --- rebuild clean coordinate variables ---
    lat = ds.latitude.values
    lon = ds.longitude.values

    ds = ds.assign_coords(
        latitude=("latitude", lat.astype("float32")),
        longitude=("longitude", lon.astype("float32"))
    )

    # remove any problematic encodings inherited from CHIRPS
    for var in ds.coords:
        ds[var].encoding.clear()

    # --- Determine correct slice direction ---
    lat = ds.latitude
    bbox = cfg["project"]["bbox"]

    if lat.values[0] > lat.values[-1]:
        # latitude descending
        ds_sub = ds.sel(
            latitude=slice(bbox["lat_max"], bbox["lat_min"]),
            longitude=slice(bbox["lon_min"], bbox["lon_max"])
        )
    else:
        # latitude ascending
        ds_sub = ds.sel(
            latitude=slice(bbox["lat_min"], bbox["lat_max"]),
            longitude=slice(bbox["lon_min"], bbox["lon_max"])
        )
        # make sure we don't carry encoding trouble to disk
        for v in ds_sub.variables:
            ds_sub[v].encoding.clear()

    output_path = interim_path / "chirps_morocco.nc"
    ds_sub.to_netcdf(output_path, engine="netcdf4", format="NETCDF4")

    logger.info(f"Saved subset to {output_path}")




if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger("CHIRPS")
    years = range(1981, 2026)
    logger.info("==== Starting CHIRPS data ingestion pipeline ====")
    download_chirps(years, cfg["paths"]["raw"], logger)
    subset_morocco(cfg, logger)
    logger.info("==== CHIRPS ingestion pipeline completed successfully ====")
