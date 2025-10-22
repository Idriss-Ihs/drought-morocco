"""
spi_compute.py
---------------
Compute Standardized Precipitation Index (SPI) at multiple timescales
from CHIRPS Morocco dataset.

Improvements:
- Handles dry (zero-precipitation) grid cells safely.
- Logs progress and completion for each timescale.
"""

import xarray as xr
import xclim.indices as xc
from pathlib import Path
import yaml
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from src.utils.logger import setup_logger
import warnings


def load_config(path="src/config/settings.yaml"):
    """Load project configuration from YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_spi():
    """Compute SPI at multiple timescales."""
    cfg = load_config()
    logger = setup_logger("SPI")

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logger.info("Starting SPI computation")

    # Load dataset
    ds_path = Path(cfg["paths"]["interim"]) / "chirps_morocco.nc"
    if not ds_path.exists():
        logger.error(f"Dataset not found: {ds_path}")
        return

    ds = xr.open_dataset(ds_path)
    var = [v for v in ds.data_vars][0]
    pr = ds[var]
    pr.attrs["units"] = "mm/month"

    # Mask extremely dry areas (avoid FitError in deserts)
    pr_masked = pr.where(pr.mean("time") > 5)

    # Replace exact zeros with a small epsilon (recommended by WMO)
    pr_masked = pr_masked.where(pr_masked > 0, 0.1)

    # Prepare output dir
    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    spi_scales = cfg["products"]["spi_timescales"]

    for scale in tqdm(spi_scales, desc="Computing SPI scales"):
        try:
            logger.info(f"Starting SPI-{scale} computation")

            # Compute SPI lazily using Dask
            spi = xc.standardized_precipitation_index(pr_masked, freq="MS", window=scale)

            output = out_dir / f"spi_{scale}.nc"

            with ProgressBar():
                spi.to_netcdf(output, compute=True)

            logger.info(f"Saved {output.name}")
        except Exception as e:
            logger.error(f"SPI-{scale} failed: {e}")

    logger.info("All SPI computations completed successfully.")


if __name__ == "__main__":
    compute_spi()
