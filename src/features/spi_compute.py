import xarray as xr
import xclim.indices as xc
from pathlib import Path
import yaml
from tqdm import tqdm
from src.utils.logger import setup_logger


def load_config(path="src/config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_spi():
    cfg = load_config()

    logger = setup_logger("SPI")
    logger.info("Starting SPI computation")
    ds = xr.open_dataset(Path(cfg["paths"]["interim"]) / "chirps_morocco.nc")

    # precipitation variable may be 'precip' or 'precipitation'
    var = [v for v in ds.data_vars][0]
    pr = ds[var]
    pr.attrs["units"] = "mm/month"

    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for scale in tqdm(cfg["products"]["spi_timescales"], desc="SPI scales"):
        spi = xc.standardized_precipitation_index(pr, freq="MS", window=scale)
        output = out_dir / f"spi_{scale}.nc"
        spi.to_netcdf(output)
        logger.info(f"Saved {output.name}")

    logger.info("SPI computation completed successfully")

if __name__ == "__main__":
    compute_spi()
