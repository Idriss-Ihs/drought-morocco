"""
spi_gamma_fast.py
-----------------
Compute SPI (true gamma-based) at multiple timescales with:
- Per-calendar-month fits
- Zero-inflation correction
- Method-of-moments gamma parameters (fast & robust)
- Dask-parallelized ufunc across grid

This avoids slow/fragile MLE and FitError crashes while staying faithful to SPI.
"""

from pathlib import Path
import numpy as np
import xarray as xr
import yaml
from scipy.stats import norm
from dask.diagnostics import ProgressBar
from src.utils.logger import setup_logger


# -------------------- Config --------------------

def load_config(path="src/config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------- Core math --------------------

def _gamma_mom(shape_series):
    """
    Compute Gamma(k, theta) parameters via Method of Moments from a 1D array of positives.
    Returns (k, theta) or (np.nan, np.nan) if not identifiable.
    """
    x = np.asarray(shape_series, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    n = x.size
    if n < 24:                     # need enough years per month to be stable
        return np.nan, np.nan
    m = x.mean()
    s2 = x.var(ddof=1)
    if m <= 0 or s2 <= 0:
        return np.nan, np.nan
    k = (m * m) / s2
    theta = s2 / m
    # Guard against degenerate params
    if not np.isfinite(k) or not np.isfinite(theta) or k <= 0 or theta <= 0:
        return np.nan, np.nan
    return k, theta


def _spi_1d_gamma_zero_infl(x, window):
    """
    Compute SPI for a single 1D time series (monthly) using:
    - rolling sum over 'window' months
    - per-calendar-month gamma fit (MoM)
    - zero-inflated CDF: H = q0 + (1 - q0) * G(x; k, theta)
    - z = Phi^{-1}(H)

    Returns an array with same length as input, NaN for first (window-1) months.
    """
    x = np.asarray(x, dtype=float)
    # Rolling sum
    roll = np.full_like(x, np.nan, dtype=float)
    if window >= 1:
        # cumulative sum trick, faster than pandas
        c = np.nancumsum(np.where(np.isfinite(x), x, 0.0))
        # number of valid points per window
        valid = np.cumsum(np.isfinite(x).astype(int))
        roll[window - 1:] = c[window - 1:] - np.concatenate(([0.0], c[:-window]))
        # if a window has any NaN, mark result as NaN (strict)
        nvalid = valid[window - 1:] - np.concatenate(([0], valid[:-window]))
        mask_incomplete = nvalid != window
        roll[window - 1:][mask_incomplete] = np.nan
    else:
        roll[:] = x

    # Group by calendar month index (0..11)
    months = np.arange(x.size) % 12
    out = np.full_like(roll, np.nan, dtype=float)

    for m in range(12):
        idx = (months == m)
        xm = roll[idx]
        if xm.size == 0:
            continue

        # Zero-inflation probability over history (for that calendar month)
        finite = np.isfinite(xm)
        if finite.sum() < 24:
            continue

        xm_hist = xm[finite]
        q0 = np.mean(xm_hist <= 0.0)  # P(X=0)
        # Fit Gamma on positives only using MoM
        k, theta = _gamma_mom(xm_hist[xm_hist > 0.0])
        if not np.isfinite(k) or not np.isfinite(theta):
            # can't fit -> leave NaN for this month
            continue

        # For each time point of that month: compute SPI
        xm_use = xm.copy()
        # CDF part:
        # For x <= 0: H = q0 (all mass at 0)
        # For x > 0: H = q0 + (1 - q0) * G(x; k, theta)
        # Use scipy.special.gammainc(k, x/theta) for Gamma CDF (regularized)
        from scipy.special import gammainc

        H = np.full_like(xm_use, np.nan, dtype=float)
        pos = np.isfinite(xm_use) & (xm_use > 0)
        zero = np.isfinite(xm_use) & (xm_use <= 0)

        # gamma CDF for positive x
        G = np.zeros_like(xm_use, dtype=float)
        G[pos] = gammainc(k, xm_use[pos] / theta)  # lower regularized incomplete gamma
        H[pos] = q0 + (1.0 - q0) * G[pos]
        H[zero] = q0  # exactly zero (or negative, should not happen)

        # Clamp H to (0,1) open interval to avoid infs in ppf
        eps = 1e-10
        H = np.clip(H, eps, 1.0 - eps)

        z = norm.ppf(H)
        out[idx] = z

    return out


# -------------------- Driver --------------------

def compute_spi_gamma_fast():
    cfg = load_config()
    logger = setup_logger("SPI_GAMMA_FAST")
    logger.info("Starting gamma-based SPI computation (fast/MoM)")

    ds_path = Path(cfg["paths"]["interim"]) / "chirps_morocco.nc"
    if not ds_path.exists():
        logger.error(f"Dataset not found: {ds_path}")
        return

    ds = xr.open_dataset(ds_path)
    # pick precip variable
    var = [v for v in ds.data_vars][0]
    pr = ds[var]
    pr.attrs["units"] = "mm/month"

    # Small epsilon to avoid strictly-zero windows breaking MoM variance
    pr = pr.where(np.isfinite(pr), np.nan)
    pr = pr.where(pr > 0, 0.0)  # keep zeros: we model zero-inflation explicitly

    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dask chunking for speed 
    # Chunk time fully (computes per-month groups), and moderate spatial chunks.
    if not pr.chunks:
        pr = pr.chunk({"time": -1, "latitude": 60, "longitude": 60})

    from tqdm import tqdm
    for window in tqdm(cfg["products"]["spi_timescales"], desc="SPI scales"):
        logger.info(f"Computing SPI-{window} (gamma, MoM, zero-inflated)")

        # apply per-grid 1D function with dask parallelization
        spi = xr.apply_ufunc(
            _spi_1d_gamma_zero_infl,
            pr,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            kwargs={"window": window},
            output_dtypes=[pr.dtype],
        )

        # Clip to typical SPI range
        spi = spi.clip(min=-3, max=3)
        spi.name = "spi"
        spi.attrs.update(
            {
                "long_name": f"Standardized Precipitation Index (window={window} months)",
                "method": "Gamma (Method of Moments) with zero-inflation, per-calendar-month",
                "units": "standard_score",
            }
        )

        out_path = out_dir / f"spi_{window}.nc"
        # Execute with visible progress
        with ProgressBar():
            spi.to_netcdf(out_path, compute=True)

        logger.info(f"Saved {out_path.name}")

    logger.info("✅ All gamma-based SPI computations completed.")


if __name__ == "__main__":
    compute_spi_gamma_fast()
