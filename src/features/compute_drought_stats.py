"""
compute_drought_stats.py
------------------------
Compute drought frequency & severity metrics per province and year
from monthly SPI time series.

Inputs:
    data/processed/provincial_spi.csv
        Columns like: time, province, 1, 3, 6, 12  (SPI scales)

Outputs:
    data/processed/provincial_drought_stats.csv   # per province-year per scale
    data/processed/provincial_spi_classes.csv     # monthly categorical SPI (optional, for viz)

Usage:
    python -m src.features.compute_drought_stats
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger


# -----------------------------
# Config
# -----------------------------

INPUT_CSV = Path("data/processed/provincial_spi.csv")
OUT_STATS = Path("data/processed/provincial_drought_stats.csv")
OUT_CLASSES = Path("data/processed/provincial_spi_classes.csv")

# SPI thresholds (WMO-style bins)
# negative side for drought, positive for wet
BINS = [-np.inf, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, np.inf]
LABELS = [
    "Extremely dry",   # <= -2.0
    "Severely dry",    # (-2.0, -1.5]
    "Moderately dry",  # (-1.5, -1.0]
    "Near normal",     # (-1.0, 1.0]
    "Moderately wet",  # (1.0, 1.5]
    "Very wet",        # (1.5, 2.0]
    "Extremely wet",   # > 2.0
]

# Metrics will be computed for these SPI scales if present:
SCALES = ["1", "3", "6", "12"]


# -----------------------------
# Helpers
# -----------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: '1','3','6','12' -> 'spi_1','spi_3','spi_6','spi_12'.
    Keeps 'time' (datetime) and 'province'.
    """
    rename_map = {}
    for s in SCALES:
        if s in df.columns:
            rename_map[s] = f"spi_{s}"
        elif f"spi_{s}" in df.columns:
            # already good
            pass
    if rename_map:
        df = df.rename(columns=rename_map)

    # enforce time dtype
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # keep only expected columns + whatever else
    return df


def _classify_spi(series: pd.Series) -> pd.Series:
    """Categorize SPI values into drought/wet classes for plotting/analytics."""
    return pd.cut(series, bins=BINS, labels=LABELS, right=True, include_lowest=True)


def _max_spell_length(mask: pd.Series) -> int:
    """
    Given a boolean Series (True if in drought), compute the maximum consecutive True run length.
    Example: [T,T,F,T] -> max spell = 2
    """
    if mask.empty:
        return 0
    # convert True/False to 1/0 and find longest consecutive ones
    arr = mask.astype(int).values
    # trick: when zeros break stretches of ones
    if arr.sum() == 0:
        return 0
    # compute run lengths
    max_run = 0
    current = 0
    for v in arr:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _yearly_metrics_for_scale(df: pd.DataFrame, spi_col: str) -> pd.DataFrame:
    """
    Compute yearly metrics for a single SPI scale column (e.g. 'spi_3').

    Input df columns: ['time','province', spi_col]
    Output columns (by group province-year-scale):
        - mean_spi
        - drought_months_moderate (SPI < -1.0)
        - drought_months_severe   (SPI < -1.5)
        - drought_months_extreme  (SPI <= -2.0)
        - wet_months_moderate     (SPI > 1.0)
        - wet_months_very         (SPI > 1.5)
        - wet_months_extreme      (SPI > 2.0)
        - max_drought_spell_moderate (max consecutive months with SPI < -1.0)
        - n_months (count of valid months)
    """
    df = df[["time", "province", spi_col]].dropna(subset=[spi_col]).copy()
    df["year"] = df["time"].dt.year

    # per province-year list of monthly values (to compute spells)
    def _agg(group: pd.DataFrame) -> pd.Series:
        vals = group[spi_col]
        # counts
        drought_mod = (vals < -1.0).sum()
        drought_sev = (vals < -1.5).sum()
        drought_ext = (vals <= -2.0).sum()
        wet_mod = (vals > 1.0).sum()
        wet_very = (vals > 1.5).sum()
        wet_ext = (vals > 2.0).sum()
        # spells
        max_spell = _max_spell_length(vals < -1.0)
        return pd.Series({
            "mean_spi": vals.mean(),
            "drought_months_moderate": int(drought_mod),
            "drought_months_severe": int(drought_sev),
            "drought_months_extreme": int(drought_ext),
            "wet_months_moderate": int(wet_mod),
            "wet_months_very": int(wet_very),
            "wet_months_extreme": int(wet_ext),
            "max_drought_spell_moderate": int(max_spell),
            "n_months": int(vals.notna().sum()),
        })

    out = (
        df.groupby(["province", "year"])
          .apply(_agg, include_groups=False)
          .reset_index()
    )
    out["scale"] = spi_col.replace("spi_", "")  # keep numeric like '3'
    return out


def _long_table_with_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a 'long' monthly table with one row per (time, province, scale),
    including the SPI class label for viz / dashboarding.
    """
    value_cols = [c for c in df.columns if c.startswith("spi_")]
    long_df = df.melt(id_vars=["time", "province"], value_vars=value_cols,
                      var_name="scale", value_name="spi")
    long_df["scale"] = long_df["scale"].str.replace("spi_", "", regex=False)
    long_df["spi_class"] = _classify_spi(long_df["spi"])
    return long_df


# -----------------------------
# Main
# -----------------------------

def main():
    logger = setup_logger("DROUGHT_STATS")
    logger.info("Starting provincial drought metrics computation")

    if not INPUT_CSV.exists():
        logger.error(f"Input not found: {INPUT_CSV}")
        raise SystemExit(1)

    df = pd.read_csv(INPUT_CSV)
    df = _ensure_columns(df)

    # keep only the columns we need
    keep_cols = ["time", "province"] + [c for c in df.columns if c.startswith("spi_")]
    df = df[keep_cols].copy().sort_values(["province", "time"])
    logger.info(f"Loaded monthly SPI: {df.shape[0]} rows, {len([c for c in df.columns if c.startswith('spi_')])} scales")

    # ----- monthly classes (optional but useful for viz / QA) -----
    long_monthly = _long_table_with_classes(df)
    OUT_CLASSES.parent.mkdir(parents=True, exist_ok=True)
    long_monthly.to_csv(OUT_CLASSES, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote monthly SPI classes: {OUT_CLASSES} ({long_monthly.shape[0]} rows)")

    # ----- yearly metrics per scale -----
    yearly_blocks = []
    for s in SCALES:
        col = f"spi_{s}"
        if col in df.columns:
            block = _yearly_metrics_for_scale(df[["time", "province", col]], col)
            yearly_blocks.append(block)
            logger.info(f"Computed yearly metrics for SPI-{s}: {block.shape[0]} rows")
        else:
            logger.warning(f"Missing column for SPI-{s}, skipping.")

    if not yearly_blocks:
        logger.error("No SPI columns found. Aborting.")
        raise SystemExit(1)

    stats = pd.concat(yearly_blocks, ignore_index=True)

    # Tidy ordering
    metric_cols = [
        "mean_spi",
        "drought_months_moderate",
        "drought_months_severe",
        "drought_months_extreme",
        "wet_months_moderate",
        "wet_months_very",
        "wet_months_extreme",
        "max_drought_spell_moderate",
        "n_months",
    ]
    stats = stats[["province", "year", "scale"] + metric_cols].sort_values(["province", "year", "scale"])

    # save
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")
    logger.info(f"✅ Wrote provincial drought stats: {OUT_STATS} ({stats.shape[0]} rows)")

    # quick console preview
    print(stats.head(10))


if __name__ == "__main__":
    main()
