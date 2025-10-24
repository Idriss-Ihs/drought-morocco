"""
plot_provincial_trends.py
-------------------------
Time-series visuals for provincial SPI:
  1) Single-province line chart across years
  2) Small multiples (facets) for several provinces
  3) Province × Year heatmap of mean SPI

Inputs:
  - data/processed/provincial_drought_stats.csv

Usage:
  # Single province (SPI-12 trend)
  python -m src.viz.plot_provincial_trends --province "Safi" --scale 12 --kind line

  # Small multiples for a list of provinces
  python -m src.viz.plot_provincial_trends --provinces "Safi,Marrakech,Agadir" --scale 3 --kind facets

  # Heatmap (all provinces)
  python -m src.viz.plot_provincial_trends --scale 12 --kind heatmap
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import setup_logger


STATS_CSV = Path("data/processed/provincial_drought_stats.csv")


def load_stats():
    df = pd.read_csv(STATS_CSV)
    df["scale"] = df["scale"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def plot_single_province_line(df, province: str, scale: str = "12"):
    sub = df[(df["province"] == province) & (df["scale"] == scale)].sort_values("year")
    if sub.empty:
        raise ValueError(f"No data for province='{province}', scale='{scale}'")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sub["year"], sub["mean_spi"], marker="o", lw=1)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_title(f"{province} — SPI-{scale} mean (annual)", fontweight="bold")
    ax.set_ylabel("Mean SPI")
    ax.set_xlabel("Year")
    plt.tight_layout()
    out = Path("docs/figures") / f"trend_{province.replace(' ','_')}_spi{scale}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.show()


def plot_facets(df, provinces: list[str], scale: str = "3"):
    sub = df[(df["province"].isin(provinces)) & (df["scale"] == scale)].copy()
    if sub.empty:
        raise ValueError("No data for given provinces/scale.")
    g = sns.FacetGrid(sub, col="province", col_wrap=3, sharey=True, height=3.0)
    g.map_dataframe(sns.lineplot, x="year", y="mean_spi")
    for ax in g.axes.flat:
        ax.axhline(0, color="0.3", lw=1)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean SPI")
    g.fig.suptitle(f"SPI-{scale} mean (annual) — selected provinces", y=1.02, fontweight="bold")
    plt.tight_layout()
    out = Path("docs/figures") / f"trends_facets_spi{scale}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.show()


def plot_heatmap(df, scale: str = "12"):
    sub = df[df["scale"] == scale].copy()
    # pivot so rows=province, cols=year
    mat = sub.pivot(index="province", columns="year", values="mean_spi").sort_index()
    plt.figure(figsize=(14, max(6, 0.18 * len(mat))))
    sns.heatmap(mat, cmap="RdBu_r", vmin=-2.0, vmax=2.0, center=0.0, linewidths=0.2, linecolor="0.85")
    plt.title(f"Mean SPI (annual) — SPI-{scale}", fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Province")
    plt.tight_layout()
    out = Path("docs/figures") / f"heatmap_spi{scale}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", type=str, choices=["line", "facets", "heatmap"], required=True)
    p.add_argument("--province", type=str, help="Single province name for --kind line")
    p.add_argument("--provinces", type=str, help="Comma-separated list for --kind facets")
    p.add_argument("--scale", type=str, default="12", help="SPI scale (1,3,6,12)")
    return p.parse_args()


if __name__ == "__main__":
    logger = setup_logger("PROV_TRENDS")
    args = parse_args()
    df = load_stats()
    if args.kind == "line":
        if not args.province:
            raise SystemExit("Please provide --province 'Name'")
        plot_single_province_line(df, args.province, scale=args.scale)
    elif args.kind == "facets":
        if not args.provinces:
            raise SystemExit("Please provide --provinces 'A,B,C'")
        provinces = [p.strip() for p in args.provinces.split(",")]
        plot_facets(df, provinces, scale=args.scale)
    else:
        plot_heatmap(df, scale=args.scale)
