"""Luas ridership analysis.

Reads weekly passenger journeys by mode of public transport (CSO export),
cleans them, and produces the tables and charts used in the project:

- weekly share of passengers by mode
- yearly totals and year-on-year growth by mode
- outlier weeks (z-score beyond +/-3)
- correlation between modes
- charts saved to outputs/

Usage:
    python src/analysis.py                                  # runs on the sample data
    python src/analysis.py data/cso_weekly_passengers.csv   # runs on a real CSO export
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "sample" / "sample_weekly_passengers.csv"
OUTPUTS = ROOT / "outputs"
Z_THRESHOLD = 3


def find_column(df, *keywords):
    """Return the first column whose name contains any keyword (case-insensitive)."""
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    raise KeyError(f"No column matching {keywords} in {list(df.columns)}")


def load(path):
    """Load a CSO-style CSV into a tidy table: year, week, mode, passengers."""
    raw = pd.read_csv(path)
    week_col = find_column(raw, "week")
    mode_col = find_column(raw, "mode", "transport")
    value_col = find_column(raw, "value")

    df = raw[[week_col, mode_col, value_col]].copy()
    df.columns = ["week_label", "mode", "passengers"]

    # CSO labels weeks like "2023 Week 01"
    parts = df["week_label"].astype(str).str.extract(r"(\d{4})\D+(\d{1,2})")
    df["year"] = pd.to_numeric(parts[0], errors="coerce")
    df["week"] = pd.to_numeric(parts[1], errors="coerce")
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
    df["mode"] = df["mode"].astype(str).str.strip()
    return df


def clean(df):
    """Drop rows we can't use and report what was removed."""
    before = len(df)
    df = df.dropna(subset=["year", "week", "passengers"])
    df = df[~df["mode"].str.lower().str.contains("all")]  # drop CSO's "All modes" total row
    df = df.drop_duplicates(subset=["year", "week", "mode"])
    df = df[df["passengers"] >= 0]
    df[["year", "week"]] = df[["year", "week"]].astype(int)
    print(f"Cleaning: kept {len(df)} of {before} rows")
    return df.sort_values(["mode", "year", "week"]).reset_index(drop=True)


def weekly_share(df):
    wide = df.pivot_table(index=["year", "week"], columns="mode", values="passengers")
    share = wide.div(wide.sum(axis=1), axis=0) * 100
    return wide, share.round(2)


def yearly_growth(df):
    totals = df.groupby(["mode", "year"])["passengers"].sum().unstack("mode")
    growth = (totals.pct_change() * 100).round(1)
    return totals, growth


def outliers(df):
    # Compare each week with the same mode in the same year. Across all years,
    # the COVID drop makes the spread so wide that real spikes would be hidden.
    stats = df.groupby(["mode", "year"])["passengers"].agg(["mean", "std"])
    out = df.join(stats, on=["mode", "year"])
    out["z_score"] = ((out["passengers"] - out["mean"]) / out["std"]).round(2)
    return out.loc[out["z_score"].abs() > Z_THRESHOLD, ["year", "week", "mode", "passengers", "z_score"]]


def plot_weekly(wide):
    modes = list(wide.columns)
    fig, axes = plt.subplots(len(modes), 1, figsize=(10, 2.6 * len(modes)), sharex=True)
    if len(modes) == 1:
        axes = [axes]
    x = [f"{y}-W{w:02d}" for y, w in wide.index]
    for ax, mode in zip(axes, modes):
        ax.plot(range(len(x)), wide[mode], linewidth=1.2)
        ax.set_title(mode, loc="left", fontsize=10)
        ax.set_ylabel("Passengers (m)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}"))
        ax.grid(alpha=0.3)
    step = max(len(x) // 10, 1)
    axes[-1].set_xticks(range(0, len(x), step))
    axes[-1].set_xticklabels(x[::step], rotation=45, ha="right", fontsize=8)
    fig.suptitle("Weekly passenger journeys by mode", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "weekly_trends_by_mode.png", dpi=150)
    plt.close(fig)


def plot_yearly(totals):
    ax = (totals / 1e6).plot(kind="line", marker="o", figsize=(8, 4.5))
    ax.set_title("Yearly passenger journeys by mode")
    ax.set_ylabel("Passengers (millions)")
    ax.set_xticks(totals.index)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "yearly_totals_by_mode.png", dpi=150)
    plt.close()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    OUTPUTS.mkdir(exist_ok=True)
    print(f"Reading {path}")

    df = clean(load(path))
    wide, share = weekly_share(df)
    totals, growth = yearly_growth(df)
    flagged = outliers(df)
    corr = wide.corr().round(2)

    share.to_csv(OUTPUTS / "weekly_share_by_mode.csv")
    totals.to_csv(OUTPUTS / "yearly_totals_by_mode.csv")
    growth.to_csv(OUTPUTS / "yearly_growth_pct_by_mode.csv")
    flagged.to_csv(OUTPUTS / "outlier_weeks.csv", index=False)
    corr.to_csv(OUTPUTS / "correlation_between_modes.csv")
    plot_weekly(wide)
    plot_yearly(totals)

    print("\nYear-on-year growth (%):")
    print(growth.to_string())
    print(f"\nOutlier weeks (|z| > {Z_THRESHOLD}): {len(flagged)}")
    print("\nCorrelation between modes:")
    print(corr.to_string())
    print(f"\nTables and charts saved to {OUTPUTS}")


if __name__ == "__main__":
    main()
