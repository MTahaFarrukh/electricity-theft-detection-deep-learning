"""Exploratory data analysis on the TRAINING split only.

Validation and test customers are excluded so that nothing seen here can
influence modelling decisions with held-out data.

    python scripts/eda.py        # needs data/raw/data.csv and data/processed/splits.csv

Outputs:
    reports/figures/eda_*.png
    reports/results/eda_summary.md
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import ID_COL, load_sgcc  # noqa: E402

# Colour-blind-safe categorical pair (slots 1 and 2 of the validated reference palette).
CLASS_STYLE = {0: ("Normal", "#2a78d6"), 1: ("Theft-labelled", "#eb6834")}
MIN_OBSERVED_PER_DAY = 50
INK, INK_2, MUTED, GRID, SURFACE ="#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
})


def save(fig, out_dir: Path, name: str) -> str:
    path = out_dir / f"eda_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def customer_features(x: np.ndarray) -> pd.DataFrame:
    """Simple per-customer statistics (also candidate baseline features for Phase 5)."""
    with np.errstate(all="ignore"):
        mean = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)
        observed = ~np.isnan(x)
        return pd.DataFrame({
            "mean_kwh": mean,
            "median_kwh": np.nanmedian(x, axis=1),
            "std_kwh": std,
            "coef_variation": std / mean,
            "max_over_mean": np.nanmax(x, axis=1) / mean,
            "zero_fraction": np.nansum(x == 0, axis=1) / observed.sum(axis=1),
            "missing_fraction": 1 - observed.mean(axis=1),
        })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "data.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data" / "processed" / "splits.csv")
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "reports" / "results")
    args = parser.parse_args()

    if not args.splits.exists():
        print("ERROR: run scripts/preprocess.py first (needs the customer split).")
        return 1
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    data = load_sgcc(args.input)
    splits = pd.read_csv(args.splits, dtype={ID_COL: str})
    train_ids = set(splits.loc[splits["split"] == "train", ID_COL])
    keep = np.array([str(i) in train_ids for i in data.ids])
    x, y, dates = data.readings[keep], data.labels[keep], data.dates
    print(f"EDA on training split: {len(y)} customers ({y.sum()} theft-labelled)")

    figures, feats = [], customer_features(x)
    feats["label"] = y

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(6, 2.4))
    counts = [int((y == c).sum()) for c in (0, 1)]
    ax.barh([CLASS_STYLE[c][0] for c in (0, 1)], counts, color=[CLASS_STYLE[c][1] for c in (0, 1)], height=0.5)
    for i, n in enumerate(counts):
        ax.text(n, i, f"  {n:,} ({n / len(y):.1%})", va="center", color=INK_2)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_title("Class balance (training customers)")
    ax.grid(axis="y", visible=False)
    figures.append(("Class balance", save(fig, args.fig_dir, "class_balance")))

    # 2. Median daily consumption over time
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for c in (0, 1):
        with np.errstate(all="ignore"):
            median = np.nanmedian(x[y == c], axis=0)
        # Days with almost no readings (e.g. 2014-10-03) give meaningless medians; leave them out.
        median[(~np.isnan(x[y == c])).sum(axis=0) < MIN_OBSERVED_PER_DAY] = np.nan
        daily = pd.Series(median, index=dates).rolling(7, min_periods=1).mean()
        ax.plot(daily.index, daily.values, color=CLASS_STYLE[c][1], label=CLASS_STYLE[c][0])
    ax.set_title("Median daily consumption across customers (7-day rolling mean)")
    ax.set_ylabel("kWh")
    ax.legend(loc="upper left")
    figures.append(("Consumption over time", save(fig, args.fig_dir, "consumption_over_time")))

    # 3. Missing readings over time
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for c in (0, 1):
        ax.plot(dates, np.isnan(x[y == c]).mean(axis=0) * 100, color=CLASS_STYLE[c][1], label=CLASS_STYLE[c][0])
    ax.set_title("Share of customers with a missing reading, per day")
    ax.set_ylabel("% missing")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    figures.append(("Missing readings over time", save(fig, args.fig_dir, "missing_over_time")))

    # 4-5. Per-customer distributions
    for col, title, log, name in [
        ("mean_kwh", "Per-customer mean daily consumption", True, "customer_mean"),
        ("missing_fraction", "Per-customer share of missing readings", False, "customer_missing"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        values = feats[col].replace([np.inf, -np.inf], np.nan)
        bins = (np.logspace(-2, 3.5, 50) if log else np.linspace(0, 1, 41))
        for c in (0, 1):
            ax.hist(values[feats.label == c].dropna(), bins=bins, density=True, histtype="step",
                    linewidth=2, color=CLASS_STYLE[c][1], label=CLASS_STYLE[c][0])
        if log:
            ax.set_xscale("log")
            ax.set_xlabel("mean kWh/day (log scale)")
        else:
            ax.set_xlabel("fraction of days missing")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend()
        figures.append((title, save(fig, args.fig_dir, name)))

    # 6. Weekly profile (each customer normalised by its own mean)
    with np.errstate(all="ignore"):
        rel = x / np.nanmean(x, axis=1, keepdims=True)
    weekday = dates.dayofweek.to_numpy()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 0.38
    for k, c in enumerate((0, 1)):
        with np.errstate(all="ignore"):
            prof = [np.nanmean(rel[y == c][:, weekday == d]) for d in range(7)]
        ax.bar(np.arange(7) + (k - 0.5) * width, prof, width=width - 0.03,
               color=CLASS_STYLE[c][1], label=CLASS_STYLE[c][0])
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set_xticks(range(7), names)
    ax.set_ylim(0.9, None)
    ax.set_ylabel("consumption / customer's mean")
    ax.set_title("Weekly profile")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    figures.append(("Weekly profile", save(fig, args.fig_dir, "weekly_profile")))

    # 7. Example customers (small multiples, shared y-axis per row)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    for row, c in enumerate((0, 1)):
        candidates = np.flatnonzero((y == c) & (feats.missing_fraction.to_numpy() < 0.05))
        for ax, i in zip(axes[row], rng.choice(candidates, 3, replace=False)):
            ax.plot(dates, x[i], color=CLASS_STYLE[c][1], linewidth=0.8)
            ax.set_title(f"{CLASS_STYLE[c][0]} customer", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("kWh/day")
    fig.autofmt_xdate()
    figures.append(("Example customers", save(fig, args.fig_dir, "example_customers")))

    # Summary table
    stats = feats.drop(columns="label").replace([np.inf, -np.inf], np.nan).groupby(feats.label).median().T
    stats.columns = [CLASS_STYLE[c][0] for c in stats.columns]
    stats["theft / normal"] = stats.iloc[:, 1] / stats.iloc[:, 0]
    lines = [
        "# EDA summary (training split only)",
        "",
        "_Generated by `scripts/eda.py`. Validation and test customers are excluded._",
        "",
        f"- Customers: {len(y):,} ({int(y.sum()):,} theft-labelled, {y.mean():.2%})",
        f"- Days: {len(dates)} ({dates[0].date()} .. {dates[-1].date()})",
        "",
        "## Per-customer statistics (median across customers)",
        "",
        "| Statistic | Normal | Theft-labelled | Theft / normal |",
        "|---|---|---|---|",
    ]
    lines += [f"| {name} | {r.iloc[0]:.3f} | {r.iloc[1]:.3f} | {r.iloc[2]:.2f} |" for name, r in stats.iterrows()]
    lines += ["", "## Figures", ""] + [f"- {title}: `reports/figures/{f}`" for title, f in figures]
    (args.results_dir / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
