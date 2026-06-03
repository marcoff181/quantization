#!/usr/bin/env python3
"""
Analysis of CLIP-D detection modes on REAL images (no model/quantization breakdown).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ----- paths & constants -----
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _ROOT / "results" / "img2img" / "detector_real_all_modes.csv"
DEFAULT_FIGURES_DIR = _ROOT / "figures" / "img2img_default"

# Seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# =============================================================================
# Data preparation
# =============================================================================
def parse_detector_mode(mode_str: str) -> tuple[str, str | None]:
    """
    Split the detector_mode field.
    e.g. 'grid_max' → ('grid', 'max')
         'patch_mean' → ('patch', 'mean')
         'default' → ('default', None)
    """
    if not isinstance(mode_str, str):
        return "unknown", None
    parts = mode_str.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return parts[0], None


def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw DataFrame."""
    result = df.copy()

    # Detector mode → mode_type & aggregation
    if "detector_mode" in result.columns:
        parsed = result["detector_mode"].apply(parse_detector_mode)
        result["mode_type"] = parsed.apply(lambda v: v[0])
        result["aggregation"] = parsed.apply(lambda v: v[1])
        result["mode_label"] = result.apply(
            lambda row: f"{row['mode_type']}_{row['aggregation']}"
            if row["aggregation"] is not None
            else row["mode_type"],
            axis=1,
        )
    else:
        result["mode_type"] = "unknown"
        result["aggregation"] = None
        result["mode_label"] = "unknown"

    # Convert numeric columns
    for col in ("confidence", "elapsed_time"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


# =============================================================================
# Summary statistics
# =============================================================================
def build_summary(df: pd.DataFrame, group_cols: list[str] = None) -> pd.DataFrame:
    """Compute per-group statistics."""
    if group_cols is None:
        group_cols = []
    else:
        group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols:
        # Global summary
        total = len(df)
        fake = (df["prediction"] == "fake").sum()
        avg_conf = df["confidence"].mean()
        avg_elapsed = df["elapsed_time"].mean() if "elapsed_time" in df.columns else None
        return pd.DataFrame(
            [
                {
                    "total": total,
                    "fake_predictions": fake,
                    "real_predictions": total - fake,
                    "fake_rate": fake / total if total > 0 else 0,
                    "real_rate": (total - fake) / total if total > 0 else 0,
                    "avg_confidence_on_fake": df.loc[
                        df["prediction"] == "fake", "confidence"
                    ].mean(),
                    "avg_confidence_on_real": df.loc[
                        df["prediction"] == "real", "confidence"
                    ].mean(),
                    "avg_confidence_overall": avg_conf,
                    "avg_elapsed_time": avg_elapsed,
                }
            ]
        )

    # Grouped summary
    grouped = df.groupby(group_cols)
    rows = []
    for name, group in grouped:
        if not isinstance(name, tuple):
            name = (name,)
        total = len(group)
        fake = (group["prediction"] == "fake").sum()
        avg_conf_fake = group.loc[group["prediction"] == "fake", "confidence"].mean()
        avg_conf_real = group.loc[group["prediction"] == "real", "confidence"].mean()
        real_rate = (total - fake) / total if total > 0 else 0
        avg_conf_all = group["confidence"].mean()
        avg_elapsed = group["elapsed_time"].mean() if "elapsed_time" in group.columns else None

        row = {col: val for col, val in zip(group_cols, name)}
        row.update(
            {
                "total": total,
                "fake_predictions": fake,
                "real_predictions": total - fake,
                "real_rate": real_rate,
                "fake_rate": fake / total if total > 0 else 0,
                "avg_confidence_on_fake": avg_conf_fake,
                "avg_confidence_on_real": avg_conf_real,
                "avg_confidence_overall": avg_conf_all,
                "avg_elapsed_time": avg_elapsed,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Simple bar chart by mode
# =============================================================================
def plot_mode_bars(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    ylabel: str,
    fmt: str = ".2f",
):
    """Create a simple bar chart: x = mode_label, y = metric."""
    data = df.dropna(subset=["mode_label", metric]).copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    sns.barplot(
        data=data,
        x="mode_label",
        y=metric,
        palette="Set2",
        hue="mode_label",
        ax=ax,
        errorbar=None,
        estimator="mean",
        linewidth=1.5,
        edgecolor="white",
    )

    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if pd.notna(height):
            ax.annotate(
                f"{height:{fmt}}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.set_xlabel("Detection Mode", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================
def main(csv_path: str) -> None:
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    # Load and prepare
    df = pd.read_csv(csv_file)
    df = prepare_results(df)

    # Keep only CLIP-D entries
    clipd = df[df["detector"] == "CLIP-D"].copy()
    if clipd.empty:
        print("No CLIP-D results found. Exiting.")
        return

    print("\n" + "=" * 70)
    print("CLIP-D DETECTION MODES – REAL IMAGES ANALYSIS")
    print("=" * 70)

    # --- Overall metrics per mode ---
    overall = build_summary(clipd, group_cols=["mode_label"])
    print("\nPerformance per detection mode:\n")
    print(overall.to_string(index=False))

    # --- Visual comparisons ---
    print("\nGenerating comparison figures...")

    # Fake rate
    plot_mode_bars(
        overall,
        metric="fake_rate",
        title="Real Images: False Positive Rate by Detection Mode",
        output_path=DEFAULT_FIGURES_DIR / "real_fake_rate_by_mode.png",
        ylabel="False Positive Rate (% detected as fake)",
        fmt=".1%",
    )

    # Confidence
    plot_mode_bars(
        overall,
        metric="avg_confidence_on_real",
        title="Real Images: Average Confidence on Real Predictions by Detection Mode",
        output_path=DEFAULT_FIGURES_DIR / "real_confidence_by_mode.png",
        ylabel="Average Confidence on Real",
        fmt=".3f",
    )

    # Elapsed time
    plot_mode_bars(
        overall,
        metric="avg_elapsed_time",
        title="Real Images: Detection Time by Mode",
        output_path=DEFAULT_FIGURES_DIR / "real_elapsed_time_by_mode.png",
        ylabel="Time (seconds)",
        fmt=".2f",
    )
    
    plot_mode_bars(
        overall,
        metric="real_rate",
        title="Real Images: True Positive Rate by Detection Mode",
        output_path=DEFAULT_FIGURES_DIR / "real_true_positive_rate_by_mode.png",
        ylabel="True Positive Rate (% detected as real)",
        fmt=".1%",
    )

    print(f"All figures saved to: {DEFAULT_FIGURES_DIR}\n")
    print("Analysis complete.")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CSV)
    main(csv_arg)