#!/usr/bin/env python3
"""
Professional comparison of CLIP‑D detection modes for img2img outputs.

The input CSV must contain the columns:
    batch_label, detector, detector_mode, image_path, prediction, confidence, elapsed_time

The script produces:
  1. Console tables with performance metrics grouped by mode, model, and quantization.
  2. Grouped bar‑chart figures (one per metric) saved as high‑resolution PNG files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ----- paths & constants -----
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _ROOT / "results" / "img2img" / "detector_img2img_results.csv"
DEFAULT_FIGURES_DIR = _ROOT / "figures" / "mode_comparison"
KNOWN_QUANTIZATIONS = {"fp16", "fp8", "fp4"}

# Seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# =============================================================================
# Data preparation
# =============================================================================
def extract_batch_parts(label: object) -> tuple[str, str]:
    """Split a batch_label like 'sd35_fp16' into (model, quantization)."""
    text = str(label)
    if "_" not in text:
        return text, "unknown"
    model, quantization = text.rsplit("_", 1)
    if quantization not in KNOWN_QUANTIZATIONS:
        return text, "unknown"
    return model, quantization


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

    # Batch label → model & quantization
    if "batch_label" in result.columns:
        batch_parts = result["batch_label"].apply(extract_batch_parts)
        result["model"] = batch_parts.apply(lambda v: v[0])
        result["quantization"] = batch_parts.apply(lambda v: v[1])
    else:
        result["model"] = "unknown"
        result["quantization"] = "unknown"

    # Detector mode → mode_type & aggregation
    if "detector_mode" in result.columns:
        parsed = result["detector_mode"].apply(parse_detector_mode)
        result["mode_type"] = parsed.apply(lambda v: v[0])
        result["aggregation"] = parsed.apply(lambda v: v[1])
        # Create a combined label for plotting (if aggregation present, use it; else just mode_type)
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
    """
    Compute per‑group statistics.
    If group_cols is None, the summary is global (across the whole df).
    """
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
                    "avg_confidence_on_fake": df.loc[
                        df["prediction"] == "fake", "confidence"
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
        avg_conf_all = group["confidence"].mean()
        avg_elapsed = group["elapsed_time"].mean() if "elapsed_time" in group.columns else None

        row = {col: val for col, val in zip(group_cols, name)}
        row.update(
            {
                "total": total,
                "fake_predictions": fake,
                "real_predictions": total - fake,
                "fake_rate": fake / total if total > 0 else 0,
                "avg_confidence_on_fake": avg_conf_fake,
                "avg_confidence_overall": avg_conf_all,
                "avg_elapsed_time": avg_elapsed,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Professional bar‑chart comparison of modes
# =============================================================================
def plot_mode_comparison_bars(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    ylabel: str,
    fmt: str = ".2f",
):
    """
    Creates a figure with one subplot per model.
    Each subplot shows grouped bars: x = quantization, hue = mode_label.
    """
    # Prepare clean data
    data = df.dropna(subset=["model", "quantization", "mode_label", metric]).copy()
    if data.empty:
        return

    models = sorted(data["model"].unique())
    n_models = len(models)
    n_cols = 2 if n_models > 1 else 1
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), constrained_layout=True
    )
    axes = [axes] if n_models == 1 else list(axes.flat)

    # Consistent colour palette across all subplots
    hue_order = sorted(data["mode_label"].unique())
    palette = sns.color_palette("Set2", n_colors=len(hue_order))

    for i, ax in enumerate(axes):
        if i >= n_models:
            ax.axis("off")
            continue

        model = models[i]
        sub = data[data["model"] == model]

        if sub.empty:
            ax.axis("off")
            continue

        # Order quantizations as fp16 → fp8 → fp4
        quant_order = [q for q in ["fp16", "fp8", "fp4"] if q in sub["quantization"].unique()]
        sub["quantization"] = pd.Categorical(sub["quantization"], categories=quant_order, ordered=True)

        # Grouped bar plot
        sns.barplot(
            data=sub,
            x="quantization",
            y=metric,
            hue="mode_label",
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            errorbar=None,           # we already use mean; you can switch to "sd" or "ci" if needed
            estimator="mean",
            linewidth=1,
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
                    fontsize=8,
                    fontweight="bold",
                    xytext=(0, 2),
                    textcoords="offset points",
                )

        ax.set_title(f"Model: {model}", fontweight="bold", fontsize=13)
        ax.set_xlabel("Quantization", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Detection Mode", fontsize=9, title_fontsize=10)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="extra bold", y=1.02)
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

    # Keep only CLIP‑D entries (other detectors may be present)
    clipd = df[df["detector"] == "CLIP-D"].copy()
    if clipd.empty:
        print("No CLIP‑D results found. Exiting.")
        return

    print("\n" + "=" * 70)
    print("CLIP‑D DETECTION MODE COMPARISON – SUMMARY")
    print("=" * 70)

    # --- Overall metrics per mode (ignoring model/quant breakdown) ---
    overall = build_summary(clipd, group_cols=["mode_label"])
    print("\n1. Overall performance per detection mode:\n")
    print(overall.to_string(index=False))

    # --- Performance by model & mode ---
    by_model_mode = build_summary(clipd, group_cols=["model", "mode_label"])
    print("\n2. Performance by model & detection mode:\n")
    print(by_model_mode.to_string(index=False))

    # --- Performance by model, quantization & mode ---
    by_model_quant_mode = build_summary(
        clipd, group_cols=["model", "quantization", "mode_label"]
    )
    print("\n3. Detailed breakdown (model × quantization × mode):\n")
    print(by_model_quant_mode.to_string(index=False))

    # --- Visual comparisons ---
    print("\nGenerating professional comparison figures...")

    # Fake rate
    plot_mode_comparison_bars(
        by_model_quant_mode,
        metric="fake_rate",
        title="CLIP‑D Detection Mode Comparison – True Positive Rate",
        output_path=DEFAULT_FIGURES_DIR / "clipd_mode_fake_rate_comparison.png",
        ylabel="Fake Detection Rate",
        fmt=".1%",
    )

    # Confidence on fake predictions
    plot_mode_comparison_bars(
        by_model_quant_mode,
        metric="avg_confidence_on_fake",
        title="CLIP‑D Detection Mode Comparison – Avg Confidence on Fake",
        output_path=DEFAULT_FIGURES_DIR / "clipd_mode_confidence_comparison.png",
        ylabel="Average Confidence",
        fmt=".2f",
    )

    # Elapsed time
    if "avg_elapsed_time" in by_model_quant_mode.columns:
        plot_mode_comparison_bars(
            by_model_quant_mode,
            metric="avg_elapsed_time",
            title="CLIP‑D Detection Mode Comparison – Average Detection Time",
            output_path=DEFAULT_FIGURES_DIR / "clipd_mode_elapsed_time_comparison.png",
            ylabel="Time (seconds)",
            fmt=".2f",
        )

    print(f"All figures saved to: {DEFAULT_FIGURES_DIR}\n")
    print("Analysis complete.")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CSV)
    main(csv_arg)
