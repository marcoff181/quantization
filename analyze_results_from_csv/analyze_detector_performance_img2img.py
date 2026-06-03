#!/usr/bin/env python3
"""Analyze detector performance for img2img outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_CSV = Path("/media/CrispyMcMarkInc/detector_img2img_results.csv")
DEFAULT_FIGURES_DIR = Path("/media/CrispyMcMarkInc/figures/img2img")
KNOWN_QUANTIZATIONS = {"fp16", "fp8", "fp4"}


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def extract_batch_parts(label: object) -> tuple[str, str]:
    text = str(label)
    if "_" not in text:
        return text, "unknown"

    model, quantization = text.rsplit("_", 1)
    if quantization not in KNOWN_QUANTIZATIONS:
        return text, "unknown"
    return model, quantization


def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "batch_label" in result.columns:
        batch_parts = result["batch_label"].apply(extract_batch_parts)
        result["model"] = batch_parts.apply(lambda value: value[0])
        result["quantization"] = batch_parts.apply(lambda value: value[1])
    else:
        result["model"] = "unknown"
        result["quantization"] = "unknown"

    for column in ("confidence", "elapsed_time"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for detector in sorted(df["detector"].dropna().unique()):
        sub = df[df["detector"] == detector]
        total = len(sub)
        fake_predictions = (sub["prediction"] == "fake").sum()
        real_predictions = (sub["prediction"] == "real").sum()
        fake_rate = fake_predictions / total if total > 0 else 0
        avg_conf_fake = sub.loc[sub["prediction"] == "fake", "confidence"].mean()
        avg_conf_all = sub["confidence"].mean()
        avg_elapsed = sub["elapsed_time"].mean() if "elapsed_time" in sub.columns else None

        summary_rows.append(
            {
                "detector": detector,
                "total": total,
                "fake_predictions": fake_predictions,
                "real_predictions": real_predictions,
                "fake_rate": fake_rate,
                "avg_confidence_on_fake_predictions": avg_conf_fake,
                "avg_confidence_overall": avg_conf_all,
                "avg_elapsed_time": avg_elapsed,
            }
        )

    return pd.DataFrame(summary_rows)


def plot_metric_grid_seaborn(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    cmap: str,
    formatter,
) -> None:
    models = [model for model in sorted(df["model"].dropna().unique()) if model != "unknown"]
    if not models or metric not in df.columns:
        return

    n_models = len(models)
    n_cols = 2 if n_models > 1 else 1
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), constrained_layout=True)
    axes = [axes] if n_models == 1 else list(axes.flat)

    vmin = df[metric].min()
    vmax = df[metric].max()

    for axis_index, axis in enumerate(axes):
        if axis_index >= len(models):
            axis.axis("off")
            continue

        model = models[axis_index]
        subm = df[df["model"] == model]
        pivot = subm.pivot_table(index="detector", columns="quantization", values=metric, aggfunc="mean")

        if pivot.empty:
            axis.axis("off")
            continue

        ordered_cols = [col for col in ["fp16", "fp8", "fp4"] if col in pivot.columns]
        pivot = pivot[ordered_cols]
        annot_data = pivot.map(lambda value: formatter(value) if pd.notna(value) else "")

        sns.heatmap(
            pivot,
            ax=axis,
            cmap=cmap,
            annot=annot_data,
            fmt="",
            vmin=vmin,
            vmax=vmax,
            linewidths=1,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )

        axis.set_title(f"Model: {model}", fontweight="bold", fontsize=14, pad=10)
        axis.set_xlabel("Quantization", fontsize=12, fontweight="semibold")
        axis.set_ylabel("Detector", fontsize=12, fontweight="semibold")
        axis.tick_params(axis="x", rotation=0, labelsize=11)
        axis.tick_params(axis="y", rotation=0, labelsize=11)

    fig.suptitle(title, fontsize=18, fontweight="extra bold")
    fig.patch.set_facecolor("white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_figures(df: pd.DataFrame, output_dir: Path = DEFAULT_FIGURES_DIR) -> None:
    def summarize_group(group: pd.DataFrame) -> pd.Series:
        fake_predictions = group[group["prediction"] == "fake"]
        return pd.Series(
            {
                "fake_rate": (group["prediction"] == "fake").mean(),
                "avg_confidence_on_fake_predictions": fake_predictions["confidence"].mean(),
                "avg_elapsed_time": group["elapsed_time"].mean() if "elapsed_time" in group.columns else None,
            }
        )

    base = df.groupby(["model", "quantization", "detector"]).apply(summarize_group).reset_index()

    plot_metric_grid_seaborn(
        base,
        metric="fake_rate",
        title="Img2img Detector True Positive Rate by Model and Quantization",
        output_path=output_dir / "img2img_fake_rate_by_model_quantization_detector.png",
        cmap="mako",
        formatter=lambda value: f"{value:.1%}",
    )

    plot_metric_grid_seaborn(
        base,
        metric="avg_confidence_on_fake_predictions",
        title="Img2img Average Confidence on Fake Predictions",
        output_path=output_dir / "img2img_confidence_by_model_quantization_detector.png",
        cmap="flare",
        formatter=lambda value: f"{value:.2f}",
    )

    if "avg_elapsed_time" in base.columns and base["avg_elapsed_time"].notna().any():
        plot_metric_grid_seaborn(
            base,
            metric="avg_elapsed_time",
            title="Img2img Average Detection Time",
            output_path=output_dir / "img2img_elapsed_time_by_model_quantization_detector.png",
            cmap="crest",
            formatter=lambda value: f"{value:.2f}s",
        )


def main(csv_path: str) -> None:
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df = prepare_results(df)

    print("\n=== Img2img Detector Performance by Model and Quantization ===")
    for model in sorted(df["model"].unique()):
        if model == "unknown":
            continue
        print(f"\n--- Generation Model: {model} ---")
        subm = df[df["model"] == model]
        model_summary = build_summary(subm)
        print(model_summary.to_string(index=False))

    print("\n=== Img2img Detector Performance Overall ===")
    overall_summary = build_summary(df)
    print(overall_summary.to_string(index=False))

    save_figures(df)
    print(f"\nSaved high-resolution figures to {DEFAULT_FIGURES_DIR}")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CSV)
    main(csv_arg)