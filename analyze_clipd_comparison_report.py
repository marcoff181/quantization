#!/usr/bin/env python3
"""Professional comparison of CLIP-D methods using fake + real CSVs.

Usage example:
  python analyze_clipd_comparison_report.py \
    --method ft_default:/media/CrispyMcMarkInc/detector_img2img_results.csv:/media/CrispyMcMarkInc/detector_real_results.csv:default \
    --method patch_majority:/media/CrispyMcMarkInc/patch_detector_img2img_results.csv:/media/CrispyMcMarkInc/patch_detector_real_all_modes.csv:patch_majority \
    --detector CLIP-D \
    --output_dir /media/CrispyMcMarkInc/figures/clipd_comparison

Method spec format:
  name:fake_csv:real_csv[:detector_mode]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )
except Exception as exc:  # pragma: no cover - handled at runtime
    raise ImportError("scikit-learn is required for ROC/PR curves.") from exc


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fake_csv: Path
    real_csv: Path
    detector_mode: Optional[str]


@dataclass
class MethodData:
    name: str
    data: pd.DataFrame
    metrics: dict
    roc: Optional[tuple[np.ndarray, np.ndarray, float]]
    pr: Optional[tuple[np.ndarray, np.ndarray, float]]


def parse_method_spec(text: str) -> MethodSpec:
    parts = text.split(":")
    if len(parts) < 3:
        raise ValueError(
            "Method spec must be name:fake_csv:real_csv[:detector_mode]"
        )
    name = parts[0].strip()
    fake_csv = Path(parts[1]).expanduser()
    real_csv = Path(parts[2]).expanduser()
    detector_mode = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
    return MethodSpec(name=name, fake_csv=fake_csv, real_csv=real_csv, detector_mode=detector_mode)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _filter_rows(df: pd.DataFrame, detector: Optional[str], detector_mode: Optional[str]) -> pd.DataFrame:
    result = df.copy()
    if detector and "detector" in result.columns:
        result = result[result["detector"] == detector]
    if detector_mode:
        if "detector_mode" not in result.columns:
            raise ValueError(
                "detector_mode filter requested but column is missing in CSV"
            )
        result = result[result["detector_mode"] == detector_mode]
    return result


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "confidence" not in result.columns:
        raise ValueError("CSV must contain a confidence column")
    result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce")
    if "elapsed_time" in result.columns:
        result["elapsed_time"] = pd.to_numeric(result["elapsed_time"], errors="coerce")
    return result.dropna(subset=["confidence"]).reset_index(drop=True)


def _label_df(df: pd.DataFrame, label_value: int, source_name: str) -> pd.DataFrame:
    result = df.copy()
    result["label"] = label_value
    result["source"] = source_name
    return result


def load_method_data(spec: MethodSpec, detector: Optional[str]) -> pd.DataFrame:
    fake_df = _normalize(_filter_rows(_load_csv(spec.fake_csv), detector, spec.detector_mode))
    real_df = _normalize(_filter_rows(_load_csv(spec.real_csv), detector, spec.detector_mode))

    fake_df = _label_df(fake_df, label_value=1, source_name="fake")
    real_df = _label_df(real_df, label_value=0, source_name="real")

    combined = pd.concat([fake_df, real_df], ignore_index=True)
    if combined.empty:
        raise ValueError(f"No rows found after filtering for method: {spec.name}")
    combined["method"] = spec.name
    return combined


def compute_metrics(df: pd.DataFrame, threshold: float) -> tuple[dict, tuple, tuple]:
    y_true = df["label"].astype(int).to_numpy()
    y_score = df["confidence"].to_numpy()

    if len(np.unique(y_true)) < 2:
        raise ValueError("Both real and fake samples are required for metrics.")

    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_curve, tpr_curve)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    metrics = {
        "n_total": len(df),
        "n_fake": int((y_true == 1).sum()),
        "n_real": int((y_true == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mean_conf_fake": float(df.loc[df["label"] == 1, "confidence"].mean()),
        "mean_conf_real": float(df.loc[df["label"] == 0, "confidence"].mean()),
    }

    if "elapsed_time" in df.columns:
        metrics["mean_elapsed_time"] = float(df["elapsed_time"].mean())
        metrics["p95_elapsed_time"] = float(df["elapsed_time"].quantile(0.95))
    else:
        metrics["mean_elapsed_time"] = np.nan
        metrics["p95_elapsed_time"] = np.nan

    roc = (fpr_curve, tpr_curve, roc_auc)
    pr = (recall_curve, precision_curve, pr_auc)
    return metrics, roc, pr


def plot_roc_pr(methods: list[MethodData], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    palette = sns.color_palette("tab10", n_colors=len(methods))

    for color, method in zip(palette, methods):
        if method.roc is None or method.pr is None:
            continue
        fpr_curve, tpr_curve, roc_auc = method.roc
        recall_curve, precision_curve, pr_auc = method.pr

        axes[0].plot(fpr_curve, tpr_curve, label=f"{method.name} (AUC={roc_auc:.3f})", color=color, linewidth=2)
        axes[1].plot(recall_curve, precision_curve, label=f"{method.name} (AP={pr_auc:.3f})", color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_threshold_metrics(summary: pd.DataFrame, output_path: Path, threshold: float) -> None:
    metric_order = ["tpr", "fpr", "precision", "f1", "accuracy"]
    available = [m for m in metric_order if m in summary.columns]
    if not available:
        return

    plot_df = summary[["method", *available]].set_index("method")
    plot_df = plot_df.loc[sorted(plot_df.index)]
    annot_df = plot_df.apply(
        lambda col: col.map(
            lambda value: f"{value:.2f}" if pd.notna(value) else ""
        )
    )

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    sns.heatmap(
        plot_df,
        annot=annot_df,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title(
        f"Threshold Metrics (confidence >= {threshold:.2f})",
        fontweight="bold",
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Method")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_score_distributions(all_data: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    real_df = all_data[all_data["label"] == 0]
    fake_df = all_data[all_data["label"] == 1]
    method_order = sorted(all_data["method"].unique())

    sns.violinplot(
        data=real_df,
        x="method",
        y="confidence",
        order=method_order,
        ax=axes[0],
        cut=0,
        inner="quartile",
        linewidth=1,
    )
    axes[0].set_title("Real Images: P(fake)", fontweight="bold")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)

    sns.violinplot(
        data=fake_df,
        x="method",
        y="confidence",
        order=method_order,
        ax=axes[1],
        cut=0,
        inner="quartile",
        linewidth=1,
    )
    axes[1].set_title("Fake Images: P(fake)", fontweight="bold")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Confidence")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_latency(summary: pd.DataFrame, output_path: Path) -> None:
    if summary["mean_elapsed_time"].isna().all():
        return

    plot_df = summary[["method", "mean_elapsed_time", "p95_elapsed_time"]].copy()
    plot_df = plot_df.dropna(subset=["mean_elapsed_time"])
    plot_df = plot_df.sort_values("mean_elapsed_time", ascending=True)
    if plot_df.empty:
        return

    fig_height = max(3.5, 0.45 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height), constrained_layout=True)
    palette = sns.color_palette("Blues", n_colors=len(plot_df))
    ax.barh(plot_df["method"], plot_df["mean_elapsed_time"], color=palette)

    if not plot_df["p95_elapsed_time"].isna().all():
        ax.scatter(
            plot_df["p95_elapsed_time"],
            plot_df["method"],
            color="black",
            marker="|",
            s=120,
            label="p95",
            zorder=3,
        )

    max_time = plot_df["mean_elapsed_time"].max()
    offset = max_time * 0.01 if max_time > 0 else 0.01
    for _, row in plot_df.iterrows():
        ax.text(
            row["mean_elapsed_time"] + offset,
            row["method"],
            f"{row['mean_elapsed_time']:.2f}s",
            va="center",
            fontsize=8,
        )

    ax.set_title("Detection Latency (mean with p95 marker)", fontweight="bold")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Method")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)
    if not plot_df["p95_elapsed_time"].isna().all():
        ax.legend(loc="lower right", fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_report(method_specs: list[MethodSpec], detector: Optional[str], threshold: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    method_outputs: list[MethodData] = []
    combined_rows = []

    for spec in method_specs:
        method_df = load_method_data(spec, detector)
        metrics, roc, pr = compute_metrics(method_df, threshold)
        method_outputs.append(MethodData(name=spec.name, data=method_df, metrics=metrics, roc=roc, pr=pr))
        combined_rows.append(method_df)

    summary = pd.DataFrame(
        [{"method": m.name, **m.metrics} for m in method_outputs]
    )
    summary_path = output_dir / "clipd_comparison_metrics.csv"
    summary.to_csv(summary_path, index=False)

    all_data = pd.concat(combined_rows, ignore_index=True)
    combined_path = output_dir / "clipd_comparison_combined.csv"
    all_data.to_csv(combined_path, index=False)

    plot_roc_pr(method_outputs, output_dir / "clipd_comparison_roc_pr.png")
    plot_threshold_metrics(
        summary,
        output_dir / "clipd_comparison_threshold_metrics.png",
        threshold,
    )
    plot_score_distributions(all_data, output_dir / "clipd_comparison_score_distributions.png")
    plot_latency(summary, output_dir / "clipd_comparison_latency.png")

    print("\nSaved summary metrics to:", summary_path)
    print("Saved combined dataset to:", combined_path)
    print("Saved figures to:", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Professional CLIP-D comparison report (ROC, PR, metrics, score distributions)."
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec: name:fake_csv:real_csv[:detector_mode]",
    )
    parser.add_argument(
        "--detector",
        default="CLIP-D",
        help="Detector name to filter on (default: CLIP-D)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for metric bars (default: 0.5)",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/CrispyMcMarkInc/figures/clipd_comparison",
        help="Output directory for figures and CSV summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_specs = [parse_method_spec(text) for text in args.method]
    output_dir = Path(args.output_dir)
    build_report(method_specs, args.detector, args.threshold, output_dir)


if __name__ == "__main__":
    main()
