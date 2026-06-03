import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Usage: python analyze_detector_performance.py detector_final_results.csv

# Apply a clean, professional aesthetic
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
KNOWN_QUANTIZATIONS = {"fp16", "fp8", "fp4"}

def _looks_like_label_series(series):
    values = series.dropna().astype(str).str.strip().str.lower()
    if values.empty:
        return False
    return values.isin({"fake", "real"}).mean() >= 0.8

def _looks_like_prob_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return False
    return numeric.between(0, 1).mean() >= 0.8

def normalize_detector_csv_schema(df):
    if not {"image_path", "prediction", "confidence"}.issubset(df.columns):
        return df

    if _looks_like_label_series(df["prediction"]):
        return df

    if _looks_like_label_series(df["image_path"]) and _looks_like_prob_series(df["prediction"]):
        result = df.copy()
        result["elapsed_time"] = pd.to_numeric(result["confidence"], errors="coerce")
        result["confidence"] = pd.to_numeric(result["prediction"], errors="coerce")
        result["prediction"] = result["image_path"]
        if "detector_mode" in result.columns:
            result["image_path"] = result["detector_mode"]
            result["detector_mode"] = "unknown"
        else:
            result["image_path"] = None
        return result

    return df

def extract_batch_parts(label):
    text = str(label)
    if '_' not in text:
        return text, 'unknown'
    model, quantization = text.rsplit('_', 1)
    if quantization not in KNOWN_QUANTIZATIONS:
        return text, 'unknown'
    return model, quantization

def build_summary(df):
    summary = []
    for detector in sorted(df['detector'].dropna().unique()):
        sub = df[df['detector'] == detector]
        total = len(sub)
        fake_predictions = (sub['prediction'] == 'fake').sum()
        real_predictions = (sub['prediction'] == 'real').sum()
        fake_rate = fake_predictions / total if total > 0 else 0
        avg_conf_fake = sub.loc[sub['prediction'] == 'fake', 'confidence'].mean()
        avg_conf_all = sub['confidence'].mean()

        summary.append({
            'detector': detector,
            'total': total,
            'fake_predictions': fake_predictions,
            'real_predictions': real_predictions,
            'fake_rate': fake_rate,
            'avg_confidence_on_fake_predictions': avg_conf_fake,
            'avg_confidence_overall': avg_conf_all,
        })
    return pd.DataFrame(summary)

def prepare_results(df):
    result = normalize_detector_csv_schema(df).copy()
    if 'batch_label' in result.columns:
        batch_parts = result['batch_label'].apply(extract_batch_parts)
        result['model'] = batch_parts.apply(lambda value: value[0]) 
        result['quantization'] = batch_parts.apply(lambda value: value[1])
    else:
        result['model'] = 'unknown'
        result['quantization'] = 'unknown'

    if 'confidence' in result.columns:
        result['confidence'] = pd.to_numeric(result['confidence'], errors='coerce')
    return result

def plot_metric_grid_seaborn(df, metric, title, output_path, cmap, formatter):
    models = [model for model in sorted(df['model'].dropna().unique()) if model!= 'unknown']
    if not models:
        return

    n_models = len(models)
    n_cols = 2 if n_models > 1 else 1
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(8 * n_cols, 5 * n_rows), 
        constrained_layout=True
    )
    axes = [axes] if n_models == 1 else list(axes.flat)

    # Establish global min and max for consistent colorbar scaling across subplots
    vmin = df[metric].min()
    vmax = df[metric].max()

    for axis_index, axis in enumerate(axes):
        if axis_index >= len(models):
            axis.axis('off')
            continue

        model = models[axis_index]
        subm = df[df['model'] == model]
        pivot = subm.pivot_table(
            index='detector', columns='quantization', values=metric, aggfunc='mean'
        )
        
        if pivot.empty:
            continue

        # Force logical precision ordering rather than alphabetical
        ordered_cols = [col for col in ['fp16', 'fp8', 'fp4'] if col in pivot.columns]
        pivot = pivot[ordered_cols]

        # Apply custom formatting to a dedicated annotation dataframe
        annot_data = pivot.map(lambda v: formatter(v) if pd.notna(v) else "")

        # Draw the heatmap
        sns.heatmap(
            pivot, 
            ax=axis, 
            cmap=cmap, 
            annot=annot_data, 
            fmt="", 
            vmin=vmin, 
            vmax=vmax,
            linewidths=1, 
            linecolor='white', 
            cbar_kws={'shrink': 0.8},
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )

        # Formatting titles and labels
        axis.set_title(f"Model: {model}", fontweight='bold', fontsize=14, pad=10)
        axis.set_xlabel('Quantization', fontsize=12, fontweight='semibold')
        axis.set_ylabel('Detector', fontsize=12, fontweight='semibold')
        axis.tick_params(axis='x', rotation=0, labelsize=11)
        axis.tick_params(axis='y', rotation=0, labelsize=11)

    fig.suptitle(title, fontsize=18, fontweight='extra bold')
    fig.patch.set_facecolor('white')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def save_figures(df, output_dir='figures'):
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def summarize_group(group):
        fake_predictions = group[group['prediction'] == 'fake']
        return pd.Series({
            'fake_rate': (group['prediction'] == 'fake').mean(),
            'avg_confidence_on_fake_predictions': fake_predictions['confidence'].mean(),
        })

    base = df.groupby(['model', 'quantization', 'detector']).apply(summarize_group).reset_index()

    plot_metric_grid_seaborn(
        base,
        metric='fake_rate',
        title='Detector True Positive Rate by Model and Quantization',
        output_path=figures_dir / 'fake_rate_by_model_quantization_detector.png',
        cmap='mako',
        formatter=lambda value: f"{value:.1%}"
    )

    plot_metric_grid_seaborn(
        base,
        metric='avg_confidence_on_fake_predictions',
        title='Average Confidence on Fake Predictions',
        output_path=figures_dir / 'confidence_by_model_quantization_detector.png',
        cmap='flare',
        formatter=lambda value: f"{value:.2f}"
    )

def main(csv_path):
    df = pd.read_csv(csv_path)
    df = prepare_results(df)

    print("\n=== Detector Performance by Model and Quantization ===")
    for model in sorted(df['model'].unique()):
        if model == 'unknown': continue
        print(f"\n--- Generation Model: {model} ---")
        subm = df[df['model'] == model]
        model_summary = build_summary(subm)
        print(model_summary.to_string(index=False))

    print("\n=== Detector Performance Overall ===")
    overall_summary = build_summary(df)
    print(overall_summary.to_string(index=False))

    save_figures(df)
    print("\nSaved high-resolution figures to ./figures")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_detector_performance.py detector_final_results.csv")
        sys.exit(1)
    main(sys.argv[1])