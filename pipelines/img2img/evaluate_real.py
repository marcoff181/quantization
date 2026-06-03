#!/usr/bin/env python3
"""
Run all CLIP-D detection modes on real images in parallel, aggregate into one CSV.
"""
from pathlib import Path
from multiprocessing import Pool
import csv
import os

# Derive project root for relative paths
_ROOT = Path(__file__).resolve().parents[2]

MODES = [
    ("default", None),
    ("grid", "max"),
    ("grid", "mean"),
    ("grid", "majority"),
    ("patch", "max"),
    ("patch", "mean"),
    ("patch", "majority"),
]

def run_single_mode(mode_info):
    """Run a single detection mode in a separate process."""
    mode_type, agg = mode_info
    
    # Import here to ensure fresh state in subprocess
    import sys
    from pathlib import Path
    sys.path.insert(0, str(_ROOT / "pipelines" / "img2img"))
    import evaluate as runner
    
    mode_label = f"{mode_type}" + (f"_{agg}" if agg else "")
    temp_csv = _ROOT / "results" / "img2img" / f"patch_detector_real_{mode_label}_temp.csv"
    
    # Configure for real images with temp CSV
    runner.TEMP_IMG_DIR = _ROOT / "data" / "real"
    runner.RESULTS_CSV = temp_csv
    runner.CLIPD_DETECTION_MODE = mode_type
    if agg is not None:
        runner.CLIPD_AGGREGATION = agg
    
    print(f"[{mode_label}] Starting detection...")
    try:
        runner.main()
        print(f"[{mode_label}] ✓ Completed")
        return str(temp_csv)
    except Exception as e:
        print(f"[{mode_label}] ✗ Error: {e}")
        raise

def merge_csvs(csv_files, output_csv):
    """Merge multiple CSV files into one."""
    CSV_FIELDS = [
        "batch_label",
        "detector",
        "detector_mode",
        "image_path",
        "prediction",
        "confidence",
        "elapsed_time",
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        
        for csv_file in sorted(csv_files):
            if Path(csv_file).exists():
                with open(csv_file, "r", encoding="utf-8") as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        writer.writerow(row)
                # Clean up temp file
                os.remove(csv_file)

if __name__ == "__main__":
    output_csv = _ROOT / "results" / "img2img" / "detector_real_all_modes.csv"
    
    print(f"\n{'='*70}")
    print(f"Running {len(MODES)} detection modes in PARALLEL")
    print(f"Final results CSV: {output_csv}")
    print('='*70 + "\n")
    
    # Run all modes in parallel
    with Pool(processes=len(MODES)) as pool:
        temp_csv_files = pool.map(run_single_mode, MODES)
    
    print(f"\nMerging {len(temp_csv_files)} result files...")
    merge_csvs(temp_csv_files, output_csv)
    
    print(f"\n{'='*70}")
    print(f"✓ All modes completed!")
    print(f"Results: {output_csv}")
    print('='*70 + "\n")