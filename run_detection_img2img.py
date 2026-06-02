#!/usr/bin/env python3
"""
Phase 2: Deepfake Detection Runner for img2img outputs.

For each image, runs one or more deepfake detectors and logs results to a CSV.
The CLIP-D detector supports three modes:

    - 'grid'     : uses --grid-mode, requires an aggregation method (max/mean/majority)
    - 'patch'    : uses --use-patching, requires an aggregation method
    - 'default'  : plain detection (no extra flags)

Set the CLIPD_DETECTION_MODE and CLIPD_AGGREGATION variables below to control the behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/media/CrispyMcMarkInc")
TEMP_IMG_DIR = Path("/media/CrispyMcMarkInc/fake-images-img2img/")
DETECTOR_ROOT = PROJECT_ROOT / "Image-Deepfake-Detectors-Public-Library"
RESULTS_CSV = PROJECT_ROOT / "patch_detector_img2img_results.csv"

DETECTOR_PYTHON = Path(
    os.environ.get(
        "DETECTOR_PYTHON",
        str(PROJECT_ROOT / ".venvs" / "detector" / "bin" / "python"),
    )
)

DETECTORS_TO_RUN = ["CLIP-D"]          # add other detectors when needed
DETECTOR_WEIGHTS = "pretrained"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# CLIP‑D detection mode settings
#   CLIPD_DETECTION_MODE  : "grid", "patch", or "default"
#   CLIPD_AGGREGATION     : "max", "mean", "majority" (only relevant for grid/patch)
# ---------------------------------------------------------------------------
CLIPD_DETECTION_MODE = "patch"
CLIPD_AGGREGATION = "majority"

# ---------------------------------------------------------------------------
# CSV column layout – detector_mode will hold a string like "grid_max" or "default"
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "batch_label",
    "detector",
    "detector_mode",
    "image_path",
    "prediction",
    "confidence",
    "elapsed_time",
]

KNOWN_MODELS = ["sd35", "sd15", "sd3"]
KNOWN_QUANTS = ["fp16", "fp8", "fp4"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def initialize_results_csv(results_csv: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP-D detection on img2img outputs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(TEMP_IMG_DIR),
        help="Directory containing images (supports train/val/test subfolders with --recursive).",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=str(RESULTS_CSV),
        help="CSV file to write results to.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input_dir recursively (skips _detect_json folders).",
    )
    parser.add_argument(
        "--clipd_mode",
        type=str,
        default=CLIPD_DETECTION_MODE,
        choices=["grid", "patch", "default"],
        help="CLIP-D detection mode.",
    )
    parser.add_argument(
        "--clipd_arch",
        type=str,
        default="opencliplinearnext_clipL14commonpool",
        help="CLIP-D architecture to load.",
    )
    parser.add_argument(
        "--clipd_aggregation",
        type=str,
        default=CLIPD_AGGREGATION,
        choices=["max", "mean", "majority"],
        help="Aggregation for grid/patch modes.",
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        default=DETECTOR_WEIGHTS,
        help="Detector weights name under CLIP-D/checkpoint.",
    )
    return parser.parse_args()


def collect_images(images_dir: Path, recursive: bool) -> List[Path]:
    images: List[Path] = []
    if recursive:
        for path in images_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            if "_detect_json" in path.parts:
                continue
            images.append(path)
    else:
        images = [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    return images


def _available_detectors(detectors_to_run: List[str], weights_name: str) -> List[str]:
    available = []
    for detector_name in detectors_to_run:
        weight_path = (
            DETECTOR_ROOT
            / "detectors"
            / detector_name
            / "checkpoint"
            / weights_name
            / "weights"
            / "best.pt"
        )
        if weight_path.is_file():
            available.append(detector_name)
        else:
            logging.warning(
                "Skipping detector %s: weights not found at %s",
                detector_name,
                weight_path,
            )
    return available


def infer_label_from_filename(filename: str) -> str:
    name_lower = filename.lower()
    found_model = next((m for m in KNOWN_MODELS if m in name_lower), "unknown")
    found_quant = next((q for q in KNOWN_QUANTS if q in name_lower), "unknown")
    if found_model == "unknown" and found_quant == "unknown":
        return name_lower.split(".")[0]  # fallback to using the filename stem
    return f"{found_model}_{found_quant}"


def build_clipd_command(
    python_bin: Path,
    image: Path,
    output_json: Path,
    mode: str,
    aggregation: str,
    arch: str,
    weights_name: str,
) -> List[str]:
    """Construct the command line for CLIP‑D detection."""
    cmd = [
        str(python_bin),
        str(DETECTOR_ROOT / "detectors" / "CLIP-D" / "detect.py"),
        "--image",
        str(image),
        "--model",
        weights_name,
        "--arch",
        arch,
        "--output",
        str(output_json),
    ]

    if mode == "grid":
        cmd.extend(["--grid-mode", "--aggregation", aggregation])
    elif mode == "patch":
        cmd.extend(["--use-patching", "--aggregation", aggregation])
    # mode "default" -> no extra flags

    return cmd


def main() -> None:
    setup_logging()
    args = parse_args()

    if not DETECTOR_PYTHON.is_file():
        raise FileNotFoundError(f"Detector Python not found at {DETECTOR_PYTHON}")

    images_dir = Path(args.input_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Img2img output directory not found: {images_dir}")

    detector_weights = args.weights_name
    active_detectors = _available_detectors(DETECTORS_TO_RUN, detector_weights)
    if not active_detectors:
        logging.error("No valid detectors found. Exiting.")
        return

    images = collect_images(images_dir, args.recursive)
    if not images:
        logging.warning("No supported images found in directory: %s", images_dir)
        return

    results_csv = Path(args.results_csv)
    initialize_results_csv(results_csv)
    logging.info("Initialized results CSV: %s", results_csv)
    logging.info(
        "Found %d images in %s. Starting detection...", len(images), images_dir
    )

    # Validate CLIP‑D settings
    valid_modes = {"grid", "patch", "default"}
    clipd_mode = args.clipd_mode
    clipd_aggregation = args.clipd_aggregation
    clipd_arch = args.clipd_arch
    if clipd_mode not in valid_modes:
        raise ValueError(
            f"Invalid CLIPD_DETECTION_MODE: {clipd_mode}. Must be one of {valid_modes}"
        )
    if clipd_mode in ("grid", "patch"):
        valid_agg = {"max", "mean", "majority"}
        if clipd_aggregation not in valid_agg:
            raise ValueError(
                f"Invalid CLIPD_AGGREGATION: {clipd_aggregation}. Must be one of {valid_agg}"
            )

    # Build the descriptive mode string for CSV
    if clipd_mode == "default":
        clipd_mode_label = "default"
    else:
        clipd_mode_label = f"{clipd_mode}_{clipd_aggregation}"

    json_out_dir = images_dir / "_detect_json"
    json_out_dir.mkdir(parents=True, exist_ok=True)

    with results_csv.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)

        for image_path in images:
            batch_label = infer_label_from_filename(image_path.name)
            safe_stem = image_path.stem.replace(" ", "_")

            for detector_name in active_detectors:
                output_json = (
                    json_out_dir / f"{batch_label}_{detector_name}_{safe_stem}.json"
                )

                if detector_name == "CLIP-D":
                    detect_cmd = build_clipd_command(
                        DETECTOR_PYTHON, image_path, output_json,
                        clipd_mode, clipd_aggregation, clipd_arch, detector_weights,
                    )
                    current_mode = clipd_mode_label
                else:
                    # Fallback for other detectors (launcher.py)
                    detect_cmd = [
                        str(DETECTOR_PYTHON),
                        str(DETECTOR_ROOT / "launcher.py"),
                        "--detector",
                        detector_name,
                        "--detect",
                        "--image",
                        str(image_path),
                        "--weights",
                        detector_weights,
                        "--output",
                        str(output_json),
                    ]
                    current_mode = "default"

                try:
                    logging.info(
                        "[%s] Detect[%s] (mode=%s) | Command: %s",
                        batch_label,
                        detector_name,
                        current_mode,
                        " ".join(detect_cmd),
                    )
                    # print(f"cmd: {' '.join(detect_cmd)}")
                    # exit(0)
                    subprocess.run(
                        detect_cmd, cwd=str(DETECTOR_ROOT), check=True, timeout=300
                    )

                    if output_json.is_file():
                        payload = json.loads(output_json.read_text(encoding="utf-8"))
                        writer.writerow(
                            {
                                "batch_label": batch_label,
                                "detector": detector_name,
                                "detector_mode": current_mode,
                                "image_path": str(image_path),
                                "prediction": payload.get("prediction"),
                                "confidence": payload.get("confidence"),
                                "elapsed_time": payload.get("elapsed_time"),
                            }
                        )
                        csv_file.flush()

                except subprocess.TimeoutExpired:
                    logging.error(
                        "[%s] Detect[%s] timed out after 300s on %s",
                        batch_label,
                        detector_name,
                        image_path.name,
                    )
                except Exception as exc:
                    logging.exception(
                        "[%s] Error running detection on %s: %s",
                        batch_label,
                        image_path.name,
                        exc,
                    )

    logging.info("Detection completed. Results written to %s", results_csv)


if __name__ == "__main__":
    main()
