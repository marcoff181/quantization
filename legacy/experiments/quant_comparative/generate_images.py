#!/usr/bin/env python3
"""Phase 1: Image Generation Batch Runner for img2img."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path("/media/CrispyMcMarkInc")
TEMP_IMG_DIR = Path("/media/SSD_4TB/crispy_storage/comparative_images_img2img/")
IMG2IMG_INPUT_DIR = Path("/media/SSD_4TB/crispy_storage/CRISPY_DATASET/PreSocial/Real/")
IMG2IMG_PROMPT = os.environ.get(
    "IMG2IMG_PROMPT",
    "Ultra-realistic photograph, preserving the original subject and composition entirely. Natural daylight, true-to-life textures, physically accurate shadows and reflections, balanced exposure, neutral color grading, high dynamic range, fine detail, sharp focus, DSLR quality, 50mm lens, documentary photo style, no stylization.",
)

QUANTIZATION_PYTHON = Path(
    os.environ.get(
        "QUANTIZATION_PYTHON",
        str(PROJECT_ROOT / ".venvs" / "quantization" / "bin" / "python"),
    )
)


# Reduced model set for img2img.
BASE_IMG2IMG_BATCHES: List[Dict[str, object]] = [
    {"script": "original_img_to_img/original_img2img.py", "model": "sd15", "quant": "fp16"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd15", "quant": "fp8"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd15", "quant": "fp4"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd3", "quant": "fp16"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd3", "quant": "fp8"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd3", "quant": "fp4"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd35", "quant": "fp16"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd35", "quant": "fp8"},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd35", "quant": "fp4"},
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all img2img generation batches across model, quantization, steps, guidance and strength combinations."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[30],
        help="List of inference steps (e.g. --steps 20 30 50).",
    )
    parser.add_argument(
        "--guidance",
        nargs="+",
        type=float,
        default=[3.5],
        help="List of guidance values (e.g. --guidance 2.5 3.5 5.0).",
    )
    parser.add_argument(
        "--strength",
        nargs="+",
        type=float,
        default=[0.3],
        help="List of image strength values (e.g. --strength 0.1 0.3 0.5).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=2,
        help="Number of images to generate per prompt.",
    ) 
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Skip batches that already have all expected output images in the destination folder.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_subprocess(cmd: List[str], phase: str, batch_label: str, cwd: Path) -> None:
    logging.info("[%s] %s | Command: %s", batch_label, phase, " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def list_input_images(max_images: int) -> List[Path]:
    supported_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [
        path
        for path in sorted(IMG2IMG_INPUT_DIR.iterdir())
        if path.is_file() and path.suffix in supported_exts
    ]
    return images[:max_images] if max_images > 0 else images


def expected_output_paths(batch: Dict[str, object], input_images: List[Path]) -> List[Path]:
    model = str(batch["model"])
    quant = str(batch["quant"])
    extra_args = [str(x) for x in batch.get("extra_args", [])]

    steps = None
    guidance = None
    strength = None
    max_images = None

    for index in range(0, len(extra_args), 2):
        key = extra_args[index]
        value = extra_args[index + 1] if index + 1 < len(extra_args) else ""
        if key == "--steps":
            steps = value
        elif key == "--guidance":
            guidance = value
        elif key == "--strength":
            strength = value
        elif key == "--max_images":
            max_images = int(value)

    if steps is None or guidance is None or strength is None:
        raise ValueError(f"Missing generation parameters for batch {batch.get('label')}")

    selected_images = input_images[:max_images] if max_images is not None else input_images
    seed_start = 123
    expected_paths: List[Path] = []

    for offset, image_path in enumerate(selected_images):
        seed = seed_start + offset
        base_name = image_path.stem
        expected_name = f"{base_name}_{model}_{quant}_seed{seed}_s{steps}_g{guidance}_str{strength}.png"
        expected_paths.append(TEMP_IMG_DIR / expected_name)

    return expected_paths


def batch_is_complete(batch: Dict[str, object], input_images: List[Path]) -> bool:
    expected_paths = expected_output_paths(batch, input_images)
    return expected_paths and all(path.is_file() for path in expected_paths)


def build_batches(steps: List[int], guidance: List[float], max_images: int, strength: List[float]) -> List[Dict[str, object]]:
    batches: List[Dict[str, object]] = []
    for base_batch, step, guide, streng in product(BASE_IMG2IMG_BATCHES, steps, guidance, strength):
        batch = {
            "script": base_batch["script"],
            "model": base_batch["model"],
            "quant": base_batch["quant"],
            "extra_args": [
                "--max_images",
                str(max_images),
                "--steps",
                str(step),
                "--guidance",
                str(guide),
                "--strength",
                str(streng),
            ],
            "label": f"{base_batch['model']}_{base_batch['quant']}_s{step}_g{guide}_str{streng}",
        }
        batches.append(batch)
    return batches

def build_generation_command(batch: Dict[str, object]) -> List[str]:
    script_rel = str(batch["script"])
    model = str(batch["model"])
    quant = str(batch["quant"])
    extra_args = [str(x) for x in batch.get("extra_args", [])]

    script_path = PROJECT_ROOT / script_rel
    if not script_path.exists():
        raise FileNotFoundError(f"Generation script not found: {script_path}")

    cmd = [
        str(QUANTIZATION_PYTHON),
        str(script_path),
        "--models",
        model,
        "--quantization",
        quant,
        "--input_dir",
        str(IMG2IMG_INPUT_DIR),
        "--output_dir",
        str(TEMP_IMG_DIR),
        "--prompt",
        IMG2IMG_PROMPT,
    ]
    cmd.extend(extra_args)
    return cmd


def main() -> None:
    args = parse_args()
    setup_logging()
    TEMP_IMG_DIR.mkdir(parents=True, exist_ok=True)
    batches = build_batches(args.steps, args.guidance, args.max_images, args.strength)
    input_images = list_input_images(args.max_images)

    if not QUANTIZATION_PYTHON.is_file():
        raise FileNotFoundError(f"Quantization Python not found at {QUANTIZATION_PYTHON}")
    if not IMG2IMG_INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {IMG2IMG_INPUT_DIR}")
    if not input_images:
        raise RuntimeError(f"No input images found in {IMG2IMG_INPUT_DIR}")

    for idx, batch in enumerate(batches, start=1):

        script_rel = str(batch["script"])
        logging.info("=== Generation Batch %d/%d | %s ===", idx, len(batches), batch["label"])

        if args.complete and batch_is_complete(batch, input_images):
            logging.info("[%s] Skipping batch: all expected images already exist.", batch["label"])
            continue

        try:
            cmd = build_generation_command(batch)
            run_subprocess(cmd, phase="Generate", batch_label=batch["label"], cwd=(PROJECT_ROOT / script_rel).parent)
        except Exception as exc:
            logging.exception("Batch failed (%s): %s", batch["label"], exc)

    logging.info("Img2img generation complete. Images saved to %s", TEMP_IMG_DIR)


if __name__ == "__main__":
    main()
