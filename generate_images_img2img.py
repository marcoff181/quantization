#!/usr/bin/env python3
"""Phase 1: Image Generation Batch Runner for img2img."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path("/media/CrispyMcMarkInc")
TEMP_IMG_DIR = Path("/media/CrispyMcMarkInc/fake-images-img2img/")
IMG2IMG_INPUT_DIR = Path("/media/CrispyMcMarkInc/Real/")
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
BATCHES: List[Dict[str, object]] = [
    {"script": "original_img_to_img/original_img2img.py", "model": "sd15", "quant": "fp16", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd15", "quant": "fp8", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd15", "quant": "fp4", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd3", "quant": "fp16", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd3", "quant": "fp8", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd3", "quant": "fp4", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd35", "quant": "fp16", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd35", "quant": "fp8", "extra_args": ["--strength", "0.3"]},
    {"script": "original_img_to_img/original_img2img.py", "model": "sd35", "quant": "fp4", "extra_args": ["--strength", "0.3"]},
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_subprocess(cmd: List[str], phase: str, batch_label: str, cwd: Path) -> None:
    logging.info("[%s] %s | Command: %s", batch_label, phase, " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


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
    setup_logging()
    TEMP_IMG_DIR.mkdir(parents=True, exist_ok=True)

    if not QUANTIZATION_PYTHON.is_file():
        raise FileNotFoundError(f"Quantization Python not found at {QUANTIZATION_PYTHON}")
    if not IMG2IMG_INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {IMG2IMG_INPUT_DIR}")

    for idx, batch in enumerate(BATCHES, start=1):
        model = str(batch["model"])
        quant = str(batch["quant"])
        script_rel = str(batch["script"])
        label = f"{model}_{quant}"
        logging.info("=== Generation Batch %d/%d | %s ===", idx, len(BATCHES), label)

        try:
            cmd = build_generation_command(batch)
            run_subprocess(cmd, phase="Generate", batch_label=label, cwd=(PROJECT_ROOT / script_rel).parent)
        except Exception as exc:
            logging.exception("Batch failed (%s): %s", label, exc)

    logging.info("Img2img generation complete. Images saved to %s", TEMP_IMG_DIR)


if __name__ == "__main__":
    main()
