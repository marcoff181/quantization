#!/usr/bin/env python3
"""Phase 1b: Image Generation Batch Runner for img2img split lists."""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import List


PROJECT_ROOT = Path("/media/CrispyMcMarkInc")
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "splits"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "fake-images-img2img"
DEFAULT_SCRIPT = "original_img_to_img/original_img2img.py"

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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_subprocess(cmd: List[str], phase: str, batch_label: str, cwd: Path) -> None:
    logging.info("[%s] %s | Command: %s", batch_label, phase, " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)

def build_generation_command(
    script_path: Path,
    model: str,
    quant: str,
    input_list: Path,
    output_dir: Path,
    prompt: str,
    strength: float,
    steps: int,
    guidance: float,
    seed: int,
    device: str | None,
    max_images: int | None,
) -> List[str]:
    cmd = [
        str(QUANTIZATION_PYTHON),
        str(script_path),
        "--models",
        model,
        "--quantization",
        quant,
        "--input_list",
        str(input_list),
        "--output_dir",
        str(output_dir),
        "--prompt",
        prompt,
        "--strength",
        str(strength),
        "--steps",
        str(steps),
        "--guidance",
        str(guidance),
        "--seed",
        str(seed),
    ]
    if device:
        cmd.extend(["--device", device])
    if max_images is not None:
        cmd.extend(["--max_images", str(max_images)])
    return cmd


def resolve_split_file(split_dir: Path, prefix: str, split_name: str) -> Path:
    return split_dir / f"{prefix}_{split_name}.txt"


def read_split_list(split_list: Path) -> List[str]:
    items: List[str] = []
    with split_list.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def write_split_list(output_path: Path, items: List[str]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(f"{item}\n")


def sample_subset(items: List[str], fraction: float, seed: int, max_images: int | None) -> List[str]:
    if fraction <= 0:
        return []
    if fraction >= 1:
        subset = list(items)
    else:
        subset_count = max(1, int(len(items) * fraction))
        subset = list(items)
        rng = random.Random(seed)
        rng.shuffle(subset)
        subset = subset[:subset_count]
    if max_images is not None:
        subset = subset[:max_images]
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate img2img fakes from train/val/test split lists."
    )
    parser.add_argument("--split_dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--prefix", type=str, default="img2img")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for generated images.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sd15", "sd3", "sd35"],
        help="Models to use.",
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=["fp16"],
        help="Quantization levels to use.",
    )
    parser.add_argument(
        "--extra_quantizations",
        nargs="+",
        default=[],
        help="Extra quantizations to run on a subset of images (e.g., fp8 fp4).",
    )
    parser.add_argument(
        "--extra_fraction",
        type=float,
        default=0.0,
        help="Fraction of images per split to use for extra quantizations.",
    )
    parser.add_argument(
        "--extra_seed",
        type=int,
        default=123,
        help="Seed for sampling subset images for extra quantizations.",
    )
    parser.add_argument(
        "--extra_max_images",
        type=int,
        default=None,
        help="Max images per split for extra quantizations.",
    )
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps"])
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=IMG2IMG_PROMPT)
    parser.add_argument("--script", type=str, default=DEFAULT_SCRIPT)

    args = parser.parse_args()
    setup_logging()

    split_dir = Path(args.split_dir)
    output_root = Path(args.output_root)
    script_path = PROJECT_ROOT / args.script

    if not QUANTIZATION_PYTHON.is_file():
        raise FileNotFoundError(f"Quantization Python not found at {QUANTIZATION_PYTHON}")
    if not script_path.exists():
        raise FileNotFoundError(f"Generation script not found: {script_path}")

    output_root.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        split_list = resolve_split_file(split_dir, args.prefix, split_name)
        if not split_list.exists():
            raise FileNotFoundError(f"Split list not found: {split_list}")

        split_output = output_root / split_name
        split_output.mkdir(parents=True, exist_ok=True)

        for model in args.models:
            for quant in args.quantizations:
                label = f"{split_name}_{model}_{quant}"
                cmd = build_generation_command(
                    script_path=script_path,
                    model=model,
                    quant=quant,
                    input_list=split_list,
                    output_dir=split_output,
                    prompt=args.prompt,
                    strength=args.strength,
                    steps=args.steps,
                    guidance=args.guidance,
                    seed=args.seed,
                    device=args.device,
                    max_images=args.max_images,
                )
                try:
                    run_subprocess(
                        cmd,
                        phase="Generate",
                        batch_label=label,
                        cwd=script_path.parent,
                    )
                except Exception as exc:
                    logging.exception("Batch failed (%s): %s", label, exc)

        if args.extra_quantizations and args.extra_fraction > 0:
            base_items = read_split_list(split_list)
            split_seed = args.extra_seed + sum(ord(ch) for ch in split_name)
            subset_items = sample_subset(
                base_items,
                fraction=args.extra_fraction,
                seed=split_seed,
                max_images=args.extra_max_images,
            )
            if subset_items:
                subset_list = split_output / f"{args.prefix}_{split_name}_extra.txt"
                write_split_list(subset_list, subset_items)
                logging.info(
                    "Extra quantization subset for %s: %d/%d images",
                    split_name,
                    len(subset_items),
                    len(base_items),
                )

                for model in args.models:
                    for quant in args.extra_quantizations:
                        label = f"{split_name}_{model}_{quant}_extra"
                        cmd = build_generation_command(
                            script_path=script_path,
                            model=model,
                            quant=quant,
                            input_list=subset_list,
                            output_dir=split_output,
                            prompt=args.prompt,
                            strength=args.strength,
                            steps=args.steps,
                            guidance=args.guidance,
                            seed=args.seed,
                            device=args.device,
                            max_images=None,
                        )
                        try:
                            run_subprocess(
                                cmd,
                                phase="Generate",
                                batch_label=label,
                                cwd=script_path.parent,
                            )
                        except Exception as exc:
                            logging.exception("Batch failed (%s): %s", label, exc)

    logging.info("Img2img generation complete. Images saved under %s", output_root)


if __name__ == "__main__":
    main()
