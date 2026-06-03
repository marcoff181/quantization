#!/usr/bin/env python3
"""Create train/val/test split lists for img2img fine-tuning."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List


_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Real_cropped_1024"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "splits"
DEFAULT_PREFIX = "img2img"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def collect_images(input_dir: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in VALID_EXTENSIONS:
            images.append(path)
    return images


def write_list(output_path: Path, items: List[Path]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(f"{item}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test split files from Real_cropped_1024."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing cropped real images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write split files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Prefix for split files (e.g., img2img_train.txt).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of real images before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )

    args = parser.parse_args()
    setup_logging()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = collect_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than 0")
        if args.limit > len(images):
            raise ValueError("--limit exceeds available images")
        images = images[: args.limit]

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    total = len(images)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)
    test_count = total - train_count - val_count

    if train_count == 0 or val_count == 0 or test_count == 0:
        raise ValueError("Split ratios produced an empty split")

    train_items = images[:train_count]
    val_items = images[train_count:train_count + val_count]
    test_items = images[train_count + val_count:]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"{args.prefix}_train.txt"
    val_path = output_dir / f"{args.prefix}_val.txt"
    test_path = output_dir / f"{args.prefix}_test.txt"

    write_list(train_path, train_items)
    write_list(val_path, val_items)
    write_list(test_path, test_items)

    logging.info("Wrote %d train images to %s", len(train_items), train_path)
    logging.info("Wrote %d val images to %s", len(val_items), val_path)
    logging.info("Wrote %d test images to %s", len(test_items), test_path)


if __name__ == "__main__":
    main()
