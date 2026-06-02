#!/usr/bin/env python3
"""Generate train/val/test CSV manifests for img2img fine-tuning."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List


PROJECT_ROOT = Path("/media/CrispyMcMarkInc")
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "manifests"
DEFAULT_FAKE_ROOT = PROJECT_ROOT / "fake-images-img2img"
DEFAULT_PREFIX = "img2img"
DEFAULT_SPLITS = ["train", "val", "test"]
DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_split_list(split_path: Path) -> List[Path]:
    items: List[Path] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = (split_path.parent / path).resolve()
            items.append(path)
    return items


def find_fakes(fake_dir: Path, base_name: str, extensions: List[str]) -> List[Path]:
    matches: List[Path] = []
    for ext in extensions:
        matches.extend(sorted(fake_dir.glob(f"{base_name}_*{ext}")))
    return matches


def write_manifest(csv_path: Path, rows: List[tuple[str, int]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create CSV manifests for real/fake img2img datasets."
    )
    parser.add_argument("--split_dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fake_root", type=str, default=str(DEFAULT_FAKE_ROOT))
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--real_label", type=int, default=0)
    parser.add_argument("--fake_label", type=int, default=1)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="Extensions to include for fake files.",
    )
    parser.add_argument(
        "--require_fakes",
        action="store_true",
        help="Fail if a real image has no corresponding fakes.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not write CSVs, only print counts.",
    )

    args = parser.parse_args()
    setup_logging()

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    fake_root = Path(args.fake_root)
    extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions]

    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    if not fake_root.exists():
        raise FileNotFoundError(f"Fake root not found: {fake_root}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        split_path = split_dir / f"{args.prefix}_{split_name}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Split list not found: {split_path}")

        fake_dir = fake_root / split_name
        if not fake_dir.exists():
            raise FileNotFoundError(f"Fake split dir not found: {fake_dir}")

        real_paths = read_split_list(split_path)
        if not real_paths:
            raise RuntimeError(f"No entries in split list: {split_path}")

        rows: List[tuple[str, int]] = []
        missing_fakes = 0
        missing_reals = 0
        fake_count = 0

        for real_path in real_paths:
            if not real_path.exists():
                missing_reals += 1
                logging.warning("Missing real image: %s", real_path)
                continue

            rows.append((str(real_path), args.real_label))
            base_name = real_path.stem
            fakes = find_fakes(fake_dir, base_name, extensions)
            if not fakes:
                missing_fakes += 1
                if args.require_fakes:
                    raise RuntimeError(f"No fakes found for {base_name} in {fake_dir}")
                continue

            for fake_path in fakes:
                rows.append((str(fake_path), args.fake_label))
                fake_count += 1

        logging.info(
            "Split %s: reals=%d fakes=%d missing_reals=%d missing_fakes=%d",
            split_name,
            len(real_paths) - missing_reals,
            fake_count,
            missing_reals,
            missing_fakes,
        )

        if args.dry_run:
            continue

        output_path = output_dir / f"{args.prefix}_{split_name}.csv"
        write_manifest(output_path, rows)
        logging.info("Wrote manifest: %s", output_path)


if __name__ == "__main__":
    main()
