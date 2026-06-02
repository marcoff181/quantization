#!/usr/bin/env python3
"""Check which img2img outputs are missing for a given generation grid."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


DEFAULT_INPUT_DIR = Path("/media/SSD_4TB/crispy_storage/CRISPY_DATASET/PreSocial/Real/")
DEFAULT_OUTPUT_DIR = Path("/media/SSD_4TB/crispy_storage/comparative_images_img2img/")
DEFAULT_MODELS = ["sd15", "sd3", "sd35"]
DEFAULT_QUANTS = ["fp16", "fp8", "fp4"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

FILENAME_RE = re.compile(
    r"^(?P<source>.+)_(?P<model>sd15|sd3|sd35)_(?P<quant>fp16|fp8|fp4)_seed(?P<seed>\d+)_s(?P<steps>\d+)_g(?P<guidance>[0-9]+(?:\.[0-9]+)?)_str(?P<strength>[0-9]+(?:\.[0-9]+)?)\.(?P<ext>png|jpg|jpeg|PNG|JPG|JPEG)$"
)


@dataclass(frozen=True)
class GenerationKey:
    source: str
    model: str
    quant: str
    seed: int
    steps: int
    guidance: str
    strength: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report missing img2img images by comparing expected combinations against saved outputs."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with source input images.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory with generated img2img outputs.")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        required=True,
        help="Expected inference steps used for generation.",
    )
    parser.add_argument(
        "--guidance",
        nargs="+",
        type=float,
        required=True,
        help="Expected guidance values used for generation.",
    )
    parser.add_argument(
        "--strength",
        nargs="+",
        type=float,
        required=True,
        help="Expected strength values used for generation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Expected generation models.",
    )
    parser.add_argument(
        "--quants",
        nargs="+",
        default=DEFAULT_QUANTS,
        help="Expected quantization levels.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="If set, only the first N input images sorted alphabetically are considered.",
    )
    parser.add_argument(
        "--report_csv",
        type=Path,
        default=None,
        help="Optional CSV output path for missing combinations.",
    )
    return parser.parse_args()


def normalize_float(value: float) -> str:
    text = f"{value:g}"
    return text


def list_input_images(input_dir: Path, max_images: int | None) -> List[Path]:
    images = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTS
    )
    if max_images is not None:
        images = images[:max_images]
    return images


def build_expected_keys(
    input_images: Sequence[Path],
    models: Sequence[str],
    quants: Sequence[str],
    steps: Sequence[int],
    guidance: Sequence[float],
    strength: Sequence[float],
) -> Set[GenerationKey]:
    expected: Set[GenerationKey] = set()
    for source_path in input_images:
        source = source_path.stem
        for model in models:
            for quant in quants:
                for step in steps:
                    for guide in guidance:
                        for streng in strength:
                            expected.add(
                                GenerationKey(
                                    source=source,
                                    model=model,
                                    quant=quant,
                                    seed=-1,
                                    steps=step,
                                    guidance=normalize_float(guide),
                                    strength=normalize_float(streng),
                                )
                            )
    return expected


def parse_actual_file(path: Path) -> GenerationKey | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    data = match.groupdict()
    return GenerationKey(
        source=data["source"],
        model=data["model"],
        quant=data["quant"],
        seed=int(data["seed"]),
        steps=int(data["steps"]),
        guidance=normalize_float(float(data["guidance"])),
        strength=normalize_float(float(data["strength"])),
    )


def build_actual_index(output_dir: Path) -> Dict[Tuple[str, str, str, int, int, str, str], Path]:
    index: Dict[Tuple[str, str, str, int, int, str, str], Path] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.suffix not in IMAGE_EXTS:
            continue
        parsed = parse_actual_file(path)
        if parsed is None:
            continue
        key = (parsed.source, parsed.model, parsed.quant, parsed.seed, parsed.steps, parsed.guidance, parsed.strength)
        index[key] = path
    return index


def expected_actual_key_for_lookup(source: str, model: str, quant: str, steps: int, guidance: float, strength: float) -> Tuple[str, str, str, str, int, str, str]:
    return (
        source,
        model,
        quant,
        -1,
        steps,
        normalize_float(guidance),
        normalize_float(strength),
    )


def write_missing_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "model",
                "quant",
                "steps",
                "guidance",
                "strength",
                "expected_count",
                "found_count",
                "missing_count",
                "missing_examples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    input_images = list_input_images(args.input_dir, args.max_images)
    if not input_images:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    actual_index = build_actual_index(args.output_dir)
    combo_counter = Counter((key[1], key[2], key[4], key[5], key[6]) for key in actual_index.keys())

    expected_total = len(input_images) * len(args.models) * len(args.quants) * len(args.steps) * len(args.guidance) * len(args.strength)
    found_total = len(actual_index)
    print(f"Input images considered: {len(input_images)}")
    print(f"Expected outputs: {expected_total}")
    print(f"Found outputs: {found_total}")
    print(f"Missing outputs: {expected_total - found_total}")

    missing_rows: List[Dict[str, object]] = []
    missing_by_source = defaultdict(int)
    missing_by_combo = defaultdict(int)

    for source_path in input_images:
        source = source_path.stem
        for model in args.models:
            for quant in args.quants:
                for step in args.steps:
                    for guide in args.guidance:
                        for streng in args.strength:
                            expected_key = expected_actual_key_for_lookup(source, model, quant, step, guide, streng)
                            found_for_combo = [
                                key for key in actual_index.keys()
                                if key[0] == source and key[1] == model and key[2] == quant and key[4] == step and key[5] == normalize_float(guide) and key[6] == normalize_float(streng)
                            ]
                            if found_for_combo:
                                continue

                            missing_by_source[source] += 1
                            missing_by_combo[(model, quant, step, normalize_float(guide), normalize_float(streng))] += 1
                            missing_rows.append(
                                {
                                    "source": source,
                                    "model": model,
                                    "quant": quant,
                                    "steps": step,
                                    "guidance": normalize_float(guide),
                                    "strength": normalize_float(streng),
                                    "expected_count": 1,
                                    "found_count": 0,
                                    "missing_count": 1,
                                    "missing_examples": f"{source}_{model}_{quant}_seed*_s{step}_g{normalize_float(guide)}_str{normalize_float(streng)}.png",
                                }
                            )

    print("\nObserved counts by combo (ignoring source/seed):")
    for (model, quant, step, guide, streng), count in sorted(
        combo_counter.items(), key=lambda item: (-item[1], item[0])
    )[:40]:
        print(f"- {model} | {quant} | steps={step} | guidance={guide} | strength={streng} -> {count}")

    if missing_rows:
        print("\nMissing combinations (first 50 shown):")
        for row in missing_rows[:50]:
            print(
                f"- {row['source']} | {row['model']} | {row['quant']} | steps={row['steps']} | guidance={row['guidance']} | strength={row['strength']}"
            )
    else:
        print("\nNo missing combinations found.")

    if missing_by_combo:
        print("\nMissing by parameter combo:")
        for (model, quant, step, guide, streng), count in sorted(
            missing_by_combo.items(), key=lambda item: (-item[1], item[0])
        )[:50]:
            print(f"- {model} | {quant} | steps={step} | guidance={guide} | strength={streng}: {count} missing")

    print("\nMissing by source:")
    for source, count in sorted(missing_by_source.items(), key=lambda item: (-item[1], item[0])):
        print(f"- {source}: {count}")

    if args.report_csv is not None:
        write_missing_csv(missing_rows, args.report_csv)
        print(f"\nCSV report saved to: {args.report_csv}")


if __name__ == "__main__":
    main()