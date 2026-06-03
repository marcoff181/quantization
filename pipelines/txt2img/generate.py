#!/usr/bin/env python3
"""Phase 1: Image Generation Batch Runner."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT
TEMP_IMG_DIR = _ROOT / "data" / "fake" / "txt2img"

QUANTIZATION_PYTHON = Path(
    os.environ.get(
        "QUANTIZATION_PYTHON",
        str(PROJECT_ROOT / ".venvs" / "quantization" / "bin" / "python"),
    )
)

BASE_TXT2IMG_BATCHES: List[Dict[str, object]] = [
    {"script": "generators/txt2img/generate.py", "model": "sdxl", "quant": "fp16"},
    {"script": "generators/txt2img/generate.py", "model": "sdxl", "quant": "fp8"},
    {"script": "generators/txt2img/generate.py", "model": "sdxl", "quant": "fp4"},
    {"script": "generators/txt2img/generate.py", "model": "sd3", "quant": "fp16"},
    {"script": "generators/txt2img/generate.py", "model": "sd3", "quant": "fp8"},
    {"script": "generators/txt2img/generate.py", "model": "sd3", "quant": "fp4"},
    {"script": "generators/txt2img/generate.py", "model": "sd35", "quant": "fp16"},
    {"script": "generators/txt2img/generate.py", "model": "sd35", "quant": "fp8"},
    {"script": "generators/txt2img/generate.py", "model": "sd35", "quant": "fp4"},
    {"script": "generators/txt2img/generate.py", "model": "sd15", "quant": "fp16"},
    {"script": "generators/txt2img/generate.py", "model": "sd15", "quant": "fp8"},
    {"script": "generators/txt2img/generate.py", "model": "sd15", "quant": "fp4"},
    {"script": "generators/txt2img/generate.py", "model": "z-image", "quant": "fp16"},
    {"script": "generators/txt2img/generate.py", "model": "z-image", "quant": "fp8"},
    {"script": "generators/txt2img/generate.py", "model": "z-image", "quant": "fp4"},
    {"script": "generators/txt2img/generate.py", "model": "pg25", "quant": "fp16"},
    {"script": "generators/txt2img/generate.py", "model": "pg25", "quant": "fp8"},
    {"script": "generators/txt2img/generate.py", "model": "pg25", "quant": "fp4"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all txt2img generation batches across model, quantization, steps, and guidance combinations."
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
        "--prompts",
        type=int,
        default=2,
        help="Number of prompts to pass to txt2img script.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Prompts file to pass to txt2img. Defaults to generators/txt2img/prompts_filtered.txt when available.",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        help="Only run batches with missing outputs in the output directory.",
    )
    return parser.parse_args()

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def _script_supports_flag(script_path: Path, flag: str) -> bool:
    try:
        content = script_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return flag in content

def _resolve_prompts_file(prompts_file: Optional[str]) -> Optional[Path]:
    if prompts_file:
        path = Path(prompts_file)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / "generators" / "txt2img" / path).resolve()
    default_path = PROJECT_ROOT / "generators" / "txt2img" / "prompts_filtered.txt"
    return default_path if default_path.exists() else None

def _count_prompts(prompts_file: Path) -> Optional[int]:
    try:
        with prompts_file.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None

def run_subprocess(cmd: List[str], phase: str, batch_label: str, cwd: Path) -> None:
    logging.info("[%s] %s | Command: %s", batch_label, phase, " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _missing_prompt_indices(
    output_dir: Path,
    model: str,
    quant: str,
    prompts: int,
    step: int,
    guidance: float,
) -> List[int]:
    missing: List[int] = []
    step_str = str(step)
    guidance_str = str(guidance)
    for idx in range(prompts):
        pattern = f"{model}_{quant}_p{idx}_seed*_s{step_str}_g{guidance_str}.*"
        if not any(output_dir.glob(pattern)):
            missing.append(idx)
    return missing


def build_batches(
    steps: List[int],
    guidance: List[float],
    prompts: int,
    prompts_file: Optional[Path] = None,
) -> List[Dict[str, object]]:
    batches: List[Dict[str, object]] = []
    for base_batch, step, guide in product(BASE_TXT2IMG_BATCHES, steps, guidance):
        extra_args: List[str] = [
            "--prompts",
            str(prompts),
            "--steps",
            str(step),
            "--guidance",
            str(guide),
        ]
        if prompts_file is not None:
            extra_args.extend(["--prompts_file", str(prompts_file)])
        batch = {
            "script": base_batch["script"],
            "model": base_batch["model"],
            "quant": base_batch["quant"],
            "extra_args": extra_args,
            "label": f"{base_batch['model']}_{base_batch['quant']}_s{step}_g{guide}",
        }
        batches.append(batch)
    return batches

def build_generation_command(batch: Dict[str, object]) -> List[str]:
    script_rel = str(batch["script"])
    model = str(batch["model"])
    quant = str(batch["quant"])

    model_flag = "--model" if _script_supports_flag(PROJECT_ROOT / script_rel, "--model") else "--models"
    quant_flag = "--quant" if _script_supports_flag(PROJECT_ROOT / script_rel, "--quant") else "--quantization"

    cmd = [
        str(QUANTIZATION_PYTHON),
        str(PROJECT_ROOT / script_rel),
        model_flag, model,
        quant_flag, quant,
        "--output_dir", str(TEMP_IMG_DIR),
    ]
    extra_args = batch.get("extra_args", [])
    cmd.extend(str(x) for x in extra_args)
    return cmd

def main() -> None:
    args = parse_args()
    setup_logging()
    TEMP_IMG_DIR.mkdir(parents=True, exist_ok=True)
    prompts_file = _resolve_prompts_file(args.prompts_file)
    effective_prompts = args.prompts
    if prompts_file is not None:
        available_prompts = _count_prompts(prompts_file)
        if available_prompts is None:
            logging.warning(
                "Prompts file %s could not be read; using requested prompts (%d).",
                prompts_file,
                args.prompts,
            )
        elif available_prompts == 0:
            logging.warning(
                "Prompts file %s is empty; using requested prompts (%d).",
                prompts_file,
                args.prompts,
            )
        else:
            if args.prompts > available_prompts:
                logging.warning(
                    "Requested %d prompts but %s has %d; capping to %d.",
                    args.prompts,
                    prompts_file,
                    available_prompts,
                    available_prompts,
                )
            effective_prompts = min(args.prompts, available_prompts)

    batches = build_batches(args.steps, args.guidance, effective_prompts, prompts_file)
    
    
    if not QUANTIZATION_PYTHON.exists() or not QUANTIZATION_PYTHON.is_file():
        raise FileNotFoundError(f"Quantization Python not found at {QUANTIZATION_PYTHON}")

    for idx, batch in enumerate(batches, start=1):
        script_rel = str(batch["script"])
        missing = []
        if args.only_missing:
            missing = _missing_prompt_indices(
                TEMP_IMG_DIR,
                str(batch["model"]),
                str(batch["quant"]),
                effective_prompts,
                int(batch["extra_args"][3]),
                float(batch["extra_args"][5]),
            )
            if not missing:
                logging.info(
                    "=== Generation Batch %d/%d | %s | skip (complete) ===",
                    idx,
                    len(batches),
                    batch["label"],
                )
                continue
            preview = ", ".join(str(i) for i in missing[:20])
            suffix = "..." if len(missing) > 20 else ""
            logging.info(
                "=== Generation Batch %d/%d | %s | missing %d prompts: %s%s ===",
                idx,
                len(batches),
                batch["label"],
                len(missing),
                preview,
                suffix,
            )
        else:
            logging.info("=== Generation Batch %d/%d | %s ===", idx, len(batches), batch["label"])

        try:
            cmd = build_generation_command(batch)
            run_subprocess(cmd, phase="Generate", batch_label=batch["label"], cwd=(PROJECT_ROOT / script_rel).parent)
        except Exception as exc:
            logging.exception("Batch failed (%s): %s", batch["label"], exc)

    logging.info("Generation complete. Images saved directly to %s", TEMP_IMG_DIR)

if __name__ == "__main__":
    main()