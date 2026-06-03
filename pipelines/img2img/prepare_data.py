#!/usr/bin/env python3
"""
Center-crop real images to 1024x1024 to match img2img preprocessing.
"""
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = _ROOT / "data" / "real" / "raw"
OUTPUT_DIR = _ROOT / "data" / "real"
TARGET_SIZE = 1024

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

logging.info(f"Cropping {len(images)} images to {TARGET_SIZE}x{TARGET_SIZE}...")

for img_path in images:
    try:
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        
        # Center crop to 1024x1024 (if image is already smaller, keep it)
        if width > TARGET_SIZE or height > TARGET_SIZE:
            left = (width - TARGET_SIZE) // 2
            top = (height - TARGET_SIZE) // 2
            right = left + TARGET_SIZE
            bottom = top + TARGET_SIZE
            img = img.crop((left, top, right, bottom))
        
        output_path = OUTPUT_DIR / img_path.name
        img.save(output_path)
        logging.info(f"  ✓ {img_path.name}")
    except Exception as e:
        logging.error(f"  ✗ {img_path.name}: {e}")

logging.info(f"Done. Cropped images saved to: {OUTPUT_DIR}")