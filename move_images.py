#!/usr/bin/env python3
"""
Copy the first 100 images from the FORLAB dataset into a local 'Real' folder.
Usage: python select_first_100.py
"""

import os
import shutil
from pathlib import Path

SOURCE_DIR = Path("/media/NAS/TrueFake/PreSocial/Real/FORLAB")
TARGET_DIR = Path.cwd() / "Real"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    # Gather all image files, sorted alphabetically
    image_files = sorted(
        [f for f in SOURCE_DIR.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
    )

    if not image_files:
        print(f"No supported images found in {SOURCE_DIR}")
        return

    # Take first 100 (or fewer if less exist)
    selected = image_files[:1000]
    print(f"Selected {len(selected)} images out of {len(image_files)} total.")

    # Create target directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Copy each selected image
    for src in selected:
        dst = TARGET_DIR / src.name
        shutil.copy2(src, dst)   # preserves metadata
        print(f"Copied: {src.name}")

    print(f"\nDone. Images saved in: {TARGET_DIR.absolute()}")


if __name__ == "__main__":
    main()
