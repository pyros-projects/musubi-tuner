#!/usr/bin/env python
import shutil
import sys
from pathlib import Path


def sync_sidecars(source_dir: str, target_dir: str, delete: bool = True):
    source = Path(source_dir)
    target = Path(target_dir)

    if not source.is_dir() or not target.is_dir():
        print(f"Error: both directories must exist.\n  source: {source}\n  target: {target}")
        sys.exit(1)

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    moved = 0
    deleted_images = 0

    for img in target.iterdir():
        if img.suffix.lower() not in image_exts:
            continue

        sidecar = source / (img.stem + ".txt")
        if sidecar.exists():
            shutil.copy2(sidecar, target / sidecar.name)
            moved += 1
            if delete:
                sidecar.unlink()

        if delete:
            source_img = source / img.name
            if source_img.exists():
                source_img.unlink()
                deleted_images += 1

    print(f"Done. Moved {moved} sidecars to {target}.")
    if delete:
        print(f"Deleted {moved} .txt + {deleted_images} images from {source}.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <target_dir> [--no-delete]")
        sys.exit(1)

    delete = "--no-delete" not in sys.argv
    sync_sidecars(sys.argv[1], sys.argv[2], delete=delete)
