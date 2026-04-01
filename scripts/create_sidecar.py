#!/usr/bin/env python3
"""Create a .txt sidecar file for every video/image in a directory."""

import argparse
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


def main():
    parser = argparse.ArgumentParser(description="Create .txt sidecar files for media files.")
    parser.add_argument("folder", type=Path, help="Directory containing media files")
    parser.add_argument("text", type=str, help="Text content for each .txt file")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing .txt files")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be created")
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Not a directory: {args.folder}")

    pattern = "**/*" if args.recursive else "*"
    files = sorted(f for f in args.folder.glob(pattern) if f.suffix.lower() in MEDIA_EXTS)

    created, skipped = 0, 0
    for media_file in files:
        txt_path = media_file.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            print(f"[dry-run] {txt_path}")
        else:
            txt_path.write_text(args.text, encoding="utf-8")
        created += 1

    label = "Would create" if args.dry_run else "Created"
    print(f"{label} {created} txt files, skipped {skipped} existing.")


if __name__ == "__main__":
    main()
