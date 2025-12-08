#!/usr/bin/env python3
"""Convert MP4 files inside sibling subfolders into animated GIFs."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MP4 files inside subfolders next to this script into GIFs. "
            "Pass a subfolder name or use --all."
        )
    )
    parser.add_argument(
        "subfolder",
        nargs="?",
        help="Name of a single subfolder (relative to this script) to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every subfolder that sits next to this script.",
    )

    args = parser.parse_args()
    if args.all and args.subfolder:
        parser.error("Provide either a subfolder name or --all, not both.")
    if not args.all and not args.subfolder:
        parser.error("You must provide a subfolder name or --all.")
    return args


def iter_target_dirs(root: Path, args: argparse.Namespace) -> list[Path]:
    if args.all:
        targets = sorted(p for p in root.iterdir() if p.is_dir())
        if not targets:
            logging.warning("No subfolders found in %s", root)
        return targets

    target = (root / args.subfolder).resolve()
    if not target.exists() or not target.is_dir():
        raise SystemExit(f"Subfolder '{args.subfolder}' does not exist under {root}")
    return [target]


def convert_mp4_to_gif(mp4_path: Path, ffmpeg_bin: str) -> bool:
    gif_path = mp4_path.with_suffix(".gif")
    logging.info("Converting %s -> %s", mp4_path.name, gif_path.name)

    if gif_path.exists():
        logging.info("Overwriting existing GIF %s", gif_path)
        gif_path.unlink()

    filter_chain = (
        "[0:v]split[v0][v1];"
        "[v0]palettegen=stats_mode=diff[p];"
        "[v1][p]paletteuse=new=1"
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(mp4_path),
        "-filter_complex",
        filter_chain,
        "-gifflags",
        "+transdiff",
        "-vsync",
        "0",
        str(gif_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.error("FFmpeg failed for %s:\n%s", mp4_path, exc.stderr.strip())
        if gif_path.exists():
            gif_path.unlink(missing_ok=True)
        return False

    try:
        mp4_path.unlink()
    except OSError as exc:
        logging.error("GIF created but failed to delete %s: %s", mp4_path, exc)
        return False

    return True


def process_directory(directory: Path, ffmpeg_bin: str) -> int:
    mp4_files = sorted(directory.rglob("*.mp4"))
    if not mp4_files:
        logging.info("No MP4 files found in %s", directory)
        return 0

    success_count = 0
    for mp4_file in mp4_files:
        if convert_mp4_to_gif(mp4_file, ffmpeg_bin):
            success_count += 1
    return success_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise SystemExit("ffmpeg is required but was not found in PATH.")

    total_converted = 0
    targets = iter_target_dirs(root_dir, args)
    for target_dir in targets:
        logging.info("Processing folder %s", target_dir)
        total_converted += process_directory(target_dir, ffmpeg_bin)

    logging.info("Finished. Converted %s file(s).", total_converted)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Aborted by user.")

