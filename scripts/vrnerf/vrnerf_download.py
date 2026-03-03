#!/usr/bin/env python3
"""
Download VR-NeRF (EyefulTower) dataset at 1K JPEG resolution.

Uses AWS CLI (aws s3 cp --recursive --no-sign-request) to fetch
images-jpeg-1k for all scenes. Saves under datasets-raw/vrnerf/<scene_name>/images-jpeg-1k/.
"""
# python scripts/vrnerf_download.py

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


# EyefulTower scene names (from https://github.com/facebookresearch/EyefulTower)
EYEFUL_TOWER_SCENES = [
    "apartment",
    "kitchen",
    "office1a",
    "office1b",
    "office2",
    "office_view1",
    "office_view2",
    "riverview",
    "seating_area",
    "table",
    "workshop",
    "raf_emptyroom",
    "raf_furnishedroom",
]

S3_BASE = "s3://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/EyefulTower"
RESOLUTION_SUBDIR = "images-jpeg-1k"


def get_project_root() -> Path:
    """Resolve project root as the parent of the scripts/ directory."""
    return Path(__file__).resolve().parent.parent


def get_output_base_dir(project_root: Path) -> Path:
    """Return datasets-raw/vrnerf under project root."""
    return project_root / "datasets-raw" / "vrnerf"


def find_aws() -> str | None:
    """Return path to aws CLI if found in PATH, else None."""
    return shutil.which("aws")


def download_scene(
    scene_name: str,
    dest_dir: Path,
    aws_path: str,
) -> bool:
    """
    Download images-jpeg-1k for one scene via aws s3 cp.

    Returns True on success, False on failure.
    """
    s3_uri = f"{S3_BASE}/{scene_name}/{RESOLUTION_SUBDIR}/"
    dest_path = dest_dir / RESOLUTION_SUBDIR

    dest_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        aws_path,
        "s3",
        "cp",
        "--recursive",
        "--no-sign-request",
        s3_uri,
        str(dest_path) + "/",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Error: {result.stderr or result.stdout or 'unknown'}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr)
        return False


def main() -> int:
    aws_path = find_aws()
    if not aws_path:
        print(
            "Error: 'aws' CLI not found in PATH.\n"
            "Please install AWS CLI (e.g. pip install awscli) and ensure it is on your PATH.\n"
            "See: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html",
            file=sys.stderr,
        )
        return 1

    project_root = get_project_root()
    output_base = get_output_base_dir(project_root)

    print(f"Project root: {project_root}")
    print(f"Output base:  {output_base}")
    print(f"Scenes:      {len(EYEFUL_TOWER_SCENES)} ({RESOLUTION_SUBDIR} only)")
    print()

    failed = []
    for i, scene_name in enumerate(EYEFUL_TOWER_SCENES, start=1):
        dest_dir = output_base / scene_name
        print(f"[{i}/{len(EYEFUL_TOWER_SCENES)}] Downloading scene: {scene_name} -> {dest_dir}/")
        if not download_scene(scene_name, dest_dir, aws_path):
            failed.append(scene_name)
            print(f"  Failed: {scene_name}")
        else:
            print(f"  Done: {scene_name}")

    if failed:
        print(f"\nFailed scenes ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll scenes downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
