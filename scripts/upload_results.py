#!/usr/bin/env python3
"""Upload the local ``outputs/`` results to Hugging Face.

Each subfolder of ``outputs/`` is a different model and is uploaded to its own
dataset repo, named ``{namespace}/{prefix}-{folder}``. The ``wildjailbreak``
folder is excluded.

Uses the ``HF_TOKEN`` stored in ``.env``.

Usage:
    python scripts/upload_results.py [--namespace USER] [--prefix NAME] [--private]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_NAMESPACE = "simocorbo"
DEFAULT_PREFIX = "nerobali-results"
EXCLUDE_FOLDERS = {"wildjailbreak", ".gate_timings", "oracle_annotation"}

IGNORE_PATTERNS = ["**/__pycache__/*", "**/.DS_Store"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default=os.getenv("HF_NAMESPACE", DEFAULT_NAMESPACE),
        help=f"HF user/org that owns the repos (default: {DEFAULT_NAMESPACE}).",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("HF_RESULTS_PREFIX", DEFAULT_PREFIX),
        help=f"Repo name prefix (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repos as private if they do not exist yet.",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_UPLOAD_TOKEN")
    if not token:
        raise SystemExit("HF_UPLOAD_TOKEN not found in environment/.env")

    if not OUTPUTS_DIR.is_dir():
        raise SystemExit(f"outputs directory not found: {OUTPUTS_DIR}")

    api = HfApi(token=token)

    folders = sorted(
        p for p in OUTPUTS_DIR.iterdir()
        if p.is_dir() and p.name not in EXCLUDE_FOLDERS
    )
    if not folders:
        raise SystemExit("No model folders to upload.")

    for folder in folders:
        repo_id = f"{args.namespace}/{args.prefix}-{folder.name}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        print(f"Uploading {folder.name} -> {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(folder),
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Upload results for {folder.name}",
        )
        print(f"  https://huggingface.co/datasets/{repo_id}")

    print(f"Done: uploaded {len(folders)} model repo(s).")


if __name__ == "__main__":
    main()
