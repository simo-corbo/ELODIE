#!/usr/bin/env python3
"""Download per-model results from Hugging Face into ``outputs/``.

Finds every dataset repo named ``{namespace}/{prefix}-*`` and downloads each
into its own subfolder ``outputs/{folder}``.

Uses the ``HF_TOKEN`` stored in ``.env`` (needed for private repos).

Usage:
    python scripts/download_results.py [--namespace USER] [--prefix NAME] [--dest DIR]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_NAMESPACE = "simocorbo"
DEFAULT_PREFIX = "nerobali-results"


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
        "--dest",
        default=str(OUTPUTS_DIR),
        help="Local destination directory (default: outputs/).",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")  # optional for public repos

    api = HfApi(token=token)
    name_prefix = f"{args.namespace}/{args.prefix}-"
    repos = [
        d.id for d in api.list_datasets(author=args.namespace)
        if d.id.startswith(name_prefix)
    ]
    if not repos:
        raise SystemExit(f"No dataset repos found matching {name_prefix}*")

    dest_root = Path(args.dest)
    for repo_id in sorted(repos):
        folder = repo_id[len(name_prefix):]
        dest = dest_root / folder
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {dest}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            token=token,
            allow_patterns=None,
            ignore_patterns=[".gitattributes"],
        )

    print(f"Done: downloaded {len(repos)} model repo(s) into {dest_root}.")


if __name__ == "__main__":
    main()
