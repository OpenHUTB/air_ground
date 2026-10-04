#!/usr/bin/env python3
"""Upload the curated multi-map dataset without Windows wildcard expansion.

The script is intentionally inert unless --execute is provided.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "yutiangu/HUBT_from_a_drones_perspective"
DEFAULT_DATASET_ROOT = Path(
    r"E:\pythonProject\air_groud\get_img"
    r"\dataset_uav_multimap_town600_hutb300_ccsp300"
)

IGNORE_PATTERNS = [
    "_logs/**",
    "_discarded*/**",
    "*/_rejected*/**",
    "**/_rejected*/**",
    ".cache/**",
    "**/.cache/**",
    "**/__pycache__/**",
    "**/.ccsp_rgb_asset_qa_cache.json",
    "collection_status.json",
    "collection_qa_summary.json",
    "sequence_consolidation_report.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the remote Hugging Face dataset with local curated files."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the remote delete-and-upload operation.",
    )
    return parser.parse_args()


def validate_local_dataset(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not (root / "README.md").is_file():
        raise FileNotFoundError(f"Hugging Face README.md is missing: {root}")

    required_maps = {
        "CCSP_Zhongdian_Software_Park",
        "HutbCarlaCity",
        "Town02_Opt",
        "Town03_Opt",
        "Town04_Opt",
        "Town05_Opt",
        "Town07_Opt",
        "Town10HD",
    }
    missing = sorted(name for name in required_maps if not (root / name).is_dir())
    if missing:
        raise FileNotFoundError(f"Required map directories are missing: {missing}")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    validate_local_dataset(dataset_root)

    print(f"Repository: {args.repo_id}")
    print(f"Local dataset: {dataset_root}")
    print("Remote policy: delete old files, then upload curated local files")
    print("Excluded local patterns:")
    for pattern in IGNORE_PATTERNS:
        print(f"  - {pattern}")

    if not args.execute:
        print("[DRY-RUN] No remote operation was performed.")
        print("Run the same command with --execute to start uploading.")
        return 0

    if not os.environ.get("HF_XET_HIGH_PERFORMANCE"):
        print("[WARN] HF_XET_HIGH_PERFORMANCE is not set; upload may be slower.")

    api = HfApi()
    account = api.whoami().get("name", "unknown")
    print(f"Authenticated account: {account}")

    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=dataset_root,
        commit_message="Replace old dataset with multi-map multimodal UAV dataset",
        ignore_patterns=IGNORE_PATTERNS,
        delete_patterns=["*", "**/*"],
    )
    print(f"Upload complete: {result.repo_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
