#!/usr/bin/env python3
"""Upload committed AirGroundCoopSuite units; dry-run unless --execute is used."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .reader import AirGroundCoopReader
except ImportError:
    from reader import AirGroundCoopReader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--task", choices=("task1", "task2", "task3"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    units = list(AirGroundCoopReader(args.root).units(args.task))
    inventory = {
        "repo_id": args.repo_id,
        "execute": bool(args.execute),
        "units": [str(path) for path in units],
    }
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before --execute") from exc
    api = HfApi()
    for unit in units:
        relative = unit.relative_to(args.root).as_posix()
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(unit),
            path_in_repo=relative,
            commit_message="Upload committed AirGroundCoopSuite unit %s" % relative,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
