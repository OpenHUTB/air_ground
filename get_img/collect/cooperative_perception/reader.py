"""Read committed AirGroundCoopSuite units without touching staging/quarantine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

try:
    from .schema import SCHEMA_VERSION, TASK_IDS
except ImportError:
    from schema import SCHEMA_VERSION, TASK_IDS


class AirGroundCoopReader:
    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)

    def units(self, task_id: Optional[str] = None) -> Iterable[Path]:
        task_directories = (
            [TASK_IDS[task_id]] if task_id is not None else list(TASK_IDS.values())
        )
        for task_directory in task_directories:
            task_root = self.dataset_root / task_directory
            if not task_root.is_dir():
                continue
            for unit in sorted(path for path in task_root.iterdir() if path.is_dir()):
                manifest_path = unit / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("schema_version") == SCHEMA_VERSION
                    and manifest.get("commit_state") == "committed"
                ):
                    yield unit

    @staticmethod
    def jsonl(path: Path) -> Iterator[Dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError("Invalid JSONL %s:%d" % (path, line_number)) from exc

    def frames(self, unit: Path) -> Iterator[Dict[str, Any]]:
        return self.jsonl(Path(unit) / "frames.jsonl")
