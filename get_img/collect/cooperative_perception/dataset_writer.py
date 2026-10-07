"""Transactional scene/sequence/sample writer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schema import COMMIT_STATE_COMMITTED, SCHEMA_VERSION


class DatasetWriter:
    def __init__(self, dataset_root: Path, task_directory: str):
        self.dataset_root = Path(dataset_root)
        self.task_root = self.dataset_root / str(task_directory)
        self.staging_root = self.dataset_root / "staging" / str(task_directory)
        self.quarantine_root = self.dataset_root / "quarantine" / str(task_directory)
        self.current_unit_id: Optional[str] = None
        self.current_path: Optional[Path] = None

    def begin(self, unit_id: str, overwrite: bool = False) -> Path:
        if self.current_path is not None:
            raise RuntimeError("A dataset transaction is already active")
        self.current_unit_id = str(unit_id)
        self.current_path = self.staging_root / self.current_unit_id
        if self.current_path.exists():
            if not overwrite:
                raise FileExistsError(str(self.current_path))
            shutil.rmtree(str(self.current_path))
        self.current_path.mkdir(parents=True, exist_ok=False)
        return self.current_path

    def _path(self, relative_path: str) -> Path:
        if self.current_path is None:
            raise RuntimeError("No active dataset transaction")
        destination = self.current_path / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self._path(relative_path)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def append_jsonl(self, relative_path: str, payload: Any) -> Path:
        path = self._path(relative_path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def copy_file(self, source: Path, relative_path: str) -> Path:
        destination = self._path(relative_path)
        try:
            os.link(str(source), str(destination))
        except OSError:
            shutil.copy2(str(source), str(destination))
        return destination

    def _checksums(self) -> Dict[str, str]:
        if self.current_path is None:
            return {}
        checksums = {}
        for path in sorted(self.current_path.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[path.relative_to(self.current_path).as_posix()] = digest.hexdigest()
        return checksums

    def commit(self, metadata: Dict[str, Any]) -> Path:
        if self.current_path is None or self.current_unit_id is None:
            raise RuntimeError("No active dataset transaction")
        manifest = dict(metadata)
        manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "commit_state": COMMIT_STATE_COMMITTED,
                "checksums_sha256": self._checksums(),
            }
        )
        self.write_json("manifest.json", manifest)
        destination = self.task_root / self.current_unit_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(str(destination))
        os.replace(str(self.current_path), str(destination))
        self.current_path = None
        self.current_unit_id = None
        return destination

    def abort(self, reason: str) -> Optional[Path]:
        if self.current_path is None or self.current_unit_id is None:
            return None
        self.write_json("abort.json", {"reason": str(reason), "commit_state": "aborted"})
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = self.quarantine_root / self.current_unit_id
        suffix = 1
        while destination.exists():
            destination = self.quarantine_root / (self.current_unit_id + "_%02d" % suffix)
            suffix += 1
        os.replace(str(self.current_path), str(destination))
        self.current_path = None
        self.current_unit_id = None
        return destination

