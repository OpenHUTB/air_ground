"""Committed-unit checkpoint state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


class CheckpointManager:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"committed_units": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, committed_units: Iterable[str], extra: Dict[str, Any] = None) -> None:
        payload = {
            "committed_units": sorted(set(str(item) for item in committed_units)),
        }
        if extra:
            payload.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

