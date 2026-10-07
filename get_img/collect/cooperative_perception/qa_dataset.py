#!/usr/bin/env python3
"""Offline structural QA for committed AirGroundCoopSuite data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

try:
    from .reader import AirGroundCoopReader
except ImportError:
    from reader import AirGroundCoopReader


TIME_FIELDS = {
    "world_frame_id",
    "timestamp",
    "simulation_time",
    "sample_index",
    "source_tick",
}

PLATFORM_STATE_FIELDS = TIME_FIELDS | {
    "platform_pose",
    "sensor_pose",
    "velocity_mps",
    "acceleration_mps2",
    "heading_deg",
    "control",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_unit(unit: Path) -> Dict[str, Any]:
    errors: List[str] = []
    manifest = json.loads((unit / "manifest.json").read_text(encoding="utf-8"))
    checksums = dict(manifest.get("checksums_sha256", {}))
    for relative_path, expected in checksums.items():
        path = unit / relative_path
        if not path.is_file():
            errors.append("missing checksum file: %s" % relative_path)
        elif digest(path) != expected:
            errors.append("checksum mismatch: %s" % relative_path)
    reader = AirGroundCoopReader(unit.parent.parent)
    frames = list(reader.frames(unit))
    expected_indices = list(range(len(frames)))
    actual_indices = [int(row.get("sample_index", -1)) for row in frames]
    if actual_indices != expected_indices:
        errors.append("sample_index is not consecutive from zero")
    for row in frames:
        missing = sorted(TIME_FIELDS - set(row))
        if missing:
            errors.append("frame missing fields: %s" % ",".join(missing))
        if int(row.get("world_frame_id", -1)) != int(row.get("source_tick", -2)):
            errors.append("source_tick differs from world_frame_id")
        if not bool(row.get("sync", {}).get("sync_ok", False)):
            errors.append("frame barrier failed in committed data")
        sync = dict(row.get("sync", {}))
        expected = set(sync.get("expected_sensor_ids", []))
        received = set(sync.get("received_sensor_ids", []))
        sensor_timestamps = dict(row.get("sensor_timestamps", {}))
        if not expected.issubset(received) or set(sensor_timestamps) != expected:
            errors.append(
                "frame %s required/received/timestamp sensor sets differ"
                % row.get("world_frame_id")
            )
        tolerance = float(sync.get("timestamp_tolerance_s", 1.0e-4))
        simulation_time = float(row.get("simulation_time", float("nan")))
        values = [float(value) for value in sensor_timestamps.values()]
        if (
            not math.isfinite(simulation_time)
            or any(not math.isfinite(value) for value in values)
            or any(abs(value - simulation_time) > tolerance for value in values)
            or (values and max(values) - min(values) > tolerance)
        ):
            errors.append(
                "frame %s sensor timestamps violate tolerance"
                % row.get("world_frame_id")
            )
    platforms_path = unit / "platforms" / "platforms.json"
    if platforms_path.is_file():
        platform_data = json.loads(platforms_path.read_text(encoding="utf-8"))
        platforms = list(platform_data.get("platforms", []))
        sensors = list(platform_data.get("sensors", []))
        required_sensor_ids = {
            str(item["sensor_id"])
            for item in sensors
            if bool(item.get("required", False))
        }
        for row in frames:
            if set(row.get("sensor_timestamps", {})) != required_sensor_ids:
                errors.append(
                    "frame %s does not contain every registered required sensor"
                    % row.get("world_frame_id")
                )
        for platform in platforms:
            platform_id = str(platform.get("platform_id", ""))
            category = (
                "ground"
                if platform.get("platform_type") == "ground_vehicle"
                else "air"
            )
            states_path = unit / "platforms" / category / platform_id / "states.jsonl"
            if not states_path.is_file():
                errors.append("missing per-frame platform states: %s" % platform_id)
                continue
            states = list(AirGroundCoopReader.jsonl(states_path))
            if len(states) != len(frames):
                errors.append("platform state count mismatch: %s" % platform_id)
                continue
            for frame, state in zip(frames, states):
                missing = sorted(PLATFORM_STATE_FIELDS - set(state))
                if missing:
                    errors.append(
                        "platform %s state missing fields: %s"
                        % (platform_id, ",".join(missing))
                    )
                for field in ("world_frame_id", "sample_index", "source_tick"):
                    if int(state.get(field, -1)) != int(frame.get(field, -2)):
                        errors.append(
                            "platform %s state/frame %s mismatch"
                            % (platform_id, field)
                        )
                for field in ("timestamp", "simulation_time"):
                    state_value = float(state.get(field, float("nan")))
                    frame_value = float(frame.get(field, float("nan")))
                    if (
                        not math.isfinite(state_value)
                        or not math.isfinite(frame_value)
                        or abs(state_value - frame_value) > 1.0e-4
                    ):
                        errors.append(
                            "platform %s state/frame %s mismatch"
                            % (platform_id, field)
                        )
        if manifest.get("task_id") == "task3":
            air = [item for item in platforms if item.get("platform_type") == "air_camera_platform"]
            ground = [item for item in platforms if item.get("platform_type") == "ground_vehicle"]
            if len(air) != 3 or any(not bool(item.get("platform_is_virtual")) for item in air):
                errors.append("Task 3 must contain exactly three virtual Air platforms")
            if len(ground) != 1:
                errors.append("Task 3 must contain exactly one Ground Vehicle")
    return {
        "unit": str(unit),
        "frames": len(frames),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--task", choices=("task1", "task2", "task3"))
    args = parser.parse_args()
    reader = AirGroundCoopReader(args.root)
    reports = [audit_unit(unit) for unit in reader.units(args.task)]
    result = {
        "root": str(args.root.resolve()),
        "unit_count": len(reports),
        "passed": bool(reports) and all(item["passed"] for item in reports),
        "units": reports,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
