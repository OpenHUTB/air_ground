#!/usr/bin/env python3
"""Offline structural QA for committed AirGroundCoopSuite data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(AirGroundCoopReader.jsonl(path)) if path.is_file() else []


def sensor_files(unit: Path, platform: Dict[str, Any], modality: str) -> List[Path]:
    category = "ground" if platform.get("platform_type") == "ground_vehicle" else "air"
    root = unit / "platforms" / category / str(platform["platform_id"])
    directory = {"depth": "depth_m"}.get(str(modality), str(modality))
    extension = ".npy" if modality in {"depth", "normal", "lidar"} else ".png"
    return sorted((root / directory).glob("*" + extension))


def audit_unit(unit: Path) -> Dict[str, Any]:
    errors: List[str] = []
    manifest = json.loads((unit / "manifest.json").read_text(encoding="utf-8"))
    deferred_perception = bool(
        manifest.get("run_mode") == "synchronized_raw_acquisition"
        and manifest.get("offline_perception_status") == "pending"
    )
    runtime_fields = ["config_sha256", "code_version", "run_mode"]
    if not deferred_perception:
        runtime_fields.extend(["model_weights_sha256", "cross_platform_fusion_verified"])
    for field in runtime_fields:
        if not manifest.get(field):
            errors.append("manifest missing cooperative runtime field: %s" % field)
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
        platform_by_id = {str(item["platform_id"]): item for item in platforms}
        for sensor in sensors:
            if not bool(sensor.get("required", False)):
                continue
            files = sensor_files(
                unit,
                platform_by_id[str(sensor["platform_id"])],
                str(sensor["modality"]),
            )
            if len(files) != len(frames):
                errors.append("required sensor file count mismatch: %s" % sensor["sensor_id"])
                continue
            for path in files:
                try:
                    if path.suffix.lower() == ".npy":
                        value = np.load(str(path), mmap_mode="r")
                        if value.size == 0 or value.dtype.kind not in "fiu":
                            raise ValueError("invalid numeric array")
                    else:
                        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if value is None or value.size == 0:
                            raise ValueError("unreadable image")
                except Exception as exc:
                    errors.append("unreadable required sensor file %s: %s" % (path.name, exc))
                    break
        for index, frame in enumerate(frames):
            calibration_path = unit / "calibration" / ("%06d.json" % index)
            if not calibration_path.is_file():
                errors.append("missing calibration frame: %06d" % index)
                continue
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if not bool(calibration.get("reprojection_ok", False)):
                errors.append("calibration reprojection roundtrip failed: %06d" % index)
            for sensor in sensors:
                sensor_id = str(sensor["sensor_id"])
                record = calibration.get("sensors", {}).get(sensor_id)
                if record is None:
                    errors.append("missing sensor calibration: %s" % sensor_id)
                    continue
                if str(sensor["modality"]) in {"rgb", "depth", "semantic", "normal"}:
                    intrinsic = record.get("intrinsic")
                    if intrinsic is None or np.asarray(intrinsic).shape != (3, 3):
                        errors.append("camera intrinsic is null or invalid: %s" % sensor_id)
                if not bool(record.get("closure_ok", False)) or float(record.get("closure_max_abs_error", 1.0)) > 1e-5:
                    errors.append("transform closure failed: %s" % sensor_id)
            if int(calibration.get("world_frame_id", -1)) != int(frame.get("world_frame_id", -2)):
                errors.append("calibration/frame id mismatch: %06d" % index)
        ground_quality_path = unit / "quality/ground_vehicle.json"
        if not ground_quality_path.is_file():
            errors.append("missing ground vehicle quality report")
        else:
            ground_quality = json.loads(ground_quality_path.read_text(encoding="utf-8"))
            if ground_quality.get("route_failure"):
                errors.append("committed unit contains ground route failure")
            if int(ground_quality.get("collision_count", 0)) > 0:
                errors.append("committed unit contains ground collision")
            if int(ground_quality.get("offroad_count", 0)) > 0:
                errors.append("committed unit contains ground offroad event")
            if float(ground_quality.get("max_pose_jump_m", 0.0)) > 10.0:
                errors.append("ground vehicle pose jump exceeds 10m")
        if not deferred_perception:
            sent_rows = read_jsonl(unit / "communication/messages.jsonl")
            event_rows = read_jsonl(unit / "communication/events.jsonl")
            if not sent_rows:
                errors.append("missing real algorithm communication log")
            sent_by_id = {str(row.get("message_id")): row for row in sent_rows}
            delivered_ids = {
                str(row.get("message_id")) for row in event_rows
                if row.get("event") == "consumed-by-fusion" and row.get("message_available")
            }
            for message_id in delivered_ids:
                if message_id not in sent_by_id or sent_by_id[message_id].get("dropped"):
                    errors.append("delivered message was absent or dropped: %s" % message_id)
            algorithm_rows: List[Dict[str, Any]] = []
            for path in (unit / "algorithms").rglob("*.jsonl") if (unit / "algorithms").is_dir() else []:
                algorithm_rows.extend(read_jsonl(path))
            encoded_algorithm = json.dumps(algorithm_rows, ensure_ascii=False)
            for forbidden in ("carla_actor_id", "global_object_uuid"):
                if ('"%s"' % forbidden) in encoded_algorithm:
                    errors.append("algorithm output contains forbidden GT field: %s" % forbidden)
            fused_rows = read_jsonl(unit / "algorithms/results/cooperative.jsonl")
            cross_platform = False
            for frame_row in fused_rows:
                for fused in frame_row.get("objects", []):
                    message_ids = set(map(str, fused.get("source_message_ids", [])))
                    if not message_ids.issubset(delivered_ids):
                        errors.append("fused result references an unavailable message")
                    contributors = list(map(str, fused.get("contributing_platforms", [])))
                    cross_platform = cross_platform or (
                        any(value.startswith("vehicle_") for value in contributors)
                        and any(not value.startswith("vehicle_") for value in contributors)
                    )
            if not cross_platform:
                errors.append("no fused result has both Air and Ground provenance")
        if manifest.get("task_id") == "task1":
            air = read_jsonl(unit / "annotations/by_sensor/uav_01_rgb.jsonl")
            ground = read_jsonl(unit / "annotations/by_sensor/vehicle_01_rgb.jsonl")
            for index in range(len(frames)):
                air_ids = {row.get("global_object_uuid") for row in air if int(row.get("sample_index", -1)) == index}
                ground_ids = {row.get("global_object_uuid") for row in ground if int(row.get("sample_index", -1)) == index}
                if not (air_ids & ground_ids):
                    errors.append("Task 1 frame has no jointly visible target: %d" % index)
        if manifest.get("task_id") == "task2":
            for field in ("joint_visible_ratio", "ground_target_visible_ratio", "vehicle_dominant_ratio", "max_consecutive_both_missing_frames"):
                if manifest.get(field) is None:
                    errors.append("Task 2 manifest field is null: %s" % field)
            manifest_ground_ratio = manifest.get("ground_target_visible_ratio")
            manifest_joint_ratio = manifest.get("joint_visible_ratio")
            if manifest_ground_ratio is None or float(manifest_ground_ratio) < 0.1:
                errors.append("Task 2 ground target visibility below minimum")
            visibility = read_jsonl(unit / "events/visibility.jsonl")
            target_ids = {
                str(row.get("target_global_uuid")) for row in visibility
                if row.get("target_global_uuid") is not None
            }
            if len(target_ids) != 1:
                errors.append("Task 2 must have exactly one target UUID")
            else:
                target_id = next(iter(target_ids))
                annotation_rows = {
                    path.stem: read_jsonl(path)
                    for path in (unit / "annotations/by_sensor").glob("*.jsonl")
                }
                air_rows = [
                    row for sensor, rows in annotation_rows.items()
                    if not sensor.startswith("vehicle_")
                    for row in rows
                    if str(row.get("global_object_uuid")) == target_id
                ]
                ground_rows = [
                    row for sensor, rows in annotation_rows.items()
                    if sensor.startswith("vehicle_")
                    for row in rows
                    if str(row.get("global_object_uuid")) == target_id
                ]
                air_frames = {
                    int(row.get("sample_index", -1)) for row in air_rows
                }
                ground_frames = {
                    int(row.get("sample_index", -1)) for row in ground_rows
                }
                joint_frames = air_frames & ground_frames
                if not ground_frames:
                    errors.append("Task 2 Ground view never contains the target UUID")
                if not joint_frames:
                    errors.append("Task 2 Air and Ground never contain the same target UUID in one frame")
                actual_ground_ratio = len(ground_frames) / max(1, len(frames))
                actual_joint_ratio = len(joint_frames) / max(1, len(frames))
                if manifest_ground_ratio is not None and abs(actual_ground_ratio - float(manifest_ground_ratio)) > 1.0 / max(1, len(frames)):
                    errors.append("Task 2 manifest ground visibility disagrees with target annotations")
                if manifest_joint_ratio is not None and abs(actual_joint_ratio - float(manifest_joint_ratio)) > 1.0 / max(1, len(frames)):
                    errors.append("Task 2 manifest joint visibility disagrees with target annotations")
            for name in ("air_track.jsonl", "ground_track.jsonl", "fused_track.jsonl"):
                if len(read_jsonl(unit / name)) != len(frames):
                    errors.append("Task 2 track output count mismatch: %s" % name)
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
            if (
                not deferred_perception
                and not manifest.get("ground_tracking_contribution_verified")
            ):
                errors.append("Task 3 has no verified Ground tracking contribution")
            annotation_rows = {
                path.stem: read_jsonl(path)
                for path in (unit / "annotations/by_sensor").glob("*.jsonl")
            }
            ground_rows = annotation_rows.get("vehicle_01_rgb", [])
            air_names = sorted(name for name in annotation_rows if name.startswith("air_camera_"))
            has_shared, has_ground_only = False, False
            for index in range(len(frames)):
                air_ids = {
                    row.get("global_object_uuid")
                    for name in air_names
                    for row in annotation_rows[name]
                    if int(row.get("sample_index", -1)) == index
                }
                ground_ids = {
                    row.get("global_object_uuid") for row in ground_rows
                    if int(row.get("sample_index", -1)) == index
                }
                has_shared = has_shared or bool(air_ids & ground_ids)
                has_ground_only = has_ground_only or bool(ground_ids - air_ids)
            if not has_shared:
                errors.append("Task 3 has no Air-Ground shared GT track")
            if not has_ground_only:
                errors.append("Task 3 has no Ground-only blind-spot supplement")
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
