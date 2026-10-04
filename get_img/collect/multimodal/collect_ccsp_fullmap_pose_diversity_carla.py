#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect balanced, quality-filtered multimodal frames across CCSP."""

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_TARGET_RGB_FRAMES = 30
ZONE_COUNT = 18
ZONE_RADIUS_M = 140.0
MAX_REQUEST_PER_BATCH = 30
MIN_SPAWN_POINTS_PER_ZONE = 8
MAP_BOUNDARY_MARGIN_M = 300.0

WEATHER_PRESETS = (
    "ClearNoon",
    "ClearSunset",
    "ClearNight",
    "FoggyNoon",
    "SnowNoon",
    "DustStorm",
)

# These roadbuild candidates combine the clean diagnostic regions with distant
# spawn-point clusters. Every accepted frame is still checked at runtime for
# complete streaming and severe normal/depth discontinuities.
VALIDATED_CCSP_ZONE_CENTERS = (
    (98.878891, 14.092216),
    (-282.019745, -71.091003),
    (42.306549, 258.378998),
    (-506.667755, -167.070557),
    (289.226318, 89.016861),
    # Additional distant road clusters expand production runs beyond the five
    # diagnostic regions. Every accepted frame from these regions still passes
    # the same streaming gate and the normal/depth artifact quarantine below.
    (170.904236, -422.130707),
    (8.231360, 629.470582),
    (699.579529, 100.385612),
    (-875.677307, 344.129944),
    # Farthest-point road clusters selected from all 476 roadbuild spawn
    # points. These stay at least 160 m from the earlier production centers.
    (-73.729172, 7.775955),
    (-539.243286, -328.547791),
    (361.870392, -525.611572),
    (-647.121094, -87.445564),
    (-794.920349, 192.594193),
    (-1013.059204, 432.224182),
    (859.682007, 151.008057),
    (-759.929810, 456.517242),
    (805.467163, -69.742455),
)
VALIDATED_ZONE_MATCH_MAX_DISTANCE_M = 25.0

# This part of roadbuild contains baked photogrammetry vehicle fragments. The
# geometry can finish streaming normally, so a streaming-only check cannot
# distinguish it from a valid road surface. Reject every camera inside the
# affected area and keep collection zones far enough away that their 140 m ROI
# cannot drift back into it.
KNOWN_BAD_ASSET_CAMERA_CIRCLES = (
    (884.08, 601.00, 220.0),
)

# Severe broken mesh regions produce dense normal discontinuities together
# with large depth jumps. Requiring both avoids rejecting ordinary foliage.
MAX_BROKEN_NORMAL_EDGE_RATIO = 0.30
MAX_BROKEN_DEPTH_EDGE_RATIO = 0.06

COLLECT_DIR = Path(__file__).resolve().parent
COLLECTOR = COLLECT_DIR / "collect_rpg_small_targets_carla_v2.py"
COLLECTOR_BOOTSTRAP = COLLECT_DIR / "run_ccsp_streaming_collector.py"
CONFIG = COLLECT_DIR / "collection_config.json"
RGB_ASSET_QA = COLLECT_DIR / "qa_ccsp_rgb_unmatched_vehicles.py"
DEFAULT_OUTPUT = COLLECT_DIR.parent.parent / "dataset_ccsp_fullmap_pose_diversity"

CCSP_PYTHON = Path(
    r"E:\OpenHUTB\中电软件园\carla_0.9.15_py37\python.exe"
)
CCSP_CARLA_EGG = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor"
    r"\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
)
YOLO_QA_PYTHON = Path(r"D:\anaconda2023.09\envs\yolo11n\python.exe")
YOLO_QA_REPO = Path(r"E:\YOLO\ultralytics-main\ultralytics-main")
YOLO_QA_WEIGHTS = YOLO_QA_REPO / "yolo11n.pt"
RGB_QA_RESULT_MARKER = "CCSP_RGB_QA_JSON="


def parse_runner_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a scene-streaming-verified CCSP multimodal dataset."
        )
    )
    parser.add_argument(
        "--recollect",
        action="store_true",
        help=(
            "Archive the existing output and perform a fresh collection. "
            "Without this flag, a complete existing dataset is only verified."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_TARGET_RGB_FRAMES,
        help="需要保存的合格 RGB 帧数；必须能被六种天气整除。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="中电园多模态数据集输出目录。",
    )
    return parser.parse_args()


def archive_existing_output(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root.with_name(f"{root.name}_backup_{timestamp}")
    suffix = 1
    while backup.exists():
        backup = root.with_name(f"{root.name}_backup_{timestamp}_{suffix}")
        suffix += 1
    root.rename(backup)
    print(f"[BACKUP] Existing dataset moved to: {backup}")
    return backup


def rgb_frame_count(root: Path) -> int:
    """Count canonical RGB files across every sequence and weather folder."""
    paired_root = root / "paired_weather"
    if not paired_root.exists():
        return 0
    return sum(
        1
        for path in paired_root.glob("seq_*/rgb/*/*.png")
        if path.is_file()
    )


def sequence_rgb_count(sequence_dir: Path) -> int:
    rgb_dir = sequence_dir / "rgb"
    if not rgb_dir.exists():
        return 0
    return sum(1 for path in rgb_dir.glob("*/*.png") if path.is_file())


def ensure_all_weather_directories(root: Path) -> None:
    """Expose all six weather folders even while a partial run is in progress."""
    sequence_dir = root / "paired_weather" / "seq_0000" / "rgb"
    if not sequence_dir.parent.exists():
        return
    for weather_name in WEATHER_PRESETS:
        (sequence_dir / weather_name).mkdir(parents=True, exist_ok=True)


def validate_paths() -> None:
    for path in (
        CCSP_PYTHON,
        CCSP_CARLA_EGG,
        COLLECTOR,
        COLLECTOR_BOOTSTRAP,
        CONFIG,
        RGB_ASSET_QA,
        YOLO_QA_PYTHON,
        YOLO_QA_REPO,
        YOLO_QA_WEIGHTS,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")


def squared_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def select_collection_zones() -> Tuple[str, List[Tuple[float, float]]]:
    """Select dispersed road-dense zones from the simulator's current map."""
    sys.path.insert(0, str(CCSP_CARLA_EGG))
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(f"Cannot import CARLA API from {CCSP_CARLA_EGG}") from exc

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    carla_map = world.get_map()
    points = [
        (float(transform.location.x), float(transform.location.y))
        for transform in carla_map.get_spawn_points()
    ]
    if len(points) < ZONE_COUNT:
        raise RuntimeError(
            f"Current map exposes only {len(points)} vehicle spawn points; "
            f"at least {ZONE_COUNT} are required."
        )

    radius_sq = ZONE_RADIUS_M ** 2
    densities = [
        sum(1 for other in points if squared_distance(point, other) <= radius_sq)
        for point in points
    ]
    if "roadbuild" in str(carla_map.name).lower():
        selected = []
        for expected in VALIDATED_CCSP_ZONE_CENTERS:
            nearest_index = min(
                range(len(points)),
                key=lambda index: squared_distance(points[index], expected),
            )
            distance_m = math.sqrt(
                squared_distance(points[nearest_index], expected)
            )
            if distance_m > VALIDATED_ZONE_MATCH_MAX_DISTANCE_M:
                raise RuntimeError(
                    "The current roadbuild map does not match the validated "
                    f"CCSP test layout near {expected}; nearest road spawn is "
                    f"{distance_m:.1f}m away."
                )
            if nearest_index in selected:
                raise RuntimeError(
                    "Validated CCSP test zones resolved to duplicate spawn points."
                )
            selected.append(nearest_index)

        zones = [points[index] for index in selected]
        for zone in zones:
            for bad_x, bad_y, bad_radius_m in KNOWN_BAD_ASSET_CAMERA_CIRCLES:
                clearance_m = math.sqrt(
                    squared_distance(zone, (bad_x, bad_y))
                )
                required_clearance_m = bad_radius_m + ZONE_RADIUS_M
                if clearance_m <= required_clearance_m:
                    raise RuntimeError(
                        "A configured CCSP zone can reach the known broken "
                        f"asset area: zone={zone}, clearance={clearance_m:.1f}m, "
                        f"required>{required_clearance_m:.1f}m."
                    )
        print(
            f"[MAP] {carla_map.name}: spawn_points={len(points)}, "
            f"runtime_checked_candidate_zones={len(zones)}, radius={ZONE_RADIUS_M:.0f}m"
        )
        for zone_index, (center_x, center_y) in enumerate(zones, start=1):
            density = densities[selected[zone_index - 1]]
            print(
                f"[ZONE {zone_index:02d}] center=({center_x:.2f}, "
                f"{center_y:.2f}), spawn_points_in_radius={density}, "
                "asset_qa=runtime_visual_gate"
            )
        return carla_map.name, zones

    dense_indices = [
        index
        for index, density in enumerate(densities)
        if density >= MIN_SPAWN_POINTS_PER_ZONE
    ]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    interior_dense_indices = [
        index
        for index in dense_indices
        if (
            min_x + MAP_BOUNDARY_MARGIN_M <= points[index][0]
            <= max_x - MAP_BOUNDARY_MARGIN_M
            and min_y + MAP_BOUNDARY_MARGIN_M <= points[index][1]
            <= max_y - MAP_BOUNDARY_MARGIN_M
        )
    ]
    if len(interior_dense_indices) >= ZONE_COUNT:
        dense_indices = interior_dense_indices
    if len(dense_indices) < ZONE_COUNT:
        dense_indices = list(range(len(points)))

    centroid = (
        sum(points[index][0] for index in dense_indices) / len(dense_indices),
        sum(points[index][1] for index in dense_indices) / len(dense_indices),
    )
    first_index = max(
        dense_indices,
        key=lambda index: (
            densities[index],
            -squared_distance(points[index], centroid),
        ),
    )
    selected = [first_index]
    remaining = set(dense_indices)
    remaining.remove(first_index)
    max_density = max(densities[index] for index in dense_indices)

    while len(selected) < ZONE_COUNT and remaining:
        next_index = max(
            remaining,
            key=lambda index: (
                min(
                    squared_distance(points[index], points[chosen])
                    for chosen in selected
                )
                * (0.75 + 0.25 * densities[index] / max_density),
                densities[index],
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)

    zones = [points[index] for index in selected]
    print(
        f"[MAP] {carla_map.name}: spawn_points={len(points)}, "
        f"selected_zones={len(zones)}, radius={ZONE_RADIUS_M:.0f}m"
    )
    for zone_index, (center_x, center_y) in enumerate(zones, start=1):
        density = densities[selected[zone_index - 1]]
        print(
            f"[ZONE {zone_index:02d}] center=({center_x:.2f}, {center_y:.2f}), "
            f"spawn_points_in_radius={density}"
        )
    return carla_map.name, zones


def read_sequence_zone(sequence_dir: Path) -> Tuple[float, float]:
    metadata_path = sequence_dir / "sequence_meta.json"
    if not metadata_path.exists():
        raise ValueError("sequence metadata is missing")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    center = metadata["camera_roi"]["center"]
    return float(center[0]), float(center[1])


def existing_zone_counts(
    root: Path,
    zones: Sequence[Tuple[float, float]],
) -> Dict[int, int]:
    """Recover per-zone progress from the surviving accepted frames."""
    counts = {index: 0 for index in range(len(zones))}
    paired_root = root / "paired_weather"
    if not paired_root.exists():
        return counts

    for sequence_dir in paired_root.glob("seq_*"):
        annotation_paths = sorted((sequence_dir / "annotations").glob("*.json"))
        if not annotation_paths:
            continue

        # Compacted batches carry the exact originating zone on every frame.
        # Counting annotations also means frames removed by either QA gate stop
        # contributing immediately instead of leaving stale metadata counts.
        counted_frames = 0
        for annotation_path in annotation_paths:
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                center_value = annotation["collection_zone"]["center"]
                center = (float(center_value[0]), float(center_value[1]))
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
            nearest_index = min(
                range(len(zones)),
                key=lambda index: squared_distance(center, zones[index]),
            )
            if math.sqrt(squared_distance(center, zones[nearest_index])) <= 1.0:
                counts[nearest_index] += 1
                counted_frames += 1

        if counted_frames == len(annotation_paths):
            continue

        # A fresh collector batch has no per-frame collection_zone until its
        # first compaction, so fall back to the sequence-level ROI once.
        try:
            center = read_sequence_zone(sequence_dir)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
        nearest_index = min(
            range(len(zones)),
            key=lambda index: squared_distance(center, zones[index]),
        )
        if math.sqrt(squared_distance(center, zones[nearest_index])) <= 1.0:
            counts[nearest_index] += len(annotation_paths) - counted_frames
    return counts


def sequence_visual_artifact_metrics(
    sequence_dir: Path,
) -> List[Dict[str, float]]:
    """Measure severe geometry discontinuities in one newly collected batch."""
    metrics: List[Dict[str, float]] = []
    normal_dir = sequence_dir / "surface_normal" / "npy"
    depth_dir = sequence_dir / "depth" / "npy"
    for normal_path in sorted(normal_dir.glob("*.npy")):
        depth_path = depth_dir / normal_path.name
        annotation_path = (
            sequence_dir / "annotations" / f"{normal_path.stem}.json"
        )
        known_bad_asset_region = False
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            camera_location = annotation["camera_transform"]["location"]
            camera_xy = (
                float(camera_location["x"]),
                float(camera_location["y"]),
            )
            known_bad_asset_region = any(
                math.sqrt(squared_distance(camera_xy, (center_x, center_y)))
                <= radius_m
                for center_x, center_y, radius_m
                in KNOWN_BAD_ASSET_CAMERA_CIRCLES
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            # Missing/corrupt annotations are handled by the normal dataset
            # consistency checks; do not hide that failure as an asset match.
            known_bad_asset_region = False

        if not depth_path.exists():
            metrics.append({
                "frame": float(int(normal_path.stem)),
                "normal_edge_ratio": 1.0,
                "depth_edge_ratio": 1.0,
                "known_bad_asset_region": float(known_bad_asset_region),
                "rejected": 1.0,
            })
            continue

        normal = np.load(str(normal_path)).astype(np.float32)[::8, ::8]
        depth = np.load(str(depth_path)).astype(np.float32)[::8, ::8]
        normal_dx = np.linalg.norm(np.diff(normal, axis=1), axis=2)
        normal_dy = np.linalg.norm(np.diff(normal, axis=0), axis=2)
        depth_dx = np.abs(np.diff(depth, axis=1))
        depth_dy = np.abs(np.diff(depth, axis=0))
        normal_edge_ratio = float(
            (np.mean(normal_dx > 0.25) + np.mean(normal_dy > 0.25)) / 2.0
        )
        depth_edge_ratio = float(
            (np.mean(depth_dx > 2.0) + np.mean(depth_dy > 2.0)) / 2.0
        )
        rejected = bool(
            known_bad_asset_region
            or (
                normal_edge_ratio >= MAX_BROKEN_NORMAL_EDGE_RATIO
                and depth_edge_ratio >= MAX_BROKEN_DEPTH_EDGE_RATIO
            )
        )
        metrics.append({
            "frame": float(int(normal_path.stem)),
            "normal_edge_ratio": normal_edge_ratio,
            "depth_edge_ratio": depth_edge_ratio,
            "known_bad_asset_region": float(known_bad_asset_region),
            "rejected": float(rejected),
        })
    return metrics


def remove_frame_bundle(
    root: Path,
    sequence_dir: Path,
    frame_stem: str,
    quarantine_group: str,
    reason: Dict[str, object],
) -> int:
    """Move one synchronized multimodal frame out of an accepted sequence."""
    quarantine_dir = (
        root / quarantine_group / sequence_dir.name / frame_stem
    )
    moved = 0
    for source_path in list(sequence_dir.rglob(f"{frame_stem}.*")):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(sequence_dir)
        destination = quarantine_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination))
        moved += 1

    for csv_name in ("frame_index.csv", "groundtruth_multi.csv"):
        csv_path = sequence_dir / csv_name
        fieldnames, rows = load_csv_rows(csv_path)
        if not fieldnames:
            continue
        filtered = [
            row for row in rows
            if row.get("frame_id", "") != str(int(frame_stem))
        ]
        write_csv_rows(csv_path, fieldnames, filtered)

    if moved:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        (quarantine_dir / "rejection_reason.json").write_text(
            json.dumps(reason, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return moved


def quarantine_broken_visual_frames(
    root: Path,
    new_sequence_names: Sequence[str],
) -> List[str]:
    """Remove only frames with severe broken meshes from new batches."""
    rejected: List[str] = []
    paired_root = root / "paired_weather"
    for sequence_name in sorted(new_sequence_names):
        sequence_dir = paired_root / sequence_name
        if not sequence_dir.exists():
            continue
        metrics = sequence_visual_artifact_metrics(sequence_dir)
        failed = [item for item in metrics if bool(item["rejected"])]
        if not failed:
            continue
        for item in failed:
            frame_stem = f"{int(item['frame']):06d}"
            moved = remove_frame_bundle(
                root,
                sequence_dir,
                frame_stem,
                "_rejected_visual_artifact_frames",
                {
                    "reason": "broken_ccsp_geometry_or_texture",
                    "normal_edge_ratio": item["normal_edge_ratio"],
                    "depth_edge_ratio": item["depth_edge_ratio"],
                    "known_bad_asset_region": bool(
                        item["known_bad_asset_region"]
                    ),
                },
            )
            if moved:
                rejected.append(f"{sequence_name}/{frame_stem}")
                print(
                    f"[ASSET-QA] Removed {sequence_name}/{frame_stem}: "
                    f"normal_edge={item['normal_edge_ratio']:.3f}, "
                    f"depth_edge={item['depth_edge_ratio']:.3f}, "
                    "known_bad_asset_region="
                    f"{bool(item['known_bad_asset_region'])}"
                )
    return rejected


def quarantine_unmatched_rgb_vehicles(root: Path) -> List[str]:
    """Reject RGB frames containing unlabelled baked vehicle-like assets."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(YOLO_QA_REPO)
    command = [
        str(YOLO_QA_PYTHON),
        str(RGB_ASSET_QA),
        "--root", str(root),
        "--weights", str(YOLO_QA_WEIGHTS),
        "--imgsz", "1280",
        "--conf", "0.20",
        "--iou-match", "0.15",
        "--min-area-ratio", "0.001",
        "--device", "0",
    ]
    completed = subprocess.run(
        command,
        cwd=str(YOLO_QA_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    output_lines = completed.stdout.splitlines()
    payload_line = next(
        (
            line for line in reversed(output_lines)
            if line.startswith(RGB_QA_RESULT_MARKER)
        ),
        None,
    )
    if completed.returncode != 0 or payload_line is None:
        tail = "\n".join(output_lines[-20:])
        raise RuntimeError(
            "YOLO RGB asset QA failed; unchecked frames will not be accepted.\n"
            + tail
        )

    payload = json.loads(payload_line[len(RGB_QA_RESULT_MARKER):])
    rejected: List[str] = []
    for item in payload.get("rejected", []):
        sequence_name = str(item["sequence"])
        frame_stem = str(item["frame"])
        sequence_dir = root / "paired_weather" / sequence_name
        moved = remove_frame_bundle(
            root,
            sequence_dir,
            frame_stem,
            "_rejected_rgb_unmatched_vehicle_frames",
            {
                "reason": "rgb_vehicle_without_simulator_ground_truth",
                "detector": "COCO-pretrained YOLO11n used only for QA",
                "unmatched_vehicle_detections": item[
                    "unmatched_vehicle_detections"
                ],
            },
        )
        if moved:
            rejected.append(f"{sequence_name}/{frame_stem}")
            print(
                f"[RGB-ASSET-QA] Removed {sequence_name}/{frame_stem}: "
                f"unmatched_vehicles="
                f"{len(item['unmatched_vehicle_detections'])}"
            )
    print(
        f"[RGB-ASSET-QA] reviewed={payload.get('reviewed', 0)}, "
        f"rejected={len(rejected)}"
    )
    return rejected


def trim_weather_overflow(
    root: Path,
    new_sequence_names: Sequence[str],
    max_per_weather: int,
) -> List[str]:
    """Keep every weather at or below its exact final quota."""
    removed: List[str] = []
    paired_root = root / "paired_weather"
    counts = collected_weather_counts(root)
    for sequence_name in reversed(sorted(new_sequence_names)):
        sequence_dir = paired_root / sequence_name
        annotation_paths = sorted(
            (sequence_dir / "annotations").glob("*.json"),
            reverse=True,
        )
        for annotation_path in annotation_paths:
            try:
                annotation = json.loads(
                    annotation_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            weather = str(annotation.get("canonical_weather", ""))
            if weather not in counts or counts[weather] <= max_per_weather:
                continue
            frame_stem = annotation_path.stem
            moved = remove_frame_bundle(
                root,
                sequence_dir,
                frame_stem,
                "_rejected_weather_overflow_frames",
                {
                    "reason": "weather_quota_overflow",
                    "weather": weather,
                    "quota": max_per_weather,
                },
            )
            if moved:
                counts[weather] -= 1
                removed.append(f"{sequence_name}/{frame_stem}")
                print(
                    f"[WEATHER-QA] Removed {sequence_name}/{frame_stem}: "
                    f"{weather} exceeded quota {max_per_weather}"
                )
    return removed


def collected_weather_counts(root: Path) -> Dict[str, int]:
    counts = {name: 0 for name in WEATHER_PRESETS}
    annotation_root = root / "paired_weather"
    for annotation_path in annotation_root.glob("seq_*/annotations/*.json"):
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            weather_name = str(payload.get("canonical_weather", ""))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if weather_name in counts:
            counts[weather_name] += 1
    return counts


def plan_weather_batch(
    counts: Dict[str, int],
    requested: int,
    seed: int,
) -> List[str]:
    """Select the currently least represented weather for every requested frame."""
    working = {name: int(counts.get(name, 0)) for name in WEATHER_PRESETS}
    rng = random.Random(seed)
    planned: List[str] = []
    for _ in range(requested):
        minimum = min(working.values())
        candidates = [
            name for name in WEATHER_PRESETS if working[name] == minimum
        ]
        rng.shuffle(candidates)
        selected = candidates[0]
        planned.append(selected)
        working[selected] += 1
    return planned


def remap_annotation_value(
    value,
    old_sequence: str,
    old_stem: str,
    new_stem: str,
):
    """Recursively rewrite sequence paths and frame file names."""
    if isinstance(value, dict):
        return {
            key: remap_annotation_value(item, old_sequence, old_stem, new_stem)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            remap_annotation_value(item, old_sequence, old_stem, new_stem)
            for item in value
        ]
    if isinstance(value, str):
        value = value.replace(old_sequence, "paired_weather/seq_0000")
        return re.sub(
            r"(?<=/)" + re.escape(old_stem) + r"(?=\.)",
            new_stem,
            value,
        )
    return value


def load_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def compact_sequences_to_one(root: Path) -> int:
    """Merge accepted regional batches into one canonical seq_0000."""
    paired_root = root / "paired_weather"
    sequence_dirs = sorted(path for path in paired_root.glob("seq_*") if path.is_dir())
    total_frames = sum(sequence_rgb_count(path) for path in sequence_dirs)
    if total_frames <= 0:
        return 0
    if len(sequence_dirs) == 1 and sequence_dirs[0].name == "seq_0000":
        return total_frames

    staging = paired_root / ".single_sequence_staging"
    backup = paired_root / ".regional_sequences_backup"
    if staging.exists() or backup.exists():
        raise RuntimeError(
            "A previous compaction staging/backup directory exists. "
            f"Inspect {staging} and {backup} before retrying."
        )
    staging.mkdir(parents=True)

    merged_frame_index: List[Dict[str, str]] = []
    merged_multi: List[Dict[str, str]] = []
    merged_groundtruth: List[str] = []
    frame_index_fields: List[str] = []
    multi_fields: List[str] = []
    source_metas: List[dict] = []
    weather_counts: Dict[str, int] = {}
    zone_counts: Dict[str, int] = {}
    new_frame_id = 0

    for source_dir in sequence_dirs:
        annotation_paths = sorted((source_dir / "annotations").glob("*.json"))
        if not annotation_paths:
            continue
        try:
            source_meta = json.loads(
                (source_dir / "sequence_meta.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            source_meta = {}
        source_sequence = str(
            source_meta.get("sequence", f"paired_weather/{source_dir.name}")
        )
        source_metas.append(source_meta)
        roi = source_meta.get("camera_roi", {})
        default_center = roi.get("center")

        index_fields, index_rows = load_csv_rows(source_dir / "frame_index.csv")
        if index_fields and not frame_index_fields:
            frame_index_fields = index_fields
        index_by_id = {row.get("frame_id", ""): row for row in index_rows}

        current_multi_fields, current_multi_rows = load_csv_rows(
            source_dir / "groundtruth_multi.csv"
        )
        if current_multi_fields and not multi_fields:
            multi_fields = current_multi_fields
        multi_by_id: Dict[str, List[Dict[str, str]]] = {}
        for row in current_multi_rows:
            multi_by_id.setdefault(row.get("frame_id", ""), []).append(row)

        groundtruth_path = source_dir / "groundtruth.txt"
        groundtruth_lines = (
            groundtruth_path.read_text(encoding="utf-8").splitlines()
            if groundtruth_path.exists()
            else []
        )

        for annotation_path in annotation_paths:
            old_stem = annotation_path.stem
            old_frame_id = int(old_stem)
            new_stem = f"{new_frame_id:06d}"

            for source_file in source_dir.rglob(f"{old_stem}.*"):
                relative = source_file.relative_to(source_dir)
                if relative.parts[0] == "annotations":
                    continue
                destination = staging / relative.parent / (
                    new_stem + source_file.suffix
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source_file), str(destination))

            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            existing_collection_zone = annotation.get("collection_zone", {})
            frame_center = existing_collection_zone.get("center", default_center)
            if not isinstance(frame_center, (list, tuple)) or len(frame_center) < 2:
                shutil.rmtree(str(staging))
                raise RuntimeError(
                    f"Cannot recover collection zone for {annotation_path}. "
                    "Original regional sequences were left unchanged."
                )
            frame_center = [float(frame_center[0]), float(frame_center[1])]
            frame_radius = existing_collection_zone.get(
                "radius_m", roi.get("radius_m", ZONE_RADIUS_M)
            )
            annotation = remap_annotation_value(
                annotation,
                source_sequence,
                old_stem,
                new_stem,
            )
            annotation["sequence"] = "paired_weather/seq_0000"
            annotation["frame_id"] = new_frame_id
            annotation["source_regional_batch"] = source_dir.name
            annotation["source_frame_id"] = old_frame_id
            annotation["collection_zone"] = {
                "center": frame_center,
                "radius_m": frame_radius,
            }
            zone_key = f"{frame_center[0]:.6f},{frame_center[1]:.6f}"
            zone_counts[zone_key] = zone_counts.get(zone_key, 0) + 1
            weather = annotation.get("canonical_weather")
            if weather:
                weather_counts[weather] = weather_counts.get(weather, 0) + 1
            destination_annotation = staging / "annotations" / f"{new_stem}.json"
            destination_annotation.parent.mkdir(parents=True, exist_ok=True)
            destination_annotation.write_text(
                json.dumps(annotation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            source_index_row = index_by_id.get(str(old_frame_id))
            if source_index_row is not None:
                remapped_row = {
                    key: remap_annotation_value(
                        value,
                        source_sequence,
                        old_stem,
                        new_stem,
                    )
                    for key, value in source_index_row.items()
                }
                remapped_row["frame_id"] = str(new_frame_id)
                merged_frame_index.append(remapped_row)

            for source_row in multi_by_id.get(str(old_frame_id), []):
                remapped_row = dict(source_row)
                remapped_row["frame_id"] = str(new_frame_id)
                if remapped_row.get("track_id"):
                    remapped_row["track_id"] = (
                        f"{source_dir.name}_{remapped_row['track_id']}"
                    )
                merged_multi.append(remapped_row)

            if old_frame_id < len(groundtruth_lines):
                merged_groundtruth.append(groundtruth_lines[old_frame_id])
            else:
                merged_groundtruth.append("0,0,0,0")
            new_frame_id += 1

    if new_frame_id != total_frames:
        shutil.rmtree(str(staging))
        raise RuntimeError(
            f"Compaction found {new_frame_id} annotations for {total_frames} RGB frames. "
            "Original regional sequences were left unchanged."
        )

    required_dirs = (
        "rgb", "depth/npy", "depth/vis_16bit", "depth/color",
        "surface_normal/png", "surface_normal/npy",
        "segmentation/id", "segmentation/color", "annotations", "labels_yolo",
    )
    for relative_dir in required_dirs:
        directory = staging / relative_dir
        file_count = sum(1 for path in directory.rglob("*") if path.is_file())
        if file_count != total_frames:
            shutil.rmtree(str(staging))
            raise RuntimeError(
                f"Compaction validation failed for {relative_dir}: "
                f"expected {total_frames}, found {file_count}. "
                "Original regional sequences were left unchanged."
            )

    if frame_index_fields:
        write_csv_rows(
            staging / "frame_index.csv",
            frame_index_fields,
            merged_frame_index,
        )
    if multi_fields:
        write_csv_rows(
            staging / "groundtruth_multi.csv",
            multi_fields,
            merged_multi,
        )
    (staging / "groundtruth.txt").write_text(
        "\n".join(merged_groundtruth) + "\n",
        encoding="utf-8",
    )

    merged_meta = dict(source_metas[0]) if source_metas else {}
    merged_meta.update({
        "sequence": "paired_weather/seq_0000",
        "sequence_name": "seq_0000",
        "relative_sequence_path": "paired_weather/seq_0000",
        "frame_count": total_frames,
        "regional_batches_compacted": True,
        "regional_batch_count": len(sequence_dirs),
        "collection_zones": [
            {
                "center": [float(key.split(",")[0]), float(key.split(",")[1])],
                "accepted_frames": count,
                "radius_m": ZONE_RADIUS_M,
            }
            for key, count in zone_counts.items()
        ],
        "camera_roi": {
            "enabled": True,
            "mode": "multiple_balanced_regions",
            "radius_m": ZONE_RADIUS_M,
            "zones": len(zone_counts),
        },
        "planned_weather_counts": weather_counts,
        "captured_weather_counts": weather_counts,
        "asset_quality_gate": {
            "max_normal_edge_ratio": MAX_BROKEN_NORMAL_EDGE_RATIO,
            "max_depth_edge_ratio": MAX_BROKEN_DEPTH_EDGE_RATIO,
            "joint_edge_threshold_required": True,
            "known_bad_camera_circles": [
                [center_x, center_y, radius_m]
                for center_x, center_y, radius_m
                in KNOWN_BAD_ASSET_CAMERA_CIRCLES
            ],
        },
        "sot_primary_track_id": None,
        "sot_primary_visible_frames": 0,
        "sot_primary_policy": "not applicable: independent regional detection frames",
    })
    (staging / "sequence_meta.json").write_text(
        json.dumps(merged_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    backup.mkdir()
    try:
        for source_dir in sequence_dirs:
            shutil.move(str(source_dir), str(backup / source_dir.name))
        shutil.move(str(staging), str(paired_root / "seq_0000"))
        if sequence_rgb_count(paired_root / "seq_0000") != total_frames:
            raise RuntimeError("Final single-sequence verification failed.")
    except Exception:
        final_dir = paired_root / "seq_0000"
        if final_dir.exists():
            shutil.move(str(final_dir), str(staging))
        for saved_dir in sorted(backup.iterdir()):
            shutil.move(str(saved_dir), str(paired_root / saved_dir.name))
        raise
    else:
        shutil.rmtree(str(backup))

    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sequences"] = [merged_meta]
        manifest["sequence_layout"] = "single_sequence"
        manifest["regional_batches_compacted"] = True
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        f"[COMPACT] Merged {len(sequence_dirs)} regional batches into "
        f"paired_weather/seq_0000 ({total_frames} frames)."
    )
    return total_frames


def collector_command(
    output: Path,
    requested: int,
    preserve: bool,
    seed: int,
    zone: Tuple[float, float],
    weather_presets: Sequence[str],
) -> List[str]:
    center_x, center_y = zone
    command = [
        str(CCSP_PYTHON),
        "-u",
        str(COLLECTOR_BOOTSTRAP),
        "--config", str(CONFIG),
        "--out", str(output),
        "--host", "127.0.0.1",
        "--port", "2000",
        "--timeout", "300",
        "--frames", str(requested),
        "--sequences", "1",
        "--route", "random",
        "--random-weather",
        "--weather-presets", *weather_presets,
        # A 45-degree view keeps the camera on the drivable corridor and avoids
        # pulling CCSP's damaged photogrammetry at the roadside into frame.
        "--fov", "45",
        "--vehicles", "6",
        # Keep the original target-size and visibility thresholds. A denser
        # pedestrian population only makes valid road-facing poses less rare.
        "--walkers", "120",
        "--camera-roi-center-x", f"{center_x:.6f}",
        "--camera-roi-center-y", f"{center_y:.6f}",
        "--camera-roi-radius-m", f"{ZONE_RADIUS_M:.1f}",
        "--camera-min-position-distance-m", "12",
        "--camera-min-yaw-difference-deg", "20",
        "--camera-pose-history-window", "0",
        "--camera-grid-size-m", "15",
        "--max-frames-per-camera-grid", "3",
        "--ground-vehicles-on-spawn",
        "--vehicle-ground-clearance-m", "0.05",
        "--vehicle-spawn-min-distance-m", "35.0",
        "--vehicle-motion-mode", "static",
        "--target-semantic-source", "actor_semantic_depth",
        "--spectator-follow-camera",
        # Every candidate pose becomes a streaming source. The test bootstrap
        # keeps retrying until RGB texture/LOD content is stable, so nearby
        # poses cannot inherit a stale low-detail frame from the previous pose.
        "--streaming-warmup-frames", "30",
        "--streaming-rewarm-distance-m", "0",
        "--streaming-min-geometry-ratio", "0.99",
        # CCSP geometry is complete but many custom assets are intentionally
        # Unlabeled in CARLA. Keep reporting this ratio without using it as a
        # hard streaming gate; near-field depth geometry remains mandatory.
        "--streaming-min-labeled-ratio", "0.0",
        "--streaming-geometry-max-depth-m", "220",
        "--streaming-readiness-retries", "12",
        "--streaming-readiness-step-frames", "20",
        "--road-validation-mode", "geometry",
        "--max-camera-pose-retries", "50",
        "--max-expensive-pose-retries", "12",
        "--png-compression-level", "1",
        "--seed", str(seed),
    ]
    command.append(
        "--preserve-existing-sequences" if preserve else "--overwrite-sequences"
    )
    return command


def main() -> int:
    runner_args = parse_runner_args()
    target_rgb_frames = int(runner_args.frames)
    output = runner_args.out.resolve()
    if target_rgb_frames <= 0:
        raise ValueError("--frames must be greater than zero")
    if target_rgb_frames % len(WEATHER_PRESETS) != 0:
        raise ValueError(
            f"--frames must be divisible by {len(WEATHER_PRESETS)} so all "
            "weather presets have an exact equal quota"
        )
    # The runtime-checked regions have different road and actor capacities.
    # Require broad map coverage without forcing sparse regions to manufacture
    # repetitive views just to match the densest region exactly.
    min_frames_per_zone = max(1, target_rgb_frames // (ZONE_COUNT * 4))
    max_frames_per_zone = max(
        min_frames_per_zone,
        math.ceil(target_rgb_frames * 0.20),
    )
    # Large production runs can accept far fewer frames than requested because
    # every candidate still has to pass target-size, diversity, streaming and
    # artifact checks. Keep topping up instead of treating the diagnostic-sized
    # round budget as a collection failure.
    max_collection_rounds = max(
        60,
        math.ceil(target_rgb_frames / MAX_REQUEST_PER_BATCH) * 10,
    )
    weather_quota = target_rgb_frames // len(WEATHER_PRESETS)
    validate_paths()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CCSP_CARLA_EGG)

    if runner_args.recollect and output.exists():
        archive_existing_output(output)

    paired_root = output / "paired_weather"
    existing_sequences = sorted(
        path.name for path in paired_root.glob("seq_*") if path.is_dir()
    )
    if existing_sequences:
        recovered_rejections = quarantine_broken_visual_frames(
            output,
            existing_sequences,
        )
        recovered_weather_overflow = trim_weather_overflow(
            output,
            existing_sequences,
            weather_quota,
        )
        if recovered_rejections or recovered_weather_overflow:
            print(
                "[RESUME-QA] cleaned existing partial output: "
                f"artifact_frames={recovered_rejections}, "
                f"weather_overflow_frames={recovered_weather_overflow}"
            )
        recovered_rgb_rejections = quarantine_unmatched_rgb_vehicles(output)
        if recovered_rgb_rejections:
            print(
                "[RESUME-QA] removed RGB/GT-inconsistent vehicle frames: "
                f"{recovered_rgb_rejections}"
            )
        compact_sequences_to_one(output)
        ensure_all_weather_directories(output)

    current = rgb_frame_count(output)
    if current >= target_rgb_frames:
        compacted = compact_sequences_to_one(output)
        if compacted != target_rgb_frames:
            raise RuntimeError(
                f"Expected exactly {target_rgb_frames} existing RGB frames, "
                f"but found {compacted}."
            )
        existing_weather = collected_weather_counts(output)
        expected_per_weather = weather_quota
        if any(
            existing_weather[name] != expected_per_weather
            for name in WEATHER_PRESETS
        ):
            raise RuntimeError(
                "Existing test data does not cover all six weather presets "
                f"equally: {existing_weather}. Run again with --recollect."
            )
        print(
            f"[DONE] Existing {compacted} frames are stored in one seq_0000."
        )
        return 0

    map_name, zones = select_collection_zones()
    zone_counts = existing_zone_counts(output, zones)
    print(f"[START] CCSP multi-zone collection: {current}/{target_rgb_frames} RGB frames")
    print(f"[OUTPUT] {output}")
    print(f"[DISTRIBUTION] resumed zone counts: {zone_counts}")

    stalled_rounds = 0
    for round_index in range(1, max_collection_rounds + 1):
        if current >= target_rgb_frames:
            break
        round_start = current
        print(f"[ROUND {round_index}/{max_collection_rounds}] current={current}")

        zone_order = sorted(
            range(len(zones)),
            key=lambda index: (
                zone_counts[index] >= min_frames_per_zone,
                zone_counts[index],
                index,
            ),
        )
        print(
            "[ZONE-ORDER] "
            + ", ".join(
                f"{index + 1}:{zone_counts[index]}" for index in zone_order
            )
        )
        for zone_index in zone_order:
            zone = zones[zone_index]
            if current >= target_rgb_frames:
                break
            zone_count = zone_counts[zone_index]
            if zone_count >= max_frames_per_zone:
                continue

            reserved_for_other_zones = sum(
                max(0, min_frames_per_zone - count)
                for other_index, count in zone_counts.items()
                if other_index != zone_index
            )
            available_after_reserve = (
                target_rgb_frames - current - reserved_for_other_zones
            )
            requested = min(
                MAX_REQUEST_PER_BATCH,
                available_after_reserve,
            )
            requested = min(
                requested,
                max_frames_per_zone - zone_count,
                target_rgb_frames - current,
            )
            if requested <= 0:
                continue

            print(
                f"[ZONE {zone_index + 1:02d}] current={zone_count}, "
                f"request={requested}, center=({zone[0]:.2f}, {zone[1]:.2f})"
            )
            command_seed = 31000 + round_index * 100 + zone_index
            weather_before = collected_weather_counts(output)
            planned_weather = plan_weather_batch(
                weather_before,
                requested,
                command_seed,
            )
            print(
                f"[WEATHER] before={weather_before}, "
                f"planned={planned_weather}"
            )
            command = collector_command(
                output=output,
                requested=requested,
                preserve=current > 0,
                seed=command_seed,
                zone=zone,
                weather_presets=planned_weather,
            )
            paired_root = output / "paired_weather"
            before_sequences = {
                path.name for path in paired_root.glob("seq_*")
                if path.is_dir()
            }
            return_code = subprocess.call(command, cwd=str(COLLECT_DIR), env=env)

            after_sequences = {
                path.name for path in paired_root.glob("seq_*")
                if path.is_dir()
            }
            rejected_frames = quarantine_broken_visual_frames(
                output,
                sorted(after_sequences - before_sequences),
            )
            weather_overflow_frames = trim_weather_overflow(
                output,
                sorted(after_sequences - before_sequences),
                weather_quota,
            )

            new_count = rgb_frame_count(output)
            accepted = max(0, new_count - current)
            zone_counts[zone_index] += accepted
            print(
                f"[ZONE {zone_index + 1:02d}] return_code={return_code}, "
                f"accepted={accepted}, rejected_frames={rejected_frames}, "
                f"weather_overflow_frames={weather_overflow_frames}, "
                f"progress={current}->{new_count}"
            )
            current = new_count
            # One regional subprocess per round keeps at most seq_0000 plus
            # one temporary seq. The round-end QA then compacts them again.
            break

        rgb_rejected_frames = quarantine_unmatched_rgb_vehicles(output)
        current = rgb_frame_count(output)
        zone_counts = existing_zone_counts(output, zones)
        compact_sequences_to_one(output)
        ensure_all_weather_directories(output)
        current = rgb_frame_count(output)
        print(
            f"[ROUND-QA] rgb_asset_rejections={rgb_rejected_frames}, "
            f"accepted_total={current}"
        )

        if current <= round_start:
            stalled_rounds += 1
            print(
                f"[WARN] No net accepted frame in this round "
                f"({stalled_rounds}/5 consecutive stalls); trying a new "
                "zone and random seed without lowering quality thresholds."
            )
            if stalled_rounds >= 5:
                raise RuntimeError(
                    "Five consecutive rounds produced no accepted frame. "
                    "Keep the simulator on the intended CCSP map and inspect "
                    "the collector rejection statistics."
                )
        else:
            stalled_rounds = 0

    if current != target_rgb_frames:
        raise RuntimeError(
            f"Expected {target_rgb_frames} RGB frames, but collected {current}. "
            f"Per-zone counts: {zone_counts}"
        )

    print(f"[DONE] Map: {map_name}")
    current = compact_sequences_to_one(output)
    weather_counts = collected_weather_counts(output)
    expected_per_weather = weather_quota
    if any(
        weather_counts[name] != expected_per_weather
        for name in WEATHER_PRESETS
    ):
        raise RuntimeError(
            "Final weather distribution is not balanced across all six "
            f"presets: {weather_counts}"
        )
    print(f"[DONE] Collected exactly {current} accepted RGB frames in seq_0000.")
    print(f"[DONE] Per-zone distribution: {zone_counts}")
    print(f"[DONE] Weather distribution: {weather_counts}")
    print("[DONE] Every RGB frame has synchronized depth, segmentation and normal data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
