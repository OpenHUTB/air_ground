#!/usr/bin/env python
"""
OpenHUTB/CARLA 无人机跨相机多目标跟踪数据采集器。

正式 MCMOT 模型输入为 RGB。深度和语义相机数据作为遮挡、可见像素、
边界框和投影 QA 辅助数据保存。所有相机在同一次 world.tick()
中取帧。正式 Schema 使用 deterministic UUID5 的 global_object_uuid；
CARLA actor.id 仅用于当前 runtime 回溯和 MOT 派生格式。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import secrets
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

COLLECT_ROOT = Path(__file__).resolve().parent.parent
MULTIMODAL_DIR = COLLECT_ROOT / "multimodal"
SINGLE_CAMERA_DIR = COLLECT_ROOT / "single_camera_tracking"
for dependency_dir in (COLLECT_ROOT, MULTIMODAL_DIR, SINGLE_CAMERA_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import collect_rpg_small_targets_carla_v2 as base
import collect_uav_single_object_vot_carla as vot
from cooperative_perception import CooperativeManager, close_simulator, ensure_simulator
from cooperative_perception.cooperative_annotation import enrich_observations


carla = base.carla
TARGETS = vot.TARGETS
CLASS_NAMES = {0: "vehicle", 1: "pedestrian"}
DEFAULT_CONFIG = Path(__file__).with_name("multi_camera_mot_config.json")


@dataclass(frozen=True)
class SceneSpec:
    name: str
    weather: str
    split: str
    anchor_class: str
    scene_id: int
    weather_index: int
    index_in_weather: int


@dataclass
class CameraUnit:
    name: str
    sensors: Dict[str, Any]
    syncs: Dict[str, Any]
    transform: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="保留已完成场景，仅采集缺失场景并在末尾重新审计。",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--frames-per-scene",
        type=int,
        default=None,
        help="每个场景的最大保存帧数；满足无目标条件时会提前结束。",
    )
    parser.add_argument("--scenes-per-map", type=int, default=None)
    parser.add_argument("--weather-presets", nargs="+", default=None)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="不连接模拟器，仅重新审计现有数据集并重建 YOLO 目录。",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.out is not None:
        config["out"] = str(args.out)
    if args.frames_per_scene is not None:
        config["frames_per_scene"] = args.frames_per_scene
        config["train_frames_per_scene"] = args.frames_per_scene
        config["eval_frames_per_scene"] = args.frames_per_scene
    if args.scenes_per_map is not None:
        config["scenes_per_map"] = args.scenes_per_map
    if args.weather_presets is not None:
        config["weather_presets"] = args.weather_presets
    config["_config_path"] = str(config_path)
    return config


def resolve_run_seed(config: Dict[str, Any]) -> int:
    """Generate a fresh default seed while preserving explicit replay seeds."""
    configured = config.get("seed")
    if configured is None:
        configured = secrets.randbelow(2_147_483_646) + 1
        config["seed"] = configured
        config["seed_source"] = "generated_at_collector_start"
    else:
        config.setdefault("seed_source", "explicit")
    return int(configured)


def validate_config(config: Dict[str, Any]) -> None:
    positive = (
        "width",
        "height",
        "fov",
        "fps",
        "num_cameras",
        "sample_interval_ticks",
        "camera_layout_selection_attempts",
        "min_effective_track_seconds",
        "scenes_per_map",
        "frames_per_scene",
        "sensor_timeout",
        "max_scene_attempts",
    )
    for key in positive:
        if float(config[key]) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    for key in ("train_frames_per_scene", "eval_frames_per_scene"):
        if key in config and int(config[key]) <= 0:
            raise ValueError(f"{key} must be greater than 0")
    for key in ("vehicles", "walkers"):
        if int(config[key]) < 0:
            raise ValueError(f"{key} 不能小于 0")
    for key in ("max_active_vehicles", "max_active_pedestrians"):
        if int(config.get(key, 1)) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    if int(config["num_cameras"]) != 3:
        raise ValueError("Task 3 永久固定为 3 个虚拟 Air Camera Platform")
    if int(config["scenes_per_map"]) < 1:
        raise ValueError("每张地图至少需要 1 个场景")
    if float(config.get("sensor_tick", 0.0)) < 0.0:
        raise ValueError("sensor_tick 不能小于 0")
    minimum_separation = float(config["camera_bearing_separation_deg"])
    preferred_minimum = float(
        config["camera_preferred_bearing_separation_min_deg"]
    )
    preferred_maximum = float(
        config["camera_preferred_bearing_separation_max_deg"]
    )
    maximum_separation = float(config["camera_max_bearing_separation_deg"])
    if not (
        0.0
        < minimum_separation
        <= preferred_minimum
        <= preferred_maximum
        <= maximum_separation
        <= 180.0
    ):
        raise ValueError(
            "camera bearing separation must satisfy 0 < minimum <= "
            "preferred_minimum <= preferred_maximum <= maximum <= 180"
        )
    if float(config.get("camera_min_pair_distance_m", 0.0)) < 0.0:
        raise ValueError("camera_min_pair_distance_m 不能小于 0")
    ratio = float(config["min_visible_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("min_visible_ratio 必须在 (0, 1] 内")
    if not config["weather_presets"]:
        raise ValueError("weather_presets 不能为空")
    for key in (
        "min_valid_frame_ratio",
        "min_anchor_visible_any_camera_ratio",
        "vehicle_min_moving_frame_ratio",
        "pedestrian_min_moving_frame_ratio",
        "max_near_duplicate_pair_ratio",
        "max_dark_flat_region_ratio",
        "max_flat_region_ratio",
        "max_far_depth_region_ratio",
    ):
        value = float(config.get(key, 1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} 必须在 [0, 1] 内")
    if not 0.0 <= float(config["near_duplicate_ssim_threshold"]) <= 1.0:
        raise ValueError("near_duplicate_ssim_threshold 必须在 [0, 1] 内")
    if int(config["min_dynamic_event_count"]) < 0:
        raise ValueError("min_dynamic_event_count 不能小于 0")
    for key in (
        "vehicle_min_dynamic_event_count",
        "pedestrian_min_dynamic_event_count",
    ):
        if key in config and int(config[key]) < 0:
            raise ValueError(f"{key} 不能小于 0")
    if int(config.get("walker_activation_ticks", 10)) < 0:
        raise ValueError("walker_activation_ticks 不能小于 0")
    if float(config.get("pedestrian_activation_displacement_m", 0.05)) <= 0.0:
        raise ValueError("pedestrian_activation_displacement_m 必须大于 0")
    if float(config.get("pedestrian_moving_step_threshold_m", 0.02)) <= 0.0:
        raise ValueError("pedestrian_moving_step_threshold_m 必须大于 0")
    pedestrian_sample_hz = float(config.get("pedestrian_sample_hz", 5.0))
    if not 0.0 < pedestrian_sample_hz <= float(config["fps"]):
        raise ValueError("pedestrian_sample_hz 必须在 (0, fps] 内")
    if int(config.get("empty_camera_patience_frames", 5)) < 0:
        raise ValueError("empty_camera_patience_frames 不能小于 0")
    for key in (
        "unmodeled_check_width",
        "unmodeled_check_height",
        "unmodeled_check_tile_size",
    ):
        if int(config.get(key, 1)) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    if float(config.get("unmodeled_flat_tile_std_max", 3.0)) < 0.0:
        raise ValueError("unmodeled_flat_tile_std_max 不能小于 0")
    dark_luma = float(config.get("unmodeled_dark_luma_max", 100.0))
    if not 0.0 <= dark_luma <= 255.0:
        raise ValueError("unmodeled_dark_luma_max 必须在 [0, 255] 内")
    if float(config.get("unmodeled_far_depth_min_m", 500.0)) <= 0.0:
        raise ValueError("unmodeled_far_depth_min_m 必须大于 0")
    empty_camera_stop_count = int(config.get("empty_camera_stop_count", 2))
    if not 1 <= empty_camera_stop_count <= int(config["num_cameras"]):
        raise ValueError("empty_camera_stop_count 必须在 [1, num_cameras] 内")


def frames_for_scene(spec: SceneSpec, config: Dict[str, Any]) -> int:
    if spec.split == "train":
        return int(
            config.get("train_frames_per_scene", config["frames_per_scene"])
        )
    return int(config.get("eval_frames_per_scene", config["frames_per_scene"]))


def sample_interval_ticks_for_scene(
    spec: SceneSpec,
    config: Dict[str, Any],
) -> int:
    """Return the save interval; pedestrian scenes are sampled at 5 Hz."""
    if spec.anchor_class != "pedestrian":
        return int(config["sample_interval_ticks"])
    requested_hz = float(config.get("pedestrian_sample_hz", 5.0))
    return max(1, int(round(float(config["fps"]) / requested_hz)))


def sample_interval_seconds_for_scene(
    spec: SceneSpec,
    config: Dict[str, Any],
) -> float:
    return sample_interval_ticks_for_scene(spec, config) / float(config["fps"])


def resolve_output(config: Dict[str, Any]) -> Path:
    output = Path(config["out"])
    if not output.is_absolute():
        output = Path(config["_config_path"]).parent / output
    return output.resolve()


def prepare_output(
    root: Path,
    overwrite: bool,
    resume: bool = False,
) -> Dict[str, Path]:
    if root.exists():
        if not overwrite and not resume:
            raise FileExistsError(
                f"输出目录已存在：{root}\n"
                "请更换 --out，或明确使用 --overwrite/--resume。"
            )
        if overwrite:
            safe_name = str(root.parent / root.name).lower()
            allowed_legacy = (
                "dataset_uav" in safe_name and "multicamera_mot" in safe_name
            )
            allowed_suite = (
                "airgroundcoopsuite" in safe_name
                and ("task3_agc_mcmot" in safe_name or "derived_task3_mcmot" in safe_name)
            )
            if not (allowed_legacy or allowed_suite):
                raise RuntimeError(f"拒绝覆盖名称异常的目录：{root}")
            shutil.rmtree(root)
    paths = {
        "root": root,
        "scenes": root / "scenes",
        "qa": root / "qa_overlay",
        "splits": root / "splits",
        "yolo": root / "yolo",
        "staging": root / "_scene_staging",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if resume:
        for stale in list(paths["staging"].iterdir()):
            if stale.is_dir():
                safe_remove_staging(stale, paths["staging"])
            else:
                stale.unlink()
    return paths


def build_scene_specs(config: Dict[str, Any]) -> List[SceneSpec]:
    specs: List[SceneSpec] = []
    count = int(config["scenes_per_map"])
    weathers = list(config["weather_presets"])
    weather_occurrences: Counter = Counter()
    # A one-scene development run is a training sample.  The normal 30-scene
    # configuration still gives the intended exact 18/6/6 split.
    train_end = 1 if count == 1 else int(count * 0.60)
    val_end = train_end if count == 1 else train_end + int(count * 0.20)
    for scene_id in range(count):
        weather_index = scene_id % len(weathers)
        weather = weathers[weather_index]
        index_in_weather = int(weather_occurrences[weather])
        weather_occurrences[weather] += 1
        if count == 2:
            # The two-scene collection contract is explicit: one vehicle scene
            # followed by one pedestrian scene on every map.
            anchor_class = "vehicle" if scene_id == 0 else "pedestrian"
        else:
            anchor_class = (
                "vehicle"
                if (weather_index + index_in_weather) % 2 == 0
                else "pedestrian"
            )
        split = (
            "train"
            if scene_id < train_end
            else "val"
            if scene_id < val_end
            else "test"
        )
        specs.append(
            SceneSpec(
                name=(
                    f"{weather.lower()}_{anchor_class}_"
                    f"scene_{index_in_weather:02d}"
                ),
                weather=weather,
                split=split,
                anchor_class=anchor_class,
                scene_id=scene_id,
                weather_index=weather_index,
                index_in_weather=index_in_weather,
            )
        )
    return specs


def safe_remove_staging(path: Path, staging_root: Path) -> None:
    resolved = path.resolve()
    root = staging_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"拒绝删除 staging 目录以外的路径：{resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def accepted_scene_directory(
    path: Path,
    max_frames: Optional[int] = None,
) -> bool:
    quality_path = path / "scene_quality.json"
    if not quality_path.is_file():
        return False
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not bool(quality.get("passed", False)):
        return False
    if max_frames is None:
        return True
    meta_path = path / "scene_meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    frame_count = int(meta.get("frames", quality.get("frame_count", 0)))
    return 0 < frame_count <= int(max_frames)


def quarantine_rejected_scene(path: Path, root: Path) -> Path:
    quarantine_root = root / "rejected_scenes"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{path.name}_{int(time.time())}"
    suffix = 1
    while destination.exists():
        destination = quarantine_root / f"{path.name}_{int(time.time())}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(destination))
    return destination


def spawn_camera_units(
    world: Any,
    initial_transform: Any,
    config: Dict[str, Any],
) -> List[CameraUnit]:
    library = world.get_blueprint_library()
    sensor_types = {
        "rgb": "sensor.camera.rgb",
        "depth": "sensor.camera.depth",
        "semantic": "sensor.camera.semantic_segmentation",
    }
    units: List[CameraUnit] = []
    for camera_index in range(int(config["num_cameras"])):
        camera_name = f"cam_{camera_index:02d}"
        sensors: Dict[str, Any] = {}
        syncs: Dict[str, Any] = {}
        for modality, sensor_type in sensor_types.items():
            blueprint = base.setup_camera_blueprint(
                library,
                sensor_type,
                int(config["width"]),
                int(config["height"]),
                float(config["fov"]),
                float(config.get("sensor_tick", 0.0)),
                enable_rgb_postprocess=bool(config["enable_rgb_postprocess"]),
            )
            if modality == "rgb":
                for attribute in (
                    "motion_blur_intensity",
                    "motion_blur_max_distortion",
                    "motion_blur_min_object_screen_size",
                ):
                    if blueprint.has_attribute(attribute):
                        blueprint.set_attribute(attribute, "0.0")
            sensor = world.spawn_actor(blueprint, initial_transform)
            sensors[modality] = sensor
            syncs[modality] = base.SensorSync(
                f"{camera_name}_{modality}",
                sensor,
            )
        units.append(
            CameraUnit(
                name=camera_name,
                sensors=sensors,
                syncs=syncs,
                transform=initial_transform,
            )
        )
    return units


def destroy_actors(actors: Iterable[Any]) -> None:
    unique: Dict[int, Any] = {}
    for actor in actors:
        if actor is None:
            continue
        try:
            unique[int(actor.id)] = actor
        except (AttributeError, RuntimeError):
            continue
    live_ids: Optional[set] = None
    for actor in unique.values():
        try:
            if live_ids is None:
                live_ids = {
                    int(current.id)
                    for current in actor.get_world().get_actors()
                }
            if int(actor.id) not in live_ids:
                continue
            if hasattr(actor, "stop"):
                actor.stop()
            if actor.destroy():
                live_ids.discard(int(actor.id))
        except (AttributeError, RuntimeError):
            continue


def registered_actors(actors: Iterable[Any]) -> List[Any]:
    candidates = [actor for actor in actors if actor is not None]
    if not candidates:
        return []
    try:
        live_ids = {
            int(current.id)
            for current in candidates[0].get_world().get_actors()
        }
    except (AttributeError, RuntimeError):
        return [actor for actor in candidates if actor.is_alive]
    return [
        actor
        for actor in candidates
        if int(actor.id) in live_ids
    ]


def destroy_traffic_population(
    world: Any,
    controllers: Iterable[Any],
    walkers: Iterable[Any],
    vehicles: Iterable[Any],
) -> None:
    """Destroy parent walkers before their attached AI controllers.

    This OpenHUTB build removes a walker when its controller is destroyed.  The
    standard controller-first cleanup therefore leaves stale walker handles and
    floods the client log with ``actor not found`` messages.
    """
    live_controllers = registered_actors(controllers)
    for controller in live_controllers:
        try:
            controller.stop()
        except (AttributeError, RuntimeError):
            continue
    try:
        world.tick()
    except RuntimeError:
        pass
    destroy_actors(walkers)
    try:
        world.tick()
    except RuntimeError:
        pass
    destroy_actors(live_controllers)
    destroy_actors(vehicles)


def set_camera_unit_transform(unit: CameraUnit, transform: Any) -> None:
    for sensor in unit.sensors.values():
        sensor.set_transform(transform)
    unit.transform = transform


def drain_camera_units(units: Sequence[CameraUnit]) -> None:
    for unit in units:
        for sync in unit.syncs.values():
            sync.drain()


def tick_and_get(
    world: Any,
    units: Sequence[CameraUnit],
    timeout: float,
    before_tick=None,
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    if before_tick is not None:
        before_tick()
    frame = int(world.tick())
    data: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        data[unit.name] = {
            modality: sync.get(frame, timeout=timeout)
            for modality, sync in unit.syncs.items()
        }
    return frame, data


def live_target_actors(vehicles: Sequence[Any], walkers: Sequence[Any]) -> List[Any]:
    unique: Dict[int, Any] = {}
    for actor in registered_actors(list(vehicles) + list(walkers)):
        unique[int(actor.id)] = actor
    return list(unique.values())


def collection_target_actors(
    world: Any,
    vehicles: Sequence[Any],
    walkers: Sequence[Any],
    config: Dict[str, Any],
) -> List[Any]:
    """Return spawned actors plus optional actors kept alive by a map bridge."""
    all_vehicles = list(vehicles)
    all_walkers = list(walkers)
    if bool(config.get("include_existing_target_actors", False)):
        actors = world.get_actors()
        all_vehicles.extend(list(actors.filter("vehicle.*")))
        all_walkers.extend(list(actors.filter("walker.pedestrian.*")))
    live = live_target_actors(all_vehicles, all_walkers)
    vehicles_live = sorted(
        (
            actor
            for actor in live
            if actor.type_id.startswith("vehicle.")
        ),
        key=lambda actor: int(actor.id),
    )[: int(config.get("max_active_vehicles", len(live)))]
    pedestrians_live = sorted(
        (
            actor
            for actor in live
            if actor.type_id.startswith("walker.pedestrian.")
        ),
        key=lambda actor: int(actor.id),
    )[: int(config.get("max_active_pedestrians", len(live)))]
    return vehicles_live + pedestrians_live


def distance_2d(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def actor_speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def actor_instance_key(actor: Any, collection_seed: int) -> str:
    """Identify one spawned instance without depending on its appearance."""
    return f"{int(collection_seed)}:{int(actor.id)}"


def current_moving_anchor_actors(
    actors: Sequence[Any],
    anchor_class: Optional[str],
    config: Dict[str, Any],
) -> List[Any]:
    """Select currently moving vehicles; pedestrians use measured positions."""
    moving: List[Any] = []
    for actor in registered_actors(actors):
        is_pedestrian = actor.type_id.startswith("walker.pedestrian.")
        actor_class = "pedestrian" if is_pedestrian else "vehicle"
        if anchor_class is not None and actor_class != anchor_class:
            continue
        # OpenHUTB can report zero velocity for a walker whose transform is
        # changing.  The proven single-camera collector therefore evaluates
        # pedestrian motion from consecutive world positions instead.
        if actor_class == "pedestrian":
            continue
        threshold = float(config["vehicle_min_median_speed_mps"])
        try:
            if actor_speed_mps(actor) >= threshold:
                moving.append(actor)
        except RuntimeError:
            continue
    return moving


def pedestrian_positions(actors: Sequence[Any]) -> Dict[int, Tuple[float, float]]:
    """Snapshot live pedestrian XY positions without simulator velocity."""
    positions: Dict[int, Tuple[float, float]] = {}
    for actor in registered_actors(actors):
        if not actor.type_id.startswith("walker.pedestrian."):
            continue
        try:
            location = actor.get_location()
            positions[int(actor.id)] = (
                float(location.x),
                float(location.y),
            )
        except (AttributeError, RuntimeError):
            continue
    return positions


def pedestrians_moved_since(
    actors: Sequence[Any],
    starting_positions: Dict[int, Tuple[float, float]],
    config: Dict[str, Any],
) -> List[Any]:
    """Return walkers with real displacement during the activation window."""
    minimum = float(config.get("pedestrian_activation_displacement_m", 0.05))
    moved: List[Any] = []
    for actor in registered_actors(actors):
        if not actor.type_id.startswith("walker.pedestrian."):
            continue
        start = starting_positions.get(int(actor.id))
        if start is None:
            continue
        try:
            location = actor.get_location()
            displacement = math.hypot(
                float(location.x) - start[0],
                float(location.y) - start[1],
            )
        except (AttributeError, RuntimeError):
            continue
        if displacement >= minimum:
            moved.append(actor)
    return moved


def eligible_anchor_actors(
    actors: Sequence[Any],
    anchor_class: str,
    config: Dict[str, Any],
    eligible_pedestrian_ids: Optional[Sequence[int]] = None,
) -> List[Any]:
    if anchor_class != "pedestrian":
        return current_moving_anchor_actors(actors, anchor_class, config)
    eligible = set(map(int, eligible_pedestrian_ids or []))
    return [
        actor
        for actor in registered_actors(actors)
        if actor.type_id.startswith("walker.pedestrian.")
        and int(actor.id) in eligible
    ]


def configure_vehicle_motion(
    client: Any,
    vehicles: Sequence[Any],
    tm_port: int,
    seed: int,
    config: Dict[str, Any],
) -> None:
    """Apply a reproducible but diverse Traffic Manager motion profile."""
    traffic_manager = client.get_trafficmanager(tm_port)
    rng = random.Random(seed + 7919)
    speed_min = float(config.get("traffic_speed_difference_min_pct", -20.0))
    speed_max = float(config.get("traffic_speed_difference_max_pct", 10.0))
    gap_min = float(config.get("traffic_follow_distance_min_m", 1.5))
    gap_max = float(config.get("traffic_follow_distance_max_m", 4.0))
    lane_change = float(config.get("traffic_lane_change_probability_pct", 35.0))
    for actor in vehicles:
        if actor is None or not actor.is_alive:
            continue
        try:
            actor.set_autopilot(True, tm_port)
            traffic_manager.vehicle_percentage_speed_difference(
                actor,
                rng.uniform(speed_min, speed_max),
            )
            traffic_manager.distance_to_leading_vehicle(
                actor,
                rng.uniform(gap_min, gap_max),
            )
            traffic_manager.auto_lane_change(actor, True)
            traffic_manager.random_left_lanechange_percentage(actor, lane_change)
            traffic_manager.random_right_lanechange_percentage(actor, lane_change)
        except (AttributeError, RuntimeError):
            # OpenHUTB packages expose slightly different Traffic Manager APIs.
            # Autopilot is already enabled by the shared spawner, so an optional
            # tuning method being unavailable must not stop collection.
            continue


def refresh_walker_motion(
    world: Any,
    controllers: Sequence[Any],
    seed: int,
    config: Dict[str, Any],
) -> None:
    """Refresh AI destinations while preserving the original controllers."""
    rng = random.Random(seed + 1543)
    speed_min = float(config.get("walker_speed_min_mps", 0.9))
    speed_max = float(config.get("walker_speed_max_mps", 1.8))
    for controller in registered_actors(controllers):
        try:
            destination = None
            for _ in range(100):
                destination = world.get_random_location_from_navigation()
                if destination is not None:
                    break
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(rng.uniform(speed_min, speed_max))
        except (AttributeError, RuntimeError):
            continue


def choose_anchor_actor(
    actors: Sequence[Any],
    config: Dict[str, Any],
    rng: random.Random,
    preferred_class: Optional[str] = None,
    excluded_actor_ids: Optional[Sequence[int]] = None,
) -> Optional[Any]:
    excluded = set(map(int, excluded_actor_ids or []))
    alive = [
        actor
        for actor in actors
        if actor is not None
        and actor.is_alive
        and int(actor.id) not in excluded
    ]
    if preferred_class == "vehicle":
        preferred = [
            actor for actor in alive if actor.type_id.startswith("vehicle.")
        ]
        if preferred:
            alive = preferred
    elif preferred_class == "pedestrian":
        preferred = [
            actor
            for actor in alive
            if actor.type_id.startswith("walker.pedestrian.")
        ]
        if preferred:
            alive = preferred
    if not alive:
        return None
    radius = float(config["anchor_neighbour_radius_m"])
    scored: List[Tuple[int, Any]] = []
    locations: Dict[int, Any] = {}
    for actor in alive:
        try:
            locations[int(actor.id)] = actor.get_location()
        except RuntimeError:
            continue
    for actor in alive:
        location = locations.get(int(actor.id))
        if location is None:
            continue
        score = sum(
            1
            for other_location in locations.values()
            if distance_2d(location, other_location) <= radius
        )
        scored.append((score, actor))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(1, int(config["anchor_top_k"]))]
    weights = [max(1, score) ** 2 for score, _ in top]
    return rng.choices([actor for _, actor in top], weights=weights, k=1)[0]


def angle_difference_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def camera_rig_geometry(transforms: Sequence[Any]) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    for first_index, first in enumerate(transforms):
        for second_index in range(first_index + 1, len(transforms)):
            second = transforms[second_index]
            pairs.append(
                {
                    "cameras": [
                        f"cam_{first_index:02d}",
                        f"cam_{second_index:02d}",
                    ],
                    "horizontal_distance_m": distance_2d(
                        first.location,
                        second.location,
                    ),
                    "view_yaw_separation_deg": angle_difference_degrees(
                        float(first.rotation.yaw),
                        float(second.rotation.yaw),
                    ),
                }
            )
    return {
        "pairs": pairs,
        "minimum_horizontal_distance_m": (
            min(pair["horizontal_distance_m"] for pair in pairs)
            if pairs
            else None
        ),
        "minimum_view_yaw_separation_deg": (
            min(pair["view_yaw_separation_deg"] for pair in pairs)
            if pairs
            else None
        ),
        "maximum_view_yaw_separation_deg": (
            max(pair["view_yaw_separation_deg"] for pair in pairs)
            if pairs
            else None
        ),
    }


def look_at_transform(
    ground_location: Any,
    aim_location: Any,
    altitude: float,
) -> Any:
    camera_location = carla.Location(
        x=float(ground_location.x),
        y=float(ground_location.y),
        z=float(ground_location.z) + altitude,
    )
    dx = float(aim_location.x) - float(camera_location.x)
    dy = float(aim_location.y) - float(camera_location.y)
    dz = float(aim_location.z) - float(camera_location.z)
    horizontal = max(1e-6, math.hypot(dx, dy))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horizontal))
    return carla.Transform(
        camera_location,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )


def choose_camera_transforms(
    anchor: Any,
    road_waypoints: Sequence[Any],
    config: Dict[str, Any],
    rng: random.Random,
) -> Optional[Tuple[List[Any], float]]:
    try:
        actor_location = anchor.get_location()
        actor_velocity = anchor.get_velocity()
        actor_box_height = float(anchor.bounding_box.extent.z) * 0.5
    except RuntimeError:
        return None
    lookahead = float(config.get("camera_route_lookahead_seconds", 0.0))
    corridor_center = carla.Location(
        x=float(actor_location.x) + float(actor_velocity.x) * lookahead * 0.5,
        y=float(actor_location.y) + float(actor_velocity.y) * lookahead * 0.5,
        z=float(actor_location.z),
    )
    aim = carla.Location(
        x=float(corridor_center.x),
        y=float(corridor_center.y),
        z=float(actor_location.z) + max(0.5, actor_box_height),
    )
    anchor_class = (
        "pedestrian"
        if anchor.type_id.startswith("walker.pedestrian.")
        else "vehicle"
    )
    minimum_radius = float(
        config.get(
            f"{anchor_class}_camera_radius_min_m",
            config["camera_radius_min_m"],
        )
    )
    maximum_radius = float(
        config.get(
            f"{anchor_class}_camera_radius_max_m",
            config["camera_radius_max_m"],
        )
    )
    minimum_height = float(
        config.get(
            f"{anchor_class}_camera_height_min_m",
            config["camera_height_min_m"],
        )
    )
    maximum_height = float(
        config.get(
            f"{anchor_class}_camera_height_max_m",
            config["camera_height_max_m"],
        )
    )
    minimum_pitch = float(
        config.get(
            f"{anchor_class}_camera_pitch_min_deg",
            config["camera_pitch_min_deg"],
        )
    )
    maximum_pitch = float(
        config.get(
            f"{anchor_class}_camera_pitch_max_deg",
            config["camera_pitch_max_deg"],
        )
    )
    candidates: List[Tuple[float, Any]] = []
    shuffled_waypoints = list(road_waypoints)
    rng.shuffle(shuffled_waypoints)
    for waypoint in shuffled_waypoints:
        location = waypoint.transform.location
        radius = distance_2d(location, corridor_center)
        if not minimum_radius <= radius <= maximum_radius:
            continue
        altitude = rng.uniform(minimum_height, maximum_height)
        transform = look_at_transform(location, aim, altitude)
        pitch = float(transform.rotation.pitch)
        if not minimum_pitch <= pitch <= maximum_pitch:
            continue
        bearing = (
            math.degrees(
                math.atan2(
                    float(location.y) - float(corridor_center.y),
                    float(location.x) - float(corridor_center.x),
                )
            )
            % 360.0
        )
        candidates.append((bearing, transform))
        if len(candidates) >= 160:
            break
    required = int(config["num_cameras"])
    if len(candidates) < required:
        return None
    rng.shuffle(candidates)
    minimum_separation = float(config["camera_bearing_separation_deg"])
    preferred_minimum = float(
        config["camera_preferred_bearing_separation_min_deg"]
    )
    preferred_maximum = float(
        config["camera_preferred_bearing_separation_max_deg"]
    )
    maximum_separation = float(config["camera_max_bearing_separation_deg"])
    minimum_pair_distance = float(config["camera_min_pair_distance_m"])
    selected: Optional[List[Tuple[float, Any]]] = None
    layout_attempts = int(config.get("camera_layout_selection_attempts", 48))
    for _ in range(layout_attempts):
        pitch_targets = [
            rng.uniform(minimum_pitch, maximum_pitch)
            for _ in range(required)
        ]
        first_ranked = sorted(
            candidates,
            key=lambda candidate: abs(
                float(candidate[1].rotation.pitch) - pitch_targets[0]
            ),
        )
        proposal: List[Tuple[float, Any]] = [
            rng.choice(first_ranked[: min(12, len(first_ranked))])
        ]
        while len(proposal) < required:
            valid = [
                candidate
                for candidate in candidates
                if candidate not in proposal
                and all(
                    (
                        minimum_separation
                        <= angle_difference_degrees(candidate[0], used[0])
                        <= maximum_separation
                        and distance_2d(
                            candidate[1].location,
                            used[1].location,
                        )
                        >= minimum_pair_distance
                    )
                    for used in proposal
                )
            ]
            if not valid:
                break
            target_pitch = pitch_targets[len(proposal)]
            target_separation = rng.uniform(
                preferred_minimum,
                preferred_maximum,
            )
            ranked = sorted(
                valid,
                key=lambda candidate: abs(
                    min(
                        angle_difference_degrees(candidate[0], used[0])
                        for used in proposal
                    )
                    - target_separation
                )
                + 0.75
                * abs(float(candidate[1].rotation.pitch) - target_pitch),
            )
            proposal.append(rng.choice(ranked[: min(8, len(ranked))]))
        if len(proposal) == required:
            selected = proposal
            break
    if selected is None:
        return None
    try:
        waypoint = anchor.get_world().get_map().get_waypoint(
            actor_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        ground_z = float(waypoint.transform.location.z)
    except Exception:
        ground_z = float(actor_location.z)
    return [transform for _, transform in selected], ground_z


def build_annotations(
    world: Any,
    transform: Any,
    depth_m: np.ndarray,
    semantic_id: np.ndarray,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    annotations = vot.build_actor_annotations(
        world,
        transform,
        depth_m,
        semantic_id,
        config,
    )
    filtered: List[Dict[str, Any]] = []
    for annotation in annotations:
        class_name = str(annotation["class_name"]).lower()
        minimum = float(
            config[
                "min_vehicle_equivalent_side_px"
                if class_name == "vehicle"
                else "min_pedestrian_equivalent_side_px"
            ]
        )
        if vot.annotation_equivalent_side(annotation) < minimum:
            continue
        annotation["global_id"] = int(annotation["carla_actor_id"])
        annotation["visibility"] = float(
            annotation["visible_ratio_projected_bbox"]
        )
        annotation["occlusion"] = float(
            1.0 - annotation["visible_ratio_projected_bbox"]
        )
        filtered.append(annotation)
    filtered.sort(key=lambda item: (int(item["class_id"]), int(item["global_id"])))
    for annotation_id, annotation in enumerate(filtered):
        annotation["id"] = annotation_id
    return filtered


def enrich_world_state(
    annotations: Sequence[Dict[str, Any]],
    actor_by_id: Dict[int, Any],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for source in annotations:
        annotation = dict(source)
        actor = actor_by_id.get(int(annotation["global_id"]))
        if actor is not None and actor.is_alive:
            try:
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                annotation["world_location"] = {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                }
                annotation["world_rotation"] = {
                    "pitch": float(transform.rotation.pitch),
                    "yaw": float(transform.rotation.yaw),
                    "roll": float(transform.rotation.roll),
                }
                annotation["world_velocity_mps"] = {
                    "x": float(velocity.x),
                    "y": float(velocity.y),
                    "z": float(velocity.z),
                }
            except RuntimeError:
                pass
        enriched.append(annotation)
    return enriched


def rgb_bgr_from_image(image: Any) -> np.ndarray:
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height,
        image.width,
        4,
    )
    return bgra[:, :, :3].copy()


def save_overlay(
    rgb_bgr: np.ndarray,
    annotations: Sequence[Dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    canvas = rgb_bgr.copy()
    for annotation in annotations:
        x, y, width, height = map(int, annotation["bbox_xywh"])
        color = (
            (0, 255, 0)
            if int(annotation["class_id"]) == 0
            else (255, 80, 40)
        )
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            canvas,
            (
                f"{annotation['class_name']} gid={annotation['global_id']} "
                f"vis={annotation['visibility']:.2f}"
            ),
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        title,
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def intrinsic_matrix(width: int, height: int, fov: float) -> List[List[float]]:
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return [
        [float(focal), 0.0, float(width) / 2.0],
        [0.0, float(focal), float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ]


def calibration_dict(
    camera_name: str,
    transform: Any,
    config: Dict[str, Any],
    ground_z: float,
) -> Dict[str, Any]:
    return {
        "camera_id": camera_name,
        "image_width": int(config["width"]),
        "image_height": int(config["height"]),
        "horizontal_fov_deg": float(config["fov"]),
        "K": intrinsic_matrix(
            int(config["width"]),
            int(config["height"]),
            float(config["fov"]),
        ),
        "camera_to_world_carla": np.asarray(
            transform.get_matrix(),
            dtype=np.float64,
        ).tolist(),
        "world_to_camera_carla": np.asarray(
            transform.get_inverse_matrix(),
            dtype=np.float64,
        ).tolist(),
        "transform": base.transform_to_dict(transform),
        "ground_plane_z_m": float(ground_z),
        "coordinate_note": (
            "CARLA camera local axes: +x forward, +y right, +z up. "
            "Image axes: u right, v down."
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_mot_line(
    path: Path,
    frame_index: int,
    annotation: Dict[str, Any],
) -> None:
    x, y, width, height = annotation["bbox_xywh"]
    line = (
        f"{frame_index + 1},{int(annotation['global_id'])},"
        f"{float(x):.2f},{float(y):.2f},"
        f"{float(width):.2f},{float(height):.2f},"
        f"1,{int(annotation['class_id']) + 1},"
        f"{float(annotation['visibility']):.6f},-1\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line)


def dataset_global_id(scene_id: int, carla_actor_id: int) -> int:
    if not 0 <= int(carla_actor_id) < 100000:
        raise ValueError(
            f"carla_actor_id 超出场景命名空间范围：{carla_actor_id}"
        )
    return (int(scene_id) + 1) * 100000 + int(carla_actor_id)


def frame_quality(
    camera_payloads: Dict[str, Dict[str, Any]],
    anchor_class: str,
    config: Dict[str, Any],
) -> Tuple[bool, List[int], Dict[int, int]]:
    counts = {
        int(annotation["global_id"]): 0
        for payload in camera_payloads.values()
        for annotation in payload["annotations"]
    }
    for global_id in list(counts):
        counts[global_id] = sum(
            any(
                int(annotation["global_id"]) == global_id
                for annotation in payload["annotations"]
            )
            for payload in camera_payloads.values()
        )
    common = sorted(global_id for global_id, count in counts.items() if count >= 2)
    enough_objects = all(
        len(payload["annotations"]) >= int(config["min_objects_per_camera"])
        for payload in camera_payloads.values()
    )
    anchor_class_id = next(
        class_id
        for class_id, class_name in CLASS_NAMES.items()
        if class_name == anchor_class
    )
    minimum_anchor_objects = int(config.get("min_anchor_class_per_camera", 0))
    enough_anchor_objects = all(
        sum(
            int(annotation["class_id"]) == anchor_class_id
            for annotation in payload["annotations"]
        )
        >= minimum_anchor_objects
        for payload in camera_payloads.values()
    )
    valid = (
        enough_objects
        and enough_anchor_objects
        and len(common) >= int(config["min_common_ids_per_frame"])
    )
    return valid, common, counts


def largest_connected_region_ratio(mask: np.ndarray) -> float:
    """Return the largest 8-connected true component as a mask fraction."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.size == 0 or not np.any(binary):
        return 0.0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return 0.0
    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return float(largest / binary.size)


def unmodeled_region_metrics(
    rgb_bgr: np.ndarray,
    config: Dict[str, Any],
    depth_m: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Detect large map-edge voids on a small grid of texture statistics."""
    if not bool(config.get("reject_unmodeled_regions", False)):
        return {
            "enabled": False,
            "passed": True,
            "dark_flat_component_ratio": 0.0,
            "flat_component_ratio": 0.0,
            "far_depth_ratio": 0.0,
        }
    width = int(config.get("unmodeled_check_width", 160))
    height = int(config.get("unmodeled_check_height", 90))
    tile = int(config.get("unmodeled_check_tile_size", 10))
    gray = cv2.cvtColor(
        cv2.resize(rgb_bgr, (width, height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    rows = height // tile
    columns = width // tile
    if rows <= 0 or columns <= 0:
        raise ValueError("未建模区域检测尺寸必须不小于 tile size")
    gray = gray[: rows * tile, : columns * tile]
    means = np.empty((rows, columns), dtype=np.float32)
    deviations = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            patch = gray[
                row * tile : (row + 1) * tile,
                column * tile : (column + 1) * tile,
            ]
            means[row, column] = float(np.mean(patch))
            deviations[row, column] = float(np.std(patch))
    flat_limit = float(config.get("unmodeled_flat_tile_std_max", 3.0))
    dark_limit = float(config.get("unmodeled_dark_luma_max", 100.0))
    flat_mask = deviations <= flat_limit
    dark_flat_mask = flat_mask & (means <= dark_limit)
    flat_ratio = largest_connected_region_ratio(flat_mask)
    dark_flat_ratio = largest_connected_region_ratio(dark_flat_mask)
    maximum_flat = float(config.get("max_flat_region_ratio", 0.24))
    maximum_dark_flat = float(
        config.get("max_dark_flat_region_ratio", 0.05)
    )
    far_depth_ratio = 0.0
    if depth_m is not None:
        far_depth_minimum = float(
            config.get("unmodeled_far_depth_min_m", 500.0)
        )
        far_depth_ratio = float(
            np.mean(
                ~np.isfinite(depth_m)
                | (depth_m <= 0.0)
                | (depth_m >= far_depth_minimum)
            )
        )
    maximum_far_depth = float(
        config.get("max_far_depth_region_ratio", 0.10)
    )
    return {
        "enabled": True,
        "passed": (
            flat_ratio <= maximum_flat
            and dark_flat_ratio <= maximum_dark_flat
            and far_depth_ratio <= maximum_far_depth
        ),
        "dark_flat_component_ratio": dark_flat_ratio,
        "flat_component_ratio": flat_ratio,
        "far_depth_ratio": far_depth_ratio,
        "max_dark_flat_component_ratio": maximum_dark_flat,
        "max_flat_component_ratio": maximum_flat,
        "max_far_depth_region_ratio": maximum_far_depth,
    }


def inspect_camera_frame(
    world: Any,
    unit: CameraUnit,
    sensor_data: Dict[str, Any],
    actor_by_id: Dict[int, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    rgb_bgr = rgb_bgr_from_image(sensor_data["rgb"])
    depth_m = base.decode_carla_depth_meters(sensor_data["depth"])
    semantic_id = base.decode_semantic_segmentation(sensor_data["semantic"])
    annotations = enrich_world_state(
        build_annotations(
            world,
            unit.transform,
            depth_m,
            semantic_id,
            config,
        ),
        actor_by_id,
    )
    road_ratio = float(
        base.road_visible_ratio(
            sensor_data["semantic"],
            list(map(int, config["road_semantic_ids"])),
        )
    )
    near_ratio = float(
        np.mean(
            np.isfinite(depth_m)
            & (depth_m > 0.0)
            & (depth_m < float(config["min_near_depth_m"]))
        )
    )
    unmodeled = unmodeled_region_metrics(rgb_bgr, config, depth_m)
    view_valid = (
        road_ratio >= float(config["min_road_visible_ratio"])
        and near_ratio <= float(config["max_near_depth_ratio"])
        and bool(unmodeled["passed"])
    )
    boundary_count = 0
    if bool(config.get("reject_boundary_annotations", False)):
        margin = float(config.get("annotation_boundary_margin_px", 0.0))
        image_width = float(config["width"])
        image_height = float(config["height"])
        def touches_boundary(annotation: Dict[str, Any]) -> bool:
            return bool(
                float(annotation["bbox_xywh"][0]) <= margin
                or float(annotation["bbox_xywh"][1]) <= margin
                or float(annotation["bbox_xywh"][0])
                + float(annotation["bbox_xywh"][2])
                >= image_width - margin
                or float(annotation["bbox_xywh"][1])
                + float(annotation["bbox_xywh"][3])
                >= image_height - margin
            )
        boundary_count = sum(touches_boundary(annotation) for annotation in annotations)
        # Truncated boxes are omitted from this camera instead of invalidating
        # the entire synchronized time step.  Sequence-level visibility gates
        # still ensure that the anchor remains useful across cameras.
        annotations = [
            annotation
            for annotation in annotations
            if not touches_boundary(annotation)
        ]
    return {
        "rgb_bgr": rgb_bgr,
        "annotations": annotations,
        "road_visible_ratio": road_ratio,
        "near_depth_ratio": near_ratio,
        "unmodeled_region": unmodeled,
        "view_valid": view_valid,
        "boundary_annotation_count": boundary_count,
    }


def write_frame(
    scene_dir: Path,
    spec: SceneSpec,
    frame_index: int,
    carla_frame: int,
    camera_payloads: Dict[str, Dict[str, Any]],
    common_ids: Sequence[int],
    qa_root: Path,
    qa_indices: Sequence[int],
    config: Dict[str, Any],
) -> None:
    saved_common_ids = [
        dataset_global_id(spec.scene_id, int(global_id))
        for global_id in common_ids
    ]
    for camera_name, payload in camera_payloads.items():
        saved_annotations: List[Dict[str, Any]] = []
        for source in payload["annotations"]:
            annotation = dict(source)
            raw_actor_id = int(annotation["carla_actor_id"])
            annotation["global_id"] = dataset_global_id(
                spec.scene_id,
                raw_actor_id,
            )
            annotation["track_id"] = (
                f"{spec.name}_{annotation['class_name']}_"
                f"{annotation['global_id']}"
            )
            saved_annotations.append(annotation)
        camera_dir = scene_dir / "cameras" / camera_name
        image_path = camera_dir / "rgb" / f"{frame_index:06d}.png"
        label_path = camera_dir / "labels_yolo" / f"{frame_index:06d}.txt"
        annotation_path = (
            camera_dir / "annotations" / f"{frame_index:06d}.json"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(image_path),
            payload["rgb_bgr"],
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        base.save_yolo_label(
            label_path,
            saved_annotations,
            int(config["width"]),
            int(config["height"]),
        )
        write_json(
            annotation_path,
            {
                "scene": spec.name,
                "split": spec.split,
                "weather": spec.weather,
                "camera_id": camera_name,
                "dataset_frame": frame_index,
                "carla_frame": carla_frame,
                "image": str(image_path.relative_to(scene_dir)),
                "road_visible_ratio": payload["road_visible_ratio"],
                "near_depth_ratio": payload["near_depth_ratio"],
                "unmodeled_region": payload["unmodeled_region"],
                "common_global_ids": saved_common_ids,
                "annotations": saved_annotations,
            },
        )
        mot_path = camera_dir / "gt" / "gt.txt"
        for annotation in saved_annotations:
            write_mot_line(mot_path, frame_index, annotation)
        if frame_index in qa_indices:
            save_overlay(
                payload["rgb_bgr"],
                saved_annotations,
                qa_root
                / spec.name
                / camera_name
                / f"{frame_index:06d}_overlay.jpg",
                (
                    f"{spec.name} {camera_name} frame={frame_index} "
                    f"common={len(common_ids)}"
                ),
            )


def write_world_tracks(
    scene_dir: Path,
    spec: SceneSpec,
    frame_index: int,
    carla_frame: int,
    actor_by_id: Dict[int, Any],
    observed_ids: Sequence[int],
) -> None:
    payload = {
        "dataset_frame": frame_index,
        "carla_frame": carla_frame,
        "actors": [],
    }
    for global_id in sorted(set(map(int, observed_ids))):
        actor = actor_by_id.get(global_id)
        if actor is None or not actor.is_alive:
            continue
        try:
            transform = actor.get_transform()
            velocity = actor.get_velocity()
        except RuntimeError:
            continue
        payload["actors"].append(
            {
                "global_id": global_id,
                "carla_actor_id": global_id,
                "class_id": (
                    0 if actor.type_id.startswith("vehicle.") else 1
                ),
                "actor_type_id": actor.type_id,
                "location": {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                },
                "rotation": {
                    "pitch": float(transform.rotation.pitch),
                    "yaw": float(transform.rotation.yaw),
                    "roll": float(transform.rotation.roll),
                },
                "velocity_mps": {
                    "x": float(velocity.x),
                    "y": float(velocity.y),
                    "z": float(velocity.z),
                },
            }
        )
        payload["actors"][-1]["global_id"] = dataset_global_id(
            spec.scene_id,
            global_id,
        )
    append_jsonl(scene_dir / "global_tracks.jsonl", payload)


def ssim_gray(first_bgr: np.ndarray, second_bgr: np.ndarray, width: int) -> float:
    """Compute a lightweight luminance SSIM for online near-duplicate checks."""
    def prepare(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if width > 0 and gray.shape[1] > width:
            height = max(1, int(round(gray.shape[0] * width / gray.shape[1])))
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32)

    first = prepare(first_bgr)
    second = prepare(second_bgr)
    c1 = 6.5025
    c2 = 58.5225
    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_product = mu_first * mu_second
    sigma_first_sq = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first_sq
    sigma_second_sq = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second_sq
    sigma_product = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_product
    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_product + c2)
    denominator = (mu_first_sq + mu_second_sq + c1) * (
        sigma_first_sq + sigma_second_sq + c2
    )
    score = float(np.mean(numerator / np.maximum(denominator, 1e-12)))
    return max(-1.0, min(1.0, score))


def longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def sequence_quality_report(
    spec: SceneSpec,
    required_global_id: int,
    frame_validity: Sequence[bool],
    anchor_camera_counts: Sequence[int],
    track_world_samples: Dict[int, List[Tuple[int, float, float, float, float]]],
    track_visible_frames: Dict[int, set],
    ssim_by_camera: Dict[str, List[float]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    frame_count = len(frame_validity)
    sample_seconds = sample_interval_seconds_for_scene(spec, config)
    valid_ratio = float(sum(frame_validity) / frame_count) if frame_count else 0.0
    visible_flags = [count >= 1 for count in anchor_camera_counts]
    common_flags = [count >= 2 for count in anchor_camera_counts]
    visible_ratio = float(sum(visible_flags) / frame_count) if frame_count else 0.0
    common_ratio = float(sum(common_flags) / frame_count) if frame_count else 0.0
    longest_common_frames = longest_true_run(common_flags)
    longest_common_seconds = longest_common_frames * sample_seconds

    reappearances = 0
    seen = False
    in_gap = False
    for visible in visible_flags:
        if visible:
            if seen and in_gap:
                reappearances += 1
            seen = True
            in_gap = False
        elif seen:
            in_gap = True

    samples = track_world_samples.get(int(required_global_id), [])
    path_length = 0.0
    for previous, current in zip(samples, samples[1:]):
        path_length += math.sqrt(
            (current[1] - previous[1]) ** 2
            + (current[2] - previous[2]) ** 2
            + (current[3] - previous[3]) ** 2
        )
    displacement = 0.0
    if len(samples) >= 2:
        displacement = math.sqrt(
            (samples[-1][1] - samples[0][1]) ** 2
            + (samples[-1][2] - samples[0][2]) ** 2
            + (samples[-1][3] - samples[0][3]) ** 2
        )
    step_distances = [
        math.sqrt(
            (current[1] - previous[1]) ** 2
            + (current[2] - previous[2]) ** 2
            + (current[3] - previous[3]) ** 2
        )
        for previous, current in zip(samples, samples[1:])
    ]
    speeds = [sample[4] for sample in samples]
    median_speed = float(np.median(speeds)) if speeds else 0.0
    prefix = "pedestrian" if spec.anchor_class == "pedestrian" else "vehicle"
    if prefix == "pedestrian":
        # Match the single-camera collector: walker movement is derived from
        # consecutive transforms because OpenHUTB may report zero velocity.
        speed_threshold = 0.0
        moving_step_threshold = float(
            config.get("pedestrian_moving_step_threshold_m", 0.02)
        )
        moving_ratio = (
            float(
                sum(step >= moving_step_threshold for step in step_distances)
                / len(step_distances)
            )
            if step_distances
            else 0.0
        )
    else:
        speed_threshold = float(config["vehicle_min_median_speed_mps"])
        moving_step_threshold = None
        moving_ratio = (
            float(sum(speed >= speed_threshold for speed in speeds) / len(speeds))
            if speeds
            else 0.0
        )

    headings: List[float] = []
    for previous, current in zip(samples, samples[1:]):
        dx = current[1] - previous[1]
        dy = current[2] - previous[2]
        if math.hypot(dx, dy) >= 0.2:
            headings.append(math.degrees(math.atan2(dy, dx)))
    heading_change = 0.0
    if len(headings) >= 2:
        heading_change = max(
            angle_difference_degrees(first, second)
            for first in headings
            for second in headings
        )

    anchor_by_frame = {sample[0]: sample for sample in samples}
    minimum_other_distance = math.inf
    overtake_detected = False
    if len(samples) >= 2:
        route_dx = samples[-1][1] - samples[0][1]
        route_dy = samples[-1][2] - samples[0][2]
        route_norm = math.hypot(route_dx, route_dy)
        if route_norm > 1e-6:
            route_dx /= route_norm
            route_dy /= route_norm
        for actor_id, other_samples in track_world_samples.items():
            if int(actor_id) == int(required_global_id):
                continue
            longitudinal: List[float] = []
            pair_minimum = math.inf
            for other in other_samples:
                anchor_sample = anchor_by_frame.get(other[0])
                if anchor_sample is None:
                    continue
                relative_x = other[1] - anchor_sample[1]
                relative_y = other[2] - anchor_sample[2]
                pair_distance = math.hypot(relative_x, relative_y)
                pair_minimum = min(pair_minimum, pair_distance)
                minimum_other_distance = min(minimum_other_distance, pair_distance)
                if route_norm > 1e-6:
                    longitudinal.append(
                        relative_x * route_dx + relative_y * route_dy
                    )
            if (
                longitudinal
                and min(longitudinal) <= -2.0
                and max(longitudinal) >= 2.0
                and pair_minimum <= float(config["overtake_distance_m"])
            ):
                overtake_detected = True

    events = {
        "close_interaction": minimum_other_distance
        <= float(config["interaction_distance_m"]),
        "turn": heading_change
        >= float(config["turn_event_min_heading_change_deg"]),
        "overtake_or_crossing": overtake_detected,
        "occlusion_reappearance": reappearances > 0,
    }
    dynamic_event_count = sum(bool(value) for value in events.values())
    minimum_dynamic_event_count = int(
        config.get(
            f"{spec.anchor_class}_min_dynamic_event_count",
            config["min_dynamic_event_count"],
        )
    )

    minimum_track_seconds = float(config["min_effective_track_seconds"])
    minimum_track_frames = max(
        1,
        int(math.ceil(minimum_track_seconds / sample_seconds)),
    )
    effective_tracks = {
        dataset_global_id(spec.scene_id, actor_id): len(frames)
        for actor_id, frames in track_visible_frames.items()
        if len(frames) >= minimum_track_frames
    }
    all_ssim = [score for values in ssim_by_camera.values() for score in values]
    near_duplicate_threshold = float(config["near_duplicate_ssim_threshold"])
    near_duplicate_ratio = (
        float(sum(score >= near_duplicate_threshold for score in all_ssim) / len(all_ssim))
        if all_ssim
        else 0.0
    )

    displacement_threshold = float(
        config[f"{prefix}_min_sequence_displacement_m"]
    )
    path_length_threshold = float(
        config[f"{prefix}_min_sequence_path_length_m"]
    )
    moving_ratio_threshold = float(
        config[f"{prefix}_min_moving_frame_ratio"]
    )
    spatial_motion_passed = (
        displacement >= displacement_threshold
        and path_length >= path_length_threshold
    )
    temporal_motion_passed = (
        median_speed >= speed_threshold
        and moving_ratio >= moving_ratio_threshold
    )
    if prefix == "pedestrian":
        # Pedestrians have no minimum-speed quality threshold. Acceptance uses
        # actual displacement and accumulated path length only.
        motion_policy = "spatial_only_no_speed_limit"
        motion_gate_passed = spatial_motion_passed
        motion_failure_detail = (
            "motion_gate_failed: "
            f"displacement_m={displacement:.3f}/"
            f"{displacement_threshold:.3f}, "
            f"path_length_m={path_length:.3f}/"
            f"{path_length_threshold:.3f}"
        )
    else:
        motion_policy = "spatial_or_temporal"
        motion_gate_passed = spatial_motion_passed or temporal_motion_passed
        motion_failure_detail = (
            "motion_gate_failed: "
            f"displacement_m={displacement:.3f}/"
            f"{displacement_threshold:.3f}, "
            f"path_length_m={path_length:.3f}/"
            f"{path_length_threshold:.3f}, "
            f"median_speed_mps={median_speed:.3f}/"
            f"{speed_threshold:.3f}, "
            f"moving_frame_ratio={moving_ratio:.3f}/"
            f"{moving_ratio_threshold:.3f}"
        )

    failures: List[str] = []
    checks = (
        (
            valid_ratio >= float(config["min_valid_frame_ratio"]),
            f"valid_frame_ratio={valid_ratio:.3f}",
        ),
        (
            visible_ratio >= float(config["min_anchor_visible_any_camera_ratio"]),
            f"anchor_visible_ratio={visible_ratio:.3f}",
        ),
        (
            longest_common_seconds >= float(config["min_anchor_common_view_seconds"]),
            f"longest_common_view_seconds={longest_common_seconds:.3f}",
        ),
        (
            motion_gate_passed,
            motion_failure_detail,
        ),
        (
            len(effective_tracks) >= int(config["min_effective_tracks"]),
            f"effective_tracks={len(effective_tracks)}",
        ),
        (
            dynamic_event_count >= minimum_dynamic_event_count,
            f"dynamic_event_count={dynamic_event_count}/"
            f"{minimum_dynamic_event_count}",
        ),
        (
            near_duplicate_ratio <= float(config["max_near_duplicate_pair_ratio"]),
            f"near_duplicate_pair_ratio={near_duplicate_ratio:.3f}",
        ),
    )
    for passed, detail in checks:
        if not passed:
            failures.append(detail)

    return {
        "passed": not failures,
        "failures": failures,
        "frame_count": frame_count,
        "valid_frame_ratio": valid_ratio,
        "anchor_visible_any_camera_ratio": visible_ratio,
        "anchor_common_camera_ratio": common_ratio,
        "anchor_longest_common_view_frames": longest_common_frames,
        "anchor_longest_common_view_seconds": longest_common_seconds,
        "anchor_reappearance_count": reappearances,
        "anchor_displacement_m": float(displacement),
        "anchor_path_length_m": float(path_length),
        "anchor_median_speed_mps": median_speed,
        "anchor_moving_frame_ratio": moving_ratio,
        "motion_gate": {
            "passed": motion_gate_passed,
            "policy": motion_policy,
            "spatial_motion_passed": spatial_motion_passed,
            "temporal_motion_passed": temporal_motion_passed,
            "thresholds": {
                "displacement_m": displacement_threshold,
                "path_length_m": path_length_threshold,
                "median_speed_mps": speed_threshold,
                "moving_frame_ratio": moving_ratio_threshold,
                "moving_step_m": moving_step_threshold,
            },
        },
        "effective_track_count": len(effective_tracks),
        "effective_track_lengths": effective_tracks,
        "effective_track_minimum_seconds": minimum_track_seconds,
        "effective_track_minimum_frames": minimum_track_frames,
        "dynamic_events": events,
        "dynamic_event_count": dynamic_event_count,
        "minimum_dynamic_event_count": minimum_dynamic_event_count,
        "anchor_heading_change_deg": float(heading_change),
        "minimum_other_actor_distance_m": (
            None
            if not math.isfinite(minimum_other_distance)
            else float(minimum_other_distance)
        ),
        "ssim": {
            "pair_count": len(all_ssim),
            "mean": float(np.mean(all_ssim)) if all_ssim else None,
            "median": float(np.median(all_ssim)) if all_ssim else None,
            "maximum": float(max(all_ssim)) if all_ssim else None,
            "near_duplicate_threshold": near_duplicate_threshold,
            "near_duplicate_pair_ratio": near_duplicate_ratio,
            "by_camera": {
                name: {
                    "pair_count": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "median": float(np.median(values)) if values else None,
                    "maximum": float(max(values)) if values else None,
                }
                for name, values in sorted(ssim_by_camera.items())
            },
        },
    }


def save_cooperative_task3_frame(
    cooperative: CooperativeManager,
    world: Any,
    units: Sequence[CameraUnit],
    raw: Dict[str, Dict[str, Any]],
    ground_raw: Dict[str, Any],
    camera_payloads: Dict[str, Dict[str, Any]],
    frame_index: int,
    carla_frame: int,
    config: Dict[str, Any],
) -> None:
    staging = cooperative.writer.current_path
    if staging is None or cooperative.object_registry is None:
        raise RuntimeError("cooperative Task-3 transaction is not active")
    stem = f"{frame_index:06d}"
    observations_by_sensor: Dict[str, List[Dict[str, Any]]] = {}
    sensor_data_by_id: Dict[str, Any] = {}
    for camera_index, unit in enumerate(units, start=1):
        platform_id = f"air_camera_{camera_index:02d}"
        payload = camera_payloads[unit.name]
        unit_raw = raw[unit.name]
        root = staging / "platforms/air" / platform_id
        rgb_path = root / "rgb" / f"{stem}.png"
        depth_path = root / "depth_m" / f"{stem}.npy"
        semantic_path = root / "semantic" / f"{stem}.png"
        for path in (rgb_path, depth_path, semantic_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rgb_path), payload["rgb_bgr"])
        np.save(depth_path, base.decode_carla_depth_meters(unit_raw["depth"]).astype(np.float32))
        base.save_semantic_id(
            base.decode_semantic_segmentation(unit_raw["semantic"]),
            semantic_path,
        )
        observations = enrich_observations(
            payload["annotations"],
            cooperative.object_registry,
            f"{platform_id}_rgb",
        )
        observations_by_sensor[f"{platform_id}_rgb"] = observations
        for modality, data in unit_raw.items():
            sensor_data_by_id[f"{platform_id}_{modality}"] = data

    ground_rgb = ground_raw["vehicle_01_rgb"]
    ground_depth_data = ground_raw["vehicle_01_depth"]
    ground_semantic_data = ground_raw["vehicle_01_semantic"]
    ground_depth = base.decode_carla_depth_meters(ground_depth_data)
    ground_semantic = base.decode_semantic_segmentation(ground_semantic_data)
    ground_root = staging / "platforms/ground/vehicle_01"
    ground_rgb_path = ground_root / "rgb" / f"{stem}.png"
    ground_depth_path = ground_root / "depth_m" / f"{stem}.npy"
    ground_semantic_path = ground_root / "semantic" / f"{stem}.png"
    for path in (ground_rgb_path, ground_depth_path, ground_semantic_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    base.save_rgb(ground_rgb, ground_rgb_path, depth_m=ground_depth)
    np.save(ground_depth_path, ground_depth.astype(np.float32))
    base.save_semantic_id(ground_semantic, ground_semantic_path)
    ground_config = dict(config)
    ground_vehicle_config = cooperative.config["ground_vehicle"]
    ground_config.update(
        {
            "width": int(ground_vehicle_config["image_width"]),
            "height": int(ground_vehicle_config["image_height"]),
            "fov": float(ground_vehicle_config["camera_fov_deg"]),
        }
    )
    ground_annotations = build_annotations(
        world,
        ground_rgb.transform,
        ground_depth,
        ground_semantic,
        ground_config,
    )
    ground_observations = enrich_observations(
        ground_annotations,
        cooperative.object_registry,
        "vehicle_01_rgb",
    )
    observations_by_sensor["vehicle_01_rgb"] = ground_observations
    sensor_data_by_id.update(ground_raw)
    snapshot = world.get_snapshot()
    first_rgb = raw[units[0].name]["rgb"]
    cooperative.record_frame(
        carla_frame,
        float(first_rgb.timestamp),
        float(snapshot.timestamp.elapsed_seconds),
        observations_by_sensor,
        sensor_data_by_id,
        ground_raw,
    )


def collect_scene_attempt(
    world: Any,
    units: Sequence[CameraUnit],
    transforms: Sequence[Any],
    ground_z: float,
    spec: SceneSpec,
    staging_dir: Path,
    paths: Dict[str, Path],
    traffic_actors: Sequence[Any],
    required_global_id: int,
    config: Dict[str, Any],
    cooperative: CooperativeManager,
) -> Dict[str, Any]:
    for unit, transform in zip(units, transforms):
        set_camera_unit_transform(unit, transform)
    if bool(config.get("spectator_follow_camera", False)):
        world.get_spectator().set_transform(transforms[0])
    drain_camera_units(units)
    timeout = float(config["sensor_timeout"])
    warmup_ticks = max(2, int(config.get("streaming_warmup_ticks", 2)))
    for _ in range(warmup_ticks):
        tick_and_get(
            world,
            units,
            timeout,
            before_tick=cooperative.before_world_tick,
        )
    drain_camera_units(units)
    cooperative.ground_vehicle.drain()

    current_anchor = next(
        (
            actor
            for actor in registered_actors(traffic_actors)
            if int(actor.id) == int(required_global_id)
        ),
        None,
    )
    if current_anchor is None:
        return {
            "accepted": False,
            "reason": "anchor_actor_is_no_longer_alive",
            "quality": {
                "passed": False,
                "failures": ["anchor_actor_is_no_longer_alive"],
            },
        }
    if spec.anchor_class != "pedestrian":
        current_speed_threshold = float(config["vehicle_min_median_speed_mps"])
        try:
            current_anchor_speed = actor_speed_mps(current_anchor)
        except RuntimeError:
            current_anchor_speed = 0.0
        if current_anchor_speed < current_speed_threshold:
            return {
                "accepted": False,
                "reason": f"anchor_current_speed_mps={current_anchor_speed:.3f}",
                "quality": {
                    "passed": False,
                    "failures": [
                        f"anchor_current_speed_mps={current_anchor_speed:.3f}"
                    ],
                },
            }

    for unit in units:
        write_json(
            staging_dir / "calibration" / f"{unit.name}.json",
            calibration_dict(unit.name, unit.transform, config, ground_z),
        )

    max_frames = frames_for_scene(spec, config)
    sample_interval_ticks = sample_interval_ticks_for_scene(spec, config)
    qa_count = min(int(config["qa_frames_per_camera"]), max_frames)
    qa_indices = sorted(
        set(
            np.linspace(0, max_frames - 1, qa_count)
            .round()
            .astype(int)
            .tolist()
        )
    )
    valid_frames = 0
    common_counts: List[int] = []
    observation_counts: Counter = Counter()
    class_counts: Counter = Counter()
    frame_rows: List[Tuple[int, int, int]] = []
    frame_validity: List[bool] = []
    anchor_camera_counts: List[int] = []
    track_world_samples: Dict[
        int,
        List[Tuple[int, float, float, float, float]],
    ] = defaultdict(list)
    track_visible_frames: Dict[int, set] = defaultdict(set)
    ssim_by_camera: Dict[str, List[float]] = defaultdict(list)
    previous_rgb: Dict[str, np.ndarray] = {}
    invalid_frame_reasons: Counter = Counter()
    empty_camera_streaks: Dict[str, int] = {
        unit.name: 0 for unit in units
    }
    empty_camera_patience = int(config.get("empty_camera_patience_frames", 5))
    empty_camera_stop_count = int(config.get("empty_camera_stop_count", 2))
    termination_reason = "max_frames_reached"

    for frame_index in range(max_frames):
        actor_by_id = {
            int(actor.id): actor
            for actor in traffic_actors
            if actor is not None and actor.is_alive
        }
        raw = None
        carla_frame = -1
        for _ in range(sample_interval_ticks):
            carla_frame, raw = tick_and_get(
                world,
                units,
                timeout,
                before_tick=cooperative.before_world_tick,
            )
        assert raw is not None
        ground_raw = cooperative.collect_ground(carla_frame)
        camera_payloads: Dict[str, Dict[str, Any]] = {}
        all_views_valid = True
        for unit in units:
            payload = inspect_camera_frame(
                world,
                unit,
                raw[unit.name],
                actor_by_id,
                config,
            )
            camera_payloads[unit.name] = payload
            all_views_valid = all_views_valid and bool(payload["view_valid"])
        rejected_unmodeled_views = {
            camera_name: payload["unmodeled_region"]
            for camera_name, payload in camera_payloads.items()
            if not bool(payload["unmodeled_region"]["passed"])
        }
        if rejected_unmodeled_views:
            details = ", ".join(
                (
                    f"{camera_name}:dark_flat="
                    f"{float(metrics['dark_flat_component_ratio']):.3f},"
                    f"flat={float(metrics['flat_component_ratio']):.3f},"
                    f"far_depth={float(metrics['far_depth_ratio']):.3f}"
                )
                for camera_name, metrics in rejected_unmodeled_views.items()
            )
            return {
                "accepted": False,
                "reason": f"unmodeled_map_region_detected ({details})",
                "quality": {
                    "passed": False,
                    "failures": ["unmodeled_map_region_detected"],
                    "rejected_cameras": rejected_unmodeled_views,
                    "rejected_at_frame": frame_index,
                },
            }
        for camera_name, payload in camera_payloads.items():
            if payload["annotations"]:
                empty_camera_streaks[camera_name] = 0
            else:
                empty_camera_streaks[camera_name] += 1
        empty_cameras_over_limit = sorted(
            camera_name
            for camera_name, streak in empty_camera_streaks.items()
            if streak > empty_camera_patience
        )
        if len(empty_cameras_over_limit) >= empty_camera_stop_count:
            termination_reason = (
                "empty_camera_patience_exceeded:"
                + ",".join(empty_cameras_over_limit)
            )
            print(
                f"[STOP] {spec.name}: collected={len(frame_rows)}/{max_frames}, "
                f"cameras without targets for more than "
                f"{empty_camera_patience} frames="
                f"{empty_cameras_over_limit}",
                flush=True,
            )
            break
        valid, common_ids, visibility_counts = frame_quality(
            camera_payloads,
            spec.anchor_class,
            config,
        )
        required_visible = int(required_global_id) in set(map(int, common_ids))
        valid = valid and all_views_valid and required_visible
        frame_validity.append(bool(valid))
        anchor_camera_count = sum(
            any(
                int(annotation["global_id"]) == int(required_global_id)
                for annotation in payload["annotations"]
            )
            for payload in camera_payloads.values()
        )
        anchor_camera_counts.append(anchor_camera_count)
        if not valid:
            if not all_views_valid:
                invalid_frame_reasons["invalid_view"] += 1
            if not common_ids:
                invalid_frame_reasons["no_common_id"] += 1
            if not required_visible:
                invalid_frame_reasons["anchor_not_common"] += 1
            if any(
                int(payload["boundary_annotation_count"]) > 0
                for payload in camera_payloads.values()
            ):
                invalid_frame_reasons["boundary_annotation"] += 1
        else:
            valid_frames += 1
        common_counts.append(len(common_ids))
        observation_counts.update(visibility_counts)
        observed_ids: List[int] = []
        for camera_name, payload in camera_payloads.items():
            if camera_name in previous_rgb:
                ssim_by_camera[camera_name].append(
                    ssim_gray(
                        previous_rgb[camera_name],
                        payload["rgb_bgr"],
                        int(config["ssim_resize_width"]),
                    )
                )
            previous_rgb[camera_name] = payload["rgb_bgr"]
            for annotation in payload["annotations"]:
                class_counts[str(annotation["class_name"])] += 1
                actor_id = int(annotation["global_id"])
                observed_ids.append(actor_id)
                track_visible_frames[actor_id].add(frame_index)
        for actor_id, actor in actor_by_id.items():
            try:
                transform = actor.get_transform()
                speed = actor_speed_mps(actor)
            except RuntimeError:
                continue
            track_world_samples[int(actor_id)].append(
                (
                    frame_index,
                    float(transform.location.x),
                    float(transform.location.y),
                    float(transform.location.z),
                    float(speed),
                )
            )
        write_frame(
            staging_dir,
            spec,
            frame_index,
            carla_frame,
            camera_payloads,
            common_ids,
            paths["qa"],
            qa_indices,
            config,
        )
        save_cooperative_task3_frame(
            cooperative,
            world,
            units,
            raw,
            ground_raw,
            camera_payloads,
            frame_index,
            carla_frame,
            config,
        )
        write_world_tracks(
            staging_dir,
            spec,
            frame_index,
            carla_frame,
            actor_by_id,
            observed_ids,
        )
        frame_rows.append((frame_index, carla_frame, len(common_ids)))

    quality = sequence_quality_report(
        spec,
        required_global_id,
        frame_validity,
        anchor_camera_counts,
        track_world_samples,
        track_visible_frames,
        ssim_by_camera,
        config,
    )
    quality["invalid_frame_reasons"] = dict(invalid_frame_reasons)
    quality["max_frame_count"] = max_frames
    quality["termination_reason"] = termination_reason
    quality["empty_camera_patience_frames"] = empty_camera_patience
    quality["empty_camera_stop_count"] = empty_camera_stop_count
    quality["camera_rig_geometry"] = camera_rig_geometry(transforms)
    write_json(staging_dir / "scene_quality.json", quality)
    accepted = bool(quality["passed"])
    if accepted:
        with (staging_dir / "sync.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ["dataset_frame", "carla_frame", "common_global_id_count"]
            )
            writer.writerows(frame_rows)
        write_json(
            staging_dir / "scene_meta.json",
            {
                "scene": spec.name,
                "scene_id": int(spec.scene_id),
                "split": spec.split,
                "weather": spec.weather,
                "anchor_class": spec.anchor_class,
                "anchor_global_id": dataset_global_id(
                    spec.scene_id,
                    int(required_global_id),
                ),
                "anchor_carla_actor_id": int(required_global_id),
                "anchor_instance_key": actor_instance_key(
                    current_anchor,
                    int(config["seed"]),
                ),
                "collection_seed": int(config["seed"]),
                "frames": len(frame_rows),
                "max_frames": max_frames,
                "termination_reason": termination_reason,
                "simulation_fps": float(config["fps"]),
                "sample_interval_ticks": sample_interval_ticks,
                "sample_interval_seconds": sample_interval_seconds_for_scene(
                    spec,
                    config,
                ),
                "simulated_duration_seconds": (
                    len(frame_rows)
                    * sample_interval_ticks
                    / float(config["fps"])
                ),
                "camera_ids": [unit.name for unit in units],
                "ground_plane_z_m": ground_z,
                "camera_rig_geometry": quality["camera_rig_geometry"],
                "valid_frames": valid_frames,
                "mean_common_ids_per_frame": float(np.mean(common_counts)),
                "max_common_ids_per_frame": int(max(common_counts)),
                "global_ids_observed": [
                    dataset_global_id(spec.scene_id, int(actor_id))
                    for actor_id in sorted(observation_counts)
                ],
                "class_observations": dict(class_counts),
                "quality": quality,
            },
        )
    return {
        "accepted": accepted,
        "reason": "ok" if accepted else "; ".join(quality["failures"]),
        "valid_frames": valid_frames,
        "mean_common_ids_per_frame": (
            float(np.mean(common_counts)) if common_counts else 0.0
        ),
        "class_observations": dict(class_counts),
        "quality": quality,
    }


def collect_scene(
    world: Any,
    units: Sequence[CameraUnit],
    road_waypoints: Sequence[Any],
    traffic_actors: Sequence[Any],
    spec: SceneSpec,
    paths: Dict[str, Path],
    config: Dict[str, Any],
    rng: random.Random,
    eligible_pedestrian_ids: Optional[Sequence[int]] = None,
    used_anchor_ids: Optional[Sequence[int]] = None,
    used_anchor_instance_keys: Optional[Sequence[str]] = None,
    cooperative: Optional[CooperativeManager] = None,
) -> Dict[str, Any]:
    if cooperative is None:
        raise RuntimeError("Task 3 cooperative manager is required")
    max_attempts = int(config["max_scene_attempts"])
    max_attempts_per_anchor = max(
        1,
        int(config.get("max_camera_attempts_per_anchor", 3)),
    )
    anchor_failure_counts: Counter = Counter()
    permanently_excluded_anchor_ids = set(map(int, used_anchor_ids or []))
    permanently_excluded_instance_keys = set(
        used_anchor_instance_keys or []
    )
    temporarily_excluded_anchor_ids: set = set()
    for attempt in range(max_attempts):
        staging_dir = paths["staging"] / f"{spec.name}_attempt_{attempt:03d}"
        safe_remove_staging(staging_dir, paths["staging"])
        staging_dir.mkdir(parents=True)
        anchor_candidates = eligible_anchor_actors(
            traffic_actors,
            spec.anchor_class,
            config,
            eligible_pedestrian_ids,
        )
        anchor_candidates = [
            actor
            for actor in anchor_candidates
            if actor_instance_key(actor, int(config["seed"]))
            not in permanently_excluded_instance_keys
        ]
        anchor = choose_anchor_actor(
            anchor_candidates,
            config,
            rng,
            preferred_class=spec.anchor_class,
            excluded_actor_ids=(
                permanently_excluded_anchor_ids
                | temporarily_excluded_anchor_ids
            ),
        )
        if anchor is None and temporarily_excluded_anchor_ids:
            # Every candidate has been explored. Start another bounded pass while
            # retaining both the frame-quality requirements and the permanent
            # per-map list of anchors already saved in successful scenes.
            temporarily_excluded_anchor_ids.clear()
            anchor_failure_counts.clear()
            anchor = choose_anchor_actor(
                eligible_anchor_actors(
                    traffic_actors,
                    spec.anchor_class,
                    config,
                    eligible_pedestrian_ids,
                ),
                config,
                rng,
                preferred_class=spec.anchor_class,
                excluded_actor_ids=permanently_excluded_anchor_ids,
            )
        if anchor is None:
            safe_remove_staging(staging_dir, paths["staging"])
            raise RuntimeError(
                f"{spec.name} 当前没有正在运动的 {spec.anchor_class} 锚点"
            )
        rig = choose_camera_transforms(anchor, road_waypoints, config, rng)
        if rig is None:
            safe_remove_staging(staging_dir, paths["staging"])
            continue
        transforms, ground_z = rig
        cooperative.prepare_scene(
            spec.name,
            int(config["seed"]) + int(spec.scene_id) * 100_003 + attempt,
            str(cooperative.config["route_profile"]),
            actors=traffic_actors,
            target_actor=anchor,
        )
        for camera_index, unit in enumerate(units, start=1):
            cooperative.register_air_platform(
                f"air_camera_{camera_index:02d}",
                virtual=True,
                platform_type="air_camera_platform",
                sensors=unit.sensors,
                required_modalities=("rgb", "depth", "semantic"),
            )
        try:
            result = collect_scene_attempt(
                world,
                units,
                transforms,
                ground_z,
                spec,
                staging_dir,
                paths,
                traffic_actors,
                int(anchor.id),
                config,
                cooperative,
            )
        except (TimeoutError, RuntimeError) as exc:
            result = {"accepted": False, "reason": str(exc)}
        if result["accepted"]:
            destination = paths["scenes"] / spec.name
            if destination.exists():
                raise RuntimeError(f"场景目录意外存在：{destination}")
            shutil.move(str(staging_dir), str(destination))
            cooperative.commit_scene(
                {
                    "split": spec.split,
                    "weather": spec.weather,
                    "frames": int(cooperative.sample_index),
                    "air_platform_count": 3,
                    "air_platform_type": "air_camera_platform",
                    "air_platform_is_virtual": True,
                    "ground_vehicle_count": 1,
                    "derived_formats": ["MOTChallenge", "MCMOT"],
                    "model_input": "rgb",
                    "auxiliary_modalities": ["depth_m", "semantic"],
                    "parameter_status": "pilot_provisional",
                }
            )
            result.update(
                {
                    "scene": spec.name,
                    "split": spec.split,
                    "weather": spec.weather,
                    "attempt": attempt + 1,
                    "anchor_global_id": dataset_global_id(
                        spec.scene_id,
                        int(anchor.id),
                    ),
                    "anchor_carla_actor_id": int(anchor.id),
                    "anchor_class": spec.anchor_class,
                    "anchor_instance_key": actor_instance_key(
                        anchor,
                        int(config["seed"]),
                    ),
                }
            )
            print(
                f"[OK] {spec.name}: attempt={attempt + 1}, "
                f"common/frame={result['mean_common_ids_per_frame']:.2f}, "
                f"classes={result['class_observations']}"
            )
            return result
        cooperative.close_scene(
            abort_reason="scene_attempt_rejected: " + str(result["reason"])
        )
        print(
            f"[RETRY] {spec.name} attempt {attempt + 1}/{max_attempts}: "
            f"anchor_actor={int(anchor.id)}, {result['reason']}"
        )
        append_jsonl(
            paths["root"] / "rejected_scene_attempts.jsonl",
            {
                "scene": spec.name,
                "attempt": attempt + 1,
                "anchor_carla_actor_id": int(anchor.id),
                "anchor_class": spec.anchor_class,
                "reason": result["reason"],
                "quality": result.get("quality"),
            },
        )
        anchor_id = int(anchor.id)
        anchor_failure_counts[anchor_id] += 1
        if anchor_failure_counts[anchor_id] >= max_attempts_per_anchor:
            temporarily_excluded_anchor_ids.add(anchor_id)
        safe_remove_staging(staging_dir, paths["staging"])
    raise RuntimeError(
        f"{spec.name} 连续 {max_attempts} 次没有得到合格的三相机公共目标"
    )


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)


def normalize_existing_dataset(
    root: Path,
    specs: Sequence[SceneSpec],
) -> Dict[str, Any]:
    """
    将公开身份改为场景命名空间 ID，并同步修正场景级划分。

    CARLA actor.id 仍保存在 carla_actor_id 中。这样同一 actor 即使在多个
    场景被再次看到，也不会跨 train/val/test 形成 ReID 身份泄漏。
    """
    changed_files = 0
    normalized_scenes = 0
    spec_by_name = {spec.name: spec for spec in specs}
    for scene_dir in sorted((root / "scenes").glob("*")):
        if not scene_dir.is_dir() or scene_dir.name not in spec_by_name:
            continue
        spec = spec_by_name[scene_dir.name]
        namespace = (int(spec.scene_id) + 1) * 100000

        def raw_id(value: Any) -> int:
            integer = int(value)
            if namespace <= integer < namespace + 100000:
                return integer - namespace
            return integer

        for camera_dir in sorted((scene_dir / "cameras").glob("cam_*")):
            annotation_paths = sorted(
                (camera_dir / "annotations").glob("*.json")
            )
            mot_path = camera_dir / "gt" / "gt.txt"
            if mot_path.exists():
                mot_path.unlink()
            for annotation_path in annotation_paths:
                payload = json.loads(
                    annotation_path.read_text(encoding="utf-8")
                )
                payload["split"] = spec.split
                payload["scene_id"] = int(spec.scene_id)
                payload["common_global_ids"] = [
                    dataset_global_id(spec.scene_id, raw_id(global_id))
                    for global_id in payload.get("common_global_ids", [])
                ]
                for annotation in payload.get("annotations", []):
                    actor_id = int(
                        annotation.get(
                            "carla_actor_id",
                            raw_id(annotation["global_id"]),
                        )
                    )
                    annotation["carla_actor_id"] = actor_id
                    annotation["global_id"] = dataset_global_id(
                        spec.scene_id,
                        actor_id,
                    )
                    annotation["track_id"] = (
                        f"{spec.name}_{annotation['class_name']}_"
                        f"{annotation['global_id']}"
                    )
                    write_mot_line(
                        mot_path,
                        int(payload["dataset_frame"]),
                        annotation,
                    )
                write_json(annotation_path, payload)
                changed_files += 1

        tracks_path = scene_dir / "global_tracks.jsonl"
        if tracks_path.is_file():
            normalized_lines: List[str] = []
            for line in tracks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                for actor in payload.get("actors", []):
                    actor_id = int(
                        actor.get(
                            "carla_actor_id",
                            raw_id(actor["global_id"]),
                        )
                    )
                    actor["carla_actor_id"] = actor_id
                    actor["global_id"] = dataset_global_id(
                        spec.scene_id,
                        actor_id,
                    )
                normalized_lines.append(
                    json.dumps(payload, ensure_ascii=False)
                )
            tracks_path.write_text(
                "\n".join(normalized_lines)
                + ("\n" if normalized_lines else ""),
                encoding="utf-8",
            )
            changed_files += 1

        meta_path = scene_dir / "scene_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["scene_id"] = int(spec.scene_id)
        meta["split"] = spec.split
        meta["anchor_class"] = spec.anchor_class
        raw_anchor = int(
            meta.get(
                "anchor_carla_actor_id",
                raw_id(meta["anchor_global_id"]),
            )
        )
        meta["anchor_carla_actor_id"] = raw_anchor
        meta["anchor_global_id"] = dataset_global_id(
            spec.scene_id,
            raw_anchor,
        )
        meta["global_ids_observed"] = [
            dataset_global_id(spec.scene_id, raw_id(global_id))
            for global_id in meta.get("global_ids_observed", [])
        ]
        write_json(meta_path, meta)
        changed_files += 1
        normalized_scenes += 1
    return {
        "normalized_scenes": normalized_scenes,
        "normalized_files": changed_files,
    }


def rebuild_yolo_dataset(
    root: Path,
    specs: Optional[Sequence[SceneSpec]] = None,
) -> Dict[str, Any]:
    yolo = root / "yolo"
    if yolo.exists():
        shutil.rmtree(yolo)
    for split in ("train", "val", "test"):
        (yolo / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
    split_by_scene: Dict[str, str] = {}
    if specs is not None:
        split_by_scene = {spec.name: spec.split for spec in specs}
    counts: Counter = Counter()
    for scene_dir in sorted((root / "scenes").glob("*")):
        if not scene_dir.is_dir():
            continue
        meta_path = scene_dir / "scene_meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        split = split_by_scene.get(scene_dir.name, str(meta["split"]))
        for camera_dir in sorted((scene_dir / "cameras").glob("cam_*")):
            for image_path in sorted((camera_dir / "rgb").glob("*.png")):
                name = (
                    f"{scene_dir.name}_{camera_dir.name}_{image_path.stem}.png"
                )
                label_path = (
                    camera_dir / "labels_yolo" / f"{image_path.stem}.txt"
                )
                hardlink_or_copy(image_path, yolo / "images" / split / name)
                hardlink_or_copy(
                    label_path,
                    yolo / "labels" / split / f"{Path(name).stem}.txt",
                )
                counts[split] += 1
    data_yaml = (
        f"path: {yolo.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: vehicle\n"
        "  1: pedestrian\n"
    )
    (yolo / "data.yaml").write_text(data_yaml, encoding="utf-8")
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        scenes = sorted(
            scene.name
            for scene in (root / "scenes").glob("*")
            if scene.is_dir()
            and json.loads(
                (scene / "scene_meta.json").read_text(encoding="utf-8")
            )["split"]
            == split
        )
        (split_dir / f"{split}_scenes.txt").write_text(
            "\n".join(scenes) + ("\n" if scenes else ""),
            encoding="utf-8",
        )
    return {"yolo_image_counts": dict(counts), "data_yaml": str(yolo / "data.yaml")}


def audit_dataset(root: Path) -> Dict[str, Any]:
    scene_reports: List[Dict[str, Any]] = []
    total_images = 0
    total_annotations = Counter()
    global_actor_ids = set()
    ids_by_split: Dict[str, set] = defaultdict(set)
    images_by_split: Counter = Counter()
    annotations_by_split_class: Counter = Counter()
    weather_by_split: Dict[str, set] = defaultdict(set)
    equivalent_sides: Dict[str, List[float]] = defaultdict(list)
    occlusion_values: Dict[str, List[float]] = defaultdict(list)
    sync_errors: List[str] = []
    image_hashes: Counter = Counter()
    sequence_metrics: Dict[str, List[float]] = defaultdict(list)

    for scene_dir in sorted((root / "scenes").glob("*")):
        if not scene_dir.is_dir():
            continue
        meta_path = scene_dir / "scene_meta.json"
        if not meta_path.is_file():
            sync_errors.append(f"{scene_dir.name}: missing scene_meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        quality_path = scene_dir / "scene_quality.json"
        if not quality_path.is_file():
            sync_errors.append(f"{scene_dir.name}: missing scene_quality.json")
            quality: Dict[str, Any] = {}
        else:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if not bool(quality.get("passed", False)):
                sync_errors.append(
                    f"{scene_dir.name}: sequence quality failed: "
                    + "; ".join(map(str, quality.get("failures", [])))
                )
            for key in (
                "anchor_displacement_m",
                "anchor_path_length_m",
                "anchor_median_speed_mps",
                "anchor_moving_frame_ratio",
                "anchor_longest_common_view_seconds",
                "valid_frame_ratio",
            ):
                value = quality.get(key)
                if value is not None:
                    sequence_metrics[key].append(float(value))
            ssim_payload = quality.get("ssim", {})
            if ssim_payload.get("near_duplicate_pair_ratio") is not None:
                sequence_metrics["near_duplicate_pair_ratio"].append(
                    float(ssim_payload["near_duplicate_pair_ratio"])
                )
        split = str(meta["split"])
        weather_by_split[split].add(str(meta["weather"]))
        camera_dirs = sorted((scene_dir / "cameras").glob("cam_*"))
        frame_map: Dict[int, List[int]] = defaultdict(list)
        scene_class_counts = Counter()
        common_per_frame: Dict[int, set] = defaultdict(set)
        for camera_dir in camera_dirs:
            images = sorted((camera_dir / "rgb").glob("*.png"))
            labels = sorted((camera_dir / "labels_yolo").glob("*.txt"))
            annotations = sorted((camera_dir / "annotations").glob("*.json"))
            if not (len(images) == len(labels) == len(annotations)):
                sync_errors.append(
                    f"{scene_dir.name}/{camera_dir.name}: "
                    f"rgb={len(images)}, label={len(labels)}, ann={len(annotations)}"
                )
            total_images += len(images)
            images_by_split[split] += len(images)
            for image_path in images:
                image_hashes[
                    hashlib.sha1(image_path.read_bytes()).hexdigest()
                ] += 1
            for annotation_path in annotations:
                payload = json.loads(
                    annotation_path.read_text(encoding="utf-8")
                )
                frame_index = int(payload["dataset_frame"])
                frame_map[frame_index].append(int(payload["carla_frame"]))
                common_per_frame[frame_index].update(
                    map(int, payload["common_global_ids"])
                )
                for annotation in payload["annotations"]:
                    class_name = str(annotation["class_name"])
                    scene_class_counts[class_name] += 1
                    total_annotations[class_name] += 1
                    annotations_by_split_class[(split, class_name)] += 1
                    global_id = int(annotation["global_id"])
                    global_actor_ids.add(global_id)
                    ids_by_split[split].add(global_id)
                    _, _, box_width, box_height = annotation["bbox_xywh"]
                    equivalent_sides[class_name].append(
                        math.sqrt(float(box_width) * float(box_height))
                    )
                    occlusion_values[class_name].append(
                        float(annotation["occlusion"])
                    )
                    if float(annotation["occlusion"]) > 0.500001:
                        sync_errors.append(
                            f"{scene_dir.name}/{camera_dir.name}/"
                            f"{frame_index}: occlusion > 0.5"
                        )
        for frame_index, carla_frames in frame_map.items():
            if len(carla_frames) != len(camera_dirs) or len(set(carla_frames)) != 1:
                sync_errors.append(
                    f"{scene_dir.name} frame {frame_index}: "
                    f"CARLA frames={carla_frames}"
                )
            # A short loss of the common target is valid MOT data when the
            # sequence-level gate confirms a sufficiently long common-view run
            # and a later reappearance.  Synchronization, not visibility, is the
            # invariant checked here.
        scene_reports.append(
            {
                "scene": scene_dir.name,
                "split": meta["split"],
                "weather": meta["weather"],
                "camera_count": len(camera_dirs),
                "frames": int(meta["frames"]),
                "class_observations": dict(scene_class_counts),
                "mean_common_ids_per_frame": float(
                    np.mean([len(ids) for ids in common_per_frame.values()])
                ),
                "quality": quality,
            }
        )
    duplicate_image_files = int(
        sum(count - 1 for count in image_hashes.values() if count > 1)
    )
    identity_overlap = {}
    for first, second in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = ids_by_split[first] & ids_by_split[second]
        identity_overlap[f"{first}_{second}"] = sorted(map(int, overlap))
        if overlap:
            sync_errors.append(
                f"{first}/{second}: global_id overlap={len(overlap)}"
            )

    def distribution(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {}
        return {
            "min": float(np.min(array)),
            "q10": float(np.quantile(array, 0.10)),
            "median": float(np.median(array)),
            "q90": float(np.quantile(array, 0.90)),
            "max": float(np.max(array)),
        }

    report = {
        "root": str(root),
        "passed": (
            bool(scene_reports)
            and not sync_errors
            and duplicate_image_files == 0
        ),
        "scene_count": len(scene_reports),
        "image_count": total_images,
        "annotation_observations": dict(total_annotations),
        "images_by_split": dict(images_by_split),
        "annotations_by_split_and_class": {
            f"{split}/{class_name}": int(count)
            for (split, class_name), count in sorted(
                annotations_by_split_class.items()
            )
        },
        "weather_by_split": {
            split: sorted(values)
            for split, values in sorted(weather_by_split.items())
        },
        "global_ids_by_split": {
            split: len(values)
            for split, values in sorted(ids_by_split.items())
        },
        "identity_overlap": identity_overlap,
        "equivalent_side_px": {
            class_name: distribution(values)
            for class_name, values in equivalent_sides.items()
        },
        "occlusion": {
            class_name: distribution(values)
            for class_name, values in occlusion_values.items()
        },
        "sequence_quality_metrics": {
            name: distribution(values)
            for name, values in sorted(sequence_metrics.items())
        },
        "unique_global_actor_ids": len(global_actor_ids),
        "duplicate_image_files": duplicate_image_files,
        "errors": sync_errors,
        "scenes": scene_reports,
    }
    write_json(root / "quality_audit.json", report)
    return report


def write_dataset_manifest(
    root: Path,
    config: Dict[str, Any],
    audit: Dict[str, Any],
) -> None:
    manifest = {
        "name": "OpenHUTB UAV Multi-Camera MOT RGB",
        "version": "2.0",
        "task": [
            "object_detection",
            "multi_object_tracking",
            "multi_camera_tracking",
        ],
        "classes": CLASS_NAMES,
        "public_modalities": ["RGB"],
        "internal_annotation_sensors": ["depth", "semantic_segmentation"],
        "image": {
            "width": int(config["width"]),
            "height": int(config["height"]),
            "fov_deg": float(config["fov"]),
        },
        "capture": {
            "num_cameras": int(config["num_cameras"]),
            "run_seed": (
                int(config["seed"])
                if config.get("seed") is not None
                else None
            ),
            "seed_source": str(config.get("seed_source", "explicit")),
            "fps": float(config["fps"]),
            "sensor_tick": float(config.get("sensor_tick", 0.0)),
            "vehicle_sample_interval_ticks": int(config["sample_interval_ticks"]),
            "vehicle_sample_interval_seconds": (
                int(config["sample_interval_ticks"]) / float(config["fps"])
            ),
            "pedestrian_sample_hz": float(
                config.get("pedestrian_sample_hz", 5.0)
            ),
            "pedestrian_sample_interval_ticks": max(
                1,
                int(
                    round(
                        float(config["fps"])
                        / float(config.get("pedestrian_sample_hz", 5.0))
                    )
                ),
            ),
            "max_frames_per_scene": int(config["frames_per_scene"]),
            "train_max_frames_per_scene": int(
                config.get("train_frames_per_scene", config["frames_per_scene"])
            ),
            "eval_max_frames_per_scene": int(
                config.get("eval_frames_per_scene", config["frames_per_scene"])
            ),
            "early_stop": {
                "empty_camera_patience_frames": int(
                    config.get("empty_camera_patience_frames", 5)
                ),
                "empty_camera_stop_count": int(
                    config.get("empty_camera_stop_count", 2)
                ),
            },
            "weather_presets": list(config["weather_presets"]),
            "camera_height_m": [
                float(config["camera_height_min_m"]),
                float(config["camera_height_max_m"]),
            ],
            "camera_pitch_deg": [
                float(config["camera_pitch_min_deg"]),
                float(config["camera_pitch_max_deg"]),
            ],
            "camera_bearing_separation_deg": {
                "minimum": float(config["camera_bearing_separation_deg"]),
                "preferred_random_range": [
                    float(
                        config[
                            "camera_preferred_bearing_separation_min_deg"
                        ]
                    ),
                    float(
                        config[
                            "camera_preferred_bearing_separation_max_deg"
                        ]
                    ),
                ],
                "maximum": float(config["camera_max_bearing_separation_deg"]),
            },
            "camera_min_pair_distance_m": float(
                config["camera_min_pair_distance_m"]
            ),
        },
        "annotation_policy": {
            "global_id": (
                "(scene_id + 1) * 100000 + carla_actor_id; "
                "same across cameras and frames within a scene"
            ),
            "min_visibility": float(config["min_visible_ratio"]),
            "max_occlusion": 1.0 - float(config["min_visible_ratio"]),
            "minimum_equivalent_side_px": {
                "vehicle": float(config["min_vehicle_equivalent_side_px"]),
                "pedestrian": float(
                    config["min_pedestrian_equivalent_side_px"]
                ),
            },
        },
        "statistics": {
            key: audit[key]
            for key in (
                "scene_count",
                "image_count",
                "images_by_split",
                "annotation_observations",
                "annotations_by_split_and_class",
                "weather_by_split",
                "global_ids_by_split",
                "identity_overlap",
                "equivalent_side_px",
                "occlusion",
                "sequence_quality_metrics",
                "duplicate_image_files",
            )
        },
        "quality_audit_passed": bool(audit["passed"]),
    }
    write_json(root / "dataset_manifest.json", manifest)


def write_readme(root: Path, config: Dict[str, Any], audit: Dict[str, Any]) -> None:
    text = f"""# OpenHUTB UAV Multi-Camera MOT RGB

该数据集由 OpenHUTB/CARLA 合成，面向无人机跨相机多目标检测与跟踪。

## 数据内容

- 公开模态：RGB
- 类别：vehicle（0）、pedestrian（1）
- 相机：每个场景 {config['num_cameras']} 台无人机相机
- 仿真帧率：{config['fps']} FPS
- 车辆场景：每 {config['sample_interval_ticks']} 个同步帧保存一次
- 行人场景：{float(config.get('pedestrian_sample_hz', 5.0)):.1f} Hz 保存
- 场景长度：配置帧数是上限；至少 {config.get('empty_camera_stop_count', 2)} 台相机连续超过 {config.get('empty_camera_patience_frames', 5)} 帧没有目标时提前停止
- 相机布局：方位角和俯视角按范围随机，相机物理间距下限为 {float(config.get('camera_min_pair_distance_m', 0.0)):.1f} 米（0 表示不限制）
- 分辨率：{config['width']}x{config['height']}
- 天气：{', '.join(config['weather_presets'])}
- 场景级划分：train / val / test，避免相邻帧泄漏
- 全局身份：同一 CARLA actor.id 在所有相机和所有帧中保持一致
- 遮挡策略：可见比例低于 {config['min_visible_ratio']:.2f} 的目标不标注

## 目录

- `scenes/<scene>/cameras/<cam>/rgb`：同步 RGB 帧
- `scenes/<scene>/cameras/<cam>/labels_yolo`：YOLO 检测标签
- `scenes/<scene>/cameras/<cam>/annotations`：含 global_id 的详细标注
- `scenes/<scene>/cameras/<cam>/gt/gt.txt`：MOTChallenge 风格标注
- `scenes/<scene>/calibration`：每台相机内参和 CARLA 外参
- `scenes/<scene>/global_tracks.jsonl`：逐帧世界坐标轨迹
- `scenes/<scene>/scene_quality.json`：位移、速度、轨迹、公共视野和 SSIM 验收结果
- `yolo/data.yaml`：可直接训练 YOLO11
- `quality_audit.json`：同步、运动、重复帧、遮挡和标注审计

## 本次审计

- 场景：{audit['scene_count']}
- RGB 图像：{audit['image_count']}
- 唯一全局 actor：{audit['unique_global_actor_ids']}
- 审计通过：{audit['passed']}
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def run_collection(
    config: Dict[str, Any],
    specs: Sequence[SceneSpec],
    paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    simulator_session = ensure_simulator(
        carla,
        str(config["host"]),
        int(config["port"]),
        config.get("map"),
        dict(config.get("simulator", {})),
    )
    client = carla.Client(
        str(config["host"]),
        int(config["port"]),
    )
    client.set_timeout(float(config["timeout"]))
    if config.get("map"):
        world = client.load_world(str(config["map"]))
    else:
        world = client.get_world()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(config["fps"])
    world.apply_settings(settings)

    static_ids: List[int] = []
    vehicles: List[Any] = []
    walkers: List[Any] = []
    controllers: List[Any] = []
    units: List[CameraUnit] = []
    results: List[Dict[str, Any]] = []
    cooperative = CooperativeManager(
        client,
        world,
        carla,
        Path(config["cooperative"]["output_root"]),
        "task3",
        config["cooperative"],
    )
    try:
        if bool(config["hide_static_map_vehicles"]):
            static_ids = base.hide_static_map_vehicles(world)
        vehicle_spawn_count = min(
            int(config["vehicles"]),
            int(config.get("max_active_vehicles", config["vehicles"])),
        )
        walker_spawn_count = min(
            int(config["walkers"]),
            int(config.get("max_active_pedestrians", config["walkers"])),
        )
        vehicles, walkers, controllers = base.spawn_background_traffic(
            client,
            world,
            vehicle_spawn_count,
            walker_spawn_count,
            int(config["tm_port"]),
            int(config["seed"]),
        )
        if bool(config["remove_two_wheel_vehicles"]):
            base.destroy_live_two_wheel_vehicles(world)
            vehicles = registered_actors(vehicles)
        configure_vehicle_motion(
            client,
            vehicles,
            int(config["tm_port"]),
            int(config["seed"]),
            config,
        )
        refresh_walker_motion(
            world,
            controllers,
            int(config["seed"]),
            config,
        )
        # Let Traffic Manager and the AI walker controllers reach a stable
        # state. This history is never used to qualify a scene target.
        for _ in range(int(config.get("actor_spawn_warmup_ticks", 75))):
            world.tick()
        traffic_actors = collection_target_actors(
            world,
            vehicles,
            walkers,
            config,
        )
        if not traffic_actors:
            raise RuntimeError("没有成功生成车辆或行人")
        initial_anchor_actors = current_moving_anchor_actors(
            traffic_actors,
            "vehicle",
            config,
        )
        if not initial_anchor_actors:
            # The initial transform only provides a safe place to spawn the
            # reusable sensors. Actual targets are selected after warmup from
            # a fresh per-scene activation window.
            initial_anchor_actors = registered_actors(traffic_actors)
        if not initial_anchor_actors:
            raise RuntimeError("当前没有可用于初始化相机的交通参与者")
        carla_map = world.get_map()
        road_waypoints = carla_map.generate_waypoints(
            float(config["camera_candidate_spacing_m"])
        )
        initial_rng = random.Random(int(config["seed"]) + 1)
        initial_rig = None
        initial_anchor_ids: set = set()
        initial_anchor_limit = min(
            len(initial_anchor_actors),
            max(16, int(config["anchor_top_k"]) * 4),
        )
        for _ in range(initial_anchor_limit):
            anchor = choose_anchor_actor(
                initial_anchor_actors,
                config,
                initial_rng,
                excluded_actor_ids=initial_anchor_ids,
            )
            if anchor is None:
                break
            initial_anchor_ids.add(int(anchor.id))
            initial_rig = choose_camera_transforms(
                anchor,
                road_waypoints,
                config,
                initial_rng,
            )
            if initial_rig is not None:
                break
        if initial_rig is None:
            fallback_anchor = initial_anchor_actors[0]
            fallback_location = fallback_anchor.get_location()
            fallback_transform = carla.Transform(
                carla.Location(
                    x=float(fallback_location.x),
                    y=float(fallback_location.y),
                    z=float(fallback_location.z)
                    + float(config["camera_height_min_m"]),
                ),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
            initial_transform = fallback_transform
            print(
                "[WARN] Initial rig proposal unavailable; using a temporary "
                "sensor spawn transform. Scene collection will still search "
                "for a fully valid three-camera rig.",
                flush=True,
            )
        else:
            initial_transform = initial_rig[0][0]
        units = spawn_camera_units(world, initial_transform, config)
        rng = random.Random(int(config["seed"]) + 101)

        current_weather = None
        used_anchor_ids_by_class = {
            "vehicle": set(),
            "pedestrian": set(),
        }
        used_anchor_instance_keys_by_class = {
            "vehicle": set(),
            "pedestrian": set(),
        }
        # Rebuild the accepted-instance history from saved scenes. Identity is
        # based on collection seed + actor ID, never on the limited appearance
        # catalogue. A respawned actor is therefore a new target instance.
        for meta_path in paths["scenes"].glob("*/scene_meta.json"):
            try:
                previous_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            previous_class = str(previous_meta.get("anchor_class", ""))
            previous_instance_key = str(
                previous_meta.get("anchor_instance_key", "")
            )
            if (
                previous_class in used_anchor_instance_keys_by_class
                and previous_instance_key
            ):
                used_anchor_instance_keys_by_class[previous_class].add(
                    previous_instance_key
                )
        instance_history_path = paths["root"] / "used_anchor_instances.json"
        write_json(
            instance_history_path,
            {
                key: sorted(values)
                for key, values in used_anchor_instance_keys_by_class.items()
            },
        )
        for scene_index, spec in enumerate(specs):
            completed_scene = paths["scenes"] / spec.name
            if completed_scene.is_dir():
                if accepted_scene_directory(
                    completed_scene,
                    frames_for_scene(spec, config),
                ):
                    print(
                        f"[RESUME-SKIP] Scene {scene_index + 1}/{len(specs)}: "
                        f"{spec.name}",
                        flush=True,
                    )
                    continue
                quarantined = quarantine_rejected_scene(
                    completed_scene,
                    paths["root"],
                )
                print(
                    f"[RESUME-RECAPTURE] {spec.name}: previous scene moved to "
                    f"{quarantined}",
                    flush=True,
                )
            if current_weather != spec.weather:
                applied = base.apply_weather(world, spec.weather)
                print(f"[INFO] Weather: requested={spec.weather}, applied={applied}")
                for _ in range(int(config["weather_warmup_ticks"])):
                    world.tick()
                drain_camera_units(units)
                current_weather = spec.weather
            print(
                f"[INFO] Scene {scene_index + 1}/{len(specs)}: "
                f"{spec.name} ({spec.split}, anchor={spec.anchor_class})"
            )
            scene_interval_ticks = sample_interval_ticks_for_scene(spec, config)
            print(
                f"[INFO] Capture policy: max_frames={frames_for_scene(spec, config)}, "
                f"sample_hz={float(config['fps']) / scene_interval_ticks:.3f}, "
                f"stop_when={int(config.get('empty_camera_stop_count', 2))} "
                f"cameras exceed "
                f"{int(config.get('empty_camera_patience_frames', 5))} "
                f"consecutive target-free frames",
                flush=True,
            )
            # Keep one bounded moving population for the map, but permanently
            # exclude every anchor already saved by this run. Public IDs remain
            # namespaced by scene_id. Avoiding per-scene actor destruction is
            # also required for this OpenHUTB build: destroying attached walker
            # controllers can remove their parents and destabilize the server.
            population_limit = max(
                1,
                int(config.get("max_actor_population_attempts", 1)),
            )
            result = None
            scene_failure_reason = None
            for population_attempt in range(population_limit):
                scene_seed = (
                    int(config["seed"])
                    + spec.scene_id * 1009
                    + population_attempt * 100003
                )
                configure_vehicle_motion(
                    client,
                    vehicles,
                    int(config["tm_port"]),
                    scene_seed,
                    config,
                )
                traffic_actors = collection_target_actors(
                    world,
                    vehicles,
                    walkers,
                    config,
                )
                pedestrian_starting_positions = (
                    pedestrian_positions(traffic_actors)
                    if spec.anchor_class == "pedestrian"
                    else {}
                )
                refresh_walker_motion(
                    world,
                    controllers,
                    scene_seed,
                    config,
                )
                if spec.anchor_class == "pedestrian":
                    for _ in range(
                        int(config.get("walker_activation_ticks", 10))
                    ):
                        tick_and_get(
                            world,
                            units,
                            float(config["sensor_timeout"]),
                        )
                    drain_camera_units(units)
                traffic_actors = collection_target_actors(
                    world,
                    vehicles,
                    walkers,
                    config,
                )
                if spec.anchor_class == "pedestrian":
                    current_anchor_actors = pedestrians_moved_since(
                        traffic_actors,
                        pedestrian_starting_positions,
                        config,
                    )
                    print(
                        "[WALKER-ACTIVATION] "
                        f"observed={len(pedestrian_starting_positions)}, "
                        f"moved_xy_at_least_"
                        f"{float(config.get('pedestrian_activation_displacement_m', 0.05)):.2f}m="
                        f"{len(current_anchor_actors)}, "
                        f"controller_refs={len(controllers)}",
                        flush=True,
                    )
                else:
                    current_anchor_actors = current_moving_anchor_actors(
                        traffic_actors,
                        spec.anchor_class,
                        config,
                    )
                used_anchor_ids = used_anchor_ids_by_class[spec.anchor_class]
                used_anchor_instance_keys = used_anchor_instance_keys_by_class[
                    spec.anchor_class
                ]
                unused_anchor_actors = [
                    actor
                    for actor in current_anchor_actors
                    if int(actor.id) not in used_anchor_ids
                    and actor_instance_key(actor, int(config["seed"]))
                    not in used_anchor_instance_keys
                ]
                remaining_scenes = sum(
                    candidate.anchor_class == spec.anchor_class
                    for candidate in specs[scene_index:]
                )
                print(
                    f"[ANCHOR-POOL] class={spec.anchor_class}, "
                    f"moving={len(current_anchor_actors)}, "
                    f"unused={len(unused_anchor_actors)}, "
                    f"already_saved={len(used_anchor_ids)}, "
                    f"used_instances={len(used_anchor_instance_keys)}, "
                    f"remaining_scenes={remaining_scenes}",
                    flush=True,
                )
                if not traffic_actors:
                    if population_attempt + 1 >= population_limit:
                        scene_failure_reason = (
                            f"No vehicle or pedestrian was spawned for {spec.name}"
                        )
                        break
                    continue
                drain_camera_units(units)
                if not unused_anchor_actors:
                    if population_attempt + 1 >= population_limit:
                        scene_failure_reason = (
                            f"No unused, currently moving {spec.anchor_class} "
                            f"anchor was available for {spec.name}"
                        )
                        break
                    print(
                        f"[RETRY-TRAFFIC] {spec.name}: no unused, currently "
                        f"moving {spec.anchor_class} anchor",
                        flush=True,
                    )
                    continue
                try:
                    result = collect_scene(
                        world,
                        units,
                        road_waypoints,
                        traffic_actors,
                        spec,
                        paths,
                        config,
                        random.Random(scene_seed + 101),
                        eligible_pedestrian_ids=[
                            int(actor.id) for actor in unused_anchor_actors
                        ],
                        used_anchor_ids=used_anchor_ids,
                        used_anchor_instance_keys=used_anchor_instance_keys,
                        cooperative=cooperative,
                    )
                except RuntimeError as exc:
                    if population_attempt + 1 >= population_limit:
                        scene_failure_reason = str(exc)
                        break
                    print(
                        f"[RETRY-TRAFFIC] {spec.name}: motion refresh "
                        f"{population_attempt + 1}/{population_limit} rejected: {exc}",
                        flush=True,
                    )
                    continue
                result["actor_seed"] = scene_seed
                result["actor_population_attempt"] = population_attempt + 1
                result["current_moving_anchor_count"] = len(
                    current_anchor_actors
                )
                accepted_anchor_id = int(result["anchor_carla_actor_id"])
                if accepted_anchor_id in used_anchor_ids:
                    raise RuntimeError(
                        f"Accepted duplicate {spec.anchor_class} anchor "
                        f"{accepted_anchor_id}"
                    )
                accepted_instance_key = str(result["anchor_instance_key"])
                if accepted_instance_key in used_anchor_instance_keys:
                    raise RuntimeError(
                        f"Accepted duplicate {spec.anchor_class} instance "
                        f"{accepted_instance_key}"
                    )
                used_anchor_ids.add(accepted_anchor_id)
                used_anchor_instance_keys.add(accepted_instance_key)
                write_json(
                    instance_history_path,
                    {
                        key: sorted(values)
                        for key, values in (
                            used_anchor_instance_keys_by_class.items()
                        )
                    },
                )
                break
            if result is None:
                scene_failure_reason = scene_failure_reason or (
                    f"No accepted actor population for {spec.name}"
                )
                failed_result = {
                    "accepted": False,
                    "scene": spec.name,
                    "split": spec.split,
                    "weather": spec.weather,
                    "anchor_class": spec.anchor_class,
                    "reason": scene_failure_reason,
                }
                results.append(failed_result)
                append_jsonl(
                    paths["root"] / "failed_scenes.jsonl",
                    failed_result,
                )
                print(
                    f"[SCENE-FAILED] {spec.name}: {scene_failure_reason}; "
                    "continuing with the next scene.",
                    flush=True,
                )
                continue
            results.append(result)
    finally:
        cooperative.close()
        destroy_actors(
            sensor
            for unit in units
            for sensor in unit.sensors.values()
        )
        if bool(config.get("destroy_traffic_on_exit", False)):
            destroy_traffic_population(
                world,
                controllers,
                walkers,
                vehicles,
            )
        else:
            print(
                "[INFO] Traffic actors left for simulator-owned shutdown.",
                flush=True,
            )
        if static_ids:
            base.restore_static_map_vehicles(world, static_ids)
        world.apply_settings(original_settings)
        close_simulator(simulator_session)
    return results


def main() -> int:
    args = parse_args()
    config = load_config(args)
    validate_config(config)
    output = resolve_output(config)
    specs = build_scene_specs(config)
    if args.max_scenes is not None:
        specs = specs[: args.max_scenes]

    if args.audit_only:
        if not output.is_dir():
            raise FileNotFoundError(f"数据集不存在：{output}")
        normalization = normalize_existing_dataset(output, specs)
        yolo_report = rebuild_yolo_dataset(output)
        audit = audit_dataset(output)
        write_dataset_manifest(output, config, audit)
        write_readme(output, config, audit)
        print(
            json.dumps(
                {**normalization, **yolo_report, **audit},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if audit["passed"] else 2

    run_seed = resolve_run_seed(config)
    print(
        f"[INFO] Run seed: {run_seed} "
        f"({config.get('seed_source', 'explicit')})",
        flush=True,
    )
    paths = prepare_output(output, args.overwrite, args.resume)
    write_json(paths["root"] / "collection_config_used.json", config)
    started = time.time()
    results = run_collection(config, specs, paths)
    normalize_existing_dataset(paths["root"], specs)
    yolo_report = rebuild_yolo_dataset(paths["root"], specs)
    audit = audit_dataset(paths["root"])
    accepted_scene_names = [
        spec.name
        for spec in specs
        if accepted_scene_directory(
            paths["scenes"] / spec.name,
            frames_for_scene(spec, config),
        )
    ]
    missing_scene_names = [
        spec.name
        for spec in specs
        if spec.name not in set(accepted_scene_names)
    ]
    collection_complete = not missing_scene_names
    write_dataset_manifest(paths["root"], config, audit)
    write_readme(paths["root"], config, audit)
    write_json(
        paths["root"] / "collection_report.json",
        {
            "elapsed_seconds": time.time() - started,
            "run_seed": int(config["seed"]),
            "seed_source": str(config.get("seed_source", "explicit")),
            "scene_results": results,
            "yolo": yolo_report,
            "audit_passed": audit["passed"],
            "collection_complete": collection_complete,
            "accepted_scenes": accepted_scene_names,
            "missing_scenes": missing_scene_names,
        },
    )
    safe_remove_staging(paths["staging"], paths["root"])
    print(
        f"[DONE] root={paths['root']}, scenes={audit['scene_count']}, "
        f"images={audit['image_count']}, audit_passed={audit['passed']}, "
        f"collection_complete={collection_complete}, "
        f"missing_scenes={missing_scene_names}"
    )
    return 0 if audit["passed"] and collection_complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] 用户中断。")
        raise SystemExit(130)
