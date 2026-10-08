#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 OpenHUTB/CARLA 中采集 RGB 单目标追踪数据。。

公开数据采用 VOT 矩形框格式；RGB 是正式模型输入，深度和语义作为遮挡、穿模与
路面质量 QA 辅助数据保存。YOLO 标签会标出画面中全部合格车辆
和行人，避免把非主目标错误地当成背景。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

COLLECT_ROOT = Path(__file__).resolve().parent.parent
MULTIMODAL_DIR = COLLECT_ROOT / "multimodal"
if str(MULTIMODAL_DIR) not in sys.path:
    sys.path.insert(0, str(MULTIMODAL_DIR))
if str(COLLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECT_ROOT))

import collect_rpg_small_targets_carla_v2 as base
from cooperative_perception import CooperativeManager, close_simulator, ensure_simulator
from cooperative_perception.cooperative_annotation import (
    enrich_observations,
    target_event_record,
)


carla = base.carla
TARGETS = [
    base.TargetClass(name="vehicle", class_id=0, semantic_ids=[10]),
    base.TargetClass(name="pedestrian", class_id=1, semantic_ids=[4]),
]
CLASS_NAMES = {0: "vehicle", 1: "pedestrian"}
DEFAULT_CONFIG = Path(__file__).with_name("single_object_vot_config.json")


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    weather: str
    target_class: str
    motion_mode: str
    split: str
    sequence_id: int
    index_in_weather: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="保留已完整验收的序列，并从第一条缺失序列继续。",
    )
    parser.add_argument("--sequences-per-map", type=int, default=None)
    parser.add_argument("--frames-per-sequence", type=int, default=None)
    parser.add_argument(
        "--weather-presets",
        nargs="+",
        default=None,
        help="仅覆盖本次运行的天气列表，适合小规模验证。",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="仅运行排在前面的若干序列，适合冒烟测试。",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.out is not None:
        config["out"] = str(args.out)
    if args.sequences_per_map is not None:
        config["sequences_per_map"] = args.sequences_per_map
    if args.frames_per_sequence is not None:
        config["frames_per_sequence"] = args.frames_per_sequence
    if args.weather_presets is not None:
        config["weather_presets"] = args.weather_presets
    if (
        str(config.get("weather_assignment", "cycle")) == "random_per_run"
        and "run_seed" not in config
    ):
        run_seed = random.SystemRandom().randint(1, 2_147_000_000)
        config["run_seed"] = run_seed
        config["seed"] = run_seed
        config["weather_assignment_seed"] = run_seed + 10_007
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    positive = (
        "width",
        "height",
        "fov",
        "fps",
        "sample_interval_ticks",
        "sequences_per_map",
        "frames_per_sequence",
        "sensor_timeout",
    )
    for key in positive:
        if float(config[key]) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    for key in ("vehicles", "walkers"):
        if int(config[key]) < 0:
            raise ValueError(f"{key} 不能小于 0")
    include_existing_any = bool(config.get("include_existing_target_actors", False))
    include_existing_vehicle = bool(
        config.get("include_existing_target_vehicle_actors", include_existing_any)
    )
    include_existing_pedestrian = bool(
        config.get("include_existing_target_pedestrian_actors", include_existing_any)
    )
    if (
        int(config["vehicles"]) == 0
        and int(config["walkers"]) == 0
        and not include_existing_vehicle
        and not include_existing_pedestrian
    ):
        raise ValueError(
            "vehicles 和 walkers 不能同时为 0，除非启用现有目标 actor"
        )
    if int(config["sequences_per_map"]) < 3:
        raise ValueError("每张地图至少需要 3 条序列，才能覆盖 train/val/test")
    if float(config.get("sensor_tick", 0.0)) < 0.0:
        raise ValueError("sensor_tick 不能小于 0")
    if not 0.0 < float(config["min_visible_ratio"]) <= 1.0:
        raise ValueError("min_visible_ratio 必须在 (0, 1] 内")
    if not config["weather_presets"]:
        raise ValueError("weather_presets 不能为空")
    weather_assignment = str(config.get("weather_assignment", "cycle"))
    if weather_assignment not in {
        "cycle",
        "random",
        "random_unseeded",
        "random_per_run",
    }:
        raise ValueError(
            "weather_assignment 只能是 cycle、random、random_unseeded "
            "或 random_per_run"
        )
    if str(config.get("motion_mode")) != "rear_upper_follow":
        raise ValueError("Task 2 只允许 motion_mode=rear_upper_follow")
    interval_by_class = config.get("sample_interval_ticks_by_target_class", {})
    for target_class in ("vehicle", "pedestrian"):
        if int(interval_by_class.get(target_class, config["sample_interval_ticks"])) <= 0:
            raise ValueError(f"{target_class} 的采样 tick 间隔必须大于 0")
    target_classes = list(config.get("target_classes", []))
    if not target_classes:
        raise ValueError("target_classes 不能为空")
    unknown_classes = sorted(set(target_classes) - {"vehicle", "pedestrian"})
    if unknown_classes:
        raise ValueError(f"target_classes 包含未知类别：{unknown_classes}")
    rear_upper = dict(config.get("rear_upper_follow", {}))
    for key in (
        "back_distance_m",
        "height_above_target_m",
        "position_response_time_s",
        "rotation_response_time_s",
        "max_position_speed_mps",
        "max_yaw_rate_deg_s",
        "max_pitch_rate_deg_s",
    ):
        if float(rear_upper.get(key, 0.0)) <= 0.0:
            raise ValueError(f"rear_upper_follow/{key} 必须大于 0")
    if int(config.get("max_sequences_per_target_actor", 2)) <= 0:
        raise ValueError("max_sequences_per_target_actor 必须大于 0")
    if int(config.get("max_target_population_refreshes", 2)) < 0:
        raise ValueError("max_target_population_refreshes 不能小于 0")
    if int(config.get("min_eligible_target_candidates", 2)) <= 0:
        raise ValueError("min_eligible_target_candidates 必须大于 0")
    if int(config.get("target_population_refresh_warmup_ticks", 20)) < 0:
        raise ValueError("target_population_refresh_warmup_ticks 不能小于 0")
    for key in (
        "target_population_refresh_batch_size",
        "max_active_target_actors_by_class",
    ):
        for target_class, value in config.get(key, {}).items():
            if target_class not in {"vehicle", "pedestrian"}:
                raise ValueError(f"{key} 包含未知类别：{target_class}")
            if int(value) <= 0:
                raise ValueError(f"{key}/{target_class} 必须大于 0")
    for key in (
        "min_target_average_speed_mps",
        "min_target_moving_step_ratio",
        "target_moving_step_threshold_m",
    ):
        for target_class, value in config.get(key, {}).items():
            if target_class not in {"vehicle", "pedestrian"}:
                raise ValueError(f"{key} 包含未知类别：{target_class}")
            if float(value) < 0.0:
                raise ValueError(f"{key}/{target_class} 不能小于 0")


def frames_for_sequence(spec: SequenceSpec, config: Dict[str, Any]) -> int:
    return int(config["frames_per_sequence"])


def sample_interval_ticks_for_sequence(
    spec: SequenceSpec,
    config: Dict[str, Any],
) -> int:
    return int(
        config.get("sample_interval_ticks_by_target_class", {}).get(
            spec.target_class,
            config["sample_interval_ticks"],
        )
    )


def max_absent_ratio_for_sequence(
    spec: SequenceSpec,
    config: Dict[str, Any],
) -> float:
    return float(config["max_absent_ratio_per_sequence"])


def max_consecutive_absent_for_sequence(
    spec: SequenceSpec,
    config: Dict[str, Any],
) -> int:
    return int(config["max_consecutive_absent_frames"])


def prepare_output(
    root: Path,
    overwrite: bool,
    resume: bool = False,
) -> Dict[str, Path]:
    root = root.resolve()
    if overwrite and resume:
        raise ValueError("--overwrite 和 --resume 不能同时使用")
    if root.exists():
        if not overwrite and not resume:
            raise FileExistsError(
                f"输出目录已存在：{root}\n"
                "为避免混入旧追踪帧，换一个 out，或使用 --overwrite/--resume。"
            )
        safe_name = str(root.parent / root.name).lower()
        allowed_legacy = (
            "dataset_uav" in safe_name and "single_object_vot" in safe_name
        )
        allowed_smoke = (
            "airgroundcoopsuite" in safe_name
            and "_smoke" in safe_name
            and (
                "task2_agc_sot" in safe_name
                or "derived_task2_sot" in safe_name
            )
        )
        if not (allowed_legacy or allowed_smoke):
            raise RuntimeError(f"拒绝覆盖名称异常的目录：{root}")
        if overwrite:
            shutil.rmtree(root)
    paths = {
        "root": root,
        "vot": root / "vot",
        "yolo": root / "yolo",
        "qa": root / "qa_overlay",
        "tmp": root / "_sequence_staging",
    }
    if resume:
        # These are derived only after all source sequences finish. Rebuild
        # them from the preserved VOT sequence directories at finalization.
        for derived in (
            paths["yolo"],
            root / "splits",
        ):
            shutil.rmtree(derived, ignore_errors=True)
        shutil.rmtree(paths["tmp"], ignore_errors=True)
        for derived_file in (
            root / "dataset_manifest.json",
            root / "quality_audit.json",
            root / "README.md",
        ):
            try:
                derived_file.unlink()
            except FileNotFoundError:
                pass
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_sequence_specs(config: Dict[str, Any]) -> List[SequenceSpec]:
    """Build rear-upper-follow sequences with scene-level diversity."""
    specs: List[SequenceSpec] = []
    count = int(config["sequences_per_map"])
    weathers = list(config["weather_presets"])
    weather_assignment = str(config.get("weather_assignment", "cycle"))
    configured_weather_schedule = list(
        config.get("weather_sequence_schedule", [])
    )
    if configured_weather_schedule and len(configured_weather_schedule) != count:
        raise ValueError("weather_sequence_schedule 长度必须等于序列数")
    invalid_weathers = sorted(
        set(configured_weather_schedule) - set(weathers)
    )
    if invalid_weathers:
        raise ValueError(
            f"weather_sequence_schedule 包含未允许天气：{invalid_weathers}"
        )
    weather_rng = (
        random.SystemRandom()
        if weather_assignment == "random_unseeded"
        else random.Random(
            int(config.get("weather_assignment_seed", config["seed"]))
        )
    )
    weather_occurrences: Counter = Counter()
    if count == 3:
        split_schedule = ["train", "val", "test"]
    else:
        train_end = int(count * 0.60)
        val_end = train_end + int(count * 0.20)
        split_schedule = [
            "train" if index < train_end else "val" if index < val_end else "test"
            for index in range(count)
        ]
    class_offset = int(config.get("target_class_offset", 0))
    target_classes = list(config.get("target_classes", ("vehicle", "pedestrian")))
    motion_mode = "rear_upper_follow"
    for sequence_id in range(count):
        weather = (
            configured_weather_schedule[sequence_id]
            if configured_weather_schedule
            else weather_rng.choice(weathers)
            if weather_assignment in {
                "random",
                "random_unseeded",
                "random_per_run",
            }
            else weathers[sequence_id % len(weathers)]
        )
        index_in_weather = int(weather_occurrences[weather])
        weather_occurrences[weather] += 1
        target_class = target_classes[
            (sequence_id + class_offset) % len(target_classes)
        ]
        split = split_schedule[sequence_id]
        specs.append(
            SequenceSpec(
                name=(
                    f"{weather.lower()}_"
                    f"{target_class}_{motion_mode}_{index_in_weather:02d}"
                ),
                weather=weather,
                target_class=target_class,
                motion_mode=motion_mode,
                split=split,
                sequence_id=sequence_id,
                index_in_weather=index_in_weather,
            )
        )
    return specs


def spawn_internal_sensors(
    world,
    transform,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    sensor_types = {
        "rgb": "sensor.camera.rgb",
        "depth": "sensor.camera.depth",
        "semantic": "sensor.camera.semantic_segmentation",
    }
    sensors: Dict[str, Any] = {}
    library = world.get_blueprint_library()
    for name, sensor_type in sensor_types.items():
        blueprint = base.setup_camera_blueprint(
            library,
            sensor_type,
            int(config["width"]),
            int(config["height"]),
            float(config["fov"]),
            float(config.get("sensor_tick", 0.0)),
            enable_rgb_postprocess=bool(config["enable_rgb_postprocess"]),
        )
        if name == "rgb":
            # 保留曝光和天气后处理，但关闭相机高速移动造成的运动模糊。
            for attr_name in (
                "motion_blur_intensity",
                "motion_blur_max_distortion",
                "motion_blur_min_object_screen_size",
            ):
                if blueprint.has_attribute(attr_name):
                    blueprint.set_attribute(attr_name, "0.0")
        sensors[name] = world.spawn_actor(blueprint, transform)
    return sensors


def tick_and_get_sensors(
    world,
    sensor_sync: Dict[str, Any],
    timeout: float,
    before_tick=None,
) -> Tuple[int, Dict[str, Any]]:
    """Advance one simulation tick and consume every synchronized modality."""
    if before_tick is not None:
        before_tick()
    carla_frame = int(world.tick())
    return carla_frame, {
        name: sync.get(carla_frame, timeout=timeout)
        for name, sync in sensor_sync.items()
    }


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
    """Destroy walkers before their attached controllers on OpenHUTB."""
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


def actor_role_name(actor: Any) -> str:
    """Return an actor role without trusting every CARLA actor API variant."""
    try:
        return str(actor.attributes.get("role_name", ""))
    except (AttributeError, RuntimeError):
        return ""


def is_static_bridge_actor(actor: Any) -> bool:
    return actor_role_name(actor) in {
        "collection_static_vehicle",
        "collection_static_pedestrian",
    }


def live_targets(
    actors: Sequence[Any],
    target_class: str,
) -> List[Any]:
    prefix = "vehicle." if target_class == "vehicle" else "walker.pedestrian."
    return [
        actor
        for actor in actors
        if actor is not None
        and actor.is_alive
        and actor.type_id.startswith(prefix)
    ]


def choose_target_actor(
    actors: Sequence[Any],
    target_class: str,
    used_actor_ids: Counter,
    actor_split_assignments: Dict[int, str],
    split: str,
    max_sequences_per_actor: int,
    attempted_actor_ids: Counter,
    rng: random.Random,
) -> Any:
    candidates = eligible_target_actors(
        actors,
        target_class,
        used_actor_ids,
        actor_split_assignments,
        split,
        max_sequences_per_actor,
    )
    if not candidates:
        raise RuntimeError(
            f"没有可用于 {split} 的 {target_class} actor："
            f"每个目标最多 {max_sequences_per_actor} 段，且不能跨 split"
        )

    # Reuse an actor already owned by this split before claiming a new one.
    # This keeps enough untouched actors for val/test while still respecting
    # the configured one-or-two-sequence cap.
    minimum_attempts = min(
        attempted_actor_ids[int(actor.id)] for actor in candidates
    )
    candidates = [
        actor
        for actor in candidates
        if attempted_actor_ids[int(actor.id)] == minimum_attempts
    ]
    assigned = [
        actor
        for actor in candidates
        if actor_split_assignments.get(int(actor.id)) == split
    ]
    if assigned:
        candidates = assigned
    minimum_use = min(used_actor_ids[int(actor.id)] for actor in candidates)
    candidates = [
        actor
        for actor in candidates
        if used_actor_ids[int(actor.id)] == minimum_use
    ]
    target = rng.choice(candidates)
    attempted_actor_ids[int(target.id)] += 1
    return target


def eligible_target_actors(
    actors: Sequence[Any],
    target_class: str,
    used_actor_ids: Counter,
    actor_split_assignments: Dict[int, str],
    split: str,
    max_sequences_per_actor: int,
) -> List[Any]:
    return [
        actor
        for actor in live_targets(actors, target_class)
        if used_actor_ids[int(actor.id)] < max_sequences_per_actor
        and actor_split_assignments.get(int(actor.id), split) == split
    ]


def refresh_target_population(
    client: Any,
    world: Any,
    sensor_sync: Dict[str, Any],
    target_class: str,
    refresh_index: int,
    config: Dict[str, Any],
    vehicles: List[Any],
    walkers: List[Any],
    controllers: List[Any],
    target_vehicles: List[Any],
    target_walkers: List[Any],
) -> Dict[str, Any]:
    """Spawn a bounded batch of fresh targets for one exhausted class."""
    enabled = bool(
        config.get("target_population_refresh_enabled_by_class", {}).get(
            target_class,
            True,
        )
    )
    target_pool = target_vehicles if target_class == "vehicle" else target_walkers
    active_before = len(live_targets(target_pool, target_class))
    active_cap = int(
        config.get("max_active_target_actors_by_class", {}).get(
            target_class,
            32 if target_class == "vehicle" else 48,
        )
    )
    if not enabled:
        return {
            "spawned": 0,
            "active_before": active_before,
            "active_after": active_before,
            "reason": f"{target_class} population refresh disabled",
        }
    remaining_capacity = max(0, active_cap - active_before)
    requested = min(
        remaining_capacity,
        int(
            config.get("target_population_refresh_batch_size", {}).get(
                target_class,
                6 if target_class == "vehicle" else 8,
            )
        ),
    )
    if requested <= 0:
        return {
            "spawned": 0,
            "active_before": active_before,
            "active_after": active_before,
            "reason": (
                f"{target_class} active actor cap reached: "
                f"{active_before}/{active_cap}"
            ),
        }

    refresh_seed = (
        int(config["seed"])
        + 500_009
        + int(refresh_index) * 100_003
        + (17 if target_class == "pedestrian" else 0)
    )
    new_vehicles, new_walkers, new_controllers = base.spawn_background_traffic(
        client,
        world,
        requested if target_class == "vehicle" else 0,
        requested if target_class == "pedestrian" else 0,
        int(config["tm_port"]),
        refresh_seed,
    )
    vehicles.extend(new_vehicles)
    walkers.extend(new_walkers)
    controllers.extend(new_controllers)
    if target_class == "vehicle":
        target_vehicles.extend(new_vehicles)
    else:
        target_walkers.extend(new_walkers)

    for _ in range(int(config.get("target_population_refresh_warmup_ticks", 20))):
        world.tick()
    for item in sensor_sync.values():
        item.drain()

    active_after = len(live_targets(target_pool, target_class))
    return {
        "spawned": max(0, active_after - active_before),
        "requested": requested,
        "active_before": active_before,
        "active_after": active_after,
        "active_cap": active_cap,
        "seed": refresh_seed,
        "reason": "ok" if active_after > active_before else "spawned no live actors",
    }


def distance_xy(first, second) -> float:
    return math.hypot(float(first.x - second.x), float(first.y - second.y))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def angle_delta_degrees(current: float, desired: float) -> float:
    return (desired - current + 180.0) % 360.0 - 180.0


def move_angle_toward(
    current: float,
    desired: float,
    response_alpha: float,
    max_delta: float,
) -> float:
    delta = angle_delta_degrees(current, desired) * response_alpha
    return current + clamp(delta, -max_delta, max_delta)


def rear_upper_follow_transform(
    target,
    config: Dict[str, Any],
    state: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Compute a heading-relative, rate-limited rear-and-above camera pose."""
    follow = config["rear_upper_follow"]
    dt = max(
        1.0 / float(config["fps"]),
        float(state.get("dt_seconds", 1.0 / float(config["fps"]))),
    )
    target_transform = target.get_transform()
    target_location = target_transform.location
    forward = target_transform.get_forward_vector()
    forward_norm = max(
        1.0e-6,
        math.sqrt(float(forward.x) ** 2 + float(forward.y) ** 2),
    )
    forward_x = float(forward.x) / forward_norm
    forward_y = float(forward.y) / forward_norm
    desired_location = carla.Location(
        x=float(target_location.x)
        - float(follow["back_distance_m"]) * forward_x,
        y=float(target_location.y)
        - float(follow["back_distance_m"]) * forward_y,
        z=float(target_location.z) + float(follow["height_above_target_m"]),
    )

    previous = state.get("previous_transform")
    if previous is None:
        camera_location = desired_location
    else:
        response_time = max(1.0e-3, float(follow["position_response_time_s"]))
        alpha = 1.0 - math.exp(-dt / response_time)
        dx = alpha * float(desired_location.x - previous.location.x)
        dy = alpha * float(desired_location.y - previous.location.y)
        dz = alpha * float(desired_location.z - previous.location.z)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        max_distance = float(follow["max_position_speed_mps"]) * dt
        scale = min(1.0, max_distance / distance) if distance > 1.0e-6 else 1.0
        camera_location = carla.Location(
            x=float(previous.location.x) + dx * scale,
            y=float(previous.location.y) + dy * scale,
            z=float(previous.location.z) + dz * scale,
        )

    bbox = target.bounding_box
    aim_location = carla.Location(
        x=float(target_location.x),
        y=float(target_location.y),
        z=float(target_location.z + max(0.4, bbox.extent.z * 0.55)),
    )
    look_x = float(aim_location.x - camera_location.x)
    look_y = float(aim_location.y - camera_location.y)
    look_z = float(aim_location.z - camera_location.z)
    horizontal = max(0.1, math.hypot(look_x, look_y))
    desired_yaw = math.degrees(math.atan2(look_y, look_x))
    desired_pitch = clamp(
        math.degrees(math.atan2(look_z, horizontal)),
        float(follow["camera_pitch_min_deg"]),
        float(follow["camera_pitch_max_deg"]),
    )
    if previous is None:
        yaw = desired_yaw
        pitch = desired_pitch
    else:
        rotation_time = max(1.0e-3, float(follow["rotation_response_time_s"]))
        rotation_alpha = 1.0 - math.exp(-dt / rotation_time)
        yaw = move_angle_toward(
            float(previous.rotation.yaw),
            desired_yaw,
            rotation_alpha,
            float(follow["max_yaw_rate_deg_s"]) * dt,
        )
        pitch = move_angle_toward(
            float(previous.rotation.pitch),
            desired_pitch,
            rotation_alpha,
            float(follow["max_pitch_rate_deg_s"]) * dt,
        )
        pitch = clamp(
            pitch,
            float(follow["camera_pitch_min_deg"]),
            float(follow["camera_pitch_max_deg"]),
        )

    transform = carla.Transform(
        camera_location,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )
    state["previous_transform"] = transform
    desired_error = math.sqrt(
        float(desired_location.x - camera_location.x) ** 2
        + float(desired_location.y - camera_location.y) ** 2
        + float(desired_location.z - camera_location.z) ** 2
    )
    stats = motion_pose_stats(transform, target, "rear_upper_follow")
    stats.update(
        {
            "desired_x": float(desired_location.x),
            "desired_y": float(desired_location.y),
            "desired_z": float(desired_location.z),
            "desired_position_error_m": desired_error,
            "target_forward_x": forward_x,
            "target_forward_y": forward_y,
            "parameter_status": str(follow.get("status", "pilot_provisional")),
        }
    )
    return transform, stats


def motion_pose_stats(transform, target, motion_mode: str) -> Dict[str, Any]:
    target_location = target.get_location()
    horizontal = math.hypot(
        float(target_location.x - transform.location.x),
        float(target_location.y - transform.location.y),
    )
    return {
        "motion_mode": motion_mode,
        "altitude_m": float(transform.location.z - target_location.z),
        "ground_distance_m": horizontal,
        "pitch_deg": float(transform.rotation.pitch),
        "yaw_deg": float(transform.rotation.yaw),
        "ground_x": float(transform.location.x),
        "ground_y": float(transform.location.y),
        "ground_z": float(target_location.z),
    }


def camera_in_excluded_circle(transform, config: Dict[str, Any]) -> bool:
    camera_x = float(transform.location.x)
    camera_y = float(transform.location.y)
    return any(
        math.hypot(camera_x - float(center_x), camera_y - float(center_y))
        <= float(radius_m)
        for center_x, center_y, radius_m
        in config.get("excluded_camera_circles", [])
    )


def camera_transform_for_motion(
    carla_map,
    target,
    target_class: str,
    direction: str,
    progress: float,
    motion_mode: str,
    config: Dict[str, Any],
    state: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    del carla_map, target_class, direction, progress
    if motion_mode != "rear_upper_follow":
        raise RuntimeError(f"正式采集只支持 rear_upper_follow：{motion_mode}")
    return rear_upper_follow_transform(target, config, state)


def annotation_equivalent_side(annotation: Dict[str, Any]) -> float:
    _, _, width, height = annotation["bbox_xywh"]
    return math.sqrt(float(width) * float(height))


def bbox_iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(value) for value in first]
    bx, by, bw, bh = [float(value) for value in second]
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def is_boundary_box(
    annotation: Dict[str, Any],
    width: int,
    height: int,
    margin: int = 1,
) -> bool:
    x, y, box_width, box_height = annotation["bbox_xywh"]
    return (
        x <= margin
        or y <= margin
        or x + box_width >= width - margin
        or y + box_height >= height - margin
    )


def build_actor_annotations(
    world,
    camera_transform,
    depth_m: np.ndarray,
    semantic_id: np.ndarray,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ratio = float(config["min_visible_ratio"])
    annotations = base.build_annotations_from_actors(
        world=world,
        camera_transform=camera_transform,
        depth_m=depth_m,
        semantic_id=semantic_id,
        targets=TARGETS,
        width=int(config["width"]),
        height=int(config["height"]),
        fov=float(config["fov"]),
        min_mask_px=8,
        small_area_ratio=0.0025,
        small_max_side_px=96,
        keep_all=True,
        min_actor_visible_px=int(config["min_visible_pixels"]),
        min_actor_visible_ratio=ratio,
        min_vehicle_projected_fill_ratio=ratio,
        min_pedestrian_projected_fill_ratio=ratio,
        actor_depth_margin=float(config["actor_depth_margin_m"]),
        actor_visibility_mode="depth",
    )
    for annotation in annotations:
        annotation["boundary_truncated"] = bool(
            float(annotation.get("truncation_ratio", 0.0)) > 0.0
        )
    return annotations


def find_target_annotation(
    annotations: Sequence[Dict[str, Any]],
    actor_id: int,
    target_class: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    minimum_size = float(
        config[
            "min_vehicle_equivalent_side_px"
            if target_class == "vehicle"
            else "min_pedestrian_equivalent_side_px"
        ]
    )
    for annotation in annotations:
        if int(annotation.get("carla_actor_id", -1)) != int(actor_id):
            continue
        if annotation_equivalent_side(annotation) < minimum_size:
            return None
        return annotation
    return None


def enrich_target_annotation(annotation: Dict[str, Any]) -> str:
    visibility = float(annotation["visible_ratio_projected_bbox"])
    truncation = float(annotation.get("truncation_ratio", 0.0))
    occlusion = max(0.0, min(1.0, 1.0 - visibility))
    if truncation >= 0.05:
        state = "partial_out_of_view"
    elif visibility < 0.25:
        state = "severe_occlusion"
    elif visibility < 0.75:
        state = "partial_occlusion"
    else:
        state = "visible"
    annotation["visibility_ratio"] = visibility
    annotation["occlusion_ratio"] = occlusion
    annotation["target_state"] = state
    return state


def save_overlay(
    rgb_bgr: np.ndarray,
    annotations: Sequence[Dict[str, Any]],
    target_actor_id: int,
    target_annotation: Optional[Dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    canvas = rgb_bgr.copy()
    for annotation in annotations:
        x, y, width, height = map(int, annotation["bbox_xywh"])
        is_target = int(annotation.get("carla_actor_id", -1)) == target_actor_id
        color = (0, 255, 255) if is_target else (
            (0, 255, 0)
            if annotation["class_name"] == "vehicle"
            else (255, 80, 40)
        )
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            canvas,
            (
                "VOT target"
                if is_target
                else f"{annotation['class_name']} {annotation['carla_actor_id']}"
            ),
            (x, max(22, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if target_annotation is None:
        cv2.putText(
            canvas,
            "VOT TARGET ABSENT / OCCLUDED > 50%",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        title,
        (30, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)


def write_frame_metadata(
    file_handle,
    frame_index: int,
    carla_frame: int,
    target,
    target_class: str,
    target_annotation: Optional[Dict[str, Any]],
    annotations: Sequence[Dict[str, Any]],
    weather: str,
    camera_transform,
    pose_stats: Dict[str, float],
    view_stats: Dict[str, float],
    road_ratio: float,
) -> None:
    record = {
        "frame_index": frame_index,
        "carla_frame": int(carla_frame),
        "weather": weather,
        "target_class": target_class,
        "target_actor_id": int(target.id),
        "target_present": target_annotation is not None,
        "target_annotation": target_annotation,
        "annotations": list(annotations),
        "camera_transform": base.transform_to_dict(camera_transform),
        "camera_pose": pose_stats,
        "view_quality": {
            **view_stats,
            "road_visible_ratio": road_ratio,
        },
    }
    file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_sequence_attempt(
    world,
    sensors: Dict[str, Any],
    sensor_sync: Dict[str, Any],
    target,
    spec: SequenceSpec,
    attempt_dir: Path,
    direction: str,
    config: Dict[str, Any],
    cooperative: CooperativeManager,
) -> Tuple[bool, Dict[str, Any]]:
    color_dir = attempt_dir / "color"
    label_dir = attempt_dir / "labels_yolo"
    color_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    planned_max_frames = frames_for_sequence(spec, config)
    min_frames = max(1, int(config.get("minimum_track_length", 1)))
    sample_interval_ticks = sample_interval_ticks_for_sequence(spec, config)
    max_absent_ratio = max_absent_ratio_for_sequence(spec, config)
    max_consecutive_absent = max_consecutive_absent_for_sequence(spec, config)
    groundtruth: List[str] = []
    absence: List[str] = []
    occlusion: List[str] = []
    target_sizes: List[float] = []
    target_visible_ratios: List[float] = []
    target_occlusion_ratios: List[float] = []
    target_truncation_ratios: List[float] = []
    target_states: List[str] = []
    normalized_displacements: List[float] = []
    adjacent_scale_log_changes: List[float] = []
    previous_box_copy_ious: List[float] = []
    previous_target_box: Optional[Sequence[float]] = None
    road_ratios: List[float] = []
    near_ratios: List[float] = []
    image_differences: List[float] = []
    previous_gray: Optional[np.ndarray] = None
    target_world_step_distances: List[float] = []
    previous_target_world_location = None
    motion_state: Dict[str, Any] = {
        "dt_seconds": sample_interval_ticks / float(config["fps"]),
    }
    consecutive_absent = 0
    absent_frames = 0
    class_counts: Counter = Counter()
    termination_reason = "max_frames_reached"
    previous_cooperative_state: Optional[str] = None
    cooperative_state_counts: Counter = Counter()
    interaction_event_count = 0
    reacquisition_count = 0
    current_both_missing_frames = 0
    max_consecutive_both_missing_frames = 0

    metadata_path = attempt_dir / "annotations.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for frame_index in range(planned_max_frames):
            progress = frame_index / max(1, planned_max_frames - 1)
            try:
                camera_transform, pose_stats = camera_transform_for_motion(
                    world.get_map(),
                    target,
                    spec.target_class,
                    direction,
                    progress,
                    spec.motion_mode,
                    config,
                    motion_state,
                )
            except RuntimeError as exc:
                return False, {"reason": str(exc), "saved_frames": frame_index}

            base.set_all_sensor_transform(sensors, camera_transform)
            if bool(config.get("spectator_follow_camera", False)):
                world.get_spectator().set_transform(camera_transform)
            if frame_index == 0:
                for _ in range(int(config.get("streaming_warmup_frames", 0))):
                    cooperative.before_world_tick()
                    world.tick()
                for item in sensor_sync.values():
                    item.drain()
                cooperative.ground_vehicle.drain()
            try:
                raw = None
                carla_frame = -1
                for _ in range(sample_interval_ticks):
                    carla_frame, raw = tick_and_get_sensors(
                        world,
                        sensor_sync,
                        float(config["sensor_timeout"]),
                        before_tick=cooperative.before_world_tick,
                    )
                assert raw is not None
                ground_raw = cooperative.collect_ground(carla_frame)
                rgb_data = raw["rgb"]
                depth_data = raw["depth"]
                semantic_data = raw["semantic"]
            except (RuntimeError, TimeoutError) as exc:
                return False, {
                    "reason": f"sensor_sync: {exc}",
                    "saved_frames": frame_index,
                }

            current_target_world_location = target.get_location()
            if previous_target_world_location is not None:
                target_world_step_distances.append(
                    distance_xy(
                        current_target_world_location,
                        previous_target_world_location,
                    )
                )
            previous_target_world_location = current_target_world_location

            depth_m = base.decode_carla_depth_meters(depth_data)
            semantic_id = base.decode_semantic_segmentation(semantic_data)
            if camera_in_excluded_circle(rgb_data.transform, config):
                return False, {
                    "reason": "相机进入已知贴图/几何错误区域",
                    "saved_frames": frame_index,
                }
            streaming_stats: Dict[str, Any] = {}
            if bool(config.get("require_streaming_geometry", False)):
                streaming_ready, streaming_stats = base.streaming_scene_readiness(
                    depth_m,
                    semantic_id,
                    max_geometry_depth_m=float(
                        config.get("streaming_geometry_max_depth_m", 220.0)
                    ),
                    min_geometry_ratio=float(
                        config.get("streaming_min_geometry_ratio", 0.90)
                    ),
                    min_labeled_ratio=0.0,
                )
                if not streaming_ready:
                    return False, {
                        "reason": (
                            "地图几何未完整加载或画面包含大面积虚空："
                            f"geometry_ratio="
                            f"{streaming_stats['streaming_geometry_ratio']:.4f}"
                        ),
                        "saved_frames": frame_index,
                    }
            bad_view, view_stats = base.is_bad_camera_view(
                depth_m,
                min_near_depth_m=float(config["min_near_depth_m"]),
                max_near_depth_ratio=float(config["max_near_depth_ratio"]),
            )
            view_stats.update(streaming_stats)
            road_ratio = base.road_visible_ratio(
                semantic_data,
                [int(value) for value in config["road_semantic_ids"]],
            )
            if bad_view or road_ratio < float(config["min_road_visible_ratio"]):
                return False, {
                    "reason": (
                        f"bad_view={bad_view}, road_ratio={road_ratio:.4f}, "
                        f"near_ratio={view_stats['near_depth_ratio']:.4f}"
                    ),
                    "saved_frames": frame_index,
                }

            annotations = build_actor_annotations(
                world,
                rgb_data.transform,
                depth_m,
                semantic_id,
                config,
            )
            air_observations = enrich_observations(
                annotations,
                cooperative.object_registry,
                "uav_01_rgb",
            )
            ground_depth = base.decode_carla_depth_meters(
                ground_raw["vehicle_01_depth"]
            )
            ground_semantic = base.decode_semantic_segmentation(
                ground_raw["vehicle_01_semantic"]
            )
            ground_config = dict(config)
            ground_vehicle_config = config["cooperative"]["ground_vehicle"]
            ground_config.update(
                {
                    "width": int(ground_vehicle_config["image_width"]),
                    "height": int(ground_vehicle_config["image_height"]),
                    "fov": float(ground_vehicle_config["camera_fov_deg"]),
                }
            )
            ground_annotations = build_actor_annotations(
                world,
                ground_raw["vehicle_01_rgb"].transform,
                ground_depth,
                ground_semantic,
                ground_config,
            )
            ground_annotations = [
                annotation
                for annotation in ground_annotations
                if int(annotation.get("carla_actor_id", -1))
                != int(cooperative.ground_vehicle.vehicle.id)
            ]
            ground_observations = enrich_observations(
                ground_annotations,
                cooperative.object_registry,
                "vehicle_01_rgb",
            )
            target_annotation = find_target_annotation(
                annotations,
                int(target.id),
                spec.target_class,
                config,
            )
            if frame_index == 0 and target_annotation is None:
                return False, {
                    "reason": "VOT 第一帧主目标不可见或尺寸不足",
                    "saved_frames": 0,
                }

            if target_annotation is None:
                absent_frames += 1
                consecutive_absent += 1
                groundtruth.append("0,0,0,0")
                absence.append("1")
                occlusion.append("0")
                target_states.append("absent_unresolved")
                previous_target_box = None
            else:
                consecutive_absent = 0
                target_state = enrich_target_annotation(target_annotation)
                x, y, width, height = target_annotation["bbox_xywh"]
                groundtruth.append(f"{x},{y},{width},{height}")
                absence.append("0")
                visible_ratio = float(
                    target_annotation["visible_ratio_projected_bbox"]
                )
                occlusion.append("1" if visible_ratio < 0.75 else "0")
                target_sizes.append(annotation_equivalent_side(target_annotation))
                target_visible_ratios.append(visible_ratio)
                target_occlusion_ratios.append(
                    float(target_annotation["occlusion_ratio"])
                )
                target_truncation_ratios.append(
                    float(target_annotation.get("truncation_ratio", 0.0))
                )
                target_states.append(target_state)
                current_box = [float(value) for value in target_annotation["bbox_xywh"]]
                if previous_target_box is not None:
                    previous_center_x = previous_target_box[0] + previous_target_box[2] * 0.5
                    previous_center_y = previous_target_box[1] + previous_target_box[3] * 0.5
                    current_center_x = current_box[0] + current_box[2] * 0.5
                    current_center_y = current_box[1] + current_box[3] * 0.5
                    previous_side = math.sqrt(
                        previous_target_box[2] * previous_target_box[3]
                    )
                    normalized_displacements.append(
                        math.hypot(
                            current_center_x - previous_center_x,
                            current_center_y - previous_center_y,
                        )
                        / max(1.0, previous_side)
                    )
                    previous_area = previous_target_box[2] * previous_target_box[3]
                    current_area = current_box[2] * current_box[3]
                    adjacent_scale_log_changes.append(
                        abs(math.log(max(1.0, current_area) / max(1.0, previous_area)))
                    )
                    previous_box_copy_ious.append(
                        bbox_iou_xywh(previous_target_box, current_box)
                    )
                previous_target_box = current_box

            if consecutive_absent > max_consecutive_absent:
                return False, {
                    "reason": "主目标连续不可见帧过多",
                    "saved_frames": frame_index,
                }

            frame_name = f"{frame_index + 1:08d}"
            image_path = color_dir / f"{frame_name}.png"
            rgb = base.save_rgb(
                rgb_data,
                image_path,
                weather_name=spec.weather,
                depth_m=depth_m,
                random_seed=(
                    int(config["seed"]) * 100000
                    + int(target.id) * 100
                    + frame_index
                ),
            )
            base.save_yolo_label(
                label_dir / f"{frame_name}.txt",
                annotations,
                int(config["width"]),
                int(config["height"]),
            )
            staging = cooperative.writer.current_path
            if staging is None:
                raise RuntimeError("cooperative dataset transaction is not active")
            ground_rgb_path = (
                staging / "platforms/ground/vehicle_01/rgb" / f"{frame_name}.png"
            )
            ground_rgb_path.parent.mkdir(parents=True, exist_ok=True)
            base.save_rgb(
                ground_raw["vehicle_01_rgb"],
                ground_rgb_path,
                weather_name=spec.weather,
                depth_m=ground_depth,
                random_seed=int(config["seed"]) + frame_index,
            )
            depth_path = (
                staging / "platforms/ground/vehicle_01/depth_m" / f"{frame_name}.npy"
            )
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(depth_path, ground_depth.astype(np.float32))
            semantic_path = (
                staging / "platforms/ground/vehicle_01/semantic" / f"{frame_name}.png"
            )
            semantic_path.parent.mkdir(parents=True, exist_ok=True)
            base.save_semantic_id(ground_semantic, semantic_path)
            cooperative.writer.copy_file(
                image_path,
                f"platforms/air/uav_01/rgb/{frame_name}.png",
            )
            air_depth_path = (
                staging / "platforms/air/uav_01/depth_m" / f"{frame_name}.npy"
            )
            air_depth_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(air_depth_path, depth_m.astype(np.float32))
            air_semantic_path = (
                staging / "platforms/air/uav_01/semantic" / f"{frame_name}.png"
            )
            air_semantic_path.parent.mkdir(parents=True, exist_ok=True)
            base.save_semantic_id(semantic_id, air_semantic_path)
            target_identity = cooperative.object_registry.lookup(int(target.id))
            air_target = next(
                (
                    item
                    for item in air_observations
                    if int(item.get("carla_actor_id", -1)) == int(target.id)
                ),
                None,
            )
            ground_target = next(
                (
                    item
                    for item in ground_observations
                    if int(item.get("carla_actor_id", -1)) == int(target.id)
                ),
                None,
            )
            event = target_event_record(
                carla_frame,
                str(target_identity["global_object_uuid"]),
                air_target,
                ground_target,
                previous_cooperative_state,
            )
            previous_cooperative_state = str(event["visibility_state"])
            cooperative_state_counts[previous_cooperative_state] += 1
            if not bool(event["air_visible"]) and not bool(event["vehicle_visible"]):
                current_both_missing_frames += 1
                max_consecutive_both_missing_frames = max(
                    max_consecutive_both_missing_frames,
                    current_both_missing_frames,
                )
            else:
                current_both_missing_frames = 0
            interaction_event_count += len(event["events"])
            reacquisition_count += int("reacquisition_event" in event["events"])
            snapshot = world.get_snapshot()
            all_sensor_data = {
                "uav_01_%s" % name: data
                for name, data in raw.items()
            }
            all_sensor_data.update(ground_raw)
            frame_report = cooperative.record_frame(
                carla_frame,
                float(rgb_data.timestamp),
                float(snapshot.timestamp.elapsed_seconds),
                {
                    "uav_01_rgb": air_observations,
                    "vehicle_01_rgb": ground_observations,
                },
                all_sensor_data,
                ground_raw,
            )
            event.update(
                {
                    "timestamp": float(rgb_data.timestamp),
                    "simulation_time": float(snapshot.timestamp.elapsed_seconds),
                    "sample_index": int(frame_report["sample_index"]),
                    "source_tick": int(carla_frame),
                }
            )
            cooperative.writer.append_jsonl("events/visibility.jsonl", event)
            for annotation in annotations:
                class_counts[annotation["class_name"]] += 1

            gray = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                (320, 180),
                interpolation=cv2.INTER_AREA,
            )
            if previous_gray is not None:
                image_differences.append(
                    float(
                        np.mean(
                            np.abs(
                                gray.astype(np.float32)
                                - previous_gray.astype(np.float32)
                            )
                        )
                    )
                )
            previous_gray = gray
            road_ratios.append(float(road_ratio))
            near_ratios.append(float(view_stats["near_depth_ratio"]))
            write_frame_metadata(
                metadata_file,
                frame_index,
                carla_frame,
                target,
                spec.target_class,
                target_annotation,
                annotations,
                spec.weather,
                rgb_data.transform,
                pose_stats,
                view_stats,
                road_ratio,
            )

            if frame_index in {
                0,
                planned_max_frames // 2,
                planned_max_frames - 1,
            }:
                save_overlay(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    annotations,
                    int(target.id),
                    target_annotation,
                    attempt_dir / f"overlay_{frame_name}.png",
                    (
                        f"{spec.name} frame={frame_index + 1} "
                        f"weather={spec.weather}"
                    ),
                )

    actual_frames = len(groundtruth)
    absent_ratio = absent_frames / float(actual_frames)
    if absent_ratio > max_absent_ratio:
        return False, {
            "reason": f"主目标缺失比例过高：{absent_ratio:.4f}",
            "saved_frames": actual_frames,
        }
    if image_differences and min(image_differences) < 0.05:
        return False, {
            "reason": (
                "出现疑似完全重复的相邻帧："
                f"min_mean_abs_diff={min(image_differences):.4f}"
            ),
            "saved_frames": actual_frames,
        }

    target_path_length_m = float(sum(target_world_step_distances))
    target_motion_duration_seconds = (
        len(target_world_step_distances)
        * sample_interval_ticks
        / float(config["fps"])
    )
    target_average_speed_mps = (
        target_path_length_m / target_motion_duration_seconds
        if target_motion_duration_seconds > 0.0
        else 0.0
    )
    moving_step_threshold_m = float(
        config.get("target_moving_step_threshold_m", {}).get(
            spec.target_class,
            0.05,
        )
    )
    target_moving_step_ratio = (
        sum(
            distance >= moving_step_threshold_m
            for distance in target_world_step_distances
        )
        / float(len(target_world_step_distances))
        if target_world_step_distances
        else 0.0
    )
    min_average_speed_mps = float(
        config.get("min_target_average_speed_mps", {}).get(
            spec.target_class,
            0.0,
        )
    )
    min_moving_step_ratio = float(
        config.get("min_target_moving_step_ratio", {}).get(
            spec.target_class,
            0.0,
        )
    )
    if (
        target_average_speed_mps < min_average_speed_mps
        or target_moving_step_ratio < min_moving_step_ratio
    ):
        return False, {
            "reason": (
                "主目标自身运动不足："
                f"average_speed={target_average_speed_mps:.3f}m/s "
                f"(min={min_average_speed_mps:.3f}), "
                f"moving_step_ratio={target_moving_step_ratio:.3f} "
                f"(min={min_moving_step_ratio:.3f})"
            ),
            "saved_frames": actual_frames,
            "target_path_length_m": target_path_length_m,
            "target_average_speed_mps": target_average_speed_mps,
            "target_moving_step_ratio": target_moving_step_ratio,
        }

    (attempt_dir / "groundtruth.txt").write_text(
        "\n".join(groundtruth) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "absence.label").write_text(
        "\n".join(absence) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "occlusion.label").write_text(
        "\n".join(occlusion) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "target_state.label").write_text(
        "\n".join(target_states) + "\n",
        encoding="utf-8",
    )
    summary = {
        "sequence": spec.name,
        "sequence_id": int(spec.sequence_id),
        "weather": spec.weather,
        "split": spec.split,
        "target_class": spec.target_class,
        "motion_mode": spec.motion_mode,
        "target_actor_id": int(target.id),
        "target_actor_type": target.type_id,
        "frames": actual_frames,
        "planned_max_frames": planned_max_frames,
        "min_frames_required": min_frames,
        "termination_reason": termination_reason,
        "simulation_fps": float(config["fps"]),
        "sample_interval_ticks": sample_interval_ticks,
        "sample_interval_seconds": (
            sample_interval_ticks / float(config["fps"])
        ),
        "simulated_duration_seconds": (
            actual_frames * sample_interval_ticks / float(config["fps"])
        ),
        "max_absent_ratio_allowed": max_absent_ratio,
        "max_consecutive_absent_frames_allowed": max_consecutive_absent,
        "absent_frames": absent_frames,
        "absent_ratio": absent_ratio,
        "target_equivalent_side_px": stats(target_sizes),
        "target_visible_ratio": stats(target_visible_ratios),
        "target_occlusion_ratio": stats(target_occlusion_ratios),
        "target_truncation_ratio": stats(target_truncation_ratios),
        "target_state_counts": dict(Counter(target_states)),
        "normalized_center_displacement": stats(normalized_displacements),
        "adjacent_scale_log_change": stats(adjacent_scale_log_changes),
        "previous_frame_box_copy_iou": stats(previous_box_copy_ious),
        "target_world_step_distance_m": stats(target_world_step_distances),
        "target_path_length_m": target_path_length_m,
        "target_average_speed_mps": target_average_speed_mps,
        "target_moving_step_ratio": target_moving_step_ratio,
        "target_moving_step_threshold_m": moving_step_threshold_m,
        "road_visible_ratio": stats(road_ratios),
        "near_depth_ratio": stats(near_ratios),
        "adjacent_frame_mean_abs_difference": stats(image_differences),
        "yolo_objects": dict(class_counts),
        "vot_region_format": "x,y,width,height; absent frames are 0,0,0,0",
        "public_modalities": ["RGB"],
        "auxiliary_modalities_saved_for_qa": ["depth", "semantic"],
        "cooperative_visibility_state_counts": dict(cooperative_state_counts),
        "joint_visible_ratio": float(
            cooperative_state_counts["Joint Visible"] / max(1, actual_frames)
        ),
        "uav_dominant_ratio": float(
            cooperative_state_counts["UAV Dominant"] / max(1, actual_frames)
        ),
        "vehicle_dominant_ratio": float(
            cooperative_state_counts["Vehicle Dominant"] / max(1, actual_frames)
        ),
        "ground_target_visible_ratio": float(
            (
                cooperative_state_counts["Joint Visible"]
                + cooperative_state_counts["Vehicle Dominant"]
            )
            / max(1, actual_frames)
        ),
        "max_consecutive_both_missing_frames": int(
            max_consecutive_both_missing_frames
        ),
        "interaction_event_count": int(interaction_event_count),
        "reacquisition_count": int(reacquisition_count),
        "cooperative_threshold_status": "pilot_provisional",
    }
    (attempt_dir / "sequence_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    minimum_ground_visible_ratio = float(
        cooperative.config.get("quality", {}).get(
            "min_ground_target_visible_ratio", 0.10
        )
    )
    if summary["ground_target_visible_ratio"] < minimum_ground_visible_ratio:
        return False, {
            **summary,
            "reason": (
                "ground target visibility below cooperative acceptance gate: "
                f"{summary['ground_target_visible_ratio']:.3f} < "
                f"{minimum_ground_visible_ratio:.3f}"
            ),
        }
    return True, summary


def stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def finalize_sequence(
    attempt_dir: Path,
    destination: Path,
    qa_root: Path,
    spec: SequenceSpec,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"序列目录已存在：{destination}")
    attempt_dir.replace(destination)
    for overlay in sorted(destination.glob("overlay_*.png")):
        overlay.replace(qa_root / f"{spec.name}_{overlay.name}")


def nonempty_line_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_sequence_checkpoint(
    sequence_dir: Path,
    spec: SequenceSpec,
    config: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load one fully committed sequence or explain why it must be retried."""
    meta_path = sequence_dir / "sequence_meta.json"
    try:
        summary = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"sequence_meta unreadable: {exc}"

    expected_identity = {
        "sequence": spec.name,
        "sequence_id": int(spec.sequence_id),
        "weather": spec.weather,
        "split": spec.split,
        "target_class": spec.target_class,
        "motion_mode": spec.motion_mode,
    }
    mismatches = [
        key
        for key, expected in expected_identity.items()
        if summary.get(key) != expected
    ]
    if mismatches:
        return None, "metadata mismatch: " + ", ".join(mismatches)

    try:
        frame_count = int(summary["frames"])
    except (KeyError, TypeError, ValueError):
        return None, "invalid frame count in sequence_meta"
    min_frames = max(1, int(config.get("minimum_track_length", 1)))
    if not min_frames <= frame_count <= frames_for_sequence(spec, config):
        return None, f"frame count outside valid range: {frame_count}"

    required_line_files = (
        "groundtruth.txt",
        "absence.label",
        "occlusion.label",
        "target_state.label",
        "annotations.jsonl",
    )
    try:
        counts = {
            "color": len(list((sequence_dir / "color").glob("*.png"))),
            "labels_yolo": len(
                list((sequence_dir / "labels_yolo").glob("*.txt"))
            ),
            **{
                name: nonempty_line_count(sequence_dir / name)
                for name in required_line_files
            },
        }
    except OSError as exc:
        return None, f"checkpoint file unreadable: {exc}"
    if any(count != frame_count for count in counts.values()):
        return None, f"checkpoint count mismatch: {counts}"
    return summary, "complete"


def load_resume_checkpoints(
    root: Path,
    specs: Sequence[SequenceSpec],
    config: Dict[str, Any],
) -> Tuple[List[SequenceSpec], List[Dict[str, Any]]]:
    vot_root = root / "vot"
    specs_by_name = {spec.name: spec for spec in specs}
    completed_specs: List[SequenceSpec] = []
    summaries: List[Dict[str, Any]] = []
    rejected_root = root / "_resume_rejected"

    for sequence_dir in sorted(
        path for path in vot_root.iterdir() if path.is_dir()
    ):
        spec = specs_by_name.get(sequence_dir.name)
        if spec is None:
            reason = "not present in current sequence plan"
            summary = None
        else:
            summary, reason = load_sequence_checkpoint(
                sequence_dir,
                spec,
                config,
            )
        if summary is not None and spec is not None:
            completed_specs.append(spec)
            summaries.append(summary)
            continue

        rejected_root.mkdir(parents=True, exist_ok=True)
        destination = rejected_root / sequence_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        sequence_dir.replace(destination)
        print(
            f"[RESUME] Quarantined incomplete sequence "
            f"{sequence_dir.name}: {reason}",
            flush=True,
        )

    order = {spec.name: index for index, spec in enumerate(specs)}
    paired = sorted(
        zip(completed_specs, summaries),
        key=lambda item: order[item[0].name],
    )
    return (
        [item[0] for item in paired],
        [item[1] for item in paired],
    )


def write_resume_state(
    root: Path,
    specs: Sequence[SequenceSpec],
    completed_specs: Sequence[SequenceSpec],
    sequence_summaries: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
    status: str,
) -> None:
    completed_names = {spec.name for spec in completed_specs}
    next_spec = next(
        (spec.name for spec in specs if spec.name not in completed_names),
        None,
    )
    payload = {
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "planned_sequences": len(specs),
        "completed_sequences": len(completed_specs),
        "completed_sequence_names": [spec.name for spec in completed_specs],
        "next_sequence": next_spec,
        "sequence_summaries": list(sequence_summaries),
        "failed_attempts": list(failures),
    }
    destination = root / "resume_state.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def build_yolo_dataset(
    root: Path,
    specs: Sequence[SequenceSpec],
) -> Dict[str, Any]:
    yolo_root = root / "yolo"
    methods: Counter = Counter()
    split_counts: Dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    split_sequences: Dict[str, List[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for spec in specs:
        sequence_dir = root / "vot" / spec.name
        split_sequences[spec.split].append(spec.name)
        image_paths = sorted((sequence_dir / "color").glob("*.png"))
        for image_path in image_paths:
            stem = f"{spec.name}_{image_path.stem}"
            label_path = sequence_dir / "labels_yolo" / f"{image_path.stem}.txt"
            methods[
                hardlink_or_copy(
                    image_path,
                    yolo_root / "images" / spec.split / f"{stem}.png",
                )
            ] += 1
            methods[
                hardlink_or_copy(
                    label_path,
                    yolo_root / "labels" / spec.split / f"{stem}.txt",
                )
            ] += 1
            split_counts[spec.split]["images"] += 1
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    split_counts[spec.split][CLASS_NAMES[class_id]] += 1

    data_yaml = (
        f"path: {yolo_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: vehicle\n"
        "  1: pedestrian\n"
    )
    (yolo_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, names in split_sequences.items():
        (split_dir / f"{split}_sequences.txt").write_text(
            "\n".join(names) + "\n",
            encoding="utf-8",
        )
    return {
        "data_yaml": str((yolo_root / "data.yaml").resolve()),
        "staging_methods": dict(methods),
        "splits": {
            split: dict(counts)
            for split, counts in split_counts.items()
        },
        "split_sequences": split_sequences,
    }


def audit_dataset(
    root: Path,
    specs: Sequence[SequenceSpec],
    config: Dict[str, Any],
    sequence_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    errors: List[str] = []
    summaries_by_sequence = {
        str(summary["sequence"]): summary
        for summary in sequence_summaries
    }
    for spec in specs:
        summary = summaries_by_sequence.get(spec.name)
        if summary is None:
            errors.append(f"{spec.name}: missing sequence summary")
            continue
        expected_frames = int(summary["frames"])
        sequence_dir = root / "vot" / spec.name
        images = sorted((sequence_dir / "color").glob("*.png"))
        labels = sorted((sequence_dir / "labels_yolo").glob("*.txt"))
        gt_lines = [
            line
            for line in (sequence_dir / "groundtruth.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        absence_lines = [
            line
            for line in (sequence_dir / "absence.label")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        state_lines = [
            line
            for line in (sequence_dir / "target_state.label")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        counts = (
            len(images),
            len(labels),
            len(gt_lines),
            len(absence_lines),
            len(state_lines),
        )
        if any(count != expected_frames for count in counts):
            errors.append(f"{spec.name}: count mismatch {counts}")
        for line in gt_lines:
            values = [float(value) for value in line.split(",")]
            if len(values) != 4:
                errors.append(f"{spec.name}: invalid VOT line {line}")
                break
            x, y, width, height = values
            if width == 0 and height == 0:
                continue
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > int(config["width"])
                or y + height > int(config["height"])
            ):
                errors.append(f"{spec.name}: out-of-range VOT box {line}")
                break

    absent_ratios = [
        float(summary["absent_ratio"])
        for summary in sequence_summaries
    ]
    target_medians = [
        float(summary["target_equivalent_side_px"]["median"])
        for summary in sequence_summaries
    ]
    actor_sequence_counts: Counter = Counter()
    actor_splits: Dict[int, set] = defaultdict(set)
    motion_target_combinations: Counter = Counter()
    for summary in sequence_summaries:
        actor_id = int(summary["target_actor_id"])
        actor_sequence_counts[actor_id] += 1
        actor_splits[actor_id].add(str(summary["split"]))
        motion_target_combinations[
            (str(summary["motion_mode"]), str(summary["target_class"]))
        ] += 1
    cross_split_actors = {
        actor_id: sorted(splits)
        for actor_id, splits in actor_splits.items()
        if len(splits) > 1
    }
    if cross_split_actors:
        errors.append(
            "target actors cross train/val/test splits: "
            + json.dumps(cross_split_actors, ensure_ascii=False)
        )
    max_sequences_per_actor = int(
        config.get("max_sequences_per_target_actor", 2)
    )
    overused_actors = {
        actor_id: count
        for actor_id, count in actor_sequence_counts.items()
        if count > max_sequences_per_actor
    }
    if overused_actors:
        errors.append(
            "target actors exceed per-actor sequence cap: "
            + json.dumps(overused_actors, ensure_ascii=False)
        )
    expected_combinations = {
        (spec.motion_mode, spec.target_class)
        for spec in specs
    }
    missing_combinations = sorted(
        expected_combinations - set(motion_target_combinations)
    )
    if missing_combinations:
        errors.append(
            f"missing motion/target combinations: {missing_combinations}"
        )
    covered_cooperative_states = {
        state
        for summary in sequence_summaries
        for state, count in summary.get(
            "cooperative_visibility_state_counts", {}
        ).items()
        if int(count) > 0
    }
    required_cooperative_states = {
        "Joint Visible", "UAV Dominant", "Vehicle Dominant"
    }
    missing_cooperative_states = sorted(
        required_cooperative_states - covered_cooperative_states
    )
    if missing_cooperative_states:
        errors.append(
            "missing cooperative visibility states: "
            + json.dumps(missing_cooperative_states, ensure_ascii=False)
        )
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "sequence_count": len(specs),
        "actual_total_frames": sum(
            int(summary["frames"]) for summary in sequence_summaries
        ),
        "planned_max_total_frames": sum(
            frames_for_sequence(spec, config) for spec in specs
        ),
        "sequence_level_split": True,
        "weather_is_constant_inside_each_sequence": True,
        "cooperative_visibility_states": sorted(covered_cooperative_states),
        "target_actor_is_constant_inside_each_sequence": True,
        "target_actor_is_split_isolated": not cross_split_actors,
        "max_sequences_per_target_actor": max_sequences_per_actor,
        "unique_target_actors": len(actor_sequence_counts),
        "target_actor_sequence_count": stats(
            list(actor_sequence_counts.values())
        ),
        "motion_target_combinations": {
            f"{mode}/{target_class}": count
            for (mode, target_class), count in sorted(
                motion_target_combinations.items()
            )
        },
        "absent_ratio": stats(absent_ratios),
        "target_equivalent_side_median_px": stats(target_medians),
        "vot_files_checked": [
            "color/*.png",
            "groundtruth.txt",
            "absence.label",
            "occlusion.label",
            "target_state.label",
        ],
    }
    (root / "quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(
            "数据集质量审计失败：\n" + "\n".join(errors[:20])
        )
    return audit


def write_dataset_card(
    root: Path,
    config: Dict[str, Any],
    specs: Sequence[SequenceSpec],
    yolo_summary: Dict[str, Any],
    audit: Dict[str, Any],
) -> None:
    text = f"""# OpenHUTB UAV RGB Single-Object Tracking Dataset

## 数据内容

- 公开模态：RGB。
- 主任务：VOT 风格单目标追踪。
- 辅助任务：YOLO 车辆/行人检测。
- 分辨率：{config['width']}x{config['height']}。
- 仿真帧率：{config['fps']} FPS。
- 有效采样间隔：车辆每
  {config.get('sample_interval_ticks_by_target_class', {}).get('vehicle', config['sample_interval_ticks'])}
  个 CARLA 帧保存一次，人物每
  {config.get('sample_interval_ticks_by_target_class', {}).get('pedestrian', config['sample_interval_ticks'])}
  个 CARLA 帧保存一次。
- 序列数：{len(specs)}。
- 实际总帧数：{audit['actual_total_frames']}。
- 计划最大帧数：{audit['planned_max_total_frames']}。
- 类别：vehicle、pedestrian。

## VOT 格式

每条序列位于 `vot/<sequence_name>/`：

- `color/00000001.png`：连续 RGB 帧；
- `groundtruth.txt`：每行 `x,y,width,height`；
- `absence.label`：主目标不可见时为 1；
- `occlusion.label`：主目标明显遮挡时为 1；
- `target_state.label`：逐帧目标状态；
- `sequence_meta.json`：目标 actor、天气和质量统计；
- `annotations.jsonl`：逐帧完整标注；
- `labels_yolo/`：画面中全部车辆和行人的 YOLO 标签。

目标低于最低可见率或尺寸阈值时不写主目标框，对应 VOT 行写为
`0,0,0,0`。逐帧 JSON 同时记录可见率、遮挡率和截断率。

## 数据划分

训练、验证、测试按完整序列划分，禁止相邻帧跨集合。同一个目标 actor
只能属于一个集合，且最多拍摄
{config.get('max_sequences_per_target_actor', 2)} 条序列。三种拍摄方式均覆盖
车辆和人物。天气在每条序列内保持不变。YOLO 配置文件为 `yolo/data.yaml`。

质量审计：{audit['status']}。
YOLO 图像数：{json.dumps(yolo_summary['splits'], ensure_ascii=False)}。
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args)
    validate_config(config)
    specs = build_sequence_specs(config)
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise ValueError("--max-sequences 必须大于 0")
        specs = specs[: args.max_sequences]
    paths = prepare_output(
        Path(config["out"]),
        args.overwrite,
        resume=args.resume,
    )
    rng = random.Random(int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    simulator_session = ensure_simulator(
        carla,
        str(config["host"]),
        int(config["port"]),
        config.get("map"),
        dict(config.get("simulator", {})),
    )
    client = carla.Client(str(config["host"]), int(config["port"]))
    client.set_timeout(float(config["timeout"]))
    world = (
        client.get_world()
        if config.get("map") in (None, "", "current")
        else client.load_world(str(config["map"]))
    )
    original_settings = world.get_settings()
    original_weather = world.get_weather()
    sensors: Dict[str, Any] = {}
    sensor_sync: Dict[str, Any] = {}
    vehicles: List[Any] = []
    walkers: List[Any] = []
    controllers: List[Any] = []
    hidden_static_ids: List[int] = []
    used_actor_ids: Counter = Counter()
    actor_split_assignments: Dict[int, str] = {}
    accepted_specs: List[SequenceSpec] = []
    sequence_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    population_refresh_history: List[Dict[str, Any]] = []
    target_vehicles: List[Any] = []
    target_walkers: List[Any] = []
    cooperative = CooperativeManager(
        client,
        world,
        carla,
        Path(config["cooperative"]["output_root"]),
        "task2",
        config["cooperative"],
    )

    if args.resume:
        accepted_specs, sequence_summaries = load_resume_checkpoints(
            paths["root"],
            specs,
            config,
        )
        print(
            f"[RESUME] Preserved {len(accepted_specs)}/{len(specs)} "
            "completed sequences.",
            flush=True,
        )
        for summary in sequence_summaries:
            actor_id = int(summary["target_actor_id"])
            split = str(summary["split"])
            existing_split = actor_split_assignments.get(actor_id)
            if existing_split is not None and existing_split != split:
                raise RuntimeError(
                    f"断点中的目标 actor={actor_id} 同时属于 "
                    f"{existing_split} 和 {split}"
                )
            actor_split_assignments[actor_id] = split
            used_actor_ids[actor_id] += 1
    write_resume_state(
        paths["root"],
        specs,
        accepted_specs,
        sequence_summaries,
        failures,
        "in_progress",
    )

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(config["fps"])
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager = client.get_trafficmanager(int(config["tm_port"]))
        traffic_manager.set_synchronous_mode(True)

        base.destroy_live_two_wheel_vehicles(world)
        if bool(config["hide_static_map_vehicles"]):
            hidden_static_ids = base.hide_static_map_vehicles(world)
        vehicles, walkers, controllers = base.spawn_background_traffic(
            client,
            world,
            int(config["vehicles"]),
            int(config["walkers"]),
            int(config["tm_port"]),
            int(config["seed"]),
        )
        base.destroy_live_two_wheel_vehicles(world)

        target_vehicles = list(vehicles)
        target_walkers = list(walkers)
        include_existing_any = bool(
            config.get("include_existing_target_actors", False)
        )
        if bool(
            config.get(
                "include_existing_target_vehicle_actors",
                include_existing_any,
            )
        ):
            world_actors = world.get_actors()
            target_vehicles.extend(list(world_actors.filter("vehicle.*")))
        if bool(
            config.get(
                "include_existing_target_pedestrian_actors",
                include_existing_any,
            )
        ):
            world_actors = world.get_actors()
            target_walkers.extend(
                list(world_actors.filter("walker.pedestrian.*"))
            )
        target_vehicles = list(
            {
                int(actor.id): actor
                for actor in target_vehicles
                if (
                    actor is not None
                    and actor.is_alive
                    and not is_static_bridge_actor(actor)
                )
            }.values()
        )
        target_walkers = list(
            {
                int(actor.id): actor
                for actor in target_walkers
                if (
                    actor is not None
                    and actor.is_alive
                    and not is_static_bridge_actor(actor)
                )
            }.values()
        )

        initial_transform = None
        initial_errors: List[str] = []
        # The initial pose only exists to spawn the reusable sensor rig.
        for initial_target in (target_vehicles + target_walkers)[:80]:
            try:
                initial_transform, _ = rear_upper_follow_transform(
                    initial_target,
                    config,
                    {"dt_seconds": 1.0 / float(config["fps"])},
                )
                break
            except RuntimeError as exc:
                initial_errors.append(str(exc))
        if initial_transform is None:
            raise RuntimeError(
                "无法从现有目标生成初始相机位姿："
                + "; ".join(initial_errors[:5])
            )
        sensors = spawn_internal_sensors(world, initial_transform, config)
        sensor_sync = {
            name: base.SensorSync(name, sensor)
            for name, sensor in sensors.items()
        }
        for _ in range(int(config["warmup_frames"])):
            world.tick()
        for item in sensor_sync.values():
            item.drain()

        completed_names = {spec.name for spec in accepted_specs}
        for sequence_index, spec in enumerate(specs):
            if spec.name in completed_names:
                print(
                    f"[RESUME {sequence_index + 1}/{len(specs)}] "
                    f"skip completed {spec.name}",
                    flush=True,
                )
                continue
            print(
                f"[SEQ {sequence_index + 1}/{len(specs)}] "
                f"{spec.name} mode={spec.motion_mode} split={spec.split}"
            )
            base.apply_weather(world, spec.weather)
            for _ in range(int(config["weather_warmup_frames"])):
                world.tick()
            for item in sensor_sync.values():
                item.drain()

            accepted = False
            population_round = 0
            population_refreshes_used = 0
            max_refreshes = int(
                config.get("max_target_population_refreshes", 2)
            )
            min_candidates = int(
                config.get("min_eligible_target_candidates", 2)
            )
            max_attempts = int(config["max_sequence_attempts"])
            last_failure_reason = "未找到可用目标"
            while not accepted:
                actors = (
                    target_vehicles
                    if spec.target_class == "vehicle"
                    else target_walkers
                )
                eligible = eligible_target_actors(
                    actors,
                    spec.target_class,
                    used_actor_ids,
                    actor_split_assignments,
                    spec.split,
                    int(config.get("max_sequences_per_target_actor", 2)),
                )
                if (
                    len(eligible) < min_candidates
                    and population_refreshes_used < max_refreshes
                ):
                    refresh = refresh_target_population(
                        client,
                        world,
                        sensor_sync,
                        spec.target_class,
                        sequence_index * (max_refreshes + 1)
                        + population_refreshes_used,
                        config,
                        vehicles,
                        walkers,
                        controllers,
                        target_vehicles,
                        target_walkers,
                    )
                    population_refreshes_used += 1
                    population_refresh_history.append(
                        {
                            "sequence": spec.name,
                            "trigger": "eligible_pool_below_minimum",
                            "eligible_before": len(eligible),
                            "refresh_number": population_refreshes_used,
                            **refresh,
                        }
                    )
                    print(
                        f"[REFRESH-TARGETS] {spec.name} "
                        f"class={spec.target_class} "
                        f"refresh={population_refreshes_used}/{max_refreshes} "
                        f"eligible={len(eligible)} spawned={refresh['spawned']} "
                        f"active={refresh['active_after']} "
                        f"reason={refresh['reason']}",
                        flush=True,
                    )
                    if int(refresh["spawned"]) > 0:
                        population_round += 1
                        continue

                eligible = eligible_target_actors(
                    actors,
                    spec.target_class,
                    used_actor_ids,
                    actor_split_assignments,
                    spec.split,
                    int(config.get("max_sequences_per_target_actor", 2)),
                )
                if not eligible:
                    last_failure_reason = (
                        f"{spec.split}/{spec.target_class} 没有剩余合格目标"
                    )
                    break

                attempted_actor_ids: Counter = Counter()
                for local_attempt in range(max_attempts):
                    attempt = population_round * max_attempts + local_attempt
                    try:
                        target = choose_target_actor(
                            actors,
                            spec.target_class,
                            used_actor_ids,
                            actor_split_assignments,
                            spec.split,
                            int(config.get("max_sequences_per_target_actor", 2)),
                            attempted_actor_ids,
                            rng,
                        )
                    except RuntimeError as exc:
                        last_failure_reason = str(exc)
                        break
                    direction = "target_heading"
                    attempt_dir = paths["tmp"] / (
                        f"{spec.name}_attempt_{attempt:03d}"
                    )
                    if attempt_dir.exists():
                        shutil.rmtree(attempt_dir)
                    attempt_dir.mkdir(parents=True)
                    scene_seed = (
                        int(config["seed"])
                        + int(spec.sequence_id) * 100_003
                        + int(attempt)
                    )
                    cooperative.prepare_scene(
                        spec.name,
                        scene_seed,
                        str(cooperative.config["route_profile"]),
                        actors=list(target_vehicles) + list(target_walkers),
                        target_actor=target,
                    )
                    cooperative.register_air_platform(
                        "uav_01",
                        virtual=True,
                        platform_type="air_camera_platform",
                        sensors=sensors,
                        required_modalities=("rgb", "depth", "semantic"),
                    )
                    try:
                        success, summary = collect_sequence_attempt(
                            world,
                            sensors,
                            sensor_sync,
                            target,
                            spec,
                            attempt_dir,
                            direction,
                            config,
                            cooperative,
                        )
                    except Exception:
                        cooperative.close_scene(
                            abort_reason="sequence_attempt_exception"
                        )
                        raise
                    if success:
                        target_actor_id = int(target.id)
                        assigned_split = actor_split_assignments.get(
                            target_actor_id
                        )
                        if (
                            assigned_split is not None
                            and assigned_split != spec.split
                        ):
                            raise RuntimeError(
                                f"目标 actor={target_actor_id} 不允许从 "
                                f"{assigned_split} 跨到 {spec.split}"
                            )
                        actor_split_assignments[target_actor_id] = spec.split
                        used_actor_ids[target_actor_id] += 1
                        summary["attempt"] = attempt
                        summary["target_population_round"] = population_round
                        summary["target_population_refreshes_used"] = (
                            population_refreshes_used
                        )
                        summary["camera_road_direction"] = direction
                        (attempt_dir / "sequence_meta.json").write_text(
                            json.dumps(summary, ensure_ascii=False, indent=2)
                            + "\n",
                            encoding="utf-8",
                        )
                        finalize_sequence(
                            attempt_dir,
                            paths["vot"] / spec.name,
                            paths["qa"],
                            spec,
                        )
                        cooperative.commit_scene(
                            {
                                "split": spec.split,
                                "motion_mode": "rear_upper_follow",
                                "target_class": spec.target_class,
                                "frames": int(summary["frames"]),
                                "derived_format": "VOT/SOT",
                                "parameter_status": "pilot_provisional",
                                "joint_visible_ratio": summary["joint_visible_ratio"],
                                "uav_dominant_ratio": summary["uav_dominant_ratio"],
                                "vehicle_dominant_ratio": summary[
                                    "vehicle_dominant_ratio"
                                ],
                                "ground_target_visible_ratio": summary[
                                    "ground_target_visible_ratio"
                                ],
                                "max_consecutive_both_missing_frames": summary[
                                    "max_consecutive_both_missing_frames"
                                ],
                                "interaction_event_count": summary[
                                    "interaction_event_count"
                                ],
                                "reacquisition_count": summary[
                                    "reacquisition_count"
                                ],
                            }
                        )
                        accepted_specs.append(spec)
                        sequence_summaries.append(summary)
                        completed_names.add(spec.name)
                        write_resume_state(
                            paths["root"],
                            specs,
                            accepted_specs,
                            sequence_summaries,
                            failures,
                            "in_progress",
                        )
                        accepted = True
                        print(
                            f"[ACCEPT] actor={target.id} "
                            f"absent={summary['absent_ratio']:.3f} "
                            "target_eq_median="
                            f"{summary['target_equivalent_side_px']['median']:.1f}px"
                        )
                        break
                    last_failure_reason = str(summary["reason"])
                    cooperative.close_scene(
                        abort_reason="sequence_attempt_rejected: "
                        + last_failure_reason
                    )
                    failures.append(
                        {
                            "sequence": spec.name,
                            "attempt": attempt,
                            "target_population_round": population_round,
                            "target_actor_id": int(target.id),
                            **summary,
                        }
                    )
                    print(
                        f"[RETRY] {spec.name} attempt={attempt + 1}: "
                        f"{summary['reason']}"
                    )
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                if accepted:
                    break
                if population_refreshes_used >= max_refreshes:
                    break

                refresh = refresh_target_population(
                    client,
                    world,
                    sensor_sync,
                    spec.target_class,
                    sequence_index * (max_refreshes + 1)
                    + population_refreshes_used,
                    config,
                    vehicles,
                    walkers,
                    controllers,
                    target_vehicles,
                    target_walkers,
                )
                population_refreshes_used += 1
                population_refresh_history.append(
                    {
                        "sequence": spec.name,
                        "trigger": "attempt_round_exhausted",
                        "refresh_number": population_refreshes_used,
                        **refresh,
                    }
                )
                print(
                    f"[REFRESH-TARGETS] {spec.name} "
                    f"class={spec.target_class} "
                    f"refresh={population_refreshes_used}/{max_refreshes} "
                    f"spawned={refresh['spawned']} "
                    f"active={refresh['active_after']} "
                    f"reason={refresh['reason']}",
                    flush=True,
                )
                if int(refresh["spawned"]) <= 0:
                    last_failure_reason = str(refresh["reason"])
                    break
                population_round += 1
            if not accepted:
                write_resume_state(
                    paths["root"],
                    specs,
                    accepted_specs,
                    sequence_summaries,
                    failures,
                    "sequence_failed",
                )
                raise RuntimeError(
                    f"序列 {spec.name} 在目标池刷新上限内仍未通过："
                    f"{last_failure_reason}"
                )

        summaries_by_name = {
            str(summary["sequence"]): summary
            for summary in sequence_summaries
        }
        accepted_specs = [
            spec for spec in specs if spec.name in completed_names
        ]
        sequence_summaries = [
            summaries_by_name[spec.name] for spec in accepted_specs
        ]
        (paths["vot"] / "list.txt").write_text(
            "\n".join(spec.name for spec in accepted_specs) + "\n",
            encoding="utf-8",
        )
        yolo_summary = build_yolo_dataset(paths["root"], accepted_specs)
        audit = audit_dataset(
            paths["root"],
            accepted_specs,
            config,
            sequence_summaries,
        )
        manifest = {
            "collector": str(Path(__file__).resolve()),
            "config": config,
            "map": world.get_map().name,
            "classes": CLASS_NAMES,
            "sequence_count": len(accepted_specs),
            "total_frames": sum(
                int(summary["frames"]) for summary in sequence_summaries
            ),
            "planned_max_total_frames": sum(
                frames_for_sequence(spec, config) for spec in accepted_specs
            ),
            "sequence_summaries": sequence_summaries,
            "failed_attempts": failures,
            "target_population_refreshes": population_refresh_history,
            "yolo": yolo_summary,
            "quality_audit": audit,
        }
        (paths["root"] / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(paths["tmp"], ignore_errors=True)
        write_dataset_card(
            paths["root"],
            config,
            accepted_specs,
            yolo_summary,
            audit,
        )
        write_resume_state(
            paths["root"],
            specs,
            accepted_specs,
            sequence_summaries,
            failures,
            "complete",
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"[DONE] {paths['root']}")
    finally:
        cooperative.close()
        destroy_actors(sensors.values())
        destroy_traffic_population(
            world,
            controllers,
            walkers,
            vehicles,
        )
        try:
            client.get_trafficmanager(int(config["tm_port"])).set_synchronous_mode(
                False
            )
        except RuntimeError:
            pass
        try:
            base.restore_static_map_vehicles(world, hidden_static_ids)
        except RuntimeError:
            pass
        try:
            world.set_weather(original_weather)
            world.apply_settings(original_settings)
        except RuntimeError:
            pass
        close_simulator(simulator_session)


if __name__ == "__main__":
    main()
