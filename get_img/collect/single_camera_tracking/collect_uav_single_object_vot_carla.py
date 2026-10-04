#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 OpenHUTB/CARLA 中采集 RGB 单目标追踪数据。。

公开数据采用 VOT 矩形框格式；内部深度和语义相机仅用于遮挡、穿模与
路面质量检查，不保存为多模态数据。YOLO 标签会标出画面中全部合格车辆
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

COLLECT_ROOT = Path(__file__).resolve().parent.parent
MULTIMODAL_DIR = COLLECT_ROOT / "multimodal"
if str(MULTIMODAL_DIR) not in sys.path:
    sys.path.insert(0, str(MULTIMODAL_DIR))

import collect_rpg_small_targets_carla_v2 as base


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
        modes = list(config.get("motion_modes", []))
        if modes:
            quotient, remainder = divmod(args.sequences_per_map, len(modes))
            config["sequences_by_motion_mode"] = {
                mode: quotient + (1 if index < remainder else 0)
                for index, mode in enumerate(modes)
            }
    if args.frames_per_sequence is not None:
        config["frames_per_sequence"] = args.frames_per_sequence
        config["frames_by_motion_mode"] = {
            mode: args.frames_per_sequence
            for mode in config.get("motion_modes", [])
        }
    if args.weather_presets is not None:
        config["weather_presets"] = args.weather_presets
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
    if (
        int(config["vehicles"]) == 0
        and int(config["walkers"]) == 0
        and not bool(config.get("include_existing_target_actors", False))
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
    if weather_assignment not in {"cycle", "random", "random_unseeded"}:
        raise ValueError(
            "weather_assignment 只能是 cycle、random 或 random_unseeded"
        )
    motion_modes = list(config.get("motion_modes", []))
    if not motion_modes:
        raise ValueError("motion_modes 不能为空")
    supported_modes = {"fixed_hover", "lagged_follow", "lateral_orbit"}
    unknown_modes = sorted(set(motion_modes) - supported_modes)
    if unknown_modes:
        raise ValueError(f"不支持的运动模式：{unknown_modes}")
    configured_count = sum(
        int(config.get("sequences_by_motion_mode", {}).get(mode, 0))
        for mode in motion_modes
    )
    if configured_count != int(config["sequences_per_map"]):
        raise ValueError(
            "sequences_per_map 必须等于 sequences_by_motion_mode 的总和"
        )
    interval_by_class = config.get("sample_interval_ticks_by_target_class", {})
    for target_class in ("vehicle", "pedestrian"):
        if int(interval_by_class.get(target_class, config["sample_interval_ticks"])) <= 0:
            raise ValueError(f"{target_class} 的采样 tick 间隔必须大于 0")
    for mode in motion_modes:
        if int(config.get("frames_by_motion_mode", {}).get(mode, 0)) <= 0:
            raise ValueError(f"{mode} 的最大帧数必须大于 0")
    for mode, class_frames in config.get(
        "frames_by_motion_mode_and_target_class", {}
    ).items():
        if mode not in supported_modes:
            raise ValueError(f"类别专用帧数包含未知模式：{mode}")
        for target_class, frame_count in class_frames.items():
            if target_class not in {"vehicle", "pedestrian"}:
                raise ValueError(f"类别专用帧数包含未知类别：{target_class}")
            if int(frame_count) <= 0:
                raise ValueError(f"{mode}/{target_class} 的帧数必须大于 0")


def frames_for_sequence(spec: SequenceSpec, config: Dict[str, Any]) -> int:
    class_specific = (
        config.get("frames_by_motion_mode_and_target_class", {})
        .get(spec.motion_mode, {})
        .get(spec.target_class)
    )
    if class_specific is not None:
        return int(class_specific)
    return int(
        config.get("frames_by_motion_mode", {}).get(
            spec.motion_mode,
            config["frames_per_sequence"],
        )
    )


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
    return float(
        config.get("max_absent_ratio_by_motion_mode", {}).get(
            spec.motion_mode,
            config["max_absent_ratio_per_sequence"],
        )
    )


def max_consecutive_absent_for_sequence(
    spec: SequenceSpec,
    config: Dict[str, Any],
) -> int:
    return int(
        config.get("max_consecutive_absent_frames_by_motion_mode", {}).get(
            spec.motion_mode,
            config["max_consecutive_absent_frames"],
        )
    )


def prepare_output(root: Path, overwrite: bool) -> Dict[str, Path]:
    root = root.resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在：{root}\n"
                "为避免混入旧追踪帧，换一个 out，或明确使用 --overwrite。"
            )
        safe_name = str(root.parent / root.name).lower()
        if "dataset_uav" not in safe_name or "single_object_vot" not in safe_name:
            raise RuntimeError(f"拒绝覆盖名称异常的目录：{root}")
        shutil.rmtree(root)
    paths = {
        "root": root,
        "vot": root / "vot",
        "yolo": root / "yolo",
        "qa": root / "qa_overlay",
        "tmp": root / "_sequence_staging",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_sequence_specs(config: Dict[str, Any]) -> List[SequenceSpec]:
    """Build the configured number of sequences for every motion mode."""
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
    motion_modes = list(config["motion_modes"])
    mode_counts = config["sequences_by_motion_mode"]
    max_mode_count = max(int(mode_counts[mode]) for mode in motion_modes)
    mode_schedule = [
        mode
        for round_index in range(max_mode_count)
        for mode in motion_modes
        if round_index < int(mode_counts[mode])
    ]
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
    for sequence_id, motion_mode in enumerate(mode_schedule):
        weather = (
            configured_weather_schedule[sequence_id]
            if configured_weather_schedule
            else weather_rng.choice(weathers)
            if weather_assignment in {"random", "random_unseeded"}
            else weathers[sequence_id % len(weathers)]
        )
        index_in_weather = int(weather_occurrences[weather])
        weather_occurrences[weather] += 1
        if motion_mode == "fixed_hover":
            target_class = "vehicle"
        elif motion_mode == "lagged_follow":
            target_class = "pedestrian"
        else:
            target_class = "vehicle" if class_offset % 2 == 0 else "pedestrian"
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
) -> Tuple[int, Dict[str, Any]]:
    """Advance one simulation tick and consume every synchronized modality."""
    carla_frame = int(world.tick())
    return carla_frame, {
        name: sync.get(carla_frame, timeout=timeout)
        for name, sync in sensor_sync.items()
    }


def destroy_actors(actors: Iterable[Any]) -> None:
    for actor in actors:
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except RuntimeError:
            pass
        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass


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
    rng: random.Random,
) -> Any:
    candidates = live_targets(actors, target_class)
    if not candidates:
        raise RuntimeError(f"没有存活的 {target_class} actor")
    minimum_use = min(used_actor_ids[int(actor.id)] for actor in candidates)
    candidates = [
        actor
        for actor in candidates
        if used_actor_ids[int(actor.id)] == minimum_use
    ]
    target = rng.choice(candidates)
    used_actor_ids[int(target.id)] += 1
    return target


def distance_xy(first, second) -> float:
    return math.hypot(float(first.x - second.x), float(first.y - second.y))


def camera_transform_for_target(
    carla_map,
    target,
    target_class: str,
    direction: str,
    progress: float,
    config: Dict[str, Any],
    previous_ground_location=None,
) -> Tuple[Any, Dict[str, float]]:
    target_location = target.get_location()
    target_waypoint = carla_map.get_waypoint(
        target_location,
        project_to_road=True,
    )
    if target_waypoint is None:
        raise RuntimeError("目标附近没有道路 waypoint")

    preferred_pitch = float(
        config[
            "preferred_pitch_vehicle"
            if target_class == "vehicle"
            else "preferred_pitch_pedestrian"
        ]
    )
    wave = math.sin(progress * math.pi * 2.0)
    altitude = (
        float(config["height_min"])
        + (float(config["height_max"]) - float(config["height_min"]))
        * (0.32 + 0.12 * wave)
    )
    radius = altitude / math.tan(math.radians(abs(preferred_pitch)))
    radius *= 1.0 + 0.035 * math.sin(progress * math.pi)
    radius = float(
        np.clip(
            radius,
            float(config["radius_min"]),
            float(config["radius_max"]),
        )
    )

    step = getattr(target_waypoint, direction, None)
    candidates = step(radius) if callable(step) else []
    if not candidates:
        fallback = "next" if direction == "previous" else "previous"
        step = getattr(target_waypoint, fallback, None)
        candidates = step(radius) if callable(step) else []
    if not candidates:
        raise RuntimeError("无法沿道路找到安全的无人机地面投影点")

    if previous_ground_location is None:
        camera_waypoint = min(
            candidates,
            key=lambda candidate: abs(
                float(candidate.transform.rotation.yaw)
                - float(target_waypoint.transform.rotation.yaw)
            ),
        )
    else:
        camera_waypoint = min(
            candidates,
            key=lambda candidate: distance_xy(
                candidate.transform.location,
                previous_ground_location,
            ),
        )

    ground = camera_waypoint.transform.location
    camera_location = carla.Location(
        x=float(ground.x),
        y=float(ground.y),
        z=float(ground.z) + altitude,
    )
    bbox = target.bounding_box
    aim_location = carla.Location(
        x=float(target_location.x),
        y=float(target_location.y),
        z=float(target_location.z + max(0.4, bbox.extent.z * 0.55)),
    )
    dx = float(aim_location.x - camera_location.x)
    dy = float(aim_location.y - camera_location.y)
    dz = float(aim_location.z - camera_location.z)
    horizontal = max(0.1, math.hypot(dx, dy))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horizontal))
    if not (
        float(config["pitch_min"]) - 2.0
        <= pitch
        <= float(config["pitch_max"]) + 2.0
    ):
        raise RuntimeError(f"相机俯角超出安全范围：{pitch:.2f}")

    transform = carla.Transform(
        camera_location,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )
    return transform, {
        "altitude_m": altitude,
        "ground_distance_m": horizontal,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "ground_x": float(ground.x),
        "ground_y": float(ground.y),
        "ground_z": float(ground.z),
    }


def blend_angle_degrees(previous: float, current: float, alpha: float) -> float:
    delta = (current - previous + 180.0) % 360.0 - 180.0
    return previous + alpha * delta


def look_at_transform(camera_location, aim_location) -> Any:
    dx = float(aim_location.x - camera_location.x)
    dy = float(aim_location.y - camera_location.y)
    dz = float(aim_location.z - camera_location.z)
    horizontal = max(0.1, math.hypot(dx, dy))
    return carla.Transform(
        camera_location,
        carla.Rotation(
            pitch=math.degrees(math.atan2(dz, horizontal)),
            yaw=math.degrees(math.atan2(dy, dx)),
            roll=0.0,
        ),
    )


def blend_transforms(previous, current, position_alpha: float, rotation_alpha: float):
    location = carla.Location(
        x=float(previous.location.x)
        + position_alpha * float(current.location.x - previous.location.x),
        y=float(previous.location.y)
        + position_alpha * float(current.location.y - previous.location.y),
        z=float(previous.location.z)
        + position_alpha * float(current.location.z - previous.location.z),
    )
    rotation = carla.Rotation(
        pitch=blend_angle_degrees(
            float(previous.rotation.pitch),
            float(current.rotation.pitch),
            rotation_alpha,
        ),
        yaw=blend_angle_degrees(
            float(previous.rotation.yaw),
            float(current.rotation.yaw),
            rotation_alpha,
        ),
        roll=0.0,
    )
    return carla.Transform(location, rotation)


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
    if motion_mode == "fixed_hover":
        if "fixed_transform" not in state:
            transform, _ = camera_transform_for_target(
                carla_map,
                target,
                target_class,
                direction,
                0.0,
                config,
            )
            state["fixed_transform"] = transform
        transform = state["fixed_transform"]
    elif motion_mode == "lagged_follow":
        ideal, _ = camera_transform_for_target(
            carla_map,
            target,
            target_class,
            direction,
            progress,
            config,
            previous_ground_location=state.get("previous_ground"),
        )
        previous = state.get("previous_transform")
        transform = ideal if previous is None else blend_transforms(
            previous,
            ideal,
            float(config.get("lagged_follow_position_alpha", 0.28)),
            float(config.get("lagged_follow_rotation_alpha", 0.22)),
        )
        state["previous_ground"] = carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(target.get_location().z),
        )
        state["previous_transform"] = transform
    elif motion_mode == "lateral_orbit":
        target_transform = target.get_transform()
        target_location = target_transform.location
        sweep = float(config.get("orbit_sweep_degrees", 120.0))
        bearing = (
            float(target_transform.rotation.yaw)
            + 90.0
            - sweep * 0.5
            + sweep * progress
        )
        phase = math.sin(progress * math.pi)
        radius = float(config["radius_max"]) - (
            float(config["radius_max"]) - float(config["radius_min"])
        ) * phase
        altitude = float(config["height_min"]) + (
            float(config["height_max"]) - float(config["height_min"])
        ) * (0.25 + 0.65 * phase)
        radians = math.radians(bearing)
        camera_location = carla.Location(
            x=float(target_location.x + radius * math.cos(radians)),
            y=float(target_location.y + radius * math.sin(radians)),
            z=float(target_location.z + altitude),
        )
        bbox = target.bounding_box
        aim_location = carla.Location(
            x=float(target_location.x),
            y=float(target_location.y),
            z=float(target_location.z + max(0.4, bbox.extent.z * 0.55)),
        )
        ideal = look_at_transform(camera_location, aim_location)
        previous = state.get("previous_transform")
        transform = ideal if previous is None else blend_transforms(
            previous,
            ideal,
            1.0,
            float(config.get("orbit_rotation_alpha", 0.3)),
        )
        state["previous_transform"] = transform
    else:
        raise RuntimeError(f"不支持的运动模式：{motion_mode}")

    return transform, motion_pose_stats(transform, target, motion_mode)


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
) -> Tuple[bool, Dict[str, Any]]:
    color_dir = attempt_dir / "color"
    label_dir = attempt_dir / "labels_yolo"
    color_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    planned_max_frames = frames_for_sequence(spec, config)
    min_frames = int(
        config.get("min_frames_by_motion_mode", {}).get(spec.motion_mode, 1)
    )
    fixed_hover_stop_after_absent = int(
        config.get("fixed_hover_stop_after_absent_frames", 5)
    )
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
    motion_state: Dict[str, Any] = {}
    consecutive_absent = 0
    absent_frames = 0
    class_counts: Counter = Counter()
    termination_reason = "max_frames_reached"

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
                    world.tick()
                for item in sensor_sync.values():
                    item.drain()
            try:
                raw = None
                carla_frame = -1
                for _ in range(sample_interval_ticks):
                    carla_frame, raw = tick_and_get_sensors(
                        world,
                        sensor_sync,
                        float(config["sensor_timeout"]),
                    )
                assert raw is not None
                rgb_data = raw["rgb"]
                depth_data = raw["depth"]
                semantic_data = raw["semantic"]
            except (RuntimeError, TimeoutError) as exc:
                return False, {
                    "reason": f"sensor_sync: {exc}",
                    "saved_frames": frame_index,
                }

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

            if (
                spec.motion_mode == "fixed_hover"
                and consecutive_absent >= fixed_hover_stop_after_absent
            ):
                if len(groundtruth) < min_frames:
                    return False, {
                        "reason": (
                            "固定悬停目标过早离开画面："
                            f"saved={len(groundtruth)}, min_required={min_frames}"
                        ),
                        "saved_frames": len(groundtruth),
                    }
                if frame_index not in {
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
                            "stop=5_consecutive_absent"
                        ),
                    )
                termination_reason = "target_absent_for_consecutive_frames"
                break

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
        "road_visible_ratio": stats(road_ratios),
        "near_depth_ratio": stats(near_ratios),
        "adjacent_frame_mean_abs_difference": stats(image_differences),
        "yolo_objects": dict(class_counts),
        "vot_region_format": "x,y,width,height; absent frames are 0,0,0,0",
        "public_modalities": ["RGB"],
        "internal_quality_sensors_not_saved": ["depth", "semantic"],
    }
    (attempt_dir / "sequence_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
        "target_actor_is_constant_inside_each_sequence": True,
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

训练、验证、测试按完整序列划分，禁止相邻帧跨集合。天气按序列轮换。
YOLO 配置文件为 `yolo/data.yaml`。

质量审计：{audit['status']}。
YOLO 图像数：{json.dumps(yolo_summary['splits'], ensure_ascii=False)}。
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args)
    validate_config(config)
    paths = prepare_output(Path(config["out"]), args.overwrite)
    specs = build_sequence_specs(config)
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise ValueError("--max-sequences 必须大于 0")
        specs = specs[: args.max_sequences]
    rng = random.Random(int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

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
    accepted_specs: List[SequenceSpec] = []
    sequence_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    target_vehicles: List[Any] = []
    target_walkers: List[Any] = []

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
        if bool(config.get("include_existing_target_actors", False)):
            world_actors = world.get_actors()
            target_vehicles.extend(list(world_actors.filter("vehicle.*")))
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
        # The initial pose only exists to spawn the reusable sensor rig. Custom
        # maps may have bridge pedestrians beside a road whose topology cannot
        # be traversed in one direction, so probe several actors and directions.
        for initial_target in (target_vehicles + target_walkers)[:80]:
            target_class = (
                "pedestrian"
                if initial_target.type_id.startswith("walker.")
                else "vehicle"
            )
            for direction in ("previous", "next"):
                try:
                    initial_transform, _ = camera_transform_for_target(
                        world.get_map(),
                        initial_target,
                        target_class,
                        direction,
                        0.0,
                        config,
                    )
                    break
                except RuntimeError as exc:
                    initial_errors.append(str(exc))
            if initial_transform is not None:
                break
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

        for sequence_index, spec in enumerate(specs):
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
            for attempt in range(int(config["max_sequence_attempts"])):
                actors = (
                    target_vehicles
                    if spec.target_class == "vehicle"
                    else target_walkers
                )
                target = choose_target_actor(
                    actors,
                    spec.target_class,
                    used_actor_ids,
                    rng,
                )
                direction = (
                    "previous"
                    if (attempt + sequence_index) % 2 == 0
                    else "next"
                )
                attempt_dir = paths["tmp"] / (
                    f"{spec.name}_attempt_{attempt:02d}"
                )
                if attempt_dir.exists():
                    shutil.rmtree(attempt_dir)
                attempt_dir.mkdir(parents=True)
                success, summary = collect_sequence_attempt(
                    world,
                    sensors,
                    sensor_sync,
                    target,
                    spec,
                    attempt_dir,
                    direction,
                    config,
                )
                if success:
                    summary["attempt"] = attempt
                    summary["camera_road_direction"] = direction
                    (attempt_dir / "sequence_meta.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    finalize_sequence(
                        attempt_dir,
                        paths["vot"] / spec.name,
                        paths["qa"],
                        spec,
                    )
                    accepted_specs.append(spec)
                    sequence_summaries.append(summary)
                    accepted = True
                    print(
                        f"[ACCEPT] actor={target.id} "
                        f"absent={summary['absent_ratio']:.3f} "
                        "target_eq_median="
                        f"{summary['target_equivalent_side_px']['median']:.1f}px"
                    )
                    break
                failures.append(
                    {
                        "sequence": spec.name,
                        "attempt": attempt,
                        "target_actor_id": int(target.id),
                        **summary,
                    }
                )
                print(
                    f"[RETRY] {spec.name} attempt={attempt + 1}: "
                    f"{summary['reason']}"
                )
                shutil.rmtree(attempt_dir, ignore_errors=True)
            if not accepted:
                raise RuntimeError(
                    f"序列 {spec.name} 在最大尝试次数内仍未通过质量检查"
                )

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
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"[DONE] {paths['root']}")
    finally:
        destroy_actors(sensors.values())
        for controller in controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        destroy_actors(controllers)
        destroy_actors(walkers)
        destroy_actors(vehicles)
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


if __name__ == "__main__":
    main()
