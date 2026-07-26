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
    split: str
    index_in_weather: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequences-per-weather", type=int, default=None)
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
    if args.sequences_per_weather is not None:
        config["sequences_per_weather"] = args.sequences_per_weather
    if args.frames_per_sequence is not None:
        config["frames_per_sequence"] = args.frames_per_sequence
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
        "sequences_per_weather",
        "frames_per_sequence",
        "vehicles",
        "walkers",
        "sensor_timeout",
    )
    for key in positive:
        if float(config[key]) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    if int(config["sequences_per_weather"]) < 4:
        raise ValueError("每种天气至少需要 4 条序列，才能按序列划分 train/val/test")
    if not 0.0 < float(config["min_visible_ratio"]) <= 1.0:
        raise ValueError("min_visible_ratio 必须在 (0, 1] 内")
    if not config["weather_presets"]:
        raise ValueError("weather_presets 不能为空")


def prepare_output(root: Path, overwrite: bool) -> Dict[str, Path]:
    root = root.resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在：{root}\n"
                "为避免混入旧追踪帧，换一个 out，或明确使用 --overwrite。"
            )
        if "dataset_uav_single_object_vot" not in root.name:
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
    """
    每种天气的前两条序列进入训练集；其余序列交替进入验证和测试集。
    这样三个划分都覆盖全部天气，且车辆/行人数量平衡。
    """
    specs: List[SequenceSpec] = []
    count = int(config["sequences_per_weather"])
    for weather_index, weather in enumerate(config["weather_presets"]):
        for index in range(count):
            target_class = "vehicle" if index % 2 == 0 else "pedestrian"
            if index < 2:
                split = "train"
            elif (index + weather_index) % 2 == 0:
                split = "val"
            else:
                split = "test"
            specs.append(
                SequenceSpec(
                    name=(
                        f"{weather.lower()}_"
                        f"{target_class}_{index:02d}"
                    ),
                    weather=weather,
                    target_class=target_class,
                    split=split,
                    index_in_weather=index,
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
            0.0,
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


def annotation_equivalent_side(annotation: Dict[str, Any]) -> float:
    _, _, width, height = annotation["bbox_xywh"]
    return math.sqrt(float(width) * float(height))


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
    return [
        annotation
        for annotation in annotations
        if not is_boundary_box(
            annotation,
            int(config["width"]),
            int(config["height"]),
        )
    ]


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
    frames = int(config["frames_per_sequence"])
    groundtruth: List[str] = []
    absence: List[str] = []
    occlusion: List[str] = []
    target_sizes: List[float] = []
    target_visible_ratios: List[float] = []
    road_ratios: List[float] = []
    near_ratios: List[float] = []
    image_differences: List[float] = []
    previous_gray: Optional[np.ndarray] = None
    previous_ground = None
    consecutive_absent = 0
    absent_frames = 0
    class_counts: Counter = Counter()

    metadata_path = attempt_dir / "annotations.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for frame_index in range(frames):
            progress = frame_index / max(1, frames - 1)
            try:
                camera_transform, pose_stats = camera_transform_for_target(
                    world.get_map(),
                    target,
                    spec.target_class,
                    direction,
                    progress,
                    config,
                    previous_ground_location=previous_ground,
                )
            except RuntimeError as exc:
                return False, {"reason": str(exc), "saved_frames": frame_index}

            previous_ground = carla.Location(
                x=pose_stats["ground_x"],
                y=pose_stats["ground_y"],
                z=pose_stats["ground_z"],
            )
            base.set_all_sensor_transform(sensors, camera_transform)
            try:
                carla_frame = world.tick()
                rgb_data = sensor_sync["rgb"].get(
                    carla_frame,
                    timeout=float(config["sensor_timeout"]),
                )
                depth_data = sensor_sync["depth"].get(
                    carla_frame,
                    timeout=float(config["sensor_timeout"]),
                )
                semantic_data = sensor_sync["semantic"].get(
                    carla_frame,
                    timeout=float(config["sensor_timeout"]),
                )
            except (RuntimeError, TimeoutError) as exc:
                return False, {
                    "reason": f"sensor_sync: {exc}",
                    "saved_frames": frame_index,
                }

            depth_m = base.decode_carla_depth_meters(depth_data)
            semantic_id = base.decode_semantic_segmentation(semantic_data)
            bad_view, view_stats = base.is_bad_camera_view(
                depth_m,
                min_near_depth_m=float(config["min_near_depth_m"]),
                max_near_depth_ratio=float(config["max_near_depth_ratio"]),
            )
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
                occlusion.append("1")
            else:
                consecutive_absent = 0
                x, y, width, height = target_annotation["bbox_xywh"]
                groundtruth.append(f"{x},{y},{width},{height}")
                absence.append("0")
                visible_ratio = float(
                    target_annotation["visible_ratio_projected_bbox"]
                )
                occlusion.append("1" if visible_ratio < 0.75 else "0")
                target_sizes.append(annotation_equivalent_side(target_annotation))
                target_visible_ratios.append(visible_ratio)

            if consecutive_absent > int(config["max_consecutive_absent_frames"]):
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

            if frame_index in {0, frames // 2, frames - 1}:
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

    absent_ratio = absent_frames / float(frames)
    if absent_ratio > float(config["max_absent_ratio_per_sequence"]):
        return False, {
            "reason": f"主目标缺失比例过高：{absent_ratio:.4f}",
            "saved_frames": frames,
        }
    if image_differences and min(image_differences) < 0.05:
        return False, {
            "reason": (
                "出现疑似完全重复的相邻帧："
                f"min_mean_abs_diff={min(image_differences):.4f}"
            ),
            "saved_frames": frames,
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
    summary = {
        "sequence": spec.name,
        "weather": spec.weather,
        "split": spec.split,
        "target_class": spec.target_class,
        "target_actor_id": int(target.id),
        "target_actor_type": target.type_id,
        "frames": frames,
        "absent_frames": absent_frames,
        "absent_ratio": absent_ratio,
        "target_equivalent_side_px": stats(target_sizes),
        "target_visible_ratio": stats(target_visible_ratios),
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
    expected_frames = int(config["frames_per_sequence"])
    errors: List[str] = []
    for spec in specs:
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
        counts = (len(images), len(labels), len(gt_lines), len(absence_lines))
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
        "expected_total_frames": len(specs) * expected_frames,
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
- 帧率：{config['fps']} FPS。
- 序列数：{len(specs)}。
- 总帧数：{len(specs) * int(config['frames_per_sequence'])}。
- 类别：vehicle、pedestrian。

## VOT 格式

每条序列位于 `vot/<sequence_name>/`：

- `color/00000001.png`：连续 RGB 帧；
- `groundtruth.txt`：每行 `x,y,width,height`；
- `absence.label`：主目标不可见时为 1；
- `occlusion.label`：主目标明显遮挡时为 1；
- `sequence_meta.json`：目标 actor、天气和质量统计；
- `annotations.jsonl`：逐帧完整标注；
- `labels_yolo/`：画面中全部车辆和行人的 YOLO 标签。

遮挡超过 50% 时不写主目标框，对应 VOT 行写为 `0,0,0,0`。

## 数据划分

训练、验证、测试按完整序列划分，禁止相邻帧跨集合。每个划分都覆盖配置中的
全部天气。YOLO 配置文件为 `yolo/data.yaml`。

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

        initial_target = (
            walkers[0] if walkers else vehicles[0]
        )
        initial_transform, _ = camera_transform_for_target(
            world.get_map(),
            initial_target,
            (
                "pedestrian"
                if initial_target.type_id.startswith("walker.")
                else "vehicle"
            ),
            "previous",
            0.0,
            config,
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
                f"{spec.name} split={spec.split}"
            )
            base.apply_weather(world, spec.weather)
            for _ in range(int(config["weather_warmup_frames"])):
                world.tick()
            for item in sensor_sync.values():
                item.drain()

            accepted = False
            for attempt in range(int(config["max_sequence_attempts"])):
                actors = vehicles if spec.target_class == "vehicle" else walkers
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
            "total_frames": (
                len(accepted_specs) * int(config["frames_per_sequence"])
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
