#!/usr/bin/env python
"""
OpenHUTB/CARLA 无人机跨相机多目标跟踪数据采集器。

公开数据只保存 RGB。深度和语义相机仅在采集时用于遮挡判断、可见像素
边界框和坏视角筛选，不写入最终数据集。所有相机在同一次 world.tick()
中取帧，并以 CARLA actor.id 作为跨相机、跨帧统一的 global_id。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import collect_rpg_small_targets_carla_v2 as base
import collect_uav_single_object_vot_carla as vot


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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--frames-per-scene", type=int, default=None)
    parser.add_argument("--scenes-per-weather", type=int, default=None)
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
    if args.scenes_per_weather is not None:
        config["scenes_per_weather"] = args.scenes_per_weather
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
        "num_cameras",
        "scenes_per_weather",
        "frames_per_scene",
        "vehicles",
        "walkers",
        "sensor_timeout",
        "max_scene_attempts",
    )
    for key in positive:
        if float(config[key]) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    if int(config["num_cameras"]) < 2:
        raise ValueError("跨相机数据至少需要 2 台相机")
    if int(config["scenes_per_weather"]) < 4:
        raise ValueError("每种天气至少 4 个场景，才能覆盖 train/val/test")
    ratio = float(config["min_visible_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("min_visible_ratio 必须在 (0, 1] 内")
    if not config["weather_presets"]:
        raise ValueError("weather_presets 不能为空")


def resolve_output(config: Dict[str, Any]) -> Path:
    output = Path(config["out"])
    if not output.is_absolute():
        output = Path(config["_config_path"]).parent / output
    return output.resolve()


def prepare_output(root: Path, overwrite: bool) -> Dict[str, Path]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在：{root}\n"
                "请更换 --out，或明确使用 --overwrite。"
            )
        if "dataset_uav_multicamera_mot" not in root.name:
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
    return paths


def build_scene_specs(config: Dict[str, Any]) -> List[SceneSpec]:
    specs: List[SceneSpec] = []
    count = int(config["scenes_per_weather"])
    for weather_index, weather in enumerate(config["weather_presets"]):
        for index in range(count):
            if index < count - 2:
                split = "train"
            else:
                validation_index = (
                    count - 2 if weather_index % 2 == 0 else count - 1
                )
                split = "val" if index == validation_index else "test"
            specs.append(
                SceneSpec(
                    name=(
                        f"{weather.lower()}_"
                        f"{'vehicle' if index % 2 == 0 else 'pedestrian'}_"
                        f"scene_{index:02d}"
                    ),
                    weather=weather,
                    split=split,
                    anchor_class=(
                        "vehicle" if index % 2 == 0 else "pedestrian"
                    ),
                    scene_id=weather_index * count + index,
                    weather_index=weather_index,
                    index_in_weather=index,
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
                0.0,
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
    for actor in actors:
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except RuntimeError:
            pass
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass


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
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    frame = int(world.tick())
    data: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        data[unit.name] = {
            modality: sync.get(frame, timeout=timeout)
            for modality, sync in unit.syncs.items()
        }
    return frame, data


def live_target_actors(vehicles: Sequence[Any], walkers: Sequence[Any]) -> List[Any]:
    return [
        actor
        for actor in list(vehicles) + list(walkers)
        if actor is not None and actor.is_alive
    ]


def distance_2d(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def choose_anchor_actor(
    actors: Sequence[Any],
    config: Dict[str, Any],
    rng: random.Random,
    preferred_class: Optional[str] = None,
) -> Optional[Any]:
    alive = [actor for actor in actors if actor is not None and actor.is_alive]
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
        actor_box_height = float(anchor.bounding_box.extent.z) * 0.5
    except RuntimeError:
        return None
    aim = carla.Location(
        x=float(actor_location.x),
        y=float(actor_location.y),
        z=float(actor_location.z) + max(0.5, actor_box_height),
    )
    minimum_radius = float(config["camera_radius_min_m"])
    maximum_radius = float(config["camera_radius_max_m"])
    minimum_pitch = float(config["camera_pitch_min_deg"])
    maximum_pitch = float(config["camera_pitch_max_deg"])
    candidates: List[Tuple[float, Any]] = []
    shuffled_waypoints = list(road_waypoints)
    rng.shuffle(shuffled_waypoints)
    for waypoint in shuffled_waypoints:
        location = waypoint.transform.location
        radius = distance_2d(location, actor_location)
        if not minimum_radius <= radius <= maximum_radius:
            continue
        altitude = rng.uniform(
            float(config["camera_height_min_m"]),
            float(config["camera_height_max_m"]),
        )
        transform = look_at_transform(location, aim, altitude)
        pitch = float(transform.rotation.pitch)
        if not minimum_pitch <= pitch <= maximum_pitch:
            continue
        bearing = (
            math.degrees(
                math.atan2(
                    float(location.y) - float(actor_location.y),
                    float(location.x) - float(actor_location.x),
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
    selected: List[Tuple[float, Any]] = [candidates[0]]
    minimum_separation = float(config["camera_bearing_separation_deg"])
    while len(selected) < required:
        valid = [
            candidate
            for candidate in candidates
            if all(
                angle_difference_degrees(candidate[0], used[0])
                >= minimum_separation
                for used in selected
            )
        ]
        if not valid:
            return None
        best_score = max(
            min(
                angle_difference_degrees(candidate[0], used[0])
                for used in selected
            )
            for candidate in valid
        )
        best = [
            candidate
            for candidate in valid
            if min(
                angle_difference_degrees(candidate[0], used[0])
                for used in selected
            )
            >= best_score - 1e-6
        ]
        selected.append(rng.choice(best))
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
    valid = (
        enough_objects
        and len(common) >= int(config["min_common_ids_per_frame"])
    )
    return valid, common, counts


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
    view_valid = (
        road_ratio >= float(config["min_road_visible_ratio"])
        and near_ratio <= float(config["max_near_depth_ratio"])
    )
    return {
        "rgb_bgr": rgb_bgr,
        "annotations": annotations,
        "road_visible_ratio": road_ratio,
        "near_depth_ratio": near_ratio,
        "view_valid": view_valid,
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
) -> Dict[str, Any]:
    for unit, transform in zip(units, transforms):
        set_camera_unit_transform(unit, transform)
    drain_camera_units(units)
    timeout = float(config["sensor_timeout"])
    for _ in range(2):
        tick_and_get(world, units, timeout)
    drain_camera_units(units)

    for unit in units:
        write_json(
            staging_dir / "calibration" / f"{unit.name}.json",
            calibration_dict(unit.name, unit.transform, config, ground_z),
        )

    frames_required = int(config["frames_per_scene"])
    qa_count = min(int(config["qa_frames_per_camera"]), frames_required)
    qa_indices = sorted(
        set(
            np.linspace(0, frames_required - 1, qa_count)
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

    for frame_index in range(frames_required):
        actor_by_id = {
            int(actor.id): actor
            for actor in traffic_actors
            if actor is not None and actor.is_alive
        }
        carla_frame, raw = tick_and_get(world, units, timeout)
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
        valid, common_ids, visibility_counts = frame_quality(
            camera_payloads,
            config,
        )
        required_visible = int(required_global_id) in set(map(int, common_ids))
        valid = valid and all_views_valid and required_visible
        if not valid:
            view_stats = {
                name: {
                    "road": round(float(payload["road_visible_ratio"]), 3),
                    "near": round(float(payload["near_depth_ratio"]), 3),
                    "valid": bool(payload["view_valid"]),
                }
                for name, payload in camera_payloads.items()
            }
            return {
                "accepted": False,
                "reason": (
                    f"frame {frame_index}: view_valid={all_views_valid}, "
                    f"common={len(common_ids)}, "
                    f"anchor_common={required_visible}, "
                    f"objects={[len(p['annotations']) for p in camera_payloads.values()]}, "
                    f"views={view_stats}"
                ),
            }
        valid_frames += 1
        common_counts.append(len(common_ids))
        observation_counts.update(visibility_counts)
        observed_ids: List[int] = []
        for payload in camera_payloads.values():
            for annotation in payload["annotations"]:
                class_counts[str(annotation["class_name"])] += 1
                observed_ids.append(int(annotation["global_id"]))
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
        write_world_tracks(
            staging_dir,
            spec,
            frame_index,
            carla_frame,
            actor_by_id,
            observed_ids,
        )
        frame_rows.append((frame_index, carla_frame, len(common_ids)))

    minimum_valid = math.ceil(
        frames_required * float(config["min_valid_frame_ratio"])
    )
    accepted = valid_frames >= minimum_valid
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
                "frames": frames_required,
                "camera_ids": [unit.name for unit in units],
                "ground_plane_z_m": ground_z,
                "valid_frames": valid_frames,
                "mean_common_ids_per_frame": float(np.mean(common_counts)),
                "max_common_ids_per_frame": int(max(common_counts)),
                "global_ids_observed": [
                    dataset_global_id(spec.scene_id, int(actor_id))
                    for actor_id in sorted(observation_counts)
                ],
                "class_observations": dict(class_counts),
            },
        )
    return {
        "accepted": accepted,
        "reason": "ok" if accepted else "valid frame ratio too low",
        "valid_frames": valid_frames,
        "mean_common_ids_per_frame": (
            float(np.mean(common_counts)) if common_counts else 0.0
        ),
        "class_observations": dict(class_counts),
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
) -> Dict[str, Any]:
    max_attempts = int(config["max_scene_attempts"])
    for attempt in range(max_attempts):
        staging_dir = paths["staging"] / f"{spec.name}_attempt_{attempt:03d}"
        safe_remove_staging(staging_dir, paths["staging"])
        staging_dir.mkdir(parents=True)
        anchor = choose_anchor_actor(
            traffic_actors,
            config,
            rng,
            preferred_class=spec.anchor_class,
        )
        if anchor is None:
            safe_remove_staging(staging_dir, paths["staging"])
            continue
        rig = choose_camera_transforms(anchor, road_waypoints, config, rng)
        if rig is None:
            safe_remove_staging(staging_dir, paths["staging"])
            continue
        transforms, ground_z = rig
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
            )
        except (TimeoutError, RuntimeError) as exc:
            result = {"accepted": False, "reason": str(exc)}
        if result["accepted"]:
            destination = paths["scenes"] / spec.name
            if destination.exists():
                raise RuntimeError(f"场景目录意外存在：{destination}")
            shutil.move(str(staging_dir), str(destination))
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
                }
            )
            print(
                f"[OK] {spec.name}: attempt={attempt + 1}, "
                f"common/frame={result['mean_common_ids_per_frame']:.2f}, "
                f"classes={result['class_observations']}"
            )
            return result
        print(
            f"[RETRY] {spec.name} attempt {attempt + 1}/{max_attempts}: "
            f"{result['reason']}"
        )
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

    for scene_dir in sorted((root / "scenes").glob("*")):
        if not scene_dir.is_dir():
            continue
        meta_path = scene_dir / "scene_meta.json"
        if not meta_path.is_file():
            sync_errors.append(f"{scene_dir.name}: missing scene_meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
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
            if not common_per_frame[frame_index]:
                sync_errors.append(
                    f"{scene_dir.name} frame {frame_index}: no common ID"
                )
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
        "passed": not sync_errors and duplicate_image_files == 0,
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
        "version": "1.0",
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
            "fps": float(config["fps"]),
            "frames_per_scene": int(config["frames_per_scene"]),
            "weather_presets": list(config["weather_presets"]),
            "camera_height_m": [
                float(config["camera_height_min_m"]),
                float(config["camera_height_max_m"]),
            ],
            "camera_pitch_deg": [
                float(config["camera_pitch_min_deg"]),
                float(config["camera_pitch_max_deg"]),
            ],
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
- `yolo/data.yaml`：可直接训练 YOLO11
- `quality_audit.json`：同步、重复帧、遮挡和标注审计

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
    try:
        if bool(config["hide_static_map_vehicles"]):
            static_ids = base.hide_static_map_vehicles(world)
        vehicles, walkers, controllers = base.spawn_background_traffic(
            client,
            world,
            int(config["vehicles"]),
            int(config["walkers"]),
            int(config["tm_port"]),
            int(config["seed"]),
        )
        if bool(config["remove_two_wheel_vehicles"]):
            base.destroy_live_two_wheel_vehicles(world)
            vehicles = [
                actor
                for actor in vehicles
                if actor is not None and actor.is_alive
            ]
        traffic_actors = live_target_actors(vehicles, walkers)
        if not traffic_actors:
            raise RuntimeError("没有成功生成车辆或行人")
        carla_map = world.get_map()
        road_waypoints = carla_map.generate_waypoints(
            float(config["camera_candidate_spacing_m"])
        )
        anchor = choose_anchor_actor(
            traffic_actors,
            config,
            random.Random(int(config["seed"])),
        )
        if anchor is None:
            raise RuntimeError("无法选择初始目标密集区域")
        initial_rig = choose_camera_transforms(
            anchor,
            road_waypoints,
            config,
            random.Random(int(config["seed"]) + 1),
        )
        if initial_rig is None:
            raise RuntimeError("无法生成初始三相机道路上空视角")
        units = spawn_camera_units(world, initial_rig[0][0], config)
        rng = random.Random(int(config["seed"]) + 101)

        current_weather = None
        for scene_index, spec in enumerate(specs):
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
            result = collect_scene(
                world,
                units,
                road_waypoints,
                traffic_actors,
                spec,
                paths,
                config,
                rng,
            )
            results.append(result)
    finally:
        destroy_actors(
            sensor
            for unit in units
            for sensor in unit.sensors.values()
        )
        destroy_actors(controllers)
        destroy_actors(walkers)
        destroy_actors(vehicles)
        if static_ids:
            base.restore_static_map_vehicles(world, static_ids)
        world.apply_settings(original_settings)
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

    paths = prepare_output(output, args.overwrite)
    write_json(paths["root"] / "collection_config_used.json", config)
    started = time.time()
    results = run_collection(config, specs, paths)
    normalize_existing_dataset(paths["root"], specs)
    yolo_report = rebuild_yolo_dataset(paths["root"], specs)
    audit = audit_dataset(paths["root"])
    write_dataset_manifest(paths["root"], config, audit)
    write_readme(paths["root"], config, audit)
    write_json(
        paths["root"] / "collection_report.json",
        {
            "elapsed_seconds": time.time() - started,
            "scene_results": results,
            "yolo": yolo_report,
            "audit_passed": audit["passed"],
        },
    )
    safe_remove_staging(paths["staging"], paths["root"])
    print(
        f"[DONE] root={paths['root']}, scenes={audit['scene_count']}, "
        f"images={audit['image_count']}, passed={audit['passed']}"
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] 用户中断。")
        raise SystemExit(130)
