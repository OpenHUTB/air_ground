#!/usr/bin/env python
"""
OpenHUTB/CARLA 无人机单相机多目标跟踪数据集采集器。

最终数据只保存 RGB。深度图和语义分割图只在采集时用于可见性、遮挡和
道路视野质量判断。输出同时兼容 MOTChallenge 和 Ultralytics YOLO 检测训练。

设计参考 UAVDT 与 VisDrone-MOT 的关键设置：连续视频序列、稳定身份、无人机
视角/天气/相机运动属性，以及按完整序列划分训练、验证和测试集。
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import collect_rpg_small_targets_carla_v2 as base
import collect_uav_multicamera_mot_carla as support


carla = base.carla
CLASS_NAMES = {0: "vehicle", 1: "pedestrian"}
DEFAULT_CONFIG = Path(__file__).with_name("single_camera_mot_config.json")


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    weather: str
    split: str
    anchor_class: str
    motion_mode: str
    sequence_id: int
    index_in_weather: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--frames-per-sequence", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    path = args.config.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    if args.out is not None:
        config["out"] = str(args.out)
    if args.frames_per_sequence is not None:
        config["frames_per_sequence"] = int(args.frames_per_sequence)
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
        "max_sequence_attempts",
    )
    for key in positive:
        if float(config[key]) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    if int(config["sequences_per_weather"]) < 4:
        raise ValueError("每种天气至少需要 4 段序列，以生成 2/1/1 的 train/val/test 划分")
    if not config["weather_presets"]:
        raise ValueError("weather_presets 不能为空")
    if not config["camera_motion_modes"]:
        raise ValueError("camera_motion_modes 不能为空")
    visible_ratio = float(config["min_visible_ratio"])
    if not 0.0 < visible_ratio <= 1.0:
        raise ValueError("min_visible_ratio 必须在 (0, 1] 内")
    smoothing = float(config["camera_aim_smoothing"])
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("camera_aim_smoothing 必须在 (0, 1] 内")


def resolve_output(config: Dict[str, Any]) -> Path:
    output = Path(config["out"])
    if not output.is_absolute():
        output = Path(config["_config_path"]).parent / output
    return output.resolve()


def prepare_output(root: Path, overwrite: bool) -> Dict[str, Path]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在：{root}\n请更换 --out 或明确使用 --overwrite")
        if "dataset_uav_single_camera_mot" not in root.name:
            raise RuntimeError(f"拒绝覆盖名称异常的目录：{root}")
        shutil.rmtree(root)
    paths = {
        "root": root,
        "sequences": root / "sequences",
        "qa": root / "qa_overlay",
        "splits": root / "splits",
        "yolo": root / "yolo",
        "staging": root / "_sequence_staging",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_remove_staging(path: Path, staging_root: Path) -> None:
    resolved = path.resolve()
    root = staging_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"拒绝删除 staging 目录之外的路径：{resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def build_sequence_specs(config: Dict[str, Any]) -> List[SequenceSpec]:
    specs: List[SequenceSpec] = []
    count = int(config["sequences_per_weather"])
    modes = list(map(str, config["camera_motion_modes"]))
    sequence_id = 0
    for weather in map(str, config["weather_presets"]):
        for index in range(count):
            # 每种天气最后两段分别用于验证和测试，其余段只用于训练。
            split = "train" if index < count - 2 else "val" if index == count - 2 else "test"
            specs.append(
                SequenceSpec(
                    name=f"seq_{sequence_id:04d}_{weather.lower()}",
                    weather=weather,
                    split=split,
                    anchor_class="vehicle" if sequence_id % 2 == 0 else "pedestrian",
                    motion_mode=modes[sequence_id % len(modes)],
                    sequence_id=sequence_id,
                    index_in_weather=index,
                )
            )
            sequence_id += 1
    return specs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def interpolate_location(start: Any, end: Any, progress: float) -> Any:
    progress = min(1.0, max(0.0, float(progress)))
    return carla.Location(
        x=float(start.x) + (float(end.x) - float(start.x)) * progress,
        y=float(start.y) + (float(end.y) - float(start.y)) * progress,
        z=float(start.z) + (float(end.z) - float(start.z)) * progress,
    )


def camera_path_end(
    carla_map: Any,
    initial_transform: Any,
    motion_mode: str,
    config: Dict[str, Any],
    rng: random.Random,
) -> Any:
    start = initial_transform.location
    if motion_mode == "hover":
        return carla.Location(x=float(start.x), y=float(start.y), z=float(start.z))
    ground = carla.Location(x=float(start.x), y=float(start.y), z=float(start.z) - 25.0)
    waypoint = carla_map.get_waypoint(
        ground,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return carla.Location(x=float(start.x), y=float(start.y), z=float(start.z))
    travel = rng.uniform(
        float(config["camera_travel_min_m"]),
        float(config["camera_travel_max_m"]),
    )
    candidates = list(waypoint.next(travel)) + list(waypoint.previous(travel))
    if not candidates:
        return carla.Location(x=float(start.x), y=float(start.y), z=float(start.z))
    endpoint = rng.choice(candidates).transform.location
    return carla.Location(
        x=float(endpoint.x),
        y=float(endpoint.y),
        z=float(start.z) + float(endpoint.z) - float(waypoint.transform.location.z),
    )


def smoothed_aim(previous: Any, target: Any, alpha: float) -> Any:
    return carla.Location(
        x=float(previous.x) + alpha * (float(target.x) - float(previous.x)),
        y=float(previous.y) + alpha * (float(target.y) - float(previous.y)),
        z=float(previous.z) + alpha * (float(target.z) - float(previous.z)),
    )


def make_transform(camera_location: Any, aim: Any) -> Any:
    dx = float(aim.x) - float(camera_location.x)
    dy = float(aim.y) - float(camera_location.y)
    dz = float(aim.z) - float(camera_location.z)
    horizontal = max(1e-6, math.hypot(dx, dy))
    return carla.Transform(
        camera_location,
        carla.Rotation(
            pitch=math.degrees(math.atan2(dz, horizontal)),
            yaw=math.degrees(math.atan2(dy, dx)),
            roll=0.0,
        ),
    )


def mot_line(frame: int, track_id: int, annotation: Dict[str, Any]) -> str:
    x, y, width, height = map(float, annotation["bbox_xywh"])
    return (
        f"{frame},{track_id},{x:.2f},{y:.2f},{width:.2f},{height:.2f},"
        f"1,{int(annotation['class_id']) + 1},{float(annotation['visibility']):.6f}\n"
    )


def save_overlay(
    image: np.ndarray,
    annotations: Sequence[Dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    canvas = image.copy()
    for annotation in annotations:
        x, y, width, height = map(int, annotation["bbox_xywh"])
        color = (0, 255, 0) if int(annotation["class_id"]) == 0 else (255, 80, 40)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            canvas,
            f"{annotation['class_name']} id={annotation['track_id']} vis={annotation['visibility']:.2f}",
            (x, max(22, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def write_seqinfo(path: Path, spec: SequenceSpec, config: Dict[str, Any]) -> None:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser["Sequence"] = {
        "name": spec.name,
        "imDir": "img1",
        "frameRate": str(int(round(float(config["fps"])))),
        "seqLength": str(int(config["frames_per_sequence"])),
        "imWidth": str(int(config["width"])),
        "imHeight": str(int(config["height"])),
        "imExt": ".png",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        parser.write(file, space_around_delimiters=False)


def sequence_quality(
    annotations_by_frame: Sequence[Sequence[Dict[str, Any]]],
    anchor_actor_id: int,
    config: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    frames = len(annotations_by_frame)
    track_lengths = Counter(
        int(annotation["carla_actor_id"])
        for annotations in annotations_by_frame
        for annotation in annotations
    )
    class_counts = Counter(
        str(annotation["class_name"])
        for annotations in annotations_by_frame
        for annotation in annotations
    )
    mean_objects = float(np.mean([len(items) for items in annotations_by_frame]))
    long_threshold = math.ceil(frames * float(config["long_track_frame_ratio"]))
    long_tracks = sum(length >= long_threshold for length in track_lengths.values())
    anchor_frames = int(track_lengths.get(int(anchor_actor_id), 0))
    report = {
        "frames": frames,
        "mean_objects_per_frame": mean_objects,
        "unique_tracks": len(track_lengths),
        "long_tracks": long_tracks,
        "long_track_threshold_frames": long_threshold,
        "anchor_visible_frames": anchor_frames,
        "anchor_visible_ratio": anchor_frames / frames if frames else 0.0,
        "class_observations": dict(class_counts),
        "track_lengths": {str(key): int(value) for key, value in sorted(track_lengths.items())},
    }
    accepted = (
        mean_objects >= float(config["min_mean_objects_per_frame"])
        and len(track_lengths) >= int(config["min_unique_tracks_per_sequence"])
        and long_tracks >= int(config["min_long_tracks_per_sequence"])
        and report["anchor_visible_ratio"] >= float(config["min_anchor_visible_frame_ratio"])
    )
    return accepted, report


def collect_sequence_attempt(
    world: Any,
    unit: Any,
    initial_transform: Any,
    anchor: Any,
    spec: SequenceSpec,
    sequence_dir: Path,
    qa_root: Path,
    traffic_actors: Sequence[Any],
    config: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    frames_required = int(config["frames_per_sequence"])
    endpoint = camera_path_end(world.get_map(), initial_transform, spec.motion_mode, config, rng)
    anchor_location = anchor.get_location()
    aim = carla.Location(
        x=float(anchor_location.x),
        y=float(anchor_location.y),
        z=float(anchor_location.z) + 0.8,
    )
    timeout = float(config["sensor_timeout"])
    support.set_camera_unit_transform(unit, initial_transform)
    support.drain_camera_units([unit])
    for _ in range(2):
        support.tick_and_get(world, [unit], timeout)
    support.drain_camera_units([unit])

    qa_count = min(int(config["qa_frames_per_sequence"]), frames_required)
    qa_frames = set(np.linspace(1, frames_required, qa_count).round().astype(int).tolist())
    track_id_by_actor: Dict[int, int] = {}
    annotations_by_frame: List[List[Dict[str, Any]]] = []
    frame_records: List[Dict[str, Any]] = []
    mot_rows: List[str] = []
    alpha = float(config["camera_aim_smoothing"])

    for frame_number in range(1, frames_required + 1):
        progress = 0.0 if frames_required <= 1 else (frame_number - 1) / (frames_required - 1)
        camera_location = interpolate_location(initial_transform.location, endpoint, progress)
        if anchor is not None and anchor.is_alive:
            live_location = anchor.get_location()
            target_aim = carla.Location(
                x=float(live_location.x),
                y=float(live_location.y),
                z=float(live_location.z) + 0.8,
            )
            aim = smoothed_aim(aim, target_aim, alpha)
        transform = make_transform(camera_location, aim)
        pitch = float(transform.rotation.pitch)
        if not float(config["camera_pitch_min_deg"]) - 4.0 <= pitch <= float(config["camera_pitch_max_deg"]) + 4.0:
            return {"accepted": False, "reason": f"frame {frame_number}: pitch={pitch:.2f}"}
        support.set_camera_unit_transform(unit, transform)
        actor_by_id = {
            int(actor.id): actor
            for actor in traffic_actors
            if actor is not None and actor.is_alive
        }
        carla_frame, raw = support.tick_and_get(world, [unit], timeout)
        payload = support.inspect_camera_frame(
            world,
            unit,
            raw[unit.name],
            actor_by_id,
            config,
        )
        if not bool(payload["view_valid"]):
            return {
                "accepted": False,
                "reason": (
                    f"frame {frame_number}: road={payload['road_visible_ratio']:.3f}, "
                    f"near={payload['near_depth_ratio']:.3f}"
                ),
            }

        saved_annotations: List[Dict[str, Any]] = []
        for source in payload["annotations"]:
            actor_id = int(source["carla_actor_id"])
            if actor_id not in track_id_by_actor:
                track_id_by_actor[actor_id] = len(track_id_by_actor) + 1
            annotation = dict(source)
            annotation["track_id"] = int(track_id_by_actor[actor_id])
            annotation["visibility"] = float(annotation["visible_ratio_projected_bbox"])
            annotation["occlusion"] = 1.0 - annotation["visibility"]
            saved_annotations.append(annotation)
            mot_rows.append(mot_line(frame_number, annotation["track_id"], annotation))
        saved_annotations.sort(key=lambda item: (int(item["class_id"]), int(item["track_id"])))
        annotations_by_frame.append(saved_annotations)

        image_path = sequence_dir / "img1" / f"{frame_number:06d}.png"
        label_path = sequence_dir / "labels_yolo" / f"{frame_number:06d}.txt"
        annotation_path = sequence_dir / "annotations" / f"{frame_number:06d}.json"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), payload["rgb_bgr"], [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"图像保存失败：{image_path}")
        base.save_yolo_label(
            label_path,
            saved_annotations,
            int(config["width"]),
            int(config["height"]),
        )
        write_json(
            annotation_path,
            {
                "sequence": spec.name,
                "split": spec.split,
                "weather": spec.weather,
                "motion_mode": spec.motion_mode,
                "frame": frame_number,
                "carla_frame": int(carla_frame),
                "camera_transform": base.transform_to_dict(transform),
                "road_visible_ratio": float(payload["road_visible_ratio"]),
                "near_depth_ratio": float(payload["near_depth_ratio"]),
                "annotations": saved_annotations,
            },
        )
        if frame_number in qa_frames:
            save_overlay(
                payload["rgb_bgr"],
                saved_annotations,
                qa_root / spec.name / f"{frame_number:06d}_overlay.jpg",
                f"{spec.name} frame={frame_number} {spec.weather} {spec.motion_mode}",
            )
        frame_records.append(
            {
                "frame": frame_number,
                "carla_frame": int(carla_frame),
                "objects": len(saved_annotations),
                "road_visible_ratio": float(payload["road_visible_ratio"]),
                "near_depth_ratio": float(payload["near_depth_ratio"]),
            }
        )

    accepted, quality = sequence_quality(annotations_by_frame, int(anchor.id), config)
    if not accepted:
        return {"accepted": False, "reason": f"sequence quality: {quality}"}
    gt_path = sequence_dir / "gt" / "gt.txt"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text("".join(mot_rows), encoding="utf-8")
    write_seqinfo(sequence_dir / "seqinfo.ini", spec, config)
    write_json(
        sequence_dir / "sequence_meta.json",
        {
            "sequence": spec.name,
            "sequence_id": spec.sequence_id,
            "split": spec.split,
            "weather": spec.weather,
            "anchor_class": spec.anchor_class,
            "anchor_carla_actor_id": int(anchor.id),
            "motion_mode": spec.motion_mode,
            "frames": frames_required,
            "quality": quality,
            "frame_records": frame_records,
        },
    )
    return {"accepted": True, **quality}


def collect_one_sequence(
    world: Any,
    unit: Any,
    road_waypoints: Sequence[Any],
    traffic_actors: Sequence[Any],
    spec: SequenceSpec,
    paths: Dict[str, Path],
    config: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    max_attempts = int(config["max_sequence_attempts"])
    for attempt in range(1, max_attempts + 1):
        staging_dir = paths["staging"] / f"{spec.name}_attempt_{attempt:03d}"
        safe_remove_staging(staging_dir, paths["staging"])
        staging_dir.mkdir(parents=True, exist_ok=True)
        anchor = support.choose_anchor_actor(
            traffic_actors,
            config,
            rng,
            preferred_class=spec.anchor_class,
        )
        if anchor is None:
            safe_remove_staging(staging_dir, paths["staging"])
            continue
        rig = support.choose_camera_transforms(anchor, road_waypoints, config, rng)
        if rig is None:
            safe_remove_staging(staging_dir, paths["staging"])
            continue
        initial_transform = rig[0][0]
        try:
            result = collect_sequence_attempt(
                world,
                unit,
                initial_transform,
                anchor,
                spec,
                staging_dir,
                paths["qa"],
                traffic_actors,
                config,
                rng,
            )
        except (RuntimeError, TimeoutError) as exc:
            result = {"accepted": False, "reason": str(exc)}
        if result.get("accepted"):
            destination = paths["sequences"] / spec.name
            shutil.move(str(staging_dir), str(destination))
            result["attempt"] = attempt
            print(
                f"[OK] {spec.name}: attempt={attempt}, mean_objects={result['mean_objects_per_frame']:.2f}, "
                f"tracks={result['unique_tracks']}, long_tracks={result['long_tracks']}"
            )
            return result
        print(f"[RETRY] {spec.name} {attempt}/{max_attempts}: {result.get('reason', 'unknown')}")
        safe_remove_staging(staging_dir, paths["staging"])
    raise RuntimeError(f"{spec.name} 连续 {max_attempts} 次未得到合格序列")


def destroy_actors(actors: Iterable[Any]) -> None:
    support.destroy_actors(actors)


def run_collection(
    config: Dict[str, Any],
    specs: Sequence[SequenceSpec],
    paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    client = carla.Client(str(config["host"]), int(config["port"]))
    client.set_timeout(float(config["timeout"]))
    # map=null 时只连接当前模拟器世界，不切地图、不启动第二个实例。
    world = client.load_world(str(config["map"])) if config.get("map") else client.get_world()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(config["fps"])
    world.apply_settings(settings)

    static_ids: List[int] = []
    vehicles: List[Any] = []
    walkers: List[Any] = []
    controllers: List[Any] = []
    units: List[Any] = []
    results: List[Dict[str, Any]] = []
    try:
        if bool(config["hide_static_map_vehicles"]):
            static_ids = base.hide_static_map_vehicles(world)
        initial_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=float(config["camera_height_max_m"])),
            carla.Rotation(pitch=-40.0),
        )
        units = support.spawn_camera_units(world, initial_transform, config)
        unit = units[0]
        road_waypoints = world.get_map().generate_waypoints(float(config["camera_candidate_spacing_m"]))
        current_weather: Optional[str] = None

        for index, spec in enumerate(specs, start=1):
            if current_weather != spec.weather:
                applied = base.apply_weather(world, spec.weather)
                print(f"[INFO] Weather: requested={spec.weather}, applied={applied}")
                for _ in range(int(config["weather_warmup_ticks"])):
                    world.tick()
                support.drain_camera_units(units)
                current_weather = spec.weather

            print(
                f"[INFO] Sequence {index}/{len(specs)}: {spec.name} "
                f"({spec.split}, anchor={spec.anchor_class}, motion={spec.motion_mode})"
            )
            destroy_actors(controllers)
            destroy_actors(walkers)
            destroy_actors(vehicles)
            controllers, walkers, vehicles = [], [], []
            for _ in range(2):
                world.tick()
            sequence_seed = int(config["seed"]) + spec.sequence_id * 1009
            vehicles, walkers, controllers = base.spawn_background_traffic(
                client,
                world,
                int(config["vehicles"]),
                int(config["walkers"]),
                int(config["tm_port"]),
                sequence_seed,
            )
            if bool(config["remove_two_wheel_vehicles"]):
                base.destroy_live_two_wheel_vehicles(world)
                vehicles = [actor for actor in vehicles if actor is not None and actor.is_alive]
            traffic_actors = support.live_target_actors(vehicles, walkers)
            if not traffic_actors:
                raise RuntimeError(f"{spec.name} 没有成功生成车辆或行人")
            for _ in range(int(config["actor_spawn_warmup_ticks"])):
                world.tick()
            support.drain_camera_units(units)
            result = collect_one_sequence(
                world,
                unit,
                road_waypoints,
                traffic_actors,
                spec,
                paths,
                config,
                random.Random(sequence_seed + 101),
            )
            result.update({"sequence": spec.name, "split": spec.split, "weather": spec.weather})
            results.append(result)
    finally:
        destroy_actors(sensor for unit in units for sensor in unit.sensors.values())
        destroy_actors(controllers)
        destroy_actors(walkers)
        destroy_actors(vehicles)
        if static_ids:
            base.restore_static_map_vehicles(world, static_ids)
        world.apply_settings(original_settings)
    return results


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)


def rebalance_evaluation_splits(
    root: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """在每种天气内部交换 val/test，降低类别分布偏移。"""
    pairs: List[Tuple[str, List[Tuple[Path, Dict[str, Any]]]]] = []
    for weather in map(str, config["weather_presets"]):
        candidates: List[Tuple[Path, Dict[str, Any]]] = []
        for sequence_dir in sorted((root / "sequences").glob("seq_*")):
            meta_path = sequence_dir / "sequence_meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta["weather"]) == weather and str(meta["split"]) in {"val", "test"}:
                candidates.append((sequence_dir, meta))
        if len(candidates) != 2:
            raise RuntimeError(
                f"{weather} 应有两段 val/test 候选序列，实际为 {len(candidates)}"
            )
        pairs.append((weather, candidates))

    class_names = tuple(CLASS_NAMES.values())
    total = Counter()
    for _, candidates in pairs:
        for _, meta in candidates:
            total.update(meta["quality"]["class_observations"])

    best: Optional[Tuple[float, Tuple[int, ...], Counter]] = None
    for choices in itertools.product((0, 1), repeat=len(pairs)):
        validation = Counter()
        for pair_index, choice in enumerate(choices):
            validation.update(
                pairs[pair_index][1][choice][1]["quality"]["class_observations"]
            )
        score = 0.0
        for class_name in class_names:
            denominator = max(1.0, float(total[class_name]))
            score += abs(float(validation[class_name]) - denominator / 2.0) / denominator
        candidate = (score, tuple(choices), validation)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("无法生成平衡的 val/test 序列划分")

    _, choices, validation_counts = best
    assignment: Dict[str, str] = {}
    for pair_index, (_, candidates) in enumerate(pairs):
        validation_index = int(choices[pair_index])
        for candidate_index, (sequence_dir, meta) in enumerate(candidates):
            split = "val" if candidate_index == validation_index else "test"
            assignment[sequence_dir.name] = split
            if str(meta["split"]) != split:
                meta["split"] = split
                write_json(sequence_dir / "sequence_meta.json", meta)
                for annotation_path in sorted((sequence_dir / "annotations").glob("*.json")):
                    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                    payload["split"] = split
                    write_json(annotation_path, payload)

    test_counts = Counter(
        {
            class_name: int(total[class_name] - validation_counts[class_name])
            for class_name in class_names
        }
    )
    report = {
        "strategy": "sequence-level weather-stratified class balancing",
        "score": float(best[0]),
        "validation_counts": dict(validation_counts),
        "test_counts": dict(test_counts),
        "assignment": assignment,
    }
    write_json(root / "split_balance.json", report)
    return report


def rebuild_yolo_dataset(root: Path) -> Dict[str, int]:
    yolo = root / "yolo"
    if yolo.exists():
        shutil.rmtree(yolo)
    counts: Counter = Counter()
    sequence_lists: Dict[str, List[str]] = defaultdict(list)
    for split in ("train", "val", "test"):
        (yolo / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
    for sequence_dir in sorted((root / "sequences").glob("seq_*")):
        meta_path = sequence_dir / "sequence_meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        split = str(meta["split"])
        sequence_lists[split].append(sequence_dir.name)
        for image_path in sorted((sequence_dir / "img1").glob("*.png")):
            name = f"{sequence_dir.name}_{image_path.name}"
            label_path = sequence_dir / "labels_yolo" / f"{image_path.stem}.txt"
            hardlink_or_copy(image_path, yolo / "images" / split / name)
            hardlink_or_copy(label_path, yolo / "labels" / split / f"{Path(name).stem}.txt")
            counts[split] += 1
    (yolo / "data.yaml").write_text(
        f"path: {yolo.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: vehicle\n  1: pedestrian\n",
        encoding="utf-8",
    )
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (split_dir / f"{split}_sequences.txt").write_text(
            "\n".join(sequence_lists[split]) + ("\n" if sequence_lists[split] else ""),
            encoding="utf-8",
        )
    return {str(key): int(value) for key, value in counts.items()}


def audit_dataset(root: Path) -> Dict[str, Any]:
    errors: List[str] = []
    images_by_split: Counter = Counter()
    annotations_by_class: Counter = Counter()
    tracks_by_split: Dict[str, set] = defaultdict(set)
    track_lengths: Counter = Counter()
    equivalent_sides: Dict[str, List[float]] = defaultdict(list)
    hashes: Counter = Counter()
    sequence_reports: List[Dict[str, Any]] = []

    for sequence_dir in sorted((root / "sequences").glob("seq_*")):
        meta_path = sequence_dir / "sequence_meta.json"
        if not meta_path.is_file():
            errors.append(f"{sequence_dir.name}: missing sequence_meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        split = str(meta["split"])
        images = sorted((sequence_dir / "img1").glob("*.png"))
        labels = sorted((sequence_dir / "labels_yolo").glob("*.txt"))
        annotations = sorted((sequence_dir / "annotations").glob("*.json"))
        expected = int(meta["frames"])
        if not (len(images) == len(labels) == len(annotations) == expected):
            errors.append(
                f"{sequence_dir.name}: images={len(images)}, labels={len(labels)}, annotations={len(annotations)}, expected={expected}"
            )
        expected_names = [f"{index:06d}.png" for index in range(1, expected + 1)]
        if [path.name for path in images] != expected_names:
            errors.append(f"{sequence_dir.name}: frame filenames are not contiguous")
        mot_path = sequence_dir / "gt" / "gt.txt"
        if not mot_path.is_file():
            errors.append(f"{sequence_dir.name}: missing gt/gt.txt")
        images_by_split[split] += len(images)
        for image_path in images:
            hashes[hashlib.sha1(image_path.read_bytes()).hexdigest()] += 1
        observed_rows = 0
        for path in annotations:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame = int(payload["frame"])
            for annotation in payload["annotations"]:
                observed_rows += 1
                class_name = str(annotation["class_name"])
                annotations_by_class[class_name] += 1
                track_key = (sequence_dir.name, int(annotation["track_id"]))
                tracks_by_split[split].add(track_key)
                track_lengths[track_key] += 1
                equivalent_sides[class_name].append(
                    math.sqrt(float(annotation["bbox_xywh"][2]) * float(annotation["bbox_xywh"][3]))
                )
                if float(annotation["occlusion"]) > 0.500001:
                    errors.append(f"{sequence_dir.name}/{frame}: occlusion > 0.5")
        mot_rows = [line for line in mot_path.read_text(encoding="utf-8").splitlines() if line.strip()] if mot_path.is_file() else []
        if len(mot_rows) != observed_rows:
            errors.append(f"{sequence_dir.name}: MOT rows={len(mot_rows)}, annotation rows={observed_rows}")
        sequence_reports.append(
            {
                "sequence": sequence_dir.name,
                "split": split,
                "weather": meta["weather"],
                "motion_mode": meta["motion_mode"],
                "frames": len(images),
                "annotations": observed_rows,
                "unique_tracks": int(meta["quality"]["unique_tracks"]),
                "mean_objects_per_frame": float(meta["quality"]["mean_objects_per_frame"]),
            }
        )

    duplicate_images = sum(count - 1 for count in hashes.values() if count > 1)
    if duplicate_images:
        errors.append(f"发现 {duplicate_images} 张内容完全重复的图像")
    split_overlap = (
        tracks_by_split["train"] & tracks_by_split["val"]
        or tracks_by_split["train"] & tracks_by_split["test"]
        or tracks_by_split["val"] & tracks_by_split["test"]
    )
    if split_overlap:
        errors.append("train/val/test 之间出现序列身份重叠")
    for split in ("train", "val", "test"):
        if images_by_split[split] <= 0:
            errors.append(f"{split} 为空")

    def distribution(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "max": float(np.max(array)),
        }

    report = {
        "passed": not errors,
        "errors": errors,
        "sequence_count": len(sequence_reports),
        "images_by_split": dict(images_by_split),
        "total_images": int(sum(images_by_split.values())),
        "annotations_by_class": dict(annotations_by_class),
        "tracks_by_split": {key: len(value) for key, value in tracks_by_split.items()},
        "track_length_frames": distribution(list(track_lengths.values())),
        "equivalent_side_pixels": {
            key: distribution(values) for key, values in equivalent_sides.items()
        },
        "duplicate_images": duplicate_images,
        "sequences": sequence_reports,
    }
    write_json(root / "quality_audit.json", report)
    return report


def write_readme(root: Path, config: Dict[str, Any], audit: Dict[str, Any]) -> None:
    text = f"""# OpenHUTB UAV Single-Camera MOT RGB Dataset

本数据集用于无人机视角单相机多目标跟踪，只包含 `vehicle` 和 `pedestrian`。
深度与语义相机只参与遮挡和可见框计算，不作为公开模态保存。

## 目录

- `sequences/<seq>/img1`：连续 RGB 帧，编号从 1 开始。
- `sequences/<seq>/gt/gt.txt`：MOTChallenge 九列标注：`frame,id,x,y,w,h,conf,class,visibility`。
- `sequences/<seq>/labels_yolo`：YOLO 检测标签，类别 0=vehicle、1=pedestrian。
- `sequences/<seq>/annotations`：包含 CARLA actor ID、遮挡、相机位姿的逐帧 JSON。
- `yolo`：按完整序列划分的 YOLO train/val/test 数据。
- `qa_overlay`：每段序列抽样的可视化质检图。

## 采集约束

- 分辨率：{config['width']}x{config['height']}，帧率：{config['fps']} FPS。
- 遮挡超过 {100.0 * (1.0 - float(config['min_visible_ratio'])):.0f}% 的目标不标注。
- 两轮车辆和静态地图车辆不进入数据集。
- 训练/验证/测试按完整序列隔离，不随机拆分相邻帧。
- 总图像：{audit['total_images']}；质量审计通过：{audit['passed']}。

## 论文设计依据

- UAVDT: The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking, ECCV 2018.
- VisDrone-MOT2019: The Vision Meets Drone Multiple Object Tracking Challenge Results, ICCVW 2019.
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box, ECCV 2022.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args)
    # 复用三传感器同步工具时固定为一台相机。
    config["num_cameras"] = 1
    config["camera_bearing_separation_deg"] = 0.0
    config["min_objects_per_camera"] = 0
    config["min_anchor_class_per_camera"] = 0
    config["min_common_ids_per_frame"] = 0
    config["reject_boundary_annotations"] = True
    config["annotation_boundary_margin_px"] = 2
    validate_config(config)
    output = resolve_output(config)
    specs = build_sequence_specs(config)
    if args.max_sequences is not None:
        specs = specs[: int(args.max_sequences)]

    if args.audit_only:
        if not output.is_dir():
            raise FileNotFoundError(output)
        split_balance = rebalance_evaluation_splits(output, config)
        yolo_counts = rebuild_yolo_dataset(output)
        audit = audit_dataset(output)
        write_readme(output, config, audit)
        print(
            json.dumps(
                {"split_balance": split_balance, "yolo_images": yolo_counts, **audit},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if audit["passed"] else 2

    paths = prepare_output(output, args.overwrite)
    write_json(paths["root"] / "collection_config_used.json", config)
    started = time.time()
    results = run_collection(config, specs, paths)
    split_balance = rebalance_evaluation_splits(paths["root"], config)
    split_by_sequence = split_balance["assignment"]
    for result in results:
        result["split"] = split_by_sequence.get(result["sequence"], result["split"])
    yolo_counts = rebuild_yolo_dataset(paths["root"])
    audit = audit_dataset(paths["root"])
    write_readme(paths["root"], config, audit)
    write_json(
        paths["root"] / "dataset_manifest.json",
        {
            "dataset": paths["root"].name,
            "created_unix_time": time.time(),
            "elapsed_seconds": time.time() - started,
            "sequences": results,
            "split_balance": split_balance,
            "yolo_images": yolo_counts,
            "quality_audit": audit,
        },
    )
    print(json.dumps({"output": str(output), "yolo_images": yolo_counts, **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
