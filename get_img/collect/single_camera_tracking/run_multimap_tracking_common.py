#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared simulator and map orchestration for the tracking collectors."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
COLLECT_ROOT = PROJECT_DIR.parent
GET_IMG_ROOT = COLLECT_ROOT.parent
MULTIMODAL_DIR = COLLECT_ROOT / "multimodal"
if str(MULTIMODAL_DIR) not in sys.path:
    sys.path.insert(0, str(MULTIMODAL_DIR))

import run_recommended_multimap_multimodal_collection as runtime

WEATHERS = [
    "ClearNoon",
    "ClearSunset",
    "ClearNight",
    "FoggyNoon",
    "SnowNoon",
    "DustStorm",
]
MAPS = list(runtime.ALL_OUTPUT_NAMES)
SINGLE_MOTION_MODES = ["fixed_hover", "lagged_follow", "lateral_orbit"]
SINGLE_TARGET_CLASSES_BY_MOTION_MODE = {
    mode: ["vehicle", "pedestrian"]
    for mode in SINGLE_MOTION_MODES
}
SINGLE_SEQUENCES_BY_MOTION_MODE = {
    "fixed_hover": 18,
    "lagged_follow": 18,
    "lateral_orbit": 18,
}
SINGLE_FRAMES_BY_MOTION_MODE = {
    "fixed_hover": 80,
    "lagged_follow": 120,
    "lateral_orbit": 150,
}
SINGLE_FRAMES_BY_MOTION_AND_CLASS = {
    "lagged_follow": {"pedestrian": 80},
    "lateral_orbit": {"pedestrian": 100},
}
SINGLE_SAMPLE_INTERVAL_TICKS_BY_TARGET_CLASS = {
    "vehicle": 2,
    "pedestrian": 6,
}
SINGLE_SEQUENCES_PER_MAP = sum(SINGLE_SEQUENCES_BY_MOTION_MODE.values())
SINGLE_FRAMES_PER_SEQUENCE = 120
SINGLE_SAMPLE_INTERVAL_TICKS = 2
SINGLE_FPS = 20.0
MULTICAMERA_SCENES_PER_MAP = 100
MULTICAMERA_FRAMES_PER_SCENE = 30
MULTICAMERA_SAMPLE_INTERVAL_TICKS = 3
MULTICAMERA_FPS = 25.0
CAMERAS_PER_MULTICAMERA_SAMPLE = 3
RUN_SEED = random.SystemRandom().randint(1, 1_900_000_000)

TASKS: Dict[str, Dict[str, Any]] = {
    "multicamera": {
        "collector": COLLECT_ROOT / "multi_camera_tracking" / "collect_uav_multicamera_mot_carla.py",
        "base_config": COLLECT_ROOT / "multi_camera_tracking" / "multi_camera_mot_config.json",
        "output": (
            GET_IMG_ROOT
            / "dataset_uav_multimap_multicamera_mot_weather500"
        ),
    },
    "single_object": {
        "collector": COLLECT_ROOT / "single_camera_tracking" / "collect_uav_single_object_vot_carla.py",
        "base_config": COLLECT_ROOT / "single_camera_tracking" / "single_object_vot_config.json",
        "output": (
            GET_IMG_ROOT
            / "dataset_uav_multimap_single_object_vot_motion_full"
        ),
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(task: str, argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Collect {task} tracking data from the eight release maps."
    )
    parser.add_argument("--only", nargs="+", choices=MAPS)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help=(
            "Use this run seed for a fresh/rebuilt collection. "
            "Omit it to generate a new random seed."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Collect one tiny scene/sequence into an isolated smoke directory.",
    )
    return parser.parse_args(argv)


def weather_presets_for_map(task: str, map_name: str) -> Sequence[str]:
    if task == "single_object":
        tracking_weathers = ["ClearNoon", "ClearSunset", "ClearNight"]
        if map_name in {
            runtime.HUTB_MAP_NAME,
            runtime.CCSP_OUTPUT_NAME,
        }:
            return tracking_weathers[:2]
        return tracking_weathers
    return list(WEATHERS)


_WEATHER_SCHEDULE_CACHE: Dict[str, Sequence[str]] = {}


def map_collection_seed(map_name: str) -> int:
    return 1 + (
        RUN_SEED + (MAPS.index(map_name) + 1) * 100_003
    ) % 2_000_000_000


def map_weather_seed(map_name: str, smoke: bool) -> int:
    return 1 + (
        RUN_SEED
        + (MAPS.index(map_name) + 1) * 10_007
        + (1_000_003 if smoke else 0)
    ) % 2_000_000_000


def single_motion_mode_schedule() -> Sequence[str]:
    max_count = max(SINGLE_SEQUENCES_BY_MOTION_MODE.values())
    return [
        mode
        for round_index in range(max_count)
        for mode in SINGLE_MOTION_MODES
        if round_index < SINGLE_SEQUENCES_BY_MOTION_MODE[mode]
    ]


def random_weather_schedule_for_map(
    map_name: str,
    smoke: bool,
) -> Sequence[str]:
    cache_key = f"{map_name}|smoke={int(smoke)}"
    if cache_key not in _WEATHER_SCHEDULE_CACHE:
        allowed = list(weather_presets_for_map("single_object", map_name))
        weather_rng = random.Random(map_weather_seed(map_name, smoke))
        _WEATHER_SCHEDULE_CACHE[cache_key] = [
            weather_rng.choice(allowed)
            for _ in range(SINGLE_SEQUENCES_PER_MAP)
        ]
    return list(_WEATHER_SCHEDULE_CACHE[cache_key])


def expected_images_by_weather(
    task: str,
    smoke: bool,
    map_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    map_weathers = weather_presets_for_map(task, map_name)
    if task == "single_object":
        counts = {weather: 0 for weather in WEATHERS}
        if config is None:
            schedule = list(single_motion_mode_schedule())
            mode_frames = SINGLE_FRAMES_BY_MOTION_MODE
            class_frames = SINGLE_FRAMES_BY_MOTION_AND_CLASS
            class_offset = MAPS.index(map_name) % 2
            weather_schedule = random_weather_schedule_for_map(map_name, smoke)
        else:
            modes = list(config["motion_modes"])
            mode_counts = config["sequences_by_motion_mode"]
            max_count = max(int(mode_counts[mode]) for mode in modes)
            schedule = [
                mode
                for round_index in range(max_count)
                for mode in modes
                if round_index < int(mode_counts[mode])
            ]
            mode_frames = config["frames_by_motion_mode"]
            class_frames = config.get(
                "frames_by_motion_mode_and_target_class", {}
            )
            class_offset = int(config.get("target_class_offset", 0))
            weather_schedule = list(config["weather_sequence_schedule"])
        if smoke:
            schedule = schedule[:1]
        mode_occurrences: Dict[str, int] = {}
        target_classes_by_mode = (
            SINGLE_TARGET_CLASSES_BY_MOTION_MODE
            if config is None
            else config.get(
                "target_classes_by_motion_mode",
                SINGLE_TARGET_CLASSES_BY_MOTION_MODE,
            )
        )
        for group_index, mode in enumerate(schedule):
            occurrence = mode_occurrences.get(mode, 0)
            target_classes = list(target_classes_by_mode[mode])
            target_class = target_classes[
                (occurrence + class_offset) % len(target_classes)
            ]
            mode_occurrences[mode] = occurrence + 1
            planned_frames = (
                class_frames.get(mode, {}).get(
                    target_class,
                    mode_frames[mode],
                )
            )
            selected_weather = weather_schedule[group_index]
            counts[selected_weather] += (
                3
                if smoke
                else planned_frames
            )
        return counts
    frames = (
        2
        if smoke and task == "multicamera"
        else MULTICAMERA_FRAMES_PER_SCENE
    )
    cameras = CAMERAS_PER_MULTICAMERA_SAMPLE
    group_count = 1 if smoke else MULTICAMERA_SCENES_PER_MAP
    counts = {weather: 0 for weather in WEATHERS}
    for group_index in range(group_count):
        counts[map_weathers[group_index % len(map_weathers)]] += frames * cameras
    return counts


def build_config(
    task: str,
    map_name: str,
    output: Path,
    smoke: bool,
) -> Dict[str, Any]:
    task_spec = TASKS[task]
    config = json.loads(
        task_spec["base_config"].read_text(encoding="utf-8")
    )
    config.update(
        {
            "out": str(output),
            "map": (
                None
                if map_name in {
                    runtime.HUTB_MAP_NAME,
                    runtime.CCSP_OUTPUT_NAME,
                }
                else map_name
            ),
            "timeout": 300.0,
            "sensor_timeout": 60.0,
            "weather_presets": weather_presets_for_map(task, map_name),
            "run_seed": RUN_SEED,
            "seed": map_collection_seed(map_name),
            "include_existing_target_actors": False,
            "include_existing_target_vehicle_actors": False,
            "include_existing_target_pedestrian_actors": False,
            "spectator_follow_camera": map_name == runtime.CCSP_OUTPUT_NAME,
        }
    )
    if map_name in {runtime.HUTB_MAP_NAME, runtime.CCSP_OUTPUT_NAME}:
        config["min_road_visible_ratio"] = 0.0
    if (
        task == "single_object"
        and map_name in {runtime.HUTB_MAP_NAME, runtime.CCSP_OUTPUT_NAME}
    ):
        config.update(
            {
                # These maps need fallback walkers because their navigation
                # meshes are absent or unreliable. Avoid creating a second
                # crowd here, and keep Traffic Manager traffic sparse.
                "vehicles": 20,
                "walkers": 0,
                "include_existing_target_pedestrian_actors": True,
                "require_streaming_geometry": True,
                "streaming_geometry_max_depth_m": 220.0,
                "streaming_min_geometry_ratio": 0.90,
            }
        )
    if task == "single_object" and map_name == runtime.CCSP_OUTPUT_NAME:
        config.update(
            {
                "height_min": 18.0,
                "height_max": 24.0,
                "radius_min": 18.0,
                "radius_max": 32.0,
                "excluded_camera_circles": [
                    [884.08, 601.00, 220.0],
                ],
            }
        )

    if task == "multicamera":
        frames_per_scene = 2 if smoke else MULTICAMERA_FRAMES_PER_SCENE
        config.update(
            {
                "scenes_per_map": MULTICAMERA_SCENES_PER_MAP,
                "fps": MULTICAMERA_FPS,
                "frames_per_scene": frames_per_scene,
                "train_frames_per_scene": frames_per_scene,
                "eval_frames_per_scene": frames_per_scene,
                "sample_interval_ticks": MULTICAMERA_SAMPLE_INTERVAL_TICKS,
                "camera_bearing_separation_deg": 35.0,
                "camera_target_bearing_separation_deg": 42.0,
                "camera_max_bearing_separation_deg": 100.0,
                "camera_pitch_targets_deg": [-36.0, -40.0, -44.0],
                "pedestrian_camera_radius_min_m": 18.0,
                "pedestrian_camera_radius_max_m": 32.0,
                "pedestrian_camera_height_min_m": 16.0,
                "pedestrian_camera_height_max_m": 24.0,
                "streaming_warmup_ticks": (
                    30 if map_name == runtime.CCSP_OUTPUT_NAME else 2
                ),
                # Sparse pedestrian layouts such as Town04 can need many camera
                # proposals while preserving the strict three-view QA gates.
                "max_scene_attempts": 300,
                "max_actor_population_attempts": 4,
            }
        )
        if map_name == runtime.CCSP_OUTPUT_NAME:
            config.update(
                {
                    "pedestrian_camera_radius_min_m": 18.0,
                    "pedestrian_camera_radius_max_m": 32.0,
                    "pedestrian_camera_height_min_m": 16.0,
                    "pedestrian_camera_height_max_m": 24.0,
                }
            )
        if map_name == "Town04_Opt":
            config.update(
                {
                    "pedestrian_camera_radius_min_m": 18.0,
                    "pedestrian_camera_radius_max_m": 32.0,
                    "pedestrian_camera_height_min_m": 16.0,
                    "pedestrian_camera_height_max_m": 24.0,
                }
            )
        if map_name == "Town02_Opt":
            config.update(
                {
                    "vehicle_camera_radius_min_m": 18.0,
                    "vehicle_camera_radius_max_m": 32.0,
                    "vehicle_camera_height_min_m": 18.0,
                    "vehicle_camera_height_max_m": 24.0,
                    # Town02 has narrower/curvier roads. Keep the road-view QA
                    # enabled, but relax it enough that the third camera can
                    # still form a valid multi-view rig.
                    "min_road_visible_ratio": 0.25,
                    "camera_bearing_separation_deg": 30.0,
                    "camera_target_bearing_separation_deg": 38.0,
                    "camera_max_bearing_separation_deg": 100.0,
                    # Give Town02 additional fresh actor populations before
                    # declaring a scene unavailable.
                    "max_actor_population_attempts": 6,
                }
            )
    else:
        frames_per_sequence = 3 if smoke else SINGLE_FRAMES_PER_SEQUENCE
        config.update(
            {
                "sequences_per_map": SINGLE_SEQUENCES_PER_MAP,
                "motion_modes": list(SINGLE_MOTION_MODES),
                "target_classes_by_motion_mode": {
                    mode: list(target_classes)
                    for mode, target_classes in (
                        SINGLE_TARGET_CLASSES_BY_MOTION_MODE.items()
                    )
                },
                "max_sequences_per_target_actor": 2,
                "sequences_by_motion_mode": dict(
                    SINGLE_SEQUENCES_BY_MOTION_MODE
                ),
                "frames_by_motion_mode": {
                    mode: 3
                    for mode in SINGLE_MOTION_MODES
                }
                if smoke
                else dict(SINGLE_FRAMES_BY_MOTION_MODE),
                "frames_by_motion_mode_and_target_class": (
                    {}
                    if smoke
                    else {
                        mode: dict(class_frames)
                        for mode, class_frames in (
                            SINGLE_FRAMES_BY_MOTION_AND_CLASS.items()
                        )
                    }
                ),
                "fps": SINGLE_FPS,
                "frames_per_sequence": frames_per_sequence,
                "sample_interval_ticks": SINGLE_SAMPLE_INTERVAL_TICKS,
                "sample_interval_ticks_by_target_class": dict(
                    SINGLE_SAMPLE_INTERVAL_TICKS_BY_TARGET_CLASS
                ),
                "weather_assignment": "random_per_run",
                "weather_assignment_seed": map_weather_seed(map_name, smoke),
                "weather_sequence_schedule": list(
                    random_weather_schedule_for_map(map_name, smoke)
                ),
                "target_class_offset": MAPS.index(map_name) % 2,
                "streaming_warmup_frames": (
                    30
                    if map_name in {
                        runtime.HUTB_MAP_NAME,
                        runtime.CCSP_OUTPUT_NAME,
                    }
                    else 0
                ),
                "max_sequence_attempts": 48,
            }
        )
    return config


def image_count(task: str, map_output: Path) -> int:
    if not map_output.exists():
        return 0
    if task == "multicamera":
        return sum(
            1 for _ in map_output.glob("scenes/*/cameras/*/rgb/*.png")
        )
    return sum(1 for _ in map_output.glob("vot/*/color/*.png"))


def image_counts_by_weather(task: str, map_output: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for weather in WEATHERS:
        prefix = weather.lower() + "_"
        if task == "multicamera":
            pattern = f"scenes/{prefix}*/cameras/*/rgb/*.png"
        else:
            pattern = f"vot/{prefix}*/color/*.png"
        counts[weather] = sum(1 for _ in map_output.glob(pattern))
    return counts


def audit_passed(task: str, map_output: Path) -> bool:
    audit_path = map_output / "quality_audit.json"
    if not audit_path.is_file():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if task == "multicamera":
        return bool(audit.get("passed", False))
    return str(audit.get("status", "")).upper() == "PASS"


def single_object_plan_is_complete(
    map_output: Path,
    map_name: str,
    smoke: bool,
) -> bool:
    manifest_path = map_output / "dataset_manifest.json"
    if not manifest_path.is_file() or not audit_passed("single_object", map_output):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected_config = build_config("single_object", map_name, map_output, smoke)
    actual_config = manifest.get("config", {})
    plan_keys = (
        "motion_modes",
        "target_classes_by_motion_mode",
        "sequences_by_motion_mode",
        "frames_by_motion_mode",
        "frames_by_motion_mode_and_target_class",
        "sample_interval_ticks_by_target_class",
        "fixed_hover_stop_after_absent_frames",
        "min_frames_by_motion_mode",
        "max_sequences_per_target_actor",
        "min_target_average_speed_mps",
        "min_target_moving_step_ratio",
        "target_moving_step_threshold_m",
        "include_existing_target_vehicle_actors",
        "include_existing_target_pedestrian_actors",
        "weather_presets",
        "weather_assignment",
    )
    if any(actual_config.get(key) != expected_config.get(key) for key in plan_keys):
        return False
    expected_sequences = 1 if smoke else SINGLE_SEQUENCES_PER_MAP
    sequence_count = sum(
        1 for path in (map_output / "vot").glob("*") if path.is_dir()
    )
    return (
        int(manifest.get("sequence_count", -1)) == expected_sequences
        and sequence_count == expected_sequences
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


RESUME_CONFIG_IGNORED_KEYS = {
    "out",
    "timeout",
    "sensor_timeout",
    "run_seed",
    "seed",
    "weather_assignment_seed",
    "weather_sequence_schedule",
    "_config_path",
}


def configs_are_resume_compatible(
    saved: Dict[str, Any],
    current: Dict[str, Any],
) -> bool:
    saved_plan = {
        key: value
        for key, value in saved.items()
        if key not in RESUME_CONFIG_IGNORED_KEYS
    }
    current_plan = {
        key: value
        for key, value in current.items()
        if key not in RESUME_CONFIG_IGNORED_KEYS
    }
    return saved_plan == current_plan


def resolve_map_config(
    task: str,
    map_name: str,
    output_root: Path,
    smoke: bool,
) -> Tuple[Dict[str, Any], bool]:
    map_output = output_root / map_name
    current = build_config(task, map_name, map_output, smoke)
    config_path = output_root / "_configs" / f"{map_name}.json"
    has_sequence_output = (
        task == "single_object"
        and not smoke
        and (map_output / "vot").is_dir()
        and any(path.is_dir() for path in (map_output / "vot").iterdir())
    )
    if not has_sequence_output or not config_path.is_file():
        return current, False
    try:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return current, False
    if not configs_are_resume_compatible(saved, current):
        return current, False

    saved.update(
        {
            "out": str(map_output),
            "timeout": current["timeout"],
            "sensor_timeout": current["sensor_timeout"],
        }
    )
    return saved, True


def run_collector(
    task: str,
    python_executable: Path,
    config_path: Path,
    output: Path,
    log_path: Path,
    env: Optional[Dict[str, str]],
    smoke: bool,
    resume: bool = False,
) -> int:
    resume = resume or (
        task == "multicamera"
        and not smoke
        and (output / "scenes").is_dir()
        and any((output / "scenes").iterdir())
    )
    command = [
        str(python_executable),
        "-u",
        str(TASKS[task]["collector"]),
        "--config",
        str(config_path),
        "--out",
        str(output),
        "--resume" if resume else "--overwrite",
    ]
    if smoke:
        command.extend(
            ["--max-scenes", "1"]
            if task == "multicamera"
            else ["--max-sequences", "1"]
        )
    print("[COLLECT] " + subprocess.list2cmdline(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if resume else "w", encoding="utf-8") as log_file:
        if resume:
            log_file.write(
                f"\n[RESUME RUN {now_iso()}]\n"
            )
            log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(TASKS[task]["collector"].parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

        if task == "multicamera" and return_code != 0:
            audit_command = list(command)
            audit_command.remove("--resume" if resume else "--overwrite")
            audit_command.append("--audit-only")
            print(
                "[RECOVER] Collector exited during native cleanup; "
                "running offline audit and manifest rebuild.",
                flush=True,
            )
            audit = subprocess.run(
                audit_command,
                cwd=str(TASKS[task]["collector"].parent),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                check=False,
            )
            print(audit.stdout, end="", flush=True)
            log_file.write("\n[OFFLINE AUDIT]\n")
            log_file.write(audit.stdout)
            log_file.flush()
            if audit.returncode != 0:
                print(
                    f"[RECOVER] Offline audit failed with {audit.returncode}.",
                    flush=True,
                )
        return return_code


def collect_map(
    task: str,
    map_name: str,
    output_root: Path,
    visible: bool,
    smoke: bool,
    config: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> int:
    map_output = output_root / map_name
    if config is None:
        config = build_config(task, map_name, map_output, smoke)
    config_path = output_root / "_configs" / f"{map_name}.json"
    log_path = output_root / "_logs" / f"{map_name}.log"
    write_json(config_path, config)

    bridge: Optional[subprocess.Popen] = None
    env: Optional[Dict[str, str]] = None
    try:
        runtime.stop_all_carla_processes()
        if map_name == runtime.CCSP_OUTPUT_NAME:
            runtime.start_simulator(
                runtime.CCSP_SIMULATOR,
                visible=True,
                full_assets=True,
            )
            env = runtime.ccsp_environment()
            runtime.wait_for_server(
                runtime.CCSP_PYTHON,
                env,
                expected_server_prefix="7b2ff0449",
            )
            bridge = runtime.start_static_pedestrian_bridge(
                runtime.CCSP_PYTHON,
                count=40 if task == "single_object" else 120,
                vehicle_count=0 if task == "single_object" else 40,
                env=env,
                seed=int(config["seed"]) + 9301,
                pedestrian_motion=(
                    "shuttle" if task == "single_object" else "static"
                ),
                shuttle_distance=10.0,
            )
            python_executable = runtime.CCSP_PYTHON
        else:
            runtime.start_simulator(
                runtime.GENERIC_SIMULATOR,
                visible=visible,
                full_assets=(
                    task == "single_object"
                    and map_name == runtime.HUTB_MAP_NAME
                ),
            )
            runtime.wait_for_server(
                runtime.OPENHUTB_PYTHON,
                env=None,
                expected_server_prefix="23461d8a4",
            )
            python_executable = runtime.OPENHUTB_PYTHON
            if map_name == runtime.HUTB_MAP_NAME:
                bridge = runtime.start_static_pedestrian_bridge(
                    runtime.OPENHUTB_PYTHON,
                    count=40 if task == "single_object" else 120,
                    vehicle_count=0 if task == "single_object" else 40,
                    map_name=map_name,
                    load_map=True,
                    seed=int(config["seed"]) + 9201,
                    pedestrian_motion=(
                        "shuttle" if task == "single_object" else "static"
                    ),
                    shuttle_distance=10.0,
                )

        return run_collector(
            task,
            python_executable,
            config_path,
            map_output,
            log_path,
            env,
            smoke,
            resume=resume,
        )
    finally:
        runtime.stop_static_pedestrian_bridge(bridge)
        runtime.stop_all_carla_processes()


def main_for_task(
    task: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    global RUN_SEED
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    args = parse_args(task, argv)
    if args.run_seed is not None:
        if not 1 <= int(args.run_seed) <= 1_900_000_000:
            raise ValueError("--run-seed 必须在 1 到 1900000000 之间")
        RUN_SEED = int(args.run_seed)
        _WEATHER_SCHEDULE_CACHE.clear()
    task_spec = TASKS[task]
    output_root = Path(str(task_spec["output"]) + ("_smoke" if args.smoke else ""))
    output_root.mkdir(parents=True, exist_ok=True)
    selected = list(args.only or MAPS)
    resolved_configs: Dict[str, Dict[str, Any]] = {}
    resume_by_map: Dict[str, bool] = {}
    for map_name in selected:
        if args.rerun_complete:
            resolved_configs[map_name] = build_config(
                task,
                map_name,
                output_root / map_name,
                args.smoke,
            )
            resume_by_map[map_name] = False
        else:
            config, resume = resolve_map_config(
                task,
                map_name,
                output_root,
                args.smoke,
            )
            resolved_configs[map_name] = config
            resume_by_map[map_name] = resume
    expected_by_map = {
        map_name: expected_images_by_weather(
            task,
            args.smoke,
            map_name,
            config=resolved_configs[map_name],
        )
        for map_name in selected
    }
    planned_max_by_map = {
        map_name: sum(counts.values())
        for map_name, counts in expected_by_map.items()
    }
    status_path = output_root / "collection_status.json"
    status: Dict[str, Any] = {
        "task": task,
        "started_at": now_iso(),
        "output_root": str(output_root),
        "session_run_seed": RUN_SEED,
        "run_seed_source": (
            "command_line" if args.run_seed is not None else "system_random"
        ),
        "map_run_seeds": {
            map_name: resolved_configs[map_name].get("run_seed", RUN_SEED)
            for map_name in selected
        },
        "map_seeds": {
            map_name: int(resolved_configs[map_name]["seed"])
            for map_name in selected
        },
        "weather_assignment_seeds": {
            map_name: resolved_configs[map_name].get("weather_assignment_seed")
            for map_name in selected
        },
        "weather_count_by_map": {
            map_name: len(weather_presets_for_map(task, map_name))
            for map_name in selected
        },
        "scenes_or_sequences_per_map": (
            1
            if args.smoke
            else MULTICAMERA_SCENES_PER_MAP
            if task == "multicamera"
            else SINGLE_SEQUENCES_PER_MAP
        ),
        "samples_per_scene_or_sequence": (
            2
            if args.smoke and task == "multicamera"
            else 3
            if args.smoke
            else MULTICAMERA_FRAMES_PER_SCENE
            if task == "multicamera"
            else dict(SINGLE_FRAMES_BY_MOTION_MODE)
        ),
        "class_specific_samples_per_sequence": (
            None
            if task == "multicamera" or args.smoke
            else {
                mode: dict(class_frames)
                for mode, class_frames in (
                    SINGLE_FRAMES_BY_MOTION_AND_CLASS.items()
                )
            }
        ),
        "sample_interval_ticks": (
            MULTICAMERA_SAMPLE_INTERVAL_TICKS
            if task == "multicamera"
            else dict(SINGLE_SAMPLE_INTERVAL_TICKS_BY_TARGET_CLASS)
        ),
        "weather_presets_by_map": {
            map_name: weather_presets_for_map(task, map_name)
            for map_name in selected
        },
        "expected_images_by_map_and_weather": expected_by_map,
        "planned_max_images_by_map": planned_max_by_map,
        "maps": {},
    }
    write_json(status_path, status)

    failed_maps = []

    for map_name in selected:
        map_output = output_root / map_name
        expected_by_weather = expected_by_map[map_name]
        expected = planned_max_by_map[map_name]
        before = image_count(task, map_output)
        before_by_weather = image_counts_by_weather(task, map_output)
        weather_complete = before_by_weather == expected_by_weather
        existing_complete = (
            single_object_plan_is_complete(map_output, map_name, args.smoke)
            if task == "single_object"
            else before == expected and weather_complete
        )
        if (
            existing_complete
            and not args.rerun_complete
        ):
            print(f"[SKIP] {map_name}: {before} images", flush=True)
            status["maps"][map_name] = {
                "state": "skipped_complete",
                "resumed": False,
                "run_seed": resolved_configs[map_name].get("run_seed"),
                "map_seed": resolved_configs[map_name].get("seed"),
                "weather_assignment_seed": resolved_configs[map_name].get(
                    "weather_assignment_seed"
                ),
                "images": before,
                "images_by_weather": before_by_weather,
            }
            write_json(status_path, status)
            continue

        status["maps"][map_name] = {
            "state": "running",
            "resumed": resume_by_map[map_name],
            "run_seed": resolved_configs[map_name].get("run_seed"),
            "map_seed": resolved_configs[map_name].get("seed"),
            "weather_assignment_seed": resolved_configs[map_name].get(
                "weather_assignment_seed"
            ),
            "started_at": now_iso(),
            "images_before": before,
        }
        write_json(status_path, status)
        collection_started = time.time()
        return_code = collect_map(
            task,
            map_name,
            output_root,
            args.visible,
            args.smoke,
            config=resolved_configs[map_name],
            resume=resume_by_map[map_name],
        )
        after = image_count(task, map_output)
        after_by_weather = image_counts_by_weather(task, map_output)
        weather_complete = after_by_weather == expected_by_weather
        manifest_path = map_output / "dataset_manifest.json"
        audit_path = map_output / "quality_audit.json"
        fresh_manifest = (
            manifest_path.is_file()
            and manifest_path.stat().st_mtime >= collection_started - 1.0
        )
        fresh_audit = (
            audit_path.is_file()
            and audit_path.stat().st_mtime >= collection_started - 1.0
        )
        passed = fresh_audit and audit_passed(task, map_output)
        expected_sequences = (
            1
            if args.smoke
            else MULTICAMERA_SCENES_PER_MAP
            if task == "multicamera"
            else SINGLE_SEQUENCES_PER_MAP
        )
        actual_sequences = (
            sum(1 for path in (map_output / "vot").glob("*") if path.is_dir())
            if task == "single_object"
            else sum(1 for path in (map_output / "scenes").glob("*") if path.is_dir())
        )
        count_complete = (
            actual_sequences == expected_sequences
            if task == "single_object"
            else after == expected and weather_complete
        )
        state = (
            "complete"
            if (
                count_complete
                and passed
                and fresh_manifest
            )
            else "failed"
        )
        status["maps"][map_name] = {
            "state": state,
            "resumed": resume_by_map[map_name],
            "run_seed": resolved_configs[map_name].get("run_seed"),
            "map_seed": resolved_configs[map_name].get("seed"),
            "weather_assignment_seed": resolved_configs[map_name].get(
                "weather_assignment_seed"
            ),
            "return_code": return_code,
            "quality_audit_passed": passed,
            "fresh_manifest": fresh_manifest,
            "fresh_quality_audit": fresh_audit,
            "images": after,
            "images_by_weather": after_by_weather,
            "sequences_or_scenes": actual_sequences,
            "expected_sequences_or_scenes": expected_sequences,
            "expected_images_by_weather": expected_by_weather,
            "planned_max_images": expected,
            "finished_at": now_iso(),
        }
        write_json(status_path, status)
        if state == "complete" and return_code != 0:
            print(
                f"[WARN] {map_name}: collector exited with {return_code} after "
                "writing a complete dataset; count and quality audit passed.",
                flush=True,
            )
        if state != "complete":
            failed_maps.append(map_name)
            print(
                f"[WARN] {map_name}: collection incomplete, "
                f"collector rc={return_code}, images={after}, "
                f"groups={actual_sequences}/{expected_sequences}; "
                f"continuing with the next map. "
                f"log={output_root / '_logs' / (map_name + '.log')}",
                flush=True,
            )
            continue

    status["state"] = (
        "complete_with_failures" if failed_maps else "complete"
    )
    status["failed_maps"] = failed_maps
    status["finished_at"] = now_iso()
    write_json(status_path, status)
    if failed_maps:
        print(
            f"[DONE WITH WARNINGS] {task}: failed_maps={failed_maps}; "
            f"output={output_root}",
            flush=True,
        )
    else:
        print(f"[DONE] {task}: {output_root}", flush=True)
    return 0
