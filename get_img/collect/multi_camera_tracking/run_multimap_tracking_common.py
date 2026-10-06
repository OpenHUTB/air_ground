#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared simulator and map orchestration for the tracking collectors."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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
SPECIAL_MAP_WEATHERS = [
    weather for weather in WEATHERS if weather != "ClearNight"
]
DIFFICULT_MULTICAMERA_LAYOUT_MAPS = {
    "Town03_Opt",
    "Town04_Opt",
    "Town07_Opt",
    "Town10HD",
}
MAPS = list(runtime.ALL_OUTPUT_NAMES)
SINGLE_SEQUENCES_PER_MAP = 75
SINGLE_FRAMES_PER_SEQUENCE = 40
SINGLE_SAMPLE_INTERVAL_TICKS = 4
SINGLE_FPS = 20.0
MULTICAMERA_SCENES_PER_MAP = 30
MULTICAMERA_TRAIN_FRAMES_PER_SCENE = 50
MULTICAMERA_EVAL_FRAMES_PER_SCENE = 50
MULTICAMERA_SAMPLE_INTERVAL_TICKS = 3
MULTICAMERA_FPS = 25.0
CAMERAS_PER_MULTICAMERA_SAMPLE = 3

TASKS: Dict[str, Dict[str, Any]] = {
    "multicamera": {
        "collector": COLLECT_ROOT / "multi_camera_tracking" / "collect_uav_multicamera_mot_carla.py",
        "base_config": COLLECT_ROOT / "multi_camera_tracking" / "multi_camera_mot_config.json",
        "output": (
            GET_IMG_ROOT
            / "dataset_uav_multimap_multicamera_mot_motion_trial1"
        ),
    },
    "single_object": {
        "collector": COLLECT_ROOT / "single_camera_tracking" / "collect_uav_single_object_vot_carla.py",
        "base_config": COLLECT_ROOT / "single_camera_tracking" / "single_object_vot_config.json",
        "output": (
            GET_IMG_ROOT
            / "dataset_uav_multimap_single_object_vot_weather500"
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
        "--smoke",
        action="store_true",
        help="Collect one tiny scene/sequence into an isolated smoke directory.",
    )
    return parser.parse_args(argv)


def expected_images_by_weather(task: str, smoke: bool) -> Dict[str, int]:
    cameras = CAMERAS_PER_MULTICAMERA_SAMPLE if task == "multicamera" else 1
    group_count = (
        1
        if smoke
        else MULTICAMERA_SCENES_PER_MAP
        if task == "multicamera"
        else SINGLE_SEQUENCES_PER_MAP
    )
    counts = {weather: 0 for weather in WEATHERS}
    for group_index in range(group_count):
        if task == "multicamera":
            if smoke:
                frames = 20
            else:
                train_end = (
                    1
                    if group_count == 1
                    else int(group_count * 0.60)
                )
                frames = (
                    MULTICAMERA_TRAIN_FRAMES_PER_SCENE
                    if group_index < train_end
                    else MULTICAMERA_EVAL_FRAMES_PER_SCENE
                )
        else:
            frames = 3 if smoke else SINGLE_FRAMES_PER_SEQUENCE
        counts[WEATHERS[group_index % len(WEATHERS)]] += frames * cameras
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
            "weather_presets": list(WEATHERS),
            "include_existing_target_actors": True,
            "spectator_follow_camera": map_name == runtime.CCSP_OUTPUT_NAME,
        }
    )
    if map_name in {runtime.HUTB_MAP_NAME, runtime.CCSP_OUTPUT_NAME}:
        config["min_road_visible_ratio"] = 0.0

    if task == "multicamera":
        frames_per_scene = 20 if smoke else MULTICAMERA_TRAIN_FRAMES_PER_SCENE
        configured_seed = config.get("seed")
        if configured_seed is None:
            configured_seed = secrets.randbelow(2_147_483_646) + 1
            seed_source = "generated_by_multimap_runner"
        else:
            seed_source = "explicit"
        config.update(
            {
                "seed": int(configured_seed),
                "seed_source": seed_source,
                "scenes_per_map": MULTICAMERA_SCENES_PER_MAP,
                "fps": MULTICAMERA_FPS,
                "frames_per_scene": frames_per_scene,
                "train_frames_per_scene": frames_per_scene,
                "eval_frames_per_scene": (
                    frames_per_scene
                    if smoke
                    else MULTICAMERA_EVAL_FRAMES_PER_SCENE
                ),
                "sample_interval_ticks": MULTICAMERA_SAMPLE_INTERVAL_TICKS,
                "camera_bearing_separation_deg": 35.0,
                "camera_preferred_bearing_separation_min_deg": 42.0,
                "camera_preferred_bearing_separation_max_deg": 65.0,
                "camera_max_bearing_separation_deg": 90.0,
                # Camera bearings remain separated, but no map imposes a
                # minimum physical distance between camera origins.
                "camera_min_pair_distance_m": 0.0,
                "pedestrian_camera_radius_min_m": 18.0,
                "pedestrian_camera_radius_max_m": 32.0,
                "pedestrian_camera_height_min_m": 18.0,
                "pedestrian_camera_height_max_m": 26.0,
                "streaming_warmup_ticks": (
                    30 if map_name == runtime.CCSP_OUTPUT_NAME else 2
                ),
                # Sparse pedestrian layouts such as Town04 can need many camera
                # proposals while preserving the strict three-view QA gates.
                "max_scene_attempts": 12,
                "max_actor_population_attempts": 3,
            }
        )
        if map_name in {runtime.HUTB_MAP_NAME, runtime.CCSP_OUTPUT_NAME}:
            config.update(
                {
                    # Only the midnight preset is disabled. Sunset, fog, snow,
                    # and dust remain available on both special maps.
                    "weather_presets": list(SPECIAL_MAP_WEATHERS),
                    # These maps use the bridge pedestrians below. Disabling
                    # the second nav-mesh population prevents overlapping
                    # crowds and walkers being routed through driving lanes.
                    "walkers": 0,
                    # Moving vehicles come from the collector/Traffic Manager.
                    # The bridge only provides road-edge pedestrians, avoiding
                    # a second batch of static vehicles in the same picture.
                    "vehicles": 30,
                    "special_map_bridge_pedestrians": 24,
                    "special_map_bridge_vehicles": 0,
                    "special_map_pedestrian_motion": "forward",
                    "special_map_pedestrian_lateral_offset_m": 4.5,
                    "special_map_pedestrian_min_road_clearance_m": 2.0,
                    # Reject camera layouts that see the dark, unmodeled map
                    # boundary or another very large textureless region.
                    "reject_unmodeled_regions": True,
                    "max_dark_flat_region_ratio": 0.05,
                    "max_flat_region_ratio": 0.24,
                    "max_far_depth_region_ratio": 0.10,
                    "max_scene_attempts": 18,
                }
            )
        if map_name in DIFFICULT_MULTICAMERA_LAYOUT_MAPS:
            config.update(
                {
                    # Dense buildings and trees make the generic aerial-view
                    # gate reject otherwise useful three-camera trajectories.
                    "min_road_visible_ratio": 0.20,
                    "max_near_depth_ratio": 0.08,
                    "camera_bearing_separation_deg": 30.0,
                    "camera_preferred_bearing_separation_min_deg": 38.0,
                    "camera_preferred_bearing_separation_max_deg": 60.0,
                    "camera_max_bearing_separation_deg": 90.0,
                }
            )
        if smoke:
            config.update(
                {
                    "actor_spawn_warmup_ticks": 25,
                    "min_anchor_common_view_seconds": 0.24,
                    "min_effective_track_seconds": 0.6,
                    "vehicle_min_sequence_displacement_m": 0.5,
                    "vehicle_min_sequence_path_length_m": 0.75,
                    "pedestrian_min_sequence_displacement_m": 0.15,
                    "pedestrian_min_sequence_path_length_m": 0.25,
                }
            )
        if map_name == runtime.CCSP_OUTPUT_NAME:
            config.update(
                {
                    "pedestrian_camera_radius_min_m": 18.0,
                    "pedestrian_camera_radius_max_m": 32.0,
                    "pedestrian_camera_height_min_m": 18.0,
                    "pedestrian_camera_height_max_m": 26.0,
                    # CCSP boundary voids are often mid-gray rather than
                    # black, so its low-texture screen-space gate is stricter.
                    "max_flat_region_ratio": 0.14,
                }
            )
        if map_name == "Town04_Opt":
            config.update(
                {
                    "pedestrian_camera_radius_min_m": 18.0,
                    "pedestrian_camera_radius_max_m": 32.0,
                    "pedestrian_camera_height_min_m": 18.0,
                    "pedestrian_camera_height_max_m": 26.0,
                }
            )
        if map_name == "Town02_Opt":
            config.update(
                {
                    # Town02 roads are narrow and curved. Dense road samples
                    # and a wider pedestrian search annulus provide enough
                    # three-view candidates without weakening final image QA.
                    "camera_candidate_spacing_m": 3.0,
                    "pedestrian_camera_radius_min_m": 14.0,
                    "pedestrian_camera_radius_max_m": 38.0,
                    "pedestrian_camera_pitch_min_deg": -52.0,
                    "pedestrian_camera_pitch_max_deg": -30.0,
                    "vehicle_camera_radius_min_m": 18.0,
                    "vehicle_camera_radius_max_m": 32.0,
                    "vehicle_camera_height_min_m": 20.0,
                    "vehicle_camera_height_max_m": 26.0,
                    # Town02 has narrower/curvier roads. Keep the road-view QA
                    # enabled, but relax it enough that the third camera can
                    # still form a valid multi-view rig.
                    "min_road_visible_ratio": 0.25,
                    "camera_bearing_separation_deg": 30.0,
                    "camera_preferred_bearing_separation_min_deg": 38.0,
                    "camera_preferred_bearing_separation_max_deg": 60.0,
                    "camera_max_bearing_separation_deg": 90.0,
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
                "fps": SINGLE_FPS,
                "frames_per_sequence": frames_per_sequence,
                "sample_interval_ticks": SINGLE_SAMPLE_INTERVAL_TICKS,
                "streaming_warmup_frames": (
                    30 if map_name == runtime.CCSP_OUTPUT_NAME else 0
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


def accepted_multicamera_scene_count(map_output: Path) -> int:
    accepted = 0
    for quality_path in map_output.glob("scenes/*/scene_quality.json"):
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if bool(quality.get("passed", False)):
            accepted += 1
    return accepted


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_collector(
    task: str,
    python_executable: Path,
    config_path: Path,
    output: Path,
    log_path: Path,
    env: Optional[Dict[str, str]],
    smoke: bool,
    force_overwrite: bool = False,
) -> int:
    resume = (
        task == "multicamera"
        and not smoke
        and not force_overwrite
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
    with log_path.open("w", encoding="utf-8") as log_file:
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
                "[RECOVER] Collector exited non-zero; running offline audit "
                "and manifest rebuild for any successfully saved scenes.",
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
    force_overwrite: bool = False,
) -> int:
    map_output = output_root / map_name
    config = build_config(task, map_name, map_output, smoke)
    if map_name in {runtime.CCSP_OUTPUT_NAME, runtime.HUTB_MAP_NAME}:
        config["bridge_seed"] = secrets.randbelow(2_147_483_646) + 1
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
                count=int(config["special_map_bridge_pedestrians"]),
                vehicle_count=int(config["special_map_bridge_vehicles"]),
                env=env,
                seed=int(config["bridge_seed"]),
                pedestrian_motion=str(
                    config["special_map_pedestrian_motion"]
                ),
                lateral_offset=float(
                    config["special_map_pedestrian_lateral_offset_m"]
                ),
                min_road_clearance=float(
                    config["special_map_pedestrian_min_road_clearance_m"]
                ),
            )
            python_executable = runtime.CCSP_PYTHON
        else:
            runtime.start_simulator(runtime.GENERIC_SIMULATOR, visible=visible)
            runtime.wait_for_server(
                runtime.OPENHUTB_PYTHON,
                env=None,
                expected_server_prefix="23461d8a4",
            )
            python_executable = runtime.OPENHUTB_PYTHON
            if map_name == runtime.HUTB_MAP_NAME:
                bridge = runtime.start_static_pedestrian_bridge(
                    runtime.OPENHUTB_PYTHON,
                    count=int(config["special_map_bridge_pedestrians"]),
                    vehicle_count=int(config["special_map_bridge_vehicles"]),
                    map_name=map_name,
                    load_map=True,
                    seed=int(config["bridge_seed"]),
                    pedestrian_motion=str(
                        config["special_map_pedestrian_motion"]
                    ),
                    lateral_offset=float(
                        config["special_map_pedestrian_lateral_offset_m"]
                    ),
                    min_road_clearance=float(
                        config["special_map_pedestrian_min_road_clearance_m"]
                    ),
                )

        return run_collector(
            task,
            python_executable,
            config_path,
            map_output,
            log_path,
            env,
            smoke,
            force_overwrite,
        )
    finally:
        runtime.stop_static_pedestrian_bridge(bridge)
        runtime.stop_all_carla_processes()


def main_for_task(
    task: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    args = parse_args(task, argv)
    task_spec = TASKS[task]
    output_root = Path(str(task_spec["output"]) + ("_smoke" if args.smoke else ""))
    output_root.mkdir(parents=True, exist_ok=True)
    selected = list(args.only or MAPS)
    expected_by_weather = expected_images_by_weather(task, args.smoke)
    expected_images_per_map = sum(expected_by_weather.values())
    status_path = output_root / "collection_status.json"
    status: Dict[str, Any] = {
        "task": task,
        "started_at": now_iso(),
        "output_root": str(output_root),
        "weather_count": len(WEATHERS),
        "scenes_or_sequences_per_map": (
            1
            if args.smoke
            else MULTICAMERA_SCENES_PER_MAP
            if task == "multicamera"
            else SINGLE_SEQUENCES_PER_MAP
        ),
        "samples_per_scene_or_sequence": (
            20
            if args.smoke and task == "multicamera"
            else 3
            if args.smoke
            else MULTICAMERA_TRAIN_FRAMES_PER_SCENE
            if task == "multicamera"
            else SINGLE_FRAMES_PER_SEQUENCE
        ),
        "sample_interval_ticks": (
            MULTICAMERA_SAMPLE_INTERVAL_TICKS
            if task == "multicamera"
            else SINGLE_SAMPLE_INTERVAL_TICKS
        ),
        "expected_images_by_weather": expected_by_weather,
        "target_images_per_map": expected_images_per_map,
        "maps": {},
    }
    write_json(status_path, status)

    failed_maps = []

    for map_name in selected:
        map_output = output_root / map_name
        expected = int(status["target_images_per_map"])
        before = image_count(task, map_output)
        before_by_weather = image_counts_by_weather(task, map_output)
        accepted_scenes_before = (
            accepted_multicamera_scene_count(map_output)
            if task == "multicamera"
            else 0
        )
        count_complete = (
            accepted_scenes_before == MULTICAMERA_SCENES_PER_MAP
            if task == "multicamera"
            else before == expected
        )
        weather_complete = (
            count_complete
            if task == "multicamera"
            else before_by_weather == expected_by_weather
        )
        if (
            count_complete
            and weather_complete
            and audit_passed(task, map_output)
            and not args.rerun_complete
        ):
            if task == "multicamera":
                print(
                    f"[SKIP] {map_name}: accepted scenes="
                    f"{accepted_scenes_before}, images={before}",
                    flush=True,
                )
            else:
                print(f"[SKIP] {map_name}: {before}/{expected} images", flush=True)
            status["maps"][map_name] = {
                "state": "skipped_complete",
                "images": before,
                "images_by_weather": before_by_weather,
                "accepted_scenes": accepted_scenes_before,
            }
            write_json(status_path, status)
            continue

        status["maps"][map_name] = {
            "state": "running",
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
            args.rerun_complete,
        )
        after = image_count(task, map_output)
        after_by_weather = image_counts_by_weather(task, map_output)
        accepted_scenes_after = (
            accepted_multicamera_scene_count(map_output)
            if task == "multicamera"
            else 0
        )
        count_complete = (
            accepted_scenes_after == MULTICAMERA_SCENES_PER_MAP
            if task == "multicamera"
            else after == expected
        )
        weather_complete = (
            count_complete
            if task == "multicamera"
            else after_by_weather == expected_by_weather
        )
        passed = audit_passed(task, map_output)
        manifest_path = map_output / "dataset_manifest.json"
        fresh_manifest = (
            manifest_path.is_file()
            and manifest_path.stat().st_mtime >= collection_started - 1.0
        )
        state = (
            "complete"
            if (
                count_complete
                and weather_complete
                and passed
                and fresh_manifest
            )
            else "failed"
        )
        status["maps"][map_name] = {
            "state": state,
            "return_code": return_code,
            "quality_audit_passed": passed,
            "fresh_manifest": fresh_manifest,
            "images": after,
            "accepted_scenes": accepted_scenes_after,
            "images_by_weather": after_by_weather,
            "expected_images_by_weather": expected_by_weather,
            "expected_images": expected,
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
                f"accepted_scenes={accepted_scenes_after}; "
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
