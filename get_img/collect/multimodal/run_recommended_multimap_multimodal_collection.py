#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the existing multimodal collector over the recommended map set.

This file manages simulator processes, CARLA API selection, map loading,
output directories, and resume checks. Standard maps use the base collector;
CCSP uses its dedicated balanced quality wrapper and RGB streaming gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_DIR = Path(__file__).resolve().parent
GET_IMG_ROOT = PROJECT_DIR.parent.parent
COLLECTOR = PROJECT_DIR / "collect_rpg_small_targets_carla_v2.py"
CCSP_QUALITY_COLLECTOR = (
    PROJECT_DIR / "collect_ccsp_fullmap_pose_diversity_carla.py"
)
CONFIG = PROJECT_DIR / "collection_config.json"
STATIC_PEDESTRIAN_HELPER = PROJECT_DIR / "prepare_static_pedestrians_carla.py"
OUTPUT_ROOT = (
    GET_IMG_ROOT
    / "AirGroundCoopSuite"
    / "release"
    / "derived_task1_detection"
)

OPENHUTB_PYTHON = Path(r"D:\anaconda2023.09\envs\openhutb\python.exe")
GENERIC_SIMULATOR = Path(r"E:\OpenHUTB\中电软件园\hutb_windows_v2.10.0\CarlaUE4.exe")

CCSP_PYTHON_ROOT = Path(r"E:\OpenHUTB\中电软件园\carla_0.9.15_py37")
CCSP_PYTHON = CCSP_PYTHON_ROOT / "python.exe"
CCSP_CARLA_EGG = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor"
    r"\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
)
CCSP_SIMULATOR = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor\CarlaUE4.exe"
)

# Keep these assignments below the legacy path literals so the script remains
# usable even if an editor previously decoded the Chinese directory incorrectly.
GENERIC_SIMULATOR = Path(
    r"E:\OpenHUTB\中电软件园\hutb_windows_v2.10.0\CarlaUE4.exe"
)
CCSP_PYTHON_ROOT = Path(r"E:\OpenHUTB\中电软件园\carla_0.9.15_py37")
CCSP_PYTHON = CCSP_PYTHON_ROOT / "python.exe"
CCSP_CARLA_EGG = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor"
    r"\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
)
CCSP_SIMULATOR = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor\CarlaUE4.exe"
)

OPENHUTB_ASSET_ROOT = Path("E:/OpenHUTB") / "\u4e2d\u7535\u8f6f\u4ef6\u56ed"
GENERIC_SIMULATOR = OPENHUTB_ASSET_ROOT / "hutb_windows_v2.10.0" / "CarlaUE4.exe"
CCSP_PYTHON_ROOT = OPENHUTB_ASSET_ROOT / "carla_0.9.15_py37"
CCSP_PYTHON = CCSP_PYTHON_ROOT / "python.exe"
CCSP_CARLA_EGG = (
    OPENHUTB_ASSET_ROOT
    / "WindowsNoEditor"
    / "WindowsNoEditor"
    / "PythonAPI"
    / "carla"
    / "dist"
    / "carla-0.9.15-py3.7-win-amd64.egg"
)
CCSP_SIMULATOR = (
    OPENHUTB_ASSET_ROOT / "WindowsNoEditor" / "WindowsNoEditor" / "CarlaUE4.exe"
)

TOWN_MAPS = [
    "Town03_Opt",
    "Town05_Opt",
    # This v2.10 package starts directly in Town10HD. Loading Town10HD_Opt via
    # RPC crashes its streaming server, so use the equivalent resident map.
    "Town10HD",
    # Town12 crashes this OpenHUTB v2.10 build when the synchronized sensor
    # streams start and exposes no pedestrian navigation locations. Town02_Opt
    # is the verified fallback so the released set still has eight usable maps.
    "Town02_Opt",
    "Town04_Opt",
    "Town07_Opt",
]
HUTB_MAP_NAME = "HutbCarlaCity"
GENERIC_MAPS = TOWN_MAPS + [HUTB_MAP_NAME]
CCSP_OUTPUT_NAME = "CCSP_Zhongdian_Software_Park"
ALL_OUTPUT_NAMES = GENERIC_MAPS + [CCSP_OUTPUT_NAME]

# Sparse maps use short runs so pedestrians do not disperse too far from the
# useful road region. The lower road ratio retains mountain/rural context.
SHORT_BATCH_MAPS = {"Town04_Opt": 100, "Town07_Opt": 100}

DEFAULT_TOWN_FRAMES = 600
DEFAULT_HUTB_FRAMES = 300
DEFAULT_CCSP_FRAMES = 300
HOST = "127.0.0.1"
RPC_PORT = 2000
TM_PORT = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--town-frames",
        type=int,
        default=DEFAULT_TOWN_FRAMES,
        help="每张标准 Town 地图需要保存的合格帧数。",
    )
    parser.add_argument(
        "--hutb-frames",
        type=int,
        default=DEFAULT_HUTB_FRAMES,
        help="HutbCarlaCity 需要保存的合格帧数。",
    )
    parser.add_argument(
        "--ccsp-frames",
        type=int,
        default=DEFAULT_CCSP_FRAMES,
        help="中电软件园专用模拟器需要保存的合格帧数。",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_OUTPUT_NAMES,
        help="Collect only the named output maps.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show the simulator window instead of using RenderOffScreen.",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Collect a map again even when it already contains enough RGB frames.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate_paths() -> None:
    required = [
        COLLECTOR,
        CONFIG,
        STATIC_PEDESTRIAN_HELPER,
        CCSP_QUALITY_COLLECTOR,
        OPENHUTB_PYTHON,
        GENERIC_SIMULATOR,
        CCSP_PYTHON,
        CCSP_CARLA_EGG,
        CCSP_SIMULATOR,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def stop_all_carla_processes() -> None:
    # Simulator launchers and shipping binaries must both exit before port 2000
    # can be reused by the other package.
    for image_name in ("CarlaUE4.exe", "CarlaUE4-Win64-Shipping.exe"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", image_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(4.0)


def start_simulator(
    executable: Path,
    visible: bool,
    full_assets: bool = False,
) -> subprocess.Popen:
    args = [str(executable), "-quality-level=Epic", f"-carla-rpc-port={RPC_PORT}"]
    if full_assets:
        args.extend([
            "-NoTextureStreaming",
            "-USEALLAVAILABLECORES",
            (
                "-ExecCmds=r.ForceLOD 0,foliage.ForceLOD 0,"
                "r.Streaming.FullyLoadUsedTextures 1"
            ),
        ])
        # CCSP photogrammetry/HLOD streaming is driven by the visible spectator.
        # Do not add RenderOffScreen for this dedicated full-asset profile.
        visible = True
    if not visible:
        args.append("-RenderOffScreen")

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if not visible:
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

    print(f"[SIM] Starting: {executable}", flush=True)
    return subprocess.Popen(
        args,
        cwd=str(executable.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def ccsp_environment() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CCSP_CARLA_EGG)
    prefix_paths = [
        CCSP_PYTHON_ROOT,
        CCSP_PYTHON_ROOT / "Library" / "mingw-w64" / "bin",
        CCSP_PYTHON_ROOT / "Library" / "usr" / "bin",
        CCSP_PYTHON_ROOT / "Library" / "bin",
        CCSP_PYTHON_ROOT / "Scripts",
    ]
    env["PATH"] = os.pathsep.join(str(path) for path in prefix_paths) + os.pathsep + env.get("PATH", "")
    return env


def start_static_pedestrian_bridge(
    python_executable: Path,
    count: int,
    vehicle_count: int = 0,
    env: Optional[Dict[str, str]] = None,
    map_name: Optional[str] = None,
    load_map: bool = False,
    seed: int = 2718,
    pedestrian_motion: str = "static",
    shuttle_distance: float = 12.0,
    lateral_offset: float = 3.5,
    min_road_clearance: float = 1.25,
) -> subprocess.Popen:
    command = [
        str(python_executable),
        "-u",
        str(STATIC_PEDESTRIAN_HELPER),
        "--host",
        HOST,
        "--port",
        str(RPC_PORT),
        "--timeout",
        "300",
        "--count",
        str(count),
        "--vehicle-count",
        str(vehicle_count),
        "--seed",
        str(seed),
        "--pedestrian-motion",
        pedestrian_motion,
        "--shuttle-distance",
        str(shuttle_distance),
        "--lateral-offset",
        str(lateral_offset),
        "--min-road-clearance",
        str(min_road_clearance),
    ]
    if map_name:
        command.extend(["--map", map_name])
    if load_map:
        command.append("--load-map")
    process = subprocess.Popen(
        command,
        cwd=str(STATIC_PEDESTRIAN_HELPER.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if line:
            print("[PEDESTRIAN-BRIDGE] " + line, end="", flush=True)
            if line.startswith("READY "):
                payload = json.loads(line[len("READY "):])
                if int(payload.get("spawned_pedestrians", 0)) <= 0:
                    process.terminate()
                    raise RuntimeError("Static pedestrian bridge spawned no actors")
                return process
        elif process.poll() is not None:
            raise RuntimeError(
                "Static pedestrian bridge exited before reporting readiness"
            )


def stop_static_pedestrian_bridge(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def wait_for_server(
    python_executable: Path,
    env: Optional[Dict[str, str]],
    expected_server_prefix: str,
    timeout_seconds: float = 360.0,
) -> int:
    probe = (
        "import carla; "
        f"c=carla.Client('{HOST}',{RPC_PORT}); c.set_timeout(4); "
        "w=c.get_world(); "
        "print(c.get_server_version()); print(w.get_map().name)"
    )
    deadline = time.time() + timeout_seconds
    last_message = ""
    while time.time() < deadline:
        result = subprocess.run(
            [str(python_executable), "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            check=False,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if expected_server_prefix and expected_server_prefix not in output:
                raise RuntimeError(
                    "Connected to the wrong simulator package. Probe output:\n" + output
                )
            print(f"[SIM] Ready:\n{output}", flush=True)
            return
        last_message = (result.stderr or result.stdout).strip()
        time.sleep(5.0)
    raise TimeoutError(f"Simulator did not become ready. Last probe: {last_message}")


def rgb_frame_count(map_output: Path) -> int:
    paired = map_output / "paired_weather"
    if not paired.exists():
        return 0
    return len(list(paired.glob("seq_*/rgb/**/*.png")))


def run_collector(
    python_executable: Path,
    map_output: Path,
    frames: int,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    map_name: Optional[str] = None,
    ccsp_without_navmesh: bool = False,
    overwrite_sequences: bool = True,
    seed: Optional[int] = None,
    allow_no_pedestrians: bool = False,
    min_road_visible_ratio: Optional[float] = None,
    extra_args: Optional[List[str]] = None,
) -> None:
    command: List[str] = [
        str(python_executable),
        "-u",
        str(COLLECTOR),
        "--config",
        str(CONFIG),
        "--host",
        HOST,
        "--port",
        str(RPC_PORT),
        "--tm-port",
        str(TM_PORT),
        "--timeout",
        "300",
        "--out",
        str(map_output),
        "--cooperative-output-root",
        str(GET_IMG_ROOT / "AirGroundCoopSuite" / "release" / str(map_name)),
        "--sequences",
        "1",
        "--frames",
        str(frames),
    ]
    command.append(
        "--overwrite-sequences"
        if overwrite_sequences
        else "--preserve-existing-sequences"
    )
    if map_name:
        command.extend(["--map", map_name])
    if ccsp_without_navmesh or allow_no_pedestrians:
        # Keep the collector unchanged while allowing vehicle-only frames on
        # maps that either have no navmesh or are intentionally sparse/rural.
        command.extend(["--min-pedestrians-per-frame", "0"])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if min_road_visible_ratio is not None:
        command.extend(
            ["--min-road-visible-ratio", str(min_road_visible_ratio)]
        )
    if extra_args:
        command.extend(extra_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[COLLECT] Output: {map_output}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(COLLECTOR.parent),
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
    return return_code


def write_status(status: Dict[str, object]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "collection_status.json"
    path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def selected_names(only: Optional[Iterable[str]]) -> List[str]:
    if only is None:
        return list(ALL_OUTPUT_NAMES)
    selected = set(only)
    return [name for name in ALL_OUTPUT_NAMES if name in selected]


def target_frames_for_map(
    map_name: str,
    town_frames: int,
    hutb_frames: int,
    ccsp_frames: int,
) -> int:
    if map_name == HUTB_MAP_NAME:
        return hutb_frames
    if map_name == CCSP_OUTPUT_NAME:
        return ccsp_frames
    return town_frames


def main() -> int:
    args = parse_args()
    if min(args.town_frames, args.hutb_frames, args.ccsp_frames) <= 0:
        raise ValueError("all frame targets must be greater than zero")
    validate_paths()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs_root = OUTPUT_ROOT / "_logs"
    names = selected_names(args.only)
    status: Dict[str, object] = {
        "started_at": now_iso(),
        "output_root": str(OUTPUT_ROOT),
        "frame_targets": {
            "town": args.town_frames,
            "hutb": args.hutb_frames,
            "ccsp": args.ccsp_frames,
        },
        "collector": str(COLLECTOR),
        "config": str(CONFIG),
        "map_selection_note": (
            "Town12 was replaced by Town02_Opt because this v2.10 package "
            "crashes when Town12 synchronized sensor streams start and "
            "Town12 exposes no pedestrian navigation locations."
        ),
        "maps": {},
    }
    write_status(status)

    generic_names = [name for name in names if name in GENERIC_MAPS]
    if generic_names:
        for map_name in generic_names:
            target_frames = target_frames_for_map(
                map_name,
                args.town_frames,
                args.hutb_frames,
                args.ccsp_frames,
            )
            map_output = OUTPUT_ROOT / map_name
            before = rgb_frame_count(map_output)
            if before >= target_frames and not args.rerun_complete:
                print(f"[SKIP] {map_name}: already has {before} RGB frames", flush=True)
                status["maps"][map_name] = {
                    "state": "skipped_complete",
                    "rgb_frames": before,
                    "target_frames": target_frames,
                }
                write_status(status)
                continue
            # Restart the generic package for each map. This avoids retaining
            # map tiles and GPU resources from the previous world.
            stop_all_carla_processes()
            start_simulator(GENERIC_SIMULATOR, args.visible)
            wait_for_server(
                OPENHUTB_PYTHON,
                env=None,
                expected_server_prefix="23461d8a4",
            )
            status["maps"][map_name] = {"state": "running", "started_at": now_iso()}
            write_status(status)
            count = before
            batch_index = len(list((map_output / "paired_weather").glob("seq_*")))
            max_batch_attempts = 100 if map_name in SHORT_BATCH_MAPS else 6
            consecutive_no_progress = 0
            while count < target_frames and batch_index < max_batch_attempts:
                batch_index += 1
                missing = target_frames - count
                request_frames = min(
                    missing,
                    SHORT_BATCH_MAPS.get(map_name, missing),
                )
                if count > before or map_name in SHORT_BATCH_MAPS:
                    stop_all_carla_processes()
                    start_simulator(GENERIC_SIMULATOR, args.visible)
                    wait_for_server(
                        OPENHUTB_PYTHON,
                        env=None,
                        expected_server_prefix="23461d8a4",
                    )
                print(
                    f"[BATCH] {map_name}: request={request_frames}, "
                    f"current={count}/{target_frames}",
                    flush=True,
                )
                pedestrian_bridge: Optional[subprocess.Popen] = None
                try:
                    if map_name == HUTB_MAP_NAME:
                        pedestrian_bridge = start_static_pedestrian_bridge(
                            OPENHUTB_PYTHON,
                            count=140,
                            map_name=map_name,
                            load_map=True,
                            seed=7100 + batch_index,
                        )
                    return_code = run_collector(
                        OPENHUTB_PYTHON,
                        map_output,
                        request_frames,
                        logs_root / f"{map_name}_batch_{batch_index:02d}.log",
                        # The bridge already loaded HutbCarlaCity. Passing --map
                        # again would rebuild the world and erase its pedestrians.
                        map_name=(None if map_name == HUTB_MAP_NAME else map_name),
                        overwrite_sequences=(count == 0),
                        seed=7 + batch_index,
                        min_road_visible_ratio=(
                            0.0
                            if map_name == "HutbCarlaCity"
                            else (0.24 if map_name in SHORT_BATCH_MAPS else None)
                        ),
                        extra_args=(
                            [
                                "--walkers", "80",
                                "--pedestrian-centered-camera-probability", "0.8",
                                "--vehicle-centered-camera-probability", "0.2",
                                "--max-camera-pose-retries", "100",
                                "--height-min", "26",
                                "--height-max", "29",
                                "--radius-min", "20",
                                "--radius-max", "36",
                                "--pitch-min", "-45",
                                "--pitch-max", "-40",
                            ]
                            if map_name in SHORT_BATCH_MAPS
                            else None
                        ),
                    )
                finally:
                    stop_static_pedestrian_bridge(pedestrian_bridge)
                new_count = rgb_frame_count(map_output)
                if return_code != 0:
                    if new_count <= count:
                        consecutive_no_progress += 1
                        if (
                            map_name not in SHORT_BATCH_MAPS
                            or consecutive_no_progress >= 8
                        ):
                            raise RuntimeError(
                                f"Collector failed with exit code {return_code} and "
                                f"made no progress. See "
                                f"{logs_root / f'{map_name}_batch_{batch_index:02d}.log'}"
                            )
                        print(
                            f"[WARN] {map_name}: no valid frames in this batch; "
                            f"retrying with a new seed "
                            f"({consecutive_no_progress}/8).",
                            flush=True,
                        )
                        continue
                    print(
                        f"[WARN] {map_name}: collector exited with code "
                        f"{return_code}, but saved {new_count - count} valid frames; "
                        "continuing with an automatic top-up batch.",
                        flush=True,
                    )
                consecutive_no_progress = 0
                if new_count <= count:
                    print(f"[WARN] {map_name}: batch made no progress", flush=True)
                count = new_count
            status["maps"][map_name] = {
                "state": "complete" if count == target_frames else "count_mismatch",
                "rgb_frames": count,
                "target_frames": target_frames,
                "finished_at": now_iso(),
            }
            write_status(status)
            if count != target_frames:
                raise RuntimeError(
                    f"{map_name}: expected {target_frames} RGB frames, found {count}"
                )

    if CCSP_OUTPUT_NAME in names:
        target_frames = args.ccsp_frames
        map_output = OUTPUT_ROOT / CCSP_OUTPUT_NAME
        before = rgb_frame_count(map_output)
        if before >= target_frames and not args.rerun_complete:
            print(f"[SKIP] {CCSP_OUTPUT_NAME}: already has {before} RGB frames", flush=True)
            status["maps"][CCSP_OUTPUT_NAME] = {
                "state": "skipped_complete",
                "rgb_frames": before,
                "target_frames": target_frames,
            }
        else:
            stop_all_carla_processes()
            start_simulator(
                CCSP_SIMULATOR,
                visible=True,
                full_assets=True,
            )
            ccsp_env = ccsp_environment()
            wait_for_server(
                CCSP_PYTHON,
                env=ccsp_env,
                # The dedicated CCSP package renders roadbuild and streams the
                # external D:\model tiles. Its current packaged build id is:
                expected_server_prefix="7b2ff0449",
            )
            status["maps"][CCSP_OUTPUT_NAME] = {
                "state": "running",
                "started_at": now_iso(),
                "target_frames": target_frames,
            }
            write_status(status)

            command = [
                str(CCSP_PYTHON),
                "-u",
                str(CCSP_QUALITY_COLLECTOR),
                "--frames",
                str(target_frames),
                "--out",
                str(map_output),
            ]
            if args.rerun_complete:
                command.append("--recollect")
            log_path = logs_root / f"{CCSP_OUTPUT_NAME}_quality_collection.log"
            print(
                "[CCSP] Using runtime-checked multi-zone collection with six balanced "
                "weather presets, half-density fog, per-pose RGB/LOD streaming "
                "stability, and per-frame visual artifact QA.",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(CCSP_QUALITY_COLLECTOR.parent),
                    env=ccsp_env,
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
            count = rgb_frame_count(map_output)
            if return_code != 0:
                raise RuntimeError(
                    f"CCSP quality collector failed with exit code {return_code}. "
                    f"See {log_path}"
                )

            status["maps"][CCSP_OUTPUT_NAME] = {
                "state": "complete" if count == target_frames else "count_mismatch",
                "rgb_frames": count,
                "target_frames": target_frames,
                "finished_at": now_iso(),
            }
            if count != target_frames:
                raise RuntimeError(
                    f"{CCSP_OUTPUT_NAME}: expected {target_frames} RGB frames, "
                    f"found {count}"
                )
        write_status(status)

    status["finished_at"] = now_iso()
    states = [entry.get("state") for entry in status["maps"].values()]
    status["state"] = "complete" if all(state in {"complete", "skipped_complete"} for state in states) else "incomplete"
    write_status(status)
    print(f"[DONE] All selected maps are under: {OUTPUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[STOP] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
