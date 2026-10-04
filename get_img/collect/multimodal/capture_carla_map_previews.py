#!/usr/bin/env python
"""Capture one aerial RGB preview for every non-Opt CARLA/OpenHUTB map."""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path
from typing import Any

import carla
import cv2
import numpy as np


DEFAULT_OUTPUT = Path(__file__).with_name("carla_map_previews_no_opt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--altitude", type=float, default=120.0)
    parser.add_argument("--warmup-frames", type=int, default=12)
    parser.add_argument("--streaming-wait", type=float, default=20.0)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--no-restore", action="store_true")
    parser.add_argument(
        "--assume-current-map",
        action="store_true",
        help="Capture the current visual-only level when it has no OpenDRIVE map name.",
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Keep the simulator asynchronous; useful for streamed large maps.",
    )
    return parser.parse_args()


def short_map_name(name: str) -> str:
    return name.replace("\\", "/").rstrip("/").split("/")[-1]


def safe_world_map_name(world: carla.World, fallback: str = "") -> str:
    """Return the map name even for visual-only levels without OpenDRIVE data."""
    try:
        return short_map_name(world.get_map().name)
    except RuntimeError:
        return fallback


def central_road_transform(
    world: carla.World,
    altitude: float,
    prefer_spectator_nearby: bool = False,
) -> tuple[carla.Transform, str, int]:
    try:
        carla_map = world.get_map()
        candidates = list(carla_map.get_spawn_points())
        source = "spawn_point"
        if not candidates:
            candidates = [waypoint.transform for waypoint in carla_map.generate_waypoints(20.0)]
            source = "road_waypoint"
    except RuntimeError:
        candidates = []
        source = "visual_level_spectator"
    if not candidates:
        spectator_transform = world.get_spectator().get_transform()
        candidates = [spectator_transform]
        if source != "visual_level_spectator":
            source = "spectator_fallback"

    if prefer_spectator_nearby:
        reference = world.get_spectator().get_transform().location
        center_x = float(reference.x)
        center_y = float(reference.y)
        source = f"{source}_near_spectator"
    else:
        xs = np.asarray([item.location.x for item in candidates], dtype=np.float64)
        ys = np.asarray([item.location.y for item in candidates], dtype=np.float64)
        center_x = float(np.median(xs))
        center_y = float(np.median(ys))
    selected = min(
        candidates,
        key=lambda item: (item.location.x - center_x) ** 2 + (item.location.y - center_y) ** 2,
    )
    location = carla.Location(
        x=selected.location.x,
        y=selected.location.y,
        z=max(float(selected.location.z) + altitude, altitude),
    )
    rotation = carla.Rotation(
        pitch=-65.0,
        yaw=float(selected.rotation.yaw) + 25.0,
        roll=0.0,
    )
    return carla.Transform(location, rotation), source, len(candidates)


def image_to_bgr(image: carla.Image) -> np.ndarray:
    pixels = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = pixels.reshape((image.height, image.width, 4))
    return bgra[:, :, :3].copy()


def capture_map(
    client: carla.Client,
    map_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    current_world = client.get_world()
    current_map_name = safe_world_map_name(current_world)
    if current_map_name == map_name or (args.assume_current_map and not current_map_name):
        world = current_world
    else:
        previous_world_id = current_world.id
        try:
            world = client.load_world(map_name)
        except RuntimeError as exc:
            # Visual-only OpenHUTB levels can load successfully but have no
            # OpenDRIVE file, causing load_world() to report "failed to generate map".
            if "failed to generate map" not in str(exc).lower():
                raise
            time.sleep(2.0)
            world = client.get_world()
            if world.id == previous_world_id:
                raise
    actual_name = safe_world_map_name(world, map_name)
    original_settings = world.get_settings()
    settings_changed = False
    if not args.async_mode:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        settings_changed = True

    camera = None
    try:
        streamed_large_map = actual_name in {"Town11", "Town12", "Town13"}
        transform, pose_source, candidate_count = central_road_transform(
            world,
            args.altitude,
            prefer_spectator_nearby=streamed_large_map,
        )
        if streamed_large_map:
            world.get_spectator().set_transform(transform)
            time.sleep(max(0.0, args.streaming_wait))
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(args.width))
        blueprint.set_attribute("image_size_y", str(args.height))
        blueprint.set_attribute("fov", str(args.fov))
        blueprint.set_attribute("enable_postprocess_effects", "true")
        if pose_source == "visual_level_spectator":
            blueprint.set_attribute("exposure_compensation", "4.0")
            blueprint.set_attribute("gamma", "2.4")
        camera = world.spawn_actor(blueprint, transform)
        image_queue: queue.Queue[carla.Image] = queue.Queue()
        camera.listen(image_queue.put)

        latest = None
        if args.async_mode:
            for _ in range(max(1, args.warmup_frames)):
                latest = image_queue.get(timeout=30.0)
        else:
            for _ in range(max(1, args.warmup_frames)):
                frame = world.tick()
                while True:
                    candidate = image_queue.get(timeout=15.0)
                    if candidate.frame >= frame:
                        latest = candidate
                        break
        if latest is None:
            raise RuntimeError("camera returned no image")

        output_path = args.output / f"{actual_name}.png"
        if not cv2.imwrite(str(output_path), image_to_bgr(latest)):
            raise RuntimeError(f"failed to save {output_path}")
        return {
            "requested_map": map_name,
            "actual_map": actual_name,
            "status": "ok",
            "output": str(output_path),
            "road_pose_source": pose_source,
            "road_pose_candidates": candidate_count,
            "camera": {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z,
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll,
                "width": args.width,
                "height": args.height,
                "fov": args.fov,
                "exposure_compensation": 4.0 if pose_source == "visual_level_spectator" else 0.0,
            },
            "seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        if settings_changed:
            world.apply_settings(original_settings)


def make_contact_sheet(records: list[dict[str, Any]], output: Path) -> None:
    successful = [item for item in records if item.get("status") == "ok"]
    if not successful:
        return
    columns = 3
    tile_width = 480
    image_height = 270
    label_height = 38
    rows = (len(successful) + columns - 1) // columns
    sheet = np.full(
        (rows * (image_height + label_height), columns * tile_width, 3),
        24,
        dtype=np.uint8,
    )
    for index, record in enumerate(successful):
        image = cv2.imread(record["output"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (tile_width, image_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x1 = column * tile_width
        y1 = row * (image_height + label_height)
        sheet[y1 : y1 + image_height, x1 : x1 + tile_width] = image
        cv2.putText(
            sheet,
            str(record["actual_map"]),
            (x1 + 12, y1 + image_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to save contact sheet: {output}")


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    original_map = safe_world_map_name(client.get_world())
    if args.maps:
        maps = [short_map_name(name) for name in args.maps]
    else:
        maps = sorted(
            short_map_name(name)
            for name in client.get_available_maps()
            if not short_map_name(name).endswith("_Opt")
        )
    records: list[dict[str, Any]] = []
    print(f"[INFO] original_map={original_map}, non_opt_maps={len(maps)}")
    try:
        for index, map_name in enumerate(maps, start=1):
            print(f"[MAP {index:02d}/{len(maps):02d}] {map_name}", flush=True)
            try:
                record = capture_map(client, map_name, args)
            except Exception as exc:
                record = {
                    "requested_map": map_name,
                    "actual_map": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[ERROR] {map_name}: {record['error']}", flush=True)
            records.append(record)
    finally:
        current_map = safe_world_map_name(client.get_world())
        if not args.no_restore and original_map and current_map != original_map:
            print(f"[RESTORE] {current_map} -> {original_map}", flush=True)
            client.load_world(original_map)

    contact_sheet = args.output / "ALL_NON_OPT_MAPS_CONTACT_SHEET.png"
    make_contact_sheet(records, contact_sheet)
    report = {
        "original_map": original_map,
        "requested_count": len(maps),
        "success_count": sum(item["status"] == "ok" for item in records),
        "failure_count": sum(item["status"] != "ok" for item in records),
        "contact_sheet": str(contact_sheet),
        "maps": records,
    }
    report_path = args.output / "capture_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[DONE] success={report['success_count']}/{report['requested_count']} "
        f"contact_sheet={contact_sheet} report={report_path}"
    )


if __name__ == "__main__":
    main()
