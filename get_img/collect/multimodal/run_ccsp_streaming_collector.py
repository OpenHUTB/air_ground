#!/usr/bin/env python3
"""Run the multimodal collector with CCSP streaming and fog safeguards."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


COLLECTOR_PATH = (
    Path(__file__).resolve().parent / "collect_rpg_small_targets_carla_v2.py"
)
RGB_STABLE_REQUIRED_COMPARISONS = 5
RGB_STABLE_MAD_THRESHOLD = 0.004


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "ccsp_streaming_base_collector",
        str(COLLECTOR_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load collector module: {COLLECTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    collector = load_collector()

    fog_preset = collector.CUSTOM_WEATHER_PRESETS["FoggyNoon"]
    fog_preset["overrides"]["fog_density"] = 36.0
    fog_preset["rendering"] = (
        "ccsp_half_density_carla_weather_plus_half_depth_rgb_fog"
    )

    full_depth_fog = collector.apply_depth_fog_rgb_effect

    def apply_half_depth_fog(bgr, depth_m):
        fogged = full_depth_fog(bgr, depth_m).astype(np.float32)
        source = bgr.astype(np.float32)
        return np.clip(source * 0.5 + fogged * 0.5, 0.0, 255.0).astype(
            np.uint8
        )

    collector.apply_depth_fog_rgb_effect = apply_half_depth_fog

    # Depth coverage becomes valid as soon as a coarse HLOD mesh appears. It
    # therefore cannot tell whether CCSP's detailed geometry and textures have
    # finished streaming. Track a small blurred RGB proxy at each fixed camera
    # pose and keep the base collector in its retry loop until the rendered
    # scene has stopped changing for several consecutive captures.
    rgb_state = {
        "previous": None,
        "stable_comparisons": 0,
        "mad": 1.0,
    }
    original_set_transform = collector.set_all_sensor_transform
    original_capture = collector.capture_synchronized_sensor_frame
    original_readiness = collector.streaming_scene_readiness
    original_settled = collector.streaming_scene_has_settled

    def reset_rgb_streaming_state(sensors, transform):
        rgb_state["previous"] = None
        rgb_state["stable_comparisons"] = 0
        rgb_state["mad"] = 1.0
        return original_set_transform(sensors, transform)

    def capture_with_rgb_stability(*args, **kwargs):
        bundle = original_capture(*args, **kwargs)
        rgb_image = bundle[1]
        bgra = collector.carla_image_to_bgra(rgb_image)
        gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
        proxy = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        proxy = cv2.GaussianBlur(proxy, (7, 7), 0).astype(np.float32) / 255.0
        previous = rgb_state["previous"]
        if previous is None:
            rgb_state["stable_comparisons"] = 0
            rgb_state["mad"] = 1.0
        else:
            mad = float(np.mean(np.abs(proxy - previous)))
            rgb_state["mad"] = mad
            if mad <= RGB_STABLE_MAD_THRESHOLD:
                rgb_state["stable_comparisons"] += 1
            else:
                rgb_state["stable_comparisons"] = 0
        rgb_state["previous"] = proxy
        return bundle

    def readiness_with_rgb_stability(*args, **kwargs):
        geometry_ready, stats = original_readiness(*args, **kwargs)
        rgb_ready = bool(
            rgb_state["stable_comparisons"]
            >= RGB_STABLE_REQUIRED_COMPARISONS
        )
        stats["streaming_rgb_mad"] = float(rgb_state["mad"])
        stats["streaming_rgb_stable_comparisons"] = int(
            rgb_state["stable_comparisons"]
        )
        stats["streaming_rgb_required_comparisons"] = int(
            RGB_STABLE_REQUIRED_COMPARISONS
        )
        stats["streaming_rgb_scene_ready"] = rgb_ready
        stats["streaming_scene_ready"] = bool(geometry_ready and rgb_ready)
        stats["streaming_readiness_mode"] = (
            "geometry_and_rgb_stable"
            if stats["streaming_scene_ready"]
            else "waiting_for_geometry_or_rgb_streaming"
        )
        return bool(stats["streaming_scene_ready"]), stats

    def settled_with_rgb_stability(history, *args, **kwargs):
        settled, stats = original_settled(history, *args, **kwargs)
        rgb_ready = bool(
            history
            and history[-1].get("streaming_rgb_scene_ready", False)
        )
        stats["streaming_rgb_scene_ready"] = rgb_ready
        return bool(settled and rgb_ready), stats

    collector.set_all_sensor_transform = reset_rgb_streaming_state
    collector.capture_synchronized_sensor_frame = capture_with_rgb_stability
    collector.streaming_scene_readiness = readiness_with_rgb_stability
    collector.streaming_scene_has_settled = settled_with_rgb_stability
    result = collector.main()
    # The base collector historically returns None after a successful run.
    # Treat that as process exit code 0 instead of raising int(None).
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
