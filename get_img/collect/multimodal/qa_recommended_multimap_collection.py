#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validate the recommended multi-map multimodal dataset collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


DEFAULT_ROOT = Path(
    r"E:\pythonProject\air_groud\get_img"
    r"\dataset_uav_multimap_town600_hutb300_ccsp300"
)
TOWN_MAPS = [
    "Town03_Opt",
    "Town05_Opt",
    "Town10HD",
    "Town02_Opt",
    "Town04_Opt",
    "Town07_Opt",
]
HUTB_MAP_NAME = "HutbCarlaCity"
CCSP_MAP_NAME = "CCSP_Zhongdian_Software_Park"
COMPLETE_MAPS = TOWN_MAPS + [HUTB_MAP_NAME, CCSP_MAP_NAME]
EXPECTED_WEATHERS = {
    "ClearNoon",
    "ClearNight",
    "ClearSunset",
    "FoggyNoon",
    "SnowNoon",
    "DustStorm",
}
REQUIRED_IMAGE_KEYS = [
    "rgb",
    "depth_npy_meters",
    "depth_vis_16bit",
    "depth_color",
    "lidar_points",
    "lidar_projected_npy_meters",
    "lidar_projected_vis_16bit",
    "lidar_projected_color",
    "surface_normal_npy",
    "surface_normal",
    "segmentation",
    "segmentation_color",
    "yolo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--town-frames", type=int, default=600)
    parser.add_argument("--hutb-frames", type=int, default=300)
    parser.add_argument("--ccsp-frames", type=int, default=300)
    return parser.parse_args()


def expected_frames_for_map(map_name: str, args: argparse.Namespace) -> int:
    if map_name == HUTB_MAP_NAME:
        return args.hutb_frames
    if map_name == CCSP_MAP_NAME:
        return args.ccsp_frames
    return args.town_frames


def relative_exists(dataset_root: Path, value: str) -> bool:
    return (dataset_root / Path(value)).is_file()


def image_shape(path: Path) -> List[int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return list(image.shape)


def validate_sample_arrays(
    dataset_root: Path,
    annotation: Dict[str, Any],
    errors: List[str],
) -> Dict[str, Any]:
    image = annotation["image"]
    sample: Dict[str, Any] = {"frame_id": annotation.get("frame_id")}
    try:
        rgb_shape = image_shape(dataset_root / image["rgb"])
        depth_color_shape = image_shape(dataset_root / image["depth_color"])
        depth_16_shape = image_shape(dataset_root / image["depth_vis_16bit"])
        normal_png_shape = image_shape(dataset_root / image["surface_normal"])
        segmentation_shape = image_shape(dataset_root / image["segmentation"])
        segmentation_color_shape = image_shape(
            dataset_root / image["segmentation_color"]
        )
        sample["png_shapes"] = {
            "rgb": rgb_shape,
            "depth_color": depth_color_shape,
            "depth_16bit": depth_16_shape,
            "surface_normal": normal_png_shape,
            "segmentation": segmentation_shape,
            "segmentation_color": segmentation_color_shape,
        }
        if rgb_shape[:2] != [1080, 1920]:
            errors.append(f"RGB shape is {rgb_shape}, expected 1080x1920")
        for name, shape in sample["png_shapes"].items():
            if shape[:2] != [1080, 1920]:
                errors.append(f"{name} shape is {shape}, expected 1080x1920")

        depth = np.load(dataset_root / image["depth_npy_meters"], mmap_mode="r")
        normal = np.load(dataset_root / image["surface_normal_npy"], mmap_mode="r")
        lidar = np.load(dataset_root / image["lidar_points"], mmap_mode="r")
        projected_lidar = np.load(
            dataset_root / image["lidar_projected_npy_meters"], mmap_mode="r"
        )
        sample["array_shapes"] = {
            "depth": list(depth.shape),
            "surface_normal": list(normal.shape),
            "lidar": list(lidar.shape),
            "lidar_projected": list(projected_lidar.shape),
        }
        if list(depth.shape) != [1080, 1920]:
            errors.append(f"Depth array shape is {list(depth.shape)}")
        if list(normal.shape) != [1080, 1920, 3]:
            errors.append(f"Surface-normal array shape is {list(normal.shape)}")
        if lidar.ndim != 2 or lidar.shape[1] < 3 or lidar.shape[0] == 0:
            errors.append(f"Invalid LiDAR point shape: {list(lidar.shape)}")
        if list(projected_lidar.shape) != [1080, 1920]:
            errors.append(
                f"Projected LiDAR array shape is {list(projected_lidar.shape)}"
            )

        finite_normal = np.asarray(normal[::16, ::16], dtype=np.float32)
        normal_magnitude = np.linalg.norm(finite_normal, axis=2)
        sample["normal_magnitude_mean"] = float(np.nanmean(normal_magnitude))
        finite_depth = np.asarray(depth[::16, ::16], dtype=np.float32)
        sample["depth_finite_ratio"] = float(np.mean(np.isfinite(finite_depth)))
        sample["depth_positive_ratio"] = float(np.mean(finite_depth > 0.0))

        segmentation = cv2.imread(
            str(dataset_root / image["segmentation"]),
            cv2.IMREAD_UNCHANGED,
        )
        unique_values = sorted(int(value) for value in np.unique(segmentation))
        sample["segmentation_values"] = unique_values
        if not set(unique_values).issubset({0, 1, 2, 255}):
            errors.append(f"Unexpected segmentation values: {unique_values}")
    except Exception as exc:
        errors.append(f"Sample-array validation failed: {exc}")
    return sample


def validate_map(
    dataset_root: Path,
    map_name: str,
    expected_frames: int,
) -> Dict[str, Any]:
    map_root = dataset_root / map_name
    errors: List[str] = []
    warnings: List[str] = []
    annotation_paths = sorted(
        (map_root / "paired_weather").glob("seq_*/annotations/*.json")
    )
    rgb_paths = sorted(
        path
        for path in (map_root / "paired_weather").glob("seq_*/rgb/*/*.png")
        if path.is_file()
    )
    if len(annotation_paths) != expected_frames:
        errors.append(
            f"annotation_count={len(annotation_paths)}, expected={expected_frames}"
        )
    if len(rgb_paths) != expected_frames:
        errors.append(f"rgb_count={len(rgb_paths)}, expected={expected_frames}")

    weather_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    frames_with_both_classes = 0
    missing_files: List[str] = []
    invalid_boxes = 0
    ignored_target_instances = 0
    low_visibility_annotations = 0
    low_visibility_not_in_trainable_mask = 0
    low_visibility_frames = 0
    loaded_annotations: List[Dict[str, Any]] = []

    for annotation_path in annotation_paths:
        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)
        loaded_annotations.append(annotation)
        image = annotation.get("image", {})
        for key in REQUIRED_IMAGE_KEYS:
            value = image.get(key)
            if not value or not relative_exists(map_root, value):
                missing_files.append(f"{annotation_path.name}:{key}:{value}")

        weather = str(annotation.get("canonical_weather", ""))
        weather_counts[weather] += 1
        frame_classes = Counter(
            str(item.get("class_name", ""))
            for item in annotation.get("annotations", [])
        )
        class_counts.update(frame_classes)
        if frame_classes["vehicle"] > 0 and frame_classes["pedestrian"] > 0:
            frames_with_both_classes += 1

        trainable_actor_ids = {
            int(item["carla_actor_id"])
            for item in annotation.get("target_semantic", {}).get("instances", [])
            if bool(item.get("trainable")) and item.get("carla_actor_id") is not None
        }
        frame_has_low_visibility = False

        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        for item in annotation.get("annotations", []):
            visible_ratio = float(item.get("visible_ratio_projected_bbox", 1.0))
            if visible_ratio < 0.5:
                low_visibility_annotations += 1
                frame_has_low_visibility = True
                if int(item.get("carla_actor_id", -1)) not in trainable_actor_ids:
                    low_visibility_not_in_trainable_mask += 1
            x, y, box_width, box_height = [
                int(value) for value in item.get("bbox_xywh", [0, 0, 0, 0])
            ]
            if (
                box_width <= 0
                or box_height <= 0
                or x < 0
                or y < 0
                or x + box_width > width
                or y + box_height > height
            ):
                invalid_boxes += 1
        if frame_has_low_visibility:
            low_visibility_frames += 1
        ignored_target_instances += sum(
            not bool(item.get("trainable"))
            for item in annotation.get("target_semantic", {}).get("instances", [])
        )

    if missing_files:
        errors.append(f"missing_modality_files={len(missing_files)}")
    if frames_with_both_classes != len(annotation_paths):
        errors.append(
            f"frames_with_both_classes={frames_with_both_classes}/"
            f"{len(annotation_paths)}"
        )
    if invalid_boxes:
        errors.append(f"invalid_boxes={invalid_boxes}")
    missing_weathers = sorted(EXPECTED_WEATHERS - set(weather_counts))
    unexpected_weathers = sorted(set(weather_counts) - EXPECTED_WEATHERS)
    if missing_weathers:
        errors.append(f"missing_weathers={missing_weathers}")
    if unexpected_weathers:
        errors.append(f"unexpected_weathers={unexpected_weathers}")
    if low_visibility_annotations:
        warnings.append(
            "Detection annotations below visible_ratio 0.5: "
            f"{low_visibility_annotations} across {low_visibility_frames} frames; "
            f"{low_visibility_not_in_trainable_mask} are absent from the "
            "trainable segmentation mask."
        )

    samples: List[Dict[str, Any]] = []
    if loaded_annotations:
        indexes = sorted(
            {0, len(loaded_annotations) // 2, len(loaded_annotations) - 1}
        )
        for index in indexes:
            samples.append(
                validate_sample_arrays(
                    map_root,
                    loaded_annotations[index],
                    errors,
                )
            )

    return {
        "state": (
            "error"
            if errors
            else ("ok_with_warnings" if warnings else "ok")
        ),
        "rgb_frames": len(rgb_paths),
        "annotation_frames": len(annotation_paths),
        "frames_with_vehicle_and_pedestrian": frames_with_both_classes,
        "class_instances": dict(sorted(class_counts.items())),
        "weather_counts": dict(sorted(weather_counts.items())),
        "missing_modality_file_count": len(missing_files),
        "missing_modality_file_examples": missing_files[:10],
        "invalid_boxes": invalid_boxes,
        "ignored_target_instances": ignored_target_instances,
        "low_visibility_annotations_below_0_5": low_visibility_annotations,
        "low_visibility_frames": low_visibility_frames,
        "low_visibility_not_in_trainable_mask": (
            low_visibility_not_in_trainable_mask
        ),
        "samples": samples,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    status_path = root / "collection_status.json"
    collection_status = {}
    if status_path.is_file():
        collection_status = json.loads(status_path.read_text(encoding="utf-8"))

    maps: Dict[str, Any] = {}
    for map_name in COMPLETE_MAPS:
        expected_frames = expected_frames_for_map(map_name, args)
        print(f"[QA] {map_name}: expected={expected_frames}", flush=True)
        maps[map_name] = validate_map(root, map_name, expected_frames)

    errors = [
        f"{map_name}: {error}"
        for map_name, result in maps.items()
        for error in result["errors"]
    ]
    warnings = [
        f"{map_name}: {warning}"
        for map_name, result in maps.items()
        for warning in result["warnings"]
    ]
    summary = {
        "dataset_root": str(root),
        "expected_frame_targets": {
            "town": args.town_frames,
            "hutb": args.hutb_frames,
            "ccsp": args.ccsp_frames,
        },
        "complete_map_count": sum(not result["errors"] for result in maps.values()),
        "complete_rgb_frames": sum(result["rgb_frames"] for result in maps.values()),
        "qa_state": (
            "error"
            if errors
            else ("ok_with_warnings" if warnings else "ok")
        ),
        "collection_status": collection_status,
        "maps": maps,
        "warnings": warnings,
        "errors": errors,
    }
    output_path = root / "collection_qa_summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[QA] state={summary['qa_state']}")
    print(f"[QA] complete_maps={summary['complete_map_count']}/{len(COMPLETE_MAPS)}")
    print(f"[QA] complete_rgb_frames={summary['complete_rgb_frames']}")
    print(f"[QA] report={output_path}")
    if errors:
        for error in errors[:30]:
            print(f"[QA][ERROR] {error}")
        return 1
    for warning in warnings[:30]:
        print(f"[QA][WARN] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
