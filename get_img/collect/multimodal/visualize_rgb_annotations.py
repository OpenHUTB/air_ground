#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""按天气随机抽样独立场景并可视化四模态目标框。"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("collection_config.json")
DEFAULT_DATASET_PATH = Path(
    "E:/pythonProject/air_groud/get_img/collect/"
    "dataset_uav_small_carla_1920_low33_ped50_occ50_random_weather"
)

BOX_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从每种随机天气中分别抽取独立场景，并为 "
            "RGB/Depth/Normal/Segmentation 绘制标注框。"
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset root. By default, use the out value in collection_config.json."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="paired_weather/seq_0000",
        help="相对数据集根目录的序列路径。"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help="从每种天气中随机抽取多少个独立场景；默认每种天气 3 张。"
    )
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="随机抽样种子；相同种子和数据集会得到相同结果，默认 7。"
    )
    parser.add_argument(
        "--weather",
        action="append",
        default=None,
        help="只输出指定天气；可重复传入。默认读取配置文件中的 weather_presets。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="默认从该配置的 weather_presets 读取要检查的天气。"
    )
    parser.add_argument(
        "--all-weather",
        action="store_true",
        help="忽略配置筛选，输出数据中已有的全部天气。"
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=0,
        help="Bounding-box thickness. Use 0 to scale automatically with image resolution."
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.0,
        help="Annotation font scale. Use 0 to scale automatically with image resolution."
    )
    parser.add_argument("--thumb-width", type=int, default=640)
    parser.add_argument("--thumb-height", type=int, default=360)
    parser.add_argument(
        "--include-lidar",
        action="store_true",
        help="额外输出稀疏 LiDAR 投影深度；默认只输出四种主要模态。"
    )
    parser.add_argument(
        "--no-contact-sheets",
        dest="contact_sheets",
        action="store_false",
        default=True
    )
    return parser.parse_args()


def resolve_path(dataset_root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def normalize_gray_to_bgr(array: np.ndarray, color_map: int) -> np.ndarray:
    values = array.astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros((*array.shape[:2], 3), dtype=np.uint8)
    low, high = np.percentile(values[finite], [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    normalized = np.zeros(values.shape[:2], dtype=np.uint8)
    normalized[finite] = np.clip(
        (values[finite] - low) / (high - low) * 255.0,
        0.0,
        255.0
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, color_map)


def load_display_image(path: Path, modality: str) -> Optional[np.ndarray]:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if modality == "surface_normal" and array.ndim == 3 and array.shape[2] == 3:
            rgb = np.clip((array.astype(np.float32) + 1.0) * 127.5, 0, 255).astype(np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return normalize_gray_to_bgr(array, cv2.COLORMAP_TURBO)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        color_map = cv2.COLORMAP_HSV if modality == "segmentation" else cv2.COLORMAP_TURBO
        return normalize_gray_to_bgr(image, color_map)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        return normalize_gray_to_bgr(image[:, :, 0], cv2.COLORMAP_TURBO)
    if modality == "depth_lidar":
        # LiDAR 投影天然稀疏，仅在 QA 预览中放大光点，原始数据保持不变。
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.dilate(image, kernel, iterations=1)
    return image.copy()


def draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    text_thickness: int
) -> None:
    x, baseline_y = origin
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness
    )
    padding = max(2, text_thickness * 2)
    x = max(0, min(x, max(0, image.shape[1] - text_w - padding * 2)))
    baseline_y = max(
        text_h + baseline + padding,
        min(baseline_y, image.shape[0] - baseline - padding)
    )
    top = max(0, baseline_y - text_h - baseline - padding)
    right = min(image.shape[1] - 1, x + text_w + padding * 2)
    bottom = min(image.shape[0] - 1, baseline_y + baseline + padding)
    cv2.rectangle(image, (x, top), (right, bottom), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (x + padding, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        text_thickness,
        cv2.LINE_AA
    )


def draw_annotations(
    image: np.ndarray,
    annotations: List[Dict],
    source_label: str,
    frame_label: str,
    box_thickness: int,
    font_scale: float
) -> np.ndarray:
    output = image.copy()
    resolution_scale = max(
        1.0,
        min(output.shape[1] / 1920.0, output.shape[0] / 1080.0)
    )
    resolved_box_thickness = (
        box_thickness
        if box_thickness > 0
        else max(2, int(round(2.0 * resolution_scale)))
    )
    resolved_font_scale = (
        font_scale
        if font_scale > 0.0
        else 0.52 * resolution_scale
    )
    text_thickness = max(1, int(round(resolution_scale)))
    margin = max(4, int(round(8.0 * resolution_scale)))
    header_y = max(18, int(round(24.0 * resolution_scale)))
    draw_label(
        output,
        f"{source_label} | frame={frame_label} | objects={len(annotations)}",
        (margin, header_y),
        (255, 255, 255),
        resolved_font_scale,
        text_thickness
    )

    for ann in annotations:
        class_id = int(ann.get("class_id", 0))
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        x, y, w, h = [int(v) for v in ann["bbox_xywh"]]
        x2 = min(output.shape[1] - 1, x + w - 1)
        y2 = min(output.shape[0] - 1, y + h - 1)
        cv2.rectangle(
            output,
            (x, y),
            (x2, y2),
            color,
            resolved_box_thickness
        )

        actor_id = ann.get("carla_actor_id", ann.get("carla_instance_id", ""))
        label = f"{ann.get('class_name', 'object')} actor={actor_id}"
        label_gap = max(5, int(round(5.0 * resolution_scale)))
        label_height = max(18, int(round(18.0 * resolution_scale)))
        text_y = (
            y - label_gap
            if y >= header_y
            else min(output.shape[0] - margin, y2 + label_height)
        )
        draw_label(
            output,
            label,
            (max(0, x), text_y),
            color,
            resolved_font_scale,
            text_thickness
        )

    return output


def make_contact_sheet(
    items: List[Tuple[str, np.ndarray]],
    columns: int = 4,
    thumb_size: Tuple[int, int] = (320, 180)
) -> np.ndarray:
    if not items:
        return np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
    tile_w, tile_h = thumb_size
    rows = (len(items) + columns - 1) // columns
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 28, dtype=np.uint8)

    for index, (label, image) in enumerate(items):
        row = index // columns
        col = index % columns
        x = col * tile_w
        y = row * tile_h
        resized = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        sheet[y:y + tile_h, x:x + tile_w] = resized
        cv2.rectangle(sheet, (x, y), (x + tile_w - 1, y + 24), (0, 0, 0), -1)
        cv2.putText(
            sheet,
            label,
            (x + 6, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # 23种天气排成4列时会多出一个格子，明确标记为排版空位。
    for index in range(len(items), rows * columns):
        row = index // columns
        col = index % columns
        x = col * tile_w
        y = row * tile_h
        cv2.line(sheet, (x, y), (x + tile_w - 1, y + tile_h - 1), (70, 70, 70), 2)
        cv2.line(sheet, (x + tile_w - 1, y), (x, y + tile_h - 1), (70, 70, 70), 2)
        cv2.putText(
            sheet,
            "layout empty",
            (x + 88, y + tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )
    return sheet


def modality_sources(
    image_info: Dict,
    include_lidar: bool = False
) -> Dict[str, Optional[str]]:
    sources = {
        "depth": image_info.get("depth_color") or image_info.get("depth_vis_16bit") or image_info.get("depth_npy_meters"),
        "surface_normal": image_info.get("surface_normal") or image_info.get("surface_normal_npy"),
        "segmentation": image_info.get("segmentation_color") or image_info.get("segmentation"),
    }
    if include_lidar:
        lidar_source = (
            image_info.get("lidar_projected_color")
            or image_info.get("lidar_projected_vis_16bit")
            or image_info.get("lidar_projected_npy_meters")
        )
        if lidar_source:
            sources["depth_lidar"] = lidar_source
    return sources


def load_configured_weather_names(config_path: str) -> Optional[List[str]]:
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    values = data.get("weather_presets")
    if not isinstance(values, list):
        return None
    return [str(value) for value in values]


def load_configured_dataset_root(config_path: str) -> Optional[Path]:
    path = Path(config_path).resolve()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("out")
    if not value:
        return None
    dataset_root = Path(str(value)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = path.parent / dataset_root
    return dataset_root.resolve()


def collect_review_candidates(
    frame_label: str,
    annotations: List[Dict],
    width: int,
    height: int
) -> List[Dict]:
    candidates = []
    for ann in annotations:
        x, y, w, h = [int(v) for v in ann["bbox_xywh"]]
        reasons = []
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            reasons.append("bbox_touches_image_boundary")
        visible_ratio = ann.get("visible_ratio_projected_bbox")
        if visible_ratio is not None and float(visible_ratio) < 0.50:
            reasons.append("low_visible_ratio_below_0.50")
        if min(w, h) <= 2:
            reasons.append("bbox_min_side_le_2px")
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            reasons.append("invalid_bbox_geometry")
        if reasons:
            candidates.append({
                "frame": frame_label,
                "annotation_id": ann.get("id"),
                "actor_id": ann.get("carla_actor_id"),
                "bbox_xywh": [x, y, w, h],
                "visible_ratio_projected_bbox": visible_ratio,
                "reasons": reasons
            })
    return candidates


def select_random_annotation_files(
    annotation_dir: Path,
    step: int,
    max_frames: int,
    seed: int,
    weather_filter: Optional[List[str]] = None
) -> Tuple[List[Path], int, Dict[str, List[str]], Dict[str, int]]:
    """Randomly select independent annotation frames from every weather."""
    candidates = sorted(annotation_dir.glob("*.json"))[::step]
    if not candidates:
        raise RuntimeError(f"No annotation JSON files found: {annotation_dir}")

    candidates_by_weather: Dict[str, List[Path]] = {}
    invalid_multi_weather: List[str] = []
    for annotation_path in candidates:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        weather_names = sorted(
            data.get("image", {}).get("rgb_by_weather", {})
        )
        if len(weather_names) != 1:
            invalid_multi_weather.append(annotation_path.stem)
            continue
        weather_name = weather_names[0]
        candidates_by_weather.setdefault(weather_name, []).append(
            annotation_path
        )

    if invalid_multi_weather:
        preview = ", ".join(invalid_multi_weather[:5])
        raise RuntimeError(
            "当前可视化脚本要求每帧只有一种天气；发现旧的配对天气标注："
            f"{preview}"
        )

    selected_weather_names = (
        weather_filter
        if weather_filter
        else sorted(candidates_by_weather)
    )
    selected: List[Path] = []
    selected_by_weather: Dict[str, List[str]] = {}
    candidate_counts = {
        name: len(paths) for name, paths in sorted(candidates_by_weather.items())
    }
    rng = random.Random(seed)
    for weather_name in selected_weather_names:
        weather_candidates = candidates_by_weather.get(weather_name, [])
        sample_count = min(max_frames, len(weather_candidates))
        weather_selected = rng.sample(weather_candidates, sample_count)
        weather_selected.sort()
        selected.extend(weather_selected)
        selected_by_weather[weather_name] = [
            path.stem for path in weather_selected
        ]

    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError("指定天气中没有可供抽样的单天气标注帧。")
    return selected, len(candidates), selected_by_weather, candidate_counts


def main() -> None:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.box_thickness < 0:
        raise ValueError("--box-thickness must be 0 or positive")
    if args.font_scale < 0.0:
        raise ValueError("--font-scale must be 0 or positive")
    if args.thumb_width <= 0 or args.thumb_height <= 0:
        raise ValueError("--thumb-width and --thumb-height must be positive")

    dataset_root = (
        Path(args.dataset).resolve()
        if args.dataset
        else load_configured_dataset_root(args.config) or DEFAULT_DATASET_PATH.resolve()
    )
    seq_dir = dataset_root / args.sequence
    configured_weather_names = None
    if not args.weather and not args.all_weather:
        configured_weather_names = load_configured_weather_names(args.config)
    weather_filter = args.weather or configured_weather_names
    (
        ann_files,
        candidate_frame_count,
        selected_by_weather,
        weather_candidate_counts,
    ) = select_random_annotation_files(
        seq_dir / "annotations",
        step=args.step,
        max_frames=args.max_frames,
        seed=args.seed,
        weather_filter=weather_filter
    )
    sample_count = len(ann_files)

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else seq_dir / (
            f"qa_overlay_random{args.max_frames}_per_weather_seed{args.seed}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written_counts: Dict[str, int] = {}
    missing: List[str] = []
    weather_sheet_items: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    frame_sheet_items: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    modality_sheet_items: List[Tuple[str, np.ndarray]] = []
    review_candidates: List[Dict] = []

    for ann_path in ann_files:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        image_info = data["image"]
        annotations = data.get("annotations", [])
        frame_label = ann_path.stem
        review_candidates.extend(
            collect_review_candidates(
                frame_label=frame_label,
                annotations=annotations,
                width=int(image_info["width"]),
                height=int(image_info["height"])
            )
        )
        rgb_by_weather = image_info.get("rgb_by_weather", {})
        weather_names = sorted(rgb_by_weather)
        if args.weather:
            requested = set(args.weather)
            weather_names = [name for name in weather_names if name in requested]
        elif configured_weather_names:
            available = set(weather_names)
            weather_names = [
                name for name in configured_weather_names
                if name in available
            ]

        frame_sheet_items[frame_label] = []
        canonical_overlay = None
        canonical_weather = data.get("canonical_weather")

        for weather_name in weather_names:
            source_path = resolve_path(dataset_root, rgb_by_weather.get(weather_name))
            if source_path is None or not source_path.exists():
                missing.append(f"RGB/{weather_name}/{frame_label}: {source_path}")
                continue
            image = load_display_image(source_path, "rgb")
            if image is None:
                missing.append(f"RGB read failed/{weather_name}/{frame_label}: {source_path}")
                continue
            overlay = draw_annotations(
                image,
                annotations,
                f"RGB/{weather_name}",
                frame_label,
                args.box_thickness,
                args.font_scale
            )
            weather_out_dir = out_dir / "rgb" / weather_name
            weather_out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(weather_out_dir / f"{frame_label}_overlay.png"), overlay)
            written_counts[f"rgb/{weather_name}"] = written_counts.get(f"rgb/{weather_name}", 0) + 1
            weather_sheet_items.setdefault(weather_name, []).append(
                (frame_label, overlay)
            )
            frame_sheet_items[frame_label].append(
                (f"RGB/{weather_name}", overlay)
            )
            if weather_name == canonical_weather:
                canonical_overlay = overlay

        for modality, value in modality_sources(
            image_info,
            include_lidar=args.include_lidar
        ).items():
            source_path = resolve_path(dataset_root, value)
            if source_path is None or not source_path.exists():
                missing.append(f"{modality}/{frame_label}: {source_path}")
                continue
            image = load_display_image(source_path, modality)
            if image is None:
                missing.append(f"{modality} read failed/{frame_label}: {source_path}")
                continue
            overlay = draw_annotations(
                image,
                annotations,
                modality,
                frame_label,
                args.box_thickness,
                args.font_scale
            )
            modality_out_dir = out_dir / modality
            modality_out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(modality_out_dir / f"{frame_label}_overlay.png"), overlay)
            written_counts[modality] = written_counts.get(modality, 0) + 1
            modality_sheet_items.append((f"{frame_label}/{modality}", overlay))
            frame_sheet_items[frame_label].append((modality, overlay))

        if canonical_overlay is not None:
            modality_sheet_items.append((f"{frame_label}/RGB-{canonical_weather}", canonical_overlay))

    if args.contact_sheets:
        contact_dir = out_dir / "contact_sheets"
        contact_dir.mkdir(parents=True, exist_ok=True)
        for weather_name, items in sorted(weather_sheet_items.items()):
            sheet = make_contact_sheet(
                items,
                columns=max(1, min(args.max_frames, len(items))),
                thumb_size=(args.thumb_width, args.thumb_height)
            )
            cv2.imwrite(
                str(
                    contact_dir
                    / f"{weather_name}_random{len(items)}_rgb.png"
                ),
                sheet
            )
        for frame_label, items in sorted(frame_sheet_items.items()):
            sheet = make_contact_sheet(
                items,
                columns=len(items),
                thumb_size=(args.thumb_width, args.thumb_height)
            )
            cv2.imwrite(
                str(contact_dir / f"{frame_label}_four_modalities.png"),
                sheet
            )
        modality_sheet = make_contact_sheet(
            modality_sheet_items,
            columns=5 if args.include_lidar else 4,
            thumb_size=(args.thumb_width, args.thumb_height)
        )
        cv2.imwrite(str(contact_dir / "random_frames_all_modalities.png"), modality_sheet)

    summary = {
        "dataset": str(dataset_root),
        "sequence": args.sequence,
        "sampling": "random_without_replacement_per_weather",
        "sampling_seed": args.seed,
        "samples_per_weather": args.max_frames,
        "candidate_frame_count": candidate_frame_count,
        "weather_candidate_counts": weather_candidate_counts,
        "selected_by_weather": selected_by_weather,
        "frames": [path.stem for path in ann_files],
        "weather_filter": weather_filter or "all",
        "include_lidar": args.include_lidar,
        "written_counts": written_counts,
        "missing": missing,
        "manual_review_candidates": review_candidates,
        "output": str(out_dir)
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[DONE] frames={len(ann_files)}")
    print(f"[DONE] selected_by_weather={selected_by_weather}")
    print(f"[DONE] weather_count={len([key for key in written_counts if key.startswith('rgb/')])}")
    print(f"[DONE] image_overlays={sum(written_counts.values())}")
    print(f"[DONE] missing={len(missing)}")
    print(f"[DONE] manual_review_candidates={len(review_candidates)}")
    print(f"[DONE] output={out_dir}")


if __name__ == "__main__":
    main()
