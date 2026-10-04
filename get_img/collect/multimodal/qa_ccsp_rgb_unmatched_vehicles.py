#!/usr/bin/env python3
"""Reject CCSP RGB frames containing vehicle-like assets without GT boxes."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ultralytics import YOLO


COCO_VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck
RESULT_MARKER = "CCSP_RGB_QA_JSON="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou-match", type=float, default=0.15)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"reviewed": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("reviewed"), dict):
            return payload
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {"reviewed": {}}


def collect_entries(root: Path, reviewed: Dict[str, object]) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    annotation_root = root / "paired_weather"
    for annotation_path in sorted(annotation_root.glob("seq_*/annotations/*.json")):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            rgb_relative = annotation["image"]["rgb"]
            rgb_path = root / str(rgb_relative)
            if not rgb_path.is_file():
                continue
            image_hash = file_sha256(rgb_path)
            if image_hash in reviewed:
                continue
            image_width = int(annotation["image"]["width"])
            image_height = int(annotation["image"]["height"])
            gt_boxes = [
                [float(value) for value in item["bbox_xyxy"]]
                for item in annotation.get("annotations", [])
                if item.get("class_name") == "vehicle"
            ]
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
        entries.append({
            "annotation_path": annotation_path,
            "rgb_path": rgb_path,
            "hash": image_hash,
            "width": image_width,
            "height": image_height,
            "gt_boxes": gt_boxes,
            "sequence": annotation_path.parent.parent.name,
            "frame": annotation_path.stem,
        })
    return entries


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    cache_path = root / ".ccsp_rgb_asset_qa_cache.json"
    cache = load_cache(cache_path)
    reviewed = cache["reviewed"]
    entries = collect_entries(root, reviewed)
    if not entries:
        print(RESULT_MARKER + json.dumps({"reviewed": 0, "rejected": []}))
        return 0

    model = YOLO(str(args.weights.resolve()))
    results = model.predict(
        [str(item["rgb_path"]) for item in entries],
        imgsz=args.imgsz,
        conf=args.conf,
        classes=sorted(COCO_VEHICLE_CLASS_IDS),
        device=args.device,
        verbose=False,
        batch=min(8, len(entries)),
    )

    rejected: List[Dict[str, object]] = []
    for entry, result in zip(entries, results):
        gt_boxes = entry["gt_boxes"]
        image_area = float(entry["width"] * entry["height"])
        unmatched: List[Dict[str, object]] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in COCO_VEHICLE_CLASS_IDS:
                continue
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            area_ratio = (
                max(0.0, xyxy[2] - xyxy[0])
                * max(0.0, xyxy[3] - xyxy[1])
                / image_area
            )
            if area_ratio < args.min_area_ratio:
                continue
            best_iou = max((box_iou(xyxy, gt) for gt in gt_boxes), default=0.0)
            if best_iou < args.iou_match:
                unmatched.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox_xyxy": xyxy,
                    "area_ratio": area_ratio,
                    "best_gt_iou": best_iou,
                })

        record = {
            "sequence": entry["sequence"],
            "frame": entry["frame"],
            "unmatched_vehicle_detections": unmatched,
        }
        if unmatched:
            rejected.append(record)
        else:
            # Cache passing frames only. If an identical broken baked asset is
            # sampled again later it must be rejected again, not skipped merely
            # because its first copy was already quarantined.
            reviewed[entry["hash"]] = record

    cache["reviewed"] = reviewed
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(RESULT_MARKER + json.dumps({
        "reviewed": len(entries),
        "rejected": rejected,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
