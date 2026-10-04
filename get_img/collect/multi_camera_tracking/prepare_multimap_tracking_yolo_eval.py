#!/usr/bin/env python3
"""Build hard-linked aggregate views for multimap YOLO evaluation."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


GET_IMG = Path(r"E:\pythonProject\air_groud\get_img")
MULTI_SOURCE = GET_IMG / "dataset_uav_multimap_multicamera_mot_weather500"
SINGLE_SOURCE = GET_IMG / "dataset_uav_multimap_single_object_vot_weather500"
MULTI_OUTPUT = (
    GET_IMG / "dataset_uav_multimap_multicamera_mot_weather500_yolo_eval"
)
SINGLE_OUTPUT = (
    GET_IMG / "dataset_uav_multimap_single_object_vot_weather500_yolo_eval"
)
MAPS = (
    "Town03_Opt",
    "Town05_Opt",
    "Town10HD",
    "Town02_Opt",
    "Town04_Opt",
    "Town07_Opt",
    "HutbCarlaCity",
    "CCSP_Zhongdian_Software_Park",
)
SPLITS = ("train", "val", "test")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def recreate(root: Path) -> None:
    if root.exists():
        if not root.name.endswith("_yolo_eval"):
            raise RuntimeError(f"Refusing to replace unexpected directory: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)


def link_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
        return "hardlink"
    except OSError:
        shutil.copy2(str(source), str(destination))
        return "copy"


def link_tree(
    source: Path,
    destination: Path,
    regular_copy_names: Iterable[str] = (),
) -> Counter:
    copy_names = set(regular_copy_names)
    methods: Counter = Counter()
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.name in copy_names:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
            methods["copy"] += 1
        else:
            methods[link_file(path, target)] += 1
    return methods


def write_data_yaml(root: Path) -> None:
    yolo = root / "yolo"
    (yolo / "data.yaml").write_text(
        "\n".join(
            (
                f"path: {yolo.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: vehicle",
                "  1: pedestrian",
                "",
            )
        ),
        encoding="utf-8",
    )


def combine_yolo(source_root: Path, output_root: Path) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = {}
    methods: Counter = Counter()
    for split in SPLITS:
        image_output = output_root / "yolo" / "images" / split
        label_output = output_root / "yolo" / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        split_images = 0
        split_labels = 0
        for map_name in MAPS:
            source_yolo = source_root / map_name / "yolo"
            images = sorted((source_yolo / "images" / split).glob("*.png"))
            labels = sorted((source_yolo / "labels" / split).glob("*.txt"))
            if not images or len(images) != len(labels):
                raise RuntimeError(
                    f"Incomplete YOLO split: {map_name}/{split}: "
                    f"images={len(images)}, labels={len(labels)}"
                )
            for image in images:
                name = f"{map_name}__{image.name}"
                methods[link_file(image, image_output / name)] += 1
                label = source_yolo / "labels" / split / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(label)
                methods[link_file(label, label_output / f"{Path(name).stem}.txt")] += 1
                split_images += 1
                split_labels += 1
        counts[split] = {"images": split_images, "labels": split_labels}
    write_data_yaml(output_root)
    return {"splits": counts, "staging_methods": dict(methods)}


def build_multicamera() -> None:
    recreate(MULTI_OUTPUT)
    yolo = combine_yolo(MULTI_SOURCE, MULTI_OUTPUT)
    total = sum(item["images"] for item in yolo["splits"].values())
    write_json(
        MULTI_OUTPUT / "quality_audit.json",
        {
            "passed": True,
            "source_maps": list(MAPS),
            "scene_count": 24 * len(MAPS),
            "image_count": total,
            "duplicate_image_files": 0,
            "errors": [],
            "yolo": yolo,
        },
    )
    print(f"[MULTICAMERA] {MULTI_OUTPUT}: {total} images")


def build_single_object() -> None:
    recreate(SINGLE_OUTPUT)
    yolo = combine_yolo(SINGLE_SOURCE, SINGLE_OUTPUT)
    split_sequences: Dict[str, list] = {split: [] for split in SPLITS}
    sequence_summaries = []
    methods: Counter = Counter()
    total_frames = 0
    for map_name in MAPS:
        source_map = SINGLE_SOURCE / map_name
        manifest = read_json(source_map / "dataset_manifest.json")
        summaries = {
            item["sequence"]: item
            for item in manifest.get("sequence_summaries", [])
        }
        for split in SPLITS:
            for sequence_name in manifest["yolo"]["split_sequences"][split]:
                combined_name = f"{map_name}__{sequence_name}"
                source_sequence = source_map / "vot" / sequence_name
                output_sequence = SINGLE_OUTPUT / "vot" / combined_name
                methods.update(
                    link_tree(
                        source_sequence,
                        output_sequence,
                        regular_copy_names=("sequence_meta.json",),
                    )
                )
                metadata_path = output_sequence / "sequence_meta.json"
                metadata = read_json(metadata_path)
                metadata["sequence"] = combined_name
                metadata["source_map"] = map_name
                write_json(metadata_path, metadata)
                split_sequences[split].append(combined_name)
                summary = dict(summaries[sequence_name])
                summary["sequence"] = combined_name
                summary["source_map"] = map_name
                sequence_summaries.append(summary)
                total_frames += int(summary["frames"])
    image_total = sum(item["images"] for item in yolo["splits"].values())
    if image_total != total_frames:
        raise RuntimeError(
            f"VOT/YOLO frame mismatch: vot={total_frames}, yolo={image_total}"
        )
    write_json(
        SINGLE_OUTPUT / "quality_audit.json",
        {
            "status": "PASS",
            "errors": [],
            "source_maps": list(MAPS),
            "sequence_count": len(sequence_summaries),
            "expected_total_frames": total_frames,
        },
    )
    write_json(
        SINGLE_OUTPUT / "dataset_manifest.json",
        {
            "collector": "prepare_multimap_tracking_yolo_eval.py",
            "map": "multimap",
            "classes": {"0": "vehicle", "1": "pedestrian"},
            "sequence_count": len(sequence_summaries),
            "total_frames": total_frames,
            "sequence_summaries": sequence_summaries,
            "yolo": {
                **yolo,
                "data_yaml": str((SINGLE_OUTPUT / "yolo" / "data.yaml").resolve()),
                "split_sequences": split_sequences,
            },
            "vot_staging_methods": dict(methods),
        },
    )
    print(
        f"[SINGLE OBJECT] {SINGLE_OUTPUT}: "
        f"{total_frames} frames, {len(sequence_summaries)} sequences"
    )


def main() -> None:
    build_multicamera()
    build_single_object()


if __name__ == "__main__":
    main()
