"""Consolidate every map's seq_* directories into paired_weather/<map_name>.

The script keeps frame modalities aligned, rewrites paths and frame ids, prefixes
track ids with the source sequence, updates splits/COCO/manifest metadata, and
moves the original sequences to a rollback directory outside the dataset root.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SEQ_PATTERN = re.compile(r"^seq_\d+$")
FRAME_PATTERN = re.compile(r"^\d+$")


@dataclass(frozen=True)
class FrameRecord:
    source_sequence: str
    source_sequence_rel: str
    source_dir: Path
    old_frame_id: int
    old_stem: str
    new_frame_id: int
    new_stem: str
    annotation_path: Path
    merged_sequence: str
    merged_relative: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def union_fields(field_groups: Iterable[Iterable[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for fields in field_groups:
        for field in fields:
            if field not in seen:
                result.append(field)
                seen.add(field)
    return result


def write_csv_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_prefix(sequence: str) -> str:
    return f"{sequence}_"


def prefix_track_id(value: Any, sequence: str) -> Any:
    if not isinstance(value, str) or not value:
        return value
    prefix = source_prefix(sequence)
    return value if value.startswith(prefix) else prefix + value


def rewrite_frame_path(value: str, record: FrameRecord) -> str:
    old_prefix = record.source_sequence_rel.replace("\\", "/") + "/"
    normalized = value.replace("\\", "/")
    if old_prefix in normalized:
        normalized = normalized.replace(old_prefix, record.merged_relative + "/")
        old_token = "/" + record.old_stem + "."
        new_token = "/" + record.new_stem + "."
        normalized = normalized.replace(old_token, new_token)
    return normalized


def rewrite_object(value: Any, record: FrameRecord, *, rewrite_tracks: bool = True) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if rewrite_tracks and key in {"track_id", "sot_primary_track_id"}:
                result[key] = prefix_track_id(item, record.source_sequence)
            else:
                result[key] = rewrite_object(item, record, rewrite_tracks=rewrite_tracks)
        return result
    if isinstance(value, list):
        return [rewrite_object(item, record, rewrite_tracks=rewrite_tracks) for item in value]
    if isinstance(value, str):
        return rewrite_frame_path(value, record)
    return value


def rewrite_sequence_templates(value: Any, merged_relative: str) -> Any:
    if isinstance(value, dict):
        return {
            key: rewrite_sequence_templates(item, merged_relative)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_sequence_templates(item, merged_relative) for item in value]
    if isinstance(value, str):
        return value.replace("paired_weather/seq_XXXX", merged_relative)
    return value


def linked_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"Destination collision: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def linked_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Destination already exists: {destination}")
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            linked_copy(path, target)


def clear_directory_contents(path: Path) -> None:
    """Clear children while keeping the directory itself (it may be open on Windows)."""
    for child in list(path.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def move_directory_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in list(source.iterdir()):
        target = destination / child.name
        if target.exists():
            raise RuntimeError(f"Destination collision: {target}")
        child.rename(target)
    source.rmdir()


def discover_sequences(map_dir: Path) -> list[Path]:
    paired = map_dir / "paired_weather"
    if not paired.is_dir():
        return []
    return sorted(
        [path for path in paired.iterdir() if path.is_dir() and SEQ_PATTERN.match(path.name)],
        key=lambda path: path.name,
    )


def discover_frames(
    sequences: list[Path], merged_sequence: str, merged_relative: str
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for sequence_dir in sequences:
        annotation_dir = sequence_dir / "annotations"
        if not annotation_dir.is_dir():
            continue
        annotation_paths = sorted(
            [
                path
                for path in annotation_dir.glob("*.json")
                if FRAME_PATTERN.match(path.stem)
            ],
            key=lambda path: int(path.stem),
        )
        for annotation_path in annotation_paths:
            annotation = read_json(annotation_path)
            old_frame_id = int(annotation.get("frame_id", int(annotation_path.stem)))
            new_frame_id = len(records)
            records.append(
                FrameRecord(
                    source_sequence=sequence_dir.name,
                    source_sequence_rel=f"paired_weather/{sequence_dir.name}",
                    source_dir=sequence_dir,
                    old_frame_id=old_frame_id,
                    old_stem=annotation_path.stem,
                    new_frame_id=new_frame_id,
                    new_stem=f"{new_frame_id:06d}",
                    annotation_path=annotation_path,
                    merged_sequence=merged_sequence,
                    merged_relative=merged_relative,
                )
            )
    return records


def frame_lookup(records: list[FrameRecord]) -> dict[tuple[str, int], FrameRecord]:
    result: dict[tuple[str, int], FrameRecord] = {}
    for record in records:
        key = (record.source_sequence, record.old_frame_id)
        if key in result:
            raise RuntimeError(f"Duplicate source frame: {key}")
        result[key] = record
    return result


def iter_frame_files(record: FrameRecord) -> Iterable[Path]:
    for source in record.source_dir.rglob(record.old_stem + ".*"):
        if not source.is_file():
            continue
        if source.parent.name == "annotations" and source.suffix.lower() == ".json":
            continue
        yield source


def staging_relative_path(source: Path, record: FrameRecord) -> Path:
    relative = source.relative_to(record.source_dir)
    new_name = record.new_stem + source.name[len(record.old_stem) :]
    return relative.with_name(new_name)


def collect_source_tables(
    sequences: list[Path], records_by_key: dict[tuple[str, int], FrameRecord]
) -> tuple[list[str], list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
    frame_field_groups: list[list[str]] = []
    multi_field_groups: list[list[str]] = []
    merged_frame_rows: list[dict[str, Any]] = []
    merged_multi_rows: list[dict[str, Any]] = []
    merged_groundtruth: list[str] = []

    for sequence_dir in sequences:
        sequence = sequence_dir.name
        frame_fields, frame_rows = read_csv_rows(sequence_dir / "frame_index.csv")
        multi_fields, multi_rows = read_csv_rows(sequence_dir / "groundtruth_multi.csv")
        frame_field_groups.append(frame_fields)
        multi_field_groups.append(multi_fields)
        gt_path = sequence_dir / "groundtruth.txt"
        groundtruth_lines = (
            gt_path.read_text(encoding="utf-8-sig").splitlines() if gt_path.is_file() else []
        )

        seen_frames: set[int] = set()
        for row_index, source_row in enumerate(frame_rows):
            old_frame_id = int(source_row["frame_id"])
            record = records_by_key.get((sequence, old_frame_id))
            if record is None:
                continue
            seen_frames.add(old_frame_id)
            row = {
                key: rewrite_frame_path(str(value), record)
                for key, value in source_row.items()
            }
            row["frame_id"] = record.new_frame_id
            merged_frame_rows.append(row)
            merged_groundtruth.append(
                groundtruth_lines[row_index] if row_index < len(groundtruth_lines) else "0,0,0,0"
            )

        source_records = sorted(
            [record for record in records_by_key.values() if record.source_sequence == sequence],
            key=lambda record: record.old_frame_id,
        )
        for record in source_records:
            if record.old_frame_id in seen_frames:
                continue
            annotation = read_json(record.annotation_path)
            image = annotation.get("image", {})
            merged_frame_rows.append(
                {
                    "frame_id": record.new_frame_id,
                    "carla_frame": annotation.get("carla_frame", ""),
                    "rgb_canonical": rewrite_frame_path(str(image.get("rgb", "")), record),
                    "rgb_by_weather_json": json.dumps(
                        rewrite_object(image.get("rgb_by_weather", {}), record),
                        ensure_ascii=False,
                    ),
                    "num_annotations": len(annotation.get("annotations", [])),
                }
            )
            merged_groundtruth.append("0,0,0,0")

        for source_row in multi_rows:
            old_frame_id = int(source_row["frame_id"])
            record = records_by_key.get((sequence, old_frame_id))
            if record is None:
                continue
            row = {
                key: rewrite_frame_path(str(value), record)
                for key, value in source_row.items()
            }
            row["frame_id"] = record.new_frame_id
            if "track_id" in row:
                row["track_id"] = prefix_track_id(row["track_id"], sequence)
            merged_multi_rows.append(row)

    merged_frame_rows.sort(key=lambda row: int(row["frame_id"]))
    merged_multi_rows.sort(
        key=lambda row: (int(row["frame_id"]), int(row.get("ann_id", 0) or 0))
    )
    frame_fields = union_fields(frame_field_groups + [[
        "frame_id",
        "carla_frame",
        "rgb_canonical",
        "rgb_by_weather_json",
        "num_annotations",
    ]])
    multi_fields = union_fields(multi_field_groups)
    return frame_fields, merged_frame_rows, multi_fields, merged_multi_rows, merged_groundtruth


def build_merged_meta(
    map_dir: Path,
    sequences: list[Path],
    records: list[FrameRecord],
    multi_rows: list[dict[str, Any]],
    merged_sequence: str,
    merged_relative: str,
) -> dict[str, Any]:
    source_meta: list[dict[str, Any]] = []
    for sequence_dir in sequences:
        meta_path = sequence_dir / "sequence_meta.json"
        if meta_path.is_file():
            source_meta.append(read_json(meta_path))
    if not source_meta:
        manifest_path = map_dir / "dataset_manifest.json"
        if manifest_path.is_file():
            source_meta = list(read_json(manifest_path).get("sequences", []))

    merged = copy.deepcopy(source_meta[0]) if source_meta else {}
    weather_counts: Counter[str] = Counter()
    for record in records:
        annotation = read_json(record.annotation_path)
        canonical = annotation.get("canonical_weather")
        if canonical:
            weather_counts[str(canonical)] += 1

    track_counts: Counter[str] = Counter()
    track_areas: defaultdict[str, float] = defaultdict(float)
    for row in multi_rows:
        track = str(row.get("track_id", ""))
        if not track:
            continue
        track_counts[track] += 1
        try:
            track_areas[track] += float(row.get("area_px", 0.0) or 0.0)
        except ValueError:
            pass
    primary_track = None
    if track_counts:
        primary_track = max(track_counts, key=lambda key: (track_counts[key], track_areas[key], key))

    historical_sources: list[str] = []
    for meta, sequence_dir in zip(source_meta, sequences):
        prior = meta.get("source_sequences")
        if isinstance(prior, list) and prior:
            historical_sources.extend(str(item) for item in prior)
        else:
            historical_sources.append(sequence_dir.name)
    historical_sources = list(dict.fromkeys(historical_sources))

    merged.update(
        {
            "sequence": merged_relative,
            "sequence_name": merged_sequence,
            "relative_sequence_path": merged_relative,
            "route": "merged_multiregion",
            "center_xy": None,
            "frame_count": len(records),
            "source_sequence_count": len(historical_sources),
            "source_sequences": historical_sources,
            "captured_weather_counts": dict(sorted(weather_counts.items())),
            "planned_weather_counts": dict(sorted(weather_counts.items())),
            "sot_primary_track_id": primary_track,
            "sot_primary_visible_frames": track_counts.get(primary_track, 0),
            "sequence_consolidated": True,
        }
    )
    return merged


def build_staging(
    map_dir: Path,
    sequences: list[Path],
    staging: Path,
    merged_sequence: str,
    merged_relative: str,
) -> tuple[list[FrameRecord], dict[str, str], dict[tuple[str, int], FrameRecord], dict[str, Any]]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    records = discover_frames(sequences, merged_sequence, merged_relative)
    if not records:
        raise RuntimeError(f"No annotation frames found in {map_dir}")
    records_by_key = frame_lookup(records)
    path_mapping: dict[str, str] = {}

    for record in records:
        for source in iter_frame_files(record):
            relative = staging_relative_path(source, record)
            destination = staging / relative
            linked_copy(source, destination)
            old_rel = f"{record.source_sequence_rel}/{source.relative_to(record.source_dir).as_posix()}"
            new_rel = f"{merged_relative}/{relative.as_posix()}"
            path_mapping[old_rel] = new_rel

        annotation = read_json(record.annotation_path)
        annotation = rewrite_object(annotation, record)
        annotation["sequence"] = merged_relative
        annotation["frame_id"] = record.new_frame_id
        annotation_destination = staging / "annotations" / f"{record.new_stem}.json"
        write_json(annotation_destination, annotation)
        path_mapping[
            f"{record.source_sequence_rel}/annotations/{record.old_stem}.json"
        ] = f"{merged_relative}/annotations/{record.new_stem}.json"

    frame_fields, frame_rows, multi_fields, multi_rows, groundtruth = collect_source_tables(
        sequences, records_by_key
    )
    if len(frame_rows) != len(records):
        raise RuntimeError(
            f"frame_index row count mismatch: {len(frame_rows)} rows for {len(records)} frames"
        )
    write_csv_rows(staging / "frame_index.csv", frame_fields, frame_rows)
    if multi_fields:
        write_csv_rows(staging / "groundtruth_multi.csv", multi_fields, multi_rows)
    else:
        (staging / "groundtruth_multi.csv").write_text("", encoding="utf-8")
    (staging / "groundtruth.txt").write_text(
        "\n".join(groundtruth) + ("\n" if groundtruth else ""), encoding="utf-8"
    )
    merged_meta = build_merged_meta(
        map_dir,
        sequences,
        records,
        multi_rows,
        merged_sequence,
        merged_relative,
    )
    write_json(staging / "sequence_meta.json", merged_meta)
    validate_staging(map_dir, staging, records, merged_relative)
    return records, path_mapping, records_by_key, merged_meta


def image_path_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from image_path_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from image_path_strings(item)
    elif isinstance(value, str) and value.startswith("paired_weather/"):
        yield value


def validate_staging(
    map_dir: Path, staging: Path, records: list[FrameRecord], merged_relative: str
) -> None:
    annotations = sorted((staging / "annotations").glob("*.json"))
    if len(annotations) != len(records):
        raise RuntimeError(
            f"Annotation count mismatch: {len(annotations)} != {len(records)}"
        )
    expected_stems = [f"{index:06d}" for index in range(len(records))]
    if [path.stem for path in annotations] != expected_stems:
        raise RuntimeError("Merged annotation frame ids are not contiguous")

    missing: list[str] = []
    for index, annotation_path in enumerate(annotations):
        annotation = read_json(annotation_path)
        if int(annotation.get("frame_id", -1)) != index:
            raise RuntimeError(f"Wrong frame_id in {annotation_path}")
        if annotation.get("sequence") != merged_relative:
            raise RuntimeError(f"Wrong sequence in {annotation_path}")
        for relative in image_path_strings(annotation.get("image", {})):
            prefix = merged_relative + "/"
            if not relative.startswith(prefix):
                missing.append(relative)
                continue
            actual = staging / relative[len(prefix) :]
            if not actual.is_file():
                missing.append(relative)
    if missing:
        sample = "\n  ".join(missing[:20])
        raise RuntimeError(f"Missing {len(missing)} referenced modality files:\n  {sample}")

    _, frame_rows = read_csv_rows(staging / "frame_index.csv")
    if len(frame_rows) != len(records):
        raise RuntimeError("Merged frame_index.csv is incomplete")
    gt_lines = (staging / "groundtruth.txt").read_text(encoding="utf-8").splitlines()
    if len(gt_lines) != len(records):
        raise RuntimeError("Merged groundtruth.txt is incomplete")


def update_split_files(map_dir: Path, path_mapping: dict[str, str]) -> None:
    split_roots = [map_dir / "splits", map_dir / "splits_by_weather"]
    for split_root in split_roots:
        if not split_root.is_dir():
            continue
        for path in split_root.rglob("*.txt"):
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            rewritten: list[str] = []
            for line in lines:
                stripped = line.strip().replace("\\", "/")
                if stripped in path_mapping:
                    stripped = path_mapping[stripped]
                rewritten.append(stripped)
            path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")


def update_coco_files(
    map_dir: Path,
    path_mapping: dict[str, str],
    records_by_key: dict[tuple[str, int], FrameRecord],
    merged_relative: str,
) -> None:
    coco_root = map_dir / "coco"
    if not coco_root.is_dir():
        return
    for path in coco_root.glob("*.json"):
        document = read_json(path)
        source_sequence_by_image: dict[Any, str] = {}
        for image in document.get("images", []):
            old_file = str(image.get("file_name", "")).replace("\\", "/")
            old_sequence_rel = str(image.get("sequence", "")).replace("\\", "/")
            sequence = old_sequence_rel.rsplit("/", 1)[-1]
            old_frame_id = int(image.get("frame_id", Path(old_file).stem))
            record = records_by_key.get((sequence, old_frame_id))
            if record is None:
                raise RuntimeError(f"COCO image has no frame mapping: {path}: {old_file}")
            if old_file not in path_mapping:
                raise RuntimeError(f"COCO file path has no mapping: {path}: {old_file}")
            source_sequence_by_image[image.get("id")] = sequence
            image["file_name"] = path_mapping[old_file]
            image["frame_id"] = record.new_frame_id
            image["sequence"] = merged_relative
        for annotation in document.get("annotations", []):
            sequence = source_sequence_by_image.get(annotation.get("image_id"))
            if sequence and "track_id" in annotation:
                annotation["track_id"] = prefix_track_id(annotation["track_id"], sequence)
        write_json(path, document)


def update_manifest(
    map_dir: Path,
    merged_meta: dict[str, Any],
    frame_count: int,
    merged_relative: str,
) -> None:
    path = map_dir / "dataset_manifest.json"
    if not path.is_file():
        return
    manifest = rewrite_sequence_templates(read_json(path), merged_relative)
    manifest["sequences"] = [merged_meta]
    manifest["total_saved_frames"] = frame_count
    manifest["total_weather_rgb_images"] = len(
        list((map_dir / merged_relative / "rgb").rglob("*.png"))
    )
    manifest["sequence_consolidation"] = {
        "merged_sequence": merged_relative,
        "frame_count": frame_count,
        "source_sequence_count": merged_meta.get("source_sequence_count", 0),
        "source_sequences": merged_meta.get("source_sequences", []),
    }
    write_json(path, manifest)


def prune_missing_root_references(
    map_dir: Path, merged_relative: str
) -> dict[str, int]:
    removed_split_lines = 0
    removed_coco_images = 0
    removed_coco_annotations = 0
    for split_root in (map_dir / "splits", map_dir / "splits_by_weather"):
        if not split_root.is_dir():
            continue
        for path in split_root.rglob("*.txt"):
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            kept = [line for line in lines if (map_dir / Path(line)).is_file()]
            removed_split_lines += len(lines) - len(kept)
            if len(kept) != len(lines):
                path.write_text(
                    "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
                )

    coco_root = map_dir / "coco"
    if coco_root.is_dir():
        for path in coco_root.glob("*.json"):
            document = read_json(path)
            images = list(document.get("images", []))
            kept_images = [
                image
                for image in images
                if (map_dir / Path(str(image.get("file_name", "")))).is_file()
            ]
            kept_image_ids = {image.get("id") for image in kept_images}
            annotations = list(document.get("annotations", []))
            kept_annotations = [
                annotation
                for annotation in annotations
                if annotation.get("image_id") in kept_image_ids
            ]
            removed_coco_images += len(images) - len(kept_images)
            removed_coco_annotations += len(annotations) - len(kept_annotations)
            if len(kept_images) != len(images) or len(kept_annotations) != len(annotations):
                document["images"] = kept_images
                document["annotations"] = kept_annotations
                write_json(path, document)

    manifest_path = map_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = rewrite_sequence_templates(
            read_json(manifest_path), merged_relative
        )
        actual_frames = len(
            list((map_dir / merged_relative / "annotations").glob("*.json"))
        )
        actual_rgb = len(list((map_dir / merged_relative / "rgb").rglob("*.png")))
        manifest["total_saved_frames"] = actual_frames
        manifest["total_weather_rgb_images"] = actual_rgb
        sequences = manifest.get("sequences")
        if isinstance(sequences, list) and len(sequences) == 1:
            sequences[0]["frame_count"] = actual_frames
        standard = manifest.get("standard_artifacts")
        if isinstance(standard, dict):
            split_counts: dict[str, int] = {}
            for split_name in ("train", "val", "test"):
                path = map_dir / "splits" / f"{split_name}.txt"
                if path.is_file():
                    split_counts[split_name] = len(
                        [
                            line
                            for line in path.read_text(encoding="utf-8-sig").splitlines()
                            if line.strip()
                        ]
                    )
            standard["split_counts"] = split_counts
            write_json(manifest_path, manifest)

    return {
        "removed_split_lines": removed_split_lines,
        "removed_coco_images": removed_coco_images,
        "removed_coco_annotations": removed_coco_annotations,
    }


def validate_root_artifacts(
    map_dir: Path, frame_count: int, merged_relative: str
) -> dict[str, Any]:
    merged_dir = map_dir / merged_relative
    errors: list[str] = []
    split_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        path = map_dir / "splits" / f"{split_name}.txt"
        if not path.is_file():
            continue
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        split_counts[split_name] = len(lines)
        for relative in lines:
            if not (map_dir / Path(relative)).is_file():
                errors.append(f"missing split target: {relative}")
            if not relative.startswith(merged_relative + "/"):
                errors.append(f"old split sequence path: {relative}")

    coco_counts: dict[str, int] = {}
    coco_root = map_dir / "coco"
    if coco_root.is_dir():
        for path in coco_root.glob("*.json"):
            document = read_json(path)
            images = document.get("images", [])
            coco_counts[path.name] = len(images)
            for image in images:
                relative = str(image.get("file_name", ""))
                if not (map_dir / Path(relative)).is_file():
                    errors.append(f"missing COCO image: {path.name}: {relative}")
                if image.get("sequence") != merged_relative:
                    errors.append(f"old COCO sequence: {path.name}: {relative}")

    annotation_count = len(list((merged_dir / "annotations").glob("*.json")))
    if annotation_count != frame_count:
        errors.append(f"annotation count {annotation_count} != {frame_count}")
    if errors:
        raise RuntimeError("Root artifact validation failed:\n  " + "\n  ".join(errors[:30]))
    return {
        "frames": frame_count,
        "split_counts": split_counts,
        "coco_image_counts": coco_counts,
    }


def backup_root_artifacts(map_dir: Path, backup_map: Path) -> None:
    for name in (
        "dataset_manifest.json",
        "coco",
        "splits",
        "splits_by_weather",
    ):
        source = map_dir / name
        destination = backup_map / "root_artifacts" / name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def restore_root_artifacts(map_dir: Path, backup_map: Path) -> None:
    root_artifacts = backup_map / "root_artifacts"
    if not root_artifacts.is_dir():
        return
    for source in root_artifacts.iterdir():
        destination = map_dir / source.name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def consolidate_map(map_dir: Path, backup_map: Path) -> dict[str, Any]:
    paired = map_dir / "paired_weather"
    merged_sequence = map_dir.name
    merged_relative = f"paired_weather/{merged_sequence}"
    merged_destination = paired / merged_sequence
    sequences = discover_sequences(map_dir)
    if merged_destination.is_dir():
        if sequences:
            raise RuntimeError(
                f"Both map-named output and seq_* sources exist in {paired}"
            )
        count = len(list((merged_destination / "annotations").glob("*.json")))
        backup_root_artifacts(map_dir, backup_map)
        pruned = prune_missing_root_references(map_dir, merged_relative)
        validation = validate_root_artifacts(map_dir, count, merged_relative)
        return {
            "map": map_dir.name,
            "status": "already_consolidated",
            "sequence_name": merged_sequence,
            "frames": count,
            "pruned_stale_references": pruned,
            "validation": validation,
        }
    if not sequences:
        return {"map": map_dir.name, "status": "skipped", "reason": "no sequences"}

    staging = paired / f".{merged_sequence}_consolidation_staging"
    print(f"[BUILD] {map_dir.name}: {len(sequences)} source sequences")
    records, path_mapping, records_by_key, merged_meta = build_staging(
        map_dir,
        sequences,
        staging,
        merged_sequence,
        merged_relative,
    )
    print(f"[VERIFY] {map_dir.name}: staging passed ({len(records)} frames)")

    backup_root_artifacts(map_dir, backup_map)
    backup_sequences = backup_map / "paired_weather"
    backup_sequences.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    committed = False
    try:
        for sequence in sequences:
            destination = backup_sequences / sequence.name
            if destination.exists():
                raise RuntimeError(f"Backup destination already exists: {destination}")
            sequence.rename(destination)
            moved.append((sequence, destination))
        staging.rename(merged_destination)
        committed = True
        update_split_files(map_dir, path_mapping)
        update_coco_files(map_dir, path_mapping, records_by_key, merged_relative)
        update_manifest(map_dir, merged_meta, len(records), merged_relative)
        validation = validate_root_artifacts(map_dir, len(records), merged_relative)
    except Exception:
        if committed and merged_destination.exists():
            shutil.rmtree(merged_destination)
        if staging.exists():
            shutil.rmtree(staging)
        for original, backup in reversed(moved):
            if backup.exists() and not original.exists():
                backup.rename(original)
        restore_root_artifacts(map_dir, backup_map)
        raise

    print(f"[DONE] {map_dir.name}: one sequence, {len(records)} frames")
    return {
        "map": map_dir.name,
        "status": "consolidated",
        "source_sequences": [path.name for path in sequences],
        "frames": len(records),
        "validation": validation,
    }


def audit_map(map_dir: Path) -> dict[str, Any]:
    sequences = discover_sequences(map_dir)
    map_named = map_dir / "paired_weather" / map_dir.name
    if map_named.is_dir():
        sequences = [map_named]
    return {
        "map": map_dir.name,
        "sequences": len(sequences),
        "sequence_names": [path.name for path in sequences],
        "annotation_frames": sum(
            len(list((path / "annotations").glob("*.json"))) for path in sequences
        ),
        "rgb_images": sum(len(list((path / "rgb").rglob("*.png"))) for path in sequences),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"E:\pythonProject\air_groud\get_img\dataset_uav_multimap_town600_hutb300_ccsp300"
        ),
    )
    parser.add_argument(
        "--maps",
        nargs="*",
        help="Optional map directory names. Default: every non-underscore map directory.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform consolidation. Without this flag, only audit the dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Dataset root does not exist: {root}")
    selected = set(args.maps or [])
    maps = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith("_")
            and (not selected or path.name in selected)
        ],
        key=lambda path: path.name,
    )
    if selected:
        missing = selected - {path.name for path in maps}
        if missing:
            raise RuntimeError(f"Unknown map directories: {sorted(missing)}")

    audit = [audit_map(path) for path in maps]
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not args.execute:
        print("[DRY-RUN] Audit only. Add --execute to consolidate.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root.parent / "dataset_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    try:
        for map_dir in maps:
            backup_map = backup_root / map_dir.name / stamp
            report.append(consolidate_map(map_dir, backup_map))
    finally:
        write_json(
            root / "sequence_consolidation_report.json",
            {
                "dataset_root": str(root),
                "backup_root": str(backup_root),
                "backup_session": stamp,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "maps": report,
            },
        )
    print(f"[BACKUP] Original sequences: {backup_root}/<map>/{stamp}")
    print("[DONE] Sequence consolidation complete.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
