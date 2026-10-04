#!/usr/bin/env python
"""Generate and install a pedestrian navigation mesh for the CCSP roadbuild map.

The released CCSP simulator contains the cooked roadbuild level and an
OpenDRIVE file, but it does not contain the OBJ exported by CARLA Exporter or a
roadbuild navigation BIN.  This script reconstructs the real OpenDRIVE
``sidewalk`` lanes using CARLA's exact XODR waypoint API, adds the OpenDRIVE
crosswalk polygons, runs CARLA's RecastBuilder, and optionally installs the
result into the dedicated simulator package.

Run this script with the Python 3.7 environment matching the dedicated CARLA
package while that simulator is running on the requested RPC port.
"""

from __future__ import print_function

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import carla


DEFAULT_BUILD_DIR = Path(r"E:\OpenHUTB\中电软件园\ccsp_navigation_build")
DEFAULT_RECAST_BUILDER = Path(
    r"E:\OpenHUTB\hutb-2.3\hutb-2.3\Build\recast-build"
    r"\RecastBuilder\Release\RecastBuilder.exe"
)
DEFAULT_NAV_DIR = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor"
    r"\CarlaUE4\Content\Carla\Maps\Nav"
)
DEFAULT_MAP_NAV_DIR = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor"
    r"\CarlaUE4\Content\roadrunner\test1\map\Nav"
)
DEFAULT_XODR = Path(
    r"E:\OpenHUTB\中电软件园\WindowsNoEditor\WindowsNoEditor"
    r"\CarlaUE4\Content\roadrunner\test1\map\OpenDrive\roadbuild.xodr"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--expected-map", default="roadbuild")
    parser.add_argument("--sample-distance", type=float, default=1.0)
    parser.add_argument("--max-segment-gap", type=float, default=3.0)
    parser.add_argument("--xodr", type=Path, default=DEFAULT_XODR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument(
        "--recast-builder", type=Path, default=DEFAULT_RECAST_BUILDER
    )
    parser.add_argument("--nav-dir", type=Path, default=DEFAULT_NAV_DIR)
    parser.add_argument(
        "--map-nav-dir", type=Path, default=DEFAULT_MAP_NAV_DIR
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--install", action="store_true")
    return parser.parse_args()


def distance(a, b):
    dx = float(a.x) - float(b.x)
    dy = float(a.y) - float(b.y)
    dz = float(a.z) - float(b.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def right_vector(yaw_degrees):
    yaw = math.radians(float(yaw_degrees))
    return -math.sin(yaw), math.cos(yaw)


def offset_point(location, right_xy, offset):
    return (
        float(location.x) + right_xy[0] * offset,
        float(location.y) + right_xy[1] * offset,
        float(location.z) + 0.03,
    )


def obj_point(carla_xyz):
    """Convert CARLA (x, y, z) to the OBJ axis order used by CARLA tools."""
    return carla_xyz[0], carla_xyz[2], carla_xyz[1]


def obj_normal_y(a, b, c):
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    return abz * acx - abx * acz


class ObjWriter(object):
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.objects = []

    def add_quad(self, name, points):
        converted = [obj_point(point) for point in points]
        base = len(self.vertices) + 1
        self.vertices.extend(converted)
        first = (base, base + 1, base + 2)
        second = (base, base + 2, base + 3)
        if obj_normal_y(converted[0], converted[1], converted[2]) < 0.0:
            first = (base, base + 2, base + 1)
            second = (base, base + 3, base + 2)
        self.faces.extend((first, second))
        self.objects.append((name, 2))

    def add_polygon(self, name, points):
        if len(points) < 3:
            return
        converted = [obj_point(point) for point in points]
        base = len(self.vertices) + 1
        self.vertices.extend(converted)
        face_count = 0
        for index in range(1, len(converted) - 1):
            face = (base, base + index, base + index + 1)
            if obj_normal_y(
                converted[0], converted[index], converted[index + 1]
            ) < 0.0:
                face = (base, base + index + 1, base + index)
            self.faces.append(face)
            face_count += 1
        self.objects.append((name, face_count))

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("# CCSP pedestrian navigation geometry\n")
            for x, y, z in self.vertices:
                handle.write("v %.6f %.6f %.6f\n" % (x, y, z))
            face_cursor = 0
            for name, face_count in self.objects:
                handle.write("o %s\n" % name)
                handle.write("usemtl sidewalk\n")
                for face in self.faces[face_cursor : face_cursor + face_count]:
                    handle.write("f %d %d %d\n" % face)
                face_cursor += face_count


def split_crosswalks(points):
    polygons = []
    current = []
    for point in points:
        xyz = (float(point.x), float(point.y), float(point.z) + 0.04)
        if not current:
            current.append(xyz)
            continue
        if (
            len(current) >= 3
            and abs(xyz[0] - current[0][0]) < 1.0e-4
            and abs(xyz[1] - current[0][1]) < 1.0e-4
            and abs(xyz[2] - current[0][2]) < 1.0e-4
        ):
            polygons.append(current)
            current = []
        else:
            current.append(xyz)
    if len(current) >= 3:
        polygons.append(current)
    return polygons


def load_sidewalk_lane_sections(xodr_path):
    if not xodr_path.is_file():
        raise FileNotFoundError("OpenDRIVE file not found: %s" % xodr_path)
    root = ElementTree.parse(str(xodr_path)).getroot()
    records = []
    for road in root.findall("road"):
        road_id = int(road.attrib["id"])
        road_length = float(road.attrib["length"])
        lanes = road.find("lanes")
        if lanes is None:
            continue
        sections = list(lanes.findall("laneSection"))
        for section_index, section in enumerate(sections):
            start_s = float(section.attrib["s"])
            end_s = (
                float(sections[section_index + 1].attrib["s"])
                if section_index + 1 < len(sections)
                else road_length
            )
            for side_name in ("left", "right"):
                side = section.find(side_name)
                if side is None:
                    continue
                for lane in side.findall("lane"):
                    if lane.attrib.get("type", "").lower() != "sidewalk":
                        continue
                    records.append(
                        {
                            "road_id": road_id,
                            "section_index": section_index,
                            "lane_id": int(lane.attrib["id"]),
                            "start_s": start_s,
                            "end_s": end_s,
                        }
                    )
    return records


def sample_s_values(start_s, end_s, spacing):
    length = end_s - start_s
    if length <= 1.0e-4:
        return []
    margin = min(0.05, length * 0.25)
    first = start_s + margin
    last = end_s - margin
    if last <= first:
        return [(start_s + end_s) * 0.5]
    values = []
    value = first
    while value < last:
        values.append(value)
        value += spacing
    if not values or last - values[-1] > spacing * 0.25:
        values.append(last)
    return values


def sample_sidewalk_waypoints(carla_map, lane_section, spacing):
    sampled = []
    for road_s in sample_s_values(
        lane_section["start_s"], lane_section["end_s"], spacing
    ):
        waypoint = carla_map.get_waypoint_xodr(
            lane_section["road_id"], lane_section["lane_id"], road_s
        )
        if waypoint is None or str(waypoint.lane_type).lower() != "sidewalk":
            continue
        if sampled and distance(
            sampled[-1].transform.location, waypoint.transform.location
        ) < 1.0e-3:
            continue
        sampled.append(waypoint)
    return sampled


def build_navigation_geometry(carla_map, args):
    writer = ObjWriter()
    lane_sections = load_sidewalk_lane_sections(args.xodr)
    strip_segments = 0
    skipped_gaps = 0
    sampled_waypoints = 0
    widths = []
    for lane_section in lane_sections:
        waypoints = sample_sidewalk_waypoints(
            carla_map, lane_section, args.sample_distance
        )
        sampled_waypoints += len(waypoints)
        widths.extend(float(waypoint.lane_width) for waypoint in waypoints)
        for index in range(len(waypoints) - 1):
            first = waypoints[index]
            second = waypoints[index + 1]
            if (
                distance(first.transform.location, second.transform.location)
                > args.max_segment_gap
            ):
                skipped_gaps += 1
                continue
            first_right = right_vector(first.transform.rotation.yaw)
            second_right = right_vector(second.transform.rotation.yaw)
            first_half_width = max(0.25, float(first.lane_width) * 0.5)
            second_half_width = max(0.25, float(second.lane_width) * 0.5)
            first_left = offset_point(
                first.transform.location, first_right, -first_half_width
            )
            first_right_edge = offset_point(
                first.transform.location, first_right, first_half_width
            )
            second_left = offset_point(
                second.transform.location, second_right, -second_half_width
            )
            second_right_edge = offset_point(
                second.transform.location, second_right, second_half_width
            )
            writer.add_quad(
                "sidewalk_%d_%d_%d_%d"
                % (
                    lane_section["road_id"],
                    lane_section["section_index"],
                    lane_section["lane_id"],
                    index,
                ),
                (first_left, first_right_edge, second_right_edge, second_left),
            )
            strip_segments += 1

    crosswalks = split_crosswalks(list(carla_map.get_crosswalks()))
    for index, polygon in enumerate(crosswalks):
        writer.add_polygon("crosswalk_%04d" % index, polygon)

    stats = {
        "map": carla_map.name,
        "geometry_source": "OpenDRIVE sidewalk lanes via get_waypoint_xodr",
        "xodr_path": str(args.xodr),
        "sample_distance_m": args.sample_distance,
        "sidewalk_lane_sections": len(lane_sections),
        "sampled_sidewalk_waypoints": sampled_waypoints,
        "sidewalk_width_min_m": min(widths) if widths else None,
        "sidewalk_width_max_m": max(widths) if widths else None,
        "sidewalk_segments": strip_segments,
        "skipped_discontinuous_segments": skipped_gaps,
        "crosswalk_polygons": len(crosswalks),
        "obj_vertices": len(writer.vertices),
        "obj_faces": len(writer.faces),
    }
    return writer, stats


def run_recast(builder, obj_path):
    if not builder.is_file():
        raise FileNotFoundError("RecastBuilder not found: %s" % builder)
    completed = subprocess.run(
        [str(builder), str(obj_path)],
        cwd=str(obj_path.parent),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("RecastBuilder exited with %d" % completed.returncode)
    bin_path = obj_path.with_suffix(".bin")
    if not bin_path.is_file() or bin_path.stat().st_size == 0:
        raise RuntimeError("RecastBuilder did not create %s" % bin_path)
    return bin_path


def install_navigation(bin_path, nav_dirs):
    installed = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for nav_dir in nav_dirs:
        nav_dir.mkdir(parents=True, exist_ok=True)
        destination = nav_dir / "roadbuild.bin"
        backup = None
        if destination.exists():
            backup = destination.with_name("roadbuild.bin.backup_%s" % stamp)
            shutil.copy2(str(destination), str(backup))
        shutil.copy2(str(bin_path), str(destination))
        installed.append(
            {
                "path": str(destination),
                "previous_bin_backup": str(backup) if backup else None,
            }
        )
    return installed


def main():
    args = parse_args()
    if args.sample_distance <= 0.0:
        raise ValueError("--sample-distance must be positive")
    if args.max_segment_gap <= args.sample_distance:
        raise ValueError("--max-segment-gap must exceed --sample-distance")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    if Path(carla_map.name).name.lower() != args.expected_map.lower():
        raise RuntimeError(
            "Expected map %s, current map is %s"
            % (args.expected_map, carla_map.name)
        )

    args.build_dir.mkdir(parents=True, exist_ok=True)
    obj_path = args.build_dir / "roadbuild.obj"
    stats_path = args.build_dir / "roadbuild_navigation_stats.json"
    writer, stats = build_navigation_geometry(carla_map, args)
    writer.write(obj_path)
    stats["server_version"] = client.get_server_version()
    stats["obj_path"] = str(obj_path)
    stats["recast_builder"] = str(args.recast_builder)

    bin_path = None
    if args.build or args.install:
        bin_path = run_recast(args.recast_builder, obj_path)
        stats["bin_path"] = str(bin_path)
        stats["bin_size_bytes"] = bin_path.stat().st_size
    if args.install:
        stats["installed_paths"] = install_navigation(
            bin_path, (args.nav_dir, args.map_nav_dir)
        )

    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
