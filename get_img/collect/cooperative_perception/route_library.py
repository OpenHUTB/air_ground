"""Deterministic route-template selection shared by all three tasks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROUTE_PROFILES = {
    "coverage": (
        "straight",
        "intersection_crossing",
        "left_turn",
        "right_turn",
        "merge_diverge",
    ),
    "target_interaction": (
        "same_direction",
        "opposite_direction",
        "intersection_crossing",
        "left_turn",
        "right_turn",
        "merge_diverge",
    ),
    "multi_object_traversal": (
        "straight",
        "intersection_crossing",
        "left_turn",
        "right_turn",
        "merge_diverge",
    ),
}


@dataclass(frozen=True)
class RoutePlan:
    route_id: str
    route_profile: str
    route_template: str
    start_transform: Any
    destination: Any
    seed: int
    lineage: str
    waypoint_sequence: List[Dict[str, float]]


def _yaw_delta(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _distance(first: Any, second: Any) -> float:
    return math.sqrt(
        (float(first.x) - float(second.x)) ** 2
        + (float(first.y) - float(second.y)) ** 2
        + (float(first.z) - float(second.z)) ** 2
    )


def _relation_score(
    template: str,
    start: Any,
    destination: Any,
    target_actor: Optional[Any],
) -> float:
    start_location = start.location
    destination_location = destination.location
    distance = _distance(start_location, destination_location)
    yaw_change = _yaw_delta(start.rotation.yaw, destination.rotation.yaw)
    score = abs(distance - 120.0) * 0.05
    if template in {"straight", "same_direction"}:
        score += yaw_change * 0.25
    elif template == "opposite_direction":
        score += abs(180.0 - yaw_change) * 0.25
    elif template in {"left_turn", "right_turn", "intersection_crossing"}:
        score += abs(90.0 - yaw_change) * 0.20
    elif template == "merge_diverge":
        score += abs(45.0 - yaw_change) * 0.20
    if target_actor is not None:
        try:
            target_transform = target_actor.get_transform()
            initial_distance = _distance(start_location, target_transform.location)
            target_yaw = float(target_transform.rotation.yaw)
            relation_yaw = _yaw_delta(start.rotation.yaw, target_yaw)
            target_bearing = math.degrees(
                math.atan2(
                    float(target_transform.location.y - start_location.y),
                    float(target_transform.location.x - start_location.x),
                )
            )
            # The ground RGB camera faces the vehicle's forward direction.  A
            # target-interaction route therefore needs the shared target in
            # front of the spawn point, not merely close to it.
            forward_alignment = _yaw_delta(start.rotation.yaw, target_bearing)
            score += abs(initial_distance - 35.0) * 0.08
            score += forward_alignment * 0.60
            if template == "same_direction":
                score += relation_yaw * 0.35
            elif template == "opposite_direction":
                score += abs(180.0 - relation_yaw) * 0.35
            elif template == "intersection_crossing":
                score += abs(90.0 - relation_yaw) * 0.30
        except RuntimeError:
            score += 1000.0
    return score


def _waypoint_record(transform: Any) -> Dict[str, float]:
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "yaw_deg": float(transform.rotation.yaw),
    }


def build_route_plan(
    carla_map: Any,
    route_profile: str,
    scene_id: str,
    seed: int,
    target_actor: Optional[Any] = None,
    occupied_locations: Optional[Sequence[Any]] = None,
    min_route_distance_m: float = 60.0,
    max_route_distance_m: float = 260.0,
    min_spawn_clearance_m: float = 8.0,
) -> RoutePlan:
    """Select a reproducible legal start/destination pair for BehaviorAgent."""
    if route_profile not in ROUTE_PROFILES:
        raise ValueError("Unknown Ground Vehicle route_profile: %s" % route_profile)
    spawn_points = list(carla_map.get_spawn_points())
    if len(spawn_points) < 2:
        raise RuntimeError("The current map does not expose enough vehicle spawn points")
    rng = random.Random(int(seed))
    if route_profile == "target_interaction" and target_actor is not None:
        # A same-direction route keeps the jointly observed traffic actor in
        # the forward Ground camera across the sequence.  Crossing/opposite
        # routes only intersect briefly and lose the shared target afterward.
        template = "same_direction"
    else:
        template = ROUTE_PROFILES[route_profile][
            int(seed) % len(ROUTE_PROFILES[route_profile])
        ]
    occupied = list(occupied_locations or [])
    candidates: List[Tuple[float, Any, Any]] = []
    shuffled = list(spawn_points)
    rng.shuffle(shuffled)
    for start in shuffled:
        if any(
            _distance(start.location, location) < float(min_spawn_clearance_m)
            for location in occupied
        ):
            continue
        try:
            start_waypoint = carla_map.get_waypoint(
                start.location,
                project_to_road=True,
            )
        except RuntimeError:
            start_waypoint = None
        if start_waypoint is None:
            continue
        for destination in shuffled:
            distance = _distance(start.location, destination.location)
            if not float(min_route_distance_m) <= distance <= float(max_route_distance_m):
                continue
            score = _relation_score(template, start, destination, target_actor)
            score += rng.random() * 0.01
            candidates.append((score, start, destination))
    if not candidates:
        raise RuntimeError(
            "No legal route candidate for profile=%s template=%s"
            % (route_profile, template)
        )
    _, start, destination = min(candidates, key=lambda item: item[0])
    route_id = "%s_%s_%08d" % (route_profile, template, int(seed))
    lineage = "%s:%s:%d" % (scene_id, route_id, int(seed))
    return RoutePlan(
        route_id=route_id,
        route_profile=route_profile,
        route_template=template,
        start_transform=start,
        destination=destination.location,
        seed=int(seed),
        lineage=lineage,
        waypoint_sequence=[
            _waypoint_record(start),
            _waypoint_record(destination),
        ],
    )
