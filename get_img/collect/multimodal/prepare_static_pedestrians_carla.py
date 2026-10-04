#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load a CARLA map and keep fallback pedestrians alive for collection.

Some OpenHUTB RoadRunner levels contain vehicle spawn points but no pedestrian
navigation mesh. This connection-side bridge places pedestrian actors beside
road spawn points, then keeps the client alive while the unchanged multimodal
collector connects to the same world.
"""

import argparse
import json
import math
import random
import sys
import time

import carla


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--map")
    parser.add_argument("--load-map", action="store_true")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--vehicle-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--lateral-offset", type=float, default=3.5)
    parser.add_argument(
        "--pedestrian-motion",
        choices=("static", "shuttle"),
        default="static",
    )
    parser.add_argument("--walker-speed-min", type=float, default=0.9)
    parser.add_argument("--walker-speed-max", type=float, default=1.5)
    parser.add_argument("--shuttle-distance", type=float, default=12.0)
    return parser.parse_args()


def spawn_fallback_pedestrians(
    world,
    count,
    seed,
    lateral_offset,
    pedestrian_motion,
    speed_min,
    speed_max,
):
    rng = random.Random(seed)
    blueprints = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    spawn_points = list(world.get_map().get_spawn_points())
    if not blueprints:
        raise RuntimeError("No walker.pedestrian.* blueprints are available")
    if not spawn_points:
        raise RuntimeError("The current map has no vehicle spawn points")

    rng.shuffle(spawn_points)
    actors = []
    motion_states = []
    candidate_index = 0
    max_attempts = max(count * 12, len(spawn_points) * 8)

    while len(actors) < count and candidate_index < max_attempts:
        base = spawn_points[candidate_index % len(spawn_points)]
        lane_pass = candidate_index // len(spawn_points)
        side = -1.0 if (candidate_index + lane_pass) % 2 else 1.0
        lateral = side * (lateral_offset + 0.7 * (lane_pass % 3))
        longitudinal = (-2.0, 0.0, 2.0)[lane_pass % 3]
        yaw_rad = math.radians(float(base.rotation.yaw))

        location = carla.Location(
            x=base.location.x - math.sin(yaw_rad) * lateral
            + math.cos(yaw_rad) * longitudinal,
            y=base.location.y + math.cos(yaw_rad) * lateral
            + math.sin(yaw_rad) * longitudinal,
            z=base.location.z + 0.6,
        )
        rotation = carla.Rotation(
            pitch=0.0,
            yaw=float(base.rotation.yaw) + rng.uniform(-35.0, 35.0),
            roll=0.0,
        )
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute(
                "role_name",
                (
                    "collection_scripted_pedestrian"
                    if pedestrian_motion == "shuttle"
                    else "collection_static_pedestrian"
                ),
            )

        actor = world.try_spawn_actor(
            blueprint,
            carla.Transform(location, rotation),
        )
        if actor is not None:
            actors.append(actor)
            motion_states.append({
                "actor": actor,
                "origin_x": float(location.x),
                "origin_y": float(location.y),
                "direction_x": math.cos(yaw_rad),
                "direction_y": math.sin(yaw_rad),
                "sign": 1.0,
                "speed": rng.uniform(speed_min, speed_max),
            })
        candidate_index += 1

    return actors, motion_states, spawn_points, candidate_index


def apply_shuttle_control(state):
    control = carla.WalkerControl()
    control.direction = carla.Vector3D(
        x=state["direction_x"] * state["sign"],
        y=state["direction_y"] * state["sign"],
        z=0.0,
    )
    control.speed = state["speed"]
    control.jump = False
    state["actor"].apply_control(control)


def update_shuttle_pedestrians(states, distance):
    for state in states:
        actor = state["actor"]
        if actor is None or not actor.is_alive:
            continue
        location = actor.get_location()
        along = (
            (float(location.x) - state["origin_x"]) * state["direction_x"]
            + (float(location.y) - state["origin_y"]) * state["direction_y"]
        )
        if along >= distance:
            state["sign"] = -1.0
        elif along <= -distance:
            state["sign"] = 1.0
        apply_shuttle_control(state)


def is_supported_four_wheel_blueprint(blueprint):
    type_id = blueprint.id.lower()
    if any(
        token in type_id
        for token in (
            "bike", "bicycle", "motorcycle", "vespa", "scooter",
            "harley", "kawasaki", "yamaha",
        )
    ):
        return False
    if blueprint.has_attribute("number_of_wheels"):
        try:
            return int(str(blueprint.get_attribute("number_of_wheels"))) >= 4
        except (TypeError, ValueError):
            pass
    return True


def spawn_static_vehicles(world, spawn_points, count, seed):
    if count <= 0:
        return [], 0
    rng = random.Random(seed)
    blueprints = [
        blueprint
        for blueprint in world.get_blueprint_library().filter("vehicle.*")
        if is_supported_four_wheel_blueprint(blueprint)
    ]
    if not blueprints:
        raise RuntimeError("No supported four-wheel vehicle blueprints are available")

    actors = []
    attempts = 0
    max_attempts = max(count * 6, len(spawn_points) * 2)
    while len(actors) < count and attempts < max_attempts:
        base = spawn_points[attempts % len(spawn_points)]
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "collection_static_vehicle")
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", rng.choice(colors))

        location = carla.Location(
            x=base.location.x,
            y=base.location.y,
            z=base.location.z + 0.25,
        )
        actor = world.try_spawn_actor(
            blueprint,
            carla.Transform(location, base.rotation),
        )
        if actor is not None:
            actors.append(actor)
        attempts += 1
    return actors, attempts


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    if args.walker_speed_min <= 0.0 or args.walker_speed_max <= 0.0:
        raise ValueError("walker speed must be greater than zero")
    if args.walker_speed_min > args.walker_speed_max:
        raise ValueError("--walker-speed-min cannot exceed --walker-speed-max")
    if args.shuttle_distance <= 0.0:
        raise ValueError("--shuttle-distance must be greater than zero")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    if args.load_map:
        if not args.map:
            raise ValueError("--map is required together with --load-map")
        world = client.load_world(args.map)
        time.sleep(5.0)
    else:
        world = client.get_world()

    pedestrians, motion_states, spawn_points, pedestrian_attempts = (
        spawn_fallback_pedestrians(
            world,
            args.count,
            args.seed,
            args.lateral_offset,
            args.pedestrian_motion,
            args.walker_speed_min,
            args.walker_speed_max,
        )
    )
    vehicles, vehicle_attempts = spawn_static_vehicles(
        world,
        spawn_points,
        args.vehicle_count,
        args.seed + 1,
    )
    actors = pedestrians + vehicles
    payload = {
        "map": world.get_map().name,
        "requested_pedestrians": args.count,
        "spawned_pedestrians": len(pedestrians),
        "requested_vehicles": args.vehicle_count,
        "spawned_vehicles": len(vehicles),
        "vehicle_spawn_points": len(spawn_points),
        "pedestrian_attempts": pedestrian_attempts,
        "vehicle_attempts": vehicle_attempts,
        "pedestrian_motion": args.pedestrian_motion,
    }
    if args.pedestrian_motion == "shuttle":
        update_shuttle_pedestrians(motion_states, args.shuttle_distance)
    print("READY " + json.dumps(payload, ensure_ascii=True), flush=True)
    if not pedestrians:
        return 2

    # Actor references and the CARLA client are intentionally kept alive until
    # the parent orchestration process terminates this helper.
    while True:
        if args.pedestrian_motion == "shuttle":
            update_shuttle_pedestrians(motion_states, args.shuttle_distance)
            time.sleep(0.25)
        else:
            time.sleep(10.0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print("ERROR " + repr(exc), file=sys.stderr, flush=True)
        sys.exit(1)
