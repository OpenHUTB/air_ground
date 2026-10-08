"""One shared Ground Vehicle generator/controller for all three tasks."""

from __future__ import annotations

import importlib
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .route_library import RoutePlan, build_route_plan
from .simulator_launcher import (
    DEFAULT_CCSP_EXECUTABLE,
    DEFAULT_STANDARD_EXECUTABLE,
)
from .sync_manager import SyncManager


def _vector(vector: Any) -> Dict[str, float]:
    return {"x": float(vector.x), "y": float(vector.y), "z": float(vector.z)}


def _rotation(rotation: Any) -> Dict[str, float]:
    return {
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def _transform(transform: Any) -> Dict[str, Any]:
    return {
        "location": _vector(transform.location),
        "rotation_degree": _rotation(transform.rotation),
    }


class GroundVehicleManager:
    """Dedicated cooperative vehicle driven by a predefined route + BehaviorAgent."""

    def __init__(
        self,
        client: Any,
        world: Any,
        carla_module: Any,
        config: Dict[str, Any],
        task_id: str,
    ) -> None:
        self.client = client
        self.world = world
        self.carla = carla_module
        self.config = dict(config)
        self.task_id = str(task_id)
        self.vehicle = None
        self.agent = None
        self.route_plan: Optional[RoutePlan] = None
        self.sensors: Dict[str, Any] = {}
        self.sync = SyncManager(timeout=float(self.config.get("sensor_timeout", 15.0)))
        self.collision_sensor = None
        self.collision_count = 0
        self.collision_events: List[Dict[str, Any]] = []
        self.offroad_count = 0
        self.travel_distance = 0.0
        self.last_location = None
        self.last_control = None
        self.last_simulation_time = None
        self.current_stationary_duration_s = 0.0
        self.max_stationary_duration_s = 0.0
        self.max_pose_jump_m = 0.0
        self.sensor_frame_requests = 0
        self.valid_sensor_frames = 0
        self.state_history: List[Dict[str, Any]] = []
        self.scene_id = ""
        self.scene_seed = 0
        self.initial_route_waypoint_count = 0

    @property
    def platform_id(self) -> str:
        return str(self.config.get("platform_id", "vehicle_01"))

    @property
    def camera_sensor_id(self) -> str:
        return "%s_camera_front_rgb" % self.platform_id

    def _behavior_agent_class(self) -> Any:
        try:
            return importlib.import_module(
                "agents.navigation.behavior_agent"
            ).BehaviorAgent
        except ModuleNotFoundError as initial_exc:
            # The PyPI/conda CARLA client provides ``import carla`` but does not
            # ship the navigation agents.  OpenHUTB keeps them beside the
            # simulator under PythonAPI/carla, so make that directory importable
            # automatically for the local simulator profiles used by collectors.
            candidates: List[Path] = []
            configured = os.environ.get("CARLA_PYTHONAPI", "")
            candidates.extend(
                Path(value)
                for value in configured.split(os.pathsep)
                if value.strip()
            )
            for variable in ("CARLA_ROOT", "OPENHUTB_ROOT"):
                root = os.environ.get(variable)
                if root:
                    candidates.append(Path(root))
            candidates.extend(
                executable.parent
                for executable in (
                    DEFAULT_STANDARD_EXECUTABLE,
                    DEFAULT_CCSP_EXECUTABLE,
                )
            )

            search_paths: List[Path] = []
            for candidate in candidates:
                for path in (
                    candidate,
                    candidate / "carla",
                    candidate / "PythonAPI" / "carla",
                ):
                    if (path / "agents" / "navigation" / "behavior_agent.py").is_file():
                        resolved = path.resolve()
                        if resolved not in search_paths:
                            search_paths.append(resolved)

            for path in reversed(search_paths):
                value = str(path)
                if value not in sys.path:
                    sys.path.insert(0, value)
            importlib.invalidate_caches()

            try:
                return importlib.import_module(
                    "agents.navigation.behavior_agent"
                ).BehaviorAgent
            except ModuleNotFoundError as exc:
                searched = ", ".join(str(path) for path in search_paths) or "none"
                if exc.name and not exc.name.startswith("agents"):
                    raise RuntimeError(
                        "CARLA BehaviorAgent dependency '%s' is unavailable. "
                        "Install it in the collector Python environment. "
                        "Discovered PythonAPI paths: %s"
                        % (exc.name, searched)
                    ) from exc
                raise RuntimeError(
                    "CARLA BehaviorAgent is unavailable. Set CARLA_PYTHONAPI "
                    "to the directory containing agents (normally "
                    "<CARLA_ROOT>/PythonAPI/carla). Discovered paths: %s"
                    % searched
                ) from initial_exc

    def _occupied_locations(self, excluded_actor_ids: Iterable[int]) -> List[Any]:
        excluded = set(int(value) for value in excluded_actor_ids)
        locations = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            if int(actor.id) in excluded:
                continue
            try:
                locations.append(actor.get_location())
            except RuntimeError:
                continue
        return locations

    def spawn_vehicle(
        self,
        scene_id: str,
        scene_seed: int,
        route_profile: str,
        target_actor: Optional[Any] = None,
        excluded_actor_ids: Iterable[int] = (),
    ) -> Any:
        if self.vehicle is not None:
            raise RuntimeError("Ground Vehicle already exists; call destroy first")
        self.collision_count = 0
        self.collision_events = []
        self.offroad_count = 0
        self.travel_distance = 0.0
        self.last_location = None
        self.last_control = None
        self.last_simulation_time = None
        self.current_stationary_duration_s = 0.0
        self.max_stationary_duration_s = 0.0
        self.max_pose_jump_m = 0.0
        self.sensor_frame_requests = 0
        self.valid_sensor_frames = 0
        self.state_history = []
        self.sync = SyncManager(timeout=float(self.config.get("sensor_timeout", 15.0)))
        self.scene_id = str(scene_id)
        self.scene_seed = int(scene_seed)
        self.route_plan = build_route_plan(
            self.world.get_map(),
            route_profile=route_profile,
            scene_id=self.scene_id,
            seed=self.scene_seed,
            target_actor=target_actor,
            occupied_locations=self._occupied_locations(excluded_actor_ids),
            min_route_distance_m=float(self.config.get("min_route_distance_m", 60.0)),
            max_route_distance_m=float(self.config.get("max_route_distance_m", 260.0)),
            min_spawn_clearance_m=float(self.config.get("min_spawn_clearance_m", 8.0)),
        )
        blueprint_id = str(self.config.get("blueprint_id", "vehicle.tesla.model3"))
        library = self.world.get_blueprint_library()
        blueprint = library.find(blueprint_id)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "cooperative_vehicle")
        if blueprint.has_attribute("color") and self.config.get("color"):
            blueprint.set_attribute("color", str(self.config["color"]))
        self.vehicle = self.world.try_spawn_actor(
            blueprint,
            self.route_plan.start_transform,
        )
        if self.vehicle is None:
            raise RuntimeError(
                "Failed to spawn cooperative_vehicle at route start for %s"
                % self.route_plan.route_id
            )
        self._build_agent()
        self._spawn_sensors()
        # In synchronous mode CARLA may still report the actor at the origin
        # until the first world tick after spawning.  The first sampled state,
        # not that pre-tick placeholder, is the trajectory baseline.
        self.last_location = None
        return self.vehicle

    def _build_agent(self) -> None:
        behavior_class = self._behavior_agent_class()
        behavior = str(self.config.get("behavior", "normal"))
        options = {
            "base_vehicle_threshold": float(
                self.config.get("base_vehicle_threshold_m", 5.0)
            ),
            "ignore_traffic_lights": bool(
                self.config.get("ignore_traffic_lights", False)
            ),
        }
        try:
            self.agent = behavior_class(self.vehicle, behavior=behavior, opt_dict=options)
        except TypeError:
            self.agent = behavior_class(self.vehicle, behavior=behavior)
        self.set_route(self.route_plan)
        nominal_speed = float(self.config.get("nominal_speed_kmh", 25.0))
        jitter = float(self.config.get("speed_jitter_kmh", 2.0))
        speed_rng = random.Random(self.scene_seed + 91_919)
        target_speed = nominal_speed + speed_rng.uniform(-jitter, jitter)
        if hasattr(self.agent, "set_target_speed"):
            self.agent.set_target_speed(max(3.0, target_speed))

    def set_route(self, route_plan: RoutePlan) -> None:
        self.route_plan = route_plan
        start_location = route_plan.start_transform.location
        destination = route_plan.destination
        try:
            self.agent.set_destination(
                destination,
                start_location=start_location,
                clean=True,
            )
        except TypeError:
            try:
                self.agent.set_destination(start_location, destination, clean=True)
            except TypeError:
                self.agent.set_destination(destination)
        local_planner = getattr(self.agent, "_local_planner", None)
        queue = getattr(local_planner, "_waypoints_queue", None)
        buffer_ = getattr(local_planner, "_waypoint_buffer", None)
        self.initial_route_waypoint_count = max(
            1,
            (len(queue) if queue is not None else 0)
            + (len(buffer_) if buffer_ is not None else 0),
        )

    def _camera_transform(self) -> Any:
        mount = dict(self.config.get("camera_mount", {}))
        return self.carla.Transform(
            self.carla.Location(
                x=float(mount.get("x_m", 1.5)),
                y=float(mount.get("y_m", 0.0)),
                z=float(mount.get("z_m", 2.1)),
            ),
            self.carla.Rotation(
                pitch=float(mount.get("pitch_deg", -5.0)),
                yaw=float(mount.get("yaw_deg", 0.0)),
                roll=float(mount.get("roll_deg", 0.0)),
            ),
        )

    def _configure_camera(self, sensor_type: str) -> Any:
        blueprint = self.world.get_blueprint_library().find(sensor_type)
        attributes = {
            "image_size_x": int(self.config.get("image_width", 1920)),
            "image_size_y": int(self.config.get("image_height", 1080)),
            "fov": float(self.config.get("camera_fov_deg", 90.0)),
            "sensor_tick": float(self.config.get("sensor_tick", 0.0)),
        }
        for name, value in attributes.items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(name, str(value))
        if sensor_type == "sensor.camera.rgb":
            if blueprint.has_attribute("enable_postprocess_effects"):
                blueprint.set_attribute(
                    "enable_postprocess_effects",
                    "true" if bool(self.config.get("enable_rgb_postprocess", True)) else "false",
                )
            for name in (
                "motion_blur_intensity",
                "motion_blur_max_distortion",
                "motion_blur_min_object_screen_size",
            ):
                if blueprint.has_attribute(name):
                    blueprint.set_attribute(name, "0.0")
        return blueprint

    def _spawn_sensors(self) -> None:
        modalities = list(self.config.get("modalities", ["rgb", "depth"]))
        camera_types = {
            "rgb": "sensor.camera.rgb",
            "depth": "sensor.camera.depth",
            "semantic": "sensor.camera.semantic_segmentation",
            "instance": "sensor.camera.instance_segmentation",
        }
        camera_transform = self._camera_transform()
        for modality in modalities:
            if modality == "normal":
                continue
            if modality == "lidar":
                sensor = self._spawn_lidar(camera_transform)
            else:
                sensor_type = camera_types.get(modality)
                if sensor_type is None:
                    raise ValueError("Unsupported Ground sensor modality: %s" % modality)
                sensor = self.world.spawn_actor(
                    self._configure_camera(sensor_type),
                    camera_transform,
                    attach_to=self.vehicle,
                )
            sensor_id = "%s_%s" % (self.platform_id, modality)
            self.sensors[modality] = sensor
            self.sync.register_sensor(sensor_id, sensor, required=True)
        collision_blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_blueprint,
            self.carla.Transform(),
            attach_to=self.vehicle,
        )
        self.collision_sensor.listen(self._on_collision)

    def _spawn_lidar(self, transform: Any) -> Any:
        blueprint = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar = dict(self.config.get("lidar", {}))
        attributes = {
            "channels": int(lidar.get("channels", 64)),
            "range": float(lidar.get("range_m", 120.0)),
            "points_per_second": int(lidar.get("points_per_second", 300000)),
            "rotation_frequency": float(lidar.get("rotation_frequency_hz", 20.0)),
            "upper_fov": float(lidar.get("upper_fov_deg", 10.0)),
            "lower_fov": float(lidar.get("lower_fov_deg", -30.0)),
            "sensor_tick": float(self.config.get("sensor_tick", 0.0)),
        }
        for name, value in attributes.items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(name, str(value))
        return self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.vehicle,
        )

    def _on_collision(self, event: Any) -> None:
        self.collision_count += 1
        impulse = getattr(event, "normal_impulse", None)
        impulse_magnitude = 0.0
        if impulse is not None:
            impulse_magnitude = math.sqrt(
                float(impulse.x) ** 2
                + float(impulse.y) ** 2
                + float(impulse.z) ** 2
            )
        other_actor = getattr(event, "other_actor", None)
        self.collision_events.append(
            {
                "world_frame_id": int(getattr(event, "frame", -1)),
                "other_actor_id": (
                    None if other_actor is None else int(other_actor.id)
                ),
                "other_actor_type": (
                    None if other_actor is None else str(other_actor.type_id)
                ),
                "normal_impulse_magnitude": float(impulse_magnitude),
            }
        )

    def run_step(self) -> Any:
        if self.agent is None or self.vehicle is None:
            raise RuntimeError("Ground Vehicle is not prepared")
        control = self.agent.run_step()
        self.vehicle.apply_control(control)
        self.last_control = control
        return control

    def collect_frame(self, world_frame_id: int) -> Dict[str, Any]:
        self.sensor_frame_requests += 1
        data = self.sync.collect(int(world_frame_id))
        self.valid_sensor_frames += 1
        if "normal" in self.config.get("modalities", []) and "depth" in self.sensors:
            data["%s_normal" % self.platform_id] = data[
                "%s_depth" % self.platform_id
            ]
        return data

    def drain(self) -> None:
        self.sync.drain()

    def _route_completion_ratio(self) -> float:
        if self.agent is None:
            return 0.0
        local_planner = getattr(self.agent, "_local_planner", None)
        queue = getattr(local_planner, "_waypoints_queue", None)
        buffer_ = getattr(local_planner, "_waypoint_buffer", None)
        if queue is None:
            return 0.0
        remaining = len(queue) + (len(buffer_) if buffer_ is not None else 0)
        initial = max(1, int(self.initial_route_waypoint_count))
        return max(0.0, min(1.0, 1.0 - float(remaining) / initial))

    def get_state(
        self,
        world_frame_id: int,
        timestamp: float,
        simulation_time: float,
        sample_index: int,
    ) -> Dict[str, Any]:
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        location = transform.location
        if self.last_location is not None:
            step_distance = math.sqrt(
                (float(location.x) - self.last_location[0]) ** 2
                + (float(location.y) - self.last_location[1]) ** 2
                + (float(location.z) - self.last_location[2]) ** 2
            )
            self.travel_distance += step_distance
            self.max_pose_jump_m = max(self.max_pose_jump_m, step_distance)
        self.last_location = (
            float(location.x),
            float(location.y),
            float(location.z),
        )
        speed = math.sqrt(
            float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
        )
        if self.last_simulation_time is not None:
            dt = max(0.0, float(simulation_time) - float(self.last_simulation_time))
            if speed < float(self.config.get("moving_speed_mps", 0.2)):
                self.current_stationary_duration_s += dt
            else:
                self.current_stationary_duration_s = 0.0
            self.max_stationary_duration_s = max(
                self.max_stationary_duration_s,
                self.current_stationary_duration_s,
            )
        self.last_simulation_time = float(simulation_time)
        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True)
        if waypoint is None:
            self.offroad_count += 1
        traffic_light_state = None
        try:
            traffic_light_state = str(self.vehicle.get_traffic_light_state())
        except RuntimeError:
            pass
        control = self.last_control or self.vehicle.get_control()
        state = {
            "world_frame_id": int(world_frame_id),
            "timestamp": float(timestamp),
            "simulation_time": float(simulation_time),
            "sample_index": int(sample_index),
            "source_tick": int(world_frame_id),
            "platform_id": self.platform_id,
            "platform_pose": _transform(transform),
            "sensor_pose": {
                "%s_%s" % (self.platform_id, modality): _transform(
                    sensor.get_transform()
                )
                for modality, sensor in sorted(self.sensors.items())
            },
            "velocity_mps": _vector(velocity),
            "speed_mps": float(speed),
            "acceleration_mps2": _vector(acceleration),
            "heading_deg": float(transform.rotation.yaw),
            "control": {
                "throttle": float(control.throttle),
                "steer": float(control.steer),
                "brake": float(control.brake),
                "hand_brake": bool(control.hand_brake),
                "reverse": bool(control.reverse),
            },
            "route_id": self.route_plan.route_id,
            "route_profile": self.route_plan.route_profile,
            "route_template": self.route_plan.route_template,
            "route_lineage": self.route_plan.lineage,
            "waypoint_id": None if waypoint is None else int(waypoint.id),
            "destination": _vector(self.route_plan.destination),
            "traffic_light_state": traffic_light_state,
            "route_completion_ratio": self._route_completion_ratio(),
            "travel_distance_m": float(self.travel_distance),
            "collision_count": int(self.collision_count),
            "collision_events": list(self.collision_events),
            "offroad_count": int(self.offroad_count),
            "stationary_duration_s": float(self.current_stationary_duration_s),
            "max_pose_jump_m": float(self.max_pose_jump_m),
            "threshold_status": "pilot_provisional",
        }
        if "normal" in self.config.get("modalities", []) and "%s_depth" % self.platform_id in state["sensor_pose"]:
            state["sensor_pose"]["%s_normal" % self.platform_id] = dict(
                state["sensor_pose"]["%s_depth" % self.platform_id]
            )
        self.state_history.append(state)
        return state

    def check_qa(self) -> Dict[str, Any]:
        speeds = [float(item["speed_mps"]) for item in self.state_history]
        moving_threshold = float(self.config.get("moving_speed_mps", 0.2))
        return {
            "route_completion_ratio": self._route_completion_ratio(),
            "travel_distance_m": float(self.travel_distance),
            "motion_ratio": float(
                sum(speed >= moving_threshold for speed in speeds) / max(1, len(speeds))
            ),
            "max_stationary_duration_s": float(self.max_stationary_duration_s),
            "collision_count": int(self.collision_count),
            "collision_events": list(self.collision_events),
            "offroad_count": int(self.offroad_count),
            "valid_sensor_frame_ratio": float(
                self.valid_sensor_frames / max(1, self.sensor_frame_requests)
            ),
            "max_pose_jump_m": float(self.max_pose_jump_m),
            "route_failure": bool(
                not self.state_history
                or self.route_plan is None
                or self.agent is None
                or self.initial_route_waypoint_count <= 0
            ),
            "threshold_status": "pilot_provisional",
        }

    def scene_metadata(self) -> Dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "platform_type": "ground_vehicle",
            "platform_is_virtual": False,
            "role_name": "cooperative_vehicle",
            "actor_id": int(self.vehicle.id),
            "blueprint_id": str(self.config.get("blueprint_id", "vehicle.tesla.model3")),
            "spawn_point": _transform(self.route_plan.start_transform),
            "route_id": self.route_plan.route_id,
            "route_profile": self.route_plan.route_profile,
            "route_template": self.route_plan.route_template,
            "route_seed": int(self.route_plan.seed),
            "route_lineage": self.route_plan.lineage,
            "route_waypoint_sequence": self.route_plan.waypoint_sequence,
            "camera_mount": dict(self.config.get("camera_mount", {})),
            "pilot_provisional_fields": [
                "blueprint_id",
                "camera_mount",
                "camera_fov_deg",
                "nominal_speed_kmh",
                "route_thresholds",
            ],
        }

    def destroy(self) -> None:
        actors = list(self.sensors.values())
        if self.collision_sensor is not None:
            actors.append(self.collision_sensor)
        for actor in actors:
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except RuntimeError:
                pass
        for actor in actors:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except RuntimeError:
                pass
        self.vehicle = None
        self.agent = None
        self.sensors = {}
        self.collision_sensor = None
        self.last_location = None
        self.last_simulation_time = None
