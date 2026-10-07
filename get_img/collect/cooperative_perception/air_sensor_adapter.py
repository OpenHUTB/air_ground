"""Register existing airborne sensor rigs without replacing their collectors."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class AirSensorAdapter:
    def __init__(
        self,
        platform_id: str,
        sensors: Dict[str, Any],
        virtual: bool,
    ) -> None:
        self.platform_id = str(platform_id)
        self.sensors = dict(sensors)
        self.virtual = bool(virtual)
        self._last_location = None
        self._last_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._last_simulation_time = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "platform_type": (
                "air_camera_platform" if self.virtual else "uav"
            ),
            "platform_is_virtual": self.virtual,
            "sensors": [
                {
                    "sensor_id": "%s_%s" % (self.platform_id, modality),
                    "modality": modality,
                }
                for modality in sorted(self.sensors)
            ],
        }

    def transforms(self) -> List[Dict[str, Any]]:
        records = []
        for modality, sensor in sorted(self.sensors.items()):
            transform = sensor.get_transform()
            records.append(
                {
                    "sensor_id": "%s_%s" % (self.platform_id, modality),
                    "location": {
                        "x": float(transform.location.x),
                        "y": float(transform.location.y),
                        "z": float(transform.location.z),
                    },
                    "rotation_degree": {
                        "pitch": float(transform.rotation.pitch),
                        "yaw": float(transform.rotation.yaw),
                        "roll": float(transform.rotation.roll),
                    },
                }
            )
        return records

    def frame_state(
        self,
        world_frame_id: int,
        timestamp: float,
        simulation_time: float,
        sample_index: int,
    ) -> Dict[str, Any]:
        transforms = self.transforms()
        if not transforms:
            raise RuntimeError("Air platform has no bound sensors: %s" % self.platform_id)
        primary = next(
            (item for item in transforms if item["sensor_id"].endswith("_rgb")),
            transforms[0],
        )
        location = primary["location"]
        velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        acceleration = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self._last_location is not None and self._last_simulation_time is not None:
            dt = float(simulation_time) - float(self._last_simulation_time)
            if dt > 1.0e-9:
                velocity = {
                    axis: (float(location[axis]) - float(self._last_location[axis])) / dt
                    for axis in ("x", "y", "z")
                }
                acceleration = {
                    axis: (float(velocity[axis]) - float(self._last_velocity[axis])) / dt
                    for axis in ("x", "y", "z")
                }
        self._last_location = dict(location)
        self._last_velocity = dict(velocity)
        self._last_simulation_time = float(simulation_time)
        sensor_pose = {
            item["sensor_id"]: {
                "location": item["location"],
                "rotation_degree": item["rotation_degree"],
            }
            for item in transforms
        }
        return {
            "world_frame_id": int(world_frame_id),
            "timestamp": float(timestamp),
            "simulation_time": float(simulation_time),
            "sample_index": int(sample_index),
            "source_tick": int(world_frame_id),
            "platform_id": self.platform_id,
            "platform_type": "air_camera_platform" if self.virtual else "uav",
            "platform_is_virtual": self.virtual,
            "platform_pose": {
                "location": primary["location"],
                "rotation_degree": primary["rotation_degree"],
            },
            "sensor_pose": sensor_pose,
            "velocity_mps": velocity,
            "acceleration_mps2": acceleration,
            "heading_deg": float(primary["rotation_degree"]["yaw"]),
            "control": {"mode": "kinematic_sensor_platform"},
        }
