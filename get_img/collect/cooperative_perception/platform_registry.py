"""Platform and sensor registry for the scene-oriented schema."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class PlatformRegistry:
    def __init__(self) -> None:
        self.platforms: Dict[str, Dict[str, Any]] = {}
        self.sensors: Dict[str, Dict[str, Any]] = {}

    def register_platform(
        self,
        platform_id: str,
        platform_type: str,
        dynamic: bool,
        virtual: bool,
        parent_actor_id: Any = None,
    ) -> Dict[str, Any]:
        record = {
            "platform_id": str(platform_id),
            "platform_type": str(platform_type),
            "dynamic": bool(dynamic),
            "platform_is_virtual": bool(virtual),
            "parent_actor_id": (
                None if parent_actor_id is None else int(parent_actor_id)
            ),
        }
        self.platforms[str(platform_id)] = record
        return record

    def register_sensor(
        self,
        sensor_id: str,
        platform_id: str,
        modality: str,
        required: bool,
        model_input: bool,
    ) -> Dict[str, Any]:
        if platform_id not in self.platforms:
            raise KeyError("Unknown platform_id: %s" % platform_id)
        record = {
            "sensor_id": str(sensor_id),
            "platform_id": str(platform_id),
            "modality": str(modality),
            "required": bool(required),
            "model_input": bool(model_input),
        }
        self.sensors[str(sensor_id)] = record
        return record

    def as_dict(self) -> Dict[str, Iterable[Dict[str, Any]]]:
        return {
            "platforms": list(self.platforms.values()),
            "sensors": list(self.sensors.values()),
        }

