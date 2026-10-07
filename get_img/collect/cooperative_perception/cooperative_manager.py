"""Lifecycle façade inserted into the three existing collectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .association_manager import AssociationManager
from .air_sensor_adapter import AirSensorAdapter
from .calibration_manager import CalibrationManager
from .communication_manager import CommunicationManager
from .dataset_writer import DatasetWriter
from .ground_vehicle_manager import GroundVehicleManager
from .object_registry import ObjectRegistry
from .platform_registry import PlatformRegistry
from .quality_manager import QualityManager
from .schema import SCHEMA_VERSION, TASK_IDS


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class CooperativeManager:
    def __init__(
        self,
        client: Any,
        world: Any,
        carla_module: Any,
        dataset_root: Path,
        task_id: str,
        config: Dict[str, Any],
    ) -> None:
        if task_id not in TASK_IDS:
            raise ValueError("Unknown cooperative task_id: %s" % task_id)
        self.client = client
        self.world = world
        self.carla = carla_module
        self.dataset_root = Path(dataset_root)
        self.task_id = str(task_id)
        base_config_path = Path(__file__).with_name("configs") / "cooperative_base.json"
        base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
        self.config = _deep_merge(base_config, dict(config))
        self.platform_registry = PlatformRegistry()
        self.calibration = CalibrationManager(
            closure_atol=float(self.config.get("transform_closure_atol", 1e-5))
        )
        self.association = AssociationManager()
        self.quality = QualityManager(dict(self.config.get("quality", {})))
        self.writer = DatasetWriter(self.dataset_root, TASK_IDS[self.task_id])
        self.ground_vehicle = GroundVehicleManager(
            client,
            world,
            carla_module,
            dict(self.config.get("ground_vehicle", {})),
            task_id,
        )
        self.object_registry: Optional[ObjectRegistry] = None
        self.communication: Optional[CommunicationManager] = None
        self.scene_id = ""
        self.scene_seed = 0
        self.air_adapters: Dict[str, AirSensorAdapter] = {}
        self.sample_index = 0

    def prepare_scene(
        self,
        scene_id: str,
        scene_seed: int,
        route_profile: str,
        actors: Iterable[Any],
        target_actor: Optional[Any] = None,
        overwrite_staging: bool = True,
    ) -> Path:
        self.close_scene(abort_reason="replaced_by_next_scene")
        self.scene_id = str(scene_id)
        self.scene_seed = int(scene_seed)
        self.sample_index = 0
        self.air_adapters = {}
        self.object_registry = ObjectRegistry(
            self.scene_id,
            dataset_namespace=str(
                self.config.get("dataset_namespace", "air-ground-coop-suite-v1")
            ),
        )
        self.object_registry.register_many(list(actors))
        self.ground_vehicle.spawn_vehicle(
            scene_id=self.scene_id,
            scene_seed=self.scene_seed,
            route_profile=route_profile,
            target_actor=target_actor,
            # Existing traffic and the target are real occupancy constraints.
            # They must remain in the clearance check for the dedicated
            # cooperative vehicle's spawn point.
            excluded_actor_ids=(),
        )
        self.platform_registry.register_platform(
            self.ground_vehicle.platform_id,
            "ground_vehicle",
            dynamic=True,
            virtual=False,
            parent_actor_id=int(self.ground_vehicle.vehicle.id),
        )
        for modality in self.ground_vehicle.config.get("modalities", []):
            self.platform_registry.register_sensor(
                "%s_%s" % (self.ground_vehicle.platform_id, modality),
                self.ground_vehicle.platform_id,
                str(modality),
                required=True,
                model_input=self.task_id == "task1" or modality == "rgb",
            )
        self.object_registry.register(self.ground_vehicle.vehicle)
        self.communication = CommunicationManager(
            self.scene_seed,
            dict(self.config.get("communication", {})),
        )
        path = self.writer.begin(self.scene_id, overwrite=overwrite_staging)
        self.writer.write_json(
            "scene.json",
            {
                "schema_version": SCHEMA_VERSION,
                "scene_id": self.scene_id,
                "task_id": self.task_id,
                "scene_seed": self.scene_seed,
                "seed_lineage": "%s:%d" % (self.scene_id, self.scene_seed),
                "threshold_status": "pilot_provisional",
                "topology": dict(self.config.get("topology", {})),
                "frame_contract": {
                    "barrier_key": "world_frame_id",
                    "required_time_fields": [
                        "world_frame_id",
                        "timestamp",
                        "simulation_time",
                        "sample_index",
                        "source_tick",
                    ],
                    "missing_or_wrong_sensor_policy": "reject_entire_frame",
                    "previous_frame_fill": "forbidden",
                },
                "ground_vehicle": self.ground_vehicle.scene_metadata(),
            },
        )
        self.writer.write_json(
            "annotations/objects.json",
            self.object_registry.records(),
        )
        return path

    def register_air_platform(
        self,
        platform_id: str,
        virtual: bool,
        platform_type: Optional[str] = None,
        sensors: Optional[Dict[str, Any]] = None,
        required_modalities: Optional[Iterable[str]] = None,
    ) -> None:
        self.platform_registry.register_platform(
            platform_id,
            platform_type or ("air_camera_platform" if virtual else "uav"),
            dynamic=True,
            virtual=virtual,
        )
        if sensors is not None:
            self.air_adapters[str(platform_id)] = AirSensorAdapter(
                str(platform_id), sensors, virtual
            )
            required = set(
                str(value)
                for value in (
                    required_modalities
                    if required_modalities is not None
                    else sensors.keys()
                )
            )
            for modality in sorted(sensors):
                self.platform_registry.register_sensor(
                    "%s_%s" % (platform_id, modality),
                    str(platform_id),
                    str(modality),
                    required=str(modality) in required,
                    model_input=self.task_id == "task1" or modality == "rgb",
                )

    def before_world_tick(self) -> None:
        self.ground_vehicle.run_step()

    def collect_ground(self, world_frame_id: int) -> Dict[str, Any]:
        return self.ground_vehicle.collect_frame(world_frame_id)

    def record_frame(
        self,
        world_frame_id: int,
        timestamp: float,
        simulation_time: float,
        observations_by_sensor: Dict[str, Iterable[Dict[str, Any]]],
        sensor_data_by_id: Dict[str, Any],
        ground_sensor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        sync_report = self.quality.frame_sync_report(
            world_frame_id,
            simulation_time,
            sensor_data_by_id,
            [
                sensor_id
                for sensor_id, record in self.platform_registry.sensors.items()
                if bool(record["required"])
            ],
        )
        if not bool(sync_report["sync_ok"]):
            raise RuntimeError(
                "frame barrier rejected world_frame_id=%s: %s"
                % (world_frame_id, sync_report)
            )
        sample_index = int(self.sample_index)
        state = self.ground_vehicle.get_state(
            world_frame_id,
            timestamp,
            simulation_time,
            sample_index,
        )
        temporal_observations: Dict[str, Any] = {}
        for sensor_id, observations in observations_by_sensor.items():
            sensor_timestamp = float(
                getattr(sensor_data_by_id.get(sensor_id), "timestamp", timestamp)
            )
            temporal_observations[sensor_id] = []
            for observation in observations:
                record = dict(observation)
                record.update(
                    {
                        "world_frame_id": int(world_frame_id),
                        "timestamp": sensor_timestamp,
                        "simulation_time": float(simulation_time),
                        "sample_index": sample_index,
                        "source_tick": int(world_frame_id),
                    }
                )
                temporal_observations[sensor_id].append(record)
                self.writer.append_jsonl(
                    "annotations/by_sensor/%s.jsonl" % sensor_id,
                    record,
                )
        associations = self.association.build(temporal_observations)
        self.writer.append_jsonl("frames.jsonl", {
            "world_frame_id": int(world_frame_id),
            "timestamp": float(timestamp),
            "simulation_time": float(simulation_time),
            "sample_index": sample_index,
            "source_tick": int(world_frame_id),
            "sensor_timestamps": sync_report["sensor_timestamps"],
            "sync": sync_report,
        })
        self.writer.append_jsonl("platforms/ground/vehicle_01/states.jsonl", state)
        for platform_id, adapter in sorted(self.air_adapters.items()):
            air_state = adapter.frame_state(
                world_frame_id,
                timestamp,
                simulation_time,
                sample_index,
            )
            self.writer.append_jsonl(
                "platforms/air/%s/states.jsonl" % platform_id,
                air_state,
            )
        ground_vehicle_transform = self.ground_vehicle.vehicle.get_transform()
        calibrations: Dict[str, Any] = {}
        ground_sensor_transforms: Dict[str, Any] = {}
        for modality, sensor in sorted(self.ground_vehicle.sensors.items()):
            sensor_id = "%s_%s" % (self.ground_vehicle.platform_id, modality)
            sensor_transform = sensor.get_transform()
            ground_sensor_transforms[sensor_id] = sensor_transform
            calibrations[sensor_id] = self.calibration.sensor_calibration(
                ground_vehicle_transform,
                sensor_transform,
            )
        if "normal" in self.ground_vehicle.config.get("modalities", []) and "depth" in self.ground_vehicle.sensors:
            calibrations["%s_normal" % self.ground_vehicle.platform_id] = dict(
                calibrations["%s_depth" % self.ground_vehicle.platform_id]
            )
            ground_sensor_transforms["%s_normal" % self.ground_vehicle.platform_id] = (
                self.ground_vehicle.sensors["depth"].get_transform()
            )
        air_primary_transforms: Dict[str, Any] = {}
        for platform_id, adapter in sorted(self.air_adapters.items()):
            primary_sensor = adapter.sensors.get("rgb")
            if primary_sensor is None:
                primary_sensor = next(iter(adapter.sensors.values()))
            platform_transform = primary_sensor.get_transform()
            air_primary_transforms[platform_id] = platform_transform
            for modality, sensor in sorted(adapter.sensors.items()):
                sensor_id = "%s_%s" % (platform_id, modality)
                calibrations[sensor_id] = self.calibration.sensor_calibration(
                    platform_transform,
                    sensor.get_transform(),
                )
        pairwise = {}
        ground_primary = self.ground_vehicle.sensors.get("rgb")
        if ground_primary is not None:
            for platform_id, air_transform in sorted(air_primary_transforms.items()):
                record = self.calibration.transform_between(
                    ground_primary.get_transform(),
                    air_transform,
                )
                pairwise["vehicle_01_rgb_from_%s_rgb" % platform_id] = {
                    "T_target_sensor_from_source_sensor": record[
                        "T_target_from_source"
                    ],
                    "T_source_sensor_from_target_sensor": record[
                        "T_source_from_target"
                    ],
                    "closure_max_abs_error": record[
                        "closure_max_abs_error"
                    ],
                    "closure_ok": record["closure_ok"],
                }
        self.writer.write_json(
            "calibration/%06d.json" % sample_index,
            {
                "world_frame_id": int(world_frame_id),
                "timestamp": float(timestamp),
                "simulation_time": float(simulation_time),
                "sample_index": sample_index,
                "source_tick": int(world_frame_id),
                "sensors": calibrations,
                "cross_platform": pairwise,
                "reprojection_error_px": None,
                "reprojection_threshold_status": "pilot_provisional",
            },
        )
        for association in associations:
            record = dict(association)
            record.update(
                {
                    "world_frame_id": int(world_frame_id),
                    "timestamp": float(timestamp),
                    "simulation_time": float(simulation_time),
                    "sample_index": sample_index,
                    "source_tick": int(world_frame_id),
                }
            )
            self.writer.append_jsonl("associations/cross_view.jsonl", record)
        for air_platform_id in sorted(self.air_adapters):
            for source, target in (
                (air_platform_id, "vehicle_01"),
                ("vehicle_01", air_platform_id),
            ):
                communication_record = self.communication.frame_record(
                    world_frame_id,
                    source,
                    target,
                    [target],
                )
                communication_record.update(
                    {
                        "timestamp": float(timestamp),
                        "simulation_time": float(simulation_time),
                        "sample_index": sample_index,
                        "source_tick": int(world_frame_id),
                    }
                )
                self.writer.append_jsonl(
                    "communication/topology.jsonl",
                    communication_record,
                )
        self.sample_index += 1
        return {
            "sync": sync_report,
            "ground_state": state,
            "associations": associations,
            "sample_index": sample_index,
        }

    def commit_scene(self, extra_manifest: Dict[str, Any]) -> Path:
        self.writer.write_json("platforms/platforms.json", self.platform_registry.as_dict())
        self.writer.write_json("quality/ground_vehicle.json", self.ground_vehicle.check_qa())
        payload = dict(extra_manifest)
        payload.update(
            {
                "scene_id": self.scene_id,
                "task_id": self.task_id,
                "scene_seed": self.scene_seed,
                "schema_version": SCHEMA_VERSION,
                "communication_seed": self.scene_seed,
                "communication_profile": self.communication.profile_name,
                "raw_sensor_data_modified_by_communication": False,
            }
        )
        destination = self.writer.commit(payload)
        self.ground_vehicle.destroy()
        self.object_registry = None
        self.communication = None
        self.scene_id = ""
        return destination

    def close_scene(self, abort_reason: Optional[str] = None) -> None:
        if self.writer.current_path is not None:
            self.writer.abort(abort_reason or "collector_closed_before_commit")
        if self.ground_vehicle.vehicle is not None:
            self.ground_vehicle.destroy()

    def close(self) -> None:
        self.close_scene(abort_reason="collector_closed")
