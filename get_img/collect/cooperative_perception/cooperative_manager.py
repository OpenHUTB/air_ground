"""Lifecycle façade inserted into the three existing collectors."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .association_manager import AssociationManager
from .air_sensor_adapter import AirSensorAdapter
from .calibration_manager import CalibrationManager, backproject_pixel, project_world_point
from .communication_manager import CommunicationManager
from .dataset_writer import DatasetWriter
from .ground_vehicle_manager import GroundVehicleManager
from .object_registry import ObjectRegistry
from .platform_registry import PlatformRegistry
from .quality_manager import QualityManager
from .runtime import CooperativePerceptionRuntime
from .schema import SCHEMA_VERSION, TASK_IDS


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _code_version() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


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
        self.runtime: Optional[CooperativePerceptionRuntime] = None
        self.target_actor_id: Optional[int] = None
        self.defer_perception = bool(
            self.config.get("perception", {}).get("defer_until_replay", False)
        )

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
        self.platform_registry = PlatformRegistry()
        self.object_registry = ObjectRegistry(
            self.scene_id,
            dataset_namespace=str(
                self.config.get("dataset_namespace", "air-ground-coop-suite-v1")
            ),
        )
        self.object_registry.register_many(list(actors))
        self.target_actor_id = None if target_actor is None else int(target_actor.id)
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
            self.scene_id,
        )
        if self.defer_perception:
            self.runtime = None
        else:
            try:
                self.runtime = CooperativePerceptionRuntime(
                    self.task_id, self.scene_id, self.communication, self.config
                )
            except Exception:
                self.ground_vehicle.destroy()
                self.object_registry = None
                self.communication = None
                self.scene_id = ""
                raise
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
                "perception_execution": (
                    "deferred_until_offline_replay"
                    if self.defer_perception
                    else "online"
                ),
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

    @staticmethod
    def _camera_array(data: Any) -> np.ndarray:
        array = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(
            int(data.height), int(data.width), 4
        )
        return array[:, :, :3].copy()

    @staticmethod
    def _metric_depth(data: Any) -> np.ndarray:
        array = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(
            int(data.height), int(data.width), 4
        ).astype(np.float64)
        encoded = array[:, :, 2] + array[:, :, 1] * 256.0 + array[:, :, 0] * 65536.0
        return (encoded / 16777215.0 * 1000.0).astype(np.float32)

    @staticmethod
    def _lidar_points(data: Any) -> np.ndarray:
        return np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].copy()

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
        platform_states: Dict[str, Dict[str, Any]] = {
            self.ground_vehicle.platform_id: state
        }
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
            platform_states[platform_id] = air_state
        ground_vehicle_transform = self.ground_vehicle.vehicle.get_transform()
        calibrations: Dict[str, Any] = {}
        ground_sensor_transforms: Dict[str, Any] = {}
        for modality, sensor in sorted(self.ground_vehicle.sensors.items()):
            sensor_id = "%s_%s" % (self.ground_vehicle.platform_id, modality)
            sensor_transform = sensor.get_transform()
            ground_sensor_transforms[sensor_id] = sensor_transform
            calibrations[sensor_id] = self.calibration.sensor_calibration_from_actor(
                ground_vehicle_transform, sensor
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
                calibrations[sensor_id] = self.calibration.sensor_calibration_from_actor(
                    platform_transform, sensor
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
        reprojection_errors = []
        for calibration_record in calibrations.values():
            if calibration_record.get("intrinsic") is None:
                continue
            width = float(calibration_record["image_width"])
            height = float(calibration_record["image_height"])
            pixel = [width * 0.6, height * 0.4]
            world_point = backproject_pixel(pixel, 20.0, calibration_record)
            projected = project_world_point(world_point, calibration_record)
            reprojection_errors.append(
                float(((projected[0] - pixel[0]) ** 2 + (projected[1] - pixel[1]) ** 2) ** 0.5)
            )
        reprojection_error = max(reprojection_errors or [float("inf")])
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
                "reprojection_error_px": reprojection_error,
                "reprojection_ok": bool(reprojection_error <= 1.0e-5),
                "reprojection_threshold_px": 1.0e-5,
            },
        )
        algorithm: Optional[Dict[str, Any]] = None
        if self.runtime is not None:
            platform_bundles: Dict[str, Dict[str, Any]] = {}
            for platform_id in sorted(platform_states):
                rgb_id = "%s_rgb" % platform_id
                depth_id = "%s_depth" % platform_id
                if rgb_id not in sensor_data_by_id or depth_id not in sensor_data_by_id:
                    continue
                bundle = {
                    "sensor_id": rgb_id,
                    "rgb": self._camera_array(sensor_data_by_id[rgb_id]),
                    "depth_m": self._metric_depth(sensor_data_by_id[depth_id]),
                    "timestamp": float(getattr(sensor_data_by_id[rgb_id], "timestamp", timestamp)),
                    "world_frame_id": int(world_frame_id),
                }
                lidar_id = "%s_lidar" % platform_id
                if lidar_id in sensor_data_by_id and lidar_id in calibrations:
                    bundle["lidar_points"] = self._lidar_points(sensor_data_by_id[lidar_id])
                    bundle["T_world_from_lidar"] = calibrations[lidar_id]["T_world_from_sensor"]
                platform_bundles[platform_id] = bundle
            if set(platform_bundles) != set(platform_states):
                raise RuntimeError("every cooperative platform requires RGB and metric depth")
            vot_initial_boxes: Optional[Dict[str, Any]] = None
            if self.task_id == "task2" and sample_index == 0 and self.target_actor_id is not None:
                vot_initial_boxes = {}
                for sensor_id, rows in temporal_observations.items():
                    platform_id = str(sensor_id).rsplit("_rgb", 1)[0]
                    match = next((row for row in rows if int(row.get("carla_actor_id", -1)) == self.target_actor_id), None)
                    if match is not None:
                        vot_initial_boxes[platform_id] = {
                            "bbox_xyxy": [float(value) for value in match["bbox_xyxy"]],
                            "class_name": str(match["class_name"]),
                        }
            algorithm = self.runtime.process_frame(
                world_frame_id, float(timestamp), platform_bundles, calibrations,
                platform_states, vot_initial_boxes
            )
            for platform_id, result in algorithm["local_perception"].items():
                self.writer.append_jsonl("algorithms/local/%s.jsonl" % platform_id, result)
            for mode in ("air_only", "ground_only", "cooperative"):
                self.writer.append_jsonl(
                    "algorithms/results/%s.jsonl" % mode,
                    {"world_frame_id": int(world_frame_id), "timestamp": float(timestamp),
                     "objects": algorithm[mode]},
                )
            for message in algorithm["messages_sent"]:
                self.writer.append_jsonl("communication/messages.jsonl", message.record())
            for message in algorithm["messages_delivered"]:
                self.writer.append_jsonl(
                    "communication/events.jsonl",
                    {"event": "consumed-by-fusion", **message.record()},
                )
            if self.task_id == "task2":
                self.writer.append_jsonl("air_track.jsonl", {"world_frame_id": int(world_frame_id), "tracks": algorithm["air_only"]})
                self.writer.append_jsonl("ground_track.jsonl", {"world_frame_id": int(world_frame_id), "tracks": algorithm["ground_only"]})
                self.writer.append_jsonl("fused_track.jsonl", {"world_frame_id": int(world_frame_id), "tracks": algorithm["cooperative"], "target": algorithm["fused_sot"]})
                if algorithm["handoff_event"] is not None:
                    self.writer.append_jsonl("handoff_events.jsonl", algorithm["handoff_event"])
            if self.task_id == "task3":
                for platform_id, tracks in algorithm["local_tracks"].items():
                    self.writer.append_jsonl(
                        "local_tracks/%s.jsonl" % platform_id,
                        {"world_frame_id": int(world_frame_id), "tracks": tracks},
                    )
                self.writer.append_jsonl(
                    "fused_tracks.jsonl",
                    {"world_frame_id": int(world_frame_id), "tracks": algorithm["cooperative"]},
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
            "algorithm": algorithm,
        }

    def commit_scene(self, extra_manifest: Dict[str, Any]) -> Path:
        self.writer.write_json("platforms/platforms.json", self.platform_registry.as_dict())
        ground_qa = self.ground_vehicle.check_qa()
        self.writer.write_json("quality/ground_vehicle.json", ground_qa)
        ground_blockers = self.quality.ground_commit_blockers(ground_qa)
        if ground_blockers:
            self.close_scene(abort_reason="ground_vehicle_qa:" + ",".join(ground_blockers))
            raise RuntimeError("ground vehicle QA prevents commit: %s" % ground_blockers)
        payload = dict(extra_manifest)
        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.defer_perception:
            payload.update(
                {
                    "run_mode": "synchronized_raw_acquisition",
                    "offline_perception_status": "pending",
                    "model_weights_sha256": None,
                    "cross_platform_fusion_verified": False,
                    "ground_tracking_contribution_verified": False,
                    "ownership_handoff_verified": False,
                }
            )
        else:
            if self.runtime is None or not self.runtime.detector_is_real_model:
                self.close_scene(abort_reason="real_detector_not_verified")
                raise RuntimeError("a real local detector is required for commit")
            if not self.runtime.cross_platform_fusion_seen:
                self.close_scene(abort_reason="no_algorithmic_air_ground_fusion")
                raise RuntimeError("no fused result used arrived Air and Ground messages")
            if self.task_id in ("task2", "task3") and not self.runtime.ground_contribution_seen:
                self.close_scene(abort_reason="ground_did_not_contribute_to_tracking")
                raise RuntimeError("Ground messages did not contribute to tracking")
            payload.update(
                {
                "scene_id": self.scene_id,
                "task_id": self.task_id,
                "scene_seed": self.scene_seed,
                "schema_version": SCHEMA_VERSION,
                "communication_seed": self.scene_seed,
                "communication_profile": self.communication.profile_name,
                "raw_sensor_data_modified_by_communication": False,
                "model_weights_sha256": self.runtime.model_hash,
                "run_mode": "online_cooperative_perception",
                "config_sha256": config_hash,
                "code_version": _code_version(),
                "cross_platform_fusion_verified": True,
                "ground_tracking_contribution_verified": bool(
                    self.runtime.ground_contribution_seen
                ),
                "ownership_handoff_verified": bool(self.runtime.handoff_seen),
                }
            )
        payload.update(
            {
                "scene_id": self.scene_id,
                "task_id": self.task_id,
                "scene_seed": self.scene_seed,
                "schema_version": SCHEMA_VERSION,
                "communication_seed": self.scene_seed,
                "communication_profile": self.communication.profile_name,
                "raw_sensor_data_modified_by_communication": False,
                "config_sha256": config_hash,
                "code_version": _code_version(),
            }
        )
        destination = self.writer.commit(payload)
        self.ground_vehicle.destroy()
        self.object_registry = None
        self.communication = None
        self.runtime = None
        self.scene_id = ""
        return destination

    def close_scene(self, abort_reason: Optional[str] = None) -> None:
        if self.writer.current_path is not None:
            self.writer.abort(abort_reason or "collector_closed_before_commit")
        if self.ground_vehicle.vehicle is not None:
            self.ground_vehicle.destroy()

    def close(self) -> None:
        self.close_scene(abort_reason="collector_closed")
