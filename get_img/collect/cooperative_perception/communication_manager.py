"""Deterministic, stateful communication used by perception algorithms."""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PROFILES = {
    "Ideal": {"latency_ms": [0.0, 0.0], "packet_loss": [0.0, 0.0], "byte_budget": None},
    "Stable": {"latency_ms": [20.0, 50.0], "packet_loss": [0.0, 0.01], "byte_budget": 8_000_000},
    "Constrained": {"latency_ms": [80.0, 150.0], "packet_loss": [0.05, 0.10], "byte_budget": 1_000_000},
    "Intermittent": {"latency_ms": [100.0, 300.0], "packet_loss": [0.10, 0.30], "byte_budget": 250_000},
}


@dataclass
class AlgorithmMessage:
    message_id: str
    source_platform: str
    target_platform: str
    source_frame_id: int
    source_timestamp: float
    send_time: float
    scheduled_arrival_time: float
    actual_arrival_time: Optional[float]
    payload_type: str
    payload_size_bytes: int
    payload_reference: str
    payload: Any
    dropped: bool
    drop_reason: Optional[str]
    latency_ms: float
    communication_profile: str
    expires_at: float
    compressed: bool = False
    consumed_by_fusion: bool = False

    def record(self, include_payload: bool = False) -> Dict[str, Any]:
        row = asdict(self)
        if not include_payload:
            row.pop("payload", None)
        row["message_available"] = bool(not self.dropped and self.actual_arrival_time is not None)
        return row


class NetworkSimulator:
    """Direction-aware queue; only ``receive`` exposes arrived packets."""

    def __init__(self, scene_id: str, scene_seed: int, profile_name: str,
                 profile: Dict[str, Any], max_age_ms: float = 500.0,
                 ground_messages_enabled: bool = True) -> None:
        self.scene_id = str(scene_id)
        self.scene_seed = int(scene_seed)
        self.profile_name = str(profile_name)
        self.profile = dict(profile)
        self.max_age_ms = float(max_age_ms)
        self.ground_messages_enabled = bool(ground_messages_enabled)
        self.pending: List[AlgorithmMessage] = []
        self.history: List[AlgorithmMessage] = []
        self._budget_usage: Dict[Tuple[int, str, str], int] = {}
        self._sequence = 0

    def _rng(self, frame_id: int, source: str, target: str) -> random.Random:
        material = "%s|%d|%d|%s|%s" % (
            self.scene_id, self.scene_seed, int(frame_id), source, target
        )
        return random.Random(int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16))

    @staticmethod
    def _encoded_size(payload: Any) -> int:
        return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def _fit_budget(self, payload: Any, budget_left: Optional[int]) -> Tuple[Any, int, bool, Optional[str]]:
        size = self._encoded_size(payload)
        if budget_left is None or size <= budget_left:
            return payload, size, False, None
        if isinstance(payload, list):
            ordered = sorted(payload, key=lambda item: float(item.get("confidence", 0.0))
                             if isinstance(item, dict) else 0.0, reverse=True)
            kept: List[Any] = []
            for item in ordered:
                candidate = kept + [item]
                if self._encoded_size(candidate) > max(0, int(budget_left)):
                    break
                kept = candidate
            if kept:
                return kept, self._encoded_size(kept), True, None
        return payload, size, False, "budget_rejected"

    def send(self, source_platform: str, target_platform: str, source_frame_id: int,
             source_timestamp: float, send_time: float, payload_type: str,
             payload: Any, payload_reference: str) -> AlgorithmMessage:
        source, target = str(source_platform), str(target_platform)
        rng = self._rng(source_frame_id, source, target)
        latency_range = self.profile.get("latency_ms", [0.0, 0.0])
        loss_range = self.profile.get("packet_loss", [0.0, 0.0])
        latency_ms = rng.uniform(float(latency_range[0]), float(latency_range[1]))
        loss_probability = rng.uniform(float(loss_range[0]), float(loss_range[1]))
        key = (int(source_frame_id), source, target)
        budget = self.profile.get("byte_budget")
        budget_left = None if budget is None else int(budget) - self._budget_usage.get(key, 0)
        fitted, size, compressed, budget_reason = self._fit_budget(payload, budget_left)
        ground_disabled = source.startswith("vehicle_") and not self.ground_messages_enabled
        packet_lost = rng.random() < loss_probability
        dropped = bool(budget_reason or ground_disabled or packet_lost)
        reason = budget_reason or ("ground_messages_disabled" if ground_disabled else None)
        if reason is None and packet_lost:
            reason = "packet_loss"
        if not dropped:
            self._budget_usage[key] = self._budget_usage.get(key, 0) + size
        self._sequence += 1
        message_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "%s|%d|%s|%s|%d" % (
            self.scene_id, int(source_frame_id), source, target, self._sequence
        )))
        message = AlgorithmMessage(
            message_id, source, target, int(source_frame_id), float(source_timestamp),
            float(send_time), float(send_time) + latency_ms / 1000.0, None,
            str(payload_type), int(size), str(payload_reference), fitted, dropped,
            reason, float(latency_ms), self.profile_name,
            float(send_time) + self.max_age_ms / 1000.0, bool(compressed), False,
        )
        self.history.append(message)
        if not dropped:
            self.pending.append(message)
        return message

    def receive(self, current_time: float, target_platform: Optional[str] = None) -> List[AlgorithmMessage]:
        delivered: List[AlgorithmMessage] = []
        keep: List[AlgorithmMessage] = []
        for message in self.pending:
            if target_platform is not None and message.target_platform != target_platform:
                keep.append(message)
            elif float(current_time) > message.expires_at:
                message.dropped, message.drop_reason = True, "expired"
            elif message.scheduled_arrival_time <= float(current_time):
                message.actual_arrival_time = float(current_time)
                delivered.append(message)
            else:
                keep.append(message)
        self.pending = keep
        return delivered

    @staticmethod
    def compensate_detection(detection: Dict[str, Any], fusion_timestamp: float) -> Dict[str, Any]:
        result = dict(detection)
        position = list(result.get("world_position", [0.0, 0.0, 0.0]))
        velocity = list(result.get("world_velocity", [0.0, 0.0, 0.0]))
        age = max(0.0, float(fusion_timestamp) - float(result.get("timestamp", fusion_timestamp)))
        result["world_position"] = [float(position[i]) + float(velocity[i]) * age for i in range(3)]
        result["motion_compensation_s"] = age
        return result


class CommunicationManager:
    """Compatibility facade plus the real algorithm message queue."""

    def __init__(self, scene_seed: int, config: Dict[str, Any], scene_id: str = ""):
        self.scene_seed = int(scene_seed)
        self.config = dict(config or {})
        profiles = dict(self.config.get("profiles", DEFAULT_PROFILES))
        requested = self.config.get("profile")
        names = list(profiles.keys())
        self.profile_name = str(requested) if requested in profiles else names[self.scene_seed % len(names)]
        self.profile = dict(profiles[self.profile_name])
        self.network = NetworkSimulator(
            scene_id, self.scene_seed, self.profile_name, self.profile,
            float(self.config.get("max_age_ms", 500.0)),
            bool(self.config.get("ground_messages_enabled", True)),
        )

    def frame_record(self, world_frame_id: int, source_platform: str,
                     target_platform: str, available_neighbors: Iterable[str]) -> Dict[str, Any]:
        source, target = str(source_platform), str(target_platform)
        rng = self.network._rng(world_frame_id, source, target)
        latency_range = self.profile.get("latency_ms", [0.0, 0.0])
        loss_range = self.profile.get("packet_loss", [0.0, 0.0])
        material = "%s|%s|%s|%s" % (self.scene_seed, world_frame_id, source, target)
        return {
            "world_frame_id": int(world_frame_id), "source_platform": source,
            "target_platform": target,
            "communication_seed": int(hashlib.sha256(material.encode()).hexdigest()[:16], 16),
            "communication_profile": self.profile_name,
            "latency_ms": rng.uniform(float(latency_range[0]), float(latency_range[1])),
            "packet_loss": rng.uniform(float(loss_range[0]), float(loss_range[1])),
            "byte_budget": self.profile.get("byte_budget"),
            "available_neighbors": list(available_neighbors),
            "message_available": False, "metadata_only": True,
            "raw_sensor_data_modified": False,
        }
