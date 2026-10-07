"""Scene-level deterministic communication metadata."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable


DEFAULT_PROFILES = {
    "Ideal": {"latency_ms": [0.0, 0.0], "packet_loss": [0.0, 0.0], "byte_budget": None},
    "Stable": {"latency_ms": [20.0, 50.0], "packet_loss": [0.0, 0.01], "byte_budget": 8_000_000},
    "Constrained": {"latency_ms": [80.0, 150.0], "packet_loss": [0.05, 0.10], "byte_budget": 1_000_000},
    "Intermittent": {"latency_ms": [100.0, 300.0], "packet_loss": [0.10, 0.30], "byte_budget": 250_000},
}


class CommunicationManager:
    def __init__(self, scene_seed: int, config: Dict[str, Any]):
        self.scene_seed = int(scene_seed)
        self.config = dict(config or {})
        names = list(self.config.get("profiles", DEFAULT_PROFILES).keys())
        self.profile_name = names[self.scene_seed % len(names)]
        self.profile = dict(
            self.config.get("profiles", DEFAULT_PROFILES)[self.profile_name]
        )

    def frame_record(
        self,
        world_frame_id: int,
        source_platform: str,
        target_platform: str,
        available_neighbors: Iterable[str],
    ) -> Dict[str, Any]:
        seed = self.scene_seed * 1_000_003 + int(world_frame_id)
        rng = random.Random(seed)
        latency_range = self.profile.get("latency_ms", [0.0, 0.0])
        loss_range = self.profile.get("packet_loss", [0.0, 0.0])
        latency = rng.uniform(float(latency_range[0]), float(latency_range[1]))
        packet_loss = rng.uniform(float(loss_range[0]), float(loss_range[1]))
        return {
            "world_frame_id": int(world_frame_id),
            "source_platform": str(source_platform),
            "target_platform": str(target_platform),
            "communication_seed": int(seed),
            "communication_profile": self.profile_name,
            "latency_ms": latency,
            "packet_loss": packet_loss,
            "byte_budget": self.profile.get("byte_budget"),
            "available_neighbors": list(available_neighbors),
            "message_available": bool(rng.random() >= packet_loss),
            "raw_sensor_data_modified": False,
            "threshold_status": "pilot_provisional",
        }

