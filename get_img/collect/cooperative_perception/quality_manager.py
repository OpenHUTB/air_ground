"""Shared cooperative QA helpers with explicitly provisional thresholds."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


class QualityManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})

    def frame_sync_report(
        self,
        world_frame_id: int,
        simulation_time: float,
        sensor_data: Dict[str, Any],
        expected_sensor_ids: Iterable[str],
    ) -> Dict[str, Any]:
        expected = sorted(set(str(value) for value in expected_sensor_ids))
        missing = sorted(set(expected) - set(sensor_data))
        mismatched = sorted(
            sensor_id
            for sensor_id, data in sensor_data.items()
            if int(getattr(data, "frame", -1)) != int(world_frame_id)
        )
        timestamps = {
            sensor_id: float(getattr(sensor_data[sensor_id], "timestamp", float("nan")))
            for sensor_id in expected
            if sensor_id in sensor_data
        }
        timestamp_tolerance = float(
            self.config.get("sensor_timestamp_tolerance_s", 1.0e-4)
        )
        invalid_timestamps = sorted(
            sensor_id
            for sensor_id, value in timestamps.items()
            if not math.isfinite(value)
            or abs(float(value) - float(simulation_time)) > timestamp_tolerance
        )
        timestamp_spread = (
            max(timestamps.values()) - min(timestamps.values())
            if timestamps
            else float("inf")
        )
        sync_ok = (
            not missing
            and not mismatched
            and not invalid_timestamps
            and timestamp_spread <= timestamp_tolerance
        )
        return {
            "world_frame_id": int(world_frame_id),
            "simulation_time": float(simulation_time),
            "sync_ok": sync_ok,
            "expected_sensor_ids": expected,
            "received_sensor_ids": sorted(sensor_data),
            "missing_sensor_ids": missing,
            "mismatched_sensor_ids": mismatched,
            "invalid_timestamp_sensor_ids": invalid_timestamps,
            "sensor_timestamps": timestamps,
            "timestamp_spread_s": float(timestamp_spread),
            "timestamp_tolerance_s": timestamp_tolerance,
            "modalities_complete": sync_ok,
        }

    def trajectory_report(self, states: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(states)
        distances: List[float] = []
        moving = 0
        for previous, current in zip(rows, rows[1:]):
            a = previous["platform_pose"]["location"]
            b = current["platform_pose"]["location"]
            distance = math.sqrt(
                (float(a["x"]) - float(b["x"])) ** 2
                + (float(a["y"]) - float(b["y"])) ** 2
                + (float(a["z"]) - float(b["z"])) ** 2
            )
            distances.append(distance)
            moving += int(distance > float(self.config.get("moving_step_m", 0.02)))
        return {
            "route_completion_ratio": rows[-1].get("route_completion_ratio", 0.0) if rows else 0.0,
            "travel_distance_m": float(sum(distances)),
            "motion_ratio": float(moving / max(1, len(distances))),
            "collision_count": max([int(row.get("collision_count", 0)) for row in rows] or [0]),
            "offroad_count": max([int(row.get("offroad_count", 0)) for row in rows] or [0]),
            "threshold_status": "pilot_provisional",
        }
