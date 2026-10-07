"""Exact world-frame barrier for Air and Ground sensors."""

from __future__ import annotations

import queue
import time
from typing import Any, Dict, Iterable, Optional


class SensorQueue:
    def __init__(self, sensor_id: str, sensor: Any):
        self.sensor_id = str(sensor_id)
        self.sensor = sensor
        self.queue = queue.Queue()
        sensor.listen(self.queue.put)

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def get(self, world_frame_id: int, timeout: float) -> Any:
        deadline = time.time() + float(timeout)
        last_frame = None
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            try:
                data = self.queue.get(timeout=remaining)
            except queue.Empty:
                break
            last_frame = int(data.frame)
            if last_frame == int(world_frame_id):
                return data
            if last_frame < int(world_frame_id):
                continue
            raise TimeoutError(
                "Sensor %s skipped frame %s and produced %s"
                % (self.sensor_id, world_frame_id, last_frame)
            )
        raise TimeoutError(
            "Timed out waiting for sensor %s at frame %s; last=%s"
            % (self.sensor_id, world_frame_id, last_frame)
        )


class SyncManager:
    def __init__(self, timeout: float = 15.0):
        self.timeout = float(timeout)
        self._queues: Dict[str, SensorQueue] = {}
        self._expected = []

    def register_sensor(self, sensor_id: str, sensor: Any, required: bool = True) -> None:
        if sensor_id in self._queues:
            raise ValueError("Duplicate sensor_id: %s" % sensor_id)
        self._queues[sensor_id] = SensorQueue(sensor_id, sensor)
        if required:
            self._expected.append(sensor_id)

    def expected_sensor_ids(self) -> Iterable[str]:
        return tuple(self._expected)

    def drain(self) -> None:
        for sensor_queue in self._queues.values():
            sensor_queue.drain()

    def collect(
        self,
        world_frame_id: int,
        sensor_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        requested = list(sensor_ids) if sensor_ids is not None else list(self._expected)
        return {
            sensor_id: self._queues[sensor_id].get(world_frame_id, self.timeout)
            for sensor_id in requested
        }

