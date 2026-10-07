"""Transform graph and CARLA/OpenCV coordinate conversion."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .schema import COORDINATE_CONVENTION


CARLA_SENSOR_TO_OPENCV = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def transform_matrix(transform: Any) -> np.ndarray:
    if hasattr(transform, "get_matrix"):
        return np.asarray(transform.get_matrix(), dtype=np.float64).reshape(4, 4)
    location = transform.location
    rotation = transform.rotation
    yaw = np.deg2rad(float(rotation.yaw))
    pitch = np.deg2rad(float(rotation.pitch))
    roll = np.deg2rad(float(rotation.roll))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )
    matrix[:3, 3] = [float(location.x), float(location.y), float(location.z)]
    return matrix


def matrix_list(matrix: np.ndarray) -> List[List[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix)]


class CalibrationManager:
    def __init__(self, closure_atol: float = 1e-5):
        self.closure_atol = float(closure_atol)

    def sensor_calibration(
        self,
        platform_transform: Any,
        sensor_transform: Any,
        intrinsic: Any = None,
    ) -> Dict[str, Any]:
        world_from_platform = transform_matrix(platform_transform)
        world_from_sensor = transform_matrix(sensor_transform)
        platform_from_sensor = np.linalg.inv(world_from_platform).dot(
            world_from_sensor
        )
        sensor_from_world = np.linalg.inv(world_from_sensor)
        opencv_camera_from_world = CARLA_SENSOR_TO_OPENCV.dot(sensor_from_world)
        closure_error = float(
            np.max(np.abs(world_from_sensor.dot(sensor_from_world) - np.eye(4)))
        )
        return {
            "coordinate_convention": COORDINATE_CONVENTION,
            "matrix_storage": "row-major",
            "vector_convention": "column-vector",
            "units": "meter",
            "intrinsic": intrinsic,
            "T_world_from_platform": matrix_list(world_from_platform),
            "T_platform_from_sensor": matrix_list(platform_from_sensor),
            "T_world_from_sensor": matrix_list(world_from_sensor),
            "T_sensor_from_world": matrix_list(sensor_from_world),
            "T_opencv_camera_from_world": matrix_list(opencv_camera_from_world),
            "closure_max_abs_error": closure_error,
            "closure_ok": bool(closure_error <= self.closure_atol),
        }

    def transform_between(self, target_transform: Any, source_transform: Any) -> Dict[str, Any]:
        world_from_target = transform_matrix(target_transform)
        world_from_source = transform_matrix(source_transform)
        target_from_source = np.linalg.inv(world_from_target).dot(world_from_source)
        source_from_target = np.linalg.inv(target_from_source)
        closure_error = float(
            np.max(np.abs(target_from_source.dot(source_from_target) - np.eye(4)))
        )
        return {
            "T_target_from_source": matrix_list(target_from_source),
            "T_source_from_target": matrix_list(source_from_target),
            "closure_max_abs_error": closure_error,
            "closure_ok": bool(closure_error <= self.closure_atol),
        }

