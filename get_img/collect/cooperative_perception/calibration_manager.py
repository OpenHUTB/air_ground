"""Transform graph and CARLA/OpenCV coordinate conversion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def camera_intrinsic(width: int, height: int, fov_degrees: float) -> List[List[float]]:
    """Return a CARLA pinhole intrinsic matrix (horizontal field of view)."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or not 0.0 < float(fov_degrees) < 180.0:
        raise ValueError("invalid camera geometry")
    focal = width / (2.0 * np.tan(np.deg2rad(float(fov_degrees)) / 2.0))
    return matrix_list(
        np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    )


def camera_model_from_sensor(sensor: Any) -> Optional[Dict[str, Any]]:
    """Read camera geometry from a CARLA sensor without guessing actor APIs."""
    attributes = dict(getattr(sensor, "attributes", {}) or {})
    if not {"image_size_x", "image_size_y", "fov"}.issubset(attributes):
        return None
    width = int(attributes["image_size_x"])
    height = int(attributes["image_size_y"])
    fov = float(attributes["fov"])
    return {
        "intrinsic": camera_intrinsic(width, height, fov),
        "image_width": width,
        "image_height": height,
        "horizontal_fov_deg": fov,
        "distortion_model": "none",
        "distortion_coefficients": [],
    }


def project_world_point(point_world: Sequence[float], calibration: Dict[str, Any]) -> Tuple[float, float, float]:
    intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64)
    camera_from_world = np.asarray(calibration["T_opencv_camera_from_world"], dtype=np.float64)
    point = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    camera = camera_from_world.dot(point)[:3]
    if camera[2] <= 0.0:
        raise ValueError("world point is behind the camera")
    pixel = intrinsic.dot(camera)
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]), float(camera[2])


def backproject_pixel(pixel: Sequence[float], depth_m: float, calibration: Dict[str, Any]) -> List[float]:
    if float(depth_m) <= 0.0:
        raise ValueError("depth_m must be positive")
    intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64)
    world_from_camera = np.linalg.inv(
        np.asarray(calibration["T_opencv_camera_from_world"], dtype=np.float64)
    )
    ray = np.linalg.inv(intrinsic).dot(
        np.asarray([float(pixel[0]), float(pixel[1]), 1.0], dtype=np.float64)
    )
    camera = ray * float(depth_m)
    world = world_from_camera.dot(np.asarray([camera[0], camera[1], camera[2], 1.0]))
    return [float(value) for value in world[:3]]


class CalibrationManager:
    def __init__(self, closure_atol: float = 1e-5):
        self.closure_atol = float(closure_atol)

    def sensor_calibration(
        self,
        platform_transform: Any,
        sensor_transform: Any,
        intrinsic: Any = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        fov_degrees: Optional[float] = None,
        distortion_model: str = "none",
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
        if intrinsic is None and image_width and image_height and fov_degrees:
            intrinsic = camera_intrinsic(image_width, image_height, fov_degrees)
        return {
            "coordinate_convention": COORDINATE_CONVENTION,
            "matrix_storage": "row-major",
            "vector_convention": "column-vector",
            "units": "meter",
            "intrinsic": intrinsic,
            "image_width": None if image_width is None else int(image_width),
            "image_height": None if image_height is None else int(image_height),
            "horizontal_fov_deg": None if fov_degrees is None else float(fov_degrees),
            "distortion_model": str(distortion_model),
            "distortion_coefficients": [],
            "carla_sensor_axes": {"forward": "+x", "right": "+y", "up": "+z"},
            "opencv_camera_axes": {"right": "+x", "down": "+y", "forward": "+z"},
            "T_world_from_platform": matrix_list(world_from_platform),
            "T_platform_from_sensor": matrix_list(platform_from_sensor),
            "T_world_from_sensor": matrix_list(world_from_sensor),
            "T_sensor_from_world": matrix_list(sensor_from_world),
            "T_opencv_camera_from_world": matrix_list(opencv_camera_from_world),
            "closure_max_abs_error": closure_error,
            "closure_ok": bool(closure_error <= self.closure_atol),
        }

    def sensor_calibration_from_actor(
        self, platform_transform: Any, sensor: Any
    ) -> Dict[str, Any]:
        model = camera_model_from_sensor(sensor)
        kwargs = {} if model is None else {
            "intrinsic": model["intrinsic"],
            "image_width": model["image_width"],
            "image_height": model["image_height"],
            "fov_degrees": model["horizontal_fov_deg"],
            "distortion_model": model["distortion_model"],
        }
        return self.sensor_calibration(platform_transform, sensor.get_transform(), **kwargs)

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
