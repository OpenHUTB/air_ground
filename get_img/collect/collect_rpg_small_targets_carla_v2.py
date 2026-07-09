#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_rpg_small_targets_carla_v2.py

CARLA / OpenHUTB 无人机视角小目标多模态数据集采集脚本。

功能：
1. 连接 CARLA/OpenHUTB 模拟器；
2. 设置同步模式；
3. 创建空中“无人机视角”传感器平台；
4. 同步采集：
   - RGB 图像
   - 深度图 Depth
   - 语义分割 Semantic Segmentation
   - 实例分割 Instance Segmentation
   - 可选 IMU / GNSS / LiDAR
5. 使用 instance segmentation 的可见像素生成一实例一目标 bbox；
6. 根据深度图保存每个目标区域的深度 crop / normalized disparity；
7. 输出：
   - RGB 图像
   - depth npy
   - depth 16-bit 可视化图
   - semantic 图
   - instance 图
   - full-frame normalized disparity 图
   - per-object mask / depth crop / disparity crop
   - mask 图
   - YOLO 标签
   - JSON 标注
   - groundtruth.txt
   - groundtruth_multi.csv
   - dataset_manifest.json

重点：
- 不用 AirSim；
- 使用 CARLA 原生 sensor；
- 修复 _queue.Empty；
- 加入 sensor warm-up；
- 加入稳健同步队列；
- 加入超时跳帧保护；
- 默认 sensor_tick=0.0，每个 world.tick 都输出一帧。

运行示例：

先改脚本同目录的 collection_config.json，然后直接运行：

python collect_rpg_small_targets_carla_v2.py

python collect_rpg_small_targets_carla_v2.py ^
  --host 127.0.0.1 ^
  --port 2000 ^
  --map Town10HD ^
  --out dataset_rpg_uav_carla ^
  --sequences 1 ^
  --frames 300 ^
  --width 1280 ^
  --height-img 720 ^
  --vehicles 20 ^
  --walkers 10 ^
  --target vehicle:0:10 ^
  --target pedestrian:1:4 ^
  --target traffic_sign:2:12 ^
  --target traffic_light:3:18

如果你有自定义 RPG / missile / rocket 模型，并且在 CARLA 中标成 Dynamic=20：

python collect_rpg_small_targets_carla_v2.py ^
  --out dataset_rpg_uav_carla ^
  --map Town10HD ^
  --sequences 3 ^
  --frames 500 ^
  --target rpg:0:20
"""

import argparse
import csv
import glob
import json
import math
import os
import queue
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 1. 可选：强制指定 OpenHUTB / CARLA 自带 PythonAPI
# ============================================================
# 如果你遇到：
# WARNING: Version mismatch detected
# Client API version != Simulator API version
#
# 建议把下面 CARLA_EGG_GLOB 改成你本机 OpenHUTB 的 egg 路径。
# 例如：
# CARLA_EGG_GLOB = r"E:\OpenHUTB\hutb\PythonAPI\carla\dist\carla-*.egg"
#
# 如果你已经正确设置 PYTHONPATH，可以保持为 None。
CARLA_EGG_GLOB = None

if CARLA_EGG_GLOB is not None:
    egg_list = glob.glob(CARLA_EGG_GLOB)
    if len(egg_list) == 0:
        raise RuntimeError(f"没有找到 carla egg 文件，请检查路径：{CARLA_EGG_GLOB}")
    sys.path.insert(0, egg_list[0])


try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "无法 import carla。\n"
        "请确认 CARLA/OpenHUTB PythonAPI 已加入 PYTHONPATH。\n"
        "例如：\n"
        "set PYTHONPATH=%PYTHONPATH%;E:\\OpenHUTB\\hutb\\PythonAPI\\carla\\dist\\carla-xxx.egg\n"
    ) from exc


# ============================================================
# 2. CARLA 语义标签说明
# ============================================================
"""
常见 CARLA semantic id：

0  Unlabeled
1  Building
2  Fence
3  Other
4  Pedestrian
5  Pole
6  RoadLine
7  Road
8  SideWalk
9  Vegetation
10 Vehicle
11 Wall
12 TrafficSign
13 Sky
14 Ground
15 Bridge
16 RailTrack
17 GuardRail
18 TrafficLight
19 Static
20 Dynamic
21 Water
22 Terrain

如果你导入了 RPG / missile / rocket 自定义模型，
最简单做法是先把它标成：
- Static = 19
或
- Dynamic = 20

然后运行：
--target rpg:0:20
或：
--target rpg:0:19
"""


@dataclass
class TargetClass:
    name: str
    class_id: int
    semantic_ids: List[int]


# ============================================================
# 3. 稳健传感器同步队列
# ============================================================

class SensorSync:
    """
    CARLA 同步模式传感器队列。

    原来简单写法：

        data = queue.get(timeout=2.0)
        if data.frame == frame:
            return data

    在高分辨率、多传感器、交通复杂时容易 _queue.Empty。

    这里做了增强：
    1. 可清空旧数据；
    2. 可丢弃旧帧；
    3. 超时时抛 TimeoutError，而不是直接 _queue.Empty；
    4. 核心相机必须严格同帧；如果拿到比目标帧更新的帧，抛错并丢帧。
    """

    def __init__(self, name: str, sensor: carla.Sensor):
        self.name = name
        self.sensor = sensor
        self.queue: queue.Queue = queue.Queue()
        self.sensor.listen(self.queue.put)

    def drain(self) -> None:
        """清空创建传感器和 warm-up 阶段积压的旧帧。"""
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def get(self, frame: int, timeout: float = 10.0):
        """
        获取指定 CARLA frame 的传感器数据。
        """
        start_time = time.time()
        last_frame = None

        while time.time() - start_time < timeout:
            remaining = max(0.1, timeout - (time.time() - start_time))

            try:
                data = self.queue.get(timeout=remaining)
            except queue.Empty:
                break

            last_frame = data.frame

            if data.frame == frame:
                return data

            if data.frame < frame:
                # 旧帧，丢弃
                continue

            if data.frame > frame:
                raise TimeoutError(
                    f"Sensor '{self.name}' skipped target frame {frame}, "
                    f"got newer frame {data.frame}. Strict sync requires exact frames."
                )

        raise TimeoutError(
            f"Timeout waiting for sensor '{self.name}' at frame {frame}. "
            f"Last received frame: {last_frame}. "
            f"建议：降低分辨率/减少交通/增大 timeout/修复 CARLA API 版本不匹配。"
        )


# ============================================================
# 4. 参数解析
# ============================================================

def parse_target(s: str) -> TargetClass:
    """
    --target 格式：

        name:class_id:semantic_id[,semantic_id...]

    示例：

        vehicle:0:10
        pedestrian:1:4
        traffic_sign:2:12
        rpg:0:20
        rpg:0:19,20
    """
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"target 格式错误：{s}\n"
            f"正确格式：name:class_id:semantic_ids，例如 vehicle:0:10"
        )

    name = parts[0].strip()
    class_id = int(parts[1])
    semantic_ids = [int(x.strip()) for x in parts[2].split(",") if x.strip() != ""]

    if len(name) == 0:
        raise argparse.ArgumentTypeError("target name 不能为空。")
    if len(semantic_ids) == 0:
        raise argparse.ArgumentTypeError("semantic_ids 不能为空。")

    return TargetClass(name=name, class_id=class_id, semantic_ids=semantic_ids)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("collection_config.json")


def _parse_target_config_item(item: Any) -> TargetClass:
    if isinstance(item, str):
        return parse_target(item)

    if isinstance(item, dict):
        try:
            name = str(item["name"])
            class_id = int(item["class_id"])
            semantic_ids = [int(x) for x in item["semantic_ids"]]
        except KeyError as exc:
            raise argparse.ArgumentTypeError(
                f"target 配置缺少字段：{exc}"
            ) from exc

        return TargetClass(
            name=name,
            class_id=class_id,
            semantic_ids=semantic_ids
        )

    raise argparse.ArgumentTypeError(
        "target 配置必须是字符串或对象，例如 vehicle:0:10"
    )


def load_collection_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None or str(config_path).strip() == "":
        return {}

    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON 配置文件格式错误：{path}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"JSON 配置文件顶层必须是对象：{path}")

    aliases = {
        "height_image": "height_img",
        "target_actor_filters": "target_actor_filter",
        "targets": "target",
        "min_vehicle_visible_ratio": "min_actor_visible_ratio",
        "min_vehicle_visible_px": "min_actor_visible_px"
    }

    config: Dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        normalized_key = aliases.get(key, key).replace("-", "_")
        config[normalized_key] = value

    if "target" in config and config["target"] is not None:
        if not isinstance(config["target"], list):
            raise RuntimeError("JSON 配置中的 target 必须是列表。")
        config["target"] = [
            _parse_target_config_item(item)
            for item in config["target"]
        ]

    if "target_actor_filter" in config and config["target_actor_filter"] is None:
        config["target_actor_filter"] = []

    return config


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    config: Dict[str, Any]
) -> None:
    if not config:
        return

    valid_dests = {
        action.dest
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    filtered = {
        key: value
        for key, value in config.items()
        if key in valid_dests
    }
    unknown_keys = sorted(set(config.keys()) - set(filtered.keys()))

    if unknown_keys:
        print(
            "[WARN] JSON 配置中存在脚本未使用的字段，已忽略："
            + ", ".join(unknown_keys)
        )

    parser.set_defaults(**filtered)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="JSON 配置文件路径；默认读取脚本同目录 collection_config.json"
    )
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_collection_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])

    # CARLA 连接参数
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=20.0)

    # 地图和输出
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--out", type=str, default="dataset_uav_small_carla")
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--frames", type=int, default=300)

    # 图像参数
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height-img", type=int, default=720)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument(
        "--enable-rgb-postprocess",
        dest="enable_rgb_postprocess",
        action="store_true",
        default=False,
        help="开启 RGB 相机后处理，使采集图更接近模拟器视口效果。"
    )
    parser.add_argument(
        "--disable-rgb-postprocess",
        dest="enable_rgb_postprocess",
        action="store_false",
        help="关闭 RGB 相机后处理，减少自动曝光/运动模糊等影响。"
    )
    parser.add_argument("--fps", type=float, default=20.0)

    # UAV 视角参数
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--road-centered-camera", action="store_true", default=True)
    parser.add_argument(
        "--fixed-config-center",
        dest="road_centered_camera",
        action="store_false",
        help="使用配置中的 center_x/center_y，不自动选择道路中心。"
    )
    parser.add_argument("--height", type=float, default=130.0)
    parser.add_argument("--height-min", type=float, default=120.0)
    parser.add_argument("--height-max", type=float, default=140.0)
    parser.add_argument("--radius-min", type=float, default=80.0)
    parser.add_argument("--radius-max", type=float, default=180.0)
    parser.add_argument("--pitch", type=float, default=-65.0)
    parser.add_argument("--pitch-min", type=float, default=None)
    parser.add_argument("--pitch-max", type=float, default=None)
    parser.add_argument("--route", type=str, default="orbit", choices=["orbit", "random"])
    parser.add_argument("--orbit-degrees-per-sequence", type=float, default=60.0)
    parser.add_argument("--safe-camera-min-z", type=float, default=120.0)
    parser.add_argument("--max-camera-pose-retries", type=int, default=20)
    parser.add_argument("--min-near-depth-m", type=float, default=5.0)
    parser.add_argument("--max-near-depth-ratio", type=float, default=0.05)
    parser.add_argument("--min-road-visible-ratio", type=float, default=0.25)
    parser.add_argument("--road-semantic-ids", type=int, nargs="+", default=[6, 7, 8])

    # 天气
    parser.add_argument("--weather", type=str, default="ClearNoon")
    parser.add_argument(
        "--keep-current-weather",
        dest="keep_current_weather",
        action="store_true",
        default=False,
        help="保持模拟器当前天气，不调用 world.set_weather。"
    )
    parser.add_argument(
        "--apply-config-weather",
        dest="keep_current_weather",
        action="store_false",
        help="按 --weather 或 --random-weather 设置天气。"
    )
    parser.add_argument("--random-weather", action="store_true")

    # 目标类别
    parser.add_argument(
        "--annotation-source",
        type=str,
        default="hybrid",
        choices=["instance", "actor", "hybrid"],
        help=(
            "标注来源：instance=只用实例分割；"
            "actor=车辆/行人只用 OpenHUTB/CARLA actor 坐标投影；"
            "hybrid=车辆/行人用 actor，其他目标用实例分割。"
        )
    )
    parser.add_argument(
        "--target",
        type=parse_target,
        action="append",
        default=None,
        help=(
            "目标类别格式：name:class_id:semantic_ids。"
            "例如：vehicle:0:10 pedestrian:1:4 rpg:0:20"
        )
    )

    # 小目标筛选
    parser.add_argument("--min-mask-px", type=int, default=8)
    parser.add_argument("--small-area-ratio", type=float, default=0.0025)
    parser.add_argument("--small-max-side-px", type=int, default=96)
    parser.add_argument("--keep-all", action="store_true")
    parser.add_argument("--min-actor-visible-px", type=int, default=12)
    parser.add_argument("--min-actor-visible-ratio", type=float, default=0.01)
    parser.add_argument("--actor-depth-margin", type=float, default=3.0)
    parser.add_argument(
        "--actor-visibility-mode",
        type=str,
        default="depth",
        choices=["depth", "semantic_depth", "projected"],
        help=(
            "actor 可见性过滤方式：depth=只用深度过滤遮挡；"
            "semantic_depth=语义+深度，最严格；"
            "projected=只用 3D 投影框，不过滤遮挡。"
        )
    )

    # 深度图可视化最大距离
    parser.add_argument("--max-depth-vis", type=float, default=250.0)
    parser.add_argument("--min-disparity-depth", type=float, default=1.0)
    parser.add_argument("--max-disparity-depth", type=float, default=250.0)

    # 交通流
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=10)
    parser.add_argument("--no-traffic", action="store_true")
    parser.add_argument(
        "--hide-static-map-vehicles",
        action="store_true",
        default=True,
        help="隐藏地图自带的静态车辆环境对象，只保留脚本生成的 vehicle.* actor。"
    )
    parser.add_argument(
        "--keep-static-map-vehicles",
        dest="hide_static_map_vehicles",
        action="store_false",
        help="保留地图自带静态车辆。"
    )

    # 可选：主动生成自定义小目标 actor。
    # 例如导入 RPG / missile / small UAV blueprint 后：
    # --target-actor-filter "static.prop.rpg*" --target-actor-count 5 --target rpg:0:20
    parser.add_argument("--target-actor-filter", type=str, action="append", default=[])
    parser.add_argument("--target-actor-count", type=int, default=0)
    parser.add_argument("--target-spawn-radius-min", type=float, default=15.0)
    parser.add_argument("--target-spawn-radius-max", type=float, default=80.0)
    parser.add_argument("--target-spawn-z-offset", type=float, default=0.2)

    # 可选 LiDAR
    parser.add_argument("--lidar", action="store_true")

    # 同步和稳定性
    parser.add_argument("--sensor-timeout", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--max-drop-frames", type=int, default=100)

    # 随机种子
    parser.add_argument("--seed", type=int, default=7)

    apply_config_defaults(parser, config_defaults)
    return parser.parse_args()


# ============================================================
# 5. 文件夹创建
# ============================================================

def make_dirs(seq_dir: Path) -> Dict[str, Path]:
    dirs = {
        "rgb": seq_dir / "rgb",
        "depth_npy": seq_dir / "depth_npy",
        "depth_vis": seq_dir / "depth_vis",
        "disparity_vis": seq_dir / "disparity_vis",
        "semantic": seq_dir / "semantic",
        "instance": seq_dir / "instance",
        "mask": seq_dir / "masks",
        "object_depth_npy": seq_dir / "object_depth_npy",
        "object_depth_vis": seq_dir / "object_depth_vis",
        "object_disparity_vis": seq_dir / "object_disparity_vis",
        "ann": seq_dir / "annotations",
        "yolo": seq_dir / "labels_yolo",
        "lidar": seq_dir / "lidar"
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ============================================================
# 6. 图像处理
# ============================================================

def carla_image_to_bgra(image: carla.Image) -> np.ndarray:
    """
    CARLA 图像 raw_data 是 BGRA uint8。
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    return arr.reshape((image.height, image.width, 4))


def save_rgb(image: carla.Image, path: Path) -> np.ndarray:
    """
    保存 RGB 相机图像。

    CARLA raw_data 是 BGRA。
    OpenCV 保存需要 BGR。
    """
    bgra = carla_image_to_bgra(image)
    bgr = bgra[:, :, :3]
    cv2.imwrite(str(path), bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def decode_carla_depth_meters(depth_image: carla.Image) -> np.ndarray:
    """
    将 CARLA depth camera 的 RGB 编码深度解码为米。

    CARLA depth 编码：

        normalized = (R + G * 256 + B * 256 * 256) / (256^3 - 1)
        depth_m = normalized * 1000

    raw_data 是 BGRA，因此：
        B = bgra[:, :, 0]
        G = bgra[:, :, 1]
        R = bgra[:, :, 2]
    """
    bgra = carla_image_to_bgra(depth_image).astype(np.float32)

    b = bgra[:, :, 0]
    g = bgra[:, :, 1]
    r = bgra[:, :, 2]

    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    depth_m = normalized * 1000.0

    return depth_m.astype(np.float32)


def metric_depth_to_u16(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    """
    将米制深度转换为 16-bit PNG 友好的数组。

    0 表示无效/背景；非零值按 [0, max_depth_m] 线性映射。
    """
    d = np.nan_to_num(depth_m, nan=0.0, posinf=max_depth_m, neginf=0.0)
    d = np.clip(d, 0.0, max_depth_m)
    return (d / max_depth_m * 65535.0).astype(np.uint16)


def depth_to_disparity_u16(
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    生成 normalized disparity 16-bit 图。

    disparity 与 1/depth 成正比，近处更亮、远处更暗。
    """
    if min_depth_m <= 0.0:
        raise ValueError("min_depth_m must be > 0")
    if max_depth_m <= min_depth_m:
        raise ValueError("max_depth_m must be larger than min_depth_m")

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if valid_mask is not None:
        valid = valid & valid_mask.astype(bool)

    out = np.zeros(depth_m.shape, dtype=np.uint16)
    if not np.any(valid):
        return out

    d = np.clip(depth_m[valid].astype(np.float32), min_depth_m, max_depth_m)
    inv = 1.0 / d
    inv_min = 1.0 / max_depth_m
    inv_max = 1.0 / min_depth_m
    normalized = (inv - inv_min) / (inv_max - inv_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    out[valid] = (normalized * 65535.0).astype(np.uint16)
    return out


def save_depth(depth_m: np.ndarray, npy_path: Path, vis_path: Path, max_depth_m: float) -> None:
    """
    保存：
    1. 原始深度 npy，单位米；
    2. 16-bit 深度可视化 png。
    """
    np.save(npy_path, depth_m.astype(np.float32))

    cv2.imwrite(str(vis_path), metric_depth_to_u16(depth_m, max_depth_m))


def save_disparity(
    depth_m: np.ndarray,
    path: Path,
    min_depth_m: float,
    max_depth_m: float
) -> None:
    cv2.imwrite(
        str(path),
        depth_to_disparity_u16(depth_m, min_depth_m, max_depth_m)
    )


def is_bad_camera_view(
    depth_m: np.ndarray,
    min_near_depth_m: float,
    max_near_depth_ratio: float
) -> Tuple[bool, Dict[str, float]]:
    """
    判断相机是否进入建筑/贴近遮挡物。

    如果大量像素深度极近，通常意味着相机穿模进楼体或贴着墙。
    """
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        return True, {
            "near_depth_ratio": 1.0,
            "valid_ratio": 0.0,
            "min_depth": 0.0
        }

    valid_depth = depth_m[valid]
    near_ratio = float(np.mean(valid_depth < min_near_depth_m))
    valid_ratio = float(np.mean(valid))
    min_depth = float(np.min(valid_depth))

    return near_ratio > max_near_depth_ratio, {
        "near_depth_ratio": near_ratio,
        "valid_ratio": valid_ratio,
        "min_depth": min_depth
    }


def road_visible_ratio(
    semantic_image: carla.Image,
    road_semantic_ids: List[int]
) -> float:
    """
    计算画面中道路/车道线/人行道等可行驶区域的语义像素比例。

    默认 CARLA 语义：
    6 RoadLine, 7 Road, 8 SideWalk
    """
    bgra = carla_image_to_bgra(semantic_image)
    road_ids = np.array(road_semantic_ids, dtype=np.uint8)

    raw_masks = [
        np.isin(bgra[:, :, channel].astype(np.uint8), road_ids)
        for channel in range(3)
    ]
    raw_ratio = max(float(np.mean(mask)) for mask in raw_masks)

    # OpenHUTB/CARLA 有些版本返回 raw semantic id，有些版本返回 CityScapes 调色板颜色。
    cityscapes_rgb = {
        6: (157, 234, 50),   # RoadLine
        7: (128, 64, 128),   # Road
        8: (244, 35, 232),   # SideWalk
    }
    palette_mask = np.zeros(bgra.shape[:2], dtype=bool)
    for sem_id in road_semantic_ids:
        rgb = cityscapes_rgb.get(int(sem_id))
        if rgb is None:
            continue
        r, g, b = rgb
        palette_mask |= (
            (bgra[:, :, 0] == b)
            & (bgra[:, :, 1] == g)
            & (bgra[:, :, 2] == r)
        )

    palette_ratio = float(np.mean(palette_mask))
    return max(raw_ratio, palette_ratio)


def save_segmentation_raw(image: carla.Image, path: Path) -> np.ndarray:
    """
    保存 semantic / instance 原始图像。
    """
    bgra = carla_image_to_bgra(image)
    cv2.imwrite(str(path), bgra[:, :, :3])
    return bgra


def decode_instance_segmentation(instance_image: carla.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    解码 CARLA instance segmentation。

    CARLA instance segmentation 中：
    - R 通道通常为 semantic id；
    - G/B 通道编码 instance id。

    raw_data 是 BGRA：
    - B = bgra[:, :, 0]
    - G = bgra[:, :, 1]
    - R = bgra[:, :, 2]

    返回：
    - semantic_id: HxW uint8
    - instance_id: HxW uint16
    """
    bgra = carla_image_to_bgra(instance_image)

    semantic_id = bgra[:, :, 2].astype(np.uint8)

    instance_id = (
        bgra[:, :, 1].astype(np.uint16) * 256
        + bgra[:, :, 0].astype(np.uint16)
    )

    return semantic_id, instance_id


# ============================================================
# 7. 标注生成
# ============================================================

def connected_components_from_mask(mask: np.ndarray, min_pixels: int):
    """
    从 mask 中提取连通域。

    返回：
        [(x, y, w, h, area, component_mask), ...]
    """
    mask_u8 = (mask.astype(np.uint8) * 255)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    comps = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if area < min_pixels:
            continue

        comp_mask = labels == i

        comps.append((
            int(x),
            int(y),
            int(w),
            int(h),
            int(area),
            comp_mask
        ))

    return comps


def build_annotations_from_instance(
    instance_image: carla.Image,
    depth_m: np.ndarray,
    targets: List[TargetClass],
    min_mask_px: int,
    small_area_ratio: float,
    small_max_side_px: int,
    keep_all: bool
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    根据 instance segmentation 生成目标标注。

    核心逻辑：
    1. 解码 semantic_id 和 instance_id；
    2. 按 semantic_id 找目标类别；
    3. 每个 semantic_id + instance_id 只生成一个目标；
    4. 用该实例全部可见像素 mask 生成 bbox；
    5. 在同一个可见 mask 内统计深度。
    """
    semantic_id, instance_id = decode_instance_segmentation(instance_image)

    img_h, img_w = semantic_id.shape

    annotations: List[Dict[str, Any]] = []
    ann_id = 0

    for target in targets:
        target_semantic_ids = np.array(target.semantic_ids, dtype=np.uint8)
        target_semantic_mask = np.isin(semantic_id, target_semantic_ids)

        visible_instance_ids = np.unique(instance_id[target_semantic_mask])

        for inst_id in visible_instance_ids:
            if inst_id == 0:
                continue

            instance_mask = target_semantic_mask & (instance_id == inst_id)
            area = int(np.count_nonzero(instance_mask))

            if area < min_mask_px:
                continue

            ys, xs = np.where(instance_mask)
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())
            bw = x2 - x1 + 1
            bh = y2 - y1 + 1

            area_ratio = area / float(img_w * img_h)

            is_small = bool(
                area_ratio <= small_area_ratio
                and max(bw, bh) <= small_max_side_px
            )

            if not keep_all and not is_small:
                continue

            depth_values = depth_m[instance_mask]
            depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0.0)]

            if depth_values.size > 0:
                depth_stats = {
                    "mean": float(np.mean(depth_values)),
                    "median": float(np.median(depth_values)),
                    "min": float(np.min(depth_values)),
                    "max": float(np.max(depth_values))
                }
            else:
                depth_stats = {
                    "mean": None,
                    "median": None,
                    "min": None,
                    "max": None
                }

            annotation = {
                "id": ann_id,
                "track_id": f"{target.name}_{int(inst_id)}",
                "class_id": int(target.class_id),
                "class_name": target.name,
                "semantic_ids": target.semantic_ids,
                "carla_instance_id": int(inst_id),
                "bbox_xywh": [x1, y1, bw, bh],
                "bbox_xyxy": [x1, y1, x2, y2],
                "area_px": int(area),
                "area_ratio": float(area_ratio),
                "small_target": is_small,
                "depth_m": depth_stats
            }

            annotations.append(annotation)
            ann_id += 1

    return annotations, semantic_id, instance_id


def is_actor_target_class(target: TargetClass) -> bool:
    name = target.name.lower()
    return (
        name in ("vehicle", "car", "truck", "bus", "pedestrian", "person", "walker")
        or 10 in target.semantic_ids
        or 4 in target.semantic_ids
    )


def target_for_actor(
    actor: carla.Actor,
    targets: List[TargetClass]
) -> Optional[TargetClass]:
    """
    根据 actor.type_id 将 OpenHUTB/CARLA actor 映射到数据集类别。
    """
    type_id = actor.type_id

    if type_id.startswith("vehicle."):
        for target in targets:
            if target.name.lower() in ("vehicle", "car", "truck", "bus"):
                return target
            if 10 in target.semantic_ids:
                return target

    if type_id.startswith("walker.pedestrian."):
        for target in targets:
            if target.name.lower() in ("pedestrian", "person", "walker"):
                return target
            if 4 in target.semantic_ids:
                return target

    return None


def project_world_location_to_image(
    location: carla.Location,
    world_to_camera: np.ndarray,
    width: int,
    height: int,
    fov: float
) -> Optional[Tuple[float, float, float]]:
    """
    将世界坐标点投影到 RGB 图像。

    返回 (u, v, depth)，其中 depth 是相机前向距离。
    """
    point = np.array(
        [location.x, location.y, location.z, 1.0],
        dtype=np.float64
    )
    point_camera = world_to_camera @ point

    depth = float(point_camera[0])
    if depth <= 0.05:
        return None

    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    u = width / 2.0 + focal * float(point_camera[1]) / depth
    v = height / 2.0 - focal * float(point_camera[2]) / depth

    return u, v, depth


def build_annotations_from_actors(
    world: carla.World,
    camera_transform: carla.Transform,
    depth_m: np.ndarray,
    semantic_id: np.ndarray,
    targets: List[TargetClass],
    width: int,
    height: int,
    fov: float,
    min_mask_px: int,
    small_area_ratio: float,
    small_max_side_px: int,
    keep_all: bool,
    min_actor_visible_px: int,
    min_actor_visible_ratio: float,
    actor_depth_margin: float,
    actor_visibility_mode: str
) -> List[Dict[str, Any]]:
    """
    使用 OpenHUTB/CARLA actor 坐标接口生成车辆/行人 bbox。

    actor 3D bbox 只作为候选区域。默认使用深度过滤遮挡，
    避免 OpenHUTB 语义图漏标车辆时把所有车过滤掉。
    """
    world_to_camera = np.array(
        camera_transform.get_inverse_matrix(),
        dtype=np.float64
    )
    annotations: List[Dict[str, Any]] = []

    actors = list(world.get_actors().filter("vehicle.*"))
    actors.extend(list(world.get_actors().filter("walker.pedestrian.*")))

    for actor in actors:
        target = target_for_actor(actor, targets)
        if target is None:
            continue

        try:
            vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
        except Exception:
            continue

        projected = [
            p
            for p in (
                project_world_location_to_image(
                    vertex,
                    world_to_camera,
                    width,
                    height,
                    fov
                )
                for vertex in vertices
            )
            if p is not None
        ]

        if len(projected) == 0:
            continue

        us = [p[0] for p in projected]
        vs = [p[1] for p in projected]
        depths = [p[2] for p in projected]

        x1 = max(0, int(math.floor(min(us))))
        y1 = max(0, int(math.floor(min(vs))))
        x2 = min(width - 1, int(math.ceil(max(us))))
        y2 = min(height - 1, int(math.ceil(max(vs))))

        if x2 <= x1 or y2 <= y1:
            continue

        projected_bw = x2 - x1 + 1
        projected_bh = y2 - y1 + 1
        projected_area = int(projected_bw * projected_bh)

        if projected_area < min_mask_px:
            continue

        sem_crop = semantic_id[y1:y2 + 1, x1:x2 + 1]
        depth_crop = depth_m[y1:y2 + 1, x1:x2 + 1]
        semantic_match = np.isin(
            sem_crop,
            np.array(target.semantic_ids, dtype=np.uint8)
        )

        actor_depth_min = max(0.0, float(np.min(depths)) - actor_depth_margin)
        actor_depth_max = float(np.max(depths)) + actor_depth_margin
        depth_visible = (
            np.isfinite(depth_crop)
            & (depth_crop > 0.0)
            & (depth_crop >= actor_depth_min)
            & (depth_crop <= actor_depth_max)
        )

        if actor_visibility_mode == "semantic_depth":
            visible_mask = semantic_match & depth_visible
        elif actor_visibility_mode == "depth":
            visible_mask = depth_visible
        else:
            visible_mask = np.ones(depth_crop.shape, dtype=bool)

        visible_px = int(np.count_nonzero(visible_mask))
        visible_ratio_projected = visible_px / float(projected_area)

        if (
            visible_px < min_actor_visible_px
            or visible_ratio_projected < min_actor_visible_ratio
        ):
            continue

        rel_ys, rel_xs = np.where(visible_mask)
        x1_visible = x1 + int(rel_xs.min())
        y1_visible = y1 + int(rel_ys.min())
        x2_visible = x1 + int(rel_xs.max())
        y2_visible = y1 + int(rel_ys.max())

        bw = x2_visible - x1_visible + 1
        bh = y2_visible - y1_visible + 1
        area = visible_px
        bbox_area = int(bw * bh)
        area_ratio = area / float(width * height)
        is_small = bool(
            area_ratio <= small_area_ratio
            and max(bw, bh) <= small_max_side_px
        )

        if not keep_all and not is_small:
            continue

        depth_values = depth_crop[visible_mask]
        depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0.0)]

        if depth_values.size > 0:
            depth_stats = {
                "mean": float(np.mean(depth_values)),
                "median": float(np.median(depth_values)),
                "min": float(np.min(depth_values)),
                "max": float(np.max(depth_values))
            }
        else:
            depth_stats = {
                "mean": float(np.mean(depths)),
                "median": float(np.median(depths)),
                "min": float(np.min(depths)),
                "max": float(np.max(depths))
            }

        annotations.append({
            "id": len(annotations),
            "track_id": f"{target.name}_actor_{actor.id}",
            "class_id": int(target.class_id),
            "class_name": target.name,
            "semantic_ids": target.semantic_ids,
            "carla_actor_id": int(actor.id),
            "carla_instance_id": int(actor.id),
            "actor_type_id": actor.type_id,
            "annotation_source": "actor_visible_pixels",
            "bbox_xywh": [x1_visible, y1_visible, bw, bh],
            "bbox_xyxy": [x1_visible, y1_visible, x2_visible, y2_visible],
            "projected_bbox_xywh": [x1, y1, projected_bw, projected_bh],
            "projected_bbox_xyxy": [x1, y1, x2, y2],
            "area_px": area,
            "bbox_area_px": bbox_area,
            "area_ratio": float(area_ratio),
            "visible_px": visible_px,
            "visible_ratio_projected_bbox": float(visible_ratio_projected),
            "visibility_depth_range_m": [actor_depth_min, actor_depth_max],
            "actor_visibility_mode": actor_visibility_mode,
            "small_target": is_small,
            "depth_m": depth_stats
        })

    return annotations


def combine_actor_and_instance_annotations(
    actor_anns: List[Dict[str, Any]],
    instance_anns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    annotations = actor_anns + instance_anns
    for ann_id, ann in enumerate(annotations):
        ann["id"] = ann_id
        ann.setdefault("annotation_source", "instance_segmentation")
    return annotations


def save_yolo_label(
    path: Path,
    anns: List[Dict[str, Any]],
    width: int,
    height: int
) -> None:
    """
    保存 YOLO 格式标签：

        class_id cx cy w h

    坐标均归一化到 0~1。
    """
    lines = []

    for ann in anns:
        x, y, w, h = ann["bbox_xywh"]

        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        nw = w / width
        nh = h / height

        lines.append(
            f"{ann['class_id']} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def save_annotation_modalities(
    mask_dir: Path,
    object_depth_npy_dir: Path,
    object_depth_vis_dir: Path,
    object_disparity_vis_dir: Path,
    frame_stem: str,
    anns: List[Dict[str, Any]],
    semantic_id: np.ndarray,
    instance_id: np.ndarray,
    depth_m: np.ndarray,
    max_depth_vis_m: float,
    min_disparity_depth_m: float,
    max_disparity_depth_m: float
) -> None:
    """
    保存每条 annotation 对应的 mask、深度 crop 和 normalized disparity crop。
    """
    for ann in anns:
        x, y, w, h = ann["bbox_xywh"]

        if ann.get("annotation_source") == "actor_visible_pixels":
            mode = ann.get("actor_visibility_mode", "depth")
            depth_min, depth_max = ann["visibility_depth_range_m"]
            px, py, pw, ph = ann.get("projected_bbox_xywh", ann["bbox_xywh"])
            mask = np.zeros(depth_m.shape, dtype=bool)
            projected_depth = depth_m[py:py + ph, px:px + pw]
            depth_mask = (
                np.isfinite(projected_depth)
                & (projected_depth > 0.0)
                & (projected_depth >= float(depth_min))
                & (projected_depth <= float(depth_max))
            )
            if mode == "semantic_depth":
                sem_ids = np.array(ann["semantic_ids"], dtype=np.uint8)
                projected_semantic = semantic_id[py:py + ph, px:px + pw]
                projected_mask = (
                    np.isin(projected_semantic, sem_ids)
                    & depth_mask
                )
                ann["mask_type"] = "actor_visible_semantic_depth_pixels"
            elif mode == "depth":
                projected_mask = depth_mask
                ann["mask_type"] = "actor_visible_depth_pixels"
            else:
                projected_mask = np.ones(projected_depth.shape, dtype=bool)
                ann["mask_type"] = "actor_projected_bbox_pixels"

            mask[py:py + ph, px:px + pw] = projected_mask
        elif ann.get("annotation_source") == "actor_bbox":
            mask = np.zeros(depth_m.shape, dtype=bool)
            mask[y:y + h, x:x + w] = True
            ann["mask_type"] = "bbox_rectangle"
        else:
            inst_id = int(ann["carla_instance_id"])
            sem_ids = np.array(ann["semantic_ids"], dtype=np.uint8)
            mask = (instance_id == inst_id) & np.isin(semantic_id, sem_ids)
            ann["mask_type"] = "instance_pixels"

        mask_u8 = (mask.astype(np.uint8) * 255)

        mask_path = mask_dir / f"{frame_stem}_ann{ann['id']:03d}.png"
        cv2.imwrite(str(mask_path), mask_u8)

        depth_crop = depth_m[y:y + h, x:x + w].astype(np.float32)
        mask_crop = mask[y:y + h, x:x + w]
        depth_crop_masked = np.where(mask_crop, depth_crop, np.nan).astype(np.float32)

        depth_npy_path = object_depth_npy_dir / f"{frame_stem}_ann{ann['id']:03d}.npy"
        depth_vis_path = object_depth_vis_dir / f"{frame_stem}_ann{ann['id']:03d}.png"
        disparity_path = object_disparity_vis_dir / f"{frame_stem}_ann{ann['id']:03d}.png"

        np.save(depth_npy_path, depth_crop_masked)
        cv2.imwrite(str(depth_vis_path), metric_depth_to_u16(depth_crop_masked, max_depth_vis_m))
        cv2.imwrite(
            str(disparity_path),
            depth_to_disparity_u16(
                depth_crop,
                min_disparity_depth_m,
                max_disparity_depth_m,
                valid_mask=mask_crop
            )
        )

        ann["files"] = {
            "mask": str(mask_path),
            "object_depth_npy_meters": str(depth_npy_path),
            "object_depth_vis_16bit": str(depth_vis_path),
            "object_disparity_vis_16bit": str(disparity_path)
        }


# ============================================================
# 8. CARLA 工具函数
# ============================================================

def vector_to_dict(v) -> Dict[str, float]:
    return {
        "x": float(v.x),
        "y": float(v.y),
        "z": float(v.z)
    }


def rotation_to_dict(r) -> Dict[str, float]:
    return {
        "pitch": float(r.pitch),
        "yaw": float(r.yaw),
        "roll": float(r.roll)
    }


def transform_to_dict(t: carla.Transform) -> Dict[str, Any]:
    return {
        "location": vector_to_dict(t.location),
        "rotation": rotation_to_dict(t.rotation)
    }


def try_load_world(client: carla.Client, map_name: Optional[str]) -> carla.World:
    if map_name is None or map_name.strip() == "":
        world = client.get_world()
        print(f"[INFO] Using current world: {world.get_map().name}")
        return world

    print(f"[INFO] Loading map: {map_name}")
    world = client.load_world(map_name)
    print(f"[INFO] Loaded world: {world.get_map().name}")
    return world


def get_vehicle_environment_label():
    """
    不同 CARLA/OpenHUTB 版本的 CityObjectLabel 命名可能不同。
    优先查找车辆类标签，用于隐藏地图自带静态车。
    """
    label_class = getattr(carla, "CityObjectLabel", None)
    if label_class is None:
        return None

    for name in ("Vehicles", "Vehicle", "Cars", "Car"):
        if hasattr(label_class, name):
            return getattr(label_class, name)

    return None


def hide_static_map_vehicles(world: carla.World) -> List[int]:
    """
    隐藏地图环境对象中的静态车辆。

    这不会影响脚本生成的 vehicle.* actor，只影响地图自带环境对象。
    返回被隐藏的环境对象 id，便于脚本退出时恢复。
    """
    if not hasattr(world, "get_environment_objects") or not hasattr(world, "enable_environment_objects"):
        print("[WARN] This OpenHUTB/CARLA API has no environment object controls.")
        return []

    vehicle_label = get_vehicle_environment_label()
    if vehicle_label is None:
        print("[WARN] carla.CityObjectLabel has no vehicle label; static map vehicles were not hidden.")
        return []

    try:
        env_objects = world.get_environment_objects(vehicle_label)
        env_ids = [obj.id for obj in env_objects]
        if len(env_ids) == 0:
            print("[INFO] No static map vehicle environment objects found.")
            return []

        world.enable_environment_objects(set(env_ids), False)
        print(f"[INFO] Hidden static map vehicle environment objects: {len(env_ids)}")
        return env_ids

    except Exception as exc:
        print(f"[WARN] Failed to hide static map vehicles: {exc}")
        return []


def restore_static_map_vehicles(world: carla.World, env_ids: List[int]) -> None:
    if len(env_ids) == 0:
        return

    if not hasattr(world, "enable_environment_objects"):
        return

    try:
        world.enable_environment_objects(set(env_ids), True)
        print(f"[INFO] Restored static map vehicle environment objects: {len(env_ids)}")
    except Exception as exc:
        print(f"[WARN] Failed to restore static map vehicles: {exc}")


def apply_weather(world: carla.World, preset_name: Optional[str]) -> str:
    if preset_name is None:
        print("[INFO] Keeping current simulator weather.")
        return "current"

    preset_name = str(preset_name).strip()
    if preset_name.lower() in ("", "current", "keep", "keep_current"):
        print("[INFO] Keeping current simulator weather.")
        return "current"

    presets = {
        "ClearNoon": carla.WeatherParameters.ClearNoon,
        "CloudyNoon": carla.WeatherParameters.CloudyNoon,
        "WetNoon": carla.WeatherParameters.WetNoon,
        "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
        "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
        "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
        "HardRainNoon": carla.WeatherParameters.HardRainNoon,
        "ClearSunset": carla.WeatherParameters.ClearSunset,
        "CloudySunset": carla.WeatherParameters.CloudySunset,
        "WetSunset": carla.WeatherParameters.WetSunset,
        "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
        "MidRainSunset": carla.WeatherParameters.MidRainSunset,
        "HardRainSunset": carla.WeatherParameters.HardRainSunset
    }

    if preset_name not in presets:
        print(f"[WARN] Unknown weather preset '{preset_name}', fallback to ClearNoon.")
        preset_name = "ClearNoon"

    weather = presets[preset_name]
    world.set_weather(weather)
    print(f"[INFO] Applied weather: {preset_name}")
    return preset_name


def get_weather_names() -> List[str]:
    return [
        "ClearNoon",
        "CloudyNoon",
        "WetNoon",
        "WetCloudyNoon",
        "SoftRainNoon",
        "MidRainyNoon",
        "HardRainNoon",
        "ClearSunset",
        "CloudySunset",
        "WetSunset",
        "SoftRainSunset",
        "MidRainSunset",
        "HardRainSunset"
    ]


def setup_camera_blueprint(
    blueprint_library,
    sensor_type: str,
    width: int,
    height: int,
    fov: float,
    sensor_tick: float,
    enable_rgb_postprocess: bool = False
):
    bp = blueprint_library.find(sensor_type)

    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))

    # sensor_tick=0.0 表示每个 simulation tick 都输出一次。
    if bp.has_attribute("sensor_tick"):
        bp.set_attribute("sensor_tick", str(sensor_tick))

    if sensor_type == "sensor.camera.rgb":
        if bp.has_attribute("enable_postprocess_effects"):
            bp.set_attribute(
                "enable_postprocess_effects",
                "true" if enable_rgb_postprocess else "false"
            )
        if not enable_rgb_postprocess:
            for attr_name in (
                "motion_blur_intensity",
                "motion_blur_max_distortion",
                "motion_blur_min_object_screen_size"
            ):
                if bp.has_attribute(attr_name):
                    bp.set_attribute(attr_name, "0.0")

    return bp


def spawn_camera_set(
    world: carla.World,
    width: int,
    height: int,
    fov: float,
    sensor_tick: float,
    initial_transform: carla.Transform,
    enable_rgb_postprocess: bool
) -> Dict[str, carla.Sensor]:
    """
    创建 RGB / depth / semantic / instance 四个相机。

    注意：
    这里不是 attach 到车辆，而是独立 spawn 到空中，
    后面每帧用 set_transform 模拟 UAV 运动。
    """
    blueprint_library = world.get_blueprint_library()

    sensor_types = {
        "rgb": "sensor.camera.rgb",
        "depth": "sensor.camera.depth",
        "semantic": "sensor.camera.semantic_segmentation",
        "instance": "sensor.camera.instance_segmentation"
    }

    sensors: Dict[str, carla.Sensor] = {}

    for name, sensor_type in sensor_types.items():
        print(f"[INFO] Spawning sensor: {name} -> {sensor_type}")
        bp = setup_camera_blueprint(
            blueprint_library,
            sensor_type,
            width,
            height,
            fov,
            sensor_tick,
            enable_rgb_postprocess=enable_rgb_postprocess
        )
        sensor = world.spawn_actor(bp, initial_transform)
        sensors[name] = sensor

    return sensors


def spawn_optional_sensors(
    world: carla.World,
    sensor_tick: float,
    initial_transform: carla.Transform,
    enable_lidar: bool
) -> Dict[str, carla.Sensor]:
    """
    创建 IMU / GNSS / 可选 LiDAR。
    """
    blueprint_library = world.get_blueprint_library()
    sensors: Dict[str, carla.Sensor] = {}

    try:
        imu_bp = blueprint_library.find("sensor.other.imu")
        if imu_bp.has_attribute("sensor_tick"):
            imu_bp.set_attribute("sensor_tick", str(sensor_tick))
        sensors["imu"] = world.spawn_actor(imu_bp, initial_transform)
        print("[INFO] Spawning sensor: imu")
    except Exception as exc:
        print(f"[WARN] IMU unavailable: {exc}")

    try:
        gnss_bp = blueprint_library.find("sensor.other.gnss")
        if gnss_bp.has_attribute("sensor_tick"):
            gnss_bp.set_attribute("sensor_tick", str(sensor_tick))
        sensors["gnss"] = world.spawn_actor(gnss_bp, initial_transform)
        print("[INFO] Spawning sensor: gnss")
    except Exception as exc:
        print(f"[WARN] GNSS unavailable: {exc}")

    if enable_lidar:
        try:
            lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("channels", "32")
            lidar_bp.set_attribute("range", "150")
            lidar_bp.set_attribute("points_per_second", "56000")
            lidar_bp.set_attribute("rotation_frequency", "20")

            if lidar_bp.has_attribute("sensor_tick"):
                lidar_bp.set_attribute("sensor_tick", str(sensor_tick))

            sensors["lidar"] = world.spawn_actor(lidar_bp, initial_transform)
            print("[INFO] Spawning sensor: lidar")
        except Exception as exc:
            print(f"[WARN] LiDAR unavailable: {exc}")

    return sensors


def set_all_sensor_transform(
    sensors: Dict[str, carla.Sensor],
    transform: carla.Transform
) -> None:
    for sensor in sensors.values():
        sensor.set_transform(transform)


def uav_orbit_transform(
    center_x: float,
    center_y: float,
    height: float,
    radius: float,
    theta: float,
    pitch: float
) -> carla.Transform:
    """
    生成环绕式 UAV 视角。

    CARLA 坐标：
    - z 向上；
    - pitch 负数表示相机向下看。
    """
    x = center_x + radius * math.cos(theta)
    y = center_y + radius * math.sin(theta)
    z = height

    yaw = math.degrees(math.atan2(center_y - y, center_x - x))

    location = carla.Location(x=x, y=y, z=z)
    rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)

    return carla.Transform(location, rotation)


def random_uav_transform(
    center_x: float,
    center_y: float,
    height_min: float,
    height_max: float,
    radius_min: float,
    radius_max: float,
    pitch_min: float,
    pitch_max: float
) -> carla.Transform:
    height = random.uniform(height_min, height_max)
    radius = random.uniform(radius_min, radius_max)
    theta = random.uniform(0.0, 2.0 * math.pi)
    pitch = random.uniform(pitch_min, pitch_max)

    return uav_orbit_transform(
        center_x=center_x,
        center_y=center_y,
        height=height,
        radius=radius,
        theta=theta,
        pitch=pitch
    )


def random_road_center(carla_map) -> Tuple[float, float]:
    spawn_points = carla_map.get_spawn_points()
    if len(spawn_points) == 0:
        return 0.0, 0.0

    sp = random.choice(spawn_points)
    return float(sp.location.x), float(sp.location.y)


def spawn_background_traffic(
    client: carla.Client,
    world: carla.World,
    number_of_vehicles: int,
    number_of_walkers: int,
    tm_port: int,
    seed: int
) -> Tuple[List[carla.Actor], List[carla.Actor], List[carla.Actor]]:
    """
    生成背景车辆和行人。

    返回：
    - vehicles
    - walkers
    - walker_controllers

    注意：
    walkers 如果没有 controller，默认不会自己走。
    这里给行人创建 controller，让场景更丰富。
    """
    random.seed(seed)

    blueprint_library = world.get_blueprint_library()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(seed)

    vehicles: List[carla.Actor] = []
    walkers: List[carla.Actor] = []
    walker_controllers: List[carla.Actor] = []

    vehicle_bps = blueprint_library.filter("vehicle.*")
    random.shuffle(spawn_points)

    for sp in spawn_points[:number_of_vehicles]:
        bp = random.choice(vehicle_bps)

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", random.choice(colors))

        actor = world.try_spawn_actor(bp, sp)

        if actor is not None:
            actor.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(actor)

    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    controller_bp = blueprint_library.find("controller.ai.walker")

    for _ in range(number_of_walkers):
        loc = world.get_random_location_from_navigation()

        if loc is None:
            continue

        transform = carla.Transform(loc)
        walker_bp = random.choice(walker_bps)

        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(walker_bp, transform)

        if walker is not None:
            walkers.append(walker)

    # 给行人创建控制器
    for walker in walkers:
        controller = world.try_spawn_actor(
            controller_bp,
            carla.Transform(),
            attach_to=walker
        )

        if controller is not None:
            walker_controllers.append(controller)

    # tick 一下，保证 controller 生效
    world.tick()

    for controller in walker_controllers:
        try:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(random.uniform(0.8, 1.6))
        except Exception:
            pass

    print(
        f"[INFO] Spawned traffic: vehicles={len(vehicles)}, "
        f"walkers={len(walkers)}, walker_controllers={len(walker_controllers)}"
    )

    return vehicles, walkers, walker_controllers


def spawn_target_actors(
    world: carla.World,
    filters: List[str],
    count: int,
    center_x: float,
    center_y: float,
    radius_min: float,
    radius_max: float,
    z_offset: float,
    seed: int
) -> List[carla.Actor]:
    """
    可选生成自定义小目标 actor。

    目标是否能在 semantic / instance 中被标为 rpg、missile 等类别，
    取决于资产在 OpenHUTB/CARLA 中的语义标签设置。
    """
    if count <= 0 or len(filters) == 0:
        return []

    random.seed(seed + 1009)
    blueprint_library = world.get_blueprint_library()
    carla_map = world.get_map()

    blueprints = []
    for pattern in filters:
        blueprints.extend(list(blueprint_library.filter(pattern)))

    if len(blueprints) == 0:
        print(f"[WARN] No target actor blueprints matched: {filters}")
        return []

    actors: List[carla.Actor] = []

    for _ in range(count):
        bp = random.choice(blueprints)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "small_target")

        theta = random.uniform(0.0, 2.0 * math.pi)
        radius = random.uniform(radius_min, radius_max)
        probe_loc = carla.Location(
            x=center_x + radius * math.cos(theta),
            y=center_y + radius * math.sin(theta),
            z=0.0
        )

        waypoint = carla_map.get_waypoint(
            probe_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is not None:
            transform = waypoint.transform
            transform.location.z += z_offset
        else:
            transform = carla.Transform(
                carla.Location(x=probe_loc.x, y=probe_loc.y, z=z_offset)
            )

        actor = world.try_spawn_actor(bp, transform)
        if actor is not None:
            actors.append(actor)

    print(
        f"[INFO] Spawned target actors: count={len(actors)}, "
        f"filters={filters}"
    )
    return actors


def save_lidar_np(lidar_data, path: Path) -> None:
    arr = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    arr = arr.reshape((-1, 4))
    np.save(path, arr)


def safe_get_optional_sensor(
    sync: Dict[str, SensorSync],
    name: str,
    frame: int,
    timeout: float
):
    if name not in sync:
        return None

    try:
        return sync[name].get(frame, timeout=timeout)
    except Exception as exc:
        print(f"[WARN] Optional sensor '{name}' failed at frame {frame}: {exc}")
        return None


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[INFO] Python executable:", sys.executable)
    print("[INFO] CARLA module:", carla.__file__)

    if args.target is None:
        args.target = [
            TargetClass("vehicle", 0, [10]),
            TargetClass("pedestrian", 1, [4]),
            TargetClass("traffic_sign", 2, [12]),
            TargetClass("traffic_light", 3, [18])
        ]

    if args.pitch_min is None:
        args.pitch_min = args.pitch
    if args.pitch_max is None:
        args.pitch_max = args.pitch
    if args.pitch_min > args.pitch_max:
        args.pitch_min, args.pitch_max = args.pitch_max, args.pitch_min

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = try_load_world(client, args.map)

    original_settings = world.get_settings()

    fixed_delta_seconds = 1.0 / args.fps

    # 这些 actor 后面 cleanup 要销毁
    traffic_actors: List[carla.Actor] = []
    walker_actors: List[carla.Actor] = []
    walker_controllers: List[carla.Actor] = []
    sensors: Dict[str, carla.Sensor] = {}
    hidden_static_vehicle_ids: List[int] = []

    try:
        # ------------------------------------------------------------
        # 设置同步模式
        # ------------------------------------------------------------
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        world.apply_settings(settings)

        print("[INFO] Applied synchronous mode.")
        print(f"[INFO] fixed_delta_seconds={fixed_delta_seconds}")

        # ------------------------------------------------------------
        # 隐藏地图自带静态车辆，只保留脚本生成的 vehicle actor。
        # ------------------------------------------------------------
        if args.hide_static_map_vehicles:
            hidden_static_vehicle_ids = hide_static_map_vehicles(world)

        # ------------------------------------------------------------
        # 天气
        # ------------------------------------------------------------
        weather_names = get_weather_names()

        # ------------------------------------------------------------
        # 生成背景交通
        # ------------------------------------------------------------
        if not args.no_traffic:
            vehicles, walkers, controllers = spawn_background_traffic(
                client=client,
                world=world,
                number_of_vehicles=args.vehicles,
                number_of_walkers=args.walkers,
                tm_port=args.tm_port,
                seed=args.seed
            )

            traffic_actors.extend(vehicles)
            walker_actors.extend(walkers)
            walker_controllers.extend(controllers)

        target_actors = spawn_target_actors(
            world=world,
            filters=args.target_actor_filter,
            count=args.target_actor_count,
            center_x=args.center_x,
            center_y=args.center_y,
            radius_min=args.target_spawn_radius_min,
            radius_max=args.target_spawn_radius_max,
            z_offset=args.target_spawn_z_offset,
            seed=args.seed
        )
        traffic_actors.extend(target_actors)

        # ------------------------------------------------------------
        # 创建无人机视角传感器
        # ------------------------------------------------------------
        initial_transform = carla.Transform(
            carla.Location(
                x=args.center_x,
                y=args.center_y,
                z=args.height
            ),
            carla.Rotation(
                pitch=args.pitch,
                yaw=0.0,
                roll=0.0
            )
        )

        # 关键修复点：
        # sensor_tick = 0.0 表示每个 world.tick 都输出。
        sensor_tick = 0.0

        camera_sensors = spawn_camera_set(
            world=world,
            width=args.width,
            height=args.height_img,
            fov=args.fov,
            sensor_tick=sensor_tick,
            initial_transform=initial_transform,
            enable_rgb_postprocess=args.enable_rgb_postprocess
        )

        optional_sensors = spawn_optional_sensors(
            world=world,
            sensor_tick=sensor_tick,
            initial_transform=initial_transform,
            enable_lidar=args.lidar
        )

        sensors = {**camera_sensors, **optional_sensors}

        sync: Dict[str, SensorSync] = {
            name: SensorSync(name, sensor)
            for name, sensor in sensors.items()
        }

        # ------------------------------------------------------------
        # 传感器 warm-up
        # ------------------------------------------------------------
        print(f"[INFO] Warming up sensors for {args.warmup_frames} frames ...")

        for _ in range(args.warmup_frames):
            world.tick()
            time.sleep(0.01)

        for sensor_sync in sync.values():
            sensor_sync.drain()

        print("[INFO] Sensor warm-up finished.")

        # ------------------------------------------------------------
        # 数据集总 manifest
        # ------------------------------------------------------------
        manifest = {
            "dataset": "OpenHUTB-CARLA-UAV-Small-Multimodal",
            "carla_module": carla.__file__,
            "host": args.host,
            "port": args.port,
            "map": world.get_map().name,
            "image_size": [args.width, args.height_img],
            "fov": args.fov,
            "rgb_postprocess": args.enable_rgb_postprocess,
            "fps": args.fps,
            "targets": [
                {
                    "name": t.name,
                    "class_id": t.class_id,
                    "semantic_ids": t.semantic_ids
                }
                for t in args.target
            ],
            "small_target_rule": {
                "min_mask_px": args.min_mask_px,
                "small_area_ratio": args.small_area_ratio,
                "small_max_side_px": args.small_max_side_px,
                "keep_all": args.keep_all
            },
            "annotation_policy": {
                "source": args.annotation_source,
                "actor_classes": "vehicle.* and walker.pedestrian.* use OpenHUTB/CARLA actor coordinates plus semantic/depth visibility filtering in hybrid/actor mode",
                "instance_classes": "non-actor targets use instance segmentation in hybrid/instance mode",
                "object_unit": "one actor bbox or one visible semantic_id + carla_instance_id per annotation",
                "bbox": "visible actor pixels for vehicles/pedestrians, tight instance mask bbox for other targets",
                "mask": "visible semantic/depth pixels for actor annotations, instance pixels for instance annotations",
                "actor_visibility": {
                    "mode": args.actor_visibility_mode,
                    "min_actor_visible_px": args.min_actor_visible_px,
                    "min_actor_visible_ratio": args.min_actor_visible_ratio,
                    "actor_depth_margin": args.actor_depth_margin
                },
                "depth": "metric full-frame depth plus per-object masked depth crop",
                "disparity": "normalized inverse-depth visualization"
            },
            "target_actor_generation": {
                "filters": args.target_actor_filter,
                "count": args.target_actor_count,
                "spawn_radius": [
                    args.target_spawn_radius_min,
                    args.target_spawn_radius_max
                ],
                "z_offset": args.target_spawn_z_offset
            },
            "camera_safety": {
                "road_centered_camera": args.road_centered_camera,
                "safe_camera_min_z": args.safe_camera_min_z,
                "max_camera_pose_retries": args.max_camera_pose_retries,
                "min_near_depth_m": args.min_near_depth_m,
                "max_near_depth_ratio": args.max_near_depth_ratio,
                "min_road_visible_ratio": args.min_road_visible_ratio,
                "road_semantic_ids": args.road_semantic_ids
            },
            "static_map_vehicle_filter": {
                "hide_static_map_vehicles": args.hide_static_map_vehicles,
                "hidden_environment_object_count": len(hidden_static_vehicle_ids)
            },
            "weather_policy": {
                "keep_current_weather": args.keep_current_weather,
                "random_weather": args.random_weather,
                "configured_weather": args.weather
            },
            "modalities": [
                "rgb",
                "depth_meters",
                "normalized_disparity",
                "semantic_segmentation",
                "instance_segmentation",
                "object_masks",
                "object_depth_crops",
                "object_disparity_crops",
                "imu_optional",
                "gnss_optional",
                "lidar_optional"
            ],
            "sequences": []
        }

        # ------------------------------------------------------------
        # sequence 循环
        # ------------------------------------------------------------
        total_drop_frames = 0
        total_saved_frames = 0

        for seq_idx in range(args.sequences):
            seq_name = f"seq_{seq_idx:04d}"
            seq_dir = out_root / seq_name
            dirs = make_dirs(seq_dir)

            if args.keep_current_weather:
                weather_name = "current"
            elif args.random_weather:
                weather_name = random.choice(weather_names)
            else:
                weather_name = args.weather

            weather_name = apply_weather(world, weather_name)

            orbit_phase = random.uniform(0.0, 2.0 * math.pi)
            orbit_radius = random.uniform(args.radius_min, args.radius_max)
            orbit_height = args.height
            if args.road_centered_camera:
                base_center_x, base_center_y = random_road_center(world.get_map())
            else:
                base_center_x, base_center_y = args.center_x, args.center_y

            seq_meta = {
                "sequence": seq_name,
                "weather": weather_name,
                "route": args.route,
                "center_xy": [base_center_x, base_center_y],
                "road_centered_camera": args.road_centered_camera,
                "min_road_visible_ratio": args.min_road_visible_ratio,
                "road_semantic_ids": args.road_semantic_ids,
                "height": args.height,
                "height_range": [args.height_min, args.height_max],
                "radius": orbit_radius,
                "radius_range": [args.radius_min, args.radius_max],
                "pitch": args.pitch,
                "pitch_range": [args.pitch_min, args.pitch_max]
            }

            manifest["sequences"].append(seq_meta)

            (seq_dir / "sequence_meta.json").write_text(
                json.dumps(seq_meta, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            gt_path = seq_dir / "groundtruth.txt"
            multi_gt_path = seq_dir / "groundtruth_multi.csv"
            index_path = seq_dir / "frame_index.csv"

            with open(gt_path, "w", encoding="utf-8") as gt_file, \
                 open(multi_gt_path, "w", newline="", encoding="utf-8") as multi_file, \
                 open(index_path, "w", newline="", encoding="utf-8") as index_file:

                multi_writer = csv.writer(multi_file)
                multi_writer.writerow([
                    "frame_id",
                    "carla_frame",
                    "ann_id",
                    "class_id",
                    "class_name",
                    "track_id",
                    "annotation_source",
                    "carla_actor_id",
                    "carla_instance_id",
                    "x",
                    "y",
                    "w",
                    "h",
                    "area_px",
                    "area_ratio",
                    "small_target",
                    "depth_mean",
                    "depth_median",
                    "depth_min",
                    "depth_max"
                ])

                index_writer = csv.writer(index_file)
                index_writer.writerow([
                    "frame_id",
                    "carla_frame",
                    "rgb",
                    "depth_npy",
                    "depth_vis",
                    "disparity_vis",
                    "semantic",
                    "instance",
                    "num_annotations"
                ])

                sot_candidates: List[List[Dict[str, Any]]] = []

                for frame_i in range(args.frames):
                    # ------------------------------------------------
                    # 生成 UAV 相机位姿，并过滤穿模/近距离遮挡视角。
                    # ------------------------------------------------
                    accepted_view = False
                    last_bad_view_stats: Optional[Dict[str, Any]] = None
                    best_view_stats: Optional[Dict[str, Any]] = None

                    for pose_try in range(max(1, args.max_camera_pose_retries)):
                        if args.route == "orbit" and pose_try == 0:
                            theta = orbit_phase + math.radians(args.orbit_degrees_per_sequence) * frame_i / max(args.frames, 1)
                            transform = uav_orbit_transform(
                                center_x=base_center_x,
                                center_y=base_center_y,
                                height=max(orbit_height, args.safe_camera_min_z),
                                radius=max(orbit_radius, args.radius_min),
                                theta=theta,
                                pitch=args.pitch
                            )
                        else:
                            if args.road_centered_camera:
                                center_x, center_y = random_road_center(world.get_map())
                            else:
                                center_x, center_y = args.center_x, args.center_y

                            transform = random_uav_transform(
                                center_x=center_x,
                                center_y=center_y,
                                height_min=max(args.height_min, args.safe_camera_min_z),
                                height_max=max(args.height_max, args.safe_camera_min_z),
                                radius_min=args.radius_min,
                                radius_max=args.radius_max,
                                pitch_min=args.pitch_min,
                                pitch_max=args.pitch_max
                            )

                        set_all_sensor_transform(sensors, transform)
                        carla_frame = world.tick()

                        try:
                            rgb_img = sync["rgb"].get(
                                carla_frame,
                                timeout=args.sensor_timeout
                            )
                            depth_img = sync["depth"].get(
                                carla_frame,
                                timeout=args.sensor_timeout
                            )
                            semantic_img = sync["semantic"].get(
                                carla_frame,
                                timeout=args.sensor_timeout
                            )
                            instance_img = sync["instance"].get(
                                carla_frame,
                                timeout=args.sensor_timeout
                            )

                        except TimeoutError as exc:
                            total_drop_frames += 1
                            print(
                                f"[WARN] Drop frame_i={frame_i}, "
                                f"carla_frame={carla_frame}: {exc}"
                            )

                            if total_drop_frames >= args.max_drop_frames:
                                raise RuntimeError(
                                    f"连续或累计丢帧过多：{total_drop_frames}。"
                                    f"请降低分辨率/减少车辆行人/检查 CARLA API 版本。"
                                )

                            continue

                        depth_m = decode_carla_depth_meters(depth_img)
                        bad_view, bad_view_stats = is_bad_camera_view(
                            depth_m,
                            min_near_depth_m=args.min_near_depth_m,
                            max_near_depth_ratio=args.max_near_depth_ratio
                        )
                        bad_view_stats["road_visible_ratio"] = road_visible_ratio(
                            semantic_img,
                            args.road_semantic_ids
                        )
                        last_bad_view_stats = bad_view_stats
                        candidate_stats = dict(bad_view_stats)
                        candidate_stats["bad_view"] = bool(bad_view)

                        if best_view_stats is None:
                            best_view_stats = candidate_stats
                        else:
                            best_bad = bool(best_view_stats.get("bad_view", True))
                            best_road = float(best_view_stats.get("road_visible_ratio", -1.0))
                            candidate_road = float(candidate_stats.get("road_visible_ratio", -1.0))
                            if (
                                (best_bad and not bad_view)
                                or (best_bad == bool(bad_view) and candidate_road > best_road)
                            ):
                                best_view_stats = candidate_stats

                        if (
                            not bad_view
                            and bad_view_stats["road_visible_ratio"] >= args.min_road_visible_ratio
                        ):
                            accepted_view = True
                            break

                    if not accepted_view:
                        total_drop_frames += 1
                        print(
                            f"[WARN] Drop frame_i={frame_i}: bad camera view after "
                            f"{args.max_camera_pose_retries} retries, "
                            f"last_stats={last_bad_view_stats}, best_stats={best_view_stats}"
                        )

                        if total_drop_frames >= args.max_drop_frames:
                            raise RuntimeError(
                                f"坏视角/丢帧过多：{total_drop_frames}。"
                                f"请提高 safe_camera_min_z 或增大 radius_min。"
                            )

                        continue

                    # ------------------------------------------------
                    # 文件路径
                    # ------------------------------------------------
                    stem = f"{frame_i:06d}"

                    rgb_path = dirs["rgb"] / f"{stem}.png"
                    depth_npy_path = dirs["depth_npy"] / f"{stem}.npy"
                    depth_vis_path = dirs["depth_vis"] / f"{stem}.png"
                    disparity_vis_path = dirs["disparity_vis"] / f"{stem}.png"
                    semantic_path = dirs["semantic"] / f"{stem}.png"
                    instance_path = dirs["instance"] / f"{stem}.png"
                    yolo_path = dirs["yolo"] / f"{stem}.txt"
                    ann_path = dirs["ann"] / f"{stem}.json"

                    # ------------------------------------------------
                    # 保存图像
                    # ------------------------------------------------
                    save_rgb(rgb_img, rgb_path)

                    save_depth(
                        depth_m,
                        depth_npy_path,
                        depth_vis_path,
                        args.max_depth_vis
                    )
                    save_disparity(
                        depth_m,
                        disparity_vis_path,
                        args.min_disparity_depth,
                        args.max_disparity_depth
                    )

                    save_segmentation_raw(semantic_img, semantic_path)
                    save_segmentation_raw(instance_img, instance_path)

                    # ------------------------------------------------
                    # 生成标注
                    # ------------------------------------------------
                    if args.annotation_source == "hybrid":
                        instance_targets = [
                            target for target in args.target
                            if not is_actor_target_class(target)
                        ]
                    elif args.annotation_source == "instance":
                        instance_targets = args.target
                    else:
                        instance_targets = []

                    instance_anns, semantic_id, instance_id = build_annotations_from_instance(
                        instance_image=instance_img,
                        depth_m=depth_m,
                        targets=instance_targets,
                        min_mask_px=args.min_mask_px,
                        small_area_ratio=args.small_area_ratio,
                        small_max_side_px=args.small_max_side_px,
                        keep_all=args.keep_all
                    )

                    if args.annotation_source in ("actor", "hybrid"):
                        actor_anns = build_annotations_from_actors(
                            world=world,
                            camera_transform=transform,
                            depth_m=depth_m,
                            semantic_id=semantic_id,
                            targets=args.target,
                            width=args.width,
                            height=args.height_img,
                            fov=args.fov,
                            min_mask_px=args.min_mask_px,
                            small_area_ratio=args.small_area_ratio,
                            small_max_side_px=args.small_max_side_px,
                            keep_all=args.keep_all,
                            min_actor_visible_px=args.min_actor_visible_px,
                            min_actor_visible_ratio=args.min_actor_visible_ratio,
                            actor_depth_margin=args.actor_depth_margin,
                            actor_visibility_mode=args.actor_visibility_mode
                        )
                    else:
                        actor_anns = []

                    anns = combine_actor_and_instance_annotations(
                        actor_anns=actor_anns,
                        instance_anns=instance_anns
                    )

                    save_yolo_label(
                        yolo_path,
                        anns,
                        width=args.width,
                        height=args.height_img
                    )

                    save_annotation_modalities(
                        mask_dir=dirs["mask"],
                        object_depth_npy_dir=dirs["object_depth_npy"],
                        object_depth_vis_dir=dirs["object_depth_vis"],
                        object_disparity_vis_dir=dirs["object_disparity_vis"],
                        frame_stem=stem,
                        anns=anns,
                        semantic_id=semantic_id,
                        instance_id=instance_id,
                        depth_m=depth_m,
                        max_depth_vis_m=args.max_depth_vis,
                        min_disparity_depth_m=args.min_disparity_depth,
                        max_disparity_depth_m=args.max_disparity_depth
                    )

                    # ------------------------------------------------
                    # 可选传感器
                    # ------------------------------------------------
                    imu_data = None
                    gnss_data = None
                    lidar_path = None

                    imu = safe_get_optional_sensor(
                        sync,
                        "imu",
                        carla_frame,
                        timeout=1.0
                    )

                    if imu is not None:
                        imu_data = {
                            "accelerometer": vector_to_dict(imu.accelerometer),
                            "gyroscope": vector_to_dict(imu.gyroscope),
                            "compass": float(imu.compass)
                        }

                    gnss = safe_get_optional_sensor(
                        sync,
                        "gnss",
                        carla_frame,
                        timeout=1.0
                    )

                    if gnss is not None:
                        gnss_data = {
                            "latitude": float(gnss.latitude),
                            "longitude": float(gnss.longitude),
                            "altitude": float(gnss.altitude)
                        }

                    lidar = safe_get_optional_sensor(
                        sync,
                        "lidar",
                        carla_frame,
                        timeout=1.0
                    )

                    if lidar is not None:
                        lidar_path = dirs["lidar"] / f"{stem}.npy"
                        save_lidar_np(lidar, lidar_path)

                    # ------------------------------------------------
                    # JSON 标注
                    # ------------------------------------------------
                    ann_json = {
                        "sequence": seq_name,
                        "frame_id": frame_i,
                        "carla_frame": int(carla_frame),
                        "timestamp": float(rgb_img.timestamp),
                        "camera_transform": transform_to_dict(transform),
                        "image": {
                            "width": args.width,
                            "height": args.height_img,
                            "rgb": str(rgb_path),
                            "depth_npy_meters": str(depth_npy_path),
                            "depth_vis_16bit": str(depth_vis_path),
                            "disparity_vis_16bit": str(disparity_vis_path),
                            "semantic": str(semantic_path),
                            "instance": str(instance_path),
                            "yolo": str(yolo_path),
                            "lidar": None if lidar_path is None else str(lidar_path)
                        },
                        "annotations": anns,
                        "sensors": {
                            "imu": imu_data,
                            "gnss": gnss_data
                        }
                    }

                    ann_path.write_text(
                        json.dumps(ann_json, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )

                    # ------------------------------------------------
                    # SOT groundtruth.txt 候选
                    # 序列结束后选择出现帧数最多的稳定 track。
                    # ------------------------------------------------
                    sot_candidates.append(anns)

                    # ------------------------------------------------
                    # 多目标 CSV
                    # ------------------------------------------------
                    for ann in anns:
                        x, y, w, h = ann["bbox_xywh"]
                        d = ann["depth_m"]

                        multi_writer.writerow([
                            frame_i,
                            int(carla_frame),
                            ann["id"],
                            ann["class_id"],
                            ann["class_name"],
                            ann["track_id"],
                            ann.get("annotation_source", "instance_segmentation"),
                            ann.get("carla_actor_id", ""),
                            ann["carla_instance_id"],
                            x,
                            y,
                            w,
                            h,
                            ann["area_px"],
                            f"{ann['area_ratio']:.10f}",
                            int(ann["small_target"]),
                            "" if d["mean"] is None else f"{d['mean']:.6f}",
                            "" if d["median"] is None else f"{d['median']:.6f}",
                            "" if d["min"] is None else f"{d['min']:.6f}",
                            "" if d["max"] is None else f"{d['max']:.6f}"
                        ])

                    # ------------------------------------------------
                    # frame_index.csv
                    # ------------------------------------------------
                    index_writer.writerow([
                        frame_i,
                        int(carla_frame),
                        str(rgb_path),
                        str(depth_npy_path),
                        str(depth_vis_path),
                        str(disparity_vis_path),
                        str(semantic_path),
                        str(instance_path),
                        len(anns)
                    ])
                    total_saved_frames += 1

                    if frame_i % 20 == 0:
                        print(
                            f"[{seq_name}] frame={frame_i:06d}, "
                            f"carla_frame={carla_frame}, "
                            f"anns={len(anns)}, "
                            f"weather={weather_name}"
                        )

                track_counts = Counter()
                track_area = Counter()
                for frame_anns in sot_candidates:
                    for ann in frame_anns:
                        track_counts[ann["track_id"]] += 1
                        track_area[ann["track_id"]] += ann["area_px"]

                primary_track_id = None
                if len(track_counts) > 0:
                    primary_track_id = max(
                        track_counts.keys(),
                        key=lambda k: (track_counts[k], track_area[k])
                    )

                for frame_anns in sot_candidates:
                    primary = None
                    if primary_track_id is not None:
                        matches = [
                            ann for ann in frame_anns
                            if ann["track_id"] == primary_track_id
                        ]
                        if len(matches) > 0:
                            primary = matches[0]

                    if primary is not None:
                        x, y, w, h = primary["bbox_xywh"]
                        gt_file.write(f"{x},{y},{w},{h}\n")
                    else:
                        gt_file.write("0,0,0,0\n")

                seq_meta["sot_primary_track_id"] = primary_track_id
                seq_meta["sot_primary_visible_frames"] = (
                    int(track_counts[primary_track_id])
                    if primary_track_id is not None
                    else 0
                )
                seq_meta["sot_primary_policy"] = (
                    "track with most visible frames, tie-broken by total visible area"
                )
                (seq_dir / "sequence_meta.json").write_text(
                    json.dumps(seq_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

        # ------------------------------------------------------------
        # 写总 manifest
        # ------------------------------------------------------------
        if total_saved_frames == 0:
            raise RuntimeError(
                "没有保存任何图像：所有帧都被视角过滤或同步超时丢弃。"
                "请查看上方 WARN 中的 best_stats；通常需要降低 min_road_visible_ratio，"
                "或调整 height/radius/pitch 让相机真正对准路面。"
            )

        manifest["total_drop_frames"] = total_drop_frames
        manifest["total_saved_frames"] = total_saved_frames

        (out_root / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"[DONE] Dataset saved to: {out_root.resolve()}")
        print(f"[DONE] total_saved_frames={total_saved_frames}")
        print(f"[DONE] total_drop_frames={total_drop_frames}")

    finally:
        # ------------------------------------------------------------
        # 清理
        # ------------------------------------------------------------
        print("[CLEANUP] Destroying sensors and restoring settings ...")

        for sensor in sensors.values():
            try:
                sensor.stop()
            except Exception:
                pass

        for sensor in sensors.values():
            try:
                sensor.destroy()
            except Exception:
                pass

        # 先停 walker controller
        for controller in walker_controllers:
            try:
                controller.stop()
            except Exception:
                pass

        # 再销毁 controller / walker / vehicle
        for actor in walker_controllers:
            try:
                actor.destroy()
            except Exception:
                pass

        for actor in walker_actors:
            try:
                actor.destroy()
            except Exception:
                pass

        for actor in traffic_actors:
            try:
                actor.destroy()
            except Exception:
                pass

        restore_static_map_vehicles(world, hidden_static_vehicle_ids)

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("[CLEANUP] Finished.")


if __name__ == "__main__":
    main()
