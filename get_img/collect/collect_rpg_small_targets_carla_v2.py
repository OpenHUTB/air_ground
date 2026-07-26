#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_rpg_small_targets_carla_v2.py

CARLA / OpenHUTB 无人机视角小目标多模态数据集采集脚本。

功能：
1. 连接 CARLA/OpenHUTB 模拟器；
2. 设置同步模式；
3. 创建空中“无人机视角”传感器平台；
4. 同步采集四种模态：
   - RGB / Visible 可见光图像
   - Surface Normal 表面法线图
   - Segmentation 语义分割图
   - Depth 深度图
5. 使用 OpenHUTB/CARLA actor 坐标、语义图和深度图生成目标 bbox；
6. 输出：
   - RGB 图像
   - metric depth npy 与 16-bit 深度图
   - semantic id 与语义彩色预览图
   - camera-space surface normal npy 与法线图
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
  --width 1920 ^
  --height-img 1080 ^
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


TARGET_BACKGROUND_ID = 0
TARGET_VEHICLE_ID = 1
TARGET_PEDESTRIAN_ID = 2
TARGET_IGNORE_ID = 255
TWO_WHEEL_TYPE_TOKENS = (
    "bike",
    "bicycle",
    "motorcycle",
    "vespa",
    "yamaha",
    "harley",
    "kawasaki",
    "diamondback",
    "gazelle",
    "crossbike",
)


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
    parser.add_argument("--out", type=str, default="dataset_uav_small_carla_1920_no_tiny_occ50")
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--frames", type=int, default=50)

    # 图像参数
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height-img", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument(
        "--enable-rgb-postprocess",
        dest="enable_rgb_postprocess",
        action="store_true",
        default=True,
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
    parser.add_argument("--height", type=float, default=33.0)
    parser.add_argument("--height-min", type=float, default=30.0)
    parser.add_argument("--height-max", type=float, default=36.0)
    parser.add_argument("--radius-min", type=float, default=28.0)
    parser.add_argument("--radius-max", type=float, default=52.0)
    parser.add_argument("--pitch", type=float, default=-40.0)
    parser.add_argument("--pitch-min", type=float, default=-45.0)
    parser.add_argument("--pitch-max", type=float, default=-35.0)
    parser.add_argument("--route", type=str, default="random", choices=["orbit", "random"])
    parser.add_argument("--orbit-degrees-per-sequence", type=float, default=120.0)
    parser.add_argument("--safe-camera-min-z", type=float, default=30.0)
    parser.add_argument(
        "--camera-origin-over-road",
        action="store_true",
        default=True,
        help="随机路线下从道路 waypoint 正上方生成相机，避免相机落入建筑物。"
    )
    parser.add_argument(
        "--allow-off-road-camera-origin",
        dest="camera_origin_over_road",
        action="store_false",
        help="允许使用道路中心周围的圆环相机位置。"
    )
    parser.add_argument(
        "--pedestrian-centered-camera-probability",
        type=float,
        default=0.75,
        help="随机路线下优先围绕行人选择道路上方相机位姿的概率。"
    )
    parser.add_argument(
        "--vehicle-centered-camera-probability",
        type=float,
        default=0.0,
        help="随机路线下优先围绕四轮车辆选择道路上方相机位姿的概率。"
    )
    parser.add_argument("--max-camera-pose-retries", type=int, default=120)
    parser.add_argument("--min-near-depth-m", type=float, default=5.0)
    parser.add_argument("--max-near-depth-ratio", type=float, default=0.05)
    parser.add_argument("--min-road-visible-ratio", type=float, default=0.35)
    parser.add_argument("--road-semantic-ids", type=int, nargs="+", default=[1, 2, 6, 7, 8])

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
    parser.add_argument(
        "--random-weather",
        action="store_true",
        help=(
            "每个保存帧从 weather_presets 中均衡随机选择一种天气；"
            "每个场景只采集一次四模态。"
        )
    )
    parser.add_argument(
        "--collect-all-weather-presets",
        dest="collect_all_weather_presets",
        action="store_true",
        default=False,
        help="依次采集 weather_presets 中的全部天气，并按天气目录分类保存。"
    )
    parser.add_argument(
        "--single-weather-mode",
        dest="collect_all_weather_presets",
        action="store_false",
        help="仅使用 current、固定天气或随机天气采集。"
    )
    parser.add_argument(
        "--weather-presets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "随机天气候选列表或全部天气模式的预设列表；"
            "未指定时动态读取当前 API 的全部预设。"
        )
    )
    parser.add_argument(
        "--weather-warmup-frames",
        type=int,
        default=10,
        help="每次切换天气后预热的仿真帧数，用于稳定曝光、雨滴、雾和湿地效果。"
    )

    # 目标类别
    parser.add_argument(
        "--annotation-source",
        type=str,
        default="actor",
        choices=["actor"],
        help="四模态模式只使用 OpenHUTB/CARLA actor 坐标、语义和深度生成标注。"
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
    parser.add_argument(
        "--min-target-equivalent-side-px",
        type=float,
        default=16.0,
        help=(
            "候选帧中任一目标的 sqrt(bbox_width*bbox_height) 小于等于该值时，"
            "整帧重选相机位姿；设为 0 可关闭。"
        )
    )
    parser.add_argument(
        "--min-pedestrian-equivalent-side-px",
        type=float,
        default=50.0,
        help=(
            "行人框的最小等效边长 sqrt(width*height)；低于该值时整帧重选位姿。"
        )
    )
    parser.add_argument(
        "--min-pedestrians-per-frame",
        type=int,
        default=1,
        help="每个正式保存帧至少包含的合格行人标注数量。"
    )
    parser.add_argument(
        "--reject-boundary-annotations",
        action="store_true",
        default=True,
        help="目标框接触图像边界时整帧重拍，避免截断目标。"
    )
    parser.add_argument(
        "--allow-boundary-annotations",
        dest="reject_boundary_annotations",
        action="store_false",
        help="允许保留接触图像边界的截断目标。"
    )
    parser.add_argument(
        "--annotation-boundary-margin-px",
        type=int,
        default=1,
        help="判定目标框接触图像边界时使用的像素边距。"
    )
    parser.add_argument("--keep-all", action="store_true")
    parser.add_argument("--min-actor-visible-px", type=int, default=24)
    parser.add_argument("--min-actor-visible-ratio", type=float, default=0.50)
    parser.add_argument(
        "--min-vehicle-projected-fill-ratio",
        type=float,
        default=0.25,
        help="车辆可见像素占 3D 投影框的最低比例。"
    )
    parser.add_argument(
        "--min-pedestrian-projected-fill-ratio",
        type=float,
        default=0.40,
        help="行人可见像素占 3D 投影框的最低比例。"
    )
    parser.add_argument(
        "--min-vehicle-visible-equivalent-side-px",
        type=float,
        default=40.0,
        help="目标语义图中车辆可见掩码的最小等效边长。"
    )
    parser.add_argument(
        "--min-pedestrian-visible-equivalent-side-px",
        type=float,
        default=48.0,
        help="目标语义图中行人可见掩码的最小等效边长。"
    )
    parser.add_argument(
        "--min-largest-component-ratio",
        type=float,
        default=0.75,
        help="最大连通区域占实例可见像素的最低比例，用于过滤严重遮挡。"
    )
    parser.add_argument("--actor-depth-margin", type=float, default=5.0)
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
    parser.add_argument(
        "--enable-lidar",
        dest="enable_lidar",
        action="store_true",
        default=True,
        help="在 Depth 模态中同时采集 LiDAR 原始点云和相机对齐的稀疏深度图。"
    )
    parser.add_argument(
        "--disable-lidar",
        dest="enable_lidar",
        action="store_false",
        help="关闭 LiDAR 深度采集。"
    )
    parser.add_argument("--lidar-channels", type=int, default=64)
    parser.add_argument("--lidar-range", type=float, default=250.0)
    parser.add_argument("--lidar-points-per-second", type=int, default=400000)
    parser.add_argument("--lidar-rotation-frequency", type=float, default=20.0)
    parser.add_argument("--lidar-upper-fov", type=float, default=30.0)
    parser.add_argument("--lidar-lower-fov", type=float, default=-30.0)
    parser.add_argument(
        "--normal-max-depth-jump-m",
        type=float,
        default=5.0,
        help="计算表面法线时允许的最大相邻深度突变，单位米；物体边界超过该值时法线置为无效。"
    )

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

    parser.add_argument(
        "--preserve-existing-sequences",
        dest="preserve_existing_sequences",
        action="store_true",
        default=True,
        help="保留已有序列并从下一个 seq_XXXX 继续采集，避免覆盖其他季节/天气的 RGB。"
    )
    parser.add_argument(
        "--overwrite-sequences",
        dest="preserve_existing_sequences",
        action="store_false",
        help="从 seq_0000 开始写入；同名文件可能被覆盖。"
    )

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
        "depth_npy": seq_dir / "depth" / "npy",
        "depth_vis": seq_dir / "depth" / "vis_16bit",
        "depth_color": seq_dir / "depth" / "color",
        "lidar_points": seq_dir / "depth" / "lidar" / "points",
        "lidar_projected_npy": seq_dir / "depth" / "lidar" / "projected_npy",
        "lidar_projected_vis": seq_dir / "depth" / "lidar" / "projected_vis_16bit",
        "lidar_projected_color": seq_dir / "depth" / "lidar" / "projected_color",
        "surface_normal": seq_dir / "surface_normal" / "png",
        "surface_normal_npy": seq_dir / "surface_normal" / "npy",
        "segmentation": seq_dir / "segmentation" / "id",
        "segmentation_color": seq_dir / "segmentation" / "color",
        "ann": seq_dir / "annotations",
        "yolo": seq_dir / "labels_yolo"
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def existing_sequence_state(out_root: Path) -> Tuple[int, List[Dict[str, Any]]]:
    """返回下一个序列编号和已有序列元数据，用于追加不同季节/天气的数据。"""
    indices: List[int] = []
    metadata: List[Dict[str, Any]] = []
    for seq_dir in sorted(out_root.glob("seq_*")):
        if not seq_dir.is_dir():
            continue
        try:
            indices.append(int(seq_dir.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue

        meta_path = seq_dir / "sequence_meta.json"
        if meta_path.exists():
            try:
                sequence_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                sequence_meta = {"sequence": seq_dir.name}
        else:
            sequence_meta = {"sequence": seq_dir.name}

        ann_paths = sorted((seq_dir / "annotations").glob("*.json"))
        four_modalities_complete = False
        if ann_paths:
            try:
                image_info = json.loads(
                    ann_paths[0].read_text(encoding="utf-8")
                ).get("image", {})
                four_modalities_complete = all(
                    image_info.get(key)
                    for key in (
                        "rgb",
                        "depth_npy_meters",
                        "surface_normal_npy",
                        "segmentation"
                    )
                )
            except (OSError, json.JSONDecodeError):
                pass
        sequence_meta["four_modalities_complete"] = four_modalities_complete
        metadata.append(sequence_meta)

    next_index = max(indices) + 1 if indices else 0
    return next_index, metadata


# ============================================================
# 6. 图像处理
# ============================================================

def carla_image_to_bgra(image: carla.Image) -> np.ndarray:
    """
    CARLA 图像 raw_data 是 BGRA uint8。
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    return arr.reshape((image.height, image.width, 4))


def apply_depth_fog_rgb_effect(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray]
) -> np.ndarray:
    """按相机深度生成高空仍可见的浓雾，保持几何和标注像素对齐。"""
    if depth_m is None or depth_m.shape != bgr.shape[:2]:
        return bgr.copy()
    depth = np.nan_to_num(
        depth_m.astype(np.float32),
        nan=250.0,
        posinf=250.0,
        neginf=0.0
    )
    distance = np.maximum(depth - 20.0, 0.0)
    fog_alpha = 0.08 + 0.78 * (1.0 - np.exp(-distance / 85.0))
    fog_alpha = np.clip(fog_alpha, 0.08, 0.84)[:, :, None]
    fog_color = np.array([205.0, 210.0, 214.0], dtype=np.float32)
    image = bgr.astype(np.float32)
    image = image * (1.0 - fog_alpha) + fog_color * fog_alpha
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def apply_snow_rgb_effect(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    random_seed: int
) -> np.ndarray:
    """生成确定性的阴雪、远距离雪雾和飘雪效果，不伪造地面积雪。"""
    image = bgr.astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image = image * 0.78 + gray[:, :, None] * 0.22

    haze_alpha = np.full(gray.shape, 0.06, dtype=np.float32)
    if depth_m is not None and depth_m.shape == gray.shape:
        distance_haze = np.clip((depth_m.astype(np.float32) - 30.0) / 180.0, 0.0, 1.0)
        haze_alpha += 0.20 * distance_haze
    snow_haze_color = np.array([238.0, 241.0, 245.0], dtype=np.float32)
    image = image * (1.0 - haze_alpha[:, :, None]) + snow_haze_color * haze_alpha[:, :, None]

    height, width = gray.shape
    rng = np.random.default_rng(int(random_seed))
    flake_mask = np.zeros((height, width), dtype=np.float32)
    small_count = max(120, (height * width) // 3500)
    large_count = max(20, (height * width) // 30000)

    for _ in range(small_count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, 3))
        strength = float(rng.uniform(0.35, 0.75))
        cv2.circle(flake_mask, (x, y), radius, strength, -1, cv2.LINE_AA)

    for _ in range(large_count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        length = int(rng.integers(4, 10))
        thickness = int(rng.integers(1, 3))
        strength = float(rng.uniform(0.55, 0.95))
        cv2.line(
            flake_mask,
            (x, y),
            (min(width - 1, x + length // 3), min(height - 1, y + length)),
            strength,
            thickness,
            cv2.LINE_AA
        )

    flake_mask = cv2.GaussianBlur(flake_mask, (3, 3), 0.55)
    flake_mask = np.clip(flake_mask, 0.0, 0.92)[:, :, None]
    image = image * (1.0 - flake_mask) + 255.0 * flake_mask
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def save_rgb(
    image: carla.Image,
    path: Path,
    weather_name: Optional[str] = None,
    depth_m: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> np.ndarray:
    """
    保存 RGB 相机图像。

    CARLA raw_data 是 BGRA。
    OpenCV 保存需要 BGR。
    """
    bgra = carla_image_to_bgra(image)
    bgr = bgra[:, :, :3].copy()
    if weather_name == "FoggyNoon":
        bgr = apply_depth_fog_rgb_effect(bgr, depth_m)
    elif weather_name == "SnowNoon":
        bgr = apply_snow_rgb_effect(bgr, depth_m, random_seed)
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


def metric_depth_to_color_bgr(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    """
    生成仅用于人工 QA 的 8-bit 彩色深度预览图。
    """
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.zeros(depth_m.shape, dtype=np.float32)
    if np.any(valid):
        clipped = np.clip(depth_m[valid].astype(np.float32), 0.0, max_depth_m)
        normalized[valid] = clipped / float(max_depth_m)
    gray = (normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


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

    normalized = depth_to_disparity_float32(
        depth_m=depth_m,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        valid_mask=valid_mask,
        invalid_value=0.0
    )
    return (np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)


def depth_to_disparity_float32(
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    valid_mask: Optional[np.ndarray] = None,
    invalid_value: float = 0.0
) -> np.ndarray:
    """
    生成 [0, 1] float32 normalized disparity，近处更接近 1。
    """
    if min_depth_m <= 0.0:
        raise ValueError("min_depth_m must be > 0")
    if max_depth_m <= min_depth_m:
        raise ValueError("max_depth_m must be larger than min_depth_m")

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if valid_mask is not None:
        valid = valid & valid_mask.astype(bool)

    out = np.full(depth_m.shape, invalid_value, dtype=np.float32)
    if not np.any(valid):
        return out

    d = np.clip(depth_m[valid].astype(np.float32), min_depth_m, max_depth_m)
    inv = 1.0 / d
    inv_min = 1.0 / max_depth_m
    inv_max = 1.0 / min_depth_m
    normalized = (inv - inv_min) / (inv_max - inv_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    out[valid] = normalized.astype(np.float32)
    return out


def save_depth(
    depth_m: np.ndarray,
    npy_path: Path,
    vis_path: Path,
    color_path: Path,
    max_depth_m: float
) -> None:
    """
    保存：
    1. 原始深度 npy，单位米；
    2. 16-bit 深度可视化 png。
    """
    np.save(npy_path, depth_m.astype(np.float32))

    cv2.imwrite(str(vis_path), metric_depth_to_u16(depth_m, max_depth_m))
    cv2.imwrite(str(color_path), metric_depth_to_color_bgr(depth_m, max_depth_m))


def depth_to_surface_normals(
    depth_m: np.ndarray,
    fov_degrees: float,
    max_depth_jump_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从米制深度和相机内参计算相机坐标系表面法线。

    npy 通道顺序为 (x-right, y-down, z-forward)，有效法线朝向相机；
    PNG 使用常见的 RGB=(nx, ny, nz) 映射到 [0, 255]，无效像素为黑色。
    """
    height, width = depth_m.shape
    fx = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )
    z = depth_m.astype(np.float32)
    points = np.stack(
        (
            (u - cx) * z / fx,
            (v - cy) * z / fy,
            z
        ),
        axis=-1
    )

    normals = np.zeros((height, width, 3), dtype=np.float32)
    if height < 3 or width < 3:
        return normals, np.zeros((height, width, 3), dtype=np.uint8)

    du = points[1:-1, 2:, :] - points[1:-1, :-2, :]
    dv = points[2:, 1:-1, :] - points[:-2, 1:-1, :]
    interior = np.cross(du, dv)
    norm = np.linalg.norm(interior, axis=2)

    center_depth = z[1:-1, 1:-1]
    neighbor_depths = (
        z[1:-1, :-2],
        z[1:-1, 2:],
        z[:-2, 1:-1],
        z[2:, 1:-1]
    )
    valid = np.isfinite(center_depth) & (center_depth > 0.0) & (norm > 1e-8)
    for neighbor in neighbor_depths:
        valid &= np.isfinite(neighbor) & (neighbor > 0.0)
        valid &= np.abs(neighbor - center_depth) <= max_depth_jump_m

    interior_normalized = np.zeros_like(interior, dtype=np.float32)
    interior_normalized[valid] = (
        interior[valid] / norm[valid, None]
    ).astype(np.float32)

    # 统一让法线朝向相机，避免同一平面出现随机正负方向。
    center_points = points[1:-1, 1:-1, :]
    away_from_camera = np.sum(interior_normalized * center_points, axis=2) > 0.0
    interior_normalized[away_from_camera] *= -1.0
    normals[1:-1, 1:-1, :] = interior_normalized

    normal_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    valid_full = np.linalg.norm(normals, axis=2) > 0.5
    encoded = np.clip((normals + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    normal_rgb[valid_full] = encoded[valid_full]
    return normals, normal_rgb


def save_surface_normal(
    depth_m: np.ndarray,
    npy_path: Path,
    png_path: Path,
    fov_degrees: float,
    max_depth_jump_m: float
) -> np.ndarray:
    normals, normal_rgb = depth_to_surface_normals(
        depth_m=depth_m,
        fov_degrees=fov_degrees,
        max_depth_jump_m=max_depth_jump_m
    )
    np.save(npy_path, normals.astype(np.float32))
    cv2.imwrite(str(png_path), cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
    return normals


def save_disparity(
    depth_m: np.ndarray,
    npy_path: Path,
    vis_path: Path,
    min_depth_m: float,
    max_depth_m: float
) -> None:
    disparity = depth_to_disparity_float32(depth_m, min_depth_m, max_depth_m)
    np.save(npy_path, disparity.astype(np.float32))
    cv2.imwrite(
        str(vis_path),
        (np.clip(disparity, 0.0, 1.0) * 65535.0).astype(np.uint16)
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


def decode_semantic_segmentation(semantic_image: carla.Image) -> np.ndarray:
    """
    解码 semantic segmentation 为单通道 semantic id。

    OpenHUTB/CARLA 常见 raw 格式是 BGRA 中 R 通道存 semantic id。
    如果遇到调色板图像，保守返回信息量最大的低值通道，避免把全 0 通道当成语义图。
    """
    bgra = carla_image_to_bgra(semantic_image)
    best_channel = 2
    best_score = -1.0

    for channel in range(3):
        values = bgra[:, :, channel].astype(np.uint8)
        unique = np.unique(values)
        nonzero_ratio = float(np.mean(values != 0))
        low_value_ratio = float(np.mean(values <= 40))
        score = float(unique.size) + nonzero_ratio * 10.0 + low_value_ratio
        if score > best_score:
            best_channel = channel
            best_score = score

    return bgra[:, :, best_channel].astype(np.uint8)


def save_semantic_id(semantic_id: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), semantic_id.astype(np.uint8))


def colorize_semantic_id(semantic_id: np.ndarray) -> np.ndarray:
    palette_bgr = {
        0: (0, 0, 0),
        1: (80, 80, 80),
        2: (120, 120, 120),
        3: (70, 70, 70),
        4: (220, 20, 60),
        6: (50, 234, 157),
        7: (128, 64, 128),
        8: (232, 35, 244),
        9: (35, 142, 107),
        10: (142, 0, 0),
        11: (70, 0, 0),
        12: (100, 60, 0),
        14: (153, 153, 153),
        18: (30, 170, 250),
        20: (0, 220, 220),
        21: (35, 100, 255),
        22: (152, 251, 152),
        24: (180, 130, 70),
        25: (60, 20, 220),
    }
    out = np.zeros((*semantic_id.shape, 3), dtype=np.uint8)
    for sem_id, color in palette_bgr.items():
        out[semantic_id == sem_id] = color
    unknown = ~np.isin(
        semantic_id,
        np.array(list(palette_bgr.keys()), dtype=np.uint8)
    )
    if np.any(unknown):
        values = semantic_id[unknown].astype(np.uint16)
        out[unknown, 0] = ((values * 37) % 255).astype(np.uint8)
        out[unknown, 1] = ((values * 67) % 255).astype(np.uint8)
        out[unknown, 2] = ((values * 97) % 255).astype(np.uint8)
    return out


def save_semantic_color(semantic_id: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), colorize_semantic_id(semantic_id))


def save_instance_id(instance_id: np.ndarray, png_path: Path, npy_path: Path) -> None:
    np.save(npy_path, instance_id.astype(np.uint16))
    cv2.imwrite(str(png_path), instance_id.astype(np.uint16))


def colorize_instance_id(instance_id: np.ndarray) -> np.ndarray:
    values = instance_id.astype(np.uint32)
    out = np.zeros((*instance_id.shape, 3), dtype=np.uint8)
    nonzero = values != 0
    out[:, :, 0] = ((values * 37 + 17) % 255).astype(np.uint8)
    out[:, :, 1] = ((values * 67 + 29) % 255).astype(np.uint8)
    out[:, :, 2] = ((values * 97 + 43) % 255).astype(np.uint8)
    out[~nonzero] = 0
    return out


def save_instance_color(instance_id: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), colorize_instance_id(instance_id))


def decode_instance_segmentation(instance_image: carla.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    解码 CARLA instance segmentation。

    CARLA instance segmentation 中：
    - R 通道通常为 semantic id；
    - G/B 通道编码 instance id，其中 G 是低 8 位，B 是高 8 位。

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
        bgra[:, :, 0].astype(np.uint16) * 256
        + bgra[:, :, 1].astype(np.uint16)
    )

    return semantic_id, instance_id


def largest_connected_component(
    mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
    total = int(np.count_nonzero(mask_u8))
    if count <= 1 or total == 0:
        return mask.astype(bool), 1.0 if total else 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    component_index = int(np.argmax(areas)) + 1
    component_area = int(areas[component_index - 1])
    return labels == component_index, component_area / float(total)


def filled_external_instance_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(
        contour,
        max(0.5, 0.001 * perimeter),
        True,
    )
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(filled, [contour], 1)
    return filled.astype(bool)


def bbox_intersection_over_instance(
    instance_bbox: Tuple[int, int, int, int],
    actor_bbox: List[int],
) -> float:
    ix1, iy1, ix2, iy2 = instance_bbox
    ax1, ay1, ax2, ay2 = (int(value) for value in actor_bbox)
    x1 = max(ix1, ax1)
    y1 = max(iy1, ay1)
    x2 = min(ix2, ax2)
    y2 = min(iy2, ay2)
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    instance_area = max(1, ix2 - ix1 + 1) * max(1, iy2 - iy1 + 1)
    return intersection / float(instance_area)


def target_mask_id_for_class(class_name: str) -> int:
    normalized = class_name.lower()
    if normalized in ("vehicle", "car", "truck", "bus"):
        return TARGET_VEHICLE_ID
    if normalized in ("pedestrian", "person", "walker"):
        return TARGET_PEDESTRIAN_ID
    raise ValueError(f"总采集器只发布 vehicle/pedestrian，收到：{class_name}")


def build_optimized_target_semantic_mask(
    instance_image: carla.Image,
    actor_annotations: List[Dict[str, Any]],
    min_mask_px: int,
    min_vehicle_projected_fill_ratio: float,
    min_pedestrian_projected_fill_ratio: float,
    min_vehicle_visible_equivalent_side_px: float,
    min_pedestrian_visible_equivalent_side_px: float,
    min_largest_component_ratio: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    _, instance_id = decode_instance_segmentation(instance_image)
    target_mask = np.full(
        instance_id.shape,
        TARGET_BACKGROUND_ID,
        dtype=np.uint8,
    )
    instances: List[Dict[str, Any]] = []

    # OpenHUTB 中动态 Actor 的 semantic tag 不稳定。Actor 提供类别和 3D
    # 投影框，instance 相机按完全相同的 Actor ID 提供目标轮廓。
    all_candidates: List[Dict[str, Any]] = []
    image_height, image_width = instance_id.shape
    for annotation in actor_annotations:
        x1, y1, x2, y2 = (
            int(value)
            for value in annotation.get(
                "projected_bbox_xyxy",
                annotation["bbox_xyxy"],
            )
        )
        x1 = min(max(0, x1), image_width - 1)
        x2 = min(max(0, x2), image_width - 1)
        y1 = min(max(0, y1), image_height - 1)
        y2 = min(max(0, y2), image_height - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        instance_crop = instance_id[y1:y2 + 1, x1:x2 + 1]
        projected_area = int(instance_crop.size)
        mask_id = target_mask_id_for_class(str(annotation["class_name"]))
        class_fill_threshold = (
            min_vehicle_projected_fill_ratio
            if mask_id == TARGET_VEHICLE_ID
            else min_pedestrian_projected_fill_ratio
        )

        for value in np.unique(instance_crop):
            current_id = int(value)
            if current_id != int(annotation["carla_actor_id"]):
                continue
            instance_pixels = instance_crop == value
            instance_px = int(np.count_nonzero(instance_pixels))
            if instance_px < min_mask_px:
                continue
            consistent_pixels = instance_pixels
            overlap_px = int(np.count_nonzero(consistent_pixels))
            if overlap_px < min_mask_px:
                continue
            depth_consistency = 1.0
            projected_fill_ratio = overlap_px / float(projected_area)
            if projected_fill_ratio < class_fill_threshold:
                continue

            raw_mask = np.zeros(instance_id.shape, dtype=bool)
            raw_mask[y1:y2 + 1, x1:x2 + 1] = consistent_pixels
            score = (
                102.0
                + math.log1p(overlap_px) / 10.0
                + min(projected_fill_ratio, 0.5)
            )
            all_candidates.append({
                "score": score,
                "instance_id": current_id,
                "annotation": annotation,
                "mask_id": mask_id,
                "raw_mask": raw_mask,
                "depth_consistency": depth_consistency,
                "projected_fill_ratio": projected_fill_ratio,
                "mask_source": "exact_actor_instance",
            })

    used_actor_ids = set()
    used_instance_ids = set()
    for candidate in sorted(
        all_candidates,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        annotation = candidate["annotation"]
        actor_id = int(annotation["carla_actor_id"])
        current_id = int(candidate["instance_id"])
        if actor_id in used_actor_ids or current_id in used_instance_ids:
            continue
        used_actor_ids.add(actor_id)
        used_instance_ids.add(current_id)

        raw_mask = candidate["raw_mask"]
        target_mask[raw_mask] = np.uint8(TARGET_IGNORE_ID)
        component, component_ratio = largest_connected_component(raw_mask)
        equivalent_side = float(np.sqrt(float(np.count_nonzero(component))))
        mask_id = int(candidate["mask_id"])
        visible_threshold = (
            min_vehicle_visible_equivalent_side_px
            if mask_id == TARGET_VEHICLE_ID
            else min_pedestrian_visible_equivalent_side_px
        )
        trainable = bool(
            equivalent_side >= visible_threshold
            and component_ratio >= min_largest_component_ratio
        )
        if mask_id == TARGET_PEDESTRIAN_ID:
            training_mask = component
        else:
            training_mask = filled_external_instance_contour(component)
            if training_mask is None:
                trainable = False
                training_mask = component
        ys, xs = np.where(training_mask)
        if xs.size == 0:
            trainable = False
            bbox = [0, 0, 0, 0]
        else:
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ]
        if trainable:
            target_mask[training_mask] = np.uint8(mask_id)

        instances.append({
            "class_id": int(annotation["class_id"]),
            "class_name": str(annotation["class_name"]),
            "mask_id": mask_id,
            "carla_instance_id": current_id,
            "carla_actor_id": actor_id,
            "actor_type_id": annotation.get("actor_type_id"),
            "bbox_xyxy": bbox,
            "projected_bbox_xyxy": list(
                annotation.get(
                    "projected_bbox_xyxy",
                    annotation["bbox_xyxy"],
                )
            ),
            "raw_visible_area_px": int(np.count_nonzero(raw_mask)),
            "training_area_px": int(np.count_nonzero(training_mask)),
            "visible_equivalent_side_px": equivalent_side,
            "largest_component_ratio": float(component_ratio),
            "depth_consistency": float(candidate["depth_consistency"]),
            "projected_fill_ratio": float(candidate["projected_fill_ratio"]),
            "mask_source": str(candidate["mask_source"]),
            "trainable": trainable,
        })

    return target_mask, instances


def colorize_target_semantic_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == TARGET_VEHICLE_ID] = (0, 220, 0)
    color[mask == TARGET_PEDESTRIAN_ID] = (255, 80, 20)
    color[mask == TARGET_IGNORE_ID] = (255, 0, 255)
    return color


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

    if type_id.startswith("vehicle.") and not is_two_wheel_type(type_id):
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
    min_vehicle_projected_fill_ratio: float,
    min_pedestrian_projected_fill_ratio: float,
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

        class_fill_ratio = (
            min_vehicle_projected_fill_ratio
            if target.name.lower() in ("vehicle", "car", "truck", "bus")
            else min_pedestrian_projected_fill_ratio
        )
        visible_ratio_threshold = class_fill_ratio
        if (
            visible_px < min_actor_visible_px
            or visible_ratio_projected < visible_ratio_threshold
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


def dataset_relative_path(path: Path, dataset_root: Path) -> str:
    """
    将文件路径写成相对数据集根目录的 POSIX 风格路径，方便跨机器移动数据集。
    """
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def save_annotation_modalities(
    dataset_root: Path,
    mask_dir: Path,
    object_depth_npy_dir: Path,
    object_depth_vis_dir: Path,
    object_depth_color_dir: Path,
    object_disparity_npy_dir: Path,
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
        depth_color_path = object_depth_color_dir / f"{frame_stem}_ann{ann['id']:03d}.png"
        disparity_npy_path = object_disparity_npy_dir / f"{frame_stem}_ann{ann['id']:03d}.npy"
        disparity_path = object_disparity_vis_dir / f"{frame_stem}_ann{ann['id']:03d}.png"

        np.save(depth_npy_path, depth_crop_masked)
        disparity_crop = depth_to_disparity_float32(
            depth_crop,
            min_disparity_depth_m,
            max_disparity_depth_m,
            valid_mask=mask_crop,
            invalid_value=np.nan
        )
        np.save(disparity_npy_path, disparity_crop.astype(np.float32))
        cv2.imwrite(str(depth_vis_path), metric_depth_to_u16(depth_crop_masked, max_depth_vis_m))
        cv2.imwrite(str(depth_color_path), metric_depth_to_color_bgr(depth_crop_masked, max_depth_vis_m))
        cv2.imwrite(
            str(disparity_path),
            (np.nan_to_num(disparity_crop, nan=0.0) * 65535.0).astype(np.uint16)
        )

        ann["files"] = {
            "mask": str(mask_path),
            "object_depth_npy_meters": str(depth_npy_path),
            "object_depth_vis_16bit": str(depth_vis_path),
            "object_depth_color": str(depth_color_path),
            "object_disparity_npy": str(disparity_npy_path),
            "object_disparity_vis_16bit": str(disparity_path)
        }
        ann["files"] = {
            key: dataset_relative_path(Path(value), dataset_root)
            for key, value in ann["files"].items()
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


WEATHER_PARAMETER_FIELDS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "fog_distance",
    "fog_falloff",
    "wetness",
    "scattering_intensity",
    "mie_scattering_scale",
    "rayleigh_scattering_scale",
    "dust_storm",
)

CUSTOM_WEATHER_PRESETS = {
    "FoggyNoon": {
        "base": "CloudyNoon",
        "overrides": {
            "cloudiness": 80.0,
            "fog_density": 72.0,
            "fog_distance": 0.0,
            "fog_falloff": 0.0,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wetness": 10.0,
            "wind_intensity": 8.0,
        },
        "rendering": "custom_carla_weather_plus_depth_based_rgb_fog",
    },
    "SnowNoon": {
        "base": "CloudyNoon",
        "overrides": {
            "cloudiness": 100.0,
            "fog_density": 28.0,
            "fog_distance": 8.0,
            "fog_falloff": 0.55,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wetness": 0.0,
            "wind_intensity": 35.0,
            "sun_altitude_angle": 35.0,
        },
        "rendering": "custom_carla_weather_plus_deterministic_rgb_snowfall",
        "surface_snow_accumulation": False,
    },
}


def clone_weather_parameters(base_name: str, overrides: Dict[str, float]):
    base = getattr(carla.WeatherParameters, base_name)
    weather = carla.WeatherParameters()
    for field in WEATHER_PARAMETER_FIELDS:
        if hasattr(base, field) and hasattr(weather, field):
            setattr(weather, field, float(getattr(base, field)))
    for field, value in overrides.items():
        if not hasattr(weather, field):
            raise RuntimeError(f"当前天气 API 不支持参数：{field}")
        setattr(weather, field, float(value))
    return weather


def weather_rendering_metadata(preset_name: str) -> Dict[str, Any]:
    custom = CUSTOM_WEATHER_PRESETS.get(preset_name)
    if custom is None:
        return {"rendering": "native_carla_weather_preset"}
    return dict(custom)


def apply_weather(world: carla.World, preset_name: Optional[str]) -> str:
    if preset_name is None:
        print("[INFO] Keeping current simulator weather.")
        return "current"

    preset_name = str(preset_name).strip()
    if preset_name.lower() in ("", "current", "keep", "keep_current"):
        print("[INFO] Keeping current simulator weather.")
        return "current"

    custom = CUSTOM_WEATHER_PRESETS.get(preset_name)
    if custom is not None:
        weather = clone_weather_parameters(custom["base"], custom["overrides"])
    else:
        weather = getattr(carla.WeatherParameters, preset_name, None)
    if not isinstance(weather, carla.WeatherParameters):
        raise RuntimeError(
            f"当前 OpenHUTB/CARLA API 不支持天气预设：{preset_name}。"
            "为避免目录标签与实际天气不一致，采集已停止。"
        )
    world.set_weather(weather)
    print(f"[INFO] Applied weather: {preset_name}")
    return preset_name


def get_weather_names() -> List[str]:
    """动态读取当前 OpenHUTB/CARLA PythonAPI 实际提供的天气预设。"""
    names = []
    for name in dir(carla.WeatherParameters):
        if name.startswith("_"):
            continue
        try:
            value = getattr(carla.WeatherParameters, name)
        except Exception:
            continue
        if isinstance(value, carla.WeatherParameters):
            names.append(name)
    names.extend(CUSTOM_WEATHER_PRESETS)
    return sorted(set(names))


def freeze_dynamic_actors_for_weather_sweep(
    vehicle_actors: List[carla.Actor],
    walker_actors: List[carla.Actor],
    walker_controllers: List[carla.Actor],
    tm_port: int
) -> List[Dict[str, Any]]:
    """冻结车辆和行人，使不同天气 RGB 使用完全相同的场景几何。"""
    snapshots: List[Dict[str, Any]] = []

    for controller in walker_controllers:
        try:
            controller.stop()
        except Exception:
            pass

    for actor in list(vehicle_actors) + list(walker_actors):
        try:
            snapshot = {
                "actor": actor,
                "transform": actor.get_transform(),
                "is_vehicle": actor.type_id.startswith("vehicle.")
            }
            snapshots.append(snapshot)

            if snapshot["is_vehicle"]:
                actor.set_autopilot(False, tm_port)
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_simulate_physics(False)
            actor.set_transform(snapshot["transform"])
        except Exception as exc:
            print(f"[WARN] Failed to freeze actor {getattr(actor, 'id', '?')}: {exc}")

    return snapshots


def restore_frozen_actor_transforms(snapshots: List[Dict[str, Any]]) -> None:
    for snapshot in snapshots:
        try:
            snapshot["actor"].set_transform(snapshot["transform"])
        except Exception:
            pass


def resume_dynamic_actors_after_weather_sweep(
    world: carla.World,
    snapshots: List[Dict[str, Any]],
    walker_controllers: List[carla.Actor],
    tm_port: int
) -> None:
    """恢复车辆自动驾驶和行人控制器。"""
    for snapshot in snapshots:
        actor = snapshot["actor"]
        try:
            actor.set_transform(snapshot["transform"])
            actor.set_simulate_physics(True)
            if snapshot["is_vehicle"]:
                actor.set_autopilot(True, tm_port)
        except Exception as exc:
            print(f"[WARN] Failed to resume actor {getattr(actor, 'id', '?')}: {exc}")

    for controller in walker_controllers:
        try:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(random.uniform(0.8, 1.6))
        except Exception:
            pass


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
    创建 RGB / depth / semantic 三个公开模态相机，以及内部实例标注相机。

    Surface Normal 由同帧 depth 和相机内参离线计算，保证与 RGB 像素严格对齐。

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


def spawn_lidar_sensor(
    world: carla.World,
    sensor_tick: float,
    initial_transform: carla.Transform,
    channels: int,
    lidar_range: float,
    points_per_second: int,
    rotation_frequency: float,
    upper_fov: float,
    lower_fov: float
) -> carla.Sensor:
    """创建与相机共位姿的机械式 LiDAR，作为 Depth 模态的稀疏深度来源。"""
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    attributes = {
        "channels": channels,
        "range": lidar_range,
        "points_per_second": points_per_second,
        "rotation_frequency": rotation_frequency,
        "upper_fov": upper_fov,
        "lower_fov": lower_fov,
        "sensor_tick": sensor_tick,
        "dropoff_general_rate": 0.0,
        "dropoff_intensity_limit": 0.0,
        "dropoff_zero_intensity": 0.0,
        "noise_stddev": 0.0
    }
    for name, value in attributes.items():
        if bp.has_attribute(name):
            bp.set_attribute(name, str(value))
    sensor = world.spawn_actor(bp, initial_transform)
    print(
        "[INFO] Spawning sensor: lidar -> sensor.lidar.ray_cast "
        f"channels={channels}, range={lidar_range}, pps={points_per_second}"
    )
    return sensor


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
    pitch = random.uniform(pitch_min, pitch_max)
    radius = ground_aim_radius(height, pitch, radius_min, radius_max)
    theta = random.uniform(0.0, 2.0 * math.pi)

    return uav_orbit_transform(
        center_x=center_x,
        center_y=center_y,
        height=height,
        radius=radius,
        theta=theta,
        pitch=pitch
    )


def random_road_uav_transform(
    carla_map,
    height_min: float,
    height_max: float,
    radius_min: float,
    radius_max: float,
    pitch_min: float,
    pitch_max: float
) -> carla.Transform:
    """Generate an oblique UAV pose whose horizontal origin is over a road."""
    spawn_points = carla_map.get_spawn_points()
    if len(spawn_points) == 0:
        raise RuntimeError("地图没有车辆 spawn point，无法生成道路上方相机位姿。")

    spawn_transform = random.choice(spawn_points)
    waypoint = None
    try:
        waypoint = carla_map.get_waypoint(
            spawn_transform.location,
            project_to_road=True
        )
    except (RuntimeError, TypeError):
        waypoint = None

    ground_transform = waypoint.transform if waypoint is not None else spawn_transform
    ground_location = ground_transform.location
    altitude = random.uniform(height_min, height_max)
    pitch = random.uniform(pitch_min, pitch_max)
    aim_distance = ground_aim_radius(
        altitude,
        pitch,
        radius_min,
        radius_max
    )

    target_location = None
    if waypoint is not None:
        step_name = "next" if random.random() < 0.5 else "previous"
        step = getattr(waypoint, step_name, None)
        if callable(step):
            candidates = step(aim_distance)
            if candidates:
                target_location = random.choice(candidates).transform.location

    if target_location is not None:
        yaw = math.degrees(
            math.atan2(
                float(target_location.y - ground_location.y),
                float(target_location.x - ground_location.x)
            )
        )
    else:
        yaw = float(ground_transform.rotation.yaw)
        if random.random() < 0.5:
            yaw += 180.0

    location = carla.Location(
        x=float(ground_location.x),
        y=float(ground_location.y),
        z=float(ground_location.z) + altitude
    )
    rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
    return carla.Transform(location, rotation)


def random_pedestrian_centered_road_uav_transform(
    carla_map,
    pedestrian_actors: List[carla.Actor],
    height_min: float,
    height_max: float,
    radius_min: float,
    radius_max: float,
    pitch_min: float,
    pitch_max: float
) -> carla.Transform:
    """Aim from a road waypoint above ground toward a live pedestrian."""
    live_pedestrians = [
        actor for actor in pedestrian_actors
        if actor is not None and actor.is_alive
    ]
    if not live_pedestrians:
        return random_road_uav_transform(
            carla_map=carla_map,
            height_min=height_min,
            height_max=height_max,
            radius_min=radius_min,
            radius_max=radius_max,
            pitch_min=pitch_min,
            pitch_max=pitch_max
        )

    target_actor = random.choice(live_pedestrians)
    target_location = target_actor.get_location()
    target_waypoint = carla_map.get_waypoint(
        target_location,
        project_to_road=True
    )
    if target_waypoint is None:
        return random_road_uav_transform(
            carla_map=carla_map,
            height_min=height_min,
            height_max=height_max,
            radius_min=radius_min,
            radius_max=radius_max,
            pitch_min=pitch_min,
            pitch_max=pitch_max
        )

    altitude = random.uniform(height_min, height_max)
    sampled_pitch = random.uniform(pitch_min, pitch_max)
    aim_distance = ground_aim_radius(
        altitude,
        sampled_pitch,
        radius_min,
        radius_max
    )

    direction_names = ["previous", "next"]
    random.shuffle(direction_names)
    camera_waypoint = None
    for direction_name in direction_names:
        step = getattr(target_waypoint, direction_name, None)
        if not callable(step):
            continue
        candidates = step(aim_distance)
        if candidates:
            camera_waypoint = random.choice(candidates)
            break

    if camera_waypoint is None:
        return random_road_uav_transform(
            carla_map=carla_map,
            height_min=height_min,
            height_max=height_max,
            radius_min=radius_min,
            radius_max=radius_max,
            pitch_min=pitch_min,
            pitch_max=pitch_max
        )

    camera_ground = camera_waypoint.transform.location
    location = carla.Location(
        x=float(camera_ground.x),
        y=float(camera_ground.y),
        z=float(camera_ground.z) + altitude
    )
    dx = float(target_location.x - location.x)
    dy = float(target_location.y - location.y)
    dz = float(target_location.z - location.z)
    horizontal_distance = max(0.1, math.hypot(dx, dy))
    yaw = math.degrees(math.atan2(dy, dx)) + random.uniform(-6.0, 6.0)
    exact_pitch = math.degrees(math.atan2(dz, horizontal_distance))
    pitch = min(pitch_max, max(pitch_min, exact_pitch + random.uniform(-2.5, 2.5)))
    return carla.Transform(
        location,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
    )


def ground_aim_radius(
    height: float,
    pitch: float,
    radius_min: float,
    radius_max: float
) -> float:
    """Return the horizontal distance that aims the camera at ground center."""
    downward_angle = min(89.9, max(0.1, abs(float(pitch))))
    radius = float(height) / math.tan(math.radians(downward_angle))
    return min(max(radius, float(radius_min)), float(radius_max))


def random_road_center(carla_map) -> Tuple[float, float]:
    spawn_points = carla_map.get_spawn_points()
    if len(spawn_points) == 0:
        return 0.0, 0.0

    sp = random.choice(spawn_points)
    return float(sp.location.x), float(sp.location.y)


def is_two_wheel_type(type_id: str) -> bool:
    normalized = type_id.lower()
    return any(token in normalized for token in TWO_WHEEL_TYPE_TOKENS)


def vehicle_blueprint_wheel_count(blueprint: carla.ActorBlueprint) -> int:
    if not blueprint.has_attribute("number_of_wheels"):
        return 0
    try:
        return int(blueprint.get_attribute("number_of_wheels").as_int())
    except Exception:
        try:
            return int(str(blueprint.get_attribute("number_of_wheels")))
        except Exception:
            return 0


def is_supported_vehicle_blueprint(
    blueprint: carla.ActorBlueprint,
) -> bool:
    return (
        not is_two_wheel_type(blueprint.id)
        and vehicle_blueprint_wheel_count(blueprint) >= 4
    )


def vehicle_actor_wheel_count(actor: carla.Actor) -> Optional[int]:
    try:
        value = actor.attributes.get("number_of_wheels")
        return int(value) if value is not None else None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def destroy_live_two_wheel_vehicles(world: carla.World) -> int:
    removed = 0
    for actor in list(world.get_actors().filter("vehicle.*")):
        wheel_count = vehicle_actor_wheel_count(actor)
        is_two_wheel = (
            (wheel_count is not None and wheel_count < 4)
            or is_two_wheel_type(actor.type_id)
        )
        if not is_two_wheel:
            continue
        try:
            actor.destroy()
            removed += 1
        except Exception:
            pass
    print(f"[INFO] Removed live bicycle/motorcycle actors: {removed}")
    return removed


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

    vehicle_bps = [
        blueprint
        for blueprint in blueprint_library.filter("vehicle.*")
        if is_supported_vehicle_blueprint(blueprint)
    ]
    if not vehicle_bps:
        raise RuntimeError("没有可用的四轮 vehicle blueprint。")
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


def decode_lidar_points(lidar_data) -> np.ndarray:
    """返回 LiDAR 局部坐标点云 [x-forward, y-right, z-up, intensity]。"""
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    return points.reshape((-1, 4)).copy()


def project_lidar_to_camera_depth(
    lidar_points: np.ndarray,
    lidar_to_camera: np.ndarray,
    width: int,
    height: int,
    fov_degrees: float,
    max_range_m: float
) -> np.ndarray:
    """将共位姿 LiDAR 点云投影为与 RGB 对齐的稀疏米制深度图。"""
    sparse_depth = np.zeros((height, width), dtype=np.float32)
    if lidar_points.size == 0:
        return sparse_depth

    lidar_xyz = lidar_points[:, :3].astype(np.float64)
    homogeneous = np.column_stack(
        (lidar_xyz, np.ones(lidar_xyz.shape[0], dtype=np.float64))
    )
    xyz = (lidar_to_camera @ homogeneous.T).T[:, :3].astype(np.float32)
    forward = xyz[:, 0]
    ranges = np.linalg.norm(xyz, axis=1)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(ranges)
        & (forward > 0.01)
        & (ranges > 0.0)
        & (ranges <= max_range_m)
    )
    if not np.any(valid):
        return sparse_depth

    xyz = xyz[valid]
    forward = xyz[:, 0]
    ranges = ranges[valid]
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    u = cx + focal * (xyz[:, 1] / forward)
    v = cy - focal * (xyz[:, 2] / forward)
    px = np.rint(u).astype(np.int32)
    py = np.rint(v).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(inside):
        return sparse_depth

    px = px[inside]
    py = py[inside]
    camera_forward_depth = forward[inside]
    flat_indices = py * width + px
    flat_depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(flat_depth, flat_indices, camera_forward_depth)
    finite = np.isfinite(flat_depth)
    flat_depth[~finite] = 0.0
    return flat_depth.reshape((height, width))


def save_lidar_depth(
    lidar_data,
    camera_data,
    points_path: Path,
    projected_npy_path: Path,
    projected_vis_path: Path,
    projected_color_path: Path,
    width: int,
    height: int,
    fov_degrees: float,
    max_range_m: float,
    max_depth_vis_m: float
) -> Dict[str, Any]:
    points = decode_lidar_points(lidar_data)
    lidar_to_world = np.asarray(
        lidar_data.transform.get_matrix(),
        dtype=np.float64
    )
    world_to_camera = np.asarray(
        camera_data.transform.get_inverse_matrix(),
        dtype=np.float64
    )
    lidar_to_camera = world_to_camera @ lidar_to_world
    sparse_depth = project_lidar_to_camera_depth(
        lidar_points=points,
        lidar_to_camera=lidar_to_camera,
        width=width,
        height=height,
        fov_degrees=fov_degrees,
        max_range_m=max_range_m
    )
    np.save(points_path, points.astype(np.float32))
    np.save(projected_npy_path, sparse_depth.astype(np.float32))
    cv2.imwrite(
        str(projected_vis_path),
        metric_depth_to_u16(sparse_depth, max_depth_vis_m)
    )
    cv2.imwrite(
        str(projected_color_path),
        metric_depth_to_color_bgr(sparse_depth, max_depth_vis_m)
    )
    valid_projected = sparse_depth > 0.0
    return {
        "point_count": int(points.shape[0]),
        "projected_pixel_count": int(np.count_nonzero(valid_projected)),
        "projected_pixel_ratio": float(np.mean(valid_projected)),
        "coordinate_system": "x-forward, y-right, z-up",
        "projected_depth_value": "camera-forward depth in meters",
        "lidar_measurement_transform": transform_to_dict(lidar_data.transform),
        "camera_measurement_transform": transform_to_dict(camera_data.transform),
        "lidar_to_camera_matrix": lidar_to_camera.tolist()
    }


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


def weather_to_dict(weather) -> Dict[str, float]:
    out = {}
    for field in WEATHER_PARAMETER_FIELDS:
        if hasattr(weather, field):
            out[field] = float(getattr(weather, field))
    return out


def number_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if len(values) == 0:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def read_dataset_frame_records(out_root: Path) -> List[Dict[str, Any]]:
    records = []
    for ann_path in sorted(out_root.glob("**/seq_*/annotations/*.json")):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        image_info = data.get("image", {})
        required_four_modalities = (
            image_info.get("rgb"),
            image_info.get("depth_npy_meters"),
            image_info.get("surface_normal_npy"),
            image_info.get("segmentation"),
            image_info.get("yolo")
        )
        if not all(required_four_modalities):
            # 保留旧季节 RGB，但不把缺少配对模态的旧帧混入四模态训练划分。
            continue
        records.append({
            "annotation_path": ann_path,
            "annotation_rel": dataset_relative_path(ann_path, out_root),
            "data": data,
            "rgb": str(data["image"]["rgb"]),
            "yolo": str(data["image"]["yolo"]),
        })
    return records


def split_frame_records(
    records: List[Dict[str, Any]],
    seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n < 5:
        return {"train": shuffled, "val": [], "test": []}

    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.20)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_split_files(
    out_root: Path,
    splits: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, str]:
    split_dir = out_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = {}
    for name, records in splits.items():
        split_path = split_dir / f"{name}.txt"
        split_path.write_text(
            "\n".join(record["rgb"] for record in records),
            encoding="utf-8"
        )
        rel_paths[name] = dataset_relative_path(split_path, out_root)
    return rel_paths


def write_weather_split_files(
    out_root: Path,
    splits: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, str]]:
    """为每个天气 RGB 生成与基础场景完全相同的 train/val/test 划分。"""
    weather_names = sorted({
        weather_name
        for records in splits.values()
        for record in records
        for weather_name in record["data"].get("image", {}).get("rgb_by_weather", {})
    })
    result: Dict[str, Dict[str, str]] = {}
    for weather_name in weather_names:
        weather_dir = out_root / "splits_by_weather" / weather_name
        weather_dir.mkdir(parents=True, exist_ok=True)
        result[weather_name] = {}
        for split_name, records in splits.items():
            paths = []
            for record in records:
                path = record["data"].get("image", {}).get(
                    "rgb_by_weather", {}
                ).get(weather_name)
                if path:
                    paths.append(str(path))
            split_path = weather_dir / f"{split_name}.txt"
            split_path.write_text("\n".join(paths), encoding="utf-8")
            result[weather_name][split_name] = dataset_relative_path(
                split_path,
                out_root
            )
    return result


def write_yolo_dataset_files(
    out_root: Path,
    targets: List[TargetClass],
    split_paths: Dict[str, str]
) -> Dict[str, str]:
    classes = sorted({int(t.class_id): t.name for t in targets}.items())
    classes_path = out_root / "classes.txt"
    classes_path.write_text(
        "\n".join(name for _, name in classes),
        encoding="utf-8"
    )

    yaml_lines = [
        "path: .",
        f"train: {split_paths.get('train', 'splits/train.txt')}",
        f"val: {split_paths.get('val', 'splits/val.txt')}",
        f"test: {split_paths.get('test', 'splits/test.txt')}",
        "names:",
    ]
    for class_id, name in classes:
        yaml_lines.append(f"  {class_id}: {name}")

    data_yaml_path = out_root / "data.yaml"
    data_yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return {
        "classes": dataset_relative_path(classes_path, out_root),
        "data_yaml": dataset_relative_path(data_yaml_path, out_root),
    }


def build_coco_dict(
    records: List[Dict[str, Any]],
    targets: List[TargetClass],
    dataset_name: str
) -> Dict[str, Any]:
    categories = [
        {
            "id": int(class_id) + 1,
            "name": name,
            "supercategory": "object",
        }
        for class_id, name in sorted({int(t.class_id): t.name for t in targets}.items())
    ]
    images = []
    annotations = []
    ann_id = 1
    for image_id, record in enumerate(records, start=1):
        data = record["data"]
        image_info = data["image"]
        images.append({
            "id": image_id,
            "file_name": image_info["rgb"],
            "width": int(image_info["width"]),
            "height": int(image_info["height"]),
            "frame_id": int(data["frame_id"]),
            "sequence": data["sequence"],
        })
        for ann in data.get("annotations", []):
            x, y, w, h = [float(v) for v in ann["bbox_xywh"]]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(ann["class_id"]) + 1,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
                "track_id": ann.get("track_id"),
                "carla_actor_id": ann.get("carla_actor_id"),
                "annotation_source": ann.get("annotation_source"),
            })
            ann_id += 1
    return {
        "info": {
            "description": dataset_name,
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def write_coco_files(
    out_root: Path,
    records: List[Dict[str, Any]],
    splits: Dict[str, List[Dict[str, Any]]],
    targets: List[TargetClass],
    dataset_name: str
) -> Dict[str, str]:
    coco_dir = out_root / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    all_path = coco_dir / "instances_all.json"
    all_path.write_text(
        json.dumps(build_coco_dict(records, targets, dataset_name), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    paths["all"] = dataset_relative_path(all_path, out_root)
    for split_name, split_records in splits.items():
        split_path = coco_dir / f"instances_{split_name}.json"
        split_path.write_text(
            json.dumps(build_coco_dict(split_records, targets, dataset_name), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        paths[split_name] = dataset_relative_path(split_path, out_root)
    return paths


def compute_dataset_quality_report(
    out_root: Path,
    records: List[Dict[str, Any]],
    road_semantic_ids: List[int]
) -> Dict[str, Any]:
    frame_counts = []
    bbox_w = []
    bbox_h = []
    bbox_area = []
    bbox_max_side = []
    bbox_area_ratio = []
    depth_mean = []
    road_ratios = []
    rgb_small = []

    for record in records:
        data = record["data"]
        anns = data.get("annotations", [])
        frame_counts.append(len(anns))
        width = float(data["image"]["width"])
        height = float(data["image"]["height"])

        rgb_path = out_root / data["image"]["rgb"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
        if rgb is not None:
            rgb_small.append(cv2.resize(rgb, (64, 36)).astype(np.float32) / 255.0)

        segmentation_rel = data["image"].get(
            "segmentation",
            data["image"].get("semantic_id")
        )
        semantic_id = None
        if segmentation_rel is not None:
            semantic_id = cv2.imread(
                str(out_root / segmentation_rel),
                cv2.IMREAD_UNCHANGED
            )
        if semantic_id is not None:
            road_mask = np.isin(
                semantic_id,
                np.array(road_semantic_ids, dtype=np.uint8)
            )
            road_ratios.append(float(np.mean(road_mask)))

        for ann in anns:
            _, _, w, h = [float(v) for v in ann["bbox_xywh"]]
            bbox_w.append(w)
            bbox_h.append(h)
            bbox_area.append(w * h)
            bbox_max_side.append(max(w, h))
            bbox_area_ratio.append((w * h) / (width * height))
            d = ann.get("depth_m", {})
            if d.get("mean") is not None:
                depth_mean.append(float(d["mean"]))

    adjacent_mad = []
    for i in range(1, len(rgb_small)):
        adjacent_mad.append(float(np.mean(np.abs(rgb_small[i] - rgb_small[i - 1]))))

    empty_frames = int(sum(1 for count in frame_counts if count == 0))
    small_96 = float(np.mean(np.array(bbox_max_side) <= 96.0)) if bbox_max_side else 0.0
    small_48 = float(np.mean(np.array(bbox_max_side) <= 48.0)) if bbox_max_side else 0.0

    return {
        "frames": len(records),
        "annotations": int(sum(frame_counts)),
        "empty_frames": empty_frames,
        "empty_frame_ratio": float(empty_frames / max(len(records), 1)),
        "annotations_per_frame": number_stats([float(v) for v in frame_counts]),
        "bbox_width_px": number_stats(bbox_w),
        "bbox_height_px": number_stats(bbox_h),
        "bbox_area_px": number_stats(bbox_area),
        "bbox_max_side_px": number_stats(bbox_max_side),
        "bbox_area_ratio": number_stats(bbox_area_ratio),
        "depth_mean_m": number_stats(depth_mean),
        "road_visible_ratio": number_stats(road_ratios),
        "adjacent_rgb_mean_abs_diff_64x36": number_stats(adjacent_mad),
        "small_target_ratio_max_side_le_96": small_96,
        "very_small_target_ratio_max_side_le_48": small_48,
    }


def write_quality_report_files(
    out_root: Path,
    report: Dict[str, Any]
) -> Dict[str, str]:
    json_path = out_root / "quality_report.json"
    md_path = out_root / "QUALITY_REPORT.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    def format_metric(value: Optional[float], digits: int = 3) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# Dataset Quality Report",
        "",
        f"- Frames: {report['frames']}",
        f"- Annotations: {report['annotations']}",
        f"- Empty frames: {report['empty_frames']} ({report['empty_frame_ratio']:.3f})",
        f"- Small target ratio, max side <= 96 px: {report['small_target_ratio_max_side_le_96']:.3f}",
        f"- Very small target ratio, max side <= 48 px: {report['very_small_target_ratio_max_side_le_48']:.3f}",
        f"- Mean annotations/frame: {format_metric(report['annotations_per_frame']['mean'])}",
        f"- Median bbox max side px: {format_metric(report['bbox_max_side_px']['median'])}",
        f"- Mean road visible ratio: {format_metric(report['road_visible_ratio']['mean'])}",
        f"- Mean adjacent RGB MAD: {format_metric(report['adjacent_rgb_mean_abs_diff_64x36']['mean'])}",
        "",
        "This report is generated automatically after collection.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "quality_report_json": dataset_relative_path(json_path, out_root),
        "quality_report_md": dataset_relative_path(md_path, out_root),
    }


def write_dataset_readme(
    out_root: Path,
    manifest: Dict[str, Any],
    report: Dict[str, Any]
) -> str:
    path = out_root / "DATASET_README.md"
    assignment_policy = manifest.get("weather_policy", {}).get(
        "assignment_policy",
        "one_weather_per_frame"
    )
    random_one_weather = (
        assignment_policy == "balanced_random_one_weather_per_frame"
    )
    rgb_description = (
        "one randomly assigned weather RGB image per independent frame"
        if random_one_weather
        else "weather-organized RGB images"
    )
    weather_note = (
        "Each frame is assigned exactly one weather; weather folders contain disjoint frame ids."
        if random_one_weather
        else "Weather variants may share one frozen camera/actor geometry."
    )
    lines = [
        "# OpenHUTB-CARLA UAV Small-Object Multimodal Dataset",
        "",
        "## Contents",
        "",
        "- `paired_weather/seq_XXXX/`: multimodal collection sequences.",
        f"- `rgb/<preset>/`: {rgb_description}.",
        "- `surface_normal/png/`: camera-space surface-normal PNG.",
        "- `surface_normal/npy/`: camera-space surface normals in float32, channel order x-right/y-down/z-forward.",
        "- `segmentation/id/`: target semantic PNG (0 background, 1 vehicle, 2 pedestrian, 255 ignore).",
        "- `segmentation/color/`: target semantic color preview PNG.",
        "- `depth/npy/`: metric depth arrays in meters, float32.",
        "- `depth/vis_16bit/`: 16-bit depth visualization.",
        "- `depth/color/`: 8-bit color depth preview for QA.",
        "- `depth/lidar/points/`: raw LiDAR point clouds in float32 Nx4 (x, y, z, intensity).",
        "- `depth/lidar/projected_npy/`: RGB-aligned sparse LiDAR camera-forward depth maps in meters, float32; zero means no return.",
        "- `depth/lidar/projected_vis_16bit/`: 16-bit sparse LiDAR depth visualization.",
        "- `depth/lidar/projected_color/`: 8-bit sparse LiDAR depth preview for QA.",
        "- `annotations/`: per-frame JSON annotations.",
        "- `labels_yolo/`: YOLO detection labels.",
        "- `splits/`: train/val/test image lists.",
        "- `coco/`: COCO detection annotations.",
        "",
        "## Summary",
        "",
        f"- Frames: {report['frames']}",
        f"- Annotations: {report['annotations']}",
        f"- Image size: {manifest.get('image_size')}",
        f"- Map: {manifest.get('map')}",
        f"- FOV: {manifest.get('fov')}",
        f"- FPS: {manifest.get('fps')}",
        f"- Weather presets: {manifest.get('weather_policy', {}).get('weather_presets', [])}",
        f"- Empty frame ratio: {report['empty_frame_ratio']:.3f}",
        f"- Small target ratio max side <= 96 px: {report['small_target_ratio_max_side_le_96']:.3f}",
        "",
        "## Notes",
        "",
        "Paths inside JSON/CSV files are relative to the dataset root.",
        weather_note,
        "RGB, depth, normals, segmentation and labels are synchronized for every saved frame.",
        "Camera depth is dense; LiDAR depth is sparse and is projected into RGB pixel coordinates using a co-located sensor transform.",
        "FoggyNoon combines custom CARLA fog parameters with depth-based RGB visibility attenuation for high-altitude views.",
        "SnowNoon combines an overcast CARLA setup with deterministic depth haze and falling-snow RGB rendering; it does not simulate snow accumulation on surfaces.",
        "Surface normals are derived from synchronized metric depth and camera intrinsics; invalid/discontinuity pixels are zero.",
        "COCO category ids are one-based; YOLO class ids remain zero-based.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_relative_path(path, out_root)


def write_config_snapshot(out_root: Path, config_path: Optional[str]) -> Optional[str]:
    if config_path is None or str(config_path).strip() == "":
        return None
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return None
    out_path = out_root / "collection_config_snapshot.json"
    out_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return dataset_relative_path(out_path, out_root)


def write_dataset_standard_artifacts(
    out_root: Path,
    args: argparse.Namespace,
    manifest: Dict[str, Any]
) -> Dict[str, Any]:
    records = read_dataset_frame_records(out_root)
    splits = split_frame_records(records, args.seed)
    split_paths = write_split_files(out_root, splits)
    weather_split_paths = write_weather_split_files(out_root, splits)
    yolo_paths = write_yolo_dataset_files(out_root, args.target, split_paths)
    coco_paths = write_coco_files(
        out_root=out_root,
        records=records,
        splits=splits,
        targets=args.target,
        dataset_name=manifest["dataset"],
    )
    report = compute_dataset_quality_report(out_root, records, args.road_semantic_ids)
    report_paths = write_quality_report_files(out_root, report)
    readme_path = write_dataset_readme(out_root, manifest, report)
    config_snapshot = write_config_snapshot(out_root, args.config)

    return {
        "splits": split_paths,
        "splits_by_weather": weather_split_paths,
        "yolo": yolo_paths,
        "coco": coco_paths,
        "quality": report_paths,
        "readme": readme_path,
        "config_snapshot": config_snapshot,
        "split_counts": {name: len(items) for name, items in splits.items()},
    }


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

    if args.annotation_source != "actor":
        print(
            "[WARN] 四模态采集模式不使用实例分割传感器；"
            "annotation_source 已强制改为 actor。"
        )
        args.annotation_source = "actor"

    if args.pitch_min is None:
        args.pitch_min = args.pitch
    if args.pitch_max is None:
        args.pitch_max = args.pitch
    if args.pitch_min > args.pitch_max:
        args.pitch_min, args.pitch_max = args.pitch_max, args.pitch_min
    if args.height_min > args.height_max:
        args.height_min, args.height_max = args.height_max, args.height_min
    if args.safe_camera_min_z <= 0.0:
        raise ValueError("--safe-camera-min-z 必须大于 0")
    if args.min_target_equivalent_side_px < 0.0:
        raise ValueError("--min-target-equivalent-side-px 不能小于 0")
    if args.min_pedestrian_equivalent_side_px < 0.0:
        raise ValueError("--min-pedestrian-equivalent-side-px 不能小于 0")
    if args.min_pedestrians_per_frame < 0:
        raise ValueError("--min-pedestrians-per-frame 不能小于 0")
    if args.annotation_boundary_margin_px < 0:
        raise ValueError("--annotation-boundary-margin-px 不能小于 0")
    if not 0.0 <= args.pedestrian_centered_camera_probability <= 1.0:
        raise ValueError(
            "--pedestrian-centered-camera-probability 必须在 0 到 1 之间"
        )
    if not 0.0 <= args.vehicle_centered_camera_probability <= 1.0:
        raise ValueError(
            "--vehicle-centered-camera-probability 必须在 0 到 1 之间"
        )
    if (
        args.pedestrian_centered_camera_probability
        + args.vehicle_centered_camera_probability
        > 1.0
    ):
        raise ValueError("行人中心与车辆中心视角概率之和不能超过 1")
    for value, name in (
        (
            args.min_vehicle_projected_fill_ratio,
            "--min-vehicle-projected-fill-ratio",
        ),
        (
            args.min_pedestrian_projected_fill_ratio,
            "--min-pedestrian-projected-fill-ratio",
        ),
        (
            args.min_largest_component_ratio,
            "--min-largest-component-ratio",
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须在 0 到 1 之间")
    released_classes = {
        (target_mask_id_for_class(target.name), int(target.class_id))
        for target in args.target
    }
    if released_classes != {
        (TARGET_VEHICLE_ID, 0),
        (TARGET_PEDESTRIAN_ID, 1),
    }:
        raise ValueError(
            "当前总采集器只允许 vehicle:0:10 和 pedestrian:1:4 两类。"
        )
    if args.target_actor_filter or args.target_actor_count:
        raise ValueError("当前总采集器已禁用车辆/行人以外的自定义目标。")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = try_load_world(client, args.map)

    original_settings = world.get_settings()
    original_weather = world.get_weather()

    fixed_delta_seconds = 1.0 / args.fps

    # 这些 actor 后面 cleanup 要销毁
    traffic_actors: List[carla.Actor] = []
    vehicle_actors: List[carla.Actor] = []
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
        destroy_live_two_wheel_vehicles(world)
        if args.hide_static_map_vehicles:
            hidden_static_vehicle_ids = hide_static_map_vehicles(world)

        # ------------------------------------------------------------
        # 天气
        # ------------------------------------------------------------
        weather_names = get_weather_names()
        collection_jobs: List[Dict[str, Any]] = []
        existing_sequence_metadata: List[Dict[str, Any]] = []

        random_weather_per_frame = bool(
            args.random_weather
            and not args.keep_current_weather
            and not args.collect_all_weather_presets
        )
        if args.collect_all_weather_presets or random_weather_per_frame:
            selected_weather_names = args.weather_presets or weather_names
        elif args.keep_current_weather:
            selected_weather_names = ["current"]
        else:
            selected_weather_names = [args.weather]

        unknown_weather_names = sorted(
            set(selected_weather_names) - set(weather_names) - {"current"}
        )
        if unknown_weather_names:
            raise RuntimeError(
                "当前 OpenHUTB/CARLA API 不支持这些天气预设："
                + ", ".join(unknown_weather_names)
            )
        if not selected_weather_names:
            raise RuntimeError("天气候选列表不能为空。")

        paired_root = out_root / "paired_weather"
        paired_root.mkdir(parents=True, exist_ok=True)
        if args.preserve_existing_sequences:
            _, legacy_metadata = existing_sequence_state(out_root)
            start_index, paired_metadata = existing_sequence_state(paired_root)
            existing_sequence_metadata = legacy_metadata + paired_metadata
        else:
            start_index = 0

        for local_seq_idx in range(args.sequences):
            if random_weather_per_frame:
                weather_schedule = [
                    selected_weather_names[index % len(selected_weather_names)]
                    for index in range(args.frames)
                ]
                random.shuffle(weather_schedule)
            else:
                weather_schedule = []
            collection_jobs.append({
                "weather_candidates": list(selected_weather_names),
                "weather_schedule": weather_schedule,
                "sequence_index": start_index + local_seq_idx,
                "sequence_root": paired_root
            })

        rgb_images_per_frame = (
            len(selected_weather_names)
            if args.collect_all_weather_presets
            else 1
        )
        print(
            f"[INFO] sequences={len(collection_jobs)}, "
            f"weather_candidates={len(selected_weather_names)}, "
            f"weather_policy={'balanced_random_per_frame' if random_weather_per_frame else ('all_paired' if args.collect_all_weather_presets else 'single')}, "
            f"geometry_frames={len(collection_jobs) * args.frames}, "
            f"rgb_images={len(collection_jobs) * args.frames * rgb_images_per_frame}"
        )

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
            vehicle_actors.extend(vehicles)
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

        sensors = dict(camera_sensors)
        if args.enable_lidar:
            sensors["lidar"] = spawn_lidar_sensor(
                world=world,
                sensor_tick=sensor_tick,
                initial_transform=initial_transform,
                channels=args.lidar_channels,
                lidar_range=args.lidar_range,
                points_per_second=args.lidar_points_per_second,
                rotation_frequency=args.lidar_rotation_frequency,
                upper_fov=args.lidar_upper_fov,
                lower_fov=args.lidar_lower_fov
            )

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
                "min_target_equivalent_side_px": (
                    args.min_target_equivalent_side_px
                ),
                "min_pedestrian_equivalent_side_px": (
                    args.min_pedestrian_equivalent_side_px
                ),
                "min_pedestrians_per_frame": args.min_pedestrians_per_frame,
                "reject_boundary_annotations": args.reject_boundary_annotations,
                "annotation_boundary_margin_px": (
                    args.annotation_boundary_margin_px
                ),
                "keep_all": args.keep_all
            },
            "annotation_policy": {
                "source": args.annotation_source,
                "actor_classes": "vehicle.* and walker.pedestrian.* use OpenHUTB/CARLA actor coordinates plus semantic/depth visibility filtering",
                "object_unit": "one OpenHUTB/CARLA actor bbox per annotation",
                "bbox": "visible actor pixels filtered by metric depth",
                "actor_visibility": {
                    "mode": args.actor_visibility_mode,
                    "min_actor_visible_px": args.min_actor_visible_px,
                    "min_actor_visible_ratio": args.min_actor_visible_ratio,
                    "min_vehicle_projected_fill_ratio": (
                        args.min_vehicle_projected_fill_ratio
                    ),
                    "min_pedestrian_projected_fill_ratio": (
                        args.min_pedestrian_projected_fill_ratio
                    ),
                    "actor_depth_margin": args.actor_depth_margin
                },
                "target_semantic": {
                    "internal_sensor": "sensor.camera.instance_segmentation",
                    "published_classes": ["vehicle", "pedestrian"],
                    "mask_values": {
                        "background": TARGET_BACKGROUND_ID,
                        "vehicle": TARGET_VEHICLE_ID,
                        "pedestrian": TARGET_PEDESTRIAN_ID,
                        "ignore": TARGET_IGNORE_ID,
                    },
                    "min_largest_component_ratio": (
                        args.min_largest_component_ratio
                    ),
                    "training_mask_source": (
                        "filled_largest_external_instance_contour"
                    ),
                },
                "depth": "metric full-frame depth"
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
                "camera_origin_over_road": args.camera_origin_over_road,
                "pedestrian_centered_camera_probability": (
                    args.pedestrian_centered_camera_probability
                ),
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
                "collect_all_weather_presets": args.collect_all_weather_presets,
                "weather_presets": list(selected_weather_names),
                "weather_rendering_definitions": {
                    name: weather_rendering_metadata(name)
                    for name in selected_weather_names
                },
                "directory_layout": "paired_weather/seq_XXXX/rgb/<preset>/frame.png",
                "assignment_policy": (
                    "balanced_random_one_weather_per_frame"
                    if random_weather_per_frame
                    else (
                        "all_weather_variants_share_one_geometry"
                        if args.collect_all_weather_presets
                        else "one_weather_per_frame"
                    )
                ),
                "rgb_images_per_geometry_frame": (
                    len(selected_weather_names)
                    if args.collect_all_weather_presets
                    else 1
                ),
                "weather_warmup_frames": args.weather_warmup_frames,
                "keep_current_weather": args.keep_current_weather,
                "random_weather": args.random_weather,
                "configured_weather": args.weather,
                "weather_parameters": weather_to_dict(world.get_weather())
            },
            "modalities": [
                "rgb_visible",
                "surface_normal",
                "target_semantic_segmentation",
                "metric_depth"
            ],
            "depth_sources": {
                "camera_depth": {
                    "dense": True,
                    "unit": "meters",
                    "aligned_to_rgb": True
                },
                "lidar": {
                    "enabled": args.enable_lidar,
                    "dense": False,
                    "projected_value": "camera-forward depth",
                    "unit": "meters",
                    "channels": args.lidar_channels,
                    "range_m": args.lidar_range,
                    "points_per_second": args.lidar_points_per_second,
                    "rotation_frequency_hz": args.lidar_rotation_frequency,
                    "upper_fov_degrees": args.lidar_upper_fov,
                    "lower_fov_degrees": args.lidar_lower_fov,
                    "aligned_to_rgb": True
                }
            },
            "sequences": existing_sequence_metadata
        }

        # ------------------------------------------------------------
        # sequence 循环
        # ------------------------------------------------------------
        total_drop_frames = 0
        total_saved_frames = 0

        for job in collection_jobs:
            seq_idx = int(job["sequence_index"])
            seq_name = f"seq_{seq_idx:04d}"
            seq_dir = Path(job["sequence_root"]) / seq_name
            dirs = make_dirs(seq_dir)
            weather_candidates = [
                str(name) for name in job["weather_candidates"]
            ]
            weather_schedule = [
                str(name) for name in job.get("weather_schedule", [])
            ]
            sequence_canonical_weather = (
                None
                if random_weather_per_frame or args.collect_all_weather_presets
                else weather_candidates[0]
            )
            rgb_weather_dirs = {
                weather_name: dirs["rgb"] / weather_name
                for weather_name in weather_candidates
            }
            for rgb_weather_dir in rgb_weather_dirs.values():
                rgb_weather_dir.mkdir(parents=True, exist_ok=True)
            sequence_id = dataset_relative_path(seq_dir, out_root)

            orbit_phase = random.uniform(0.0, 2.0 * math.pi)
            orbit_height = max(args.height, args.safe_camera_min_z)
            orbit_radius = ground_aim_radius(
                orbit_height,
                args.pitch,
                args.radius_min,
                args.radius_max
            )
            if args.road_centered_camera:
                base_center_x, base_center_y = random_road_center(world.get_map())
            else:
                base_center_x, base_center_y = args.center_x, args.center_y

            seq_meta = {
                "sequence": sequence_id,
                "sequence_name": seq_name,
                "relative_sequence_path": sequence_id,
                "paired_weather_rgb": bool(
                    args.collect_all_weather_presets
                    and len(weather_candidates) > 1
                ),
                "four_modalities_complete": True,
                "modalities": [
                    "rgb_visible",
                    "surface_normal",
                    "target_semantic_segmentation",
                    "metric_depth"
                ],
                "weather_candidates": weather_candidates,
                "weather_assignment_policy": (
                    "balanced_random_one_weather_per_frame"
                    if random_weather_per_frame
                    else (
                        "all_paired"
                        if args.collect_all_weather_presets
                        else "single"
                    )
                ),
                "planned_weather_counts": dict(Counter(weather_schedule)),
                "canonical_weather": sequence_canonical_weather,
                "weather_warmup_frames": args.weather_warmup_frames,
                "route": args.route,
                "center_xy": [base_center_x, base_center_y],
                "road_centered_camera": args.road_centered_camera,
                "camera_origin_over_road": args.camera_origin_over_road,
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
                    "rgb_canonical",
                    "rgb_by_weather_json",
                    "depth_npy",
                    "depth_vis",
                    "depth_color",
                    "lidar_points",
                    "lidar_projected_npy",
                    "lidar_projected_vis",
                    "lidar_projected_color",
                    "surface_normal_npy",
                    "surface_normal",
                    "segmentation",
                    "segmentation_color",
                    "num_annotations"
                ])

                sot_candidates: List[List[Dict[str, Any]]] = []
                captured_weather_counts: Counter = Counter()
                active_weather_name: Optional[str] = None

                for frame_i in range(args.frames):
                    if random_weather_per_frame:
                        frame_weather_variants = [weather_schedule[frame_i]]
                    else:
                        frame_weather_variants = list(weather_candidates)
                    canonical_weather = (
                        "ClearNoon"
                        if "ClearNoon" in frame_weather_variants
                        else frame_weather_variants[0]
                    )

                    # 单天气模式在四模态同步采集前切换天气。每个场景只拍一次，
                    # RGB、深度、语义和法线因此对应同一个仿真时刻。
                    if not args.collect_all_weather_presets:
                        if active_weather_name != canonical_weather:
                            active_weather_name = apply_weather(
                                world,
                                canonical_weather
                            )
                            if (
                                active_weather_name != "current"
                                and args.weather_warmup_frames > 0
                            ):
                                for _ in range(args.weather_warmup_frames):
                                    world.tick()
                                for sensor_sync in sync.values():
                                    sensor_sync.drain()

                    # ------------------------------------------------
                    # 生成 UAV 相机位姿，并过滤穿模/近距离遮挡视角。
                    # ------------------------------------------------
                    accepted_view = False
                    last_bad_view_stats: Optional[Dict[str, Any]] = None
                    best_view_stats: Optional[Dict[str, Any]] = None
                    accepted_annotations: Optional[List[Dict[str, Any]]] = None
                    accepted_semantic_sensor_id: Optional[np.ndarray] = None
                    accepted_target_semantic_id: Optional[np.ndarray] = None
                    accepted_target_instances: Optional[List[Dict[str, Any]]] = None

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

                            if args.camera_origin_over_road:
                                centered_draw = random.random()
                                use_pedestrian_center = bool(
                                    walker_actors
                                    and centered_draw
                                    < args.pedestrian_centered_camera_probability
                                )
                                use_vehicle_center = bool(
                                    vehicle_actors
                                    and not use_pedestrian_center
                                    and centered_draw
                                    < (
                                        args.pedestrian_centered_camera_probability
                                        + args.vehicle_centered_camera_probability
                                    )
                                )
                                if use_pedestrian_center:
                                    transform = (
                                        random_pedestrian_centered_road_uav_transform(
                                            carla_map=world.get_map(),
                                            pedestrian_actors=walker_actors,
                                            height_min=max(
                                                args.height_min,
                                                args.safe_camera_min_z
                                            ),
                                            height_max=max(
                                                args.height_max,
                                                args.safe_camera_min_z
                                            ),
                                            radius_min=args.radius_min,
                                            radius_max=args.radius_max,
                                            pitch_min=args.pitch_min,
                                            pitch_max=args.pitch_max
                                        )
                                    )
                                elif use_vehicle_center:
                                    transform = (
                                        random_pedestrian_centered_road_uav_transform(
                                            carla_map=world.get_map(),
                                            pedestrian_actors=vehicle_actors,
                                            height_min=max(
                                                args.height_min,
                                                args.safe_camera_min_z
                                            ),
                                            height_max=max(
                                                args.height_max,
                                                args.safe_camera_min_z
                                            ),
                                            radius_min=args.radius_min,
                                            radius_max=args.radius_max,
                                            pitch_min=args.pitch_min,
                                            pitch_max=args.pitch_max
                                        )
                                    )
                                else:
                                    transform = random_road_uav_transform(
                                        carla_map=world.get_map(),
                                        height_min=max(
                                            args.height_min,
                                            args.safe_camera_min_z
                                        ),
                                        height_max=max(
                                            args.height_max,
                                            args.safe_camera_min_z
                                        ),
                                        radius_min=args.radius_min,
                                        radius_max=args.radius_max,
                                        pitch_min=args.pitch_min,
                                        pitch_max=args.pitch_max
                                    )
                            else:
                                transform = random_uav_transform(
                                    center_x=center_x,
                                    center_y=center_y,
                                    height_min=max(
                                        args.height_min,
                                        args.safe_camera_min_z
                                    ),
                                    height_max=max(
                                        args.height_max,
                                        args.safe_camera_min_z
                                    ),
                                    radius_min=args.radius_min,
                                    radius_max=args.radius_max,
                                    pitch_min=args.pitch_min,
                                    pitch_max=args.pitch_max
                                )

                        set_all_sensor_transform(sensors, transform)
                        carla_frame = world.tick()

                        lidar_data = None
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
                            if args.enable_lidar:
                                lidar_data = sync["lidar"].get(
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
                        near_depth_bad_view, bad_view_stats = is_bad_camera_view(
                            depth_m,
                            min_near_depth_m=args.min_near_depth_m,
                            max_near_depth_ratio=args.max_near_depth_ratio
                        )
                        bad_view_stats["road_visible_ratio"] = road_visible_ratio(
                            semantic_img,
                            args.road_semantic_ids
                        )
                        candidate_semantic_sensor_id = decode_semantic_segmentation(
                            semantic_img
                        )
                        candidate_annotations = build_annotations_from_actors(
                            world=world,
                            camera_transform=transform,
                            depth_m=depth_m,
                            semantic_id=candidate_semantic_sensor_id,
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
                            min_vehicle_projected_fill_ratio=(
                                args.min_vehicle_projected_fill_ratio
                            ),
                            min_pedestrian_projected_fill_ratio=(
                                args.min_pedestrian_projected_fill_ratio
                            ),
                            actor_depth_margin=args.actor_depth_margin,
                            actor_visibility_mode=args.actor_visibility_mode
                        )
                        boundary_margin = args.annotation_boundary_margin_px
                        filtered_boundary_annotation_count = sum(
                            int(annotation["bbox_xywh"][0]) <= boundary_margin
                            or int(annotation["bbox_xywh"][1]) <= boundary_margin
                            or (
                                int(annotation["bbox_xywh"][0])
                                + int(annotation["bbox_xywh"][2])
                                >= args.width - boundary_margin
                            )
                            or (
                                int(annotation["bbox_xywh"][1])
                                + int(annotation["bbox_xywh"][3])
                                >= args.height_img - boundary_margin
                            )
                            for annotation in candidate_annotations
                        )
                        if (
                            args.reject_boundary_annotations
                            and filtered_boundary_annotation_count
                        ):
                            candidate_annotations = [
                                annotation
                                for annotation in candidate_annotations
                                if not (
                                    int(annotation["bbox_xywh"][0])
                                    <= boundary_margin
                                    or int(annotation["bbox_xywh"][1])
                                    <= boundary_margin
                                    or (
                                        int(annotation["bbox_xywh"][0])
                                        + int(annotation["bbox_xywh"][2])
                                        >= args.width - boundary_margin
                                    )
                                    or (
                                        int(annotation["bbox_xywh"][1])
                                        + int(annotation["bbox_xywh"][3])
                                        >= args.height_img - boundary_margin
                                    )
                                )
                            ]
                        (
                            candidate_target_semantic_id,
                            candidate_target_instances,
                        ) = build_optimized_target_semantic_mask(
                            instance_image=instance_img,
                            actor_annotations=candidate_annotations,
                            min_mask_px=args.min_mask_px,
                            min_vehicle_projected_fill_ratio=(
                                args.min_vehicle_projected_fill_ratio
                            ),
                            min_pedestrian_projected_fill_ratio=(
                                args.min_pedestrian_projected_fill_ratio
                            ),
                            min_vehicle_visible_equivalent_side_px=(
                                args.min_vehicle_visible_equivalent_side_px
                            ),
                            min_pedestrian_visible_equivalent_side_px=(
                                args.min_pedestrian_visible_equivalent_side_px
                            ),
                            min_largest_component_ratio=(
                                args.min_largest_component_ratio
                            ),
                        )
                        target_vehicle_count = int(
                            np.count_nonzero(
                                candidate_target_semantic_id
                                == TARGET_VEHICLE_ID
                            )
                            > 0
                        )
                        target_pedestrian_count = int(
                            np.count_nonzero(
                                candidate_target_semantic_id
                                == TARGET_PEDESTRIAN_ID
                            )
                            > 0
                        )
                        bad_view_stats["target_vehicle_present"] = bool(
                            target_vehicle_count
                        )
                        bad_view_stats["target_pedestrian_present"] = bool(
                            target_pedestrian_count
                        )
                        bad_view_stats["target_trainable_instances"] = int(
                            sum(
                                bool(instance["trainable"])
                                for instance in candidate_target_instances
                            )
                        )
                        bad_view_stats["target_unmatched_instances"] = int(
                            sum(
                                not bool(instance["trainable"])
                                and instance["carla_actor_id"] is None
                                for instance in candidate_target_instances
                            )
                        )
                        equivalent_sides = [
                            math.sqrt(
                                max(0.0, float(ann["bbox_xywh"][2]))
                                * max(0.0, float(ann["bbox_xywh"][3]))
                            )
                            for ann in candidate_annotations
                        ]
                        pedestrian_equivalent_sides = [
                            side
                            for ann, side in zip(
                                candidate_annotations,
                                equivalent_sides
                            )
                            if ann.get("class_name") == "pedestrian"
                        ]
                        min_equivalent_side = (
                            min(equivalent_sides) if equivalent_sides else None
                        )
                        tiny_target_count = sum(
                            side <= args.min_target_equivalent_side_px
                            for side in equivalent_sides
                        ) if args.min_target_equivalent_side_px > 0.0 else 0
                        undersized_pedestrian_count = sum(
                            side < args.min_pedestrian_equivalent_side_px
                            for side in pedestrian_equivalent_sides
                        ) if args.min_pedestrian_equivalent_side_px > 0.0 else 0
                        boundary_margin = args.annotation_boundary_margin_px
                        boundary_annotation_count = sum(
                            int(ann["bbox_xywh"][0]) <= boundary_margin
                            or int(ann["bbox_xywh"][1]) <= boundary_margin
                            or (
                                int(ann["bbox_xywh"][0])
                                + int(ann["bbox_xywh"][2])
                                >= args.width - boundary_margin
                            )
                            or (
                                int(ann["bbox_xywh"][1])
                                + int(ann["bbox_xywh"][3])
                                >= args.height_img - boundary_margin
                            )
                            for ann in candidate_annotations
                        )
                        bad_view_stats["annotation_count"] = len(
                            candidate_annotations
                        )
                        bad_view_stats["min_target_equivalent_side_px"] = (
                            min_equivalent_side
                        )
                        bad_view_stats["tiny_target_count"] = int(
                            tiny_target_count
                        )
                        bad_view_stats["pedestrian_count"] = len(
                            pedestrian_equivalent_sides
                        )
                        bad_view_stats[
                            "min_pedestrian_equivalent_side_px"
                        ] = (
                            min(pedestrian_equivalent_sides)
                            if pedestrian_equivalent_sides
                            else None
                        )
                        bad_view_stats["undersized_pedestrian_count"] = int(
                            undersized_pedestrian_count
                        )
                        bad_view_stats["boundary_annotation_count"] = int(
                            boundary_annotation_count
                        )
                        bad_view_stats[
                            "ignored_boundary_annotation_count"
                        ] = int(filtered_boundary_annotation_count)
                        bad_view = bool(
                            near_depth_bad_view
                            or tiny_target_count > 0
                            or undersized_pedestrian_count > 0
                            or len(pedestrian_equivalent_sides)
                            < args.min_pedestrians_per_frame
                            or target_vehicle_count == 0
                            or target_pedestrian_count == 0
                            or (
                                args.reject_boundary_annotations
                                and boundary_annotation_count > 0
                            )
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
                            accepted_annotations = candidate_annotations
                            accepted_semantic_sensor_id = (
                                candidate_semantic_sensor_id
                            )
                            accepted_target_semantic_id = (
                                candidate_target_semantic_id
                            )
                            accepted_target_instances = (
                                candidate_target_instances
                            )
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

                    if (
                        accepted_annotations is None
                        or accepted_semantic_sensor_id is None
                        or accepted_target_semantic_id is None
                        or accepted_target_instances is None
                    ):
                        raise RuntimeError(
                            f"frame {frame_i} 已接受位姿但缺少同步标注或语义图"
                        )
                    anns = accepted_annotations
                    semantic_sensor_id = accepted_semantic_sensor_id
                    target_semantic_id = accepted_target_semantic_id
                    target_semantic_instances = accepted_target_instances

                    # ------------------------------------------------
                    # 文件路径
                    # ------------------------------------------------
                    stem = f"{frame_i:06d}"

                    rgb_paths_by_weather = {
                        weather_name: rgb_weather_dirs[weather_name] / f"{stem}.png"
                        for weather_name in frame_weather_variants
                    }
                    rgb_path = rgb_paths_by_weather[canonical_weather]
                    depth_npy_path = dirs["depth_npy"] / f"{stem}.npy"
                    depth_vis_path = dirs["depth_vis"] / f"{stem}.png"
                    depth_color_path = dirs["depth_color"] / f"{stem}.png"
                    lidar_points_path = dirs["lidar_points"] / f"{stem}.npy"
                    lidar_projected_npy_path = dirs["lidar_projected_npy"] / f"{stem}.npy"
                    lidar_projected_vis_path = dirs["lidar_projected_vis"] / f"{stem}.png"
                    lidar_projected_color_path = dirs["lidar_projected_color"] / f"{stem}.png"
                    surface_normal_npy_path = dirs["surface_normal_npy"] / f"{stem}.npy"
                    surface_normal_path = dirs["surface_normal"] / f"{stem}.png"
                    segmentation_path = dirs["segmentation"] / f"{stem}.png"
                    segmentation_color_path = dirs["segmentation_color"] / f"{stem}.png"
                    yolo_path = dirs["yolo"] / f"{stem}.txt"
                    ann_path = dirs["ann"] / f"{stem}.json"

                    rgb_images_by_weather: Dict[str, carla.Image] = {}
                    weather_capture_metadata: Dict[str, Dict[str, Any]] = {}
                    if args.collect_all_weather_presets:
                        # 兼容旧的配对天气模式：仅在显式开启时冻结场景并循环天气。
                        frozen_snapshots: List[Dict[str, Any]] = []
                        traffic_lights_frozen = False
                        try:
                            frozen_snapshots = freeze_dynamic_actors_for_weather_sweep(
                                vehicle_actors=vehicle_actors,
                                walker_actors=walker_actors,
                                walker_controllers=walker_controllers,
                                tm_port=args.tm_port
                            )
                            if hasattr(world, "freeze_all_traffic_lights"):
                                world.freeze_all_traffic_lights(True)
                                traffic_lights_frozen = True

                            for weather_variant in frame_weather_variants:
                                restore_frozen_actor_transforms(frozen_snapshots)
                                applied_weather = apply_weather(world, weather_variant)
                                if args.weather_warmup_frames > 0:
                                    for _ in range(args.weather_warmup_frames):
                                        world.tick()
                                    for sensor_sync in sync.values():
                                        sensor_sync.drain()

                                restore_frozen_actor_transforms(frozen_snapshots)
                                rgb_carla_frame = world.tick()
                                weather_rgb_img = sync["rgb"].get(
                                    rgb_carla_frame,
                                    timeout=args.sensor_timeout
                                )
                                sync["depth"].get(
                                    rgb_carla_frame,
                                    timeout=args.sensor_timeout
                                )
                                sync["semantic"].get(
                                    rgb_carla_frame,
                                    timeout=args.sensor_timeout
                                )
                                sync["instance"].get(
                                    rgb_carla_frame,
                                    timeout=args.sensor_timeout
                                )
                                if args.enable_lidar:
                                    sync["lidar"].get(
                                        rgb_carla_frame,
                                        timeout=args.sensor_timeout
                                    )

                                rgb_images_by_weather[weather_variant] = weather_rgb_img
                                weather_capture_metadata[weather_variant] = {
                                    "applied_weather": applied_weather,
                                    "carla_frame": int(rgb_carla_frame),
                                    "timestamp": float(weather_rgb_img.timestamp),
                                    "weather_parameters": weather_to_dict(world.get_weather()),
                                    "rendering": weather_rendering_metadata(weather_variant)
                                }
                        finally:
                            if traffic_lights_frozen:
                                try:
                                    world.freeze_all_traffic_lights(False)
                                except Exception:
                                    pass
                            resume_dynamic_actors_after_weather_sweep(
                                world=world,
                                snapshots=frozen_snapshots,
                                walker_controllers=walker_controllers,
                                tm_port=args.tm_port
                            )
                    else:
                        # 随机/固定单天气直接使用已通过视角检查的同步 RGB。
                        rgb_images_by_weather[canonical_weather] = rgb_img
                        weather_capture_metadata[canonical_weather] = {
                            "applied_weather": active_weather_name,
                            "carla_frame": int(carla_frame),
                            "timestamp": float(rgb_img.timestamp),
                            "weather_parameters": weather_to_dict(world.get_weather()),
                            "rendering": weather_rendering_metadata(canonical_weather)
                        }

                    if len(rgb_images_by_weather) != len(frame_weather_variants):
                        raise RuntimeError(
                            f"frame {frame_i} RGB 天气采集不完整："
                            f"{len(rgb_images_by_weather)}/{len(frame_weather_variants)}"
                        )

                    for weather_variant, weather_rgb_img in rgb_images_by_weather.items():
                        save_rgb(
                            weather_rgb_img,
                            rgb_paths_by_weather[weather_variant],
                            weather_name=weather_variant,
                            depth_m=depth_m,
                            random_seed=(args.seed * 1000003 + seq_idx * 1009 + frame_i)
                        )
                    captured_weather_counts.update(rgb_images_by_weather.keys())

                    save_depth(
                        depth_m,
                        depth_npy_path,
                        depth_vis_path,
                        depth_color_path,
                        args.max_depth_vis
                    )
                    lidar_metadata = None
                    if args.enable_lidar:
                        if lidar_data is None:
                            raise RuntimeError(
                                f"frame {frame_i} 缺少同步 LiDAR 数据。"
                            )
                        lidar_metadata = save_lidar_depth(
                            lidar_data=lidar_data,
                            camera_data=depth_img,
                            points_path=lidar_points_path,
                            projected_npy_path=lidar_projected_npy_path,
                            projected_vis_path=lidar_projected_vis_path,
                            projected_color_path=lidar_projected_color_path,
                            width=args.width,
                            height=args.height_img,
                            fov_degrees=args.fov,
                            max_range_m=args.lidar_range,
                            max_depth_vis_m=args.max_depth_vis
                        )
                    save_surface_normal(
                        depth_m=depth_m,
                        npy_path=surface_normal_npy_path,
                        png_path=surface_normal_path,
                        fov_degrees=args.fov,
                        max_depth_jump_m=args.normal_max_depth_jump_m
                    )
                    save_semantic_id(target_semantic_id, segmentation_path)
                    cv2.imwrite(
                        str(segmentation_color_path),
                        colorize_target_semantic_mask(target_semantic_id),
                    )

                    save_yolo_label(
                        yolo_path,
                        anns,
                        width=args.width,
                        height=args.height_img
                    )

                    # ------------------------------------------------
                    # JSON 标注
                    # ------------------------------------------------
                    ann_json = {
                        "sequence": sequence_id,
                        "frame_id": frame_i,
                        "carla_frame": int(carla_frame),
                        "timestamp": float(depth_img.timestamp),
                        "paired_weather_rgb": bool(
                            args.collect_all_weather_presets
                            and len(frame_weather_variants) > 1
                        ),
                        "canonical_weather": canonical_weather,
                        "weather_variants": frame_weather_variants,
                        "weather_captures": weather_capture_metadata,
                        "camera_transform": transform_to_dict(transform),
                        "image": {
                            "width": args.width,
                            "height": args.height_img,
                            "rgb": dataset_relative_path(rgb_path, out_root),
                            "rgb_by_weather": {
                                weather_variant: dataset_relative_path(path, out_root)
                                for weather_variant, path in rgb_paths_by_weather.items()
                            },
                            "depth_npy_meters": dataset_relative_path(depth_npy_path, out_root),
                            "depth_vis_16bit": dataset_relative_path(depth_vis_path, out_root),
                            "depth_color": dataset_relative_path(depth_color_path, out_root),
                            "lidar_points": (
                                dataset_relative_path(lidar_points_path, out_root)
                                if args.enable_lidar else None
                            ),
                            "lidar_projected_npy_meters": (
                                dataset_relative_path(lidar_projected_npy_path, out_root)
                                if args.enable_lidar else None
                            ),
                            "lidar_projected_vis_16bit": (
                                dataset_relative_path(lidar_projected_vis_path, out_root)
                                if args.enable_lidar else None
                            ),
                            "lidar_projected_color": (
                                dataset_relative_path(lidar_projected_color_path, out_root)
                                if args.enable_lidar else None
                            ),
                            "surface_normal_npy": dataset_relative_path(surface_normal_npy_path, out_root),
                            "surface_normal": dataset_relative_path(surface_normal_path, out_root),
                            "segmentation": dataset_relative_path(segmentation_path, out_root),
                            "segmentation_color": dataset_relative_path(segmentation_color_path, out_root),
                            "yolo": dataset_relative_path(yolo_path, out_root)
                        },
                        "lidar": lidar_metadata,
                        "target_semantic": {
                            "mask_values": {
                                "background": TARGET_BACKGROUND_ID,
                                "vehicle": TARGET_VEHICLE_ID,
                                "pedestrian": TARGET_PEDESTRIAN_ID,
                                "ignore": TARGET_IGNORE_ID,
                            },
                            "training_mask_source": (
                                "filled_largest_external_instance_contour"
                            ),
                            "instances": target_semantic_instances,
                        },
                        "annotations": anns
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
                        dataset_relative_path(rgb_path, out_root),
                        json.dumps(
                            {
                                weather_variant: dataset_relative_path(path, out_root)
                                for weather_variant, path in rgb_paths_by_weather.items()
                            },
                            ensure_ascii=False
                        ),
                        dataset_relative_path(depth_npy_path, out_root),
                        dataset_relative_path(depth_vis_path, out_root),
                        dataset_relative_path(depth_color_path, out_root),
                        dataset_relative_path(lidar_points_path, out_root) if args.enable_lidar else "",
                        dataset_relative_path(lidar_projected_npy_path, out_root) if args.enable_lidar else "",
                        dataset_relative_path(lidar_projected_vis_path, out_root) if args.enable_lidar else "",
                        dataset_relative_path(lidar_projected_color_path, out_root) if args.enable_lidar else "",
                        dataset_relative_path(surface_normal_npy_path, out_root),
                        dataset_relative_path(surface_normal_path, out_root),
                        dataset_relative_path(segmentation_path, out_root),
                        dataset_relative_path(segmentation_color_path, out_root),
                        len(anns)
                    ])
                    total_saved_frames += 1

                    if frame_i % 20 == 0:
                        print(
                            f"[{sequence_id}] frame={frame_i:06d}, "
                            f"carla_frame={carla_frame}, "
                            f"anns={len(anns)}, "
                            f"weather={canonical_weather}, "
                            f"weather_rgb={len(frame_weather_variants)}"
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
                seq_meta["captured_weather_counts"] = dict(
                    captured_weather_counts
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

        all_frame_records = read_dataset_frame_records(out_root)
        total_weather_rgb_images = sum(
            len(record["data"].get("image", {}).get("rgb_by_weather", {}))
            for record in all_frame_records
        )
        manifest["total_drop_frames_this_run"] = total_drop_frames
        manifest["total_saved_frames_this_run"] = total_saved_frames
        manifest["total_saved_frames"] = len(all_frame_records)
        manifest["total_weather_rgb_images"] = total_weather_rgb_images
        manifest["standard_artifacts"] = write_dataset_standard_artifacts(
            out_root=out_root,
            args=args,
            manifest=manifest
        )

        (out_root / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"[DONE] Dataset saved to: {out_root.resolve()}")
        print(f"[DONE] total_saved_frames_this_run={total_saved_frames}")
        print(f"[DONE] total_saved_frames_all_sequences={len(all_frame_records)}")
        print(f"[DONE] total_weather_rgb_images={total_weather_rgb_images}")
        print(f"[DONE] total_drop_frames_this_run={total_drop_frames}")

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
            world.set_weather(original_weather)
            print("[INFO] Restored simulator weather used before collection.")
        except Exception as exc:
            print(f"[WARN] Failed to restore original weather: {exc}")

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("[CLEANUP] Finished.")


if __name__ == "__main__":
    main()
