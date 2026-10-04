#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""连接 OpenHUTB/CARLA 模拟器并查看或切换地图。

示例：
    直接运行：使用下方 TARGET_MAP_NAME 配置的地图
    python switch_carla_map.py
    python switch_carla_map.py --list
    python switch_carla_map.py --current
    python switch_carla_map.py Town15  # 临时覆盖代码中的地图配置
    python switch_carla_map.py Town10HD --host 127.0.0.1 --port 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import PurePosixPath
from typing import Dict, List

try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "无法导入 carla。请使用 openhutb 环境运行，或把 OpenHUTB PythonAPI "
        "加入 PYTHONPATH。"
    ) from exc


# ======================== 地图配置区 ========================
# 在这里修改要切换的地图。直接运行本文件时会加载这张地图。
TARGET_MAP_NAME = "HutbCarlaCity"
# Town12/Town15 等大地图首次加载较慢，建议至少设置为 300 秒。
RPC_TIMEOUT_SECONDS = 300.0
# True：检测到正在主动控制的无人机或 hero/ego/player 载具时取消换图。
# CARLA 换地图会重建整个 World，无法原样保留当前飞行器 actor。
PROTECT_ACTIVE_UAV = True
# ============================================================


UAV_TYPE_KEYWORDS = ("drone", "uav", "quadcopter", "quadrotor", "aircraft")
ACTIVE_CONTROL_ROLE_NAMES = {"hero", "ego", "player"}


# 这些地图具有道路场景，适合车辆/行人数据采集。
RECOMMENDED_ROAD_MAPS = {
    "Town01",
    "Town01_Opt",
    "Town02",
    "Town02_Opt",
    "Town03",
    "Town03_Opt",
    "Town04",
    "Town04_Opt",
    "Town05",
    "Town05_Opt",
    "Town06",
    "Town06_Opt",
    "Town07",
    "Town07_Opt",
    "Town10HD",
    "Town10HD_Opt",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
    "HutbCarlaCity",
    "baidutest2test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "map_name",
        nargs="?",
        help=(
            "临时指定要加载的地图；未填写时使用代码顶部的 "
            "TARGET_MAP_NAME。"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="模拟器地址。")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC 端口。")
    parser.add_argument(
        "--timeout",
        type=float,
        default=RPC_TIMEOUT_SECONDS,
        help="连接和加载地图的超时时间（秒）。",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出模拟器当前注册的全部地图。",
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="显示当前地图。",
    )
    parser.add_argument(
        "--allow-world-reset",
        action="store_true",
        help=(
            "明确允许销毁当前测试世界中的飞行器并切换地图。"
            "默认关闭，主动操作的飞行器仍受保护。"
        ),
    )
    return parser.parse_args()


def short_map_name(map_path: str) -> str:
    """把 /Game/Carla/Maps/Town10HD 转换为 Town10HD。"""
    normalized = str(map_path).replace("\\", "/").rstrip("/")
    return PurePosixPath(normalized).name


def available_map_lookup(client: "carla.Client") -> Dict[str, str]:
    """创建不区分大小写的短名称到 CARLA 注册名称映射。"""
    result: Dict[str, str] = {}
    for registered_name in client.get_available_maps():
        short_name = short_map_name(registered_name)
        result[short_name.casefold()] = registered_name
    return result


def print_available_maps(lookup: Dict[str, str]) -> None:
    names: List[str] = sorted(
        (short_map_name(value) for value in lookup.values()),
        key=str.casefold,
    )
    print(f"[INFO] 可加载地图数量：{len(names)}")
    for name in names:
        category = "道路地图" if name in RECOMMENDED_ROAD_MAPS else "测试/素材地图"
        print(f"  {name:<28} {category}")


def resolve_requested_map(
    requested_name: str,
    lookup: Dict[str, str],
) -> str:
    short_requested = short_map_name(requested_name)
    registered_name = lookup.get(short_requested.casefold())
    if registered_name is None:
        available = ", ".join(
            sorted(short_map_name(value) for value in lookup.values())
        )
        raise ValueError(
            f"地图不存在：{requested_name}\n"
            f"模拟器当前可加载地图：{available}"
        )
    return registered_name


def find_protected_uav_actors(world: "carla.World") -> List[str]:
    """查找不能在换图时销毁的主动控制飞行器或 hero 载具。"""
    protected: List[str] = []
    for actor in world.get_actors():
        type_id = str(getattr(actor, "type_id", ""))
        attributes = getattr(actor, "attributes", {}) or {}
        role_name = str(attributes.get("role_name", "")).strip()
        type_id_lower = type_id.casefold()
        role_name_lower = role_name.casefold()

        is_uav = any(keyword in type_id_lower for keyword in UAV_TYPE_KEYWORDS)
        is_controlled_vehicle = (
            type_id_lower.startswith("vehicle.")
            and role_name_lower in ACTIVE_CONTROL_ROLE_NAMES
        )
        if is_uav or is_controlled_vehicle:
            protected.append(
                f"id={actor.id}, type={type_id}, role={role_name or '-'}"
            )
    return protected


def main() -> int:
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    try:
        world = client.get_world()
        lookup = available_map_lookup(client)
    except RuntimeError as exc:
        raise RuntimeError(
            f"无法连接模拟器 {args.host}:{args.port}。请先启动 OpenHUTB/CARLA。"
        ) from exc

    current_name = short_map_name(world.get_map().name)
    if args.current:
        print(f"[INFO] 当前地图：{current_name}")

    if args.list:
        print_available_maps(lookup)

    # 查询命令未附带地图名称时只查询，不执行地图切换。
    if (args.current or args.list) and args.map_name is None:
        return 0

    requested_map_name = args.map_name or TARGET_MAP_NAME
    target_registered_name = resolve_requested_map(requested_map_name, lookup)
    target_short_name = short_map_name(target_registered_name)
    if current_name.casefold() == target_short_name.casefold():
        print(f"[INFO] 当前已经是 {target_short_name}，无需重新加载。")
        return 0

    if PROTECT_ACTIVE_UAV and not args.allow_world_reset:
        protected_uavs = find_protected_uav_actors(world)
        if protected_uavs:
            details = "\n".join(f"  - {item}" for item in protected_uavs)
            raise RuntimeError(
                "检测到正在主动控制的飞行器或 hero 载具，已取消地图切换。\n"
                "CARLA 的 load_world() 会销毁当前世界中的全部 actor，"
                "无法在换图时原样保留飞行器。\n"
                f"受保护 actor：\n{details}"
            )

    print(f"[INFO] 正在切换：{current_name} -> {target_short_name}")
    print(f"[INFO] 大地图首次加载可能需要数分钟，最长等待 {args.timeout:.0f} 秒。")
    try:
        loaded_world = client.load_world(target_registered_name)
    except RuntimeError as exc:
        raise RuntimeError(
            f"切换到 {target_short_name} 时等待超过 {args.timeout:.0f} 秒。\n"
            "如果模拟器画面仍在加载，请继续等待后再用 --current 查询；"
            "如果画面已经无响应，请关闭全部 CarlaUE4 进程，只启动一个"
            "模拟器实例后重试。"
        ) from exc
    loaded_name = short_map_name(loaded_world.get_map().name)
    if loaded_name.casefold() != target_short_name.casefold():
        raise RuntimeError(
            f"地图加载验证失败：请求={target_short_name}，实际={loaded_name}"
        )

    print(f"[DONE] 地图切换成功：{loaded_name}")
    if loaded_name not in RECOMMENDED_ROAD_MAPS:
        print("[WARN] 这是测试/素材地图，不建议用于车辆与行人数据采集。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
