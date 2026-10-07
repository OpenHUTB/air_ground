# Task 1：空地协同多模态感知

Task 1 是 `Air-Ground Cooperative Multimodal Perception`。每个 scene 使用 1 个 Air 平台和 1 辆由共享 `GroundVehicleManager` 控制的专用 Ground Vehicle；Ground Vehicle 按 `coverage` 路线行驶，不围绕单一 Target 运动。

## 必选模态

每个正式 sample 必须同时具有：

- Air：RGB、metric Depth、Normal、Semantic Segmentation、LiDAR
- Ground Vehicle：RGB、metric Depth、Normal、Semantic Segmentation、LiDAR

Ground Normal 由同帧 metric Depth 计算，继承 Depth 的 `world_frame_id` 与 Sensor timestamp。Depth 以 `float32 .npy`、单位 meter 保存；彩色或 16-bit 图仅用于预览。任一必选数据缺失，整帧不提交。

## 硬同步与状态

每个正式 frame 保存 `world_frame_id`、`timestamp`、`simulation_time`、连续 `sample_index` 和 `source_tick`。十项必选数据必须具有同一个 `world_frame_id`，时间戳误差必须在配置容差内。禁止用上一帧补齐。

Air 与 Vehicle 的 platform pose、sensor pose、velocity、acceleration、heading、control 均逐帧绑定。标定使用方向明确的 4×4 row-major 矩阵与 column-vector 规则，并保存闭环误差。`global_object_uuid` 由 dataset namespace、scene ID 与 object spawn serial 通过 UUID5 确定。

统一 source of truth 位于：

```text
E:\pythonProject\air_groud\get_img\AirGroundCoopSuite
```

COCO/YOLO Detection 是 Task 1 的派生兼容格式。

## Town03_Opt smoke

当前配置只采 3 个小 batch，每个 10 个正式 frame：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$ROOT = "E:\pythonProject\air_groud\get_img\collect\multimodal"
& $PYTHON "$ROOT\collect_rpg_small_targets_carla_v2.py" `
  --config "$ROOT\collection_config.json" `
  --map Town03_Opt `
  --sequences 3 `
  --frames 10
```

Smoke 输出固定在 `AirGroundCoopSuite\_smoke\Town03_Opt`。重点检查两个平台的 RGB、Depth、Normal、Segmentation、LiDAR，以及 `frames.jsonl`、逐帧 calibration、annotations、associations 和 quality。

主要调参项：

- Ground Camera：`cooperative.ground_vehicle.camera_mount`、`camera_fov_deg`
- Ground 路线/速度：`route_profile`、`nominal_speed_kmh`、`speed_jitter_kmh`
- Ground LiDAR：`cooperative.ground_vehicle.lidar`
- Air 视角：`height_min/max`、`radius_min/max`、`pitch_min/max`

所有尚未用 pilot 统计冻结的参数均标记为 `pilot_provisional`。不要自动启动、关闭或强杀 CARLA；人工确认 smoke 前不要启动 pilot/full collection。
