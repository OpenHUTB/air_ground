# Task 3：空地协同多相机多目标跟踪

Task 3 从 smoke、pilot 到 full release 永久固定为：

```text
3 virtual aerial camera platforms + 1 Ground Vehicle
```

三个 Air 节点是可移动的独立虚拟空中传感器平台，不绑定真实 UAV actor。Metadata 必须为 `platform_type="air_camera_platform"`、`platform_is_virtual=true`。数据结构仍保持 `Platform -> Sensor`。不设计第二辆 Ground Vehicle。

Ground Vehicle 由共享 `GroundVehicleManager` 生成，按 `multi_object_traversal` 预定义路线与 BehaviorAgent 正常穿行多目标场景。

## 模态与格式

RGB 是正式 MCMOT 模型输入；metric Depth 与 Semantic 保存为 visibility、occlusion、bbox 与投影 QA 辅助数据，不把 Task 3 定义成多模态跟踪。

统一 scene-oriented Schema 是唯一 source of truth：

```text
scene.json
frames.jsonl
platforms/
calibration/
annotations/
associations/
communication/
quality/
manifest.json
```

MOTChallenge/MCMOT 是派生格式。正式对象身份采用 `global_object_uuid` 与 scene 内连续 `scene_object_id`；`carla_actor_id` 只用于当前 runtime 回溯。

## 同步

三个 Air Camera 的 RGB/Depth/Semantic 与 Ground Vehicle 的 RGB/Depth/Semantic 必须通过同一 `world_frame_id` 屏障。每个正式 frame 保存 `timestamp`、`simulation_time`、连续 `sample_index` 和 `source_tick`，并逐帧保存四个平台的 pose、sensor pose、velocity、acceleration、heading、control。缺失、错帧或时间戳不一致时整帧拒绝，禁止上一帧补齐。

## Town03_Opt smoke

当前配置为 3 个短 scene、每个 15 个正式 frame：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$ROOT = "E:\pythonProject\air_groud\get_img\collect\multi_camera_tracking"
& $PYTHON "$ROOT\collect_uav_multicamera_mot_carla.py" `
  --config "$ROOT\multi_camera_mot_config.json" `
  --max-scenes 3 `
  --frames-per-scene 15
```

输出固定在 `E:\pythonProject\air_groud\get_img\AirGroundCoopSuite\_smoke\Town03_Opt`。检查 `air_camera_01`、`air_camera_02`、`air_camera_03` 和 `vehicle_01` 的 RGB，核对 `frames.jsonl`、`platforms.json`、逐帧 calibration、`annotations/by_sensor`、`associations/cross_view.jsonl` 与 communication metadata。

主要调参项：Air Camera 高度、半径、pitch 和方位分离参数；Ground Camera 的 `camera_mount`、`camera_fov_deg`；Ground Vehicle 的 `nominal_speed_kmh`、`speed_jitter_kmh` 与 route 配置。所有新阈值先保持 `pilot_provisional`。

不要自动启动或关闭 CARLA，不要在人工确认 smoke 结果前开始 pilot/full collection。
