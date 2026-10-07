# Task 2：空地协同单目标跟踪

Task 2 使用 1 个真实 UAV 视角、1 辆独立 Ground Vehicle 和 1 个独立 Target。正式模型输入是 RGB；Depth 与 Semantic 只作为 bbox、遮挡、可见性和 QA 辅助数据保存。

Ground Vehicle 由共享 `GroundVehicleManager` 生成，角色为 `cooperative_vehicle`，使用 `BehaviorAgent` 按 `target_interaction` 预定义路线行驶。它不是 Target，也不会主动追随 Target。Target 是另一辆车辆或行人。

## 唯一 UAV 运动策略

所有正式序列固定使用：

```text
motion_mode = rear_upper_follow
```

期望位置按 Target 当前前向向量计算：

```text
desired_uav_position = target_position
                     - back_distance_m * target_forward
                     + height_above_target_m * world_up
```

Camera 始终朝向 Target。位置采用响应时间与最大速度限幅，yaw/pitch 采用响应时间与最大角速度限幅。Target 转弯时目标后上方位置随 heading 连续变化。`back_distance_m`、高度、响应时间、最大速度、角速度和 pitch 范围均为 `pilot_provisional`，人工 smoke 后再冻结。

## 数据与同步

统一 source of truth 写入：

```text
E:\pythonProject\air_groud\get_img\AirGroundCoopSuite
```

VOT/SOT 只是派生兼容格式。每个正式 frame 以 `world_frame_id` 建立硬屏障，同时记录 `timestamp`、`simulation_time`、连续 `sample_index` 和 `source_tick`。Air RGB/Depth/Semantic 与 Vehicle RGB/Depth/Semantic 必须同帧且时间戳通过检查；任一必选 Sensor 超时、错帧或时间不一致会拒绝整帧，禁止复制上一帧补齐。

每帧分别保存 Air 与 Vehicle 的 platform pose、sensor pose、velocity、acceleration、heading 和 control。目标身份使用 deterministic UUID5 的 `global_object_uuid`，同时保存 scene 内连续的 `scene_object_id` 与仅供运行时回溯的 `carla_actor_id`。

## 协同状态

逐帧记录 `Joint Visible`、`UAV Dominant`、`Vehicle Dominant`、`Occlusion`、视角切换和 `Reacquisition`。这些新指标的阈值在 pilot 前不作为最终标准。

## Town03_Opt smoke

当前 smoke 配置为 3 条短 sequence、每条 15 个正式 frame，仅写入：

```text
E:\pythonProject\air_groud\get_img\AirGroundCoopSuite\_smoke\Town03_Opt
```

用户手动启动模拟器后运行：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$ROOT = "E:\pythonProject\air_groud\get_img\collect\single_camera_tracking"
& $PYTHON "$ROOT\collect_uav_single_object_vot_carla.py" `
  --config "$ROOT\single_object_vot_config.json" `
  --max-sequences 3 `
  --frames-per-sequence 15
```

主要调参位置：

- 后向距离/高度：`rear_upper_follow.back_distance_m`、`height_above_target_m`
- 平滑与限速：`position_response_time_s`、`rotation_response_time_s`、`max_position_speed_mps`、`max_yaw_rate_deg_s`、`max_pitch_rate_deg_s`
- 车载相机：`cooperative.ground_vehicle.camera_mount`、`camera_fov_deg`
- Ground Vehicle 路线与速度：`route_profile`、`nominal_speed_kmh`、`speed_jitter_kmh`

不要自动启动或关闭 CARLA，不要在人工确认 smoke 结果前开始 pilot/full collection。
