# AirGroundCoopSuite shared layer

本目录是 Task 1/2/3 共用的正式数据层。它只写入 `E:\pythonProject\air_groud\get_img\AirGroundCoopSuite`，不会把旧 UAV-only 数据复制、拼接或改名为协同数据。

核心组件包括统一的 `GroundVehicleManager`、route template、严格帧屏障、平台与对象注册、Transform Graph、跨视角 association、通信条件、QA、事务写入、读取和上传入口。

## 固定 Schema

- `schema_version = 1.0.0`
- SI 单位
- 4×4 homogeneous matrix，row-major storage
- column-vector：`p_target = T_target_from_source @ p_source`
- CARLA world 为左手坐标系；OpenCV camera 为 x-right/y-down/z-forward
- 对象 ID：`global_object_uuid`、`scene_object_id`、`carla_actor_id`

每个正式 frame 至少保存 `world_frame_id`、`timestamp`、`simulation_time`、`sample_index`、`source_tick`。所有必选 SensorData 先通过帧号和时间戳屏障，再进入事务目录。任何必选数据超时、错帧或时间异常都会拒绝整帧；绝不复制上一帧。

## 事务目录

- `staging/`：尚未完成的事务
- `quarantine/`：失败或中止的事务
- `task1_agc_multimodal_perception/`：已提交 Task 1 单元
- `task2_agc_sot/`：已提交 Task 2 sequence
- `task3_agc_mcmot/`：已提交 Task 3 scene

每个已提交单元带 SHA-256 manifest。只应从已提交任务目录训练或上传。

## 离线 QA

采集完成后可执行（不会连接 CARLA）：

```powershell
python E:\pythonProject\air_groud\get_img\collect\cooperative_perception\qa_dataset.py `
  E:\pythonProject\air_groud\get_img\AirGroundCoopSuite\_smoke\Town03_Opt
```

QA 会检查 checksum、时间字段、连续 sample index、必选 Sensor 时间戳集合、frame barrier、每个平台逐帧 pose/velocity/acceleration/heading/control，以及 Task 3 固定的 3 个 virtual Air + 1 个 Ground Vehicle。

## 上传

上传入口默认只打印 committed unit 清单；显式添加 `--execute` 才会调用 Hugging Face：

```powershell
python E:\pythonProject\air_groud\get_img\collect\cooperative_perception\upload_air_ground_coop_suite.py `
  E:\pythonProject\air_groud\get_img\AirGroundCoopSuite\release `
  organization/dataset-name
```

通信 profile 只控制算法可用信息，不删除、不延迟写入、也不修改原始 Sensor Data 或 Ground Truth。所有尚未通过 pilot 统计冻结的阈值保持 `pilot_provisional`。
