# 无人机多相机多目标追踪数据集

本项目在 OpenHUTB/CARLA 中采集三台同步无人机相机的 RGB 多目标追踪数据。相机在同一个 `world.tick()` 获取传感器帧，并以 CARLA actor 身份构造跨相机、跨帧一致的 `global_id`。

正式结果位于：

```text
E:\pythonProject\air_groud\get_img\dataset_uav_multimap_multicamera_mot_motion_trial1
```

## 数据特点

- 公开模态：RGB。
- 任务：目标检测、多目标追踪、跨相机身份关联。
- 类别：`vehicle`、`pedestrian`。
- 每个场景 3 台同步相机。
- 分辨率：1920×1080。
- 仿真帧率：25 FPS；车辆场景每 3 个同步 tick 保存一次，行人场景按 5 Hz 保存。
- 当前正式配置每张地图采 30 个场景：15 段车辆、15 段行人；每段最长 50 个三相机同步时刻，实际长度可以更短。
- 至少 2 台相机连续超过 5 帧没有目标时，当前片段立即停止。
- 预热只用于启动交通流，不保存预热目标名单；车辆按当前速度选择，行人则在每次正式尝试前用新的连续位置变化确认正在行走。
- 三台相机使用随机方位角与俯视角布局，不设置物理间距下限，也不使用固定夹角；仍通过最小视角差避免视角重合。
- 每段序列自动检查位移、路径长度、有效轨迹、公共视野持续时间和相邻帧 SSIM；行人不设最低速度门槛，只要确实在行进。
- 每张地图内，已经作为合格片段主目标的车辆或行人实例不会再次被选中；身份由采集种子与 actor ID 共同确定，不使用有限的外观模型做去重。目标删除后重新生成属于新实例，失败尝试不会占用目标名额。
- 湖工商和中电园额外检测地图边界的大面积深色/低纹理未建模区域，命中后整套相机布局作废并重新选择，不保存坏画面。
- 提供 YOLO、MOTChallenge、详细 JSON、相机标定和世界轨迹。
- 深度与语义相机只用于采集期遮挡和质量判断，不写入公开结果。

## 主要文件

| 文件 | 用途 |
|---|---|
| `collect_uav_multicamera_mot_carla.py` | 三相机同步 MOT 主采集器 |
| `multi_camera_mot_config.json` | 单次采集基础配置 |
| `run_multimap_multicamera_mot_collection.py` | 八地图正式批量入口 |
| `run_multimap_tracking_common.py` | 模拟器启动、地图切换、配置生成和完成度审计 |
| `prepare_multimap_tracking_yolo_eval.py` | 为单相机和多相机正式结果建立 YOLO 评估视图 |
| `upload_multicamera_mot_weather500_hf_mirror.py` | Hugging Face 镜像上传工具 |
| `*.jsonl` | 上传批次、提交和运行历史 |
| `*_completed_files.txt` | 大文件上传断点记录 |

主采集器会复用 `../multimodal/` 的 CARLA 基础函数和 `../single_camera_tracking/` 的目标类别定义，因此三个项目目录的相对位置不应随意改变。

## 环境要求

- Windows 和可运行的 OpenHUTB/CARLA 模拟器。
- 与模拟器版本匹配的 CARLA Python API。
- Python 依赖至少包括 `numpy` 和 `opencv-python`。
- 默认 RPC 端口 `2000`，Traffic Manager 端口 `8000`。
- 上传功能需要 `huggingface_hub` 和有效凭据。
- 三路 Full HD 连续数据写盘量较大，应使用空间充足且写入稳定的磁盘。

八地图运行器复用多模态项目中的模拟器路径和地图定义。换机器时需要检查 `run_multimap_tracking_common.py` 与 `../multimodal/run_recommended_multimap_multimodal_collection.py` 中的本机路径。

## 快速开始

### 单地图直接采集

先启动模拟器，再执行：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$PROJECT = "E:\pythonProject\air_groud\get_img\collect\multi_camera_tracking"

& $PYTHON "$PROJECT\collect_uav_multicamera_mot_carla.py" `
  --config "$PROJECT\multi_camera_mot_config.json" `
  --out "E:\pythonProject\air_groud\get_img\dataset_uav_multicamera_mot_motion_trial1_direct" `
  --overwrite
```

小规模验证：

```powershell
& $PYTHON "$PROJECT\collect_uav_multicamera_mot_carla.py" `
  --config "$PROJECT\multi_camera_mot_config.json" `
  --out "E:\pythonProject\air_groud\get_img\dataset_uav_multicamera_mot_smoke" `
  --max-scenes 1 `
  --overwrite
```

### 八地图批量运行

```powershell
& $PYTHON "$PROJECT\run_multimap_multicamera_mot_collection.py" --visible
```

只运行指定地图：

```powershell
& $PYTHON "$PROJECT\run_multimap_multicamera_mot_collection.py" `
  --only Town03_Opt CCSP_Zhongdian_Software_Park `
  --visible
```

当前正式配置为 `scenes_per_map=30`。如果只想先采一张地图，可以运行：

```powershell
& $PYTHON "$PROJECT\run_multimap_multicamera_mot_collection.py" `
  --only Town03_Opt `
  --visible
```

不写 `--only` 会依次运行全部 8 张地图，共采集 240 段序列；每张地图包含 15 段车辆序列和 15 段行人序列。

运行隔离的小规模冒烟采集：

```powershell
& $PYTHON "$PROJECT\run_multimap_multicamera_mot_collection.py" `
  --only Town03_Opt `
  --smoke `
  --visible
```

| 批量入口参数 | 含义 |
|---|---|
| `--only MAP...` | 只运行指定地图 |
| `--visible` | 显示模拟器窗口 |
| `--rerun-complete` | 完成的地图也使用 `--overwrite` 真正重新采集，不复用旧场景 |
| `--smoke` | 使用隔离输出，只采一个极小场景 |

## 核心配置

| 配置项 | 当前基础值 | 说明 |
|---|---:|---|
| `num_cameras` | 3 | 每个场景的同步相机数 |
| `scenes_per_map` | 30 | 每张地图采集 15 段车辆和 15 段行人 |
| `train_frames_per_scene` | 50 | train 场景最大同步时刻数 |
| `eval_frames_per_scene` | 50 | val/test 场景最大同步时刻数 |
| `sample_interval_ticks` | 3 | 车辆场景每隔多少仿真 tick 保存一次 |
| `pedestrian_sample_hz` | 5 | 行人场景保存频率 |
| `walker_activation_ticks` | 30 | 每次正式尝试用于确认行人实际位移的观察 tick 数 |
| `pedestrian_activation_displacement_m` | 0.05 m | 行人进入候选所需的最小水平位移，不使用瞬时速度判断 |
| `empty_camera_patience_frames` | 5 | 单台相机允许连续无目标的帧数 |
| `empty_camera_stop_count` | 2 | 达到无目标上限后触发停止的相机数 |
| `seed` | `null` | 默认每次运行自动生成新种子；实际种子会保存，填写整数可复现指定采集 |
| `camera_height_min_m` / `max_m` | 28 / 34 m | 通用相机高度范围 |
| `camera_radius_min_m` / `max_m` | 24 / 46 m | 相机与锚点距离范围 |
| `camera_bearing_separation_deg` | 35° | 任意相机方位角硬性最小间隔 |
| `camera_preferred_bearing_separation_min_deg` / `max_deg` | 42° / 65° | 每次随机抽取的偏好方位角范围，不固定夹角 |
| `camera_min_pair_distance_m` | 0 m | 所有地图均不限制相机物理间距；视角差要求仍然生效 |
| `min_visible_ratio` | 0.50 | 目标最低可见比例 |
| `min_common_ids_per_frame` | 1 | 每时刻至少一个跨相机共同身份 |
| `min_valid_frame_ratio` | 0.35 | 场景最低有效时刻比例 |
| `min_anchor_common_view_seconds` | 1.5 s | 锚点连续跨相机公共视野下限 |
| `min_effective_track_seconds` | 1.5 s | 有效轨迹最短持续时间，按车辆/行人采样率换算帧数 |
| `vehicle_min_dynamic_event_count` | 1 | 车辆段至少包含一次交汇、超车、转弯、遮挡重现等动态事件 |
| `pedestrian_min_dynamic_event_count` | 0 | 行人段不要求动态事件，只要求真实持续行进并通过轨迹质量检查 |
| `max_active_vehicles` | 30 | 单张地图同时参与采集的车辆上限，避免画面过密 |
| `max_active_pedestrians` | 24 | 单张地图同时参与采集的行人上限，避免画面过密并为15个唯一主行人保留余量 |
| `min_vehicle_equivalent_side_px` | 32 px | 车辆等效边长下限 |
| `min_pedestrian_equivalent_side_px` | 36 px | 行人等效边长下限 |
| `reject_unmodeled_regions` | 特殊地图开启 | 检测地图边界黑块和大面积低纹理未建模区域 |
| `max_dark_flat_region_ratio` | 0.05 | 最大连续深色低纹理区域比例，超过即重选相机 |
| `max_flat_region_ratio` | 湖工商 0.24 / 中电园 0.14 | 最大连续低纹理区域比例，超过即重选相机 |
| `max_far_depth_region_ratio` | 0.10 | 深度相机中无几何或超过 500 米区域的最大比例 |

八地图正式运行器会把连接超时、传感器超时、最大尝试次数和特殊地图参数提高到生产值，因此 `_configs/<map>.json` 才是每次正式运行的最终配置。

查看完整参数：

```powershell
& $PYTHON "$PROJECT\collect_uav_multicamera_mot_carla.py" --help
```

## 输出结构

```text
<map_root>/
├─ scenes/<scene_name>/
│  ├─ cameras/<camera_name>/
│  │  ├─ rgb/
│  │  ├─ labels_yolo/
│  │  ├─ annotations/
│  │  └─ gt/gt.txt
│  ├─ calibration/
│  ├─ global_tracks.jsonl
│  ├─ scene_quality.json
│  └─ scene_meta.json
├─ yolo/
│  ├─ images/{train,val,test}/
│  ├─ labels/{train,val,test}/
│  └─ data.yaml
├─ collection_config_used.json
├─ used_anchor_instances.json
├─ rejected_scene_attempts.jsonl
├─ failed_scenes.jsonl
├─ dataset_manifest.json
├─ quality_audit.json
└─ README.md
```

`global_id` 在同一场景的所有相机和所有帧中保持一致。训练、验证和测试按完整场景划分，身份集合之间不应重叠。

`used_anchor_instances.json` 按车辆和行人分别记录本次运行中已经成功作为主目标的生成实例。实例键由采集种子和 CARLA actor ID 组成，不依赖有限的外观模型；同一存活实例不会再次成为主目标，删除后重新生成的 actor 视为新实例。

## 续跑与离线审计

在已有场景目录上继续补采：

```powershell
& $PYTHON "$PROJECT\collect_uav_multicamera_mot_carla.py" `
  --config "$PROJECT\multi_camera_mot_config.json" `
  --out "E:\path\to\existing_dataset" `
  --resume
```

只审计现有数据并重建 YOLO 目录：

```powershell
& $PYTHON "$PROJECT\collect_uav_multicamera_mot_carla.py" `
  --config "$PROJECT\multi_camera_mot_config.json" `
  --out "E:\path\to\existing_dataset" `
  --audit-only
```

`--overwrite` 与 `--resume` 互斥。批量运行器发现已有场景时会优先续跑；若原生资源清理阶段异常退出，还会尝试离线审计并重建清单。

## 质量标准

采集器允许短暂遮挡、离开公共视野和重新出现，不再因为单帧不合格立即丢弃整段。序列级验收包括：

- 车辆锚点运动采用组合门槛：净位移与累计路径同时达标，或速度中位数与运动帧比例同时达标。
- 行人锚点不读取瞬时速度作为门槛：在正式尝试开始后的 30 个 tick 内至少产生 `0.05 m` 实际位移才进入候选，序列验收继续检查实际位移和累计路径。
- 普通 Town 地图沿用单相机与多模态项目已经验证的 AI Walker Controller 流程，并始终保留原始 walker/controller 引用；不再把同一行人在 AI 控制与直接 `WalkerControl` 之间切换。
- 有效轨迹数量及轨迹长度。
- 锚点在任意相机中的可见比例，以及连续跨相机公共视野时长。
- 相邻帧 SSIM 和近重复帧比例。
- 合格同步时刻比例、边界标注和视角质量。

不合格候选会写入 `rejected_scene_attempts.jsonl` 后自动更换锚点、相机布局或 actor population 重采。最终 `quality_audit.json` 会再次检查同步关系、重复图像、运动质量、身份划分泄漏、标注范围和数据清单完整性。

## YOLO 评估视图

`prepare_multimap_tracking_yolo_eval.py` 会为两套追踪数据建立聚合评估目录，优先使用硬链接，失败时才复制文件。该脚本会重建名称以 `_yolo_eval` 结尾的目标目录，运行前应检查文件顶部的源目录和输出目录。

## 上传说明

上传工具依赖同目录的批次历史、提交历史和完成文件列表进行镜像式断点续传。正式运行前应核对脚本顶部的 `LOCAL_FOLDER`、`REPO_ID` 和数据集标识。不要随意删除 `*_completed_files.txt`，否则会失去现有上传进度。

## 注意事项

- `--overwrite` 会清理并重建目标数据集，只应用于确认可以覆盖的目录。
- 三路相机必须在同一个仿真 tick 内同步，不能分别推进世界。
- 不要按单张图片拆分数据集；必须按完整场景划分。
- CCSP 和 HutbCarlaCity 的路面语义、地图流送和静态行人策略与普通 Town 地图不同，应通过批量运行器生成专用配置。
- 中电园和湖工商不使用 `ClearNight` 深夜天气，其他日落、雾、雪和沙尘天气保留。这两张地图使用 24 名道路外侧单向前行的行人，并关闭额外导航行人，避免行人群集中在车道。
- 截断在画面边缘的标注不再被整个删除，仍须通过可见比例和遮挡检查。所有地图都不设置相机物理间距下限；Town03、Town04、Town07、Town10HD 和 Town02 使用 30° 最小视角差，其余地图使用 35°。Town02 另外将道路相机候选点加密到 3 米。
