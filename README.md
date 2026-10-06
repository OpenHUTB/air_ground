# OpenHUTB/CARLA 无人机数据采集项目

本仓库包含三套基于 OpenHUTB/CARLA 的无人机视角数据采集代码，分别用于多模态感知、单相机单目标追踪和多相机多目标追踪。

本文档只说明 `get_img/collect` 下三个项目的代码结构、用途和运行方式。采集结果、论文材料和实验产物不属于本文档的说明范围。

## 项目总览

| 项目 | 主要任务 | 公开数据 | 主要标注 | 正式入口 |
|---|---|---|---|---|
| [`multimodal`](./get_img/collect/multimodal/) | 多模态检测与感知 | RGB、表面法线、语义分割、深度、可选 LiDAR | YOLO、COCO、逐帧 JSON | `run_recommended_multimap_multimodal_collection.py` |
| [`single_camera_tracking`](./get_img/collect/single_camera_tracking/) | 单相机单目标追踪（VOT/SOT） | RGB | VOT、YOLO、逐帧 JSON | `run_multimap_single_object_vot_collection.py` |
| [`multi_camera_tracking`](./get_img/collect/multi_camera_tracking/) | 多相机多目标追踪与跨相机身份关联 | 三相机同步 RGB | MOTChallenge、YOLO、逐帧 JSON、全局轨迹 | `run_multimap_multicamera_mot_collection.py` |

三个项目支持以下八张地图：

- `Town02_Opt`
- `Town03_Opt`
- `Town04_Opt`
- `Town05_Opt`
- `Town07_Opt`
- `Town10HD`
- `HutbCarlaCity`
- `CCSP_Zhongdian_Software_Park`

正式采集规模、天气、相机参数和质量阈值以各项目的 JSON 配置及批量运行器生成的配置快照为准。

## 目录结构

```text
get_img/collect/
|-- multimodal/
|   |-- collect_rpg_small_targets_carla_v2.py
|   |-- collection_config.json
|   |-- run_recommended_multimap_multimodal_collection.py
|   |-- collect_ccsp_fullmap_pose_diversity_carla.py
|   |-- run_ccsp_streaming_collector.py
|   |-- prepare_static_pedestrians_carla.py
|   |-- generate_ccsp_pedestrian_navigation.py
|   |-- capture_carla_map_previews.py
|   |-- switch_carla_map.py
|   |-- consolidate_multimap_sequences.py
|   |-- qa_recommended_multimap_collection.py
|   |-- qa_ccsp_rgb_unmatched_vehicles.py
|   |-- visualize_rgb_annotations.py
|   `-- upload_multimap_dataset_to_hf.py
|-- single_camera_tracking/
|   |-- collect_uav_single_object_vot_carla.py
|   |-- single_object_vot_config.json
|   |-- run_multimap_single_object_vot_collection.py
|   |-- run_multimap_tracking_common.py
|   `-- upload_uav_dataset_to_huggingface.py
`-- multi_camera_tracking/
    |-- collect_uav_multicamera_mot_carla.py
    |-- multi_camera_mot_config.json
    |-- run_multimap_multicamera_mot_collection.py
    |-- run_multimap_tracking_common.py
    |-- prepare_multimap_tracking_yolo_eval.py
    `-- upload_multicamera_mot_weather500_hf_mirror.py
```

## 代码依赖关系

`multimodal/collect_rpg_small_targets_carla_v2.py` 提供传感器同步、深度解码、三维投影、目标可见性和边界框生成等基础能力。两个追踪项目会直接复用这些函数，多相机项目还会复用单相机项目中的目标类别定义。

```text
multimodal/collect_rpg_small_targets_carla_v2.py
        |-- single_camera_tracking/collect_uav_single_object_vot_carla.py
        `-- multi_camera_tracking/collect_uav_multicamera_mot_carla.py
                `-- single_camera_tracking 中的目标类别定义
```

两个追踪项目的批量运行器还会复用多模态项目的地图列表、模拟器启动配置和特殊地图处理逻辑。因此，请保持三个目录的相对位置不变。

## 运行环境

- Windows。
- 已安装并可运行的 OpenHUTB/CARLA 模拟器。
- 与模拟器版本匹配的 CARLA Python API。
- Python 依赖至少包括 `numpy` 和 `opencv-python`。
- 质量检查脚本 `qa_ccsp_rgb_unmatched_vehicles.py` 需要 `ultralytics`。
- 上传脚本需要 `huggingface_hub`。
- 默认 CARLA RPC 端口为 `2000`，Traffic Manager 端口为 `8000`。

批量运行器中包含本机 Python、模拟器和特殊地图资源的路径。换机器运行前，应先检查以下文件中的路径配置：

- `multimodal/run_recommended_multimap_multimodal_collection.py`
- `single_camera_tracking/run_multimap_tracking_common.py`
- `multi_camera_tracking/run_multimap_tracking_common.py`
- 三个项目的 JSON 配置文件
- 上传脚本中的本地目录、数据集仓库和认证设置

以下示例假设当前 PowerShell 位于仓库根目录：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$COLLECT = Join-Path (Get-Location) "get_img\collect"
```

## 一、多模态数据采集

### 项目功能

多模态项目在同一仿真时刻采集无人机视角的 RGB、表面法线、目标语义分割、稠密深度和可选 LiDAR，并生成车辆、行人的检测标注与质量报告。深度、语义和实例信息同时用于遮挡判断、可见框计算和坏视角过滤。

### 代码说明

| 文件 | 作用 |
|---|---|
| `collect_rpg_small_targets_carla_v2.py` | 主采集器；负责同步传感器、生成目标标注、划分数据集并写出质量报告 |
| `collection_config.json` | 单次采集的默认配置，包括地图、输出目录、分辨率、相机姿态、天气、目标数量和质量阈值 |
| `run_recommended_multimap_multimodal_collection.py` | 八地图批量入口；负责启动模拟器、切换地图、生成地图专用配置、续跑检查和日志记录 |
| `collect_ccsp_fullmap_pose_diversity_carla.py` | CCSP 地图专用包装器；控制区域覆盖、位姿多样性、天气均衡和质量过滤 |
| `run_ccsp_streaming_collector.py` | CCSP 流送稳定性与雾天保护入口，在调用主采集器前检查 RGB 是否稳定 |
| `prepare_static_pedestrians_carla.py` | 为缺少可用行人导航网格的地图生成并维持辅助行人或车辆 actor |
| `generate_ccsp_pedestrian_navigation.py` | 根据 CCSP OpenDRIVE 道路信息生成并可选安装行人导航网格 |
| `capture_carla_map_previews.py` | 批量切换地图并保存空中 RGB 预览图，用于地图检查 |
| `switch_carla_map.py` | 查看可用地图、查看当前地图或切换 CARLA 地图 |
| `consolidate_multimap_sequences.py` | 将同一地图的多个 `seq_*` 目录合并，保持模态对齐并更新划分、COCO 和清单文件 |
| `qa_recommended_multimap_collection.py` | 检查八地图结果的帧数、模态完整性、数组形状、标注和数据清单 |
| `qa_ccsp_rgb_unmatched_vehicles.py` | 使用 YOLO 检查 CCSP RGB 中疑似存在但未匹配真值框的车辆 |
| `visualize_rgb_annotations.py` | 随机抽样数据并绘制 RGB、多模态与检测标注，供人工复核 |
| `upload_multimap_dataset_to_hf.py` | 校验本地数据后上传 Hugging Face；只有传入 `--execute` 才会真正上传 |

### 运行方式

运行单次采集前先启动与 Python API 匹配的模拟器：

```powershell
$PROJECT = Join-Path $COLLECT "multimodal"

& $PYTHON "$PROJECT\collect_rpg_small_targets_carla_v2.py" `
  --config "$PROJECT\collection_config.json" `
  --out "E:\path\to\dataset_multimodal"
```

运行八地图批量采集：

```powershell
& $PYTHON "$PROJECT\run_recommended_multimap_multimodal_collection.py" --visible
```

只采集指定地图：

```powershell
& $PYTHON "$PROJECT\run_recommended_multimap_multimodal_collection.py" `
  --only Town03_Opt HutbCarlaCity `
  --visible
```

主采集器的完整参数可通过以下命令查看：

```powershell
& $PYTHON "$PROJECT\collect_rpg_small_targets_carla_v2.py" --help
```

### 主要输出

```text
<map_root>/
|-- paired_weather/<map>/
|   |-- rgb/<weather>/
|   |-- surface_normal/{png,npy}/
|   |-- segmentation/{id,color}/
|   |-- depth/{npy,vis_16bit,color,lidar}/
|   |-- annotations/
|   |-- labels_yolo/
|   |-- frame_index.csv
|   `-- sequence_meta.json
|-- splits/
|-- splits_by_weather/
|-- coco/
|-- data.yaml
|-- dataset_manifest.json
|-- collection_config_snapshot.json
`-- quality_report.json
```

## 二、单相机单目标追踪

### 项目功能

单相机项目采集连续 RGB 单目标追踪序列。每条序列指定一个车辆或行人作为主目标，并在整条序列中保持同一 CARLA actor 身份。主目标采用 VOT 矩形框格式，画面中其他合格车辆和行人同时写入 YOLO 标签。

深度和语义相机只在采集期间用于遮挡、道路、目标尺寸和穿模检查，不保存到公开结果中。训练集、验证集和测试集按完整序列划分，避免相邻帧和目标身份泄漏。

### 代码说明

| 文件 | 作用 |
|---|---|
| `collect_uav_single_object_vot_carla.py` | VOT/SOT 主采集器；负责目标选择、相机运动、序列验收、VOT/YOLO 标注、断点状态和最终审计 |
| `single_object_vot_config.json` | 单次采集基础配置，包括三种相机运动模式、目标类别配额、帧数、采样间隔和质量阈值 |
| `run_multimap_single_object_vot_collection.py` | 单相机八地图批量入口，将任务交给公共运行器 |
| `run_multimap_tracking_common.py` | 负责模拟器进程、地图切换、地图专用配置、运行种子、完成度审计、续跑与日志 |
| `upload_uav_dataset_to_huggingface.py` | 支持大批次、限流、失败重试和断点续传的 Hugging Face 上传工具 |

采集器支持固定悬停、滞后跟随和侧向环绕三种拍摄方式。实际序列数、每种目标配额和帧数由 `single_object_vot_config.json` 及批量运行时生成的地图配置决定。

### 运行方式

单地图直接采集：

```powershell
$PROJECT = Join-Path $COLLECT "single_camera_tracking"

& $PYTHON "$PROJECT\collect_uav_single_object_vot_carla.py" `
  --config "$PROJECT\single_object_vot_config.json" `
  --out "E:\path\to\dataset_single_object" `
  --overwrite
```

小规模验证：

```powershell
& $PYTHON "$PROJECT\collect_uav_single_object_vot_carla.py" `
  --config "$PROJECT\single_object_vot_config.json" `
  --out "E:\path\to\dataset_single_object_smoke" `
  --max-sequences 1 `
  --frames-per-sequence 5 `
  --overwrite
```

八地图批量运行：

```powershell
& $PYTHON "$PROJECT\run_multimap_single_object_vot_collection.py" --visible
```

只运行指定地图的隔离冒烟测试：

```powershell
& $PYTHON "$PROJECT\run_multimap_single_object_vot_collection.py" `
  --only Town03_Opt `
  --smoke `
  --visible
```

### 主要输出

```text
<map_root>/
|-- vot/<sequence_name>/
|   |-- color/
|   |-- labels_yolo/
|   |-- groundtruth.txt
|   |-- absence.label
|   |-- occlusion.label
|   |-- target_state.label
|   |-- annotations.jsonl
|   `-- sequence_meta.json
|-- yolo/
|   |-- images/{train,val,test}/
|   |-- labels/{train,val,test}/
|   `-- data.yaml
|-- resume_state.json
|-- dataset_manifest.json
`-- quality_audit.json
```

`groundtruth.txt` 每行使用 `x,y,width,height`。主目标不可见或严重遮挡时写入 `0,0,0,0`，对应状态记录在 `absence.label` 或 `occlusion.label` 中。

## 三、多相机多目标追踪

### 项目功能

多相机项目使用三台无人机相机采集同步 RGB 序列。所有相机在同一次 `world.tick()` 中取帧，并使用 CARLA actor 身份生成跨相机、跨帧一致的 `global_id`，用于目标检测、多目标追踪和跨相机身份关联。

深度和语义相机只用于采集期的遮挡、可见框和坏视角检查。数据按完整场景划分，不能按单张图片随机拆分。

### 代码说明

| 文件 | 作用 |
|---|---|
| `collect_uav_multicamera_mot_carla.py` | 三相机同步 MOT 主采集器；负责相机布局、同步采样、全局身份、场景验收、MOT/YOLO 标注和离线审计 |
| `multi_camera_mot_config.json` | 单次采集基础配置，包括相机数量、场景数量、采样频率、视角布局、运动和质量阈值 |
| `run_multimap_multicamera_mot_collection.py` | 多相机八地图批量入口，将任务交给公共运行器 |
| `run_multimap_tracking_common.py` | 负责模拟器进程、地图切换、地图专用配置、随机种子、完成度审计、续跑和日志 |
| `prepare_multimap_tracking_yolo_eval.py` | 为单相机和多相机结果建立聚合 YOLO 评估目录，优先创建硬链接，失败时复制 |
| `upload_multicamera_mot_weather500_hf_mirror.py` | 支持镜像站、限流、失败重试和断点续传的 Hugging Face 上传工具 |

### 运行方式

单地图直接采集：

```powershell
$PROJECT = Join-Path $COLLECT "multi_camera_tracking"

& $PYTHON "$PROJECT\collect_uav_multicamera_mot_carla.py" `
  --config "$PROJECT\multi_camera_mot_config.json" `
  --out "E:\path\to\dataset_multicamera_mot" `
  --overwrite
```

只采集一个场景进行验证：

```powershell
& $PYTHON "$PROJECT\collect_uav_multicamera_mot_carla.py" `
  --config "$PROJECT\multi_camera_mot_config.json" `
  --out "E:\path\to\dataset_multicamera_mot_smoke" `
  --max-scenes 1 `
  --overwrite
```

八地图批量运行：

```powershell
& $PYTHON "$PROJECT\run_multimap_multicamera_mot_collection.py" --visible
```

续采已有结果：

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

### 主要输出

```text
<map_root>/
|-- scenes/<scene_name>/
|   |-- cameras/<camera_name>/
|   |   |-- rgb/
|   |   |-- labels_yolo/
|   |   |-- annotations/
|   |   `-- gt/gt.txt
|   |-- calibration/
|   |-- global_tracks.jsonl
|   |-- scene_quality.json
|   `-- scene_meta.json
|-- yolo/
|   |-- images/{train,val,test}/
|   |-- labels/{train,val,test}/
|   `-- data.yaml
|-- collection_config_used.json
|-- dataset_manifest.json
`-- quality_audit.json
```

同一场景内的 `global_id` 在所有相机和所有帧中保持一致。每台相机的 `gt/gt.txt` 使用 MOTChallenge 格式，标定目录保存相机内参与外参。

## 续跑、审计与安全说明

- 多模态项目可保留已有序列并继续补采；`--overwrite-sequences` 会覆盖同名序列。
- 单相机项目通过 `resume_state.json` 记录已经验收的序列；普通续跑不需要 `--overwrite`。
- 多相机项目使用 `--resume` 补采已有结果，`--overwrite` 与 `--resume` 互斥。
- `--overwrite`、`--overwrite-sequences` 和 `--rerun-complete` 可能重建现有结果，运行前必须确认输出目录。
- `consolidate_multimap_sequences.py` 默认只分析；确认报告后再传入 `--execute` 修改数据。
- 不要删除 `_configs/`、`_logs/`、配置快照、质量报告和上传断点文件，这些文件用于复现、续跑和问题定位。
- 单目标追踪必须按完整序列划分；多相机追踪必须按完整场景划分。
- 多相机数据必须保持同一仿真 tick 的同步关系，不能让各相机分别推进世界。
- CCSP 和 HutbCarlaCity 的地图流送、路面语义、行人生成和模拟器版本与普通 Town 地图不同，建议通过批量入口运行。

## 建议工作流程

1. 检查 Python、CARLA API、模拟器、端口和输出路径。
2. 使用隔离目录运行单地图或冒烟测试。
3. 检查 RGB、标注、配置快照和质量审计结果。
4. 确认磁盘空间后启动批量采集。
5. 根据 `collection_status.json`、`resume_state.json` 和 `_logs/` 检查进度。
6. 运行项目对应的 QA 或离线审计。
7. 确认 `dataset_manifest.json`、数据划分和质量报告无误后再上传。
