# OpenHUTB/CARLA 无人机数据集采集总览

本目录包含三套正式的数据集采集项目，分别面向多模态小目标感知、单相机单目标追踪和多相机多目标追踪。代码已按照实际生成的数据集、采集状态、配置快照和运行日志完成归档。

三个项目共享相同的 OpenHUTB/CARLA 地图、天气、车辆和行人目标体系，但保存模态、序列组织方式和质量门槛不同。

## 项目总览

| 项目 | 公开数据 | 主要任务 | 八地图正式规模 | 详细文档 |
|---|---|---|---:|---|
| `multimodal/` | RGB、Surface Normal、Segmentation、Depth、LiDAR | 多模态小目标检测与感知 | 4200 组同步帧 | [多模态 README](./multimodal/README.md) |
| `single_camera_tracking/` | RGB、VOT、辅助 YOLO | 单相机单目标追踪（VOT/SOT） | 24000 张 RGB | [单相机 README](./single_camera_tracking/README.md) |
| `multi_camera_tracking/` | RGB、MOT、YOLO、标定和世界轨迹 | 多相机多目标追踪与身份关联 | 72000 张 RGB | [多相机 README](./multi_camera_tracking/README.md) |

正式结果目录分别为：

```text
E:\pythonProject\air_groud\get_img\dataset_uav_multimap_town600_hutb300_ccsp300
E:\pythonProject\air_groud\get_img\dataset_uav_multimap_single_object_vot_weather500
E:\pythonProject\air_groud\get_img\dataset_uav_multimap_multicamera_mot_weather500
```

## 目录结构

```text
collect/
├─ README.md
├─ multimodal/
│  ├─ collect_rpg_small_targets_carla_v2.py
│  ├─ collection_config.json
│  ├─ run_recommended_multimap_multimodal_collection.py
│  ├─ collect_ccsp_fullmap_pose_diversity_carla.py
│  ├─ run_ccsp_streaming_collector.py
│  ├─ qa_*.py
│  ├─ consolidate_multimap_sequences.py
│  ├─ upload_multimap_dataset_to_hf.py
│  └─ README.md
├─ single_camera_tracking/
│  ├─ collect_uav_single_object_vot_carla.py
│  ├─ single_object_vot_config.json
│  ├─ run_multimap_single_object_vot_collection.py
│  ├─ run_multimap_tracking_common.py
│  ├─ upload_uav_dataset_to_huggingface.py
│  └─ README.md
└─ multi_camera_tracking/
   ├─ collect_uav_multicamera_mot_carla.py
   ├─ multi_camera_mot_config.json
   ├─ run_multimap_multicamera_mot_collection.py
   ├─ run_multimap_tracking_common.py
   ├─ prepare_multimap_tracking_yolo_eval.py
   ├─ upload_multicamera_mot_weather500_hf_mirror.py
   └─ README.md
```

上传目录中的 `*.jsonl` 和 `*_completed_files.txt` 是断点与历史记录，不是临时文件。需要继续上传时应保留。

## 代码依赖关系

多模态主采集器包含 CARLA 传感器、相机投影、深度解码、可见性判断和标注生成等基础函数。两个追踪项目在此基础上实现各自的数据组织与审计逻辑。

```text
multimodal/collect_rpg_small_targets_carla_v2.py
        ├─ single_camera_tracking/collect_uav_single_object_vot_carla.py
        └─ multi_camera_tracking/collect_uav_multicamera_mot_carla.py
                  └─ 复用单相机项目的目标类别定义
```

两个追踪批量运行器还会复用多模态项目中的地图列表、模拟器启动方式和 CCSP 环境配置。因此三个子目录应保持当前相对位置。

## 正式入口

| 项目 | 主采集器 | 默认配置 | 八地图批量入口 |
|---|---|---|---|
| 多模态 | `multimodal/collect_rpg_small_targets_carla_v2.py` | `multimodal/collection_config.json` | `multimodal/run_recommended_multimap_multimodal_collection.py` |
| 单相机 | `single_camera_tracking/collect_uav_single_object_vot_carla.py` | `single_camera_tracking/single_object_vot_config.json` | `single_camera_tracking/run_multimap_single_object_vot_collection.py` |
| 多相机 | `multi_camera_tracking/collect_uav_multicamera_mot_carla.py` | `multi_camera_tracking/multi_camera_mot_config.json` | `multi_camera_tracking/run_multimap_multicamera_mot_collection.py` |

命令行参数优先于 JSON 配置。八地图批量运行器还会针对地图覆盖连接超时、采样规模、流送预热和质量门槛；正式运行最终使用的配置会保存到结果目录。

## 运行环境

### 必需组件

- Windows。
- OpenHUTB/CARLA 模拟器及与其版本匹配的 Python API。
- Python 依赖至少包括 `numpy` 和 `opencv-python`。
- 上传工具需要 `huggingface_hub`。
- 足够的磁盘容量和持续写入性能。

默认服务端口：

| 服务 | 默认端口 |
|---|---:|
| CARLA RPC | 2000 |
| Traffic Manager | 8000 |

本机当前正式运行环境使用：

```text
D:\anaconda2023.09\envs\openhutb\python.exe
```

批量运行器还包含 OpenHUTB、CCSP 模拟器和专用 Python API 的本机绝对路径。复制到其他电脑后，必须先检查以下配置：

- `multimodal/run_recommended_multimap_multimodal_collection.py` 中的 Python 和模拟器路径。
- 两个 `run_multimap_tracking_common.py` 中生成输出和调用采集器的路径逻辑。
- 三个 JSON 配置中的 `out`、端口和地图设置。
- 上传脚本中的 `LOCAL_FOLDER`、`REPO_ID` 和凭据环境。

## 支持的地图和天气

八地图正式任务包含：

- `Town03_Opt`
- `Town05_Opt`
- `Town10HD`
- `Town02_Opt`
- `Town04_Opt`
- `Town07_Opt`
- `HutbCarlaCity`
- `CCSP_Zhongdian_Software_Park`

统一天气集合：

- `ClearNoon`
- `ClearSunset`
- `ClearNight`
- `FoggyNoon`
- `SnowNoon`
- `DustStorm`

CCSP 与 HutbCarlaCity 的地图流送、行人生成、路面语义和模拟器版本与普通 Town 地图不同，应优先通过批量运行器启动，不建议直接套用普通地图参数。

## 快速开始

在 PowerShell 中定义公共路径：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$COLLECT = "E:\pythonProject\air_groud\get_img\collect"
```

### 多模态八地图采集

```powershell
& $PYTHON "$COLLECT\multimodal\run_recommended_multimap_multimodal_collection.py" `
  --visible
```

### 单相机追踪冒烟验证

```powershell
& $PYTHON "$COLLECT\single_camera_tracking\run_multimap_single_object_vot_collection.py" `
  --only Town03_Opt `
  --smoke `
  --visible
```

### 多相机追踪冒烟验证

```powershell
& $PYTHON "$COLLECT\multi_camera_tracking\run_multimap_multicamera_mot_collection.py" `
  --only Town03_Opt `
  --smoke `
  --visible
```

追踪任务确认冒烟结果正常后，去掉 `--smoke` 即可运行正式规模。多模态项目的单次小规模验证方法见其项目 README。

## 推荐工作流程

1. 检查 Python、CARLA API、模拟器和输出路径。
2. 确认目标磁盘剩余空间。
3. 先运行单地图、小规模任务验证连接和传感器同步。
4. 检查输出图像、配置快照和质量审计。
5. 启动八地图批量任务。
6. 根据 `collection_status.json` 和 `_logs/` 检查完成状态。
7. 运行项目专用 QA 或离线审计。
8. 确认 `dataset_manifest.json`、数据数量和划分正确后再上传。

## 正式结果规模

| 项目 | 单图规模 | 八地图合计 |
|---|---:|---:|
| 多模态 | 普通 Town 600 帧；HUTB/CCSP 各 300 帧 | 4200 组同步帧 |
| 单相机 VOT/SOT | 75 序列 × 40 帧 | 24000 张 RGB |
| 多相机 MOT | 100 场景 × 30 时刻 × 3 相机 | 72000 张 RGB |

这些数字来自现有正式结果的状态文件和数据清单。基础 JSON 中的值可能被批量运行器覆盖，复现实验时应以结果目录中的 `_configs/`、`collection_config_used.json` 或配置快照为准。

## 状态、续跑与覆盖行为

| 项目 | 续跑策略 | 覆盖风险 |
|---|---|---|
| 多模态 | 可保留已有序列并补采；批量入口默认跳过达到目标的地图 | `--overwrite-sequences` 会覆盖同名序列 |
| 单相机 | 批量入口跳过完整地图；小规模任务使用隔离 smoke 输出 | `--overwrite` 会重建指定输出 |
| 多相机 | `--resume` 保留已完成场景并补缺失场景 | `--overwrite` 会清理并重建指定输出 |

以下选项应谨慎使用：

- `--overwrite`
- `--overwrite-sequences`
- `--rerun-complete`
- 数据整理工具的 `--execute`

运行前应确认它们指向的确切目录，尤其不要把临时验证任务写入正式数据集根目录。

## 质量与可复现性文件

正式结果中应保留：

- `collection_status.json`：批量任务整体状态。
- `_configs/` 或配置快照：每张地图最终使用的配置。
- `_logs/`：采集输出和异常定位依据。
- `dataset_manifest.json`：模态、类别、规模和划分摘要。
- `quality_audit.json` 或质量报告：数据完整性与质量门槛结果。
- 数据集内 README：公开数据结构和标注格式。

这些文件也是判断数据是否由当前正式代码生成的主要证据。

## 项目间的主要区别

| 维度 | 多模态 | 单相机追踪 | 多相机追踪 |
|---|---|---|---|
| 保存模态 | RGB、法线、分割、深度、LiDAR | RGB | RGB |
| 时间连续性 | 以独立场景为主 | 连续单目标序列 | 三路同步连续场景 |
| 身份范围 | 帧级检测标注 | 单序列主目标身份 | 同一场景跨相机、跨帧身份 |
| 主要标注 | YOLO、COCO、JSON | VOT、YOLO | MOT、YOLO、JSON、全局轨迹 |
| 数据划分单位 | 帧/合并序列 | 完整序列 | 完整场景 |
| 内部质量传感器 | 深度、语义、实例、LiDAR | 深度、语义 | 深度、语义 |

## 安全注意事项

- 不要在不确认目标路径的情况下运行覆盖选项。
- 不要删除上传断点文件，除非确定需要从头重新上传。
- 不要按单帧随机拆分追踪数据，否则会产生相邻帧或身份泄漏。
- 多相机数据必须保持同一仿真 tick 的同步关系。
- 采集器异常退出后，应先检查日志和现有数据，再决定续跑或重建。
- 正式数据目录不应作为测试输出目录。

## 更多文档

- [多模态项目完整说明](./multimodal/README.md)
- [单相机 VOT/SOT 项目完整说明](./single_camera_tracking/README.md)
- [多相机 MOT 项目完整说明](./multi_camera_tracking/README.md)

旧的单相机多目标 MOT、独立语义分割采集、独立 GBuffer 法线采集和环境探针没有参与这三套正式结果，已从正式采集目录中移除。当前目录只保留生产流程、质量检查、结果整理和上传所需内容。
