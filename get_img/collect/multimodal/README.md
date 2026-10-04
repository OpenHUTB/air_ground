# 无人机小目标多模态数据集采集

本项目在 OpenHUTB/CARLA 中采集无人机视角的小目标多模态数据。每个保存帧包含严格同步的 RGB、表面法线、目标语义分割、稠密深度及可选 LiDAR，并提供车辆和行人的 YOLO、COCO 与逐帧 JSON 标注。

正式结果位于：

```text
E:\pythonProject\air_groud\get_img\dataset_uav_multimap_town600_hutb300_ccsp300
```

## 数据内容

- RGB：每个独立场景只分配一种天气。
- Surface Normal：相机坐标系法线，提供 PNG 和 float32 NPY。
- Segmentation：`0=background`、`1=vehicle`、`2=pedestrian`、`255=ignore`。
- Depth：与 RGB 对齐的米制稠密深度及可视化图。
- LiDAR：原始点云和投影到 RGB 平面的稀疏深度。
- Detection：YOLO、COCO、逐帧 JSON 和 CSV 标注。
- Quality metadata：配置快照、数据清单、质量报告和数据划分。

目标类别为 `vehicle` 和 `pedestrian`。深度、语义、实例信息会共同用于遮挡判断、可见框计算和坏视角过滤。

## 正式采集规模

八地图运行器使用以下目标数量：

| 地图类型 | 每图合格帧数 |
|---|---:|
| Town03、Town05、Town10HD、Town02、Town04、Town07 | 600 |
| HutbCarlaCity | 300 |
| CCSP 中电软件园 | 300 |

天气包含 `ClearNoon`、`ClearSunset`、`ClearNight`、`FoggyNoon`、`SnowNoon` 和 `DustStorm`。

## 主要文件

| 文件 | 用途 |
|---|---|
| `collect_rpg_small_targets_carla_v2.py` | 四模态主采集器 |
| `collection_config.json` | 单次采集默认配置和参数说明 |
| `run_recommended_multimap_multimodal_collection.py` | 八地图正式批量运行入口 |
| `collect_ccsp_fullmap_pose_diversity_carla.py` | CCSP 多区域、天气均衡和质量过滤包装器 |
| `run_ccsp_streaming_collector.py` | CCSP RGB 流送稳定性和雾效保护入口 |
| `prepare_static_pedestrians_carla.py` | 特殊地图静态行人辅助进程 |
| `consolidate_multimap_sequences.py` | 将分批序列合并成地图级正式序列 |
| `qa_recommended_multimap_collection.py` | 审计八地图正式结果 |
| `qa_ccsp_rgb_unmatched_vehicles.py` | 检查 CCSP 未匹配车辆外观 |
| `visualize_rgb_annotations.py` | 抽样绘制 RGB 标注框供人工复核 |
| `upload_multimap_dataset_to_hf.py` | 上传整理后的正式数据集 |

## 环境要求

- Windows 和可运行的 OpenHUTB/CARLA 模拟器。
- 与模拟器版本匹配的 CARLA Python API。
- Python 依赖至少包括 `numpy`、`opencv-python`；上传工具还需要 `huggingface_hub`。
- 默认 RPC 端口为 `2000`，Traffic Manager 端口为 `8000`。
- Full HD 多模态数据占用空间很大，正式运行前应检查目标磁盘余量。

批量运行器中保存了本机 Python、模拟器和 CCSP 包路径。换机器运行时，先检查 `OPENHUTB_ASSET_ROOT`、`OPENHUTB_PYTHON`、`CCSP_PYTHON` 等路径。

## 快速开始

### 单次采集

先启动与 Python API 匹配的模拟器，然后执行：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$PROJECT = "E:\pythonProject\air_groud\get_img\collect\multimodal"

& $PYTHON "$PROJECT\collect_rpg_small_targets_carla_v2.py" `
  --config "$PROJECT\collection_config.json" `
  --out "E:\pythonProject\air_groud\get_img\dataset_multimodal_new"
```

建议先复制并修改 `collection_config.json`，不要直接把试运行写入正式结果目录。

### 八地图正式运行

```powershell
& $PYTHON "$PROJECT\run_recommended_multimap_multimodal_collection.py" --visible
```

只运行指定地图：

```powershell
& $PYTHON "$PROJECT\run_recommended_multimap_multimodal_collection.py" `
  --only Town03_Opt HutbCarlaCity `
  --visible
```

| 运行器参数 | 含义 |
|---|---|
| `--town-frames N` | 每张普通 Town 地图的目标帧数 |
| `--hutb-frames N` | HutbCarlaCity 的目标帧数 |
| `--ccsp-frames N` | CCSP 的目标帧数 |
| `--only MAP...` | 只运行指定地图 |
| `--visible` | 显示模拟器窗口 |
| `--rerun-complete` | 已达到目标数量的地图也重新执行 |

## 核心配置

| 配置项 | 当前基础值 | 说明 |
|---|---:|---|
| `width` / `height_img` | 1920 / 1080 | 输出分辨率 |
| `fov` | 55° | 水平视场角 |
| `fps` | 20 | 仿真固定帧率 |
| `route` | `random` | 每帧随机道路视角；`orbit` 用于连续轨迹 |
| `height_min` / `height_max` | 26 / 32 m | 随机相机高度 |
| `radius_min` / `radius_max` | 24 / 46 m | 相机与瞄准区域的水平距离 |
| `pitch_min` / `pitch_max` | -45° / -35° | 相机俯角范围 |
| `vehicles` / `walkers` | 30 / 40 | 尝试生成的动态目标数量 |
| `min_actor_visible_px` | 24 | 目标最低可见像素数 |
| `min_actor_visible_ratio` | 0.50 | 目标最低可见比例 |
| `sensor_timeout` | 10 s | 同步传感器等待超时 |

命令行参数优先于 JSON。完整参数可运行：

```powershell
& $PYTHON "$PROJECT\collect_rpg_small_targets_carla_v2.py" --help
```

## 输出结构

```text
<map_root>/
├─ paired_weather/<map>/
│  ├─ rgb/<weather>/
│  ├─ surface_normal/{png,npy}/
│  ├─ segmentation/{id,color}/
│  ├─ depth/{npy,vis_16bit,color,lidar}/
│  ├─ annotations/
│  ├─ labels_yolo/
│  ├─ frame_index.csv
│  └─ sequence_meta.json
├─ splits/
├─ splits_by_weather/
├─ coco/
├─ data.yaml
├─ dataset_manifest.json
├─ collection_config_snapshot.json
├─ quality_report.json
└─ QUALITY_REPORT.md
```

各模态使用相同帧 ID。JSON 和 CSV 内的文件路径以数据集根目录为基准。

## 质量检查与整理

审计八地图正式采集结果：

```powershell
& $PYTHON "$PROJECT\qa_recommended_multimap_collection.py" `
  --root "E:\pythonProject\air_groud\get_img\dataset_uav_multimap_town600_hutb300_ccsp300"
```

正式合并工具默认先分析；只有传入 `--execute` 才会实际改动数据目录：

```powershell
& $PYTHON "$PROJECT\consolidate_multimap_sequences.py" `
  --root "E:\path\to\multimap_dataset"
```

确认报告后再追加 `--execute`。

## 注意事项

- `--overwrite-sequences` 会覆盖同名序列，运行前确认输出目录。
- `--preserve-existing-sequences` 用于保留已有序列并继续补采。
- CCSP 必须使用匹配的专用模拟器和 Python API；其地图流送策略与普通 Town 地图不同。
- `FoggyNoon` 和 `SnowNoon` 包含额外 RGB 合成处理，不等同于只调用原生天气预设。
- 批量运行日志、配置快照和质量报告是复现实验的重要组成部分，不应在发布前删除。
