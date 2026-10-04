# 无人机单相机单目标追踪数据集

本项目在 OpenHUTB/CARLA 中采集无人机视角的 RGB 单目标追踪序列，主任务格式参考 VOT/SOT。每条序列只指定一个主目标，主目标在完整序列中保持同一 CARLA actor 身份；画面内其他合格车辆和行人同时提供 YOLO 检测标签。

正式数据集输出到（不会覆盖之前的试采结果）：

```text
E:\pythonProject\air_groud\get_img\dataset_uav_multimap_single_object_vot_motion_full
```

## 数据特点

- 公开模态：RGB。
- 主任务：单相机单目标追踪（VOT/SOT）。
- 辅助任务：车辆和行人 YOLO 检测。
- 类别：`vehicle`、`pedestrian`。
- 分辨率：1920×1080。
- 仿真帧率：20 FPS；车辆每 2 tick 保存一次（10 Hz），人物每 6 tick 保存一次（约 3.33 Hz）。
- 每张地图分别采集固定悬停、滞后跟随和侧向环绕各 5 条序列，共 15 条。
- 固定悬停最长 80 帧，连续 5 帧未出现目标即结束；不足 30 帧则重采。
- 人物滞后跟随保存 80 帧（0.3 秒间隔，约 24 秒）。
- 侧向环绕：车辆 150 帧（0.1 秒间隔），人物 100 帧（0.3 秒间隔）。
- 根据侧向环绕目标类别，每张地图计划最多约 1300–1550 张 RGB；固定悬停
  允许提前结束，因此八张地图实际预计约 9400–11400 张。
- 训练、验证、测试按完整序列划分，避免相邻帧泄漏。
- 深度与语义传感器只用于遮挡、道路、目标尺寸和穿模检查，不作为公开模态保存。
- 中电园和湖工商使用 40 个带行走控制的补充行人，不再补充静止车辆；
  普通交通参与者调整为 8 辆车、0 个额外导航行人，采集器还会跳过旧版
  静止桥接 actor。
- 两张特殊地图启用完整贴图加载、30 帧预热和场景几何完整度检查；中电园还会
  避开已知的贴图/虚空区域。
- 三种运动模式分别从地图允许的天气中独立随机选择，不设置固定天气种子，
  每次重新启动批量采集都会重新抽取。其余七张地图只会抽取 `ClearNoon`、
  `ClearSunset`、`ClearNight`；中电园只会抽取 `ClearNoon` 和
  `ClearSunset`。
- 中电园单相机高度单独调整为 18–24 m、水平距离调整为 18–32 m，以提高
  行人首帧尺寸和可见性；其他地图仍使用 30–36 m 高度。

## 主要文件

| 文件 | 用途 |
|---|---|
| `collect_uav_single_object_vot_carla.py` | VOT/SOT 主采集器 |
| `single_object_vot_config.json` | 单次采集基础配置 |
| `run_multimap_single_object_vot_collection.py` | 八地图正式批量入口 |
| `run_multimap_tracking_common.py` | 模拟器启动、地图切换、配置生成和完成度审计 |
| `upload_uav_dataset_to_huggingface.py` | Hugging Face 上传工具 |
| `hf_*.jsonl` | 上传批次、提交和运行历史 |
| `hf_upload_completed_files.txt` | 上传断点记录 |

采集器复用相邻 `../multimodal/collect_rpg_small_targets_carla_v2.py` 中的 CARLA 传感器、投影和可见性基础函数，因此三个项目目录的相对位置不应随意改变。

## 环境要求

- Windows 和可运行的 OpenHUTB/CARLA 模拟器。
- 与模拟器版本匹配的 CARLA Python API。
- Python 依赖至少包括 `numpy` 和 `opencv-python`。
- 默认 RPC 端口 `2000`，Traffic Manager 端口 `8000`。
- 上传功能需要 `huggingface_hub` 和有效凭据。

八地图运行器还会复用多模态项目中的模拟器路径和地图定义。换机器时需要检查 `run_multimap_tracking_common.py` 与 `../multimodal/run_recommended_multimap_multimodal_collection.py` 中的本机路径。

## 快速开始

### 单地图直接采集

先启动模拟器，再执行：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
$PROJECT = "E:\pythonProject\air_groud\get_img\collect\single_camera_tracking"

& $PYTHON "$PROJECT\collect_uav_single_object_vot_carla.py" `
  --config "$PROJECT\single_object_vot_config.json" `
  --out "E:\pythonProject\air_groud\get_img\dataset_single_object_new" `
  --overwrite
```

`--overwrite` 会重建目标输出，请勿指向需要保留的数据集。

小规模验证：

```powershell
& $PYTHON "$PROJECT\collect_uav_single_object_vot_carla.py" `
  --config "$PROJECT\single_object_vot_config.json" `
  --out "E:\pythonProject\air_groud\get_img\dataset_single_object_smoke" `
  --max-sequences 1 `
  --frames-per-sequence 5 `
  --overwrite
```

### 八地图批量运行

```powershell
& $PYTHON "$PROJECT\run_multimap_single_object_vot_collection.py" --visible
```

只运行指定地图：

```powershell
& $PYTHON "$PROJECT\run_multimap_single_object_vot_collection.py" `
  --only Town03_Opt CCSP_Zhongdian_Software_Park `
  --visible
```

运行隔离的小规模冒烟采集：

```powershell
& $PYTHON "$PROJECT\run_multimap_single_object_vot_collection.py" `
  --only Town03_Opt `
  --smoke `
  --visible
```

| 批量入口参数 | 含义 |
|---|---|
| `--only MAP...` | 只运行指定地图 |
| `--visible` | 显示模拟器窗口 |
| `--rerun-complete` | 完成的地图也重新采集 |
| `--smoke` | 使用隔离输出，只采一条极小序列 |

## 核心配置

| 配置项 | 当前基础值 | 说明 |
|---|---:|---|
| `sequences_per_map` | 15 | 正式采集每张地图的完整序列数 |
| `sequences_by_motion_mode` | 每种 5 条 | 三种运动模式的正式配额 |
| `frames_by_motion_mode` | 最长 80 / 120 / 150 | 三种模式的默认/车辆上限 |
| `frames_by_motion_mode_and_target_class` | 人物跟随 80 / 人物环绕 100 | 保持人物序列原覆盖时长 |
| `fixed_hover_stop_after_absent_frames` | 5 | 固定悬停连续缺失 5 帧即结束 |
| `min_frames_by_motion_mode.fixed_hover` | 30 | 固定悬停最低有效长度 |
| `sample_interval_ticks_by_target_class` | 车辆 2 / 人物 6 | 分别对应 10 Hz 和约 3.33 Hz |
| `height_min` / `height_max` | 30 / 36 m | 相机高度范围 |
| `radius_min` / `radius_max` | 28 / 52 m | 相机与目标的水平距离 |
| `pitch_min` / `pitch_max` | -45° / -35° | 相机俯角范围 |
| `vehicles` / `walkers` | 20 / 30 | 尝试生成的目标数量 |
| `min_visible_ratio` | 0.10 | 主目标最低可见比例 |
| `min_visible_pixels` | 24 | 目标最低可见像素数 |
| `max_absent_ratio_by_motion_mode` | 固定悬停 0.25 / 其余 0.40 | 单序列最大缺失帧比例 |
| `max_consecutive_absent_frames_by_motion_mode` | 固定悬停 5 / 其余 40 | 最大连续缺失帧数 |
| `min_vehicle_equivalent_side_px` | 40 px | 车辆等效边长下限 |
| `min_pedestrian_equivalent_side_px` | 35 px | 行人等效边长下限 |

命令行可覆盖序列数、帧数、天气和输出目录。查看完整参数：

```powershell
& $PYTHON "$PROJECT\collect_uav_single_object_vot_carla.py" --help
```

## 输出结构

```text
<map_root>/
├─ vot/<sequence_name>/
│  ├─ color/00000001.png
│  ├─ groundtruth.txt
│  ├─ absence.label
│  ├─ occlusion.label
│  ├─ target_state.label
│  ├─ sequence_meta.json
│  ├─ annotations.jsonl
│  └─ labels_yolo/
├─ yolo/
│  ├─ images/{train,val,test}/
│  ├─ labels/{train,val,test}/
│  └─ data.yaml
├─ dataset_manifest.json
├─ quality_audit.json
└─ README.md
```

`groundtruth.txt` 每行采用 `x,y,width,height`。当主目标不可见或严重遮挡时写入 `0,0,0,0`，同时在 `absence.label` 或 `occlusion.label` 中记录状态。

## 质量与续跑

- 第一帧必须存在合格主目标，否则整条尝试失败。
- 一条序列超过缺失比例或连续缺失上限时不会进入正式结果。
- `quality_audit.json` 应为 `PASS`，并检查序列级数据划分和 VOT 必需文件。
- 批量运行器按地图核对图像数量、天气分布、清单新鲜度和质量审计。
- `--smoke` 输出与正式数据隔离，适合验证环境和地图兼容性。
- 特殊地图的几何检查用于拒绝未完整加载或大面积虚空画面，不能修复地图资源包
  自身的贴图缺失；若错误覆盖大部分地图，需修复或替换对应地图资源。

## 上传说明

上传工具会读取同目录的历史和完成文件以支持断点续传。移动或删除这些记录会导致重新扫描或重复提交。正式上传前应检查脚本顶部的本地数据目录与 Hugging Face `REPO_ID`。

## 注意事项

- 正式单相机项目是 VOT/SOT，不是旧的单相机多目标 MOT 方案。
- `--overwrite` 是破坏性选项，只应用于确认可以重建的输出目录。
- 同一序列内天气和主目标身份必须保持不变。
- 不要按帧随机拆分训练集、验证集和测试集，否则会产生相邻帧泄漏。
- CARLA RPC 端口默认是 2000；不要同时运行另一个采集聊天或采集脚本。
