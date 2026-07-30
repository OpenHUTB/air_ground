# Python 脚本说明

本文件只说明 `E:\pythonProject\air_groud\get_img\collect` 中各个 Python
脚本的用途、输入输出和运行方式。

## 运行环境

需要连接 OpenHUTB/CARLA 模拟器的采集脚本使用：

```powershell
$PYTHON = "D:\anaconda2023.09\envs\openhutb\python.exe"
```

运行采集脚本前，需要先启动 OpenHUTB 模拟器，并确认服务端口为 `2000`。
审计、后处理和可视化脚本不要求模拟器保持运行。

## 脚本总览

| Python 文件 | 类型 | 是否需要模拟器 | 主要用途 |
|---|---|---:|---|
| `collect_rpg_small_targets_carla_v2.py` | 采集 | 是 | 采集 RGB、Surface Normal、Segmentation、Depth 四模态无人机小目标数据 |
| `visualize_rgb_annotations.py` | 可视化 | 否 | 按天气随机抽取图片并绘制标注和四模态对照图 |
| `qa_check_dataset_modalities.py` | 审计 | 否 | 检查四种模态的帧对齐、尺寸、缺失文件和标签质量 |
| `collect_target_semantic_instance_carla.py` | 采集 | 是 | 专项采集车辆和行人的目标语义分割数据 |
| `audit_target_semantic_dataset.py` | 审计 | 否 | 审计目标语义分割数据并生成随机叠加图 |
| `collect_surface_normal_gbuffer_carla.py` | 采集 | 是 | 从 Unreal GBufferA 专项采集 Surface Normal |
| `refine_surface_normal_gbuffer_dataset.py` | 后处理 | 否 | 对 GBuffer Surface Normal 做边缘保持优化 |
| `collect_uav_single_object_vot_carla.py` | 采集 | 是 | 采集 VOT 格式的无人机单目标跟踪数据 |
| `collect_uav_multicamera_mot_carla.py` | 采集 | 是 | 采集三相机同步的跨相机多目标跟踪数据 |
| `try.py` | 临时测试 | 否 | 检查当前 Python 环境加载的 CARLA 模块路径 |

## 四模态总采集脚本

### `collect_rpg_small_targets_carla_v2.py`

项目中的主采集脚本，用于生成无人机视角的小目标多模态数据。

采集内容：

- RGB 可见光图像
- Surface Normal 表面法线图
- Segmentation 语义分割图
- Depth 深度图
- 车辆和行人边界框
- 相机参数、目标属性和质量检查信息

脚本使用 `collection_config.json` 作为默认配置。命令行参数的优先级高于
JSON 配置，因此可以临时覆盖帧数、分辨率、高度、天气和目标数量。

推荐运行：

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_rpg_small_targets_carla_v2.py" `
  --config "E:\pythonProject\air_groud\get_img\collect\collection_config.json"
```

查看全部参数：

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_rpg_small_targets_carla_v2.py" --help
```

重要参数：

- `--out`：输出目录。
- `--sequences`：采集序列数量。
- `--frames`：每个序列的帧数。
- `--width`、`--height-img`：图像分辨率。
- `--height-min`、`--height-max`：无人机高度范围。
- `--pitch-min`、`--pitch-max`：相机俯视角范围。
- `--vehicles`、`--walkers`：生成车辆和行人的数量。
- `--weather-presets`：指定采集天气。
- `--min-actor-visible-ratio`：目标最低可见比例。
- `--overwrite-sequences`：覆盖已有序列。

## 四模态检查与可视化

### `visualize_rgb_annotations.py`

从数据集中按天气随机选择图片，为 RGB、Depth、Surface Normal 和
Segmentation 绘制对应目标框，并可生成四模态联系表。

默认每种天气随机选择 3 张图片。相同 `--seed` 会得到相同的随机结果。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\visualize_rgb_annotations.py" `
  --dataset "E:\pythonProject\air_groud\get_img\collect\dataset_uav_small_carla_config" `
  --max-frames 3 `
  --all-weather
```

常用参数：

- `--dataset`：数据集根目录。
- `--max-frames`：每种天气随机抽取的图片数。
- `--weather`：只可视化指定天气。
- `--all-weather`：处理数据集中的全部天气。
- `--out-dir`：可视化结果目录。
- `--no-contact-sheets`：不生成联系表。

### `qa_check_dataset_modalities.py`

用于检查四模态数据是否一一对应，主要检查：

- 各模态文件数量和帧编号是否一致。
- 图像宽高是否一致。
- RGB 与标签是否对齐。
- 深度、法线和分割图是否存在空白或异常值。
- 训练、验证和测试索引是否指向有效文件。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\qa_check_dataset_modalities.py" `
  --dataset "E:\pythonProject\air_groud\get_img\collect\dataset_uav_small_carla_config"
```

使用 `--weather` 可只检查一种天气，使用 `--max-frames` 可限制检查帧数。

## 语义分割专项脚本

### `collect_target_semantic_instance_carla.py`

只针对车辆和行人生成监督语义分割数据。

公开标签中的像素值：

- `0`：背景
- `1`：车辆
- `2`：行人
- `255`：忽略区域

内部实例分割、深度和 Actor 3D 框只用于匹配动态 Actor、过滤遮挡和生成
干净标签，不作为额外公开模态。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_target_semantic_instance_carla.py" `
  --out "E:\pythonProject\air_groud\get_img\collect\dataset_target_semantic" `
  --frames 500
```

快速验证采集逻辑：

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_target_semantic_instance_carla.py" `
  --out "E:\pythonProject\air_groud\get_img\collect\dataset_target_semantic_probe" `
  --probe-only
```

### `audit_target_semantic_dataset.py`

检查 RGB、目标语义掩码和 YOLO 分割标签是否对应，并生成
`audit_report.json` 和随机标注叠加图。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\audit_target_semantic_dataset.py" `
  "E:\pythonProject\air_groud\get_img\collect\dataset_target_semantic" `
  --random-overlays 6
```

## Surface Normal 专项脚本

### `collect_surface_normal_gbuffer_carla.py`

直接从 Unreal Engine 的 GBufferA 读取相机坐标系法线，用于专项采集和评估
Surface Normal。脚本同时保存对应 RGB 输入、法线标签、有效区域和质量报告。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_surface_normal_gbuffer_carla.py" `
  --out "E:\pythonProject\air_groud\get_img\collect\dataset_surface_normal_gbuffer" `
  --frames 500
```

主要质量参数：

- `--min-normal-valid-ratio`：最低有效法线比例。
- `--min-edge-alignment`：法线边缘与图像结构的最低对齐要求。
- `--valid-erode-px`：有效区域边缘收缩像素数。
- `--median-kernel`：中值滤波核大小。

### `refine_surface_normal_gbuffer_dataset.py`

对已经采集的 GBuffer 法线数据做中值滤波、双边滤波、有效区域腐蚀和边缘保持
优化，不连接模拟器，也不会重新采集其他模态。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\refine_surface_normal_gbuffer_dataset.py" `
  --source-root "E:\pythonProject\air_groud\get_img\collect\dataset_surface_normal_gbuffer" `
  --output-root "E:\pythonProject\air_groud\get_img\collect\dataset_surface_normal_refined"
```

该脚本不会原地修改源数据，必须分别指定 `--source-root` 和
`--output-root`。

## 跟踪数据采集脚本

### `collect_uav_single_object_vot_carla.py`

采集无人机视角的 RGB 单目标跟踪序列，主目标标注采用 VOT 矩形框格式。
脚本内部使用深度图和语义图进行遮挡、道路和穿模检查，但最终不保存为多模态
数据。

主要输出：

- 连续 RGB 帧。
- `groundtruth.txt`。
- `sequence_meta.json`。
- 全部合格车辆和行人的 YOLO 标签。
- `quality_audit.json` 和数据集清单。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_uav_single_object_vot_carla.py" `
  --config "E:\pythonProject\air_groud\get_img\collect\single_object_vot_config.json" `
  --overwrite
```

小规模试运行可以使用 `--max-sequences` 和 `--frames-per-sequence`。

### `collect_uav_multicamera_mot_carla.py`

采集跨相机多目标跟踪数据。每个场景使用 3 个同步无人机相机，所有相机在同一次
`world.tick()` 中取帧。

该脚本负责：

- 保存各相机 RGB 序列。
- 使用场景级 `global_id` 统一同一目标在不同相机和不同帧中的身份。
- 输出每相机 MOT `gt.txt`。
- 输出逐帧 JSON 标注、相机内外参和全局轨迹。
- 生成场景隔离的 YOLO train/val/test 目录。
- 检查重复图像、同步关系、遮挡率和身份划分泄漏。

推荐运行：

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_uav_multicamera_mot_carla.py" `
  --config "E:\pythonProject\air_groud\get_img\collect\multi_camera_mot_config.json" `
  --overwrite
```

只重新审计已有数据，不连接模拟器：

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\collect_uav_multicamera_mot_carla.py" `
  --out "E:\pythonProject\air_groud\get_img\collect\dataset_uav_multicamera_mot_rgb_1080" `
  --audit-only
```

小规模试运行可使用：

- `--max-scenes`
- `--frames-per-scene`
- `--scenes-per-weather`
- `--weather-presets`

## 临时环境测试脚本

### `try.py`

打印当前环境加载的 `carla` Python 模块路径，并输出一行测试文字。它不采集
数据，也不应作为正式实验入口。

```powershell
& $PYTHON "E:\pythonProject\air_groud\get_img\collect\try.py"
```

## YOLO 侧相关 Python 脚本

下面两个脚本位于
`E:\YOLO\ultralytics-main\ultralytics-main`，用于处理
`collect_uav_multicamera_mot_carla.py` 生成的数据。

### `train_yolo11x_uav_multicamera_mot.py`

训练和测试 YOLO11x 车辆、行人检测器，并输出标准 mAP、固定阈值
TP/FP/FN、漏检率和误检率。

```powershell
& "D:\anaconda2023.09\envs\yolo11n\python.exe" `
  "E:\YOLO\ultralytics-main\ultralytics-main\train_yolo11x_uav_multicamera_mot.py"
```

### `evaluate_yolo11x_multicamera_tracking.py`

读取 YOLO11x 检测结果和相机标定，通过世界坐标投影完成同帧跨相机关联和
跨帧身份关联，输出 IDP、IDR、IDF1、MOTA、IDSW 和跨相机配对一致率。

```powershell
& "D:\anaconda2023.09\envs\yolo11n\python.exe" `
  "E:\YOLO\ultralytics-main\ultralytics-main\evaluate_yolo11x_multicamera_tracking.py"
```

## 推荐使用顺序

四模态数据：

```text
collect_rpg_small_targets_carla_v2.py
-> qa_check_dataset_modalities.py
-> visualize_rgb_annotations.py
```

语义分割专项：

```text
collect_target_semantic_instance_carla.py
-> audit_target_semantic_dataset.py
```

Surface Normal 专项：

```text
collect_surface_normal_gbuffer_carla.py
-> refine_surface_normal_gbuffer_dataset.py
```

跨相机多目标跟踪：

```text
collect_uav_multicamera_mot_carla.py
-> train_yolo11x_uav_multicamera_mot.py
-> evaluate_yolo11x_multicamera_tracking.py
```
