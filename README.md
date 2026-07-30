# 空地一体数据集

该项目为空地一体数据集相关的源代码和运行说明，收集好的数据位于[huggingface](https://huggingface.co/datasets/yutiangu/HUBT_from_a_drones_perspective)。


# Python 脚本说明

本文件只说明 `E:\pythonProject\air_groud\get_img\collect` 中各个 Python
脚本的用途、输入输出和运行方式。

## 运行环境

需要连接 OpenHUTB/CARLA 模拟器的采集脚本使用：


运行采集脚本前，需要先启动 OpenHUTB 模拟器。
## 脚本总览

| Python 文件 | 类型 | 是否需要模拟器 | 主要用途 |
|---|---|---:|---|
| `collect_rpg_small_targets_carla_v2.py` | 采集 | 是 | 采集 RGB、Surface Normal、Segmentation、Depth 四模态无人机小目标数据 |
| `collect_uav_single_object_vot_carla.py` | 采集 | 是 | 采集 VOT 格式的无人机单目标跟踪数据 |
| `collect_uav_multicamera_mot_carla.py` | 采集 | 是 | 采集三相机同步的跨相机多目标跟踪数据 |


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


