# 空地一体无人机多模态数据集

本项目包含基于 OpenHUTB/CARLA 的无人机视角多地图数据采集、质量检查和标注工具。
整理后的数据集发布在
[Hugging Face：HUTB From a Drone's Perspective](https://huggingface.co/datasets/yutiangu/HUBT_from_a_drones_perspective)。

## 当前数据集

当前版本包含 8 张仿真地图中的 4,081 组同步数据，图像分辨率为
1920 x 1080。数据覆盖车辆和行人检测、目标语义分割、深度估计、表面法向估计、
LiDAR 与跨地图/跨天气鲁棒性研究。

| 项目 | 数量 |
|---|---:|
| 地图 | 8 |
| 同步帧 | 4,081 |
| 训练集 | 2,856 |
| 验证集 | 816 |
| 测试集 | 409 |
| 车辆框 | 6,707 |
| 行人框 | 9,568 |
| 可训练目标实例 | 16,275 |
| 天气与光照条件 | 6 |

每一帧只属于一种天气，因此不同天气目录中的 RGB 文件是互不重复的帧，并不是
同一帧的多天气渲染。

## 地图分布

| 地图 | 帧数 | 训练 | 验证 | 测试 | 车辆 | 行人 |
|---|---:|---:|---:|---:|---:|---:|
| `CCSP_Zhongdian_Software_Park` | 181 | 126 | 36 | 19 | 182 | 311 |
| `HutbCarlaCity` | 300 | 210 | 60 | 30 | 533 | 618 |
| `Town02_Opt` | 600 | 420 | 120 | 60 | 1,244 | 1,108 |
| `Town03_Opt` | 600 | 420 | 120 | 60 | 864 | 1,198 |
| `Town04_Opt` | 600 | 420 | 120 | 60 | 698 | 2,081 |
| `Town05_Opt` | 600 | 420 | 120 | 60 | 880 | 883 |
| `Town07_Opt` | 600 | 420 | 120 | 60 | 901 | 2,342 |
| `Town10HD` | 600 | 420 | 120 | 60 | 1,405 | 1,027 |
| **总计** | **4,081** | **2,856** | **816** | **409** | **6,707** | **9,568** |

## 同步模态与标注

每一组已发布数据均包含以下同步内容：

| 内容 | 格式 |
|---|---|
| 可见光 RGB | 按天气分类的 8 位 PNG |
| 相机深度 | 米制 `float32` NPY、16 位 PNG 和彩色预览图 |
| 表面法向 | 相机坐标系 `float32` NPY 和 PNG 预览图 |
| 目标语义分割 | 类别 ID PNG 和彩色预览图 |
| LiDAR | 原始点云 NPY、RGB 对齐投影 NPY 和预览图 |
| 目标检测 | YOLO TXT、COCO JSON 和逐帧 JSON |

检测类别采用零起始 YOLO ID：`0` 为 `vehicle`，`1` 为 `pedestrian`。
目标语义分割中，像素值 `0` 为背景，`1` 为可训练车辆，`2` 为可训练行人，
`255` 为识别到但不用于训练的目标。

## 天气分布

| 天气 | 帧数 |
|---|---:|
| `ClearNoon` | 692 |
| `ClearSunset` | 690 |
| `ClearNight` | 682 |
| `FoggyNoon` | 681 |
| `SnowNoon` | 669 |
| `DustStorm` | 667 |
| **总计** | **4,081** |

## 数据目录

每张地图独立保存，整合后的序列目录使用地图名称，不再使用多个 `seq_xxxx`
目录：

```text
<MapName>/
|-- paired_weather/<MapName>/
|   |-- rgb/<Weather>/<frame>.png
|   |-- depth/
|   |-- surface_normal/
|   |-- segmentation/
|   |-- labels_yolo/
|   |-- annotations/
|   `-- frame_index.csv
|-- splits/{train,val,test}.txt
|-- splits_by_weather/
|-- coco/{train,val,test}.json
|-- data.yaml
|-- dataset_manifest.json
`-- quality_report.json
```

JSON、CSV、COCO 和数据划分文件中的路径均相对于对应的地图目录。

## YOLO 使用示例

每张地图都包含独立的 `data.yaml`。下载完整数据后，可在数据集根目录运行：

```powershell
yolo detect train model=yolo11x.pt data=Town02_Opt/data.yaml imgsz=1920 epochs=100
```

进行跨地图泛化实验时，建议保留一张或多张完整地图作为测试域，而不是将所有地图
随机混合后再划分。

## 质量与限制

- 4,081 帧均通过同步模态文件检查，并具有匹配的 RGB、深度、表面法向、目标分割、
  LiDAR、YOLO 和 JSON 数据。
- 数据来自仿真环境，与真实无人机数据之间存在域差异。
- 检测标注面向 CARLA 车辆与行人 Actor；没有 Actor 身份的烘焙静态网格不保证
  获得检测标签。
- 中电软件园自定义地图仍可能出现少量资源流送、网格或贴图瑕疵，训练前应按任务需要
  进行额外视觉检查。
- 天气效果是仿真近似；例如雪天不表示路面已有真实积雪。


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


