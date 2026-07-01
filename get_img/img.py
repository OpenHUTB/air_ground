# 30m、50m、80m，无人机俯视拍照


import carla
import math
import queue
import os
import numpy as np
import cv2
import random

# ==============================================================================
# 核心参数配置
# ==============================================================================
NUM_PHOTOS = 10  # 控制要拍摄的照片组数 (每组包含3个高度)
HEIGHTS = [30.0, 50.0, 80.0]  # 无人机高度设定 (米)
DISTANCE_BEHIND = 15.0  # 无人机在目标后方的水平距离 (米)

# 相机内参设置 (必须与生成的传感器蓝图一致)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
FOV = 90.0

# 定义 3D 目标类型前缀与 YOLO 类别的映射关系
CLASS_MAPPING = {
    'walker.pedestrian': 0,  # YOLO 类别 0: 行人
    'vehicle': 1  # YOLO 类别 1: 车辆
}


# ==============================================================================


def build_projection_matrix(w, h, fov):
    """构建相机的内参矩阵 (Intrinsic Matrix)"""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_camera_intrinsic():
    """获取内参矩阵"""
    return build_projection_matrix(IMG_WIDTH, IMG_HEIGHT, FOV)


def calculate_transform(walker_transform, height, distance_behind):
    """计算无人机在目标后上方的绝对位姿"""
    walker_loc = walker_transform.location
    walker_rot = walker_transform.rotation

    yaw_rad = math.radians(walker_rot.yaw)
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)

    cam_x = walker_loc.x - (forward_x * distance_behind)
    cam_y = walker_loc.y - (forward_y * distance_behind)
    cam_z = walker_loc.z + height

    pitch = math.degrees(math.atan2(-height, distance_behind))
    cam_loc = carla.Location(x=cam_x, y=cam_y, z=cam_z)
    cam_rot = carla.Rotation(pitch=pitch, yaw=walker_rot.yaw, roll=0.0)

    return carla.Transform(cam_loc, cam_rot)


def setup_camera(world, blueprint_library, camera_type='sensor.camera.rgb'):
    """配置相机蓝图，强制关闭 RGB 相机的模糊特效"""
    camera_bp = blueprint_library.find(camera_type)
    camera_bp.set_attribute('image_size_x', str(IMG_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))

    if camera_type == 'sensor.camera.rgb':
        # 彻底阉割引擎级别的画面模糊特效，保证微小目标边缘绝对锐利
        for attr in ['motion_blur_intensity', 'motion_blur_max_distortion', 'blur_amount']:
            if camera_bp.has_attribute(attr):
                camera_bp.set_attribute(attr, '0.0')
    return camera_bp


def get_2d_bounding_box(actor, camera, K):
    """
    核心：将 3D 边界框的 8 个顶点通过内外参矩阵投影到 2D 像素平面。
    支持视锥体截断处理，解决边缘目标漏检问题。
    """
    bb = actor.bounding_box
    vertices = bb.get_world_vertices(actor.get_transform())

    # 获取相机到世界坐标系的逆变换矩阵 (Extrinsic Matrix)
    cam_transform = camera.get_transform()
    cam_matrix = np.array(cam_transform.get_matrix())
    world_to_sensor = np.linalg.inv(cam_matrix)

    points_2d = []
    for vertex in vertices:
        p_world = np.array([[vertex.x], [vertex.y], [vertex.z], [1.0]])
        p_sensor = np.dot(world_to_sensor, p_world)

        # Carla 左手坐标系转相机标准坐标系：相机前方为 Z，右方为 X，下方为 Y
        p_cam = np.array([p_sensor[1], -p_sensor[2], p_sensor[0]])

        # 只要顶点在相机前方（Z > 0）就可以投影
        if p_cam[2] > 0:
            p_img = np.dot(K, p_cam)
            u = p_img[0] / p_img[2]
            v = p_img[1] / p_img[2]
            points_2d.append([u[0], v[0]])

    # 只要至少有 2 个点在视野前方，就可以尝试画出截断框
    if len(points_2d) < 2:
        return None

    points_2d = np.array(points_2d)

    # 获取最小外接矩形，并限制在画面范围内（截断）
    x_min = np.clip(np.min(points_2d[:, 0]), 0, IMG_WIDTH)
    x_max = np.clip(np.max(points_2d[:, 0]), 0, IMG_WIDTH)
    y_min = np.clip(np.min(points_2d[:, 1]), 0, IMG_HEIGHT)
    y_max = np.clip(np.max(points_2d[:, 1]), 0, IMG_HEIGHT)

    # 过滤无效或过度截断的极端噪点框
    if x_max - x_min < 10 or y_max - y_min < 10:
        return None

    return int(x_min), int(y_min), int(x_max), int(y_max)


def check_occlusion(x_min, y_min, x_max, y_max, depth_image, actor, camera):
    """
    终极版：基于相机前向向量点乘的 Z-Depth 深度校验。
    解决边缘目标因欧氏距离膨胀导致误判为遮挡，以及高大车辆自我遮挡的问题。
    """
    actor_loc = actor.get_location()
    cam_transform = camera.get_transform()
    cam_loc = cam_transform.location
    forward_vec = cam_transform.get_forward_vector()

    # 1. 获取相机到目标的向量
    dx = actor_loc.x - cam_loc.x
    dy = actor_loc.y - cam_loc.y
    dz = actor_loc.z - cam_loc.z

    # 2. 通过点乘计算目标在相机光学平面上的真实正交深度 (Z-Depth)
    true_z_depth = dx * forward_vec.x + dy * forward_vec.y + dz * forward_vec.z

    # 3. 解析深度图为真实距离矩阵 (米)
    array = np.frombuffer(depth_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (IMG_HEIGHT, IMG_WIDTH, 4)).astype(np.float32)
    depth_map = (array[:, :, 2] + array[:, :, 1] * 256.0 + array[:, :, 0] * 256.0 * 256.0) / (16777215.0) * 1000.0

    # 4. 提取 2D 框内的深度信息
    roi_depth = depth_map[y_min:y_max, x_min:x_max]
    if roi_depth.size == 0:
        return 0.0

    # 5. 动态计算深度容差
    extent = actor.bounding_box.extent
    max_dimension = math.sqrt(extent.x ** 2 + extent.y ** 2 + extent.z ** 2)
    # 给定基础容差并加上物体的对角线尺寸补偿，包容大型车辆的深度跨度
    tolerance = 2.5 + max_dimension

    # 6. 计算在容差范围内的有效像素数
    visible_pixels = np.sum((roi_depth > (true_z_depth - tolerance)) & (roi_depth < (true_z_depth + tolerance)))

    # 返回遮挡率 (填充率)
    return visible_pixels / roi_depth.size


def get_sync_frame(sensor_queue, target_frame):
    """阻塞队列直到获取与目标物理帧对齐的数据"""
    while True:
        try:
            data = sensor_queue.get(True, 5.0)
        except queue.Empty:
            print("警告: 传感器队列读取超时。")
            return None
        if data.frame == target_frame:
            return data
        elif data.frame > target_frame:
            return data


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 开启同步模式，确保数据对齐
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    actor_list = []

    # 初始化数据集目录结构
    base_dir = "multimodal_uav_dataset"
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    debug_dir = os.path.join(base_dir, "debug_vis")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    K = get_camera_intrinsic()

    try:
        print("正在初始化环境资产...")
        # === 1. 生成锚点行人 (无人机将跟随拍摄该行人) ===
        walker_bp = blueprint_library.filter('walker.pedestrian.*')[0]
        spawn_location = world.get_random_location_from_navigation()
        if spawn_location is None:
            spawn_location = carla.Location(0, 0, 1)
            print("警告: 未找到导航网格，行人生成在原点。")

        walker = world.spawn_actor(walker_bp, carla.Transform(spawn_location))
        actor_list.append(walker)

        walker_controller = world.spawn_actor(blueprint_library.find('controller.ai.walker'), carla.Transform(), walker)
        actor_list.append(walker_controller)

        world.tick()  # 让行人在引擎注册
        walker_controller.start()
        walker_controller.go_to_location(world.get_random_location_from_navigation())
        walker_controller.set_max_speed(1.5)

        # === 2. 随机生成环境背景车辆 ===
        vehicle_bps = blueprint_library.filter('vehicle.*')
        spawn_points = world.get_map().get_spawn_points()
        for i in range(1, min(15, len(spawn_points))):
            try:
                veh = world.spawn_actor(random.choice(vehicle_bps), spawn_points[i])
                veh.set_autopilot(True)
                actor_list.append(veh)
            except:
                pass

        # === 3. 挂载同步双相机 (RGB + Depth) ===
        rgb_camera = world.spawn_actor(setup_camera(world, blueprint_library, 'sensor.camera.rgb'),
                                       carla.Transform(spawn_location))
        depth_camera = world.spawn_actor(setup_camera(world, blueprint_library, 'sensor.camera.depth'),
                                         carla.Transform(spawn_location))
        actor_list.extend([rgb_camera, depth_camera])

        rgb_queue, depth_queue = queue.Queue(), queue.Queue()
        rgb_camera.listen(rgb_queue.put)
        depth_camera.listen(depth_queue.put)

        global_img_count = 0

        print("环境预热中...")
        for _ in range(40):
            world.tick()

        print("开始采集数据集...")
        for i in range(NUM_PHOTOS):
            # 组间间隔 60 帧，增加背景多样性
            for _ in range(60):
                world.tick()

            walker_transform = walker.get_transform()

            # 在同一时刻执行不同高度连拍
            for h in HEIGHTS:
                # 定位无人机
                target_transform = calculate_transform(walker_transform, h, DISTANCE_BEHIND)
                rgb_camera.set_transform(target_transform)
                depth_camera.set_transform(target_transform)

                # 推进物理世界获取图像
                frame_id = world.tick()
                rgb_image = get_sync_frame(rgb_queue, frame_id)
                depth_image = get_sync_frame(depth_queue, frame_id)

                if not rgb_image or not depth_image:
                    continue

                labels = []
                # 提取 RGB 供 OpenCV 调试绘画使用
                rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8")).reshape(
                    (IMG_HEIGHT, IMG_WIDTH, 4))
                debug_img = rgb_array[:, :, :3].copy()

                # 遍历当前环境中所有的动态 Actor
                for actor in world.get_actors().filter('*'):
                    actor_type = actor.type_id

                    yolo_class_id = -1
                    if actor_type.startswith('walker.pedestrian'):
                        yolo_class_id = 0
                    elif actor_type.startswith('vehicle'):
                        yolo_class_id = 1
                    else:
                        continue  # 跳过路灯、假车等非目标物体

                    # 1. 物理投影获取 2D 框
                    bbox = get_2d_bounding_box(actor, rgb_camera, K)
                    if not bbox:
                        continue
                    x_min, y_min, x_max, y_max = bbox

                    # 2. 深度图校验计算遮挡率
                    fill_ratio = check_occlusion(x_min, y_min, x_max, y_max, depth_image, actor, rgb_camera)

                    # 3. 遮挡过滤 (丢弃有效面积 < 15% 的极度遮挡样本)
                    if fill_ratio < 0.15:
                        continue

                    # 4. 转换为 YOLO 格式并记录
                    w, h_box = x_max - x_min, y_max - y_min
                    x_center, y_center = (x_min + w / 2.0) / IMG_WIDTH, (y_min + h_box / 2.0) / IMG_HEIGHT
                    box_width, box_height = w / IMG_WIDTH, h_box / IMG_HEIGHT

                    # TXT 格式: <cls_id> <x> <y> <w> <h> <fill_ratio>
                    labels.append(
                        f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {fill_ratio:.3f}")

                    # 5. 可视化绘制
                    color = (0, 0, 255) if yolo_class_id == 0 else (255, 0, 0)  # 行人红框，车辆蓝框
                    cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(debug_img, f"cls:{yolo_class_id} occ:{fill_ratio:.2f}", (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # 只有检出目标时才保存数据
                if labels:
                    file_prefix = f"frame_{global_img_count:06d}_h{int(h)}"

                    # 保存 RGB 图
                    rgb_image.save_to_disk(os.path.join(img_dir, f"{file_prefix}.png"))

                    # 保存 YOLO 标签
                    with open(os.path.join(lbl_dir, f"{file_prefix}.txt"), 'w') as f:
                        f.write("\n".join(labels))

                    # 保存 Debug 诊断图
                    cv2.imwrite(os.path.join(debug_dir, f"{file_prefix}_DEBUG.jpg"), debug_img)

                    print(f"成功保存: {file_prefix}.png | 高度: {h}m | 共检出 {len(labels)} 个目标")
                    global_img_count += 1

    finally:
        print("正在清理环境...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list:
            actor.destroy()
        print("采集任务完成！请检查 debug_vis 文件夹确认标注质量。")


if __name__ == '__main__':
    main()