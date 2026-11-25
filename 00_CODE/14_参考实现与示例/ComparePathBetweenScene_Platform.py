import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


'''
提取并绘制站台层轨迹
Modified 2025-03-04
'''

# 受试者轨迹容器
subject_traj_total = []
subject_traj_security = []
subject_traj_passenger = []
subject_traj_robot = []

# 领导者轨迹容器
leader_traj_total = []
leader_traj_security = []
leader_traj_passenger = []
leader_traj_robot = []

# 根据第二阶段的领导者划分大组别
group_security = []
group_passenger = []
group_robot = []

target_level = "B5"


def get_grand_grand_parent_folder():
    current_script_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_script_path)
    parent_folder = os.path.dirname(current_folder)
    grandparent_folder = os.path.dirname(parent_folder)
    return grandparent_folder


def get_all_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths


def build_group_ref():
    global group_security
    global group_passenger
    global group_robot

    target_file_name = get_grand_grand_parent_folder() + os.sep + "Results" + os.sep + "Subject_Leader.csv"
    with open(target_file_name, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if len(row) > 1:
                name = row[0]
                if "Passenger" in str(row[3]):
                    group_passenger.append(name)
                if "Security" in str(row[3]):
                    group_security.append(name)
                if "Robot" in str(row[3]):
                    group_robot.append(name)
    print("Passenger组总计：" + str(len(group_passenger)))
    print("Security组总计：" + str(len(group_security)))
    print("Robot组总计：" + str(len(group_robot)))


def build_traj_list():
    global group_security
    global group_passenger
    global group_robot

    global subject_traj_total
    global subject_traj_security
    global subject_traj_passenger
    global subject_traj_robot

    global leader_traj_total
    global leader_traj_security
    global leader_traj_passenger
    global leader_traj_robot

    group_data_path = get_grand_grand_parent_folder() + os.sep + "GroupData"
    files = get_all_file_paths(group_data_path)
    progress_bar = tqdm(total=len(files))
    for file in files:
        file_name = os.path.basename(file)
        subject_name = file_name.split("_")[0]
        if target_level in file_name:
            new_subject_traj = []
            new_leader_traj = []

            # 构造轨迹列表
            with open(file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    sub_y_value = float(row[2])
                    if -15.8 < sub_y_value < -15.6:
                        sub_x = float(row[1])
                        sub_z = float(row[3])
                        new_subject_traj.append((sub_x, sub_z))
                    if "Leader" in str(row[34]):
                        leader_y_value = float(row[36])
                        if -15.8 < leader_y_value < -15.6:
                            leader_x = float(row[35])
                            leader_z = float(row[37])
                            new_leader_traj.append((leader_x, leader_z))

            # 添加到组别专属的数据容器中
            if subject_name in group_security:
                subject_traj_security.append(new_subject_traj)
                if target_level != "A1":
                    leader_traj_security.append(new_leader_traj)
                # 添加到场景公共数据容器中
                subject_traj_total.append(new_subject_traj)
                if target_level != "A1":
                    leader_traj_total.append(new_leader_traj)

            if subject_name in group_passenger:
                subject_traj_passenger.append(new_subject_traj)
                if target_level != "A1":
                    leader_traj_passenger.append(new_leader_traj)
                # 添加到场景公共数据容器中
                subject_traj_total.append(new_subject_traj)
                if target_level != "A1":
                    leader_traj_total.append(new_leader_traj)

            if subject_name in group_robot:
                subject_traj_robot.append(new_subject_traj)
                if target_level != "A1":
                    leader_traj_robot.append(new_leader_traj)
                # 添加到场景公共数据容器中
                subject_traj_total.append(new_subject_traj)
                if target_level != "A1":
                    leader_traj_total.append(new_leader_traj)

        # 更新进度条
        progress_bar.update(1)
        progress_bar.set_description(f"提取场景轨迹数据")

    # print("关卡Leader轨迹数：" + str(len(leader_traj_total)))
    # print("关卡Subject轨迹数：" + str(len(subject_traj_total)))


def draw():
    global subject_traj_total
    global subject_traj_security
    global subject_traj_passenger
    global subject_traj_robot

    global leader_traj_total
    global leader_traj_security
    global leader_traj_passenger
    global leader_traj_robot

    draw_level(subject_traj_total, leader_traj_total, "Total")
    draw_level(subject_traj_security, leader_traj_security, "Security")
    draw_level(subject_traj_passenger, leader_traj_passenger, "Passenger")
    draw_level(subject_traj_robot, leader_traj_robot, "Robot")


def get_bg_path():
    root = get_grand_grand_parent_folder()
    plat_bg_pic = root + os.sep + "Shots" + os.sep + "Plat.png"
    return plat_bg_pic


def draw_level(subject_traj, leader_traj, name):
    pic_path = get_bg_path()

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    # 读取底图
    base_image = mpimg.imread(pic_path)
    # 上下翻转底图
    flipped_base_image = np.flipud(base_image)

    # 获取底图的尺寸
    base_height, base_width = flipped_base_image.shape[:2]

    # 存储放大后的轨迹坐标
    scaled_subject_traj = []
    scaled_leader_traj = []

    scale_rate = 25.5

    # 遍历 subject_traj 中的每条轨迹并放大
    for traj in subject_traj:
        x_coords = [point[0] * scale_rate for point in traj]
        y_coords = [point[1] * scale_rate for point in traj]
        scaled_subject_traj.append(list(zip(x_coords, y_coords)))

    if target_level != "A1":
        # 遍历 leader_traj 中的每条轨迹并放大
        for traj in leader_traj:
            x_coords = [point[0] * scale_rate for point in traj]
            y_coords = [point[1] * scale_rate for point in traj]
            scaled_leader_traj.append(list(zip(x_coords, y_coords)))

    # 将 X、Y 坐标都乘以 -1
    negated_scaled_subject_traj = []
    for traj in scaled_subject_traj:
        new_traj = [(-x, -y) for x, y in traj]
        negated_scaled_subject_traj.append(new_traj)

    negated_scaled_leader_traj = []
    if target_level != "A1":
        for traj in scaled_leader_traj:
            new_traj = [(-x, -y) for x, y in traj]
            negated_scaled_leader_traj.append(new_traj)

    # 收集所有轨迹的 X 和 Y 坐标
    all_x_coords = []
    all_y_coords = []
    for traj in negated_scaled_subject_traj:
        x_coords = [point[0] for point in traj]
        y_coords = [point[1] for point in traj]
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)

    if target_level != "A1":
        for traj in negated_scaled_leader_traj:
            x_coords = [point[0] for point in traj]
            y_coords = [point[1] for point in traj]
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)

    # 计算轨迹的坐标范围
    if all_x_coords and all_y_coords:
        min_x = min(all_x_coords)
        max_x = max(all_x_coords)
        min_y = min(all_y_coords)
        max_y = max(all_y_coords)
    else:
        min_x, max_x, min_y, max_y = 0, 0, 0, 0

    # 计算底图和轨迹的综合范围
    x_range = max(max_x, base_width / 2) - min(min_x, -base_width / 2)
    y_range = max(max_y, base_height / 2) - min(min_y, -base_height / 2)

    # 计算宽高比
    aspect_ratio = x_range / y_range

    # 设置图形大小，让图形与内容的宽高比匹配
    if aspect_ratio > 1:
        width_inches = 8
        height_inches = width_inches / aspect_ratio
    else:
        height_inches = 8
        width_inches = height_inches * aspect_ratio

    plt.figure(figsize=(width_inches, height_inches))

    # 计算底图中心相对于轨迹坐标系原点的偏移量
    x_offset = -base_width / 2
    y_offset = -base_height / 2

    # 显示底图，使其中心与轨迹坐标系原点重合
    plt.imshow(flipped_base_image, origin='upper',
               extent=[x_offset, x_offset + base_width, y_offset + base_height, y_offset])

    # 定义轨迹向下移动的像素数
    y_shift = 290

    # 在底图上绘制放大且取负后的 subject_traj 轨迹，并向下移动
    for traj in negated_scaled_subject_traj:
        x_coords = [point[0] for point in traj]
        y_coords = [point[1] - y_shift for point in traj]
        # 指定颜色为蓝色，将 markersize 和 linewidth 设为原来的 0.2 倍
        plt.plot(x_coords, y_coords, marker='o', color='gold', markersize=0.025, linewidth=0.01)

    if target_level != "A1":
        # 在底图上绘制放大且取负后的 leader_traj 轨迹，并向下移动
        for traj in negated_scaled_leader_traj:
            x_coords = [point[0] for point in traj]
            y_coords = [point[1] - y_shift for point in traj]
            # 指定颜色为蓝色，将 markersize 和 linewidth 设为原来的 0.2 倍
            plt.plot(x_coords, y_coords, marker='o', color='tomato', markersize=0.025, linewidth=0.01)

    # 设置 X、Y 坐标轴比例尺一致
    plt.axis('equal')

    # 反转 Y 轴，使 Y 轴正向朝下
    plt.gca().invert_yaxis()

    plt.title(f"Trajectory Level:{target_level} Group:{name}")

    # 移除坐标轴和边框
    plt.axis('off')

    # 调整坐标轴范围，使内容充满图片且底图显示完全
    plt.xlim(-base_width / 2, base_width / 2)
    plt.ylim(-base_height / 2, base_height / 2)

    # 去掉 plt.tight_layout()，并设置保存图片时的参数让图片紧凑
    plt.savefig(name + '.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    build_group_ref()
    build_traj_list()
    draw()
