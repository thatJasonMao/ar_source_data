import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

'''
提取受试者对环境的注视点，相对于受试者自身的坐标
对相对位置关系进行DBSCAN聚类
考虑到部分关卡眼动交互（尤其是对领导者），直接合并所有关卡的注视数据
Modified 2025-4-9
注：本脚本只基于静态参数进行绘制
'''

# 根据第二阶段的领导者划分大组别（受试者容器）
group_security = []
group_passenger = []
group_robot = []

# 各组受试者对应的<数据容器>
infos_security = []
infos_passenger = []
infos_robot = []

# 各组受试者对应的<相对位置容器>
local_pos_security = []
local_pos_passenger = []
local_pos_robot = []
local_pos_total = []

# 邻点距离上限
epsilon = 0.097

# 核心邻点数下限
min_samples = 10

# 结果文件输出路径
sub_folder_path = "Static_DBSCAN_OnAll"

target_contour_value = [0.02]


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


def get_raw_data_path():
    target_file_name = get_grand_grand_parent_folder() + os.sep + "GroupData_Gaze_On_All"
    return target_file_name


def get_local_position(x_a, y_a, rotation_a, x_b, y_b):
    rotation_rad = np.radians(rotation_a)

    translated_x = x_b - x_a
    translated_y = y_b - y_a

    rotation_matrix = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad), np.cos(rotation_rad)]
    ])

    translated_vector = np.array([translated_x, translated_y])
    local_position = np.dot(rotation_matrix, translated_vector)

    return local_position


def get_data(path):
    raw_infos = []
    try:
        with open(path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if len(row) > 1:
                    if "Leader" not in str(row[7]):
                        sub_pos_x = float(row[1])
                        sub_pos_y = float(row[2])
                        sub_pos_z = float(row[3])
                        sub_rot_y = float(row[5])

                        leader_pos_x = float(row[7])
                        leader_pos_y = float(row[8])
                        leader_pos_z = float(row[9])

                        frame_data = [sub_pos_x, sub_pos_y, sub_pos_z, sub_rot_y, leader_pos_x, leader_pos_y, leader_pos_z]
                        raw_infos.append(frame_data)
                    else:
                        print(f"略过一处不合规数据 {path}")
    except:
        print("文件读取错误：" + path)
    return raw_infos


def build_data_ref():
    global group_security
    global group_passenger
    global group_robot

    global infos_security
    global infos_passenger
    global infos_robot

    global local_pos_security
    global local_pos_passenger
    global local_pos_robot
    global local_pos_total

    # 第一轮遍历 构建原始数据
    files = get_all_file_paths(get_raw_data_path())

    for file in files:
        level_name = os.path.basename(file)
        subject_name = level_name.split("_")[0]
        if subject_name in group_security:
            infos_security = infos_security + get_data(file)

    for file in files:
        level_name = os.path.basename(file)
        subject_name = level_name.split("_")[0]
        if subject_name in group_passenger:
            infos_passenger = infos_passenger + get_data(file)

    for file in files:
        level_name = os.path.basename(file)
        subject_name = level_name.split("_")[0]
        if subject_name in group_robot:
            infos_robot = infos_robot + get_data(file)

    # 第二轮遍历 构建相对位置数据
    for info in infos_security:
        new_pos = get_local_position(info[0], info[2], info[3], info[4], info[6])
        local_pos_security.append(new_pos)
        local_pos_total.append(new_pos)

    for info in infos_passenger:
        new_pos = get_local_position(info[0], info[2], info[3], info[4], info[6])
        local_pos_passenger.append(new_pos)
        local_pos_total.append(new_pos)

    for info in infos_robot:
        new_pos = get_local_position(info[0], info[2], info[3], info[4], info[6])
        local_pos_robot.append(new_pos)
        local_pos_total.append(new_pos)


def execute_dbscan(data_container, name):
    print(f"{name}数据容器长度：{len(data_container)}")
    # 转换数据格式为numpy数组
    points = np.array(data_container)

    # 执行DBSCAN聚类
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = db.labels_

    # 统计聚类信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_percent = np.sum(labels == -1) / len(labels) * 100
    print(f"{name}聚类完成：找到{n_clusters}个簇，噪声点占比{noise_percent:.1f}%")

    # 可视化设置
    plt.figure(figsize=(12, 12))
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 24
    cmap = ListedColormap(['#999999'] + plt.cm.terrain(np.linspace(0, 1, n_clusters)).tolist())

    # 绘制散点图（跳过噪声点）
    for cluster_id in range(-1, n_clusters):
        mask = (labels == cluster_id)
        if cluster_id == -1:
            plt.scatter(points[mask, 0], points[mask, 1], c='#999999',
                        s=10, alpha=0.3, label='Noise')
        else:
            plt.scatter(points[mask, 0], points[mask, 1],
                        c=[cmap(cluster_id + 1)],  # +1跳过第一个灰色
                        s=10, alpha=0.6, label=f'Cluster {cluster_id}')

    # 图例和坐标轴设置
    plt.title(f'{name} DBSCAN Clustering', fontsize=45, y=1.05)
    plt.xlabel('X Position', fontsize=40)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.12, 0.3)
    plt.ylabel('Y Position', rotation=270, va='top', ha='right', fontsize=40)

    plt.xticks(ticks=[-10, -5, 0, 5, 10], labels=['-10', '-5', '0', '5', '10'], fontsize=40)
    plt.yticks(ticks=[0, 5, 10, 15, 20], labels=['0', '5', '10', '15', '20'], fontsize=40)

    plt.grid(alpha=0.3)
    # 暂时隐藏图例
    # plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlim(-10, 10)
    plt.ylim(0, 20)
    plt.gca().set_aspect('equal')  # 强制等比例坐标系

    # 保存结果图片
    output_path = f"{sub_folder_path}/{name}_DBSCAN.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=600)
    plt.close()

    # 统计聚类信息
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

    # 计算轮廓系数均值
    m_labels = db.labels_
    m_mask = m_labels != -1
    # 暂时略去轮廓系数计算 节省性能
    # silhouette_avg = silhouette_score(points[m_mask], labels[m_mask])

    valid_points = points[m_mask]
    if len(valid_points) > 1:  # 需要至少2个点才能计算KDE
        # KDE图的坐标轴界限
        xmin, xmax = -10, 10
        ymin, ymax = 0, 20
        # 创建网格数据
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # 计算核密度估计
        kernel = gaussian_kde(valid_points.T)
        f = np.reshape(kernel(positions).T, xx.shape)

        # 绘制KDE图形
        plt.figure(figsize=(12, 12))
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 32
        plt.contourf(xx, yy, f, levels=15, cmap='gist_earth', vmin=0, vmax=0.1)
        plt.colorbar(shrink=0.75, aspect=12)

        plt.xticks(ticks=[-10, -5, 0, 5, 10], labels=['-10', '-5', '0', '5', '10'])
        plt.yticks(ticks=[0, 5, 10, 15, 20], labels=['0', '5', '10', '15', '20'])

        # 略去散点图绘制
        # plt.scatter(valid_points[:, 0], valid_points[:, 1], s=5, c='white', alpha=0.3)

        sorted_contour_values = sorted(target_contour_value)
        contour_lines = plt.contour(xx, yy, f, levels=sorted_contour_values, colors='red', linewidths=2)

        # 图形设置
        plt.title(f'{name} 2D KDE Visualization', y=1.05)
        plt.xlabel('X Position', labelpad=15)

        plt.tick_params(axis='x', which='both', pad=8)  # X轴刻度标签间距
        plt.tick_params(axis='y', which='both', pad=12)  # Y轴刻度标签间距

        plt.ylabel('Y Position', rotation=270, va='bottom', ha='center', labelpad=36)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # 保存KDE图片
        kde_path = f"{sub_folder_path}/{name}_KDE.png"
        plt.savefig(kde_path, bbox_inches='tight', dpi=600)
        plt.close()

        max_idx = np.unravel_index(np.argmax(f), f.shape)
        peak_density = f[max_idx]
        peak_x = xx[max_idx]
        peak_y = yy[max_idx]

    # 准备输出内容
    output_content = []
    output_content.append(f"=== {name} Density Estimation Report ===")
    output_content.append(f"KDE峰值概率密度: {peak_density:.5f}")
    output_content.append(f"KDE峰值坐标: ({peak_x:.5f}, {peak_y:.5f})")
    output_content.append("------------------------------")

    output_content.append(f"=== {name} Group Cluster Report ===")
    output_content.append(f"Total clusters: {n_clusters}")
    # output_content.append(f"Silhouette Score: {silhouette_avg:.3f}")
    output_content.append(f"Total points: {len(points)}")
    output_content.append(f"Noise points: {np.sum(labels == -1)} ({np.mean(labels == -1)*100:.1f}%)")
    output_content.append("------------------------------")

    # 遍历所有簇（排除噪声点）
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue

        cluster_points = points[labels == cluster_id]
        angles = np.degrees(np.arctan2(cluster_points[:, 1], cluster_points[:, 0]))  # 计算角度(Y/X)
        angles = (angles + 360) % 360  # 转换到0-360范围

        distances = np.sqrt(cluster_points[:, 0] ** 2 + cluster_points[:, 1] ** 2)

        # 记录簇信息
        output_content.append(f"Cluster {cluster_id}:")
        output_content.append(f"  Point count: {len(cluster_points)}")
        output_content.append(f"  Angle range: {np.min(angles):.1f}° - {np.max(angles):.1f}°")
        output_content.append(f"  Distance range: {np.min(distances):.2f}m - {np.max(distances):.2f}m")
        output_content.append("------------------------------")

    # 写入文件
    file_name = f"{name}_cluster_info_eps{epsilon}_minsamp{min_samples}.txt"
    with open(os.path.join(sub_folder_path, file_name), 'w') as f:
        f.write('\n'.join(output_content))


if __name__ == "__main__":
    if not os.path.exists(sub_folder_path):
        os.mkdir(sub_folder_path)
    build_group_ref()
    build_data_ref()
    execute_dbscan(local_pos_security, "Security")
    execute_dbscan(local_pos_passenger, "Passenger")
    execute_dbscan(local_pos_robot, "Robot")
    # execute_dbscan(local_pos_total, "Total")
