import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

from common import PROJECT_ROOT, RAW_DATA_DIR, GAZE_TOPO_DIR, SCENARIO_FOLDERS

RELATIVE_POS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_RelativePosition")

RESULTS_BASE_DIR = os.path.join(GAZE_TOPO_DIR, "Results_GazePattern")

epsilon = 0.097
min_samples = 10
target_contour_value = [0.02]


def get_local_position_2d(sub_x, sub_z, sub_rot_y, gaze_x, gaze_z):
    rotation_rad = np.radians(sub_rot_y)
    translated_x = gaze_x - sub_x
    translated_z = gaze_z - sub_z
    rotation_matrix = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad),  np.cos(rotation_rad)],
    ])
    translated_vector = np.array([translated_x, translated_z])
    local_plane = np.dot(rotation_matrix, translated_vector)
    return np.array([float(local_plane[0]), float(local_plane[1])])


def collect_relative_positions_from_raw(scenario_path):
    all_positions = []
    subject_folders = sorted([
        f for f in os.listdir(scenario_path)
        if os.path.isdir(os.path.join(scenario_path, f)) and f != "Results"
    ])
    for subject_folder in subject_folders:
        subject_path = os.path.join(scenario_path, subject_folder)
        eye_files = [
            f for f in os.listdir(subject_path)
            if "EyeGaze_Simulation_Info" in f and f.endswith(".csv")
        ]
        sub_files = [
            f for f in os.listdir(subject_path)
            if "Subject_Simulation_Info" in f and f.endswith(".csv")
        ]
        if not eye_files or not sub_files:
            continue
        for eye_file in eye_files:
            base_eye = os.path.basename(eye_file)
            parts_eye = base_eye.split("_")
            if len(parts_eye) < 6:
                continue
            eye_ts = parts_eye[4] + "_" + parts_eye[5]
            matched_sub = None
            for sub_file in sub_files:
                base_sub = os.path.basename(sub_file)
                parts_sub = base_sub.split("_")
                if len(parts_sub) < 6:
                    continue
                sub_ts = parts_sub[4] + "_" + parts_sub[5]
                if eye_ts == sub_ts:
                    matched_sub = sub_file
                    break
            if matched_sub is None:
                continue
            try:
                eye_df = pd.read_csv(os.path.join(subject_path, eye_file), index_col=False)
                sub_df = pd.read_csv(os.path.join(subject_path, matched_sub), index_col=False)
            except Exception:
                continue
            if "TimeStamp" not in eye_df.columns or "TimeStamp" not in sub_df.columns:
                continue
            eye_df = eye_df.drop_duplicates(subset=["TimeStamp"], keep="first")
            sub_df = sub_df.drop_duplicates(subset=["TimeStamp"], keep="first")
            eye_mat = eye_df.to_numpy()
            sub_mat = sub_df.to_numpy()
            eye_cols = list(eye_df.columns)
            sub_cols = list(sub_df.columns)
            i_ts_eye = eye_cols.index("TimeStamp")
            i_ts_sub = sub_cols.index("TimeStamp")
            eye_by_ts = {}
            for r in range(len(eye_df)):
                ts = str(eye_mat[r, i_ts_eye])
                eye_by_ts[ts] = r
            for r in range(len(sub_df)):
                ts = str(sub_mat[r, i_ts_sub])
                if ts not in eye_by_ts:
                    continue
                er = eye_by_ts[ts]
                try:
                    sx = float(sub_mat[r, 1])
                    sz = float(sub_mat[r, 3])
                    srot_y = float(sub_mat[r, 5])
                    gx = float(eye_mat[er, 1])
                    gz = float(eye_mat[er, 3])
                except (ValueError, IndexError):
                    continue
                loc = get_local_position_2d(sx, sz, srot_y, gx, gz)
                all_positions.append(loc)
    return np.array(all_positions)


def load_scenario_relative_positions(scenario_name):
    scenario_rel_dir = os.path.join(RELATIVE_POS_BASE_DIR, scenario_name)
    combined_csv = os.path.join(scenario_rel_dir, "AllSubjects_RelativePosInfo.csv")
    if os.path.exists(combined_csv):
        df = pd.read_csv(combined_csv)
        if "LocalPosX" in df.columns and "LocalPosY" in df.columns:
            x = df["LocalPosX"].to_numpy(dtype=float)
            y = df["LocalPosY"].to_numpy(dtype=float)
            return np.column_stack([x, y])
    all_points = []
    if os.path.isdir(scenario_rel_dir):
        subject_dirs = sorted([
            d for d in os.listdir(scenario_rel_dir)
            if os.path.isdir(os.path.join(scenario_rel_dir, d))
        ])
        for sub_dir in subject_dirs:
            csv_path = os.path.join(scenario_rel_dir, sub_dir, "RelativePosInfo.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if "LocalPosX" in df.columns and "LocalPosY" in df.columns:
                    x = df["LocalPosX"].to_numpy(dtype=float)
                    y = df["LocalPosY"].to_numpy(dtype=float)
                    all_points.append(np.column_stack([x, y]))
    if all_points:
        return np.vstack(all_points)
    scenario_path = os.path.join(RAW_DATA_DIR, scenario_name)
    if os.path.isdir(scenario_path):
        return collect_relative_positions_from_raw(scenario_path)
    return np.empty((0, 2))


def execute_gaze_pattern_analysis(points, scenario_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  {scenario_name} 数据点总数：{len(points)}")
    if len(points) < 2:
        print(f"  [跳过] {scenario_name} 数据点不足")
        return
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = db.labels_
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    noise_percent = noise_count / len(labels) * 100
    print(f"  聚类完成：找到{n_clusters}个簇，噪声点占比{noise_percent:.1f}%")
    cmap = ListedColormap(['#999999'] + plt.cm.terrain(np.linspace(0, 1, n_clusters)).tolist())
    plt.figure(figsize=(12, 12))
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 24
    for cluster_id in range(-1, n_clusters):
        mask = (labels == cluster_id)
        if cluster_id == -1:
            plt.scatter(points[mask, 0], points[mask, 1], c='#999999',
                        s=10, alpha=0.3, label='Noise')
        else:
            plt.scatter(points[mask, 0], points[mask, 1],
                        c=[cmap(cluster_id + 1)],
                        s=10, alpha=0.6, label=f'Cluster {cluster_id}')
    plt.title(f'{scenario_name} DBSCAN Clustering', fontsize=45, y=1.05)
    plt.xlabel('X Position', fontsize=40)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.12, 0.3)
    plt.ylabel('Y Position', rotation=270, va='top', ha='right', fontsize=40)
    plt.xticks(ticks=[-10, -5, 0, 5, 10], labels=['-10', '-5', '0', '5', '10'], fontsize=40)
    plt.yticks(ticks=[0, 5, 10, 15, 20], labels=['0', '5', '10', '15', '20'], fontsize=40)
    plt.grid(alpha=0.3)
    plt.xlim(-10, 10)
    plt.ylim(0, 20)
    plt.gca().set_aspect('equal')
    dbscan_path = os.path.join(output_dir, f"{scenario_name}_DBSCAN.png")
    plt.savefig(dbscan_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"  DBSCAN聚类图已保存")
    m_mask = labels != -1
    valid_points = points[m_mask]
    peak_density = 0.0
    peak_x = 0.0
    peak_y = 0.0
    if len(valid_points) > 1:
        xmin, xmax = -10, 10
        ymin, ymax = 0, 20
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = gaussian_kde(valid_points.T)
        f = np.reshape(kernel(positions).T, xx.shape)
        plt.figure(figsize=(12, 12))
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 32
        plt.contourf(xx, yy, f, levels=15, cmap='gist_earth', vmin=0, vmax=0.1)
        plt.colorbar(shrink=0.75, aspect=12)
        plt.xticks(ticks=[-10, -5, 0, 5, 10], labels=['-10', '-5', '0', '5', '10'])
        plt.yticks(ticks=[0, 5, 10, 15, 20], labels=['0', '5', '10', '15', '20'])
        sorted_contour_values = sorted(target_contour_value)
        plt.contour(xx, yy, f, levels=sorted_contour_values, colors='red', linewidths=2)
        plt.title(f'{scenario_name} 2D KDE Visualization', y=1.05)
        plt.xlabel('X Position', labelpad=15)
        plt.tick_params(axis='x', which='both', pad=8)
        plt.tick_params(axis='y', which='both', pad=12)
        plt.ylabel('Y Position', rotation=270, va='bottom', ha='center', labelpad=36)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal')
        kde_2d_path = os.path.join(output_dir, f"{scenario_name}_KDE_2D.png")
        plt.savefig(kde_2d_path, bbox_inches='tight', dpi=600)
        plt.close()
        print(f"  2D KDE图已保存")
        fig = plt.figure(figsize=(12, 12))
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 30
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy, f, cmap='gist_earth',
                               rstride=1, cstride=1,
                               linewidth=0, antialiased=True)
        fig.colorbar(surf, shrink=0.75, aspect=12)
        ax.view_init(elev=30, azim=235)
        ax.set_title(f'{scenario_name} 3D KDE Visualization', x=0.62)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 25
        ax.zaxis.set_tick_params(pad=10)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ztext = ax.set_zlabel('Density')
        ax.set_zlim(0, 0.1)
        kde_3d_path = os.path.join(output_dir, f"{scenario_name}_KDE_3D.png")
        plt.savefig(kde_3d_path, dpi=600, bbox_inches='tight', bbox_extra_artists=[ztext])
        plt.close()
        print(f"  3D KDE图已保存")
        max_idx = np.unravel_index(np.argmax(f), f.shape)
        peak_density = f[max_idx]
        peak_x = xx[max_idx]
        peak_y = yy[max_idx]
    output_content = []
    output_content.append(f"=== {scenario_name} Gaze Pattern Analysis Report ===")
    output_content.append(f"")
    output_content.append(f"--- Density Estimation ---")
    output_content.append(f"KDE峰值概率密度: {peak_density:.5f}")
    output_content.append(f"KDE峰值坐标: ({peak_x:.5f}, {peak_y:.5f})")
    output_content.append(f"")
    output_content.append(f"--- DBSCAN Clustering ---")
    output_content.append(f"Total clusters: {n_clusters}")
    output_content.append(f"Total points: {len(points)}")
    output_content.append(f"Noise points: {noise_count} ({noise_percent:.1f}%)")
    output_content.append(f"")
    for cluster_id in sorted(unique_labels):
        if cluster_id == -1:
            continue
        cluster_points = points[labels == cluster_id]
        angles = np.degrees(np.arctan2(cluster_points[:, 1], cluster_points[:, 0]))
        angles = (angles + 360) % 360
        distances = np.sqrt(cluster_points[:, 0] ** 2 + cluster_points[:, 1] ** 2)
        output_content.append(f"Cluster {cluster_id}:")
        output_content.append(f"  Point count: {len(cluster_points)}")
        output_content.append(f"  Angle range: {np.min(angles):.1f}° - {np.max(angles):.1f}°")
        output_content.append(f"  Distance range: {np.min(distances):.2f}m - {np.max(distances):.2f}m")
        output_content.append(f"")
    info_name = f"{scenario_name}_cluster_info_eps{epsilon}_minsamp{min_samples}.txt"
    with open(os.path.join(output_dir, info_name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))
    print(f"  分析报告已保存")


def main():
    print(f"\n{'=' * 70}")
    print(f"眼动注视模式解析 - DBSCAN + KDE 聚类分析")
    print(f"{'=' * 70}")
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"相对位置数据源: {RELATIVE_POS_BASE_DIR}")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    print(f"场景数量: {len(SCENARIO_FOLDERS)}")
    print(f"DBSCAN参数: eps={epsilon}, min_samples={min_samples}")
    print(f"{'=' * 70}\n")
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    total_scenarios = len(SCENARIO_FOLDERS)
    for scenario_idx, scenario_folder in enumerate(SCENARIO_FOLDERS, 1):
        scenario_path = os.path.join(RAW_DATA_DIR, scenario_folder)
        if not os.path.exists(scenario_path):
            print(f"[{scenario_idx}/{total_scenarios}] [警告] 场景目录不存在: {scenario_folder}")
            continue
        print(f"\n{'=' * 70}")
        print(f"[{scenario_idx}/{total_scenarios}] 处理场景: {scenario_folder}")
        print(f"{'=' * 70}")
        points = load_scenario_relative_positions(scenario_folder)
        if len(points) == 0:
            print(f"  [跳过] 未找到该场景的相对位置数据")
            continue
        scenario_output_dir = os.path.join(RESULTS_BASE_DIR, scenario_folder)
        execute_gaze_pattern_analysis(points, scenario_folder, scenario_output_dir)
    print(f"\n{'=' * 70}")
    print(f"全部处理完成！")
    print(f"{'=' * 70}")
    print(f"结果已保存到: {RESULTS_BASE_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
