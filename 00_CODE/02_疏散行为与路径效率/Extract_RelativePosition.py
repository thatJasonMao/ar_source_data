import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS, excel_col_to_index

# 结果输出根目录（所有场景的结果都输出到这里）
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_RelativePosition")


OUTPUT_RELATIVE_COLUMNS = [
    "TimeStamp",
    "SubjectWorldX",
    "SubjectWorldY",
    "SubjectWorldZ",
    "SubjectRotY_deg",
    "GazeWorldX",
    "GazeWorldY",
    "GazeWorldZ",
    "LocalPosX",
    "LocalPosY",
]

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


def _cell_is_nonempty(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True


def _parse_float(v):
    if not _cell_is_nonempty(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def calculate_heatmap_center_and_percentage_boundary(data, x_range, y_range, bins=(50, 50), top_percentage=0.1):
    data = np.asarray(data, dtype=float)
    x = data[:, 0]
    y = data[:, 1]
    center_x = float(np.mean(x))
    center_y = float(np.mean(y))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
    total_intensity = np.sum(heatmap)
    if total_intensity <= 0:
        return (center_x, center_y), [], np.array([]), float(total_intensity)
    sorted_intensity = np.sort(heatmap.flatten())[::-1]
    threshold_index = min(int(top_percentage * len(sorted_intensity)), len(sorted_intensity) - 1)
    threshold_value = sorted_intensity[threshold_index]
    high_intensity_points = np.argwhere(heatmap >= threshold_value)
    x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
    y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
    boundary_points = [(x_bin_centers[i], y_bin_centers[j]) for i, j in high_intensity_points]
    return (center_x, center_y), boundary_points, sorted_intensity, float(total_intensity)


def calculate_concentration_index(sorted_intensity, total_intensity, top_percentage=0.1):
    if total_intensity <= 0 or len(sorted_intensity) == 0:
        return 0.0
    threshold_index = min(int(top_percentage * len(sorted_intensity)), len(sorted_intensity) - 1)
    top_intensity = np.sum(sorted_intensity[:threshold_index])
    return float(top_intensity / total_intensity)


def save_relative_pos_heatmap(output_folder: str, subject_name: str, out_df: pd.DataFrame) -> None:
    if out_df is None or len(out_df) == 0:
        return
    if "LocalPosX" not in out_df.columns or "LocalPosY" not in out_df.columns:
        return

    x = out_df["LocalPosX"].to_numpy(dtype=float)
    y = out_df["LocalPosY"].to_numpy(dtype=float)
    if len(x) == 0:
        return

    x_range = (-10.0, 10.0)
    y_range = (0.0, 20.0)

    bin_size = 0.2
    x_bins = max(1, int(np.ceil((x_range[1] - x_range[0]) / bin_size)))
    y_bins = max(1, int(np.ceil((y_range[1] - y_range[0]) / bin_size)))

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins], range=[x_range, y_range])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    vmax = float(np.max(heatmap)) if np.max(heatmap) > 0 else 1.0

    data_xy = np.column_stack([x, y])
    center, _, sorted_intensity, total_intensity = calculate_heatmap_center_and_percentage_boundary(
        data_xy, x_range, y_range, bins=(50, 50), top_percentage=0.1
    )
    concentration_index = calculate_concentration_index(sorted_intensity, total_intensity, top_percentage=0.1)

    title = f"Gaze Relative Position (Local) - {subject_name}"
    print(f"    热力图质心: ({center[0]:.2f}, {center[1]:.2f}), 前10%集中度指数: {concentration_index:.4f}")

    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap.T, extent=extent, origin="lower", cmap="gist_earth", vmin=0, vmax=vmax, aspect="auto")

    ax.scatter(center[0], center[1], color="red", label="Heatmap Center", zorder=5)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(title)
    ax.set_xlabel("Local X")
    ax.set_ylabel("Local Y")
    plt.colorbar(im, ax=ax, shrink=0.75)

    tx, ty = 0.02, 0.98
    ax.text(
        tx, ty,
        f"Mass Center: ({center[0]:.3f}, {center[1]:.3f})",
        transform=ax.transAxes,
        fontsize=11,
        color="black",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )
    ax.text(
        tx, ty - 0.08,
        f"Top 10% Concentration Index: {concentration_index:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        color="black",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )

    out_png = os.path.join(output_folder, f"{subject_name}_RelativePos_Heatmap.png")
    plt.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"    热力图已保存: {out_png}")


def process_subject(subject_path: str) -> dict:
    result = {
        "subject": os.path.basename(subject_path),
        "processed": 0,
        "skipped": 0,
        "valid_rows": 0,
    }

    idx_sub_x = excel_col_to_index("B")
    idx_sub_y = excel_col_to_index("C")
    idx_sub_z = excel_col_to_index("D")
    idx_sub_rot_y = excel_col_to_index("I")

    idx_gaze_x = excel_col_to_index("H")
    idx_gaze_y = excel_col_to_index("I")
    idx_gaze_z = excel_col_to_index("J")

    eye_files = glob.glob(os.path.join(subject_path, "*EyeGaze_Simulation_Info_*.csv"))
    subject_files = glob.glob(os.path.join(subject_path, "*Subject_Simulation_Info_*.csv"))

    if not eye_files:
        print("    [跳过] 未找到EyeGaze文件")
        result["skipped"] += 1
        return result
    if not subject_files:
        print("    [跳过] 未找到Subject文件")
        result["skipped"] += 1
        return result

    matched_pairs = []
    for eye_file in eye_files:
        base_eye = os.path.basename(eye_file)
        parts_eye = base_eye.split("_")
        if len(parts_eye) < 6:
            continue
        eye_timestamp = parts_eye[4] + "_" + parts_eye[5]

        for subject_file in subject_files:
            base_sub = os.path.basename(subject_file)
            parts_sub = base_sub.split("_")
            if len(parts_sub) < 6:
                continue
            subject_timestamp = parts_sub[4] + "_" + parts_sub[5]
            if eye_timestamp == subject_timestamp:
                matched_pairs.append((eye_file, subject_file))
                break

    if not matched_pairs:
        print("    [跳过] 未找到时间戳匹配的文件对")
        result["skipped"] += 1
        return result

    for eye_file, subject_file in matched_pairs:
        print(f"    处理: {os.path.basename(eye_file)}")

        try:
            eye_df = pd.read_csv(eye_file, index_col=False)
        except Exception as e:
            print(f"      [错误] 读取EyeGaze失败: {e}")
            continue

        try:
            subject_df = pd.read_csv(subject_file, index_col=False)
        except Exception as e:
            print(f"      [错误] 读取Subject失败: {e}")
            continue

        if "TimeStamp" not in eye_df.columns or "TimeStamp" not in subject_df.columns:
            print("      [跳过] 缺少TimeStamp列")
            continue

        n_eye_cols = eye_df.shape[1]
        n_sub_cols = subject_df.shape[1]
        need_eye = max(idx_gaze_x, idx_gaze_y, idx_gaze_z)
        need_sub = max(idx_sub_x, idx_sub_y, idx_sub_z, idx_sub_rot_y)

        if need_eye >= n_eye_cols or need_sub >= n_sub_cols:
            print(f"      [跳过] 列数不足")
            continue

        eye_df = eye_df.drop_duplicates(subset=["TimeStamp"], keep="first")
        subject_df = subject_df.drop_duplicates(subset=["TimeStamp"], keep="first")

        eye_mat = eye_df.to_numpy()
        sub_mat = subject_df.to_numpy()
        eye_cols = list(eye_df.columns)
        sub_cols = list(subject_df.columns)

        i_ts_eye = eye_cols.index("TimeStamp")
        i_ts_sub = sub_cols.index("TimeStamp")

        eye_by_ts = {}
        for r in range(len(eye_df)):
            ts = str(eye_mat[r, i_ts_eye])
            eye_by_ts[ts] = r

        rows_out = []
        skipped = 0
        for r in range(len(subject_df)):
            ts = str(sub_mat[r, i_ts_sub])
            if ts not in eye_by_ts:
                continue
            er = eye_by_ts[ts]

            sx = _parse_float(sub_mat[r, idx_sub_x])
            sy = _parse_float(sub_mat[r, idx_sub_y])
            sz = _parse_float(sub_mat[r, idx_sub_z])
            srot_y = _parse_float(sub_mat[r, idx_sub_rot_y])

            gx = _parse_float(eye_mat[er, idx_gaze_x])
            gy = _parse_float(eye_mat[er, idx_gaze_y])
            gz = _parse_float(eye_mat[er, idx_gaze_z])

            if any(v is None for v in (sx, sy, sz, srot_y, gx, gy, gz)):
                skipped += 1
                continue

            loc = get_local_position_2d(sx, sz, srot_y, gx, gz)
            rows_out.append({
                "TimeStamp": ts,
                "SubjectWorldX": sx,
                "SubjectWorldY": sy,
                "SubjectWorldZ": sz,
                "SubjectRotY_deg": srot_y,
                "GazeWorldX": gx,
                "GazeWorldY": gy,
                "GazeWorldZ": gz,
                "LocalPosX": float(loc[0]),
                "LocalPosY": float(loc[1]),
            })

        out_df = pd.DataFrame(rows_out, columns=OUTPUT_RELATIVE_COLUMNS)
        print(f"      有效行: {len(out_df)}, 跳过: {skipped}")

        if len(out_df) > 0:
            subject_name = os.path.basename(subject_path)
            scenario_name = os.path.basename(os.path.dirname(subject_path))
            output_subject_dir = os.path.join(RESULTS_BASE_DIR, scenario_name, subject_name)
            os.makedirs(output_subject_dir, exist_ok=True)

            output_path = os.path.join(output_subject_dir, "RelativePosInfo.csv")
            out_df.to_csv(output_path, index=False)
            print(f"      结果已保存: {output_path}")

            save_relative_pos_heatmap(output_subject_dir, subject_name, out_df)
            result["processed"] += 1
            result["valid_rows"] += len(out_df)
        else:
            result["skipped"] += 1

    return result


def main():
    print(f"\n{'=' * 70}")
    print(f"相对注视位置计算 - 批量处理所有场景")
    print(f"{'=' * 70}")
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    print(f"场景数量: {len(SCENARIO_FOLDERS)}")
    print(f"{'=' * 70}\n")

    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    total_scenarios = len(SCENARIO_FOLDERS)
    grand_total_processed = 0
    grand_total_valid_rows = 0

    for scenario_idx, scenario_folder in enumerate(SCENARIO_FOLDERS, 1):
        scenario_path = os.path.join(RAW_DATA_DIR, scenario_folder)

        if not os.path.exists(scenario_path):
            print(f"[{scenario_idx}/{total_scenarios}] [警告] 目录不存在: {scenario_folder}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{scenario_idx}/{total_scenarios}] 处理场景: {scenario_folder}")
        print(f"{'=' * 70}")

        subject_folders = sorted([
            f for f in os.listdir(scenario_path)
            if os.path.isdir(os.path.join(scenario_path, f)) and f != "Results"
        ])

        if not subject_folders:
            print(f"  [跳过] 该场景下没有受试者数据")
            continue

        print(f"找到 {len(subject_folders)} 个受试者\n")

        scenario_total_processed = 0
        scenario_total_valid_rows = 0
        scenario_all_data = []

        for subject_idx, subject_folder in enumerate(subject_folders, 1):
            subject_path = os.path.join(scenario_path, subject_folder)
            print(f"[{subject_idx}/{len(subject_folders)}] {subject_folder}")
            result = process_subject(subject_path)
            scenario_total_processed += result["processed"]
            scenario_total_valid_rows += result["valid_rows"]

            subject_output_dir = os.path.join(RESULTS_BASE_DIR, scenario_folder, subject_folder)
            csv_path = os.path.join(subject_output_dir, "RelativePosInfo.csv")
            if os.path.exists(csv_path):
                subject_df = pd.read_csv(csv_path)
                if not subject_df.empty and "LocalPosX" in subject_df.columns and "LocalPosY" in subject_df.columns:
                    scenario_all_data.append(subject_df)

        print(f"\n场景 {scenario_folder} 处理完成:")
        print(f"  成功处理: {scenario_total_processed} 个文件对")
        print(f"  有效数据行: {scenario_total_valid_rows}")

        if scenario_all_data:
            combined_df = pd.concat(scenario_all_data, ignore_index=True)
            scenario_output_dir = os.path.join(RESULTS_BASE_DIR, scenario_folder)
            os.makedirs(scenario_output_dir, exist_ok=True)

            combined_csv_path = os.path.join(scenario_output_dir, "AllSubjects_RelativePosInfo.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"  场景总体数据已保存: {combined_csv_path}")

            overall_name = f"整体-{scenario_folder}"
            save_relative_pos_heatmap(scenario_output_dir, overall_name, combined_df)

        grand_total_processed += scenario_total_processed
        grand_total_valid_rows += scenario_total_valid_rows

    print(f"\n{'=' * 70}")
    print(f"全部处理完成！")
    print(f"{'=' * 70}")
    print(f"处理场景数: {total_scenarios}")
    print(f"成功处理: {grand_total_processed} 个文件对")
    print(f"有效数据行: {grand_total_valid_rows}")
    print(f"结果已保存到: {RESULTS_BASE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
