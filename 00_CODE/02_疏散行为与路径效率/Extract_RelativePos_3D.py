import os
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS

# 结果输出根目录（与 ExtractRelativePosition.py 保持一致）
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_RelativePosition")

OUTPUT_RELATIVE_COLUMNS = [
    "TimeStamp",
    "SubjectWorldX",
    "SubjectWorldY",
    "SubjectWorldZ",
    "SubjectRotX_deg",
    "SubjectRotY_deg",
    "SubjectRotZ_deg",
    "GazeWorldX",
    "GazeWorldY",
    "GazeWorldZ",
    "LocalPosX",
    "LocalPosY",
    "LocalPosZ",
]


def excel_col_to_index(col: str) -> int:
    """Excel column label -> 0-based index (A->0, B->1, ..., Z->25)."""
    col = col.strip().upper()
    n = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column: {col!r}")
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def get_local_position_3d(sub_x, sub_y, sub_z, sub_rot_y, gaze_x, gaze_y, gaze_z):
    """
    与 DrawGazeOnAllPolar.get_local_position 一致：
    水平面取世界 X 与世界 Z，绕 Y 的 rotation（度）逆时针旋转到受试者局部坐标。
    这里在原有二维逻辑基础上扩展为三维：
    - LocalPosX / LocalPosY：保持原二维热力图平面的局部坐标（由世界 XZ 旋转得到）
    - LocalPosZ：垂直方向差值（世界 Y 差）
    """
    rotation_rad = np.radians(sub_rot_y)
    translated_x = gaze_x - sub_x
    translated_z = gaze_z - sub_z
    translated_y = gaze_y - sub_y
    rotation_matrix = np.array(
        [
            [np.cos(rotation_rad), -np.sin(rotation_rad)],
            [np.sin(rotation_rad), np.cos(rotation_rad)],
        ]
    )
    translated_vector = np.array([translated_x, translated_z])
    local_plane = np.dot(rotation_matrix, translated_vector)
    return np.array([float(local_plane[0]), float(local_plane[1]), float(translated_y)])


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


def _subject_second_horizontal_col_index(n_cols: int) -> int:
    """
    Subject「水平面第二轴」：用户指定为 Excel Z 列；若当前 CSV 列数不足，则回退为 PosZ（D 列，index 3），
    与 DrawGaze 中 sub_pos_z 用法一致。
    """
    z_idx = excel_col_to_index("Z")
    if z_idx < n_cols:
        return z_idx
    return excel_col_to_index("D")


def calculate_heatmap_center_and_percentage_boundary(data, x_range, y_range, bins=(50, 50), top_percentage=0.1):
    """
    与 DrawGazeOnAllPolar 思路一致：散点均值作质心；在固定 range 的 2D 直方图上取热力最高 top_percentage 的网格中心作为边界点集。
    """
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


def save_relative_pos_heatmap(folder_path: str, out_df: pd.DataFrame) -> None:
    """
    参照 DrawGazeOnAllPolar.plot_scatter：histogram2d + imshow(gist_earth) + 质心/集中度标注。
    显示范围固定为 Local X ∈ [-10, 10]，Local Y ∈ [0, 20]；保存到 folder_path/RelativePos_Heatmap.png
    """
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

    title = f"Gaze Relative Position (Local) — {os.path.basename(folder_path)}"
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
        tx,
        ty,
        f"Mass Center: ({center[0]:.3f}, {center[1]:.3f})",
        transform=ax.transAxes,
        fontsize=11,
        color="black",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )
    ax.text(
        tx,
        ty - 0.08,
        f"Top 10% Concentration Index: {concentration_index:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        color="black",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )

    out_png = os.path.join(folder_path, f"{os.path.basename(folder_path)}_RelativePos_Heatmap.png")
    plt.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"    热力图已保存: {out_png}")


def _build_relative_table(
    eye_df,
    subject_df,
    idx_sub_x,
    idx_sub_y,
    idx_sub_z,
    idx_sub_rot_x,
    idx_sub_rot_y,
    idx_sub_rot_z,
    idx_gaze_x,
    idx_gaze_y,
    idx_gaze_z,
):
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
        srot_x = _parse_float(sub_mat[r, idx_sub_rot_x])
        srot_y = _parse_float(sub_mat[r, idx_sub_rot_y])
        srot_z = _parse_float(sub_mat[r, idx_sub_rot_z])

        gx = _parse_float(eye_mat[er, idx_gaze_x])
        gy = _parse_float(eye_mat[er, idx_gaze_y])
        gz = _parse_float(eye_mat[er, idx_gaze_z])

        if any(v is None for v in (sx, sy, sz, srot_x, srot_y, srot_z, gx, gy, gz)):
            skipped += 1
            continue

        loc = get_local_position_3d(sx, sy, sz, srot_y, gx, gy, gz)
        rows_out.append(
            {
                "TimeStamp": ts,
                "SubjectWorldX": sx,
                "SubjectWorldY": sy,
                "SubjectWorldZ": sz,
                "SubjectRotX_deg": srot_x,
                "SubjectRotY_deg": srot_y,
                "SubjectRotZ_deg": srot_z,
                "GazeWorldX": gx,
                "GazeWorldY": gy,
                "GazeWorldZ": gz,
                "LocalPosX": float(loc[0]),
                "LocalPosY": float(loc[1]),
                "LocalPosZ": float(loc[2]),
            }
        )

    # 指定列顺序，确保即便无有效行也能输出完整三维表头，便于下游稳定读取。
    out_df = pd.DataFrame(rows_out, columns=OUTPUT_RELATIVE_COLUMNS)
    return out_df, skipped


def align_gaze_subject_data():
    """
    处理每个受试者数据文件夹：仅当 EyeGaze 与 Subject 文件名时间戳一致时才配对；
    行对齐仅保留 TimeStamp 字符串完全相同的行；
    在双方用于坐标/旋转的单元格均非空且可解析为 float 时，按 DrawGaze 规则计算注视点相对受试者局部坐标。
    输出 RelativePosInfo.csv 到 Results_RelativePosition/<场景>/<受试者>/。
    """
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    print(f"场景数量: {len(SCENARIO_FOLDERS)}")
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    idx_sub_x = excel_col_to_index("B")
    idx_sub_y = excel_col_to_index("C")
    idx_sub_z = excel_col_to_index("D")
    idx_sub_rot_x = excel_col_to_index("H")
    idx_sub_rot_y = excel_col_to_index("I")
    idx_sub_rot_z = excel_col_to_index("J")

    idx_gaze_x = excel_col_to_index("H")
    idx_gaze_y = excel_col_to_index("I")
    idx_gaze_z = excel_col_to_index("J")

    for scenario_folder in SCENARIO_FOLDERS:
        scenario_path = os.path.join(RAW_DATA_DIR, scenario_folder)
        if not os.path.isdir(scenario_path):
            print(f"[警告] 场景目录不存在，跳过: {scenario_folder}")
            continue

        subject_folders = sorted([
            f
            for f in os.listdir(scenario_path)
            if os.path.isdir(os.path.join(scenario_path, f))
            and f != "Results"
            and not f.startswith(".")
        ])
        print(f"\n[{scenario_folder}] 找到 {len(subject_folders)} 个受试者文件夹")

        for folder in subject_folders:
            folder_path = os.path.join(scenario_path, folder)
            print(f"\n正在处理受试者: {scenario_folder}/{folder}")

            eye_files = glob.glob(
                os.path.join(folder_path, "*EyeGaze_Simulation_Info_*.csv")
            )
            subject_files = glob.glob(
                os.path.join(folder_path, "*Subject_Simulation_Info_*.csv")
            )

            if not eye_files:
                print("  未找到eyegaze文件，跳过此文件夹")
                continue
            if not subject_files:
                print("  未找到subject文件，跳过此文件夹")
                continue

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
                print("  未找到时间戳匹配的eyegaze和subject文件，跳过此文件夹")
                continue

            for eye_file, subject_file in matched_pairs:
                print(f"  处理文件对: {os.path.basename(eye_file)} 和 {os.path.basename(subject_file)}")

                try:
                    # EyeGaze 行末常有多余逗号，字段数多于表头；必须关闭首列作索引的推断，否则 TimeStamp 会整列错位。
                    eye_df = pd.read_csv(eye_file, index_col=False)
                except Exception as e:
                    print(f"    读取eyegaze文件失败: {e}")
                    continue
                try:
                    subject_df = pd.read_csv(subject_file, index_col=False)
                except Exception as e:
                    print(f"    读取subject文件失败: {e}")
                    continue

                if "TimeStamp" not in eye_df.columns or "TimeStamp" not in subject_df.columns:
                    print("    缺少 TimeStamp 列，跳过")
                    continue

                n_eye_cols = eye_df.shape[1]
                n_sub_cols = subject_df.shape[1]
                need_eye = max(idx_gaze_x, idx_gaze_y, idx_gaze_z)
                need_sub = max(
                    idx_sub_x,
                    idx_sub_y,
                    idx_sub_z,
                    idx_sub_rot_x,
                    idx_sub_rot_y,
                    idx_sub_rot_z,
                )
                if need_eye >= n_eye_cols or need_sub >= n_sub_cols:
                    print(
                        f"    列数不足（Eyegaze 需要 > {need_eye}, Subject 需要 > {need_sub}），跳过"
                    )
                    continue

                eye_df = eye_df.drop_duplicates(subset=["TimeStamp"], keep="first")
                subject_df = subject_df.drop_duplicates(subset=["TimeStamp"], keep="first")

                out_df, skipped = _build_relative_table(
                    eye_df,
                    subject_df,
                    idx_sub_x,
                    idx_sub_y,
                    idx_sub_z,
                    idx_sub_rot_x,
                    idx_sub_rot_y,
                    idx_sub_rot_z,
                    idx_gaze_x,
                    idx_gaze_y,
                    idx_gaze_z,
                )

                print(f"    有效相对位置行数: {len(out_df)}，因空/非数值跳过: {skipped}")

                # 与 ExtractRelativePosition.py 对齐：把结果输出到 Results_RelativePosition/<场景>/<受试者>/
                output_subject_dir = os.path.join(
                    RESULTS_BASE_DIR, scenario_folder, folder
                )
                os.makedirs(output_subject_dir, exist_ok=True)

                output_path = os.path.join(output_subject_dir, "RelativePosInfo.csv")
                out_df.to_csv(output_path, index=False)
                print(f"    结果已保存到: {output_path}")
                save_relative_pos_heatmap(output_subject_dir, out_df)

    print("\n所有受试者数据处理完成！")
    verify_alignment()


def verify_alignment():
    print("\n开始验证输出...")

    if not os.path.isdir(RESULTS_BASE_DIR):
        print(f"未找到输出根目录: {RESULTS_BASE_DIR}")
        return

    # 与 ExtractRelativePosition.py 对齐：只搜索 Results_RelativePosition/<场景>/<受试者>/RelativePosInfo.csv
    result_files = glob.glob(
        os.path.join(RESULTS_BASE_DIR, "*", "*", "RelativePosInfo.csv")
    )
    if not result_files:
        print("未找到任何RelativePosInfo.csv文件")
        return

    required = [
        "TimeStamp",
        "SubjectWorldX",
        "SubjectWorldY",
        "SubjectWorldZ",
        "SubjectRotX_deg",
        "SubjectRotY_deg",
        "SubjectRotZ_deg",
        "GazeWorldX",
        "GazeWorldY",
        "GazeWorldZ",
        "LocalPosX",
        "LocalPosY",
        "LocalPosZ",
    ]

    for result_file in result_files[:3]:
        print(f"\n验证文件: {result_file}")
        try:
            df = pd.read_csv(result_file, index_col=False)
            if len(df) == 0:
                print("  文件为空")
                continue
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  缺少列: {missing}")
                continue
            row = df.iloc[0]
            sx, sy, sz = row["SubjectWorldX"], row["SubjectWorldY"], row["SubjectWorldZ"]
            srot_y = row["SubjectRotY_deg"]
            gx, gy, gz = row["GazeWorldX"], row["GazeWorldY"], row["GazeWorldZ"]
            loc = get_local_position_3d(sx, sy, sz, srot_y, gx, gy, gz)
            if (
                abs(loc[0] - row["LocalPosX"]) > 1e-4
                or abs(loc[1] - row["LocalPosY"]) > 1e-4
                or abs(loc[2] - row["LocalPosZ"]) > 1e-4
            ):
                print("  警告: 首行 LocalPos 与重算不一致")
            else:
                print("  首行 LocalPos 与 get_local_position 一致")
        except Exception as e:
            print(f"  验证出错: {e}")

    print("\n验证完成！")


if __name__ == "__main__":
    align_gaze_subject_data()
