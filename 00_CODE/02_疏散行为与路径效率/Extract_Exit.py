import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS, excel_col_to_index


RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_ExitChoice")

EXIT_POINTS = {
    "A": np.array([38.5000,-11.3200,-80.5180], dtype=float),
    "B": np.array([38.5000,-11.3200,-35.2800], dtype=float),
    "C": np.array([-38.3000,-11.3200,-80.5180], dtype=float),
    "D": np.array([-38.3000,-11.3200,-35.2820], dtype=float),
    "E": np.array([38.5000,-11.3200,11.5150], dtype=float),
    "F": np.array([38.4900,-11.3200,56.8680], dtype=float),
    "G": np.array([-38.3000,-11.3200,11.5150], dtype=float),
    "H": np.array([-38.3000,-11.3200,56.7000], dtype=float),
}

OUTPUT_COLUMNS = [
    "Scenario",
    "Subject",
    "FileName",
    "EndX",
    "EndY",
    "EndZ",
    "ChosenExit",
    "ChosenDistance",
    "DistanceToA",
    "DistanceToB",
    "DistanceToC",
    "DistanceToD",
    "DistanceToE",
    "DistanceToF",
    "DistanceToG",
    "DistanceToH",
    "TotalPoints",
]


def safe_float(value):
    """
    安全转换为 float。
    如果为空值、非数字或 NaN，则返回 None。
    """

    try:
        v = float(value)
    except (ValueError, TypeError):
        return None

    if np.isnan(v):
        return None

    return v


def calculate_3d_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    计算两个三维点之间的欧氏距离。
    """

    return float(np.linalg.norm(point_a - point_b))


def analyze_exit_choice(end_position: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    """
    根据受试者最终位置，判断其最接近哪个出口点。

    返回：
        chosen_exit: 最近出口编号，例如 A/B/C...
        chosen_distance: 到最近出口的距离
        distance_map: 到所有出口的距离
    """

    distance_map = {}

    for exit_name, exit_position in EXIT_POINTS.items():
        distance_map[exit_name] = calculate_3d_distance(
            end_position,
            exit_position
        )

    chosen_exit = min(distance_map, key=distance_map.get)
    chosen_distance = distance_map[chosen_exit]

    return chosen_exit, chosen_distance, distance_map


def process_subject(subject_path: str, scenario_name: str) -> List[Dict]:
    subject_name = os.path.basename(subject_path)
    results = []

    subject_files = sorted(
        glob.glob(os.path.join(subject_path, "*Subject_Simulation_Info_*.csv"))
    )

    if not subject_files:
        return results

    idx_x = excel_col_to_index("B")
    idx_y = excel_col_to_index("C")
    idx_z = excel_col_to_index("D")

    for file_idx, file_path in enumerate(subject_files, 1):
        try:
            df = pd.read_csv(file_path, index_col=False)
        except Exception as e:
            print(f"    [错误] 读取失败 {os.path.basename(file_path)}: {e}")
            continue

        if df.empty:
            print(f"    [跳过] 空文件: {os.path.basename(file_path)}")
            continue

        mat = df.to_numpy()
        n_cols = mat.shape[1]

        if idx_x >= n_cols or idx_y >= n_cols or idx_z >= n_cols:
            print(f"    [跳过] 列数不足: {os.path.basename(file_path)}")
            continue

        last_x = safe_float(mat[-1, idx_x])
        last_y = safe_float(mat[-1, idx_y])
        last_z = safe_float(mat[-1, idx_z])

        if last_x is None or last_y is None or last_z is None:
            print(f"    [跳过] 末行坐标解析失败: {os.path.basename(file_path)}")
            continue

        end_position = np.array([last_x, last_y, last_z], dtype=float)

        chosen_exit, chosen_distance, distance_map = analyze_exit_choice(
            end_position
        )

        if len(subject_files) == 1:
            label = subject_name
        else:
            label = f"{subject_name}_{file_idx}"

        results.append({
            "Scenario": scenario_name,
            "Subject": label,
            "FileName": os.path.basename(file_path),
            "EndX": last_x,
            "EndY": last_y,
            "EndZ": last_z,
            "ChosenExit": chosen_exit,
            "ChosenDistance": chosen_distance,
            "DistanceToA": distance_map["A"],
            "DistanceToB": distance_map["B"],
            "DistanceToC": distance_map["C"],
            "DistanceToD": distance_map["D"],
            "DistanceToE": distance_map["E"],
            "DistanceToF": distance_map["F"],
            "DistanceToG": distance_map["G"],
            "DistanceToH": distance_map["H"],
            "TotalPoints": len(mat),
        })

    return results


def save_scenario_csv(scenario_name: str, data: List[Dict]) -> None:
    if not data:
        return

    scenario_dir = os.path.join(RESULTS_BASE_DIR, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    df = pd.DataFrame(data, columns=OUTPUT_COLUMNS)

    out_path = os.path.join(scenario_dir, "ExitChoice.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"  场景结果已保存: {out_path}")


def main():
    print(f"\n{'=' * 70}")
    print("终点选择判定 - 批量处理所有场景")
    print(f"{'=' * 70}")
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    print(f"场景数量: {len(SCENARIO_FOLDERS)}")
    print(f"{'=' * 70}\n")

    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    total_scenarios = len(SCENARIO_FOLDERS)
    all_results = []

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
            print("  [跳过] 该场景下没有受试者数据")
            continue

        print(f"找到 {len(subject_folders)} 个受试者\n")

        scenario_data = []

        for subject_idx, subject_folder in enumerate(subject_folders, 1):
            subject_path = os.path.join(scenario_path, subject_folder)

            print(f"[{subject_idx}/{len(subject_folders)}] {subject_folder}")

            subject_results = process_subject(subject_path, scenario_folder)

            if not subject_results:
                print("    未找到 Subject_Simulation_Info 文件")
                continue

            for r in subject_results:
                print(
                    f"    {r['Subject']}: 选择终点 = {r['ChosenExit']} "
                    f"(距离 = {r['ChosenDistance']:.4f})"
                )

            scenario_data.extend(subject_results)
            all_results.extend(subject_results)

        save_scenario_csv(scenario_folder, scenario_data)

    if all_results:
        overall_path = os.path.join(
            RESULTS_BASE_DIR,
            "AllSubjects_ExitChoice.csv"
        )

        overall_df = pd.DataFrame(all_results, columns=OUTPUT_COLUMNS)
        overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")

        print(f"\n总体结果已保存: {overall_path}")

    print(f"\n{'=' * 70}")
    print("全部处理完成！")
    print(f"{'=' * 70}")
    print(f"处理场景数: {total_scenarios}")
    print(f"总受试者记录数: {len(all_results)}")
    print(f"结果已保存到: {RESULTS_BASE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()