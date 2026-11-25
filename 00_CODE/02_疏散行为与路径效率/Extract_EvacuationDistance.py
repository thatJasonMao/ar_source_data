import os
from typing import List, Dict

import numpy as np
import pandas as pd

from tqdm import tqdm

from common import PROJECT_ROOT, INDICATORS_DIR, load_trials, excel_col_to_index


RESULTS_BASE_DIR = os.path.join(INDICATORS_DIR, "Results_EvacuationDistance")

OUTPUT_COLUMNS = [
    "Scenario",
    "Subject",
    "FileName",
    "StartX",
    "StartZ",
    "EndX",
    "EndZ",
    "EvacuationDistance",
    "TotalPoints",
]


def calculate_xz_displacement(x1: float, z1: float, x2: float, z2: float) -> float:
    return float(np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2))


def calculate_trajectory_length(mat: np.ndarray, idx_x: int, idx_z: int) -> float:
    total = 0.0

    for i in range(1, len(mat)):
        try:
            x1 = float(mat[i - 1, idx_x])
            z1 = float(mat[i - 1, idx_z])
            x2 = float(mat[i, idx_x])
            z2 = float(mat[i, idx_z])
        except (ValueError, TypeError, IndexError):
            continue

        total += calculate_xz_displacement(x1, z1, x2, z2)

    return total


def process_trial(file_path: str, scenario_name: str, subject_name: str) -> Dict:
    """仅计算 common.load_trials() 清单指定的单个试次。"""
    idx_x = excel_col_to_index("B")
    idx_z = excel_col_to_index("D")
    try:
        df = pd.read_csv(file_path, index_col=False)
    except Exception as e:
        print(f"    [错误] 读取失败 {os.path.basename(file_path)}: {e}")
        raise RuntimeError(f"试次无法计算：{file_path}")

    if df.empty:
        print(f"    [错误] 空文件: {os.path.basename(file_path)}")
        raise RuntimeError(f"试次无法计算：{file_path}")

    mat = df.to_numpy()
    n_cols = mat.shape[1]

    if idx_x >= n_cols or idx_z >= n_cols:
        print(f"    [错误] 列数不足: {os.path.basename(file_path)}")
        raise RuntimeError(f"试次无法计算：{file_path}")

    try:
        first_x = float(mat[0, idx_x])
        first_z = float(mat[0, idx_z])
        last_x = float(mat[-1, idx_x])
        last_z = float(mat[-1, idx_z])
    except (ValueError, TypeError, IndexError) as e:
        print(f"    [错误] 起终点坐标无法转换为数字: {os.path.basename(file_path)}: {e}")
        raise RuntimeError(f"试次无法计算：{file_path}")

    distance = calculate_trajectory_length(mat, idx_x, idx_z)

    return {
        "Scenario": scenario_name,
        "Subject": subject_name,
        "FileName": os.path.basename(file_path),
        "StartX": first_x,
        "StartZ": first_z,
        "EndX": last_x,
        "EndZ": last_z,
        "EvacuationDistance": distance,
        "TotalPoints": len(mat),
    }



def save_scenario_csv(scenario_name: str, data: List[Dict]) -> None:
    if not data:
        return

    scenario_dir = os.path.join(RESULTS_BASE_DIR, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    df = pd.DataFrame(data, columns=OUTPUT_COLUMNS)

    out_path = os.path.join(scenario_dir, "EvacuationDistance.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"  场景结果已保存: {out_path}")


def main():
    trials = load_trials()
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"固定主队列: {trials.ParticipantID.nunique()} 人 / {len(trials)} 试次")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    all_results = []
    for trial in tqdm(trials.itertuples(index=False), total=len(trials),
                      desc="Evacuation Distance"):
        all_results.append(process_trial(
            trial.SubjectFile, trial.Scenario, trial.ParticipantID))

    overall_df = pd.DataFrame(all_results, columns=OUTPUT_COLUMNS)
    expected = set(trials[["Scenario", "ParticipantID", "FileName"]]
                   .itertuples(index=False, name=None))
    actual = set(overall_df[["Scenario", "Subject", "FileName"]]
                 .itertuples(index=False, name=None))
    if len(overall_df) != 270 or actual != expected:
        raise RuntimeError("计算结果与 common 的 270 试次清单不一致，停止写入")

    # 全部试次完成并核验后再写入，计算失败时不发布不完整结果。
    for scenario, group in overall_df.groupby("Scenario", sort=False):
        save_scenario_csv(scenario, group.to_dict("records"))
    overall_path = os.path.join(RESULTS_BASE_DIR, "AllSubjects_EvacuationDistance.csv")
    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    print(f"总体结果已保存: {overall_path}（{len(overall_df)} 试次）")


if __name__ == "__main__":
    main()
