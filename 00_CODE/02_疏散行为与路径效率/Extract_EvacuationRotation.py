import os
from typing import List, Dict

import numpy as np
import pandas as pd

from tqdm import tqdm

from common import PROJECT_ROOT, INDICATORS_DIR, load_trials, excel_col_to_index


RESULTS_BASE_DIR = os.path.join(INDICATORS_DIR, "Results_EvacuationRotation")

OUTPUT_COLUMNS = [
    "Scenario",
    "Subject",
    "FileName",
    "StartRotY_deg",
    "EndRotY_deg",
    "TotalRotation_deg",
    "TotalPoints",
]


def safe_float(value):
    """
    安全转换为 float。
    转换失败或为空值时返回 None。
    """

    try:
        v = float(value)
    except (ValueError, TypeError):
        return None

    if np.isnan(v):
        return None

    return v


def angular_difference_deg(angle1, angle2):
    """
    计算两个角度之间的最小夹角差，单位为度。

    例如：
        359 -> 1，真实转角为 2 度，而不是 358 度。
        -179 -> 179，真实转角为 2 度，而不是 358 度。

    返回值范围为 0 到 180。
    """

    diff = (angle2 - angle1 + 180) % 360 - 180
    return abs(diff)


def calculate_total_rotation(rot_values):
    """
    根据旋转角序列计算累计转角。
    使用最小角度差，避免 0/360 或 -180/180 跨界造成异常大值。
    """

    total_rotation = 0.0
    valid_count = 0

    for i in range(1, len(rot_values)):
        prev_rot = rot_values[i - 1]
        curr_rot = rot_values[i]

        if prev_rot is None or curr_rot is None:
            continue

        total_rotation += angular_difference_deg(prev_rot, curr_rot)
        valid_count += 1

    return total_rotation, valid_count


def process_trial(file_path: str, scenario_name: str, subject_name: str) -> Dict:
    """仅计算 common.load_trials() 清单指定的单个试次。"""
    idx_rot = excel_col_to_index("I")
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

    if idx_rot >= n_cols:
        print(f"    [错误] 列数不足: {os.path.basename(file_path)}")
        raise RuntimeError(f"试次无法计算：{file_path}")

    rot_values = []

    for r in range(len(mat)):
        rot = safe_float(mat[r, idx_rot])
        rot_values.append(rot)

    valid_rot_values = [v for v in rot_values if v is not None]

    if len(valid_rot_values) < 2:
        print(f"    [错误] 有效旋转数据不足: {os.path.basename(file_path)}")
        raise RuntimeError(f"试次无法计算：{file_path}")

    first_rot = valid_rot_values[0]
    last_rot = valid_rot_values[-1]

    total_rotation, valid_count = calculate_total_rotation(rot_values)

    return {
        "Scenario": scenario_name,
        "Subject": subject_name,
        "FileName": os.path.basename(file_path),
        "StartRotY_deg": first_rot,
        "EndRotY_deg": last_rot,
        "TotalRotation_deg": total_rotation,
        "TotalPoints": valid_count + 1,
    }



def save_scenario_csv(scenario_name: str, data: List[Dict]) -> None:
    if not data:
        return

    scenario_dir = os.path.join(RESULTS_BASE_DIR, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    df = pd.DataFrame(data, columns=OUTPUT_COLUMNS)

    out_path = os.path.join(scenario_dir, "EvacuationRotation.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"  场景结果已保存: {out_path}")


def main():
    trials = load_trials()
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"固定主队列: {trials.ParticipantID.nunique()} 人 / {len(trials)} 试次")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    all_results = []
    for trial in tqdm(trials.itertuples(index=False), total=len(trials),
                      desc="Evacuation Rotation"):
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
    overall_path = os.path.join(RESULTS_BASE_DIR, "AllSubjects_EvacuationRotation.csv")
    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    print(f"总体结果已保存: {overall_path}（{len(overall_df)} 试次）")


if __name__ == "__main__":
    main()
