"""
Experiment_Data_A5 公共基础模块

集中管理：
- 项目根目录路径索引
- 场景目录名称列表
- 受试者分组识别与颜色映射
- Excel 列字母到数字索引转换
- 全量检索 → 全量剔除：试次队列（45 名受试者 / 270 个试次）的唯一权威口径
"""

import os

# ---------------------------------------------------------------------------
# 路径索引
# ---------------------------------------------------------------------------

# 工程根目录（common.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 原始数据根目录：10 个实验场景目录（见 SCENARIO_FOLDERS）均位于该目录下，
# 即 <工程根>/RAW_DATA/<场景目录>/<受试者目录>/*.csv
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "RAW_DATA")

# 基础指标结果根目录：疏散距离/速度/时间/转角、MLE 等基础指标输出统一存放于
# <工程根>/INDICATORS/Results_* 下（与工程根下的其他 Results_* 目录分开管理）
INDICATORS_DIR = os.path.join(PROJECT_ROOT, "INDICATORS")

# 注视拓扑分析结果根目录：注视模式聚类、平面/球面注视热力图等注视分析结果
# 统一存放于 <工程根>/GAZE_TOPO/Results_* 下
GAZE_TOPO_DIR = os.path.join(PROJECT_ROOT, "GAZE_TOPO")

# 10 个实验场景目录（5 种辅助方式 × 2 种烟雾条件）
SCENARIO_FOLDERS = [
    "StaticSign-HighSmoke",
    "StaticSign-LowSmoke",
    "Full-HighSmoke",
    "Full-LowSmoke",
    "ArrowRibbon-HighSmoke",
    "ArrowRibbon-LowSmoke",
    "MiniMap-HighSmoke",
    "MiniMap-LowSmoke",
    "TextBar-HighSmoke",
    "TextBar-LowSmoke",
]


# 引导模态 × 烟雾的统一配色（与 graph_0911/gaze_depth/gaze_depth_0915_2.py 一致）。
# 与 A/B/C 受试者分组颜色 GROUP_COLORS 含义不同，绘制实验条件时使用此表。
GUIDANCE_SMOKE_COLORS = {'StaticSign': {'LowSmoke': '#E6DAF5', 'HighSmoke': '#6B5C89'}, 'ArrowRibbon': {'LowSmoke': '#83E5FC', 'HighSmoke': '#398BD1'}, 'MiniMap': {'LowSmoke': '#BDE6EA', 'HighSmoke': '#357887'}, 'TextBar': {'LowSmoke': '#FDD8C7', 'HighSmoke': '#E99E74'}, 'Full': {'LowSmoke': '#FFC8C8', 'HighSmoke': '#CF5149'}}
SCENARIO_COLORS = {
    f"{guidance}-{smoke}": color
    for guidance, smokes in GUIDANCE_SMOKE_COLORS.items()
    for smoke, color in smokes.items()
}


def get_scenario_color(scenario):
    """按正式场景名返回颜色；未知条件报错，避免静默用错配色。"""
    try:
        return SCENARIO_COLORS[scenario]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario color: {scenario!r}") from exc


# ---------------------------------------------------------------------------
# 分组索引
# ---------------------------------------------------------------------------

# 组别颜色映射（A 组 → ArrowRibbon, B 组 → MiniMap, C 组 → TextBar）
GROUP_COLORS = {
    "A": "aqua",
    "B": "deeppink",
    "C": "yellow",
}

# 组别可读标签
GROUP_LABELS = {
    "A": "ArrowRibbon",
    "B": "MiniMap",
    "C": "TextBar",
}

# 所有受试者都参与的混合场景（A/B/C 三组受试者都包含数据）
MIXED_SCENARIO_PREFIXES = {"StaticSign", "Full"}


def get_subject_group(file_path):
    """从文件路径中提取受试者组别（A / B / C），若无法识别则返回 None。"""
    subject_name = os.path.basename(os.path.dirname(file_path))
    for g in ("A", "B", "C"):
        if ("_" + g) in subject_name:
            return g
    return None


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------

def excel_col_to_index(col):
    """将 Excel 列字母（如 'B', 'K', 'AA'）转换为 0-based 数字索引。"""
    col = col.strip().upper()
    n = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column: {col!r}")
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


# ---------------------------------------------------------------------------
# 唯一试次入口：全量检索 → 完整黑名单排除 → 固定 45 人 / 270 试次
# 调用：from common import load_trials; trials = load_trials()
# 无队列选择、宽松模式、外部会话注入或命令行入口。
# 未登记的重复、缺失或身份变更一律报错，不自动选取“最新一次”。
# ---------------------------------------------------------------------------

# 剔除理由（字符串即落盘口径，勿随意改动，历史产物按此对齐）
_EXCLUSION_REASON_TEST = "technical_test_folder"
_EXCLUSION_REASON_SUPERSEDED = "superseded_by_later_rerun_same_condition"
_EXCLUSION_REASON_NO_EYE = "no_eye_tracking_data_aoi_unavailable"

# 第 2 层：整人剔除名单。这三名受试者的全部会话整段无有效眼动数据
# （ValidEyeSampleN = 0 / ValidDurationCoverage = 0，采集于 2026-04-26 当晚），
# 因此无法计算任何依赖 AOI 的指标。证据见
# Results_Guidance_Spatial_Causal_Analysis/tables/eye_tracking_data_quality_audit.csv
# 与 tables/excluded_aoi_unavailable_participants_and_trials.csv
_EXCLUDED_PARTICIPANTS_NO_EYE = ("LAT_A2", "LXC_C1", "MC_B2")

# 全量剔除登记表（试次级，明码逐条列出）：(场景, 受试者, FileName, 理由)
_EXCLUDED_SESSIONS = (
    # ① 第 1 层 · TEST 技术测试目录（3 条；设计性剔除，与数据质量无关）
    ("StaticSign-HighSmoke", "TEST", "A5-1_Subject_Simulation_Info_2026_04_28_12_31_50.csv", _EXCLUSION_REASON_TEST),
    ("StaticSign-LowSmoke", "TEST", "A5-2_Subject_Simulation_Info_2026_04_26_21_53_34.csv", _EXCLUSION_REASON_TEST),
    ("StaticSign-LowSmoke", "TEST", "A5-2_Subject_Simulation_Info_2026_04_27_15_35_34.csv", _EXCLUSION_REASON_TEST),
    # ② 第 1 层 · 同场景同受试者重复采集，保留后跑的版本（1 条）
    ("ArrowRibbon-HighSmoke", "YZN_A1", "A5-2_Subject_Simulation_Info_2026_04_26_15_23_10.csv", _EXCLUSION_REASON_SUPERSEDED),
    # ③ 第 2 层 · 整段无眼动（18 条 = LAT_A2 / LXC_C1 / MC_B2 × 6 场景）
    ("StaticSign-HighSmoke", "LAT_A2", "A5-1_Subject_Simulation_Info_2026_04_26_20_20_13.csv", _EXCLUSION_REASON_NO_EYE),
    ("StaticSign-LowSmoke", "LAT_A2", "A5-2_Subject_Simulation_Info_2026_04_26_20_23_50.csv", _EXCLUSION_REASON_NO_EYE),
    ("Full-HighSmoke", "LAT_A2", "A5-5_Subject_Simulation_Info_2026_04_26_20_34_06.csv", _EXCLUSION_REASON_NO_EYE),
    ("Full-LowSmoke", "LAT_A2", "A5-6_Subject_Simulation_Info_2026_04_26_20_39_09.csv", _EXCLUSION_REASON_NO_EYE),
    ("ArrowRibbon-HighSmoke", "LAT_A2", "A5-3_Subject_Simulation_Info_2026_04_26_20_27_37.csv", _EXCLUSION_REASON_NO_EYE),
    ("ArrowRibbon-LowSmoke", "LAT_A2", "A5-4_Subject_Simulation_Info_2026_04_26_20_30_56.csv", _EXCLUSION_REASON_NO_EYE),
    ("StaticSign-HighSmoke", "LXC_C1", "A5-1_Subject_Simulation_Info_2026_04_26_19_48_44.csv", _EXCLUSION_REASON_NO_EYE),
    ("StaticSign-LowSmoke", "LXC_C1", "A5-2_Subject_Simulation_Info_2026_04_26_19_51_49.csv", _EXCLUSION_REASON_NO_EYE),
    ("Full-HighSmoke", "LXC_C1", "A5-5_Subject_Simulation_Info_2026_04_26_20_01_58.csv", _EXCLUSION_REASON_NO_EYE),
    ("Full-LowSmoke", "LXC_C1", "A5-6_Subject_Simulation_Info_2026_04_26_20_04_46.csv", _EXCLUSION_REASON_NO_EYE),
    ("TextBar-HighSmoke", "LXC_C1", "A5-3_Subject_Simulation_Info_2026_04_26_19_55_24.csv", _EXCLUSION_REASON_NO_EYE),
    ("TextBar-LowSmoke", "LXC_C1", "A5-4_Subject_Simulation_Info_2026_04_26_19_58_37.csv", _EXCLUSION_REASON_NO_EYE),
    ("StaticSign-HighSmoke", "MC_B2", "A5-1_Subject_Simulation_Info_2026_04_26_20_46_52.csv", _EXCLUSION_REASON_NO_EYE),
    ("StaticSign-LowSmoke", "MC_B2", "A5-2_Subject_Simulation_Info_2026_04_26_20_50_24.csv", _EXCLUSION_REASON_NO_EYE),
    ("Full-HighSmoke", "MC_B2", "A5-5_Subject_Simulation_Info_2026_04_26_21_02_02.csv", _EXCLUSION_REASON_NO_EYE),
    ("Full-LowSmoke", "MC_B2", "A5-6_Subject_Simulation_Info_2026_04_26_21_04_43.csv", _EXCLUSION_REASON_NO_EYE),
    ("MiniMap-HighSmoke", "MC_B2", "A5-3_Subject_Simulation_Info_2026_04_26_20_55_39.csv", _EXCLUSION_REASON_NO_EYE),
    ("MiniMap-LowSmoke", "MC_B2", "A5-4_Subject_Simulation_Info_2026_04_26_20_58_58.csv", _EXCLUSION_REASON_NO_EYE),
)

# 锁定已核实的 270 个 (Scenario, ParticipantID, FileName)，防止等量替换
# 绕过人数/条数检查。这里只存身份摘要，不读取其他分析脚本的产物。
_PRIMARY_KEYS_SHA256 = "5cf0be00021e36a4d5a4c1159e53bcaad1ad1aee2c5dfe5d08ec14aa909b85c2"


def load_trials():
    """从 RAW_DATA 全量检索并排除黑名单，仅返回完整的 45 人 / 270 试次。

    每行一个试次：Scenario / Guidance / Smoke / ParticipantID /
    SubjectFolder / Group / FileName / SubjectFile / EyeGazeFile。
    所有检查在返回前执行；任何异常都抛错，不返回部分结果。
    眼动整人排除依据既有质量审计，不在基础组件内重算眼动指标。
    """
    import hashlib
    import json
    import re
    import pandas as pd

    rows = []
    for scenario in SCENARIO_FOLDERS:
        scenario_dir = os.path.join(RAW_DATA_DIR, scenario)
        if not os.path.isdir(scenario_dir):
            raise FileNotFoundError(f"场景目录不存在：{scenario_dir}")
        guidance, smoke = scenario.rsplit("-", 1)
        for participant in sorted(os.listdir(scenario_dir)):
            folder = os.path.join(scenario_dir, participant)
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                if "_Subject_" not in filename or not filename.lower().endswith(".csv"):
                    continue
                subject_file = os.path.join(folder, filename)
                if not os.path.isfile(subject_file):
                    raise RuntimeError(f"Subject 记录不是文件：{subject_file}")
                rows.append({
                    "Scenario": scenario, "Guidance": guidance, "Smoke": smoke,
                    "ParticipantID": participant, "SubjectFolder": participant,
                    "Group": get_subject_group(subject_file), "FileName": filename,
                    "SubjectFile": subject_file,
                    "EyeGazeFile": os.path.join(folder, filename.replace("_Subject_", "_EyeGaze_")),
                })
    if not rows:
        raise RuntimeError(f"未检索到 Subject 会话：{RAW_DATA_DIR}")
    frame = pd.DataFrame(rows)
    columns = ["Scenario", "ParticipantID", "FileName"]
    keys = list(frame[columns].itertuples(index=False, name=None))
    blacklist = {entry[:3] for entry in _EXCLUDED_SESSIONS}
    if len(blacklist) != 22 or len(_EXCLUDED_SESSIONS) != 22:
        raise RuntimeError("完整黑名单必须包含 22 个唯一试次")
    missing = blacklist - set(keys)
    if missing:
        raise RuntimeError(f"黑名单会话缺失或已改名：{sorted(missing)}")
    result = frame.loc[[key not in blacklist for key in keys]].copy()
    if len(result) != 270 or result.ParticipantID.nunique() != 45:
        raise RuntimeError(
            f"黑名单排除后必须为 45 人 / 270 试次，实际为 "
            f"{result.ParticipantID.nunique()} 人 / {len(result)} 试次")
    if result.duplicated(["Scenario", "ParticipantID"]).any():
        raise RuntimeError("存在黑名单未登记的重复会话，不能自动选择版本")
    for participant, trials in result.groupby("ParticipantID"):
        match = re.fullmatch(r".+_([ABC])([1-9]|1[0-6])", participant)
        if match is None or participant in _EXCLUDED_PARTICIPANTS_NO_EYE:
            raise RuntimeError(f"非法或禁止纳入的受试者：{participant}")
        expected = {f"{mode}-{smoke}" for mode in
                    ("StaticSign", "Full", GROUP_LABELS[match.group(1)])
                    for smoke in ("HighSmoke", "LowSmoke")}
        if len(trials) != 6 or set(trials.Scenario) != expected:
            raise RuntimeError(f"受试者六场景不完整或组别错配：{participant}")
    actual_keys = sorted(result[columns].itertuples(index=False, name=None))
    digest = hashlib.sha256(json.dumps(actual_keys, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != _PRIMARY_KEYS_SHA256:
        raise RuntimeError("试次身份与已确认的 45 人 / 270 试次不一致，请核对原始目录和文件名")
    for eye_file in result.EyeGazeFile:
        if not os.path.isfile(eye_file):
            raise FileNotFoundError(f"主队列缺少对应 EyeGaze 文件：{eye_file}")
    return result.sort_values(columns).reset_index(drop=True)
