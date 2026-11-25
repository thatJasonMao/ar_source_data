"""
Experiment_Data_A5 — MLE 显著性全面对比

对 MLE 数据进行系统性显著性检验，覆盖 6 大模块：
  模块1: 烟雾效应 (High vs Low Smoke)
  模块2: 场景推进效应 (Scene Progression)
  模块3: 组间比较 (Group A vs B vs C)
  模块4: 引导跟随行为 (Guidance Following)
  模块5: 出口选择模式 (Exit Choice Pattern)
  模块6: 补充与稳健性分析

数据源:
  - Results_ExitChoice/AllSubjects_ExitChoice.csv
  - INDICATORS/Results_LyapunovMLE/个人MLE统计表.csv
"""

import os
import re
import json
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

from common import PROJECT_ROOT, INDICATORS_DIR

warnings.filterwarnings("ignore")

# ── 路径常量 ──
EXIT_CHOICE_CSV = os.path.join(PROJECT_ROOT, "Results_ExitChoice", "AllSubjects_ExitChoice.csv")
MLE_CSV        = os.path.join(INDICATORS_DIR, "Results_LyapunovMLE", "个人MLE统计表.csv")
OUTPUT_JSON    = os.path.join(INDICATORS_DIR, "Results_LyapunovMLE", "comprehensive_significance_results.json")

# 引导出口映射 (文件名场景号 → 引导出口)
GUIDED_EXIT_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}

# 场景目录列表
SCENARIO_FOLDERS = [
    "StaticSign-HighSmoke", "StaticSign-LowSmoke",
    "ArrowRibbon-HighSmoke", "ArrowRibbon-LowSmoke",
    "MiniMap-HighSmoke", "MiniMap-LowSmoke",
    "TextBar-HighSmoke", "TextBar-LowSmoke",
    "Full-HighSmoke", "Full-LowSmoke",
]


# ═══════════════════════════════════════════════════════════════
# 统计工具函数
# ═══════════════════════════════════════════════════════════════

def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def mannwhitney_u(a, b):
    """
    Mann-Whitney U 检验（含 tie correction + rank-biserial correlation 效应量）。

    返回 (U, z, p, r_rb)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, np.nan

    combined = np.concatenate([a, b])
    ranks = rankdata(combined)

    R1 = np.sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # tie correction
    _, counts = np.unique(ranks, return_counts=True)
    tie_correction = 0.0
    for c in counts:
        tie_correction += (c ** 3 - c)
    N = n1 + n2
    sigma = np.sqrt(n1 * n2 * (N + 1 - tie_correction / (N * (N - 1))) / 12.0) if N > 1 else 1.0

    if sigma < 1e-12:
        z = 0.0
        p = 1.0
    else:
        # continuity correction
        z = (abs(U - n1 * n2 / 2.0) - 0.5) / sigma
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    r_rb = 1.0 - 2.0 * U / (n1 * n2) if n1 * n2 > 0 else 0.0
    return U, z, p, r_rb


def wilcoxon_paired(a, b):
    """
    Wilcoxon 符号秩检验（剔除零差值）。

    返回 (W, z, p)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    d = a - b
    d = d[np.abs(d) > 1e-12]  # 剔除零差值
    n = len(d)
    if n < 3:
        return np.nan, np.nan, np.nan
    try:
        W, p = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
        return W, W, p
    except Exception:
        return np.nan, np.nan, np.nan


# ═══════════════════════════════════════════════════════════════
# 数据加载与预处理
# ═══════════════════════════════════════════════════════════════

def load_and_merge():
    """
    读取 ExitChoice 和 MLE 数据，清洗并合并为统一结构。
    返回: (df_merged, subject_info_dict)
      subject_info_dict = {(scenario_folder, subject_clean): {mle, chosen, scene_num, guided, group}}
    """
    print("=" * 70)
    print("MLE 显著性全面对比")
    print("=" * 70)

    # ── 按固定列读取 ExitChoice ──
    df_exit = pd.read_csv(EXIT_CHOICE_CSV, usecols=["Scenario", "Subject", "FileName", "ChosenExit"],
                          encoding="utf-8-sig")
    df_exit = df_exit.dropna(subset=["Scenario", "Subject", "ChosenExit"])

    # 提取场景号
    df_exit["scene_num"] = df_exit["FileName"].apply(
        lambda x: int(m.group(1)) if isinstance(x, str) and (m := re.search(r"A5-(\d)", x)) else None
    )
    df_exit = df_exit.dropna(subset=["scene_num"])
    df_exit["scene_num"] = df_exit["scene_num"].astype(int)

    # Subject 清洗
    df_exit["Subject_clean"] = df_exit["Subject"].apply(lambda x: re.sub(r"_\d+$", "", str(x)))

    # 排除 TEST
    df_exit = df_exit[~df_exit["Subject_clean"].str.upper().str.startswith("TEST")]

    # 提取组别
    df_exit["group"] = df_exit["Subject_clean"].apply(
        lambda x: m.group(1) if isinstance(x, str) and (m := re.search(r"_([ABC])\d", x)) else None
    )

    # ── 按固定列读取 MLE ──
    df_mle = pd.read_csv(MLE_CSV, usecols=["场景", "受试者", "MLE"], encoding="utf-8-sig")
    df_mle = df_mle.dropna(subset=["场景", "受试者", "MLE"])
    df_mle.rename(columns={"场景": "Scenario", "受试者": "Subject"}, inplace=True)
    df_mle["Subject_clean"] = df_mle["Subject"].apply(lambda x: re.sub(r"_\d+$", "", str(x)))
    df_mle = df_mle[~df_mle["Subject_clean"].str.upper().str.startswith("TEST")]

    # ── 合并 ──
    df_merged = pd.merge(
        df_exit, df_mle,
        on=["Scenario", "Subject", "Subject_clean"],
        how="inner"
    )
    print(f"合并后有效记录: {len(df_merged)} 条")
    print(f"涉及受试者: {df_merged['Subject_clean'].nunique()} 人")
    print(f"涉及场景: {df_merged['Scenario'].nunique()} 个")

    # ── 构建 subject_info 索引 ──
    subject_info = {}
    for _, row in df_merged.iterrows():
        key = (row["Scenario"], row["Subject_clean"])
        subject_info[key] = {
            "mle": row["MLE"],
            "chosen": row["ChosenExit"],
            "scene_num": int(row["scene_num"]),
            "guided": GUIDED_EXIT_MAP.get(int(row["scene_num"]), ""),
            "group": row["group"],
        }

    # 构建 mle_by_subj_scene
    mle_by_subj_scene = defaultdict(dict)
    for _, row in df_merged.iterrows():
        mle_by_subj_scene[row["Subject_clean"]][int(row["scene_num"])] = row["MLE"]

    # 构建 chosen_by_subj_scene
    chosen_by_subj_scene = defaultdict(dict)
    for _, row in df_merged.iterrows():
        chosen_by_subj_scene[row["Subject_clean"]][int(row["scene_num"])] = row["ChosenExit"]

    # 构建 group_of_subj
    group_of_subj = {}
    for _, row in df_merged.iterrows():
        group_of_subj[row["Subject_clean"]] = row["group"]

    return df_merged, subject_info, mle_by_subj_scene, chosen_by_subj_scene, group_of_subj


# ═══════════════════════════════════════════════════════════════
# 测试运行器
# ═══════════════════════════════════════════════════════════════

class ResultCollector:
    def __init__(self):
        self.results = []
    def add(self, title, method, n1, n2, stat_val, p, effect="", sig=""):
        if not sig:
            sig = _sig(p) if not np.isnan(p) else ""
        self.results.append({
            "title": title, "method": method,
            "n1": n1, "n2": n2,
            "statistic": float(stat_val) if not np.isnan(stat_val) else None,
            "p": float(p) if not np.isnan(p) else None,
            "effect": effect, "sig": sig,
        })
    def add_mwu(self, title, a, b, tag=""):
        U, z, p, r_rb = mannwhitney_u(a, b)
        effect = f"r_rb={r_rb:.3f}" if not np.isnan(r_rb) else ""
        title_full = f"[{tag}] {title}" if tag else title
        self.add(title_full, "Mann-Whitney U", len(a), len(b), U, p, effect)
        print(f"  {title_full}: n1={len(a)} n2={len(b)} U={U:.1f} p={p:.4f} {effect} {_sig(p)}")
    def add_wilcoxon(self, title, a, b, tag=""):
        W, _, p = wilcoxon_paired(a, b)
        title_full = f"[{tag}] {title}" if tag else title
        self.add(title_full, "Wilcoxon", len(a), len(b), W, p)
        print(f"  {title_full}: n_pairs={len(a)} W={W:.1f} p={p:.4f} {_sig(p)}")


# ═══════════════════════════════════════════════════════════════
# 模块实现
# ═══════════════════════════════════════════════════════════════

def module1_smoke_effect(R, subject_info, mle_by_subj_scene, group_of_subj):
    """模块1: 烟雾效应"""
    print("\n" + "=" * 70)
    print("模块1: 烟雾效应 — High Smoke vs Low Smoke")
    print("=" * 70)

    sn_map = {"StaticSign": 1, "ArrowRibbon": 2, "MiniMap": 3, "TextBar": 4, "Full": 5}
    # 全量 高烟 vs 低烟
    print("\n--- 测试1.1: 全量高烟 vs 全量低烟 ---")
    high_all = [info["mle"] for k, info in subject_info.items() if "HighSmoke" in k[0]]
    low_all  = [info["mle"] for k, info in subject_info.items() if "LowSmoke" in k[0]]
    R.add_mwu("测试1.1 全量高烟 vs 全量低烟", high_all, low_all)

    # 测试1.2: 同引导类型内
    print("\n--- 测试1.2: 同引导类型 高烟 vs 低烟 ---")
    for interv in ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]:
        high = [info["mle"] for k, info in subject_info.items()
                if k[0] == f"{interv}-HighSmoke"]
        low  = [info["mle"] for k, info in subject_info.items()
                if k[0] == f"{interv}-LowSmoke"]
        R.add_mwu(f"测试1.2 {interv} 高烟 vs 低烟", high, low)

    # 测试1.3: 配对 高烟场均值 vs 低烟场均值 (scene 1,3,5 high; 2,4,6 low)
    print("\n--- 测试1.3: 配对 高烟场均值 vs 低烟场均值 ---")
    high_vals, low_vals = [], []
    for subj, scenes in mle_by_subj_scene.items():
        h = [v for sn, v in scenes.items() if int(sn) in (1, 3, 5)]
        l = [v for sn, v in scenes.items() if int(sn) in (2, 4, 6)]
        if len(h) >= 2 and len(l) >= 2:
            high_vals.append(np.mean(h))
            low_vals.append(np.mean(l))
    if len(high_vals) >= 5:
        R.add_wilcoxon("测试1.3 配对 高烟均值 vs 低烟均值 (n≥5)", high_vals, low_vals)
    else:
        print("  [跳过] 对子数不足5")

    # 测试1.4: 奇数场 vs 偶数场
    print("\n--- 测试1.4: 奇数场(1,3,5) vs 偶数场(2,4,6) ---")
    odd_all, even_all = [], []
    for scenes in mle_by_subj_scene.values():
        odd_all.extend([v for sn, v in scenes.items() if int(sn) in (1, 3, 5)])
        even_all.extend([v for sn, v in scenes.items() if int(sn) in (2, 4, 6)])
    R.add_mwu("测试1.4 奇数场 vs 偶数场", odd_all, even_all)


def module2_scene_progression(R, mle_by_subj_scene, group_of_subj):
    """模块2: 场景推进效应"""
    print("\n" + "=" * 70)
    print("模块2: 场景推进效应 — Scene Progression")
    print("=" * 70)

    # 测试2.1: 各场景描述统计
    print("\n--- 测试2.1: 各场景描述统计 ---")
    for sn in range(1, 7):
        vals = [v for scenes in mle_by_subj_scene.values() for s, v in scenes.items() if int(s) == sn]
        if vals:
            print(f"  Scene {sn}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals, ddof=1):.4f}")

    # 测试2.2: 配对场景间比较
    print("\n--- 测试2.2: 配对场景间比较 ---")
    pairs = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,3), (3,5), (1,5), (1,6)]
    for s1, s2 in pairs:
        a, b = [], []
        for scenes in mle_by_subj_scene.values():
            if s1 in scenes and s2 in scenes:
                a.append(scenes[s1])
                b.append(scenes[s2])
        if len(a) >= 5:
            R.add_wilcoxon(f"测试2.2 S{s1}→S{s2}", a, b)
        else:
            print(f"  [跳过] S{s1}→S{s2} 对子数不足5 ({len(a)})")

    # 测试2.3: 前半段 vs 后半段
    print("\n--- 测试2.3: 前半段(1,2,3) vs 后半段(4,5,6) ---")
    first_half, second_half = [], []
    for scenes in mle_by_subj_scene.values():
        fh = [v for sn, v in scenes.items() if int(sn) in (1, 2, 3)]
        sh = [v for sn, v in scenes.items() if int(sn) in (4, 5, 6)]
        if len(fh) >= 2 and len(sh) >= 2:
            first_half.append(np.mean(fh))
            second_half.append(np.mean(sh))
    if len(first_half) >= 5:
        R.add_wilcoxon("测试2.3 前半段 vs 后半段", first_half, second_half)
    else:
        print(f"  [跳过] 对子数不足5 ({len(first_half)})")

    # 测试2.4: 各组场景推进趋势 (描述性)
    print("\n--- 测试2.4: 各组场景推进趋势 (描述性) ---")
    for grp in ["A", "B", "C"]:
        subjs = [s for s, g in group_of_subj.items() if g == grp]
        print(f"  组 {grp}:")
        for sn in range(1, 7):
            vals = [mle_by_subj_scene[s][sn] for s in subjs if sn in mle_by_subj_scene[s]]
            if vals:
                print(f"    Scene {sn}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals, ddof=1):.4f}")


def module3_group_comparison(R, mle_by_subj_scene, group_of_subj):
    """模块3: 组间比较"""
    print("\n" + "=" * 70)
    print("模块3: 组间比较 — Group A vs B vs C")
    print("=" * 70)

    # 获取每组每个场景的 MLE 列表
    def group_scene_mle(grp, sn=None):
        subjs = [s for s, g in group_of_subj.items() if g == grp]
        if sn is None:
            return [v for s in subjs for v in mle_by_subj_scene[s].values()]
        return [mle_by_subj_scene[s][sn] for s in subjs if sn in mle_by_subj_scene[s]]

    # 测试3.1: 全量合并组间比较
    print("\n--- 测试3.1: 全量合并组间比较 ---")
    for g1, g2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        R.add_mwu(f"测试3.1 组{g1} vs 组{g2} (全量)", group_scene_mle(g1), group_scene_mle(g2))

    # 测试3.2: Scene 1 基线组间
    print("\n--- 测试3.2: Scene 1 基线组间 ---")
    for g1, g2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        R.add_mwu(f"测试3.2 组{g1} vs 组{g2} (S1基线)", group_scene_mle(g1, 1), group_scene_mle(g2, 1))

    # 测试3.3: 各组 S1→S3 干预效应
    print("\n--- 测试3.3: 各组 S1→S3 干预效应 ---")
    for grp in ["A", "B", "C"]:
        subjs = [s for s, g in group_of_subj.items() if g == grp]
        a = [mle_by_subj_scene[s][1] for s in subjs if 1 in mle_by_subj_scene[s] and 3 in mle_by_subj_scene[s]]
        b = [mle_by_subj_scene[s][3] for s in subjs if 1 in mle_by_subj_scene[s] and 3 in mle_by_subj_scene[s]]
        if len(a) >= 5:
            R.add_wilcoxon(f"测试3.3 组{grp} S1→S3", a, b)
        else:
            print(f"  [跳过] 组{grp} 对子数不足5 ({len(a)})")

    # 测试3.4: 各组 S1→S4
    print("\n--- 测试3.4: 各组 S1→S4 ---")
    for grp in ["A", "B", "C"]:
        subjs = [s for s, g in group_of_subj.items() if g == grp]
        a = [mle_by_subj_scene[s][1] for s in subjs if 1 in mle_by_subj_scene[s] and 4 in mle_by_subj_scene[s]]
        b = [mle_by_subj_scene[s][4] for s in subjs if 1 in mle_by_subj_scene[s] and 4 in mle_by_subj_scene[s]]
        if len(a) >= 5:
            R.add_wilcoxon(f"测试3.4 组{grp} S1→S4", a, b)
        else:
            print(f"  [跳过] 组{grp} 对子数不足5 ({len(a)})")

    # 测试3.5: 各组 S1→S5, S1→S6
    print("\n--- 测试3.5: 各组 S1→S5, S1→S6 ---")
    for grp in ["A", "B", "C"]:
        for target in [5, 6]:
            subjs = [s for s, g in group_of_subj.items() if g == grp]
            a = [mle_by_subj_scene[s][1] for s in subjs if 1 in mle_by_subj_scene[s] and target in mle_by_subj_scene[s]]
            b = [mle_by_subj_scene[s][target] for s in subjs if 1 in mle_by_subj_scene[s] and target in mle_by_subj_scene[s]]
            if len(a) >= 5:
                R.add_wilcoxon(f"测试3.5 组{grp} S1→S{target}", a, b)
            else:
                print(f"  [跳过] 组{grp} S1→S{target} 对子数不足5 ({len(a)})")


def module4_guidance_following(R, subject_info, mle_by_subj_scene):
    """模块4: 引导跟随行为"""
    print("\n" + "=" * 70)
    print("模块4: 引导跟随行为 — Guidance Following")
    print("=" * 70)

    # 测试4.1: 全量 跟随 vs 不跟随
    print("\n--- 测试4.1: 全量 跟随 vs 不跟随 ---")
    follow_all = [info["mle"] for info in subject_info.values() if info["chosen"] == info["guided"]]
    no_follow_all = [info["mle"] for info in subject_info.values() if info["chosen"] != info["guided"]]
    R.add_mwu("测试4.1 全量 跟随 vs 不跟随", follow_all, no_follow_all)

    # 测试4.2: 逐场景 跟随 vs 不跟随
    print("\n--- 测试4.2: 逐场景 跟随 vs 不跟随 ---")
    for sc in SCENARIO_FOLDERS:
        follow = [info["mle"] for k, info in subject_info.items()
                  if k[0] == sc and info["chosen"] == info["guided"]]
        no_follow = [info["mle"] for k, info in subject_info.items()
                     if k[0] == sc and info["chosen"] != info["guided"]]
        if len(follow) >= 3 and len(no_follow) >= 3:
            R.add_mwu(f"测试4.2 {sc} 跟随 vs 不跟随", follow, no_follow)
        else:
            print(f"  [跳过] {sc} 组样本量不足 (follow={len(follow)}, no_follow={len(no_follow)})")

    # 测试4.3: 行为一致性
    print("\n--- 测试4.3: 行为一致性 ---")
    subj_follow_ratio = {}
    for (sc, subj), info in subject_info.items():
        subj_follow_ratio.setdefault(subj, []).append(info["chosen"] == info["guided"])
    always, never, mixed = [], [], []
    for subj, follows in subj_follow_ratio.items():
        ratio = sum(follows) / len(follows)
        mle_vals = [info["mle"] for (s, sj), info in subject_info.items() if sj == subj]
        avg = np.mean(mle_vals) if mle_vals else np.nan
        if ratio == 1.0:
            always.append(avg)
        elif ratio == 0.0:
            never.append(avg)
        else:
            mixed.append(avg)
    print(f"  始终跟随: n={len(always)}, 从不跟随: n={len(never)}, 混合: n={len(mixed)}")
    if len(always) >= 3 and len(never) >= 3:
        R.add_mwu("测试4.3 始终跟随 vs 从不跟随", always, never)
    else:
        print("  [跳过] 组样本量不足")


def module5_exit_choice(R, subject_info, mle_by_subj_scene):
    """模块5: 出口选择模式"""
    print("\n" + "=" * 70)
    print("模块5: 出口选择模式 — Exit Choice Pattern")
    print("=" * 70)

    # 测试5.1: 按出口分组
    print("\n--- 测试5.1: 按出口分组 ---")
    exit_mle = defaultdict(list)
    for info in subject_info.values():
        exit_mle[info["chosen"]].append(info["mle"])
    print("  按出口:", {e: f"n={len(v)} mean={np.mean(v):.4f}" for e, v in sorted(exit_mle.items())})

    # E vs 非E
    e_mle = exit_mle.get("E", [])
    non_e_mle = [v for e, vals in exit_mle.items() for v in vals if e != "E"]
    if len(e_mle) >= 3 and len(non_e_mle) >= 3:
        R.add_mwu("测试5.1 E vs 非E", e_mle, non_e_mle)
    # C vs D
    c_mle, d_mle = exit_mle.get("C", []), exit_mle.get("D", [])
    if len(c_mle) >= 3 and len(d_mle) >= 3:
        R.add_mwu("测试5.1 C vs D", c_mle, d_mle)

    # 测试5.2: 出口切换行为
    print("\n--- 测试5.2: 出口切换行为 ---")
    sticky, switcher = [], []
    for subj in set(k[1] for k in subject_info):
        exits = {info["chosen"] for (s, sj), info in subject_info.items() if sj == subj}
        mle_vals = [info["mle"] for (s, sj), info in subject_info.items() if sj == subj]
        avg = np.mean(mle_vals) if mle_vals else np.nan
        if len(exits) == 1:
            sticky.append(avg)
        else:
            switcher.append(avg)
    print(f"  始终同出口: n={len(sticky)}, 切换出口: n={len(switcher)}")
    if len(sticky) >= 3 and len(switcher) >= 3:
        R.add_mwu("测试5.2 始终同出口 vs 切换出口", sticky, switcher)
    else:
        print("  [跳过] 组样本量不足")


def module6_supplementary(R, subject_info, mle_by_subj_scene, group_of_subj):
    """模块6: 补充与稳健性分析"""
    print("\n" + "=" * 70)
    print("模块6: 补充与稳健性分析")
    print("=" * 70)

    # 测试6.1: 场景目录间两两对比
    print("\n--- 测试6.1: 场景目录间两两对比 (同烟雾) ---")
    interventions = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
    for smoke in ["HighSmoke", "LowSmoke"]:
        for i in range(len(interventions)):
            for j in range(i + 1, len(interventions)):
                g1 = f"{interventions[i]}-{smoke}"
                g2 = f"{interventions[j]}-{smoke}"
                d1 = [info["mle"] for k, info in subject_info.items() if k[0] == g1]
                d2 = [info["mle"] for k, info in subject_info.items() if k[0] == g2]
                R.add_mwu(f"测试6.1 {g1} vs {g2}", d1, d2)

    # 测试6.2: MLE 变异系数 CV
    print("\n--- 测试6.2: MLE 变异系数 (CV) ---")
    for sc in SCENARIO_FOLDERS:
        vals = [info["mle"] for k, info in subject_info.items() if k[0] == sc]
        if vals and np.mean(vals) > 0:
            cv = np.std(vals, ddof=1) / np.mean(vals)
            print(f"  {sc}: CV={cv:.4f} (n={len(vals)})")

    # 测试6.3: 个体MLE稳定性
    print("\n--- 测试6.3: 个体MLE稳定性 ---")
    intra_std = {"A": [], "B": [], "C": []}
    for subj, scenes in mle_by_subj_scene.items():
        grp = group_of_subj.get(subj)
        if grp and grp in intra_std and len(scenes) >= 3:
            intra_std[grp].append(np.std(list(scenes.values()), ddof=1))
    for g1, g2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        R.add_mwu(f"测试6.3 组{g1} vs 组{g2} 个体内MLE标准差",
                  intra_std.get(g1, []), intra_std.get(g2, []))

    # 测试6.4: MLE vs 轨迹点数量 (Pearson r)
    print("\n--- 测试6.4: MLE vs 轨迹点数量 ---")
    df_mle_full = pd.read_csv(MLE_CSV, encoding="utf-8-sig")
    valid = df_mle_full[(df_mle_full["轨迹点数量"] > 0) & df_mle_full["MLE"].notna()]
    if len(valid) >= 10:
        r, p = stats.pearsonr(valid["轨迹点数量"], valid["MLE"])
        print(f"  Pearson r={r:.4f}, p={p:.4f}, n={len(valid)}")
        R.add("测试6.4 MLE vs 轨迹点数量", "Pearson", len(valid), len(valid), r, p)

    # 测试6.5: 干预阶段 vs 基线
    print("\n--- 测试6.5: 干预阶段(S3+S4) vs 基线(S1+S2) ---")
    interv_vals, base_vals = [], []
    for scenes in mle_by_subj_scene.values():
        interv_vals.extend([v for sn, v in scenes.items() if int(sn) in (3, 4)])
        base_vals.extend([v for sn, v in scenes.items() if int(sn) in (1, 2)])
    R.add_mwu("测试6.5 干预阶段 vs 基线", interv_vals, base_vals)

    # 测试6.6: 去极值稳健性
    print("\n--- 测试6.6: 去极值稳健性 S1 vs S6 ---")
    all_mle = [info["mle"] for info in subject_info.values()]
    if len(all_mle) >= 20:
        lo, hi = np.percentile(all_mle, 5), np.percentile(all_mle, 95)
        s1_ro = [info["mle"] for k, info in subject_info.items()
                 if info["scene_num"] == 1 and lo <= info["mle"] <= hi]
        s6_ro = [info["mle"] for k, info in subject_info.items()
                 if info["scene_num"] == 6 and lo <= info["mle"] <= hi]
        R.add_mwu("测试6.6 S1 vs S6 (去5%极值)", s1_ro, s6_ro)

    # 测试6.7: 各组 S1→S2
    print("\n--- 测试6.7: 各组 S1→S2 ---")
    for grp in ["A", "B", "C"]:
        subjs = [s for s, g in group_of_subj.items() if g == grp]
        a = [mle_by_subj_scene[s][1] for s in subjs if 1 in mle_by_subj_scene[s] and 2 in mle_by_subj_scene[s]]
        b = [mle_by_subj_scene[s][2] for s in subjs if 1 in mle_by_subj_scene[s] and 2 in mle_by_subj_scene[s]]
        if len(a) >= 5:
            R.add_wilcoxon(f"测试6.7 组{grp} S1→S2", a, b)
        else:
            print(f"  [跳过] 组{grp} 对子数不足5 ({len(a)})")

    # 测试6.8: 极端值识别
    print("\n--- 测试6.8: 极端值识别 ---")
    records = [(k[1], info["scene_num"], info["mle"]) for k, info in subject_info.items()]
    records.sort(key=lambda x: x[2])
    print("  最低5个 MLE:")
    for subj, sn, mle in records[:5]:
        print(f"    ({subj}, S{sn}, MLE={mle:.4f})")
    print("  最高5个 MLE:")
    for subj, sn, mle in records[-5:]:
        print(f"    ({subj}, S{sn}, MLE={mle:.4f})")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 加载数据 ──
    df_merged, subject_info, mle_by_subj_scene, chosen_by_subj_scene, group_of_subj = load_and_merge()

    # ── 运行所有模块 ──
    R = ResultCollector()

    module1_smoke_effect(R, subject_info, mle_by_subj_scene, group_of_subj)
    module2_scene_progression(R, mle_by_subj_scene, group_of_subj)
    module3_group_comparison(R, mle_by_subj_scene, group_of_subj)
    module4_guidance_following(R, subject_info, mle_by_subj_scene)
    module5_exit_choice(R, subject_info, mle_by_subj_scene)
    module6_supplementary(R, subject_info, mle_by_subj_scene, group_of_subj)

    # ── 汇总输出 ──
    significant = [r for r in R.results if r["sig"] and r["sig"] != "ns"]
    significant.sort(key=lambda x: x["p"] if x["p"] is not None else 1.0)

    print("\n" + "=" * 70)
    print(f"最终汇总: 共 {len(R.results)} 个对比, {len(significant)} 个显著")
    print("=" * 70)
    for i, r in enumerate(significant[:30], 1):
        print(f"  {i:2d}. [{r['sig']}] {r['title']} (p={r['p']:.5f})")

    # ── 保存 JSON ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(R.results),
        "significant_count": len(significant),
        "all_results": R.results,
        "significant": significant,
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
