"""
=============================================================================
Experiment_Data_A5 — MLE显著性全面对比分析脚本
=============================================================================
基于 Hermes Agent 探究过程提炼的结构化分析框架。
覆盖29个对比角度，分为6个分析模块。

使用方法:
    python mle_comprehensive_analysis.py

依赖: Python 3.8+, 仅标准库 (csv, math, re, json, collections)
数据路径: 脚本内 DATA_ROOT 变量，指向 Experiment_Data_A5 根目录
=============================================================================
"""

import csv, math, re, json, time
from collections import defaultdict

# ======================== 配置 ========================
DATA_ROOT = r"F:\ExperimentData\Experiment_Data_A5"
EXIT_CHOICE_FILE = f"{DATA_ROOT}\\Results_ExitChoice\\AllSubjects_ExitChoice.csv"
MLE_FILE = f"{DATA_ROOT}\\Results_LyapunovMLE\\个人MLE统计表.csv"
OUTPUT_DIR = f"{DATA_ROOT}\\Results_LyapunovMLE"

# 场景号 → 引导出口 映射 (文件名 A5-N → 出口字母)
SCENE_GUIDED = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

# ======================== 数据加载 ========================
def load_data():
    """加载 ExitChoice 和 MLE 数据，构建统一的 subject_info 字典"""
    exit_data = []
    with open(EXIT_CHOICE_FILE, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            exit_data.append(row)

    mle_data = []
    with open(MLE_FILE, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            mle_data.append(row)

    # subject_info[(scenario, subject_clean)] = {mle, chosen, scene_num, guided, group}
    subject_info = {}
    for row in exit_data:
        subj = clean_subject(row['Subject'])
        scen = row['Scenario']
        if is_test(subj):
            continue
        sn = extract_scene_num(row['FileName'])
        if sn is None:
            continue
        key = (scen, subj)
        if key not in subject_info:
            subject_info[key] = {}
        subject_info[key]['chosen'] = row['ChosenExit']
        subject_info[key]['scene_num'] = sn
        subject_info[key]['guided'] = SCENE_GUIDED.get(sn)
        subject_info[key]['scenario'] = scen
        m = re.search(r'_([ABC])\d', subj)
        subject_info[key]['group'] = m.group(1) if m else '?'

    for row in mle_data:
        subj = clean_subject(row['受试者'])
        scen = row['场景']
        if is_test(subj):
            continue
        ms = row['MLE'].strip()
        if ms == '':
            continue
        key = (scen, subj)
        if key in subject_info:
            subject_info[key]['mle'] = float(ms)

    # 构建辅助索引
    mle_by_subj_scene = defaultdict(dict)  # {subj: {scene_num: mle}}
    chosen_by_subj_scene = defaultdict(dict)  # {subj: {scene_num: chosen_exit}}
    group_of_subj = {}

    for (scen, subj), info in subject_info.items():
        if 'mle' not in info:
            continue
        mle_by_subj_scene[subj][info['scene_num']] = info['mle']
        chosen_by_subj_scene[subj][info['scene_num']] = info['chosen']
        if subj not in group_of_subj:
            group_of_subj[subj] = info['group']

    return subject_info, mle_by_subj_scene, chosen_by_subj_scene, group_of_subj


# ======================== 工具函数 ========================
def clean_subject(name):
    """去除 YZN_A1_1 → YZN_A1 后缀"""
    return re.sub(r'_\d+$', '', name.strip())

def is_test(s):
    return s.upper().startswith('TEST')

def extract_scene_num(filename):
    m = re.search(r'A5-(\d)', filename)
    return int(m.group(1)) if m else None

def mean_std(vals):
    n = len(vals)
    if n == 0:
        return 0, 0
    m = sum(vals) / n
    if n == 1:
        return m, 0
    return m, math.sqrt(sum((x - m)**2 for x in vals) / (n - 1))

def norm_cdf(z):
    """标准正态 CDF 近似 (Abramowitz-Stegun)"""
    if z < 0:
        return 1 - norm_cdf(-z)
    b = [0.2316419, 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    t = 1.0 / (1.0 + b[0] * abs(z))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-z * z / 2.0) * \
           (b[1] * t + b[2] * t**2 + b[3] * t**3 + b[4] * t**4 + b[5] * t**5)

def mann_whitney_u(a, b):
    """
    Mann-Whitney U 检验 (独立样本, 双尾)
    返回: (U_statistic, p_value, rank_biserial_correlation)
    """
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return 0, 1.0, 0

    combined = [(x, 0) for x in a] + [(x, 1) for x in b]
    combined.sort(key=lambda x: x[0])

    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, combined[k][1]))
        i = j

    R1 = sum(r for r, g in ranks if g == 0)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Tie correction
    tie_counts = {}
    for v, _ in combined:
        tie_counts[v] = tie_counts.get(v, 0) + 1
    tie_corr = sum(c**3 - c for c in tie_counts.values() if c > 1)

    N = n1 + n2
    mu = n1 * n2 / 2.0
    sigma = math.sqrt((n1 * n2 / (N * (N - 1))) * ((N**3 - N) / 12.0 - tie_corr / 12.0))

    if sigma < 1e-10:
        p = 1.0 if abs(U - mu) < 0.001 else 0.0001
    else:
        z = (U - mu) / sigma
        p = 2.0 * (1.0 - norm_cdf(abs(z)))
        p = max(0.0001, min(1.0, p))

    r_rb = 1.0 - (2.0 * U) / (n1 * n2)
    return U, p, r_rb


def wilcoxon_signed_rank(a, b):
    """
    Wilcoxon 符号秩检验 (配对样本, 双尾)
    返回: (W_statistic, p_value)
    """
    n = len(a)
    if n < 5:
        return 0, 1.0

    diffs = [a[i] - b[i] for i in range(n)]
    non_zero = [(abs(d), 1 if d > 0 else 0) for d in diffs if abs(d) > 1e-10]
    if len(non_zero) < 3:
        return 0, 1.0

    non_zero.sort(key=lambda x: x[0])
    ranks = []
    i = 0
    while i < len(non_zero):
        j = i
        while j < len(non_zero) and abs(non_zero[j][0] - non_zero[i][0]) < 1e-10:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, non_zero[k][1]))
        i = j

    W_pos = sum(r for r, sign in ranks if sign == 1)
    W_neg = sum(r for r, sign in ranks if sign == 0)
    W = min(W_pos, W_neg)
    nr = len(ranks)

    mu_w = nr * (nr + 1) / 4.0
    sigma_w = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24.0)

    if sigma_w < 1e-10:
        return W, 1.0
    z = (W - mu_w) / sigma_w
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    return W, max(0.0001, min(1.0, p))


def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'


# ======================== 分析模块 ========================

def module_1_smoke_effect(subject_info, mle_by_subj_scene):
    """
    ============================================================
    模块1: 烟雾效应 (High Smoke vs Low Smoke)
    ============================================================
    测试1: 全量高烟雾 vs 全量低烟雾 (独立样本 MWU)
    测试2: 同引导类型内 高vs低烟雾 (5个类型 × 1 = 5个比较)
    测试3: 配对 高烟场(1+3+5) vs 低烟场(2+4+6) 均值 (Wilcoxon)
    测试4: 奇数场 vs 偶数场 全量合并 (独立样本 MWU)
    """
    print("\n" + "=" * 70)
    print("模块1: 烟雾效应 (High Smoke vs Low Smoke)")
    print("=" * 70)

    results = []

    # --- 测试1: 全量高烟 vs 低烟 ---
    high_all = [info['mle'] for (scen, subj), info in subject_info.items()
                if 'mle' in info and 'HighSmoke' in scen]
    low_all = [info['mle'] for (scen, subj), info in subject_info.items()
               if 'mle' in info and 'LowSmoke' in scen]
    U, p, rr = mann_whitney_u(high_all, low_all)
    mh, sh = mean_std(high_all)
    ml, sl = mean_std(low_all)
    print(f"  测试1: 全量 High(n={len(high_all)}) vs Low(n={len(low_all)})")
    print(f"    High: {mh:.2f}±{sh:.2f}  Low: {ml:.2f}±{sl:.2f}  U={U:.1f} p={p:.4f} {sig_label(p)}")
    results.append(('全量高烟雾 vs 低烟雾', p, sig_label(p)))

    # --- 测试2: 同引导类型内 ---
    print(f"\n  测试2: 同引导类型内 高vs低烟雾:")
    for gt in ['StaticSign', 'ArrowRibbon', 'MiniMap', 'TextBar', 'Full']:
        hk, lk = f"{gt}-HighSmoke", f"{gt}-LowSmoke"
        hv = [info['mle'] for (scen, subj), info in subject_info.items() if scen == hk and 'mle' in info]
        lv = [info['mle'] for (scen, subj), info in subject_info.items() if scen == lk and 'mle' in info]
        if len(hv) >= 3 and len(lv) >= 3:
            U, p, rr = mann_whitney_u(hv, lv)
            print(f"    {gt}: High({len(hv)})={mean_std(hv)[0]:.2f} Low({len(lv)})={mean_std(lv)[0]:.2f} p={p:.4f} {sig_label(p)}")
            results.append((f'{gt} 高vs低烟雾', p, sig_label(p)))

    # --- 测试3: 配对高烟场 vs 低烟场 ---
    paired_oe = []
    for subj, scenes in mle_by_subj_scene.items():
        odd = [scenes[s] for s in [1, 3, 5] if s in scenes]
        even = [scenes[s] for s in [2, 4, 6] if s in scenes]
        if len(odd) >= 2 and len(even) >= 2:
            paired_oe.append((sum(odd) / len(odd), sum(even) / len(even)))
    if len(paired_oe) >= 5:
        ov = [x[0] for x in paired_oe]
        ev = [x[1] for x in paired_oe]
        W, p = wilcoxon_signed_rank(ov, ev)
        print(f"\n  测试3: 配对 高烟场均值 vs 低烟场均值 (n={len(paired_oe)})")
        print(f"    高烟: {mean_std(ov)[0]:.2f}  低烟: {mean_std(ev)[0]:.2f}  W={W:.1f} p={p:.4f} {sig_label(p)}")
        results.append(('配对 高烟场 vs 低烟场', p, sig_label(p)))

    # --- 测试4: 奇数场 vs 偶数场 ---
    odd_all = []
    even_all = []
    for subj, scenes in mle_by_subj_scene.items():
        for sn in [1, 3, 5]:
            if sn in scenes: odd_all.append(scenes[sn])
        for sn in [2, 4, 6]:
            if sn in scenes: even_all.append(scenes[sn])
    U, p, rr = mann_whitney_u(odd_all, even_all)
    print(f"\n  测试4: 奇数场(高烟) vs 偶数场(低烟) 全量")
    print(f"    奇数: n={len(odd_all)} MLE={mean_std(odd_all)[0]:.2f}  偶数: n={len(even_all)} MLE={mean_std(even_all)[0]:.2f}")
    print(f"    U={U:.1f} p={p:.4f} {sig_label(p)}")
    results.append(('奇数场 vs 偶数场 全量', p, sig_label(p)))

    return results


def module_2_scene_progression(mle_by_subj_scene, group_of_subj):
    """
    ============================================================
    模块2: 场景推进效应 (Scene Progression / Learning Effect)
    ============================================================
    测试5: 各场景序号描述统计 (Scene 1-6)
    测试6: 配对场景间比较 (S1→2, 2→3, 3→4, 4→5, 5→6, 1→3, 3→5, 1→5, 1→6) — Wilcoxon
    测试7: 前半段(S1-3均值) vs 后半段(S4-6均值) 配对
    测试8: 各组场景推进趋势 (Group × Scene 交互)
    """
    print("\n" + "=" * 70)
    print("模块2: 场景推进效应")
    print("=" * 70)

    results = []

    # --- 测试5: 各场景描述 ---
    scene_mles = defaultdict(list)
    for subj, scenes in mle_by_subj_scene.items():
        for sn, mle in scenes.items():
            scene_mles[sn].append(mle)
    print(f"  测试5: 各场景序号 MLE 描述统计:")
    for sn in sorted(scene_mles.keys()):
        vals = scene_mles[sn]
        m, s = mean_std(vals)
        print(f"    Scene {sn}: n={len(vals)} MLE={m:.2f}±{s:.2f}")

    # --- 测试6: 配对场景比较 ---
    print(f"\n  测试6: 配对场景间比较 (Wilcoxon):")
    scene_pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 3), (3, 5), (1, 5), (1, 6)]
    for s_from, s_to in scene_pairs:
        paired = []
        for subj, scenes in mle_by_subj_scene.items():
            if s_from in scenes and s_to in scenes:
                paired.append((scenes[s_from], scenes[s_to]))
        if len(paired) < 5:
            continue
        a_vals, b_vals = [x[0] for x in paired], [x[1] for x in paired]
        W, p = wilcoxon_signed_rank(a_vals, b_vals)
        ma, sa = mean_std(a_vals)
        mb, sb = mean_std(b_vals)
        direction = "↑" if mb > ma else "↓"
        print(f"    S{s_from}→S{s_to}: {ma:.2f}→{mb:.2f} {direction} n={len(paired)} W={W:.1f} p={p:.4f} {sig_label(p)}")
        results.append((f'配对 S{s_from}→S{s_to}', p, sig_label(p)))

    # --- 测试7: 前半 vs 后半 ---
    first_half, second_half = [], []
    for subj, scenes in mle_by_subj_scene.items():
        fh = [scenes[s] for s in [1, 2, 3] if s in scenes]
        sh = [scenes[s] for s in [4, 5, 6] if s in scenes]
        if len(fh) >= 2 and len(sh) >= 2:
            first_half.append(sum(fh) / len(fh))
            second_half.append(sum(sh) / len(sh))
    if len(first_half) >= 5:
        W, p = wilcoxon_signed_rank(first_half, second_half)
        print(f"\n  测试7: 前半(S1-3) vs 后半(S4-6) 配对 n={len(first_half)}")
        print(f"    前半: {mean_std(first_half)[0]:.2f}  后半: {mean_std(second_half)[0]:.2f}  W={W:.1f} p={p:.4f} {sig_label(p)}")
        results.append(('前半段 vs 后半段', p, sig_label(p)))

    # --- 测试8: 各组场景趋势 ---
    print(f"\n  测试8: 各组场景推进趋势:")
    for group in 'ABC':
        subjs = [s for s, g in group_of_subj.items() if g == group]
        print(f"    组{group}:")
        for sn in range(1, 7):
            vals = [mle_by_subj_scene[subj][sn] for subj in subjs if sn in mle_by_subj_scene[subj]]
            if vals:
                m, s = mean_std(vals)
                print(f"      S{sn}: n={len(vals)} MLE={m:.2f}±{s:.2f}")

    return results


def module_3_group_comparison(subject_info, mle_by_subj_scene, group_of_subj):
    """
    ============================================================
    模块3: 组间比较 (Group A vs B vs C)
    ============================================================
    测试9:  全量合并 组A vs B vs C (独立 MWU, 3对)
    测试10: Scene 1 基线 组间比较 (独立 MWU, 3对)
    测试11: 各组 S1→S3 干预效应 (配对 Wilcoxon, 3组)
    测试12: 各组 S1→S4 (配对 Wilcoxon, 3组)
    测试13: 各组 S1→S5, S1→S6 (配对 Wilcoxon, 6组)
    """
    print("\n" + "=" * 70)
    print("模块3: 组间比较")
    print("=" * 70)

    results = []

    # --- 测试9: 全量合并组间 ---
    group_mles = defaultdict(list)
    for (scen, subj), info in subject_info.items():
        if 'mle' in info and info['group'] in 'ABC':
            group_mles[info['group']].append(info['mle'])

    print(f"  测试9: 全量合并 组间比较:")
    for g1, g2 in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
        U, p, rr = mann_whitney_u(group_mles[g1], group_mles[g2])
        m1, s1 = mean_std(group_mles[g1])
        m2, s2 = mean_std(group_mles[g2])
        print(f"    {g1}({len(group_mles[g1])})={m1:.2f} vs {g2}({len(group_mles[g2])})={m2:.2f}  U={U:.1f} p={p:.4f} {sig_label(p)}")
        results.append((f'全量 组{g1} vs 组{g2}', p, sig_label(p)))

    # --- 测试10: Scene 1 组间 ---
    s1_by_group = defaultdict(list)
    for (scen, subj), info in subject_info.items():
        if 'mle' in info and info['scene_num'] == 1 and info['group'] in 'ABC':
            s1_by_group[info['group']].append(info['mle'])

    print(f"\n  测试10: Scene 1 基线 组间比较:")
    for g1, g2 in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
        U, p, rr = mann_whitney_u(s1_by_group[g1], s1_by_group[g2])
        print(f"    {g1}({len(s1_by_group[g1])}) vs {g2}({len(s1_by_group[g2])})  U={U:.1f} p={p:.4f} {sig_label(p)}")
        results.append((f'Scene1 组{g1} vs 组{g2}', p, sig_label(p)))

    # --- 测试11-13: 组内配对 S1 → 各场景 ---
    for test_name, target_scenes in [
        ('测试11: S1→S3(干预)', [3]),
        ('测试12: S1→S4', [4]),
        ('测试13: S1→S5/S6', [5, 6]),
    ]:
        print(f"\n  {test_name}:")
        for group in 'ABC':
            subjs = [s for s, g in group_of_subj.items() if g == group]
            for target in target_scenes:
                s1v, stv = [], []
                for subj in subjs:
                    if 1 in mle_by_subj_scene[subj] and target in mle_by_subj_scene[subj]:
                        s1v.append(mle_by_subj_scene[subj][1])
                        stv.append(mle_by_subj_scene[subj][target])
                if len(s1v) >= 5:
                    W, p = wilcoxon_signed_rank(s1v, stv)
                    m1, s1_ = mean_std(s1v)
                    mt, st_ = mean_std(stv)
                    print(f"    组{group} S1→S{target}: {m1:.2f}→{mt:.2f} n={len(s1v)} W={W:.1f} p={p:.4f} {sig_label(p)}")
                    results.append((f'组{group} S1→S{target}', p, sig_label(p)))

    return results


def module_4_guidance_following(subject_info, mle_by_subj_scene, group_of_subj):
    """
    ============================================================
    模块4: 引导跟随行为
    ============================================================
    测试14: 全量 跟随引导 vs 不跟随 (独立 MWU)
    测试15: 按场景 跟随 vs 不跟随 (逐场景 MWU, 10个)
    测试16: 行为一致性: 始终跟随 vs 从不跟随 vs 混合
    """
    print("\n" + "=" * 70)
    print("模块4: 引导跟随行为")
    print("=" * 70)

    results = []

    # --- 测试14: 全量 ---
    follow_all, nofollow_all = [], []
    for (scen, subj), info in subject_info.items():
        if 'mle' not in info or info['guided'] is None:
            continue
        if info['chosen'] == info['guided']:
            follow_all.append(info['mle'])
        else:
            nofollow_all.append(info['mle'])

    U, p, rr = mann_whitney_u(follow_all, nofollow_all)
    print(f"  测试14: 全量 跟随(n={len(follow_all)}) vs 不跟随(n={len(nofollow_all)})")
    print(f"    跟随: {mean_std(follow_all)[0]:.2f}  不跟随: {mean_std(nofollow_all)[0]:.2f}")
    print(f"    U={U:.1f} p={p:.4f} {sig_label(p)}")
    results.append(('全量 跟随 vs 不跟随', p, sig_label(p)))

    # --- 测试15: 逐场景 ---
    print(f"\n  测试15: 逐场景 跟随 vs 不跟随:")
    scenario_types = sorted(set(info['scenario'] for info in subject_info.values() if 'scenario' in info))
    for scenario in scenario_types:
        fv, nfv = [], []
        for (scen, subj), info in subject_info.items():
            if scen != scenario or 'mle' not in info or info['guided'] is None:
                continue
            if info['chosen'] == info['guided']:
                fv.append(info['mle'])
            else:
                nfv.append(info['mle'])
        if len(fv) >= 3 and len(nfv) >= 3:
            U, p, rr = mann_whitney_u(fv, nfv)
            print(f"    {scenario}: 跟随({len(fv)})={mean_std(fv)[0]:.2f} vs 不跟随({len(nfv)})={mean_std(nfv)[0]:.2f} p={p:.4f} {sig_label(p)}")
            results.append((f'{scenario} 跟随 vs 不跟随', p, sig_label(p)))

    # --- 测试16: 行为一致性 ---
    follow_count = defaultdict(int)
    total_scenes = defaultdict(int)
    for (scen, subj), info in subject_info.items():
        if 'mle' not in info or info['guided'] is None:
            continue
        total_scenes[subj] += 1
        if info['chosen'] == info['guided']:
            follow_count[subj] += 1

    always, never, mixed = [], [], []
    for subj in total_scenes:
        ratio = follow_count[subj] / total_scenes[subj] if total_scenes[subj] > 0 else 0
        avg_mle = sum(mle_by_subj_scene[subj].values()) / len(mle_by_subj_scene[subj])
        if ratio == 1.0:
            always.append(avg_mle)
        elif ratio == 0.0:
            never.append(avg_mle)
        else:
            mixed.append(avg_mle)

    print(f"\n  测试16: 行为一致性:")
    print(f"    始终跟随: n={len(always)} MLE={mean_std(always)[0]:.2f}")
    print(f"    从不跟随: n={len(never)} MLE={mean_std(never)[0]:.2f}")
    print(f"    混合策略: n={len(mixed)} MLE={mean_std(mixed)[0]:.2f}")
    if len(always) >= 3 and len(never) >= 3:
        U, p, rr = mann_whitney_u(always, never)
        print(f"    始终 vs 从不: U={U:.1f} p={p:.4f} {sig_label(p)}")
        results.append(('始终跟随 vs 从不跟随', p, sig_label(p)))

    return results


def module_5_exit_choice(subject_info, mle_by_subj_scene, chosen_by_subj_scene):
    """
    ============================================================
    模块5: 出口选择模式
    ============================================================
    测试17: 按选择出口分组 全量比较 (E vs 非E, C vs D)
    测试18: 出口切换行为: 始终同出口 vs 切换出口
    测试19: 各出口的 MLE 描述统计
    """
    print("\n" + "=" * 70)
    print("模块5: 出口选择模式")
    print("=" * 70)

    results = []

    # --- 测试17: 按出口分组 ---
    exit_mles = defaultdict(list)
    for (scen, subj), info in subject_info.items():
        if 'mle' in info:
            exit_mles[info['chosen']].append(info['mle'])

    print(f"  测试17: 按选择出口 MLE 描述:")
    for ex in sorted(exit_mles.keys()):
        vals = exit_mles[ex]
        print(f"    出口 {ex}: n={len(vals)} MLE={mean_std(vals)[0]:.2f}±{mean_std(vals)[1]:.2f}")

    # E vs 非E
    e_vals = exit_mles.get('E', [])
    non_e = [m for ex, vals in exit_mles.items() if ex != 'E' for m in vals]
    if len(e_vals) >= 3 and len(non_e) >= 3:
        U, p, rr = mann_whitney_u(e_vals, non_e)
        print(f"    E vs 非E: U={U:.1f} p={p:.4f} {sig_label(p)}")
        results.append(('出口E vs 非E', p, sig_label(p)))

    # C vs D
    c_vals = exit_mles.get('C', [])
    d_vals = exit_mles.get('D', [])
    if len(c_vals) >= 3 and len(d_vals) >= 3:
        U, p, rr = mann_whitney_u(c_vals, d_vals)
        print(f"    C vs D: U={U:.1f} p={p:.4f} {sig_label(p)}")
        results.append(('出口C vs 出口D', p, sig_label(p)))

    # --- 测试18: 出口切换 ---
    unique_counts = {}
    for subj, scenes in chosen_by_subj_scene.items():
        exits = [scenes[s] for s in sorted(scenes.keys())]
        unique_counts[subj] = len(set(exits))

    same_exit_mle, varied_exit_mle = [], []
    for subj, scenes in mle_by_subj_scene.items():
        if subj not in unique_counts:
            continue
        avg = sum(scenes.values()) / len(scenes)
        if unique_counts[subj] == 1:
            same_exit_mle.append(avg)
        else:
            varied_exit_mle.append(avg)

    print(f"\n  测试18: 出口切换行为:")
    print(f"    始终同出口: n={len(same_exit_mle)} MLE={mean_std(same_exit_mle)[0]:.2f}")
    print(f"    切换出口:   n={len(varied_exit_mle)} MLE={mean_std(varied_exit_mle)[0]:.2f}")
    if len(same_exit_mle) >= 3 and len(varied_exit_mle) >= 3:
        U, p, rr = mann_whitney_u(same_exit_mle, varied_exit_mle)
        print(f"    U={U:.1f} p={p:.4f} {sig_label(p)}")
        results.append(('始终同出口 vs 切换出口', p, sig_label(p)))

    return results


def module_6_supplementary(subject_info, mle_by_subj_scene, group_of_subj):
    """
    ============================================================
    模块6: 补充分析
    ============================================================
    测试20: 各场景目录间两两对比 (高烟雾条件内)
    测试21: 各场景目录间两两对比 (低烟雾条件内)
    测试22: MLE 变异系数 (CV) 比较
    测试23: 个体 MLE 稳定性 (个体内标准差) 组间比较
    测试24: MLE 与轨迹点数量相关性
    测试25: 干预阶段(S3-4) vs 基线(S1-2)
    测试26: 去极值后 S1 vs S6 稳健性检验
    测试27: 各组 S1→S2 配对
    测试28: 极端值识别
    """
    print("\n" + "=" * 70)
    print("模块6: 补充分析")
    print("=" * 70)

    results = []

    # --- 测试20-21: 场景目录间两两对比 ---
    for smoke, label in [('HighSmoke', '高烟雾'), ('LowSmoke', '低烟雾')]:
        scenarios = [s for s in sorted(set(info['scenario'] for info in subject_info.values() if 'scenario' in info))
                     if smoke in s]
        print(f"\n  测试20-21: {label}条件内 场景目录两两对比:")
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                s1, s2 = scenarios[i], scenarios[j]
                v1 = [info['mle'] for (scen, subj), info in subject_info.items() if scen == s1 and 'mle' in info]
                v2 = [info['mle'] for (scen, subj), info in subject_info.items() if scen == s2 and 'mle' in info]
                if len(v1) >= 3 and len(v2) >= 3:
                    U, p, rr = mann_whitney_u(v1, v2)
                    n1 = s1.replace(f'-{smoke}', '')
                    n2 = s2.replace(f'-{smoke}', '')
                    if p < 0.2:  # 只打印 p<0.2 的 (节省输出)
                        print(f"    {n1} vs {n2}: U={U:.1f} p={p:.4f} {sig_label(p)}")
                    results.append((f'{label} {n1} vs {n2}', p, sig_label(p)))

    # --- 测试22: 变异系数 ---
    print(f"\n  测试22: MLE 变异系数 (CV = std/mean):")
    for scen in sorted(set(info['scenario'] for info in subject_info.values() if 'scenario' in info)):
        vals = [info['mle'] for (s, subj), info in subject_info.items() if s == scen and 'mle' in info]
        if len(vals) >= 5:
            m, s = mean_std(vals)
            print(f"    {scen}: CV={s/m:.3f}" if m > 0 else f"    {scen}: CV=N/A")

    # --- 测试23: 个体MLE稳定性 ---
    subj_variability = []
    for subj, scenes in mle_by_subj_scene.items():
        vals = list(scenes.values())
        if len(vals) >= 3:
            m, s = mean_std(vals)
            subj_variability.append((subj, m, s))

    print(f"\n  测试23: 个体MLE稳定性 (个体内标准差) 组间比较:")
    for group in 'ABC':
        gvars = [s[2] for s in subj_variability if group_of_subj.get(s[0]) == group]
        if gvars:
            print(f"    组{group}: {mean_std(gvars)[0]:.2f}±{mean_std(gvars)[1]:.2f} (n={len(gvars)})")
    for g1, g2 in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
        v1 = [s[2] for s in subj_variability if group_of_subj.get(s[0]) == g1]
        v2 = [s[2] for s in subj_variability if group_of_subj.get(s[0]) == g2]
        if len(v1) >= 3 and len(v2) >= 3:
            U, p, rr = mann_whitney_u(v1, v2)
            print(f"    {g1} vs {g2}: U={U:.1f} p={p:.4f} {sig_label(p)}")
            results.append((f'个体MLE变异性 组{g1} vs 组{g2}', p, sig_label(p)))

    # --- 测试24: MLE vs 轨迹点数量 ---
    import csv as csv2
    n_vals, mle_vals_n = [], []
    with open(MLE_FILE, 'r', encoding='utf-8-sig') as f:
        for row in csv2.DictReader(f):
            subj = clean_subject(row['受试者'])
            if is_test(subj): continue
            ms = row['MLE'].strip()
            if ms == '': continue
            tp = row['轨迹点数量'].strip()
            if tp in ('0', ''): continue
            n_vals.append(int(tp))
            mle_vals_n.append(float(ms))

    if len(n_vals) > 5:
        mx = sum(n_vals) / len(n_vals)
        my = sum(mle_vals_n) / len(mle_vals_n)
        cov = sum((n_vals[i] - mx) * (mle_vals_n[i] - my) for i in range(len(n_vals)))
        sx = math.sqrt(sum((x - mx)**2 for x in n_vals))
        sy = math.sqrt(sum((y - my)**2 for y in mle_vals_n))
        r = cov / (sx * sy) if sx * sy > 0 else 0
        t_stat = r * math.sqrt((len(n_vals) - 2) / (1 - r * r)) if abs(r) < 1 else 999
        p_corr = 2 * (1 - norm_cdf(abs(t_stat))) if len(n_vals) > 2 else 1.0
        print(f"\n  测试24: MLE vs 轨迹点数量 Pearson r={r:.4f} p={p_corr:.4f} {sig_label(p_corr)}")
        results.append(('MLE vs 轨迹点数量', p_corr, sig_label(p_corr)))

    # --- 测试25: 干预 vs 基线 ---
    interv, baseline = [], []
    for (scen, subj), info in subject_info.items():
        if 'mle' not in info: continue
        if info['scene_num'] in (3, 4):
            interv.append(info['mle'])
        elif info['scene_num'] in (1, 2):
            baseline.append(info['mle'])
    U, p, rr = mann_whitney_u(interv, baseline)
    print(f"\n  测试25: 干预(S3-4) vs 基线(S1-2):")
    print(f"    干预: n={len(interv)} MLE={mean_std(interv)[0]:.2f} 基线: n={len(baseline)} MLE={mean_std(baseline)[0]:.2f}")
    print(f"    U={U:.1f} p={p:.4f} {sig_label(p)}")
    results.append(('干预 vs 基线', p, sig_label(p)))

    # --- 测试26: 稳健性 ---
    all_mles_flat = [(subj, sn, mle_by_subj_scene[subj][sn])
                     for subj in mle_by_subj_scene for sn in mle_by_subj_scene[subj]]
    all_mles_flat.sort(key=lambda x: x[2])
    n_all = len(all_mles_flat)
    lo, hi = all_mles_flat[int(n_all * 0.05)][2], all_mles_flat[int(n_all * 0.95)][2]
    s1_trim = [m for _, sn, m in all_mles_flat if sn == 1 and lo <= m <= hi]
    s6_trim = [m for _, sn, m in all_mles_flat if sn == 6 and lo <= m <= hi]
    U, p, rr = mann_whitney_u(s1_trim, s6_trim)
    print(f"\n  测试26: 去极值(5%) S1 vs S6 稳健性:")
    print(f"    S1: n={len(s1_trim)} MLE={mean_std(s1_trim)[0]:.2f} S6: n={len(s6_trim)} MLE={mean_std(s6_trim)[0]:.2f}")
    print(f"    U={U:.1f} p={p:.4f} {sig_label(p)}")
    results.append(('稳健性 S1 vs S6(去极值)', p, sig_label(p)))

    # --- 测试27: 各组 S1→S2 ---
    print(f"\n  测试27: 各组 S1→S2 配对:")
    for group in 'ABC':
        subjs = [s for s, g in group_of_subj.items() if g == group]
        s1v, s2v = [], []
        for subj in subjs:
            if 1 in mle_by_subj_scene[subj] and 2 in mle_by_subj_scene[subj]:
                s1v.append(mle_by_subj_scene[subj][1])
                s2v.append(mle_by_subj_scene[subj][2])
        if len(s1v) >= 5:
            W, p = wilcoxon_signed_rank(s1v, s2v)
            print(f"    组{group}: {mean_std(s1v)[0]:.2f}→{mean_std(s2v)[0]:.2f} n={len(s1v)} W={W:.1f} p={p:.4f} {sig_label(p)}")
            results.append((f'组{group} S1→S2', p, sig_label(p)))

    # --- 测试28: 极端值 ---
    print(f"\n  测试28: MLE极端值 (最低/最高5个):")
    for label, items in [('最低', all_mles_flat[:5]), ('最高', all_mles_flat[-5:])]:
        print(f"    {label}: {[(s[0][:8], f'S{s[1]}', f'{s[2]:.2f}') for s in items]}")

    return results


# ======================== 主流程 ========================
def main():
    print("=" * 70)
    print("Experiment_Data_A5  MLE 全面对比分析")
    print(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 加载数据
    subject_info, mle_by_subj_scene, chosen_by_subj_scene, group_of_subj = load_data()
    print(f"\n数据加载完成: {len(subject_info)} 条 (场景,受试者) 记录")
    print(f"受试者数: {len(mle_by_subj_scene)}")

    # 运行所有模块
    all_results = []

    all_results.extend(module_1_smoke_effect(subject_info, mle_by_subj_scene))
    all_results.extend(module_2_scene_progression(mle_by_subj_scene, group_of_subj))
    all_results.extend(module_3_group_comparison(subject_info, mle_by_subj_scene, group_of_subj))
    all_results.extend(module_4_guidance_following(subject_info, mle_by_subj_scene, group_of_subj))
    all_results.extend(module_5_exit_choice(subject_info, mle_by_subj_scene, chosen_by_subj_scene))
    all_results.extend(module_6_supplementary(subject_info, mle_by_subj_scene, group_of_subj))

    # 最终汇总
    print("\n\n" + "=" * 70)
    print("最终汇总")
    print("=" * 70)

    # 按 p 值排序
    significant = [(title, p, sig) for title, p, sig in all_results if sig != 'ns']
    significant.sort(key=lambda x: x[1])

    print(f"\n共 {len(all_results)} 个对比, {len(significant)} 个显著:\n")
    for i, (title, p, sig) in enumerate(significant, 1):
        print(f"  {i}. [{sig}] {title}  (p={p:.4f})")

    # 保存结果
    output_path = f"{OUTPUT_DIR}\\comprehensive_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(all_results),
            'significant_count': len(significant),
            'all_results': [{'title': t, 'p': round(p, 6), 'sig': s} for t, p, s in all_results],
            'significant': [{'title': t, 'p': round(p, 6), 'sig': s} for t, p, s in significant],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {output_path}")
    return significant


if __name__ == '__main__':
    main()
