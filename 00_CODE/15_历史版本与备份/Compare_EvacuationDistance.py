import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

from common import PROJECT_ROOT, INDICATORS_DIR


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 路径设置
# =========================

INPUT_BASE_DIR = os.path.join(INDICATORS_DIR, "Results_EvacuationDistance")
SUMMARY_FILE = os.path.join(INPUT_BASE_DIR, "AllSubjects_EvacuationDistance.csv")

OUTPUT_DIR = os.path.join(INDICATORS_DIR, "Results_GroupComparison_EvacuationDistance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 绘图顺序
# =========================

GROUP_ORDER = [
    "StaticSign-HighSmoke",
    "StaticSign-LowSmoke",
    "ArrowRibbon-HighSmoke",
    "ArrowRibbon-LowSmoke",
    "MiniMap-HighSmoke",
    "MiniMap-LowSmoke",
    "TextBar-HighSmoke",
    "TextBar-LowSmoke",
    "Full-HighSmoke",
    "Full-LowSmoke",
]

PLOT_LABELS = [
    "StaticSign\nHigh Smoke",
    "StaticSign\nLow Smoke",
    "ArrowRibbon\nHigh Smoke",
    "ArrowRibbon\nLow Smoke",
    "MiniMap\nHigh Smoke",
    "MiniMap\nLow Smoke",
    "TextBar\nHigh Smoke",
    "TextBar\nLow Smoke",
    "Full\nHigh Smoke",
    "Full\nLow Smoke",
]

COLORS = [
    "#4f7d75", "#9bbfba",   # StaticSign
    "#112052", "#3d5a99",   # ArrowRibbon
    "#5f5f5f", "#aaaaaa",   # MiniMap
    "#2f667f", "#7b9fb1",   # TextBar
    "#b88a2c", "#e0c58f",   # Full
]


# =========================
# 显著性设置
# =========================

KEY_PAIRS = [
    ("StaticSign-HighSmoke", "StaticSign-LowSmoke"),
    ("ArrowRibbon-HighSmoke", "ArrowRibbon-LowSmoke"),
    ("MiniMap-HighSmoke", "MiniMap-LowSmoke"),
    ("TextBar-HighSmoke", "TextBar-LowSmoke"),
    ("Full-HighSmoke", "Full-LowSmoke"),
]

KEY_COMPARISONS_FOR_PLOT = [
    (0, 1, "StaticSign-HighSmoke", "StaticSign-LowSmoke"),
    (2, 3, "ArrowRibbon-HighSmoke", "ArrowRibbon-LowSmoke"),
    (4, 5, "MiniMap-HighSmoke", "MiniMap-LowSmoke"),
    (6, 7, "TextBar-HighSmoke", "TextBar-LowSmoke"),
    (8, 9, "Full-HighSmoke", "Full-LowSmoke"),
]


def get_sig_annotation(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def load_evacuation_distance_data():
    """
    优先读取总体结果表：
    INDICATORS/Results_EvacuationDistance/AllSubjects_EvacuationDistance.csv

    如果总体结果表不存在，则读取每个场景文件夹下的：
    INDICATORS/Results_EvacuationDistance/场景名/EvacuationDistance.csv
    """

    if os.path.exists(SUMMARY_FILE):
        print(f"读取总体结果表: {SUMMARY_FILE}")
        df = pd.read_csv(SUMMARY_FILE)
        return df

    print("[提示] 未找到总体结果表，开始读取各场景 EvacuationDistance.csv")

    all_data = []

    for scenario in GROUP_ORDER:
        scenario_file = os.path.join(INPUT_BASE_DIR, scenario, "EvacuationDistance.csv")

        if not os.path.exists(scenario_file):
            print(f"  [跳过] 未找到: {scenario_file}")
            continue

        df_scenario = pd.read_csv(scenario_file)
        all_data.append(df_scenario)
        print(f"  已读取: {scenario_file}")

    if not all_data:
        raise FileNotFoundError(
            "未找到任何 EvacuationDistance.csv。请先运行 Extract_EvacuationDistance.py。"
        )

    df = pd.concat(all_data, ignore_index=True)
    return df


def build_group_data(df):
    """
    按 GROUP_ORDER 提取每个场景的 EvacuationDistance。
    """

    required_cols = ["Scenario", "EvacuationDistance"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入数据缺少必要列: {col}")

    data_dict = {}

    print("\n各场景样本量:")
    for scenario in GROUP_ORDER:
        values = df.loc[df["Scenario"] == scenario, "EvacuationDistance"]
        values = pd.to_numeric(values, errors="coerce").dropna().tolist()
        data_dict[scenario] = values
        print(f"  {scenario}: {len(values)} 个样本")

    return data_dict


def draw_significance_bars(ax, comparisons, box_data, box_positions,
                           y_start_factor=0.12, y_step_factor=0.075):
    """
    在箱型图上绘制显著性标注。
    当前只标注同一种引导方式下 HighSmoke vs LowSmoke。
    """

    all_vals = [v for data in box_data for v in data]

    if not all_vals:
        return

    data_min = min(all_vals)
    data_max = max(all_vals)
    data_range = data_max - data_min

    if data_range == 0:
        data_range = 1

    for idx, (i, j, label_i, label_j) in enumerate(comparisons):
        data_i = box_data[i]
        data_j = box_data[j]

        if len(data_i) < 2 or len(data_j) < 2:
            print(f"  [跳过显著性] {label_i} vs {label_j}: 样本量不足")
            continue

        stat, p_value = stats.mannwhitneyu(data_i, data_j, alternative="two-sided")
        sig = get_sig_annotation(p_value)

        print(f"  [Mann-Whitney U] {label_i} vs {label_j}: "
              f"stat={stat:.4f}, p={p_value:.6f}, {sig}")

        bar_y = data_max + data_range * (y_start_factor + idx * y_step_factor)

        xi = box_positions[i]
        xj = box_positions[j]

        ax.plot(
            [xi, xi, xj, xj],
            [
                bar_y - data_range * 0.008,
                bar_y,
                bar_y,
                bar_y - data_range * 0.008,
            ],
            color="black",
            linewidth=1.0,
            clip_on=False,
        )

        p_label = f"p={p_value:.4f}" if p_value >= 0.0001 else "p<0.0001"

        ax.text(
            (xi + xj) / 2,
            bar_y + data_range * 0.012,
            f"{sig}\n{p_label}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
        )


def draw_boxplot(data_dict):
    box_data = [data_dict[group] for group in GROUP_ORDER]
    box_positions = list(range(1, len(GROUP_ORDER) + 1))

    fig, ax = plt.subplots(figsize=(13, 5.8))

    # 注意：
    # 不使用 tick_labels 参数，避免旧版本 matplotlib 报错。
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_linewidth(1.0)

    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.0)

    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)

    for median in bp["medians"]:
        median.set(color="black", linewidth=2.0)

    for mean in bp["means"]:
        mean.set(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
        )

    # 叠加散点
    rng = np.random.default_rng(42)

    for pos, data, color in zip(box_positions, box_data, COLORS):
        if len(data) == 0:
            continue

        jitter = rng.normal(0, 0.045, size=len(data))
        x = np.full(len(data), pos) + jitter

        ax.scatter(
            x,
            data,
            s=16,
            color=color,
            alpha=0.55,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )

    # 均值标注
    for data, pos in zip(box_data, box_positions):
        if len(data) == 0:
            continue

        mean_val = np.mean(data)

        ax.annotate(
            f"Mean={mean_val:.2f}",
            xy=(pos, mean_val),
            xytext=(pos + 0.12, mean_val),
            fontsize=8,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
            ),
        )

    ax.set_ylabel("Evacuation Distance", fontsize=13)
    ax.set_title(
        "Evacuation Distance by Guidance Type and Smoke Condition",
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)

    ax.set_xticks(box_positions)
    ax.set_xticklabels(PLOT_LABELS, rotation=30, ha="right")

    print("\n图上显著性标注:")
    draw_significance_bars(
        ax,
        KEY_COMPARISONS_FOR_PLOT,
        box_data,
        box_positions,
        y_start_factor=0.12,
        y_step_factor=0.075,
    )

    all_vals_flat = [v for data in box_data for v in data]

    if all_vals_flat:
        data_min = min(all_vals_flat)
        data_max = max(all_vals_flat)
        data_range = data_max - data_min

        if data_range == 0:
            data_range = 1

        new_top = data_max + data_range * 0.60
        ax.set_ylim(bottom=0, top=new_top)

    legend_elements = [
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="StaticSign",
            markerfacecolor="#4f7d75",
            markersize=9,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="ArrowRibbon",
            markerfacecolor="#112052",
            markersize=9,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="MiniMap",
            markerfacecolor="#5f5f5f",
            markersize=9,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="TextBar",
            markerfacecolor="#2f667f",
            markersize=9,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="Full",
            markerfacecolor="#b88a2c",
            markersize=9,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1, 0.85),
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "EvacuationDistance_Boxplot.png")
    out_svg = os.path.join(OUTPUT_DIR, "EvacuationDistance_Boxplot.svg")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n箱型图已保存: {out_png}")
    print(f"SVG 图已保存: {out_svg}")


def save_descriptive_stats(data_dict):
    rows = []

    for scenario in GROUP_ORDER:
        data = data_dict[scenario]

        if len(data) == 0:
            rows.append({
                "Scenario": scenario,
                "N": 0,
                "Mean": np.nan,
                "Std": np.nan,
                "Median": np.nan,
                "Min": np.nan,
                "Max": np.nan,
                "Q1": np.nan,
                "Q3": np.nan,
            })
            continue

        arr = np.array(data, dtype=float)

        rows.append({
            "Scenario": scenario,
            "N": len(arr),
            "Mean": np.mean(arr),
            "Std": np.std(arr, ddof=1) if len(arr) > 1 else 0,
            "Median": np.median(arr),
            "Min": np.min(arr),
            "Max": np.max(arr),
            "Q1": np.percentile(arr, 25),
            "Q3": np.percentile(arr, 75),
        })

    df_stats = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "轨迹长度描述性统计.csv")
    df_stats.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n描述性统计已保存: {out_path}")


def save_normality_tests(data_dict):
    rows = []

    print("\n" + "=" * 60)
    print("正态性检验")
    print("=" * 60)

    for scenario in GROUP_ORDER:
        data = data_dict[scenario]

        if len(data) >= 3:
            stat, p_value = stats.shapiro(data)
            is_normal = p_value > 0.05

            rows.append({
                "Scenario": scenario,
                "N": len(data),
                "Shapiro_W": f"{stat:.4f}",
                "p_value": f"{p_value:.6f}",
                "Normal": "是" if is_normal else "否",
            })

            print(f"{scenario}: W={stat:.4f}, p={p_value:.6f}, "
                  f"{'正态' if is_normal else '非正态'}")

        else:
            rows.append({
                "Scenario": scenario,
                "N": len(data),
                "Shapiro_W": "",
                "p_value": "",
                "Normal": "样本量不足",
            })

            print(f"{scenario}: 样本量不足，无法进行 Shapiro 检验")

    df_normality = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "轨迹长度正态性检验.csv")
    df_normality.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n正态性检验结果已保存: {out_path}")


def save_pairwise_comparisons(data_dict):
    """
    输出两类统计比较：
    1. 重点比较：同一引导方式下 HighSmoke vs LowSmoke；
    2. 全部 10 个场景之间的两两比较。

    默认使用 Mann-Whitney U 检验。
    """

    print("\n" + "=" * 60)
    print("两两对比检验")
    print("=" * 60)

    rows_key = []
    rows_all = []

    print("\n重点比较：同一引导方式下 HighSmoke vs LowSmoke")

    for group1, group2 in KEY_PAIRS:
        data1 = data_dict[group1]
        data2 = data_dict[group2]

        if len(data1) < 2 or len(data2) < 2:
            print(f"  [跳过] {group1} vs {group2}: 样本量不足")
            continue

        stat_val, p_value = stats.mannwhitneyu(
            data1,
            data2,
            alternative="two-sided",
        )

        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        std1 = np.std(data1, ddof=1) if len(data1) > 1 else 0
        std2 = np.std(data2, ddof=1) if len(data2) > 1 else 0

        sig = get_sig_annotation(p_value)

        print(f"  {group1} vs {group2}: U={stat_val:.4f}, "
              f"p={p_value:.6f}, {sig}")

        rows_key.append({
            "Group1": group1,
            "Group2": group2,
            "Comparison": f"{group1} vs {group2}",
            "Test": "Mann-Whitney U",
            "Statistic": f"{stat_val:.4f}",
            "p_value": f"{p_value:.6f}",
            "Significance": sig,
            "Group1_N": len(data1),
            "Group2_N": len(data2),
            "Group1_Mean": f"{mean1:.4f}",
            "Group1_Std": f"{std1:.4f}",
            "Group2_Mean": f"{mean2:.4f}",
            "Group2_Std": f"{std2:.4f}",
        })

    print("\n全部场景两两比较")

    for group1, group2 in itertools.combinations(GROUP_ORDER, 2):
        data1 = data_dict[group1]
        data2 = data_dict[group2]

        if len(data1) < 2 or len(data2) < 2:
            continue

        stat_val, p_value = stats.mannwhitneyu(
            data1,
            data2,
            alternative="two-sided",
        )

        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        std1 = np.std(data1, ddof=1) if len(data1) > 1 else 0
        std2 = np.std(data2, ddof=1) if len(data2) > 1 else 0

        sig = get_sig_annotation(p_value)

        rows_all.append({
            "Group1": group1,
            "Group2": group2,
            "Comparison": f"{group1} vs {group2}",
            "Test": "Mann-Whitney U",
            "Statistic": f"{stat_val:.4f}",
            "p_value": f"{p_value:.6f}",
            "Significance": sig,
            "Group1_N": len(data1),
            "Group2_N": len(data2),
            "Group1_Mean": f"{mean1:.4f}",
            "Group1_Std": f"{std1:.4f}",
            "Group2_Mean": f"{mean2:.4f}",
            "Group2_Std": f"{std2:.4f}",
        })

    df_key = pd.DataFrame(rows_key)
    key_path = os.path.join(
        OUTPUT_DIR,
        "轨迹长度重点对比_HighSmoke_vs_LowSmoke.csv"
    )
    df_key.to_csv(key_path, index=False, encoding="utf-8-sig")

    df_all = pd.DataFrame(rows_all)
    all_path = os.path.join(
        OUTPUT_DIR,
        "轨迹长度全部两两对比.csv"
    )
    df_all.to_csv(all_path, index=False, encoding="utf-8-sig")

    print(f"\n重点对比结果已保存: {key_path}")
    print(f"全部两两对比结果已保存: {all_path}")


def save_selected_raw_data(df):
    selected_rows = df[df["Scenario"].isin(GROUP_ORDER)].copy()

    selected_rows["Scenario"] = pd.Categorical(
        selected_rows["Scenario"],
        categories=GROUP_ORDER,
        ordered=True,
    )

    sort_cols = ["Scenario"]

    if "Subject" in selected_rows.columns:
        sort_cols.append("Subject")

    selected_rows = selected_rows.sort_values(sort_cols)

    selected_path = os.path.join(
        OUTPUT_DIR,
        "轨迹长度箱型图原始数据.csv"
    )

    selected_rows.to_csv(selected_path, index=False, encoding="utf-8-sig")

    print(f"\n绘图原始数据已保存: {selected_path}")


def main():
    print("\n" + "=" * 70)
    print("轨迹长度箱型图分析 - EvacuationDistance")
    print("=" * 70)
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"输入目录: {INPUT_BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    df = load_evacuation_distance_data()

    data_dict = build_group_data(df)

    save_selected_raw_data(df)

    draw_boxplot(data_dict)

    save_descriptive_stats(data_dict)
    save_normality_tests(data_dict)
    save_pairwise_comparisons(data_dict)

    print("\n显著性标注说明：*** p<0.001  ** p<0.01  * p<0.05  ns 不显著")

    print("\n" + "=" * 70)
    print("全部处理完成")
    print("=" * 70)
    print(f"结果已保存至: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()