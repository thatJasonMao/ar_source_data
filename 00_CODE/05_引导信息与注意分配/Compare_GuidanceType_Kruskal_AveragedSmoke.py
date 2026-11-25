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

DISTANCE_FILE = os.path.join(
    INDICATORS_DIR,
    "Results_EvacuationDistance",
    "AllSubjects_EvacuationDistance.csv"
)

SPEED_FILE = os.path.join(
    INDICATORS_DIR,
    "Results_EvacuationSpeed",
    "AllSubjects_EvacuationSpeed.csv"
)

TIME_FILE = os.path.join(
    INDICATORS_DIR,
    "Results_EvacuationTime",
    "AllSubjects_EvacuationTime.csv"
)

ROTATION_FILE = os.path.join(
    INDICATORS_DIR,
    "Results_EvacuationRotation",
    "AllSubjects_EvacuationRotation.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "Results_GuidanceType_Kruskal_AveragedSmoke"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 分组顺序与颜色
# =========================

GUIDANCE_ORDER = [
    "StaticSign",
    "ArrowRibbon",
    "MiniMap",
    "TextBar",
    "Full",
]

GUIDANCE_LABELS = [
    "StaticSign",
    "ArrowRibbon",
    "MiniMap",
    "TextBar",
    "Full",
]

COLORS = {
    "StaticSign": "#4f7d75",
    "ArrowRibbon": "#112052",
    "MiniMap": "#5f5f5f",
    "TextBar": "#2f667f",
    "Full": "#b88a2c",
}


# =========================
# 指标设置
# =========================

METRIC_CONFIGS = [
    {
        "metric_name": "EvacuationDistance",
        "file_path": DISTANCE_FILE,
        "value_col": "EvacuationDistance",
        "y_label": "Evacuation Distance",
        "title": "Evacuation Distance by Guidance Type",
    },
    {
        "metric_name": "AverageSpeed",
        "file_path": SPEED_FILE,
        "value_col": "AverageSpeed_m_s",
        "y_label": "Average Evacuation Speed (m/s)",
        "title": "Average Evacuation Speed by Guidance Type",
    },
    {
        "metric_name": "EvacuationTime",
        "file_path": TIME_FILE,
        "value_col": "EvacuationTime_s",
        "y_label": "Evacuation Time (s)",
        "title": "Evacuation Time by Guidance Type",
    },
    {
        "metric_name": "EvacuationRotation",
        "file_path": ROTATION_FILE,
        "value_col": "TotalRotation_deg",
        "y_label": "Cumulative Rotation (deg)",
        "title": "Cumulative Rotation by Guidance Type",
    },
]


# 是否要求同一个受试者在同一引导方式下必须同时有 HighSmoke 和 LowSmoke
# True: 只有同时具备 HighSmoke 和 LowSmoke 的记录才参与平均和后续检验
# False: 如果只有一个烟雾条件，也保留该值
REQUIRE_BOTH_SMOKE = True


# =========================
# 工具函数
# =========================

def parse_scenario(scenario):
    """
    将场景名拆成 GuidanceType 和 SmokeCondition。

    例如：
        StaticSign-HighSmoke -> StaticSign, HighSmoke
        ArrowRibbon-LowSmoke -> ArrowRibbon, LowSmoke
    """

    scenario = str(scenario).strip()

    if scenario.endswith("-HighSmoke"):
        guidance = scenario.replace("-HighSmoke", "")
        smoke = "HighSmoke"
    elif scenario.endswith("-LowSmoke"):
        guidance = scenario.replace("-LowSmoke", "")
        smoke = "LowSmoke"
    else:
        guidance = scenario
        smoke = "Unknown"

    return guidance, smoke


def get_sig_annotation(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def holm_adjust(p_values):
    """
    Holm-Bonferroni p值校正。
    输入: p值列表
    输出: 校正后的p值列表
    """

    p_values = np.array(p_values, dtype=float)
    m = len(p_values)

    if m == 0:
        return []

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.zeros(m)

    for rank, p in enumerate(sorted_p):
        adjusted_sorted[rank] = (m - rank) * p

    # 保证校正后的p值单调不下降
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.zeros(m)
    adjusted[order] = adjusted_sorted

    return adjusted.tolist()


def load_metric_data(metric_config):
    file_path = metric_config["file_path"]
    value_col = metric_config["value_col"]
    metric_name = metric_config["metric_name"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = ["Scenario", "Subject", value_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{file_path} 缺少必要列: {col}")

    df = df.copy()
    df["Metric"] = metric_name
    df["MetricValue"] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["MetricValue"])

    guidance_smoke = df["Scenario"].apply(parse_scenario)
    df["GuidanceType"] = guidance_smoke.apply(lambda x: x[0])
    df["SmokeCondition"] = guidance_smoke.apply(lambda x: x[1])

    df = df[df["GuidanceType"].isin(GUIDANCE_ORDER)]
    df = df[df["SmokeCondition"].isin(["HighSmoke", "LowSmoke"])]

    return df


def average_high_low_by_subject(df):
    """
    对同一 Subject + GuidanceType 下的 HighSmoke 和 LowSmoke 取平均。

    输出每个受试者在每种引导模式下的一条记录。
    """

    grouped = []

    group_cols = ["Metric", "Subject", "GuidanceType"]

    for keys, sub_df in df.groupby(group_cols):
        metric, subject, guidance = keys

        smoke_set = set(sub_df["SmokeCondition"].tolist())

        has_high = "HighSmoke" in smoke_set
        has_low = "LowSmoke" in smoke_set

        if REQUIRE_BOTH_SMOKE and not (has_high and has_low):
            continue

        high_values = sub_df.loc[
            sub_df["SmokeCondition"] == "HighSmoke",
            "MetricValue"
        ].tolist()

        low_values = sub_df.loc[
            sub_df["SmokeCondition"] == "LowSmoke",
            "MetricValue"
        ].tolist()

        mean_value = float(np.mean(sub_df["MetricValue"]))

        grouped.append({
            "Metric": metric,
            "Subject": subject,
            "GuidanceType": guidance,
            "MeanAcrossSmoke": mean_value,
            "HighSmokeMean": float(np.mean(high_values)) if high_values else np.nan,
            "LowSmokeMean": float(np.mean(low_values)) if low_values else np.nan,
            "SmokeConditionCount": len(smoke_set),
            "HasHighSmoke": has_high,
            "HasLowSmoke": has_low,
            "OriginalRecordCount": len(sub_df),
        })

    out_df = pd.DataFrame(grouped)

    if not out_df.empty:
        out_df["GuidanceType"] = pd.Categorical(
            out_df["GuidanceType"],
            categories=GUIDANCE_ORDER,
            ordered=True
        )
        out_df = out_df.sort_values(["Metric", "GuidanceType", "Subject"])

    return out_df


def run_kruskal_for_metric(avg_df, metric_name):
    """
    对一个指标进行 Kruskal-Wallis 总体检验。
    """

    metric_df = avg_df[avg_df["Metric"] == metric_name].copy()

    group_values = []
    group_ns = {}

    for guidance in GUIDANCE_ORDER:
        values = metric_df.loc[
            metric_df["GuidanceType"] == guidance,
            "MeanAcrossSmoke"
        ].dropna().tolist()

        group_values.append(values)
        group_ns[guidance] = len(values)

    valid_group_values = [v for v in group_values if len(v) >= 2]
    valid_group_names = [
        g for g, v in zip(GUIDANCE_ORDER, group_values)
        if len(v) >= 2
    ]

    if len(valid_group_values) < 2:
        return {
            "Metric": metric_name,
            "Test": "Kruskal-Wallis",
            "H": np.nan,
            "p_value": np.nan,
            "Significance": "insufficient groups",
            "EpsilonSquared": np.nan,
            "TotalN": sum(group_ns.values()),
            **{f"N_{g}": group_ns[g] for g in GUIDANCE_ORDER},
        }

    h_stat, p_value = stats.kruskal(*valid_group_values)

    total_n = sum(len(v) for v in valid_group_values)
    k = len(valid_group_values)

    if total_n > k:
        epsilon_squared = (h_stat - k + 1) / (total_n - k)
        epsilon_squared = max(0.0, float(epsilon_squared))
    else:
        epsilon_squared = np.nan

    return {
        "Metric": metric_name,
        "Test": "Kruskal-Wallis",
        "ComparedGroups": "; ".join(valid_group_names),
        "H": float(h_stat),
        "p_value": float(p_value),
        "Significance": get_sig_annotation(p_value),
        "EpsilonSquared": epsilon_squared,
        "TotalN": total_n,
        **{f"N_{g}": group_ns[g] for g in GUIDANCE_ORDER},
    }


def run_pairwise_for_metric(avg_df, metric_name):
    """
    Kruskal-Wallis 后的两两比较。
    默认使用 Mann-Whitney U 检验，并进行 Holm 校正。
    """

    metric_df = avg_df[avg_df["Metric"] == metric_name].copy()

    rows = []
    raw_p_values = []

    for g1, g2 in itertools.combinations(GUIDANCE_ORDER, 2):
        data1 = metric_df.loc[
            metric_df["GuidanceType"] == g1,
            "MeanAcrossSmoke"
        ].dropna().tolist()

        data2 = metric_df.loc[
            metric_df["GuidanceType"] == g2,
            "MeanAcrossSmoke"
        ].dropna().tolist()

        if len(data1) < 2 or len(data2) < 2:
            continue

        u_stat, p_value = stats.mannwhitneyu(
            data1,
            data2,
            alternative="two-sided"
        )

        n1 = len(data1)
        n2 = len(data2)

        mean1 = float(np.mean(data1))
        mean2 = float(np.mean(data2))
        median1 = float(np.median(data1))
        median2 = float(np.median(data2))
        std1 = float(np.std(data1, ddof=1)) if n1 > 1 else 0.0
        std2 = float(np.std(data2, ddof=1)) if n2 > 1 else 0.0

        # Rank-biserial correlation
        # 正值表示 g1 的值整体上倾向于高于 g2
        rank_biserial = (2 * u_stat) / (n1 * n2) - 1

        rows.append({
            "Metric": metric_name,
            "Group1": g1,
            "Group2": g2,
            "Comparison": f"{g1} vs {g2}",
            "Test": "Mann-Whitney U",
            "U": float(u_stat),
            "p_raw": float(p_value),
            "p_holm": np.nan,
            "Significance_raw": get_sig_annotation(p_value),
            "Significance_holm": "",
            "RankBiserial_Group1Higher": float(rank_biserial),
            "Group1_N": n1,
            "Group2_N": n2,
            "Group1_Mean": mean1,
            "Group2_Mean": mean2,
            "Group1_Median": median1,
            "Group2_Median": median2,
            "Group1_Std": std1,
            "Group2_Std": std2,
        })

        raw_p_values.append(p_value)

    adjusted_p_values = holm_adjust(raw_p_values)

    for row, p_adj in zip(rows, adjusted_p_values):
        row["p_holm"] = p_adj
        row["Significance_holm"] = get_sig_annotation(p_adj)

    return rows


def save_descriptive_stats(avg_df):
    rows = []

    for metric_name in avg_df["Metric"].unique():
        metric_df = avg_df[avg_df["Metric"] == metric_name]

        for guidance in GUIDANCE_ORDER:
            values = metric_df.loc[
                metric_df["GuidanceType"] == guidance,
                "MeanAcrossSmoke"
            ].dropna().tolist()

            if len(values) == 0:
                rows.append({
                    "Metric": metric_name,
                    "GuidanceType": guidance,
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

            arr = np.array(values, dtype=float)

            rows.append({
                "Metric": metric_name,
                "GuidanceType": guidance,
                "N": len(arr),
                "Mean": float(np.mean(arr)),
                "Std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "Median": float(np.median(arr)),
                "Min": float(np.min(arr)),
                "Max": float(np.max(arr)),
                "Q1": float(np.percentile(arr, 25)),
                "Q3": float(np.percentile(arr, 75)),
            })

    stats_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "DescriptiveStats_AveragedSmoke.csv")
    stats_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"描述性统计已保存: {out_path}")


def draw_single_metric_boxplot(avg_df, metric_config, kruskal_row):
    metric_name = metric_config["metric_name"]
    y_label = metric_config["y_label"]
    title = metric_config["title"]

    metric_df = avg_df[avg_df["Metric"] == metric_name].copy()

    box_data = []
    for guidance in GUIDANCE_ORDER:
        values = metric_df.loc[
            metric_df["GuidanceType"] == guidance,
            "MeanAcrossSmoke"
        ].dropna().tolist()
        box_data.append(values)

    box_positions = list(range(1, len(GUIDANCE_ORDER) + 1))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
    )

    for patch, guidance in zip(bp["boxes"], GUIDANCE_ORDER):
        patch.set_facecolor(COLORS[guidance])
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

    rng = np.random.default_rng(42)

    for pos, values, guidance in zip(box_positions, box_data, GUIDANCE_ORDER):
        if len(values) == 0:
            continue

        jitter = rng.normal(0, 0.04, size=len(values))
        x = np.full(len(values), pos) + jitter

        ax.scatter(
            x,
            values,
            s=18,
            color=COLORS[guidance],
            alpha=0.55,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )

        mean_val = np.mean(values)

        ax.annotate(
            f"Mean={mean_val:.2f}",
            xy=(pos, mean_val),
            xytext=(pos + 0.10, mean_val),
            fontsize=8,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
            ),
        )

    h_value = kruskal_row.get("H", np.nan)
    p_value = kruskal_row.get("p_value", np.nan)
    sig = kruskal_row.get("Significance", "")

    if not pd.isna(p_value):
        subtitle = f"Kruskal-Wallis: H={h_value:.3f}, p={p_value:.4f}, {sig}"
    else:
        subtitle = "Kruskal-Wallis: insufficient data"

    ax.set_title(
        f"{title}\n{subtitle}",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks(box_positions)
    ax.set_xticklabels(GUIDANCE_LABELS, rotation=25, ha="right", fontsize=11)

    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=11)

    legend_elements = [
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label=guidance,
            markerfacecolor=COLORS[guidance],
            markersize=9,
        )
        for guidance in GUIDANCE_ORDER
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1, 0.88),
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()

    out_png = os.path.join(
        OUTPUT_DIR,
        f"{metric_name}_GuidanceType_Boxplot_AveragedSmoke.png"
    )
    out_svg = os.path.join(
        OUTPUT_DIR,
        f"{metric_name}_GuidanceType_Boxplot_AveragedSmoke.svg"
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{metric_name} 箱线图已保存: {out_png}")
    print(f"{metric_name} SVG 已保存: {out_svg}")


def draw_four_metric_panel(avg_df, kruskal_df):
    """
    绘制四个指标的 2 x 2 合并箱线图。
    """

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2))
    axes = axes.flatten()

    for ax, metric_config in zip(axes, METRIC_CONFIGS):
        metric_name = metric_config["metric_name"]
        y_label = metric_config["y_label"]
        title = metric_config["title"]

        metric_df = avg_df[avg_df["Metric"] == metric_name].copy()

        box_data = []
        for guidance in GUIDANCE_ORDER:
            values = metric_df.loc[
                metric_df["GuidanceType"] == guidance,
                "MeanAcrossSmoke"
            ].dropna().tolist()
            box_data.append(values)

        box_positions = list(range(1, len(GUIDANCE_ORDER) + 1))

        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
        )

        for patch, guidance in zip(bp["boxes"], GUIDANCE_ORDER):
            patch.set_facecolor(COLORS[guidance])
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
                markersize=6,
            )

        rng = np.random.default_rng(42)

        for pos, values, guidance in zip(box_positions, box_data, GUIDANCE_ORDER):
            if len(values) == 0:
                continue

            jitter = rng.normal(0, 0.04, size=len(values))
            x = np.full(len(values), pos) + jitter

            ax.scatter(
                x,
                values,
                s=14,
                color=COLORS[guidance],
                alpha=0.50,
                edgecolors="black",
                linewidths=0.20,
                zorder=3,
            )

        k_row = kruskal_df[kruskal_df["Metric"] == metric_name]

        if not k_row.empty and not pd.isna(k_row.iloc[0]["p_value"]):
            h_value = k_row.iloc[0]["H"]
            p_value = k_row.iloc[0]["p_value"]
            sig = k_row.iloc[0]["Significance"]
            subtitle = f"H={h_value:.2f}, p={p_value:.4f}, {sig}"
        else:
            subtitle = "insufficient data"

        ax.set_title(
            f"{title}\nKruskal-Wallis: {subtitle}",
            fontsize=11,
            fontweight="bold"
        )
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xticks(box_positions)
        ax.set_xticklabels(GUIDANCE_LABELS, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=9)

    legend_elements = [
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label=guidance,
            markerfacecolor=COLORS[guidance],
            markersize=9,
        )
        for guidance in GUIDANCE_ORDER
    ]

    axes[-1].legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1, 0.88),
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()

    out_png = os.path.join(
        OUTPUT_DIR,
        "GuidanceType_FourMetrics_Boxplot_AveragedSmoke.png"
    )
    out_svg = os.path.join(
        OUTPUT_DIR,
        "GuidanceType_FourMetrics_Boxplot_AveragedSmoke.svg"
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"四指标合并箱线图已保存: {out_png}")
    print(f"四指标合并 SVG 已保存: {out_svg}")


def main():
    print("\n" + "=" * 70)
    print("不同引导模式主效应分析 - HighSmoke 和 LowSmoke 按受试者平均")
    print("=" * 70)
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"要求同时具备 HighSmoke 和 LowSmoke: {REQUIRE_BOTH_SMOKE}")
    print("=" * 70 + "\n")

    all_raw_data = []

    for config in METRIC_CONFIGS:
        print(f"读取指标: {config['metric_name']}")
        df_metric = load_metric_data(config)
        all_raw_data.append(df_metric)
        print(f"  原始有效记录数: {len(df_metric)}")

    raw_df = pd.concat(all_raw_data, ignore_index=True)

    raw_path = os.path.join(OUTPUT_DIR, "RawData_LongFormat.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n长格式原始数据已保存: {raw_path}")

    avg_df = average_high_low_by_subject(raw_df)

    if avg_df.empty:
        raise ValueError(
            "平均后的数据为空。请检查 Subject、Scenario、HighSmoke/LowSmoke 是否匹配。"
        )

    avg_path = os.path.join(OUTPUT_DIR, "AveragedSmoke_BySubjectGuidance.csv")
    avg_df.to_csv(avg_path, index=False, encoding="utf-8-sig")
    print(f"高低烟平均后的数据已保存: {avg_path}")

    print("\n平均后各指标、各引导方式样本量:")

    count_table = avg_df.pivot_table(
        index="Metric",
        columns="GuidanceType",
        values="Subject",
        aggfunc="count",
        fill_value=0
    )

    print(count_table)

    count_path = os.path.join(OUTPUT_DIR, "SampleSize_AveragedSmoke.csv")
    count_table.to_csv(count_path, encoding="utf-8-sig")
    print(f"样本量表已保存: {count_path}")

    save_descriptive_stats(avg_df)

    # Kruskal-Wallis 总体检验
    kruskal_rows = []

    print("\n" + "=" * 70)
    print("Kruskal-Wallis 总体检验")
    print("=" * 70)

    for config in METRIC_CONFIGS:
        metric_name = config["metric_name"]
        row = run_kruskal_for_metric(avg_df, metric_name)
        kruskal_rows.append(row)

        print(
            f"{metric_name}: H={row['H']:.4f}, "
            f"p={row['p_value']:.6f}, "
            f"{row['Significance']}, "
            f"epsilon^2={row['EpsilonSquared']:.4f}"
        )

    kruskal_df = pd.DataFrame(kruskal_rows)

    kruskal_path = os.path.join(OUTPUT_DIR, "KruskalWallis_GuidanceType.csv")
    kruskal_df.to_csv(kruskal_path, index=False, encoding="utf-8-sig")
    print(f"\nKruskal-Wallis 结果已保存: {kruskal_path}")

    # 两两比较
    pairwise_rows = []

    print("\n" + "=" * 70)
    print("事后两两比较: Mann-Whitney U + Holm 校正")
    print("=" * 70)

    for config in METRIC_CONFIGS:
        metric_name = config["metric_name"]
        rows = run_pairwise_for_metric(avg_df, metric_name)
        pairwise_rows.extend(rows)

        print(f"\n{metric_name}:")
        for r in rows:
            print(
                f"  {r['Comparison']}: "
                f"U={r['U']:.3f}, "
                f"p_raw={r['p_raw']:.6f}, "
                f"p_holm={r['p_holm']:.6f}, "
                f"{r['Significance_holm']}"
            )

    pairwise_df = pd.DataFrame(pairwise_rows)

    pairwise_path = os.path.join(OUTPUT_DIR, "Pairwise_MannWhitney_Holm.csv")
    pairwise_df.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    print(f"\n两两比较结果已保存: {pairwise_path}")

    # 绘图
    print("\n" + "=" * 70)
    print("绘制箱线图")
    print("=" * 70)

    for config in METRIC_CONFIGS:
        metric_name = config["metric_name"]
        k_row = kruskal_df[kruskal_df["Metric"] == metric_name].iloc[0].to_dict()
        draw_single_metric_boxplot(avg_df, config, k_row)

    draw_four_metric_panel(avg_df, kruskal_df)

    print("\n" + "=" * 70)
    print("全部处理完成")
    print("=" * 70)
    print(f"结果已保存至: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()