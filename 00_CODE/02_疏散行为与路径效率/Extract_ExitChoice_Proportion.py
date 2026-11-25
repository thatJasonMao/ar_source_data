import os
import pandas as pd

from common import PROJECT_ROOT


# =========================
# 路径设置
# =========================

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "Results_ExitChoice",
    "AllSubjects_ExitChoice.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "Results_ExitChoice_Proportion"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 顺序设置
# =========================

SCENARIO_ORDER = [
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

EXIT_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]


def load_exit_choice_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"未找到输入文件: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_cols = ["Scenario", "ChosenExit"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列: {col}")

    df = df.copy()

    df["Scenario"] = df["Scenario"].astype(str).str.strip()
    df["ChosenExit"] = df["ChosenExit"].astype(str).str.strip().str.upper()

    # 只保留 A-H 的有效出口
    df_valid = df[df["ChosenExit"].isin(EXIT_ORDER)].copy()

    invalid_count = len(df) - len(df_valid)
    if invalid_count > 0:
        print(f"[提示] 有 {invalid_count} 条记录的 ChosenExit 不在 A-H 范围内，已排除。")

    return df_valid


def get_ordered_scenarios(df):
    """
    优先使用预设顺序。
    如果数据中存在预设顺序之外的场景，也会自动追加到最后。
    """

    existing_scenarios = df["Scenario"].dropna().unique().tolist()

    ordered = [s for s in SCENARIO_ORDER if s in existing_scenarios]
    extra = sorted([s for s in existing_scenarios if s not in ordered])

    return ordered + extra


def build_summary_tables(df):
    ordered_scenarios = get_ordered_scenarios(df)

    rows_wide = []
    rows_long = []

    for scenario in ordered_scenarios:
        sub_df = df[df["Scenario"] == scenario]
        total_n = len(sub_df)

        if total_n == 0:
            continue

        count_series = sub_df["ChosenExit"].value_counts()

        row = {
            "Scenario": scenario,
            "TotalN": total_n,
        }

        for exit_name in EXIT_ORDER:
            count = int(count_series.get(exit_name, 0))
            proportion = count / total_n * 100 if total_n > 0 else 0

            row[f"{exit_name}_Count"] = count
            row[f"{exit_name}_Percent"] = proportion

            rows_long.append({
                "Scenario": scenario,
                "Exit": exit_name,
                "Count": count,
                "Percent": proportion,
                "TotalN": total_n,
            })

        rows_wide.append(row)

    wide_df = pd.DataFrame(rows_wide)
    long_df = pd.DataFrame(rows_long)

    return wide_df, long_df


def format_percentage(value):
    return f"{value:.2f}%"


def make_text_summary(wide_df):
    lines = []

    lines.append("=" * 80)
    lines.append("Exit Choice Proportion Summary")
    lines.append("=" * 80)
    lines.append("说明：比例 = 某出口选择人数 / 当前场景总人数 × 100%")
    lines.append("")

    for _, row in wide_df.iterrows():
        scenario = row["Scenario"]
        total_n = int(row["TotalN"])

        lines.append("-" * 80)
        lines.append(f"{scenario}")
        lines.append(f"Total N = {total_n}")
        lines.append("-" * 80)

        exit_items = []

        for exit_name in EXIT_ORDER:
            count = int(row[f"{exit_name}_Count"])
            percent = float(row[f"{exit_name}_Percent"])

            exit_items.append((exit_name, count, percent))

            lines.append(
                f"Exit {exit_name}: "
                f"{count} / {total_n} "
                f"({format_percentage(percent)})"
            )

        # 找出最多选择的出口
        max_count = max(item[1] for item in exit_items)
        dominant_exits = [
            item for item in exit_items
            if item[1] == max_count and item[1] > 0
        ]

        if dominant_exits:
            dominant_text = ", ".join([
                f"Exit {exit_name} ({count}/{total_n}, {format_percentage(percent)})"
                for exit_name, count, percent in dominant_exits
            ])
            lines.append("")
            lines.append(f"Most selected exit: {dominant_text}")
        else:
            lines.append("")
            lines.append("Most selected exit: None")

        lines.append("")

    lines.append("=" * 80)
    lines.append("End of summary")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_outputs(wide_df, long_df, text_summary):
    wide_path = os.path.join(
        OUTPUT_DIR,
        "ExitChoice_Proportion_ByScenario_Wide.csv"
    )

    long_path = os.path.join(
        OUTPUT_DIR,
        "ExitChoice_Proportion_ByScenario_Long.csv"
    )

    text_path = os.path.join(
        OUTPUT_DIR,
        "ExitChoice_Proportion_TextSummary.txt"
    )

    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_summary)

    print(f"\n宽表结果已保存: {wide_path}")
    print(f"长表结果已保存: {long_path}")
    print(f"文本结果已保存: {text_path}")


def main():
    print("\n" + "=" * 80)
    print("出口选择比例统计")
    print("=" * 80)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    df = load_exit_choice_data()

    if df.empty:
        raise ValueError("有效出口选择数据为空，请检查 ChosenExit 字段。")

    wide_df, long_df = build_summary_tables(df)

    text_summary = make_text_summary(wide_df)

    print(text_summary)

    save_outputs(wide_df, long_df, text_summary)

    print("\n" + "=" * 80)
    print("全部处理完成")
    print("=" * 80)


if __name__ == "__main__":
    main()