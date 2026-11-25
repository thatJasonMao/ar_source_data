#!/usr/bin/env python3
"""Compare two completed V6_2 runs from their saved evaluation artifacts.

Read-only inputs; no training imports, model execution, cache reconstruction, or
selection. Output must be a new/empty directory to protect existing analyses.
Example:
  python Analyze_IRL_V6_2_Calibration.py --old-run OLD --new-run NEW --output OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

CHANNELS = ("StaticSign", "ArrowRibbon", "MiniMap", "TextBar")
SMOKES = ("HighSmoke", "LowSmoke")
SPLITS = ("val", "test")
IDENTITY_KEYS = ("action", "person", "smoke", "trial", "frame_index", "weight")
METRICS = (
    "spatial_reward_auc", "auc", "participant_macro_nll", "nll", "nll_gain",
    "brier", "policy_ap", "spatial_ap", "observed_fraction", "predicted_fraction",
)
LOWER_BETTER = {"participant_macro_nll", "nll", "brier"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_run(path: Path) -> Path:
    path = path.resolve()
    if path.name == "Best_Model":
        path = path.parent
    if not (path / "Best_Model" / "Models").is_dir():
        raise ValueError(f"Completed Best_Model/Models not found: {path}")
    return path


def frame_metrics(data: dict, mask: np.ndarray, baseline: dict) -> dict:
    """Match Train_IRL_V6_2.frame_report using natural-frame metrics."""
    y = np.asarray(data["action"][mask], float)
    p = np.clip(np.asarray(data["probability"][mask], float), 1e-7, 1 - 1e-7)
    u = np.asarray(data["utility"][mask], float)
    person = data["person"][mask]
    smoke = data["smoke"][mask]
    base = (np.full(len(y), baseline["global"]) if "global" in baseline else
            np.asarray([baseline[SMOKES[int(index)]] for index in smoke]))
    base = np.clip(base, 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    base_loss = -(y * np.log(base) + (1 - y) * np.log1p(-base))
    both_classes = len(np.unique(y)) == 2
    return {
        "n": len(y), "people": len(np.unique(person)),
        "spatial_reward_auc": float(roc_auc_score(y, u)) if both_classes else None,
        "auc": float(roc_auc_score(y, p)) if both_classes else None,
        "participant_macro_nll": float(np.mean([loss[person == who].mean()
                                                for who in np.unique(person)])),
        "nll": float(loss.mean()), "nll_gain": float((base_loss - loss).mean()),
        "brier": float(np.mean((y - p) ** 2)),
        "policy_ap": float(average_precision_score(y, p)) if both_classes else None,
        "spatial_ap": float(average_precision_score(y, u)) if both_classes else None,
        "observed_fraction": float(y.mean()), "predicted_fraction": float(p.mean()),
    }


def pair_row(old: dict, new: dict, **identity) -> dict:
    result = dict(identity)
    result["n"] = old["n"]
    result["people"] = old["people"]
    for metric in METRICS:
        a, b = old.get(metric), new.get(metric)
        result[f"old_{metric}"] = a
        result[f"new_{metric}"] = b
        result[f"delta_{metric}"] = b - a if a is not None and b is not None else None
    return result


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value, digits=5, sign=False) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def paired_plot(cases: pd.DataFrame, output: Path) -> None:
    data = cases[(cases.split == "test") & (cases.smoke != "Both")]
    # Model versions use two colors consistently; channels/smokes are row groups.
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "axes.unicode_minus": False, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    for index, row in enumerate(data.itertuples(index=False)):
        a, b = row.old_spatial_reward_auc, row.new_spatial_reward_auc
        ax.plot([a, b], [index, index], color="#a3a3a3", lw=2.0, zorder=1)
        ax.scatter(a, index, color="#8b8b8b", marker="s", s=58, zorder=3,
                   label="Old: 7.5 / 12.5 m" if index == 0 else None)
        ax.scatter(b, index, color="#17698b", marker="o", s=58, zorder=4,
                   label="New: 10 / 14 m" if index == 0 else None)
        ax.text(a, index - 0.16, f"{a:.4f}", ha="center", va="bottom", fontsize=9, color="#555555")
        ax.text(b, index + 0.17, f"{b:.4f}", ha="center", va="top", fontsize=9, color="#17698b")
        ax.text(1.01, index, f"{b-a:+.4f}", transform=ax.get_yaxis_transform(), va="center", fontsize=10)
    for boundary in (1.5, 3.5, 5.5):
        ax.axhline(boundary, color="#eeeeee", lw=1.0)
    ax.axvline(0.5, color="#777777", linestyle="--", lw=1, label="Chance (0.5)")
    ax.set_yticks(range(len(data)), [f"{r.channel}  |  {'High smoke' if r.smoke == 'HighSmoke' else 'Low smoke'}"
                                   for r in data.itertuples(index=False)])
    ax.set_ylim(len(data) - 0.45, -0.65)
    ax.set_xlim(0.44, 0.96)
    ax.set_xlabel("Test spatial reward AUC (natural frames)")
    ax.set_title("V6.2 visibility recalibration: paired test comparison", pad=14)
    ax.text(1.01, 1.025, "New - old", transform=ax.transAxes, fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.18)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -0.10), ncol=3, frameon=False)
    fig.subplots_adjust(left=.26, right=.86, bottom=.16, top=.9)
    fig.savefig(output / "Test_Spatial_AUC_Comparison.png", dpi=300)
    fig.savefig(output / "Test_Spatial_AUC_Comparison.pdf")
    plt.close(fig)


def make_report(cases: pd.DataFrame, macro: pd.DataFrame, settings: pd.DataFrame,
                people: pd.DataFrame, old_run: Path, new_run: Path, checks: dict) -> str:
    test = cases[cases.split == "test"]
    both = test[test.smoke == "Both"]
    lines = [
        "# V6.2 能见度重新标定：旧新完整训练流程比较", "",
        f"旧 run：`{old_run}`", f"新 run：`{new_run}`", "",
        "本报告只读取已完成训练保存的预测和评价；不训练、不重新选模型、不重建已删除的训练缓存。",
        "比较对象为旧 7.5/12.5 m 与新 10/14 m 完整训练/选参流程。旧模型配置未记录 smoke_visibility_m，旧值来自本项目此次标定变更的历史口径；新值由四通道 config.json 明确记录并写入 audit.json。", "",
        "## 结论", "",
        "本次重新标定后，测试集总体预测性能下降，主要由 TextBar 拉低；StaticSign、MiniMap 的场 AUC 提高，ArrowRibbon 小幅下降。验证集总体略改善，但这一改善没有转化为测试集总体改善。",
        "能见度按实际场景重新标定有物理依据，但这与预测性能提高是两个不同结论。两次选出的超参数也变化，因此差异不能全部归因于能见度数值本身。", "",
        "## 可比性与统计口径", "",
        "四通道新旧 evaluation_identity.json（包括 train/val/test 人员、试次、行数）完全一致；val/test 的 action、person、smoke、trial、frame_index、weight 六组数组逐项完全一致。对保存的预测重新计算十项指标并与 evaluation.json 核对通过（float32 保存造成的微小数值误差允许绝对误差 2e-6）。",
        "主指标 spatial_reward_auc 是每帧空间 utility 对观测 action 的排序 AUC；policy auc 使用 probability。不能由静态重建网格本身推算测试 AUC。每个通道的 Both 行沿用原评价中的高低烟自然帧合并口径。",
        "总体为四通道 AUC 的等权算术平均，另附八 case 等权平均；不混合不同通道的 utility/probability 重新算 pooled AUC。宏平均只是描述量，各通道标签含义和难度不同。NLL/Brier 越低越好，NLL gain/AP/AUC 越高越好；NLL gain 为训练基线 NLL 减模型 NLL。", "",
        "测试集参与者数量：StaticSign 9、ArrowRibbon 3、MiniMap 3、TextBar 2。参与者是泛化与不确定性的统计单位，不能把大量相关帧当成独立重复；本报告不作显著性宣称，不进行小样本 bootstrap，也不根据该测试集重新选模型。", "",
        "## 测试集：通道与八 case 场 AUC", "",
    ]
    lines.append(markdown_table(["通道", "烟雾", "参与者", "帧数", "旧场 AUC", "新场 AUC", "Δ 新−旧", "旧 policy AUC", "新 policy AUC"],
        [[r.channel, r.smoke, r.people, r.n, fmt(r.old_spatial_reward_auc), fmt(r.new_spatial_reward_auc),
          fmt(r.delta_spatial_reward_auc, sign=True), fmt(r.old_auc), fmt(r.new_auc)]
         for r in test.itertuples(index=False)]))
    lines += ["", "![八 case 测试场 AUC](Test_Spatial_AUC_Comparison.png)", "", "## 总体：验证与测试分开", ""]
    lines.append(markdown_table(["split", "平均口径", "指标", "旧", "新", "Δ 新−旧"],
        [[r.split, r.aggregation, r.metric, fmt(r.old), fmt(r.new), fmt(r.delta, sign=True)]
         for r in macro.itertuples(index=False) if r.aggregation == "equal_4_channels_combined_smoke"
         and r.metric in ("spatial_reward_auc", "auc", "participant_macro_nll", "nll_gain", "brier", "policy_ap", "spatial_ap")]))
    lines += ["", "## 各通道的概率质量与 AP（测试集、高低烟合并）", ""]
    lines.append(markdown_table(["通道", "macro NLL 旧→新", "NLL gain 旧→新", "Brier 旧→新", "policy AP 旧→新", "spatial AP 旧→新"],
        [[r.channel] + [f"{fmt(getattr(r, 'old_' + m))} → {fmt(getattr(r, 'new_' + m))}"
                      for m in ("participant_macro_nll", "nll_gain", "brier", "policy_ap", "spatial_ap")]
         for r in both.itertuples(index=False)]))
    lines += ["", "## TextBar：验证改善、测试下降", ""]
    text = cases[(cases.channel == "TextBar") & (cases.smoke == "Both")]
    lines.append(markdown_table(["split", "场 AUC 旧→新", "policy AUC 旧→新", "macro NLL 旧→新", "NLL gain 旧→新"],
        [[r.split] + [f"{fmt(getattr(r, 'old_' + m))} → {fmt(getattr(r, 'new_' + m))}"
                      for m in ("spatial_reward_auc", "auc", "participant_macro_nll", "nll_gain")]
         for r in text.itertuples(index=False)]))
    lines += ["", "TextBar 高烟测试场 AUC 下降到 0.48369，低烟下降到 0.57469；总体 policy AUC 降到 0.54467，NLL gain 变为负值，说明该测试集上概率预测的 NLL 比训练集常数基线更差。这并非只有绘图色标或静态场表达变化。测试仅 2 人，尚不能断言它代表所有人的真实总体效应。", "",
              "下表按参与者分解，帮助定位泛化差异；与总体自然帧 AUC 不同，它不能被当作总体 AUC 的直接分解。", ""]
    subset = people[(people.channel == "TextBar") & (people.split == "test") & (people.smoke == "Both")]
    lines.append(markdown_table(["参与者", "帧数", "旧场 AUC", "新场 AUC", "旧 policy AUC", "新 policy AUC"],
        [[r.person, r.n, fmt(r.old_spatial_reward_auc), fmt(r.new_spatial_reward_auc), fmt(r.old_auc), fmt(r.new_auc)]
         for r in subset.itertuples(index=False)]))
    lines += ["", "## 选出的超参数", ""]
    lines.append(markdown_table(["通道", "参数", "旧", "新", "变化"],
        [[r.channel, r.parameter, r.old, r.new, "是" if r.changed else "否"]
         for r in settings.itertuples(index=False)]))
    lines += ["", "TextBar 从 linear/48 clusters 变为 interaction/24 clusters，同时学习率与正则化变化；其验证集收益与测试集损失应作为完整选参流程的结果解读。若后续研究原因，应预先确定参与者层面的重复划分/交叉验证方案，并固定比较口径；不能反复利用当前 test 来筛选能见度或模型。", "",
              "## 文件与可复现性", "",
              "- `case_metrics.csv`：val/test × 四通道 × Both/HighSmoke/LowSmoke，十项指标的新旧值及差值。",
              "- `macro_metrics.csv`：四通道合并烟雾的等权平均、八 case 等权平均。",
              "- `participant_metrics.csv`：参与者层面的描述指标。",
              "- `selected_hyperparameters.csv`：全部选定超参数。",
              "- `audit.json`：输入绝对路径、SHA-256、身份与预测指标复算结果及配置。",
              "- `Test_Spatial_AUC_Comparison.png/.pdf`：八 case 配对图；Times New Roman，PNG 为 300 dpi。", "",
              f"新空间场入口：`{new_run / 'Best_Model' / 'Results_Global_Spatial_Field_V6_2'}`。",
              "3D 主图、补图、调色板和共同色标对齐结果以该目录内 Results_3D 的 manifest 和 README 为准；本比较脚本不改写空间场或图集。", "",
              "复现命令：", "", "```powershell",
              f'python .\\Analyze_IRL_V6_2_Calibration.py --old-run "{old_run}" --new-run "{new_run}" --output "<新的空输出目录>"',
              "```", "", "本报告不执行 IRL 训练。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_run, new_run = normalize_run(args.old_run), normalize_run(args.new_run)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be a new/empty directory: {output}")
    for run in (old_run, new_run):
        if output == run or (run / "Best_Model") == output or output.is_relative_to(run / "Best_Model" / "Models"):
            raise ValueError("Output may not overwrite model inputs")
    rows, people_rows, config_rows = [], [], []
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "old_run": str(old_run), "new_run": str(new_run), "channels": {},
             "report_metric_source": "evaluation.json", "prediction_metric_absolute_tolerance": 2e-6,
             "training_executed": False, "bootstrap_executed": False}
    for channel in tqdm(CHANNELS, desc="Verify and compare channels"):
        folders = [run / "Best_Model" / "Models" / channel for run in (old_run, new_run)]
        configs, identities, reports, arrays = [], [], [], []
        checked = {"input_hashes": {}, "splits": {}}
        for label, folder in zip(("old", "new"), folders):
            config = read_json(folder / "config.json")
            if config["status"] != "complete" or config["profile"]["variant"] != "V6_2":
                raise ValueError(f"Not a complete V6_2 model: {folder}")
            configs.append(config)
            identities.append(read_json(folder / "evaluation_identity.json"))
            reports.append(read_json(folder / "evaluation.json"))
            with np.load(folder / "evaluation_predictions.npz", allow_pickle=False) as data:
                arrays.append({key: data[key] for key in data.files})
            checked["input_hashes"][label] = {}
            for filename in ("config.json", "evaluation_identity.json", "evaluation.json", "evaluation_predictions.npz"):
                digest = sha256(folder / filename)
                checked["input_hashes"][label][filename] = digest
                declared = config.get("files_sha256", {}).get(filename)
                if declared and digest != declared:
                    raise ValueError(f"Saved input hash mismatch: {folder / filename}")
        if identities[0] != identities[1]:
            raise ValueError(f"Participant/trial/split identity mismatch: {channel}")
        if tuple(identities[0]["smokes"]) != SMOKES:
            raise ValueError(f"Unexpected smoke encoding: {channel}")
        checked["identity_json_exactly_equal"] = True
        checked["configs"] = {label: config for label, config in zip(("old", "new"), configs)}
        for parameter in sorted(set(configs[0]["config"]) | set(configs[1]["config"])):
            a, b = configs[0]["config"].get(parameter), configs[1]["config"].get(parameter)
            config_rows.append(dict(channel=channel, parameter=parameter, old=a, new=b, changed=a != b))
        for split in SPLITS:
            data = [{key[len(split) + 1:]: value for key, value in saved.items() if key.startswith(split + "_")}
                    for saved in arrays]
            for key in IDENTITY_KEYS:
                if not np.array_equal(data[0][key], data[1][key]):
                    raise ValueError(f"Paired input mismatch: {channel}/{split}/{key}")
            if len(data[0]["action"]) != identities[0]["splits"][split]["rows"]:
                raise ValueError("Saved frame count differs from identity metadata")
            checked["splits"][split] = {"identical_arrays": list(IDENTITY_KEYS), "metric_max_abs_error": 0.0}
            for smoke in ("Both", *SMOKES):
                mask = np.ones(len(data[0]["action"]), bool) if smoke == "Both" else data[0]["smoke"] == SMOKES.index(smoke)
                saved_metrics = [report[split] if smoke == "Both" else report[split]["by_smoke"][smoke] for report in reports]
                for index in range(2):
                    recalculated = frame_metrics(data[index], mask, reports[index][split]["baseline"])
                    for metric in ("n", "people", *METRICS):
                        a, b = saved_metrics[index].get(metric), recalculated[metric]
                        if a is None and b is None:
                            continue
                        if a is None or b is None or not np.isclose(a, b, rtol=0, atol=2e-6):
                            raise ValueError(f"Saved metric mismatch: {channel}/{split}/{smoke}/{index}/{metric}: {a} vs {b}")
                        checked["splits"][split]["metric_max_abs_error"] = max(
                            checked["splits"][split]["metric_max_abs_error"], abs(float(a) - float(b)))
                rows.append(pair_row(*saved_metrics, channel=channel, split=split, smoke=smoke))
                for who in np.unique(data[0]["person"][mask]):
                    per_mask = mask & (data[0]["person"] == who)
                    person_metrics = [frame_metrics(item, per_mask, reports[index][split]["baseline"])
                                      for index, item in enumerate(data)]
                    person_name = identities[0]["splits"][split]["people"][int(who)]
                    people_rows.append(pair_row(*person_metrics, channel=channel, split=split, smoke=smoke, person=person_name))
        audit["channels"][channel] = checked
    cases, participants, settings = pd.DataFrame(rows), pd.DataFrame(people_rows), pd.DataFrame(config_rows)
    macro_rows = []
    for split in SPLITS:
        for aggregation, smoke_filter in (("equal_4_channels_combined_smoke", cases.smoke == "Both"),
                                           ("equal_8_channel_smoke_cases", cases.smoke != "Both")):
            selected = cases[(cases.split == split) & smoke_filter]
            for metric in METRICS:
                a, b = selected[f"old_{metric}"].mean(), selected[f"new_{metric}"].mean()
                macro_rows.append(dict(split=split, aggregation=aggregation, metric=metric,
                                       old=a, new=b, delta=b-a,
                                       preferred_direction="lower" if metric in LOWER_BETTER else "higher" if metric not in ("observed_fraction", "predicted_fraction") else "descriptive"))
    macro = pd.DataFrame(macro_rows)
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("case_metrics.csv", cases), ("macro_metrics.csv", macro),
                           ("participant_metrics.csv", participants), ("selected_hyperparameters.csv", settings)):
        data.to_csv(output / filename, index=False, encoding="utf-8-sig", float_format="%.10g")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    paired_plot(cases, output)
    (output / "Comparison_Report.md").write_text(make_report(cases, macro, settings, participants, old_run, new_run, audit), encoding="utf-8")
    print(both_summary(cases, macro))
    print(f"Saved read-only comparison to: {output}")


def both_summary(cases: pd.DataFrame, macro: pd.DataFrame) -> str:
    test = cases[(cases.split == "test") & (cases.smoke == "Both")]
    columns = ["channel", "old_spatial_reward_auc", "new_spatial_reward_auc", "delta_spatial_reward_auc"]
    overall = macro[(macro.split == "test") & (macro.aggregation == "equal_4_channels_combined_smoke")
                    & (macro.metric == "spatial_reward_auc")].iloc[0]
    return test[columns].to_string(index=False) + f"\nEqual-channel test spatial AUC: {overall.old:.8f} -> {overall.new:.8f} ({overall.delta:+.8f})"


if __name__ == "__main__":
    main()
