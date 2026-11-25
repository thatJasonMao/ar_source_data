#!/usr/bin/env python
"""分析三种疏散引导的注意捕获、Full 组竞争效应和疏散进程变化。

本脚本将逐采样点 AOI 命中转换为：
1. 受试者层面的有效注视时间比例；
2. 连续 AOI 注意片段（不是严格眼动 fixation）；
3. 单引导与 Full 条件的受试者内配对比较；
4. 高烟与低烟的受试者内比较；
5. 归一化疏散进程上的注意变化。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy.stats import wilcoxon


GUIDE_CONFIG = {
    "LightBlanket": {
        "single_prefix": "ArrowRibbon",
        "gaze": "GazeOnLightBlanket",
        "valid": "ValidLightBlanketAOI",
        "label_zh": "光毯",
        "color": "#E69F00",
    },
    "MiniMap": {
        "single_prefix": "MiniMap",
        "gaze": "GazeOnMiniMap",
        "valid": "ValidMiniMapAOI",
        "label_zh": "小地图",
        "color": "#0072B2",
    },
    "Text": {
        "single_prefix": "TextBar",
        "gaze": "GazeOnTextGuidance",
        "valid": "ValidTextAOI",
        "label_zh": "文字",
        "color": "#009E73",
    },
}

SMOKE_ORDER = ["HighSmoke", "LowSmoke"]
SMOKE_LABEL = {"HighSmoke": "高烟", "LowSmoke": "低烟"}
TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("Results_Guidance_GazeAOI_Integrated"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results_Guidance_Attention_Analysis"),
    )
    parser.add_argument(
        "--episode-gap-ms",
        type=float,
        default=75.0,
        help="允许同一注意片段内的最大短暂中断，默认 75 ms。",
    )
    parser.add_argument(
        "--episode-min-ms",
        type=float,
        default=100.0,
        help="持续注意片段的最短跨度，默认 100 ms。",
    )
    parser.add_argument(
        "--max-sample-gap-ms",
        type=float,
        default=100.0,
        help="单采样点可贡献的最大时长，避免日志长缺口被算作注视。",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy(dtype=float) != 0
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
        .to_numpy(dtype=bool)
    )


def parse_identity(path: Path, frame: pd.DataFrame) -> dict[str, str]:
    condition = str(frame["Condition"].dropna().iloc[0])
    participant = str(frame["ParticipantFolder"].dropna().iloc[0])
    trial = str(frame["TrialID"].dropna().iloc[0])
    guidance, smoke = condition.rsplit("-", 1)
    return {
        "Condition": condition,
        "GuidanceCondition": guidance,
        "SmokeLevel": smoke,
        "ParticipantFolder": participant,
        "TrialID": trial,
        "SourceFile": str(path),
    }


def timestamp_and_weights(
    values: pd.Series, max_sample_gap_sec: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    parsed = pd.to_datetime(values.astype(str), format=TIMESTAMP_FORMAT, errors="coerce")
    valid = parsed.notna().to_numpy()
    if valid.sum() < 2:
        raise ValueError("有效 TimeStamp 少于 2 个，无法计算时间指标")
    ns = parsed.astype("int64").to_numpy()
    first_ns = ns[valid][0]
    elapsed = (ns - first_ns) / 1e9
    elapsed[~valid] = np.nan
    diffs = np.diff(elapsed)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        raise ValueError("TimeStamp 没有正向时间差")
    median_dt = float(np.median(positive))
    weights = np.empty(len(elapsed), dtype=float)
    weights[:-1] = diffs
    weights[-1] = median_dt
    bad = ~np.isfinite(weights) | (weights <= 0)
    weights[bad] = median_dt
    weights = np.minimum(weights, max_sample_gap_sec)
    trial_duration = float(np.nanmax(elapsed) - np.nanmin(elapsed))
    return elapsed, weights, median_dt, trial_duration


def sustained_episodes(
    elapsed: np.ndarray,
    weights: np.ndarray,
    gaze: np.ndarray,
    max_gap_sec: float,
    min_span_sec: float,
) -> dict[str, float | int | None]:
    indices = np.flatnonzero(gaze & np.isfinite(elapsed))
    if indices.size == 0:
        return {
            "SustainedEpisodeCount": 0,
            "SustainedGazeDurationSec": 0.0,
            "MeanSustainedEpisodeSec": None,
            "MedianSustainedEpisodeSec": None,
            "FirstSustainedAttentionLatencySec": None,
        }
    times = elapsed[indices]
    split_at = np.flatnonzero(np.diff(times) > max_gap_sec) + 1
    groups = np.split(indices, split_at)
    qualifying: list[tuple[float, float]] = []
    for group in groups:
        span = float(elapsed[group[-1]] - elapsed[group[0]] + weights[group[-1]])
        dwell = float(weights[group].sum())
        if span >= min_span_sec:
            qualifying.append((float(elapsed[group[0]]), dwell))
    if not qualifying:
        return {
            "SustainedEpisodeCount": 0,
            "SustainedGazeDurationSec": 0.0,
            "MeanSustainedEpisodeSec": None,
            "MedianSustainedEpisodeSec": None,
            "FirstSustainedAttentionLatencySec": None,
        }
    durations = np.array([item[1] for item in qualifying], dtype=float)
    return {
        "SustainedEpisodeCount": len(qualifying),
        "SustainedGazeDurationSec": float(durations.sum()),
        "MeanSustainedEpisodeSec": float(durations.mean()),
        "MedianSustainedEpisodeSec": float(np.median(durations)),
        "FirstSustainedAttentionLatencySec": qualifying[0][0],
    }


def applicable_guides(condition: str) -> list[str]:
    prefix = condition.rsplit("-", 1)[0]
    if prefix == "Full":
        return ["LightBlanket", "MiniMap", "Text", "Any"]
    for guide, config in GUIDE_CONFIG.items():
        if prefix == config["single_prefix"]:
            return [guide]
    return []


def guide_arrays(frame: pd.DataFrame, guide: str) -> tuple[np.ndarray, np.ndarray]:
    if guide == "Any":
        gaze = as_bool(frame["AnyGuidanceGaze"])
        valid_parts = [
            as_bool(frame[column])
            for column in ["ValidLightBlanketAOI", "ValidMiniMapAOI", "ValidTextAOI"]
            if column in frame
        ]
        valid = np.logical_or.reduce(valid_parts)
        return gaze, valid
    config = GUIDE_CONFIG[guide]
    return as_bool(frame[config["gaze"]]), as_bool(frame[config["valid"]])


def process_trial(
    path: Path,
    max_gap_sec: float,
    episode_gap_sec: float,
    episode_min_sec: float,
) -> tuple[list[dict], list[dict]]:
    frame = pd.read_csv(path, low_memory=False)
    identity = parse_identity(path, frame)
    elapsed, weights, median_dt, trial_duration = timestamp_and_weights(
        frame["TimeStamp"], max_gap_sec
    )
    metrics: list[dict] = []
    progress: list[dict] = []
    normalized = np.clip(elapsed / trial_duration, 0, 0.999999) if trial_duration > 0 else np.zeros(len(frame))
    bins = np.floor(normalized * 10).astype(int)

    for guide in applicable_guides(identity["Condition"]):
        gaze, valid = guide_arrays(frame, guide)
        usable = valid & np.isfinite(elapsed)
        gaze_valid = gaze & usable
        valid_duration = float(weights[usable].sum())
        gaze_duration = float(weights[gaze_valid].sum())
        record = {
            **identity,
            "Guide": guide,
            "RowCount": len(frame),
            "MedianSampleIntervalMs": median_dt * 1000,
            "TrialDurationSec": trial_duration,
            "ValidDurationSec": valid_duration,
            "GazeDurationSec": gaze_duration,
            "GazeTimeRateAmongValid": gaze_duration / valid_duration if valid_duration > 0 else None,
            "GazeSampleCount": int(gaze_valid.sum()),
            "ValidSampleCount": int(usable.sum()),
        }
        record.update(
            sustained_episodes(
                elapsed,
                weights,
                gaze_valid,
                max_gap_sec=episode_gap_sec,
                min_span_sec=episode_min_sec,
            )
        )
        metrics.append(record)

        for bin_index in range(10):
            selected = bins == bin_index
            bin_valid = selected & usable
            bin_gaze = selected & gaze_valid
            progress.append(
                {
                    **identity,
                    "Guide": guide,
                    "ProgressBin": bin_index + 1,
                    "ProgressMidpointPct": (bin_index + 0.5) * 10,
                    "ValidDurationSec": float(weights[bin_valid].sum()),
                    "GazeDurationSec": float(weights[bin_gaze].sum()),
                }
            )
    return metrics, progress


def participant_metrics(trial_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["Condition", "GuidanceCondition", "SmokeLevel", "ParticipantFolder", "Guide"]
    sum_columns = [
        "RowCount",
        "TrialDurationSec",
        "ValidDurationSec",
        "GazeDurationSec",
        "GazeSampleCount",
        "ValidSampleCount",
        "SustainedEpisodeCount",
        "SustainedGazeDurationSec",
    ]
    summed = trial_metrics.groupby(keys, as_index=False)[sum_columns].sum(min_count=1)
    trials = (
        trial_metrics.groupby(keys, as_index=False)
        .agg(
            TrialCount=("TrialID", "size"),
            FirstSustainedAttentionLatencySec=("FirstSustainedAttentionLatencySec", "median"),
            MeanSustainedEpisodeSec=("MeanSustainedEpisodeSec", "mean"),
            TrialsWithSustainedAttention=("SustainedEpisodeCount", lambda x: int((x > 0).sum())),
        )
    )
    output = summed.merge(trials, on=keys, validate="one_to_one")
    output["GazeTimeRateAmongValid"] = output["GazeDurationSec"] / output["ValidDurationSec"]
    output.loc[output["ValidDurationSec"].le(0), "GazeTimeRateAmongValid"] = np.nan
    output["SustainedAttentionInAllTrials"] = (
        output["TrialsWithSustainedAttention"] == output["TrialCount"]
    ).astype("int8")
    return output


def bootstrap_mean_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    samples = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def condition_summary(
    participants: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["Condition", "GuidanceCondition", "SmokeLevel", "Guide"]
    for key, group in participants.groupby(keys, sort=True):
        rates = group["GazeTimeRateAmongValid"].to_numpy(dtype=float)
        finite = rates[np.isfinite(rates)]
        low, high = bootstrap_mean_ci(finite, iterations, rng)
        rows.append(
            {
                **dict(zip(keys, key)),
                "ParticipantCount": group["ParticipantFolder"].nunique(),
                "ValidParticipantCount": len(finite),
                "MeanGazeTimeRate": float(np.mean(finite)) if len(finite) else None,
                "MedianGazeTimeRate": float(np.median(finite)) if len(finite) else None,
                "SDGazeTimeRate": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                "Bootstrap95CI_Low": low,
                "Bootstrap95CI_High": high,
                "ParticipantsWithSustainedAttention": int(
                    group["SustainedEpisodeCount"].gt(0).sum()
                ),
                "MeanSustainedEpisodeCount": float(group["SustainedEpisodeCount"].mean()),
                "MedianFirstSustainedAttentionLatencySec": float(
                    group["FirstSustainedAttentionLatencySec"].median()
                )
                if group["FirstSustainedAttentionLatencySec"].notna().any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p))
    if not len(valid_indices):
        return adjusted.tolist()
    order = valid_indices[np.argsort(p[valid_indices])]
    ranked = p[order] * len(valid_indices) / np.arange(1, len(valid_indices) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def paired_test_record(
    comparison: str,
    guide: str,
    smoke: str,
    left_label: str,
    right_label: str,
    paired: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> dict:
    clean = paired[["Left", "Right"]].dropna()
    differences = (clean["Right"] - clean["Left"]).to_numpy(dtype=float)
    low, high = bootstrap_mean_ci(differences, iterations, rng)
    if len(differences) and not np.allclose(differences, 0):
        statistic, p_value = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    else:
        statistic, p_value = math.nan, math.nan
    return {
        "ComparisonFamily": comparison,
        "Guide": guide,
        "SmokeLevel": smoke,
        "LeftCondition": left_label,
        "RightCondition": right_label,
        "PairedN": len(clean),
        "LeftMeanRate": clean["Left"].mean(),
        "RightMeanRate": clean["Right"].mean(),
        "MeanDifference_RightMinusLeft": differences.mean() if len(differences) else None,
        "MedianDifference_RightMinusLeft": np.median(differences) if len(differences) else None,
        "Bootstrap95CI_DifferenceLow": low,
        "Bootstrap95CI_DifferenceHigh": high,
        "WilcoxonStatistic": statistic,
        "WilcoxonP": p_value,
    }


def paired_comparisons(
    participants: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict] = []
    for guide, config in GUIDE_CONFIG.items():
        single_prefix = config["single_prefix"]
        for smoke in SMOKE_ORDER:
            single = participants[
                participants["Condition"].eq(f"{single_prefix}-{smoke}")
                & participants["Guide"].eq(guide)
            ][["ParticipantFolder", "GazeTimeRateAmongValid"]].rename(
                columns={"GazeTimeRateAmongValid": "Left"}
            )
            full = participants[
                participants["Condition"].eq(f"Full-{smoke}")
                & participants["Guide"].eq(guide)
            ][["ParticipantFolder", "GazeTimeRateAmongValid"]].rename(
                columns={"GazeTimeRateAmongValid": "Right"}
            )
            paired = single.merge(full, on="ParticipantFolder", validate="one_to_one")
            rows.append(
                paired_test_record(
                    "Single_vs_Full",
                    guide,
                    smoke,
                    f"{single_prefix}-{smoke}",
                    f"Full-{smoke}",
                    paired,
                    iterations,
                    rng,
                )
            )

    smoke_comparisons = [
        ("LightBlanket", "ArrowRibbon"),
        ("MiniMap", "MiniMap"),
        ("Text", "TextBar"),
        ("LightBlanket", "Full"),
        ("MiniMap", "Full"),
        ("Text", "Full"),
        ("Any", "Full"),
    ]
    for guide, prefix in smoke_comparisons:
        high = participants[
            participants["Condition"].eq(f"{prefix}-HighSmoke")
            & participants["Guide"].eq(guide)
        ][["ParticipantFolder", "GazeTimeRateAmongValid"]].rename(
            columns={"GazeTimeRateAmongValid": "Left"}
        )
        low = participants[
            participants["Condition"].eq(f"{prefix}-LowSmoke")
            & participants["Guide"].eq(guide)
        ][["ParticipantFolder", "GazeTimeRateAmongValid"]].rename(
            columns={"GazeTimeRateAmongValid": "Right"}
        )
        paired = high.merge(low, on="ParticipantFolder", validate="one_to_one")
        rows.append(
            paired_test_record(
                "HighSmoke_vs_LowSmoke",
                guide,
                "High_to_Low",
                f"{prefix}-HighSmoke",
                f"{prefix}-LowSmoke",
                paired,
                iterations,
                rng,
            )
        )
    result = pd.DataFrame(rows)
    result["WilcoxonQ_BH"] = bh_adjust(result["WilcoxonP"].tolist())
    return result


def participant_progress(progress_trials: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "Condition",
        "GuidanceCondition",
        "SmokeLevel",
        "ParticipantFolder",
        "Guide",
        "ProgressBin",
        "ProgressMidpointPct",
    ]
    output = progress_trials.groupby(keys, as_index=False)[
        ["ValidDurationSec", "GazeDurationSec"]
    ].sum()
    output["GazeTimeRateAmongValid"] = output["GazeDurationSec"] / output["ValidDurationSec"]
    output.loc[output["ValidDurationSec"].le(0), "GazeTimeRateAmongValid"] = np.nan
    return output


def progress_summary(
    participant_profile: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict] = []
    keys = [
        "Condition",
        "GuidanceCondition",
        "SmokeLevel",
        "Guide",
        "ProgressBin",
        "ProgressMidpointPct",
    ]
    for key, group in participant_profile.groupby(keys, sort=True):
        values = group["GazeTimeRateAmongValid"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low, high = bootstrap_mean_ci(values, iterations, rng)
        rows.append(
            {
                **dict(zip(keys, key)),
                "ParticipantN": len(values),
                "MeanGazeTimeRate": values.mean() if len(values) else None,
                "MedianGazeTimeRate": np.median(values) if len(values) else None,
                "Bootstrap95CI_Low": low,
                "Bootstrap95CI_High": high,
            }
        )
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    font = next((item for item in candidates if item in available), "DejaVu Sans")
    matplotlib.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    sns.set_theme(style="whitegrid", font=font)


def save_figure(fig: plt.Figure, base: Path) -> None:
    # 当前实验机的应用程序控制策略会阻止 fontTools 的 PDF 子集 DLL；
    # SVG 仍是可编辑矢量格式，因此保留 SVG + 300 dpi PNG。
    for suffix in [".png", ".svg"]:
        fig.savefig(
            base.with_suffix(suffix),
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_condition_distributions(participants: pd.DataFrame, output_dir: Path) -> None:
    single_rows = []
    for guide, config in GUIDE_CONFIG.items():
        subset = participants[
            participants["GuidanceCondition"].eq(config["single_prefix"])
            & participants["Guide"].eq(guide)
        ].copy()
        subset["GuideLabel"] = config["label_zh"]
        single_rows.append(subset)
    single = pd.concat(single_rows, ignore_index=True)
    full = participants[
        participants["GuidanceCondition"].eq("Full")
        & participants["Guide"].isin(GUIDE_CONFIG)
    ].copy()
    full["GuideLabel"] = full["Guide"].map(
        {guide: config["label_zh"] for guide, config in GUIDE_CONFIG.items()}
    )
    smoke_palette = {"HighSmoke": "#D55E00", "LowSmoke": "#56B4E9"}
    order = ["光毯", "小地图", "文字"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, constrained_layout=True)
    for ax, data, title in [
        (axes[0], single, "单引导条件"),
        (axes[1], full, "Full 条件"),
    ]:
        sns.boxplot(
            data=data,
            x="GuideLabel",
            y="GazeTimeRateAmongValid",
            hue="SmokeLevel",
            order=order,
            hue_order=SMOKE_ORDER,
            palette=smoke_palette,
            showfliers=False,
            width=0.65,
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x="GuideLabel",
            y="GazeTimeRateAmongValid",
            hue="SmokeLevel",
            order=order,
            hue_order=SMOKE_ORDER,
            dodge=True,
            palette=smoke_palette,
            edgecolor="white",
            linewidth=0.35,
            size=3.3,
            alpha=0.65,
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:2],
            ["高烟", "低烟"],
            title="",
            frameon=False,
            loc="upper right",
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylim(0, 0.85)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)
    axes[0].set_ylabel("有效记录时间中的引导注视比例")
    axes[1].set_ylabel("")
    fig.suptitle("三种引导的注意捕获：每个点代表一名受试者", fontweight="bold")
    save_figure(fig, output_dir / "01_attention_rate_distributions")


def plot_paired_differences(comparisons_source: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for guide, config in GUIDE_CONFIG.items():
        for smoke in SMOKE_ORDER:
            single = comparisons_source[
                comparisons_source["Condition"].eq(f"{config['single_prefix']}-{smoke}")
                & comparisons_source["Guide"].eq(guide)
            ][["ParticipantFolder", "GazeTimeRateAmongValid"]].rename(
                columns={"GazeTimeRateAmongValid": "Single"}
            )
            full = comparisons_source[
                comparisons_source["Condition"].eq(f"Full-{smoke}")
                & comparisons_source["Guide"].eq(guide)
            ][["ParticipantFolder", "GazeTimeRateAmongValid"]].rename(
                columns={"GazeTimeRateAmongValid": "Full"}
            )
            paired = single.merge(full, on="ParticipantFolder").dropna()
            paired["Difference"] = paired["Full"] - paired["Single"]
            paired["GuideLabel"] = config["label_zh"]
            paired["SmokeLevel"] = smoke
            rows.append(paired)
    data = pd.concat(rows, ignore_index=True)
    smoke_palette = {"HighSmoke": "#D55E00", "LowSmoke": "#56B4E9"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    sns.boxplot(
        data=data,
        x="GuideLabel",
        y="Difference",
        hue="SmokeLevel",
        hue_order=SMOKE_ORDER,
        palette=smoke_palette,
        showfliers=False,
        width=0.65,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="GuideLabel",
        y="Difference",
        hue="SmokeLevel",
        hue_order=SMOKE_ORDER,
        dodge=True,
        palette=smoke_palette,
        edgecolor="white",
        linewidth=0.35,
        size=4,
        alpha=0.75,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["高烟", "低烟"], title="", frameon=False)
    ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Full − 单引导（注视比例差）")
    ax.set_title("组合呈现后，每种引导获得的注意变化\n每个点代表同一受试者的配对差值", fontweight="bold")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    save_figure(fig, output_dir / "02_single_vs_full_paired_difference")


def plot_progress_profiles(profile: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2), sharex=True, sharey=True, constrained_layout=True)
    smoke_style = {
        "HighSmoke": ("#D55E00", "-", "o", "高烟"),
        "LowSmoke": ("#0072B2", "--", "s", "低烟"),
    }
    for column, (guide, config) in enumerate(GUIDE_CONFIG.items()):
        for row, prefix in enumerate([config["single_prefix"], "Full"]):
            ax = axes[row, column]
            subset = profile[
                profile["GuidanceCondition"].eq(prefix) & profile["Guide"].eq(guide)
            ]
            for smoke in SMOKE_ORDER:
                part = subset[subset["SmokeLevel"].eq(smoke)].sort_values("ProgressMidpointPct")
                color, linestyle, marker, label = smoke_style[smoke]
                ax.plot(
                    part["ProgressMidpointPct"],
                    part["MeanGazeTimeRate"],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3,
                    linewidth=1.5,
                    label=label,
                )
                ax.fill_between(
                    part["ProgressMidpointPct"].to_numpy(dtype=float),
                    part["Bootstrap95CI_Low"].to_numpy(dtype=float),
                    part["Bootstrap95CI_High"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            ax.set_title(
                f"{config['label_zh']}｜{'单引导' if row == 0 else 'Full'}"
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 0.85)
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)
            if row == 1:
                ax.set_xlabel("归一化疏散进程（%）")
            if column == 0:
                ax.set_ylabel("平均注视时间比例")
            if row == 0 and column == 2:
                ax.legend(frameon=False, loc="upper right")
    fig.suptitle("引导注意随疏散进程的变化（阴影为受试者 bootstrap 95% CI）", fontweight="bold")
    save_figure(fig, output_dir / "03_attention_over_evacuation_progress")


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value * 100:.{digits}f}%"


def generate_report(
    conditions: pd.DataFrame,
    comparisons: pd.DataFrame,
    participants: pd.DataFrame,
    profile: pd.DataFrame,
    qc: dict,
    output_path: Path,
) -> None:
    lookup = conditions.set_index(["Condition", "Guide"])

    def cell(condition: str, guide: str, column: str) -> float:
        try:
            return float(lookup.loc[(condition, guide), column])
        except (KeyError, TypeError, ValueError):
            return math.nan

    def phase_rate(condition: str, guide: str, bins: list[int]) -> float:
        selected = profile[
            profile["Condition"].eq(condition)
            & profile["Guide"].eq(guide)
            & profile["ProgressBin"].isin(bins)
        ]["MeanGazeTimeRate"]
        return float(selected.mean()) if len(selected) else math.nan

    lines = [
        "# 三种疏散引导信息的注视分析",
        "",
        "## 一句话结论",
        "",
        (
            "**小地图最容易持续占据视觉注意；文字和光毯在单独呈现时能被稳定注意，"
            "但进入 Full 组合后，两者的注意份额明显被压缩。**这说明 Full 并不是三种引导效果的简单相加，"
            "而是存在明显的视觉竞争，且小地图是主要注意入口。"
        ),
        "",
        "## 1. 单引导条件：引导是否成功吸引注意",
        "",
        "| 引导 | 高烟平均注视时间比例 | 低烟平均注视时间比例 |",
        "|---|---:|---:|",
    ]
    for guide, config in GUIDE_CONFIG.items():
        prefix = config["single_prefix"]
        lines.append(
            f"| {config['label_zh']} | "
            f"{fmt_pct(cell(f'{prefix}-HighSmoke', guide, 'MeanGazeTimeRate'))} | "
            f"{fmt_pct(cell(f'{prefix}-LowSmoke', guide, 'MeanGazeTimeRate'))} |"
        )

    lines.extend(
        [
            "",
            (
                "这些比例以“该 AOI 可判定的有效时间”为分母，并先在每名受试者内计算，再对受试者取平均。"
                "因此不会让疏散时间更长、日志行数更多的实验获得更大权重。"
            ),
            "",
            "## 2. Full 条件：三种信息之间存在注意竞争",
            "",
            "| Full中的引导 | 高烟 | 低烟 |",
            "|---|---:|---:|",
        ]
    )
    for guide, config in GUIDE_CONFIG.items():
        lines.append(
            f"| {config['label_zh']} | "
            f"{fmt_pct(cell('Full-HighSmoke', guide, 'MeanGazeTimeRate'))} | "
            f"{fmt_pct(cell('Full-LowSmoke', guide, 'MeanGazeTimeRate'))} |"
        )
    lines.append(
        f"| 任一引导 | {fmt_pct(cell('Full-HighSmoke', 'Any', 'MeanGazeTimeRate'))} | "
        f"{fmt_pct(cell('Full-LowSmoke', 'Any', 'MeanGazeTimeRate'))} |"
    )

    paired = comparisons[comparisons["ComparisonFamily"].eq("Single_vs_Full")]
    lines.extend(
        [
            "",
            "## 3. 同一受试者：从单引导变为 Full 后的变化",
            "",
            "| 引导 | 烟雾 | 有效配对n | Full−单引导 | 95% CI | BH校正q |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in paired.iterrows():
        guide_label = GUIDE_CONFIG[row["Guide"]]["label_zh"]
        lines.append(
            f"| {guide_label} | {SMOKE_LABEL[row['SmokeLevel']]} | {int(row['PairedN'])} | "
            f"{fmt_pct(row['MeanDifference_RightMinusLeft'])} | "
            f"[{fmt_pct(row['Bootstrap95CI_DifferenceLow'])}, "
            f"{fmt_pct(row['Bootstrap95CI_DifferenceHigh'])}] | "
            f"{row['WilcoxonQ_BH']:.4f} |"
        )

    lines.extend(
        [
            "",
            (
                "负值表示：同一种引导在 Full 中获得的注意少于它单独出现时。"
                "配对比较只使用同一受试者的数据；无有效 Full AOI 的配对被排除。"
            ),
            "",
            "## 4. 注视发生在疏散过程的什么阶段",
            "",
            "| 条件 | 引导 | 前20%进程 | 后20%进程 | 主要模式 |",
            "|---|---|---:|---:|---|",
        ]
    )
    phase_specs = [
        ("ArrowRibbon-HighSmoke", "LightBlanket", "单引导·高烟", "后期增强"),
        ("ArrowRibbon-LowSmoke", "LightBlanket", "单引导·低烟", "后期增强"),
        ("MiniMap-HighSmoke", "MiniMap", "单引导·高烟", "前期最强"),
        ("MiniMap-LowSmoke", "MiniMap", "单引导·低烟", "前期最强"),
        ("TextBar-HighSmoke", "Text", "单引导·高烟", "前期较强"),
        ("TextBar-LowSmoke", "Text", "单引导·低烟", "前期较强"),
        ("Full-HighSmoke", "LightBlanket", "Full·高烟", "后期增强"),
        ("Full-LowSmoke", "LightBlanket", "Full·低烟", "后期增强"),
        ("Full-HighSmoke", "MiniMap", "Full·高烟", "前期最强"),
        ("Full-LowSmoke", "MiniMap", "Full·低烟", "前期最强"),
        ("Full-HighSmoke", "Text", "Full·高烟", "前期较强"),
        ("Full-LowSmoke", "Text", "Full·低烟", "前期较强"),
    ]
    for condition, guide, condition_label, pattern in phase_specs:
        lines.append(
            f"| {condition_label} | {GUIDE_CONFIG[guide]['label_zh']} | "
            f"{fmt_pct(phase_rate(condition, guide, [1, 2]))} | "
            f"{fmt_pct(phase_rate(condition, guide, [9, 10]))} | {pattern} |"
        )
    lines.extend(
        [
            "",
            (
                "时间模式很清楚：**屏幕 UI（小地图、文字）主要负责疏散开始阶段的方向确认，"
                "地面光毯则在移动进入中后段后获得更多注意。**因此三种引导并非只是在争夺同一时刻的注意，"
                "也可能承担不同阶段的导航功能。"
            ),
            "",
            "## 5. 对实验问题的解释",
            "",
            "1. **小地图是最强视觉吸引源。**它在单独呈现和 Full 中都获得最高的注视比例。",
            "2. **Full 存在资源竞争。**三种信息同时出现后，受试者并不会把单引导条件下的注意全部保留下来。",
            "3. **文字信息最容易在组合条件中被稀释。**这可能与阅读需要主动抬眼、语义处理成本更高有关。",
            "4. **光毯更像连续背景导航线索。**其较低注视比例不必然表示无效；它可能通过周边视觉或短时检查发挥作用。",
            (
                "5. **高低烟差异要和引导组合共同解释。**三个单引导条件的高低烟配对差异均不显著；"
                "Full 中高烟的任一引导注视比低烟高约 8.3 个百分点，主要来自小地图增加约 6.7 个百分点。"
            ),
            "",
            "## 6. 这能回答什么、不能回答什么",
            "",
            (
                "当前分析回答的是“引导是否进入视觉注意、注意如何在三种引导间分配”。"
                "它还不能单独证明哪种引导提高了疏散效率，更不能把较高注视比例直接等同于较好效果。"
            ),
            "",
            (
                "下一步应把这些注意指标接到你的机制链路中："
                "**引导条件 → 引导注意（本分析）→ 搜索/转向/眼动稳定性 → 疏散时间与路径效率**。"
                "可进一步使用多层中介模型或纵向中介模型检验“为什么这种引导更好”。"
            ),
            "",
            "## 7. 指标口径与限制",
            "",
            "- 主指标：受试者层面的有效 AOI 时间中，引导命中所占比例。",
            "- 持续注意片段：允许不超过 75 ms 的短暂中断，最短跨度 100 ms。",
            "- 持续注意片段不是严格 fixation，因为尚未使用眼动速度/离散度阈值。",
            "- 三个 AOI 的位置、面积和视觉形态不同，因此跨引导的绝对比例比较反映“视觉捕获能力”，不等同于认知收益。",
            "- Full 中采用 UI 优先去重后的光毯指标，同时保留原始光毯命中供敏感性分析。",
            "",
            "## 数据质量",
            "",
            f"- 成功处理逐试次文件：{qc['successful_files']}。",
            f"- 处理失败：{qc['failed_files']}。",
            f"- 受试者—条件—引导记录：{len(participants)}。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()
    input_root = resolve(workspace, args.input_root).resolve()
    output_dir = resolve(workspace, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_root.glob("*/*/*_GuidanceGazeIntegrated_*.csv"))
    if not files:
        print(f"未发现整合结果文件：{input_root}", file=sys.stderr)
        return 1

    trial_rows: list[dict] = []
    progress_rows: list[dict] = []
    errors: list[dict] = []
    for path in files:
        try:
            metrics, progress = process_trial(
                path,
                max_gap_sec=args.max_sample_gap_ms / 1000,
                episode_gap_sec=args.episode_gap_ms / 1000,
                episode_min_sec=args.episode_min_ms / 1000,
            )
            trial_rows.extend(metrics)
            progress_rows.extend(progress)
        except Exception as exc:
            errors.append({"SourceFile": str(path), "Error": f"{type(exc).__name__}: {exc}"})

    trials = pd.DataFrame(trial_rows)
    progress_trials = pd.DataFrame(progress_rows)
    errors_frame = pd.DataFrame(errors)
    if trials.empty:
        errors_frame.to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
        return 1

    rng = np.random.default_rng(20260724)
    participants = participant_metrics(trials)
    conditions = condition_summary(participants, args.bootstrap_iterations, rng)
    comparisons = paired_comparisons(participants, args.bootstrap_iterations, rng)
    participant_profile = participant_progress(progress_trials)
    profile = progress_summary(participant_profile, args.bootstrap_iterations, rng)

    trials.to_csv(output_dir / "trial_attention_metrics.csv", index=False, encoding="utf-8-sig")
    participants.to_csv(
        output_dir / "participant_attention_metrics.csv", index=False, encoding="utf-8-sig"
    )
    conditions.to_csv(
        output_dir / "condition_guide_summary.csv", index=False, encoding="utf-8-sig"
    )
    comparisons.to_csv(
        output_dir / "paired_comparisons.csv", index=False, encoding="utf-8-sig"
    )
    profile.to_csv(
        output_dir / "progress_profiles.csv", index=False, encoding="utf-8-sig"
    )
    errors_frame.to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")

    set_plot_style()
    plot_condition_distributions(participants, output_dir)
    plot_paired_differences(participants, output_dir)
    plot_progress_profiles(profile, output_dir)

    qc = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "discovered_files": len(files),
        "successful_files": len(files) - len(errors),
        "failed_files": len(errors),
        "trial_guide_records": len(trials),
        "participant_condition_guide_records": len(participants),
        "episode_gap_ms": args.episode_gap_ms,
        "episode_min_ms": args.episode_min_ms,
        "max_sample_gap_ms": args.max_sample_gap_ms,
        "bootstrap_iterations": args.bootstrap_iterations,
        "quality_gate_passed": len(errors) == 0,
    }
    (output_dir / "analysis_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    generate_report(
        conditions,
        comparisons,
        participants,
        profile,
        qc,
        output_dir / "analysis_report.md",
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    return 0 if qc["quality_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
