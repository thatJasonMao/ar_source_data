#!/usr/bin/env python
"""整合光毯、小地图和文字引导的逐时刻眼动 AOI 结果。

输出规则
--------
1. ArrowRibbon-H/L：只输出光毯注视信息。
2. MiniMap-H/L：只输出小地图注视信息。
3. TextBar-H/L：只输出文字引导注视信息。
4. Full-H/L：在同一个逐时刻文件中输出三种引导信息。

Full 组同时保留三种原始命中，并给出推荐的互斥主口径：
若视线先穿过无碰撞体的 UI（小地图或文字），则不把其后方地面的命中重复计作光毯。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LIGHT_CONDITIONS = {
    "ArrowRibbon-HighSmoke",
    "ArrowRibbon-LowSmoke",
    "Full-HighSmoke",
    "Full-LowSmoke",
}
UI_CONDITIONS = {
    "MiniMap-HighSmoke",
    "MiniMap-LowSmoke",
    "TextBar-HighSmoke",
    "TextBar-LowSmoke",
    "Full-HighSmoke",
    "Full-LowSmoke",
}
TARGET_CONDITIONS = LIGHT_CONDITIONS | UI_CONDITIONS

LIGHT_PATTERN = re.compile(
    r"^(?P<trial>.+)_LightBlanketGazeAOI_(?P<stamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.csv$"
)
UI_PATTERN = re.compile(
    r"^(?P<trial>.+)_MinimapTextGazeAOI_(?P<stamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.csv$"
)

IDENTITY_COLUMNS = [
    "OriginalRowIndex",
    "TimeStamp",
    "Condition",
    "GuidanceType",
    "SmokeLevel",
    "ParticipantFolder",
    "TrialID",
]

LIGHT_BASE_COLUMNS = IDENTITY_COLUMNS + [
    "Hit1_Name",
    "Hit1_X",
    "Hit1_Y",
    "Hit1_Z",
]

UI_BASE_COLUMNS = IDENTITY_COLUMNS + [
    "SubjectPosX",
    "SubjectPosY",
    "SubjectPosZ",
    "SubjectRotY",
    "GazeDirX",
    "GazeDirY",
    "GazeDirZ",
    "Hit1_X",
    "Hit1_Y",
    "Hit1_Z",
    "Hit1_Name",
    "ValidSubjectPose",
    "ValidGazeDirection",
    "ValidHit1",
    "ValidCanvasIntersection",
    "RayAnchorSource",
    "CanvasIntersectionX",
    "CanvasIntersectionY",
    "CanvasIntersectionZ",
    "CanvasLocalX_M",
    "CanvasLocalY_M",
    "CanvasLocalZ_M",
    "EyeToCanvasDistanceApproxM",
    "AOIMarginM",
]

LIGHT_DETAIL_COLUMNS = [
    "PathAvailable",
    "PathMode",
    "PathBeforeIndex",
    "PathAfterIndex",
    "PathInterpolationAlpha",
    "NearestPathSnapshotAgeMs",
    "DistanceToCarpetCenterXZ_M",
    "DistanceToCarpetCenter3D_M",
    "VerticalDifferenceM",
    "AccuracyMarginM",
    "CoreThresholdM",
    "ExpandedThresholdM",
    "GazeOnLightBlanketCore",
    "GazeOnLightBlanket",
    "AOIConfidence",
    "AOIStatus",
]

MINIMAP_DETAIL_COLUMNS = [
    "MiniMapVisible",
    "GazeOnMiniMapCore",
    "GazeOnMiniMap",
    "UIAOILabel",
]

TEXT_DETAIL_COLUMNS = [
    "TextVisible",
    "DistanceTextContentVisible",
    "GazeOnTextDirectionCore",
    "GazeOnTextDirection",
    "GazeOnTextDistancePanelCore",
    "GazeOnTextDistancePanel",
    "GazeOnTextDistanceContentCore",
    "GazeOnTextDistanceContent",
    "GazeOnTextAnyPanelCore",
    "GazeOnTextAnyPanel",
    "GazeOnTextGuidanceCore",
    "GazeOnTextGuidance",
    "UIAOILabel",
]


@dataclass(frozen=True)
class TrialKey:
    condition: str
    participant: str
    trial: str
    stamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--light-results",
        type=Path,
        default=Path("Results_LightBlanket_GazeAOI"),
        help="光毯 AOI 结果目录（相对路径按 data-root 解析）",
    )
    parser.add_argument(
        "--ui-results",
        type=Path,
        default=Path("Results_MinimapText_GazeAOI"),
        help="小地图/文字 AOI 结果目录（相对路径按 data-root 解析）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results_Guidance_GazeAOI_Integrated"),
        help="整合结果目录（相对路径按 data-root 解析）",
    )
    return parser.parse_args()


def resolve_under(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def discover_files(root: Path, pattern: re.Pattern[str], kind: str) -> tuple[dict[TrialKey, Path], list[str]]:
    found: dict[TrialKey, Path] = {}
    issues: list[str] = []
    if not root.exists():
        issues.append(f"{kind}结果目录不存在：{root}")
        return found, issues

    for path in sorted(root.glob("*/*/*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            participant = path.parent.name
            condition = path.parent.parent.name
        except IndexError:
            continue
        if condition not in TARGET_CONDITIONS:
            continue
        key = TrialKey(condition, participant, match.group("trial"), match.group("stamp"))
        if key in found:
            issues.append(f"{kind}重复试次键：{key} -> {found[key]} | {path}")
        else:
            found[key] = path
    return found, issues


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} 缺少列：{missing}")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def validate_identity(frame: pd.DataFrame, key: TrialKey, source: Path) -> None:
    require_columns(frame, IDENTITY_COLUMNS, source)
    if frame.empty:
        raise ValueError(f"{source} 没有数据行")
    if frame["OriginalRowIndex"].isna().any():
        raise ValueError(f"{source} 的 OriginalRowIndex 存在缺失")
    if frame["OriginalRowIndex"].duplicated().any():
        count = int(frame["OriginalRowIndex"].duplicated(keep=False).sum())
        raise ValueError(f"{source} 的 OriginalRowIndex 不唯一，涉及 {count} 行")
    for column, expected in [
        ("Condition", key.condition),
        ("ParticipantFolder", key.participant),
        ("TrialID", key.trial),
    ]:
        values = set(frame[column].dropna().astype(str).unique())
        if values != {expected}:
            raise ValueError(f"{source} 的 {column}={sorted(values)}，与路径键 {expected!r} 不一致")


def build_light_only(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = LIGHT_BASE_COLUMNS + ["ValidEyeHit"] + LIGHT_DETAIL_COLUMNS
    require_columns(frame, required, source)
    output = frame[LIGHT_BASE_COLUMNS + LIGHT_DETAIL_COLUMNS].copy()
    output.insert(
        output.columns.get_loc("PathAvailable"),
        "ValidLightBlanketAOI",
        (as_bool(frame["ValidEyeHit"]) & as_bool(frame["PathAvailable"])).astype("int8"),
    )
    return output


def build_minimap_only(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = UI_BASE_COLUMNS + MINIMAP_DETAIL_COLUMNS
    require_columns(frame, required, source)
    output = frame[required].copy()
    output.insert(
        output.columns.get_loc("MiniMapVisible"),
        "ValidMiniMapAOI",
        (as_bool(frame["ValidCanvasIntersection"]) & as_bool(frame["MiniMapVisible"])).astype("int8"),
    )
    return output


def build_text_only(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = UI_BASE_COLUMNS + TEXT_DETAIL_COLUMNS
    require_columns(frame, required, source)
    output = frame[required].copy()
    output.insert(
        output.columns.get_loc("TextVisible"),
        "ValidTextAOI",
        (as_bool(frame["ValidCanvasIntersection"]) & as_bool(frame["TextVisible"])).astype("int8"),
    )
    return output


def raw_label(light: bool, minimap: bool, text: bool) -> str:
    labels = []
    if light:
        labels.append("LightBlanket")
    if minimap:
        labels.append("MiniMap")
    if text:
        labels.append("Text")
    return "+".join(labels) if labels else "None"


def exclusive_label(valid_any: bool, light: bool, minimap: bool, text: bool) -> str:
    if not valid_any:
        return "Unavailable"
    if minimap and text:
        return "MiniMap+TextOverlap"
    if minimap:
        return "MiniMap"
    if text:
        return "Text"
    if light:
        return "LightBlanket"
    return "None"


def build_full(light: pd.DataFrame, ui: pd.DataFrame, light_source: Path, ui_source: Path) -> pd.DataFrame:
    light_required = IDENTITY_COLUMNS + ["ValidEyeHit"] + LIGHT_DETAIL_COLUMNS
    ui_required = UI_BASE_COLUMNS + MINIMAP_DETAIL_COLUMNS[:-1] + TEXT_DETAIL_COLUMNS
    require_columns(light, light_required, light_source)
    require_columns(ui, ui_required, ui_source)

    if len(light) != len(ui):
        raise ValueError(f"Full 组来源行数不一致：光毯 {len(light)} 行，UI {len(ui)} 行")

    light_selected = light[light_required].copy()
    rename = {
        "TimeStamp": "TimeStamp_Light",
        "Condition": "Condition_Light",
        "GuidanceType": "GuidanceType_Light",
        "SmokeLevel": "SmokeLevel_Light",
        "ParticipantFolder": "ParticipantFolder_Light",
        "TrialID": "TrialID_Light",
        "GazeOnLightBlanketCore": "GazeOnLightBlanketCoreRaw",
        "GazeOnLightBlanket": "GazeOnLightBlanketRaw",
        "AOIConfidence": "LightBlanketAOIConfidence",
        "AOIStatus": "LightBlanketAOIStatus",
    }
    light_selected = light_selected.rename(columns=rename)

    merged = ui.merge(
        light_selected,
        on="OriginalRowIndex",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        raise ValueError(f"Full 组 OriginalRowIndex 不能一一匹配：{int(unmatched.sum())} 行")
    merged = merged.drop(columns="_merge")

    identity_pairs = [
        ("TimeStamp", "TimeStamp_Light"),
        ("Condition", "Condition_Light"),
        ("GuidanceType", "GuidanceType_Light"),
        ("SmokeLevel", "SmokeLevel_Light"),
        ("ParticipantFolder", "ParticipantFolder_Light"),
        ("TrialID", "TrialID_Light"),
    ]
    mismatches: dict[str, int] = {}
    for ui_column, light_column in identity_pairs:
        left = merged[ui_column]
        right = merged[light_column]
        if ui_column == "TimeStamp":
            left_num = pd.to_numeric(left, errors="coerce")
            right_num = pd.to_numeric(right, errors="coerce")
            mismatch = ~np.isclose(left_num, right_num, rtol=0.0, atol=1e-9, equal_nan=True)
        else:
            mismatch = left.fillna("<NA>").astype(str).ne(right.fillna("<NA>").astype(str))
        count = int(np.asarray(mismatch).sum())
        if count:
            mismatches[ui_column] = count
    if mismatches:
        raise ValueError(f"Full 组两个来源的身份/时间列不一致：{mismatches}")
    merged = merged.drop(columns=[column for _, column in identity_pairs])

    valid_light = as_bool(merged["ValidEyeHit"]) & as_bool(merged["PathAvailable"])
    valid_map = as_bool(merged["ValidCanvasIntersection"]) & as_bool(merged["MiniMapVisible"])
    valid_text = as_bool(merged["ValidCanvasIntersection"]) & as_bool(merged["TextVisible"])
    light_raw = as_bool(merged["GazeOnLightBlanketRaw"])
    minimap = as_bool(merged["GazeOnMiniMap"])
    text = as_bool(merged["GazeOnTextGuidance"])
    light_exclusive = light_raw & ~minimap & ~text
    valid_any = valid_light | valid_map | valid_text

    merged["ValidLightBlanketAOI"] = valid_light.astype("int8")
    merged["ValidMiniMapAOI"] = valid_map.astype("int8")
    merged["ValidTextAOI"] = valid_text.astype("int8")
    merged["GazeOnLightBlanket"] = light_exclusive.astype("int8")
    merged["UIOverlapSuppressesLightBlanket"] = (light_raw & (minimap | text)).astype("int8")
    merged["RawGuideMatchCount"] = (
        light_raw.astype("int8") + minimap.astype("int8") + text.astype("int8")
    ).astype("int8")
    merged["ExclusiveGuideMatchCount"] = (
        light_exclusive.astype("int8") + minimap.astype("int8") + text.astype("int8")
    ).astype("int8")
    merged["AnyGuidanceGaze"] = (light_exclusive | minimap | text).astype("int8")
    merged["RawGuidanceAOILabel"] = [
        raw_label(a, b, c) for a, b, c in zip(light_raw, minimap, text)
    ]
    merged["GuidanceAOILabel"] = [
        exclusive_label(v, a, b, c)
        for v, a, b, c in zip(valid_any, light_exclusive, minimap, text)
    ]

    light_details_full = [
        "ValidLightBlanketAOI",
        "PathAvailable",
        "PathMode",
        "PathBeforeIndex",
        "PathAfterIndex",
        "PathInterpolationAlpha",
        "NearestPathSnapshotAgeMs",
        "DistanceToCarpetCenterXZ_M",
        "DistanceToCarpetCenter3D_M",
        "VerticalDifferenceM",
        "AccuracyMarginM",
        "CoreThresholdM",
        "ExpandedThresholdM",
        "GazeOnLightBlanketCoreRaw",
        "GazeOnLightBlanketRaw",
        "GazeOnLightBlanket",
        "LightBlanketAOIConfidence",
        "LightBlanketAOIStatus",
    ]
    minimap_details_full = [
        "ValidMiniMapAOI",
        "MiniMapVisible",
        "GazeOnMiniMapCore",
        "GazeOnMiniMap",
    ]
    text_details_full = [
        "ValidTextAOI",
        "TextVisible",
        "DistanceTextContentVisible",
        "GazeOnTextDirectionCore",
        "GazeOnTextDirection",
        "GazeOnTextDistancePanelCore",
        "GazeOnTextDistancePanel",
        "GazeOnTextDistanceContentCore",
        "GazeOnTextDistanceContent",
        "GazeOnTextAnyPanelCore",
        "GazeOnTextAnyPanel",
        "GazeOnTextGuidanceCore",
        "GazeOnTextGuidance",
        "UIAOILabel",
    ]
    combined = [
        "UIOverlapSuppressesLightBlanket",
        "RawGuideMatchCount",
        "ExclusiveGuideMatchCount",
        "AnyGuidanceGaze",
        "RawGuidanceAOILabel",
        "GuidanceAOILabel",
    ]
    output_columns = UI_BASE_COLUMNS + light_details_full + minimap_details_full + text_details_full + combined
    return merged[output_columns].sort_values("OriginalRowIndex").reset_index(drop=True)


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_condition_summary(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    count_columns = [
        "RowCount",
        "LightBlanketValidRows",
        "LightBlanketGazeRows",
        "MiniMapValidRows",
        "MiniMapGazeRows",
        "TextGuidanceValidRows",
        "TextGuidanceGazeRows",
        "LightBlanketRawGazeRows",
        "LightBlanketUISuppressedRows",
        "AnyGuidanceGazeRows",
        "MiniMapTextOverlapRows",
    ]
    rows: list[dict[str, object]] = []
    for condition, group in trials.groupby("Condition", sort=True):
        record: dict[str, object] = {
            "Condition": condition,
            "ExperimentCount": len(group),
        }
        for column in count_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            record[column] = values.sum(min_count=1)
        rate_specs = [
            ("LightBlanketGazeRateAmongValid", "LightBlanketGazeRows", "LightBlanketValidRows"),
            ("MiniMapGazeRateAmongValid", "MiniMapGazeRows", "MiniMapValidRows"),
            ("TextGuidanceGazeRateAmongValid", "TextGuidanceGazeRows", "TextGuidanceValidRows"),
            ("AnyGuidanceGazeRateAmongRows", "AnyGuidanceGazeRows", "RowCount"),
            (
                "LightBlanketUISuppressedRateAmongRaw",
                "LightBlanketUISuppressedRows",
                "LightBlanketRawGazeRows",
            ),
            ("MiniMapTextOverlapRateAmongRows", "MiniMapTextOverlapRows", "RowCount"),
        ]
        for output_column, numerator_column, denominator_column in rate_specs:
            numerator = record[numerator_column]
            denominator = record[denominator_column]
            record[output_column] = (
                float(numerator) / float(denominator)
                if pd.notna(numerator) and pd.notna(denominator) and float(denominator) != 0
                else None
            )
        rows.append(record)
    return pd.DataFrame(rows)


def trial_summary(
    key: TrialKey,
    output: pd.DataFrame,
    output_path: Path,
    light_path: Path | None,
    ui_path: Path | None,
) -> dict:
    record: dict[str, object] = {
        "Condition": key.condition,
        "GuidanceType": key.condition.rsplit("-", 1)[0],
        "SmokeLevel": key.condition.rsplit("-", 1)[1],
        "ParticipantFolder": key.participant,
        "TrialID": key.trial,
        "ExperimentStamp": key.stamp,
        "RowCount": len(output),
        "SourceLightFile": str(light_path) if light_path else "",
        "SourceUIFile": str(ui_path) if ui_path else "",
        "OutputFile": str(output_path),
        "TimeKeyIntegrity": True,
    }

    metric_pairs = [
        ("ValidLightBlanketAOI", "GazeOnLightBlanket"),
        ("ValidMiniMapAOI", "GazeOnMiniMap"),
        ("ValidTextAOI", "GazeOnTextGuidance"),
    ]
    for valid_column, gaze_column in metric_pairs:
        prefix = gaze_column.removeprefix("GazeOn")
        if valid_column in output and gaze_column in output:
            valid = as_bool(output[valid_column])
            gaze = as_bool(output[gaze_column])
            valid_count = int(valid.sum())
            gaze_count = int((valid & gaze).sum())
            record[f"{prefix}ValidRows"] = valid_count
            record[f"{prefix}GazeRows"] = gaze_count
            record[f"{prefix}GazeRateAmongValid"] = safe_rate(gaze_count, valid_count)
        else:
            record[f"{prefix}ValidRows"] = None
            record[f"{prefix}GazeRows"] = None
            record[f"{prefix}GazeRateAmongValid"] = None

    if "GazeOnLightBlanketRaw" in output:
        record["LightBlanketRawGazeRows"] = int(as_bool(output["GazeOnLightBlanketRaw"]).sum())
        record["LightBlanketUISuppressedRows"] = int(
            as_bool(output["UIOverlapSuppressesLightBlanket"]).sum()
        )
    else:
        record["LightBlanketRawGazeRows"] = None
        record["LightBlanketUISuppressedRows"] = None

    if "AnyGuidanceGaze" in output:
        record["AnyGuidanceGazeRows"] = int(as_bool(output["AnyGuidanceGaze"]).sum())
        record["MiniMapTextOverlapRows"] = int(
            output["GuidanceAOILabel"].astype(str).eq("MiniMap+TextOverlap").sum()
        )
    else:
        relevant = [
            column
            for column in ["GazeOnLightBlanket", "GazeOnMiniMap", "GazeOnTextGuidance"]
            if column in output
        ]
        record["AnyGuidanceGazeRows"] = (
            int(np.logical_or.reduce([as_bool(output[column]).to_numpy() for column in relevant]).sum())
            if relevant
            else 0
        )
        record["MiniMapTextOverlapRows"] = None
    return record


def data_contract() -> dict:
    return {
        "version": "1.0",
        "row_granularity": "每行对应原眼动日志的一条记录；OriginalRowIndex 与 TimeStamp 是逐行追溯键。",
        "condition_output_contract": {
            "ArrowRibbon-HighSmoke/LowSmoke": "只包含光毯 AOI 字段。",
            "MiniMap-HighSmoke/LowSmoke": "只包含小地图 AOI 字段。",
            "TextBar-HighSmoke/LowSmoke": "只包含文字引导 AOI 字段。",
            "Full-HighSmoke/LowSmoke": "同一文件包含光毯、小地图、文字引导及三者整合字段。",
        },
        "recommended_main_fields": {
            "ArrowRibbon-HighSmoke/LowSmoke": "GazeOnLightBlanket",
            "MiniMap-HighSmoke/LowSmoke": "GazeOnMiniMap",
            "TextBar-HighSmoke/LowSmoke": "GazeOnTextGuidance",
            "Full-HighSmoke/LowSmoke": "GuidanceAOILabel；二元总指标使用 AnyGuidanceGaze。",
        },
        "full_group_logic": {
            "GazeOnLightBlanketRaw": "不考虑前景 UI 时，眼动射线在地面上的落点是否命中光毯。",
            "GazeOnLightBlanket": (
                "推荐互斥口径。若同一射线命中小地图或文字 UI，则不重复计为光毯；"
                "即 RawLightBlanket AND NOT MiniMap AND NOT TextGuidance。"
            ),
            "GazeOnMiniMap": "是否命中小地图 AOI。",
            "GazeOnTextGuidance": "是否命中文字引导的有效内容 AOI（推荐文字口径）。",
            "GazeOnTextAnyPanel": "是否命中文字 UI 的任一面板，可用于敏感性分析。",
            "UIOverlapSuppressesLightBlanket": "原始光毯命中因前景 UI 命中而被去重。",
            "RawGuidanceAOILabel": "三种原始命中的组合标签，允许重叠。",
            "GuidanceAOILabel": (
                "推荐主标签：None、LightBlanket、MiniMap、Text、"
                "MiniMap+TextOverlap 或 Unavailable。"
            ),
            "AnyGuidanceGaze": "互斥光毯、小地图或文字至少一个为真。",
        },
        "important_caveat": (
            "这些列是逐采样点 AOI 命中，不等同于一次完整注视事件。"
            "计算注视次数/持续时间前，应再按时间连续性、最短持续时间和采样缺口聚合。"
        ),
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    light_root = resolve_under(data_root, args.light_results).resolve()
    ui_root = resolve_under(data_root, args.ui_results).resolve()
    output_root = resolve_under(data_root, args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    light_files, light_issues = discover_files(light_root, LIGHT_PATTERN, "光毯")
    ui_files, ui_issues = discover_files(ui_root, UI_PATTERN, "UI")
    discovery_issues = light_issues + ui_issues
    if not light_files:
        discovery_issues.append(f"未在 {light_root} 发现光毯逐试次结果")
    if not ui_files:
        discovery_issues.append(f"未在 {ui_root} 发现小地图/文字逐试次结果")

    expected_keys = sorted(
        set(light_files) | set(ui_files),
        key=lambda k: (k.condition, k.participant, k.trial, k.stamp),
    )
    summaries: list[dict] = []
    failures: list[dict] = []
    condition_successes: dict[str, int] = {}
    condition_rows: dict[str, int] = {}

    for key in expected_keys:
        light_path = light_files.get(key)
        ui_path = ui_files.get(key)
        try:
            if key.condition.startswith("ArrowRibbon-"):
                if light_path is None:
                    raise ValueError("单光毯组缺少光毯 AOI 来源文件")
                light = read_csv(light_path)
                validate_identity(light, key, light_path)
                output = build_light_only(light, light_path)
            elif key.condition.startswith("MiniMap-"):
                if ui_path is None:
                    raise ValueError("单小地图组缺少 UI AOI 来源文件")
                ui = read_csv(ui_path)
                validate_identity(ui, key, ui_path)
                output = build_minimap_only(ui, ui_path)
            elif key.condition.startswith("TextBar-"):
                if ui_path is None:
                    raise ValueError("单文字组缺少 UI AOI 来源文件")
                ui = read_csv(ui_path)
                validate_identity(ui, key, ui_path)
                output = build_text_only(ui, ui_path)
            elif key.condition.startswith("Full-"):
                if light_path is None or ui_path is None:
                    raise ValueError(
                        f"Full 组来源不完整：light={light_path is not None}, ui={ui_path is not None}"
                    )
                light = read_csv(light_path)
                ui = read_csv(ui_path)
                validate_identity(light, key, light_path)
                validate_identity(ui, key, ui_path)
                output = build_full(light, ui, light_path, ui_path)
            else:
                continue

            output_dir = output_root / key.condition / key.participant
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{key.trial}_GuidanceGazeIntegrated_{key.stamp}.csv"
            output.to_csv(output_path, index=False, encoding="utf-8-sig")
            summaries.append(trial_summary(key, output, output_path, light_path, ui_path))
            condition_successes[key.condition] = condition_successes.get(key.condition, 0) + 1
            condition_rows[key.condition] = condition_rows.get(key.condition, 0) + len(output)
        except Exception as exc:  # 继续处理其余试次，并在 errors.csv 汇总
            failures.append(
                {
                    "Condition": key.condition,
                    "ParticipantFolder": key.participant,
                    "TrialID": key.trial,
                    "ExperimentStamp": key.stamp,
                    "SourceLightFile": str(light_path) if light_path else "",
                    "SourceUIFile": str(ui_path) if ui_path else "",
                    "Error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            ["Condition", "ParticipantFolder", "TrialID", "ExperimentStamp"]
        )
    summary_frame.to_csv(output_root / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    build_condition_summary(summary_frame).to_csv(
        output_root / "condition_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(failures).to_csv(output_root / "errors.csv", index=False, encoding="utf-8-sig")

    with (output_root / "data_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(data_contract(), handle, ensure_ascii=False, indent=2)

    qc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "light_results_root": str(light_root),
        "ui_results_root": str(ui_root),
        "output_root": str(output_root),
        "discovered_light_trials": len(light_files),
        "discovered_ui_trials": len(ui_files),
        "candidate_union_trials": len(expected_keys),
        "successful_trials": len(summaries),
        "failed_trials": len(failures),
        "successful_trials_by_condition": dict(sorted(condition_successes.items())),
        "output_rows_by_condition": dict(sorted(condition_rows.items())),
        "total_output_rows": int(sum(condition_rows.values())),
        "discovery_issues": discovery_issues,
        "quality_gate_passed": not failures and not discovery_issues,
    }
    with (output_root / "batch_qc.json").open("w", encoding="utf-8") as handle:
        json.dump(qc, handle, ensure_ascii=False, indent=2)

    print(json.dumps(qc, ensure_ascii=False, indent=2))
    return 0 if qc["quality_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
