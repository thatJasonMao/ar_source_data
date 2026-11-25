"""Incremental space-time extension of the reviewed multimodal analysis.

Only reads raw sources and existing outputs. All new or replaced outputs stay
inside ``Results_Guidance_Spatial_Causal_Analysis``.
"""

from __future__ import annotations

import json
import re
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from Extract_HRV_Time_Features import load_distance_fields, parse_time, temporal_subsample
from Extract_Visual_Information_Features import (
    aoi_columns,
    derive_eye_arrays,
    integrated_path,
    metrics_for_selection,
    read_eye,
    timestamps,
    to_bool,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DOC = ROOT / "Documentation_Guidance_Spatial_Causal_Analysis"
DATA = OUT / "data"
TABLE = OUT / "tables"
FIG = OUT / "figures"
NOTE = OUT / "notes"
ZONE_DIR = OUT / "zone_annotation"
for directory in (DATA, TABLE, FIG, NOTE, DOC):
    directory.mkdir(parents=True, exist_ok=True)

TIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
KEYS = ["Scenario", "ParticipantID", "FileName"]
GUIDANCE_ORDER = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
PALETTE = {
    "StaticSign": "#6B7280", "ArrowRibbon": "#0072B2", "MiniMap": "#009E73",
    "TextBar": "#D55E00", "Full": "#CC79A7",
}
SEED = 20260804
RNG = np.random.default_rng(SEED)
BOOT_N = 300
BIN_N = 10
POSITION_TOLERANCE_MS = 120
DEBOUNCE_S = 0.60

VISUAL_METRICS = [
    "HeadAngularPathRate_deg_s", "FixationTimeProportion", "GazeDirectionEntropy",
]
HRV_METRICS = ["MeanHRBaselineDelta_bpm", "LogRMSSDBaselineDelta"]
MOVEMENT_METRICS = [
    "MeanSpeed_m_s", "MovementTurnPathRate_deg_s", "BacktrackIncrement_m",
    "NonProgressPathRatio",
]
AOI_METRICS = ["GuideDwellProportion"]

METRIC_LABELS = {
    "HeadAngularPathRate_deg_s": "头部搜索路径率",
    "FixationTimeProportion": "凝视时间比例",
    "GazeDirectionEntropy": "注视方向熵",
    "MeanHRBaselineDelta_bpm": "相对静息心率变化",
    "LogRMSSDBaselineDelta": "相对静息log(RMSSD)变化",
    "MeanSpeed_m_s": "平均速度",
    "MovementTurnPathRate_deg_s": "转向路径率",
    "BacktrackIncrement_m": "回退增量",
    "NonProgressPathRatio": "非前进路径比例",
    "GuideDwellProportion": "引导注视比例",
}

mpl.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=x.index)


def zone_metadata(zones: pd.DataFrame) -> dict[str, tuple[str, str]]:
    return {
        str(row.zone_id): (str(row.zone_type), str(row.zone_instance))
        for row in zones.itertuples(index=False)
    }


def assign_zones(frame: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    ids = np.full(n, "Unassigned", dtype=object)
    types = np.full(n, "Unassigned", dtype=object)
    instances = np.full(n, "", dtype=object)
    x = frame["PosX"].to_numpy(float)
    y = frame["PosY"].to_numpy(float)
    zpos = frame["PosZ"].to_numpy(float)
    for zone in zones.sort_values("priority", ascending=False).itertuples(index=False):
        if not isinstance(zone.collider_parts_xz, str):
            raise ValueError(f"Missing collider_parts_xz for {zone.zone_id}")
        xz_hit = np.zeros(n, dtype=bool)
        for part in json.loads(zone.collider_parts_xz):
            xz_hit |= (
                (x >= float(part["xmin"])) & (x <= float(part["xmax"]))
                & (zpos >= float(part["zmin"])) & (zpos <= float(part["zmax"]))
            )
        chosen = (
            (ids == "Unassigned") & xz_hit
            & (y >= float(zone.y_min)) & (y <= float(zone.y_max))
        )
        ids[chosen] = zone.zone_id
        types[chosen] = zone.zone_type
        instances[chosen] = str(zone.zone_instance)
    out = frame.copy()
    out["PhysicalZone"] = ids
    out["ZoneType"] = types
    out["ZoneInstance"] = instances
    return out


def debounce_zone_ids(frame: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    metadata = zone_metadata(zones)
    for _ in range(2):
        labels = out["PhysicalZone"].astype(str).to_numpy(object)
        change = np.r_[True, labels[1:] != labels[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(labels)]
        elapsed = (out["ParsedTime"] - out["ParsedTime"].iloc[0]).dt.total_seconds().to_numpy(float)
        for j, (start, end) in enumerate(zip(starts, ends)):
            duration = elapsed[end - 1] - elapsed[start] if end - start > 1 else 0.0
            if duration >= DEBOUNCE_S or j == 0 or j == len(starts) - 1:
                continue
            previous = labels[starts[j - 1]]
            following = labels[starts[j + 1]]
            if previous == following and previous != "Unassigned":
                labels[start:end] = previous
        out["PhysicalZone"] = labels
    out["ZoneType"] = out["PhysicalZone"].map(lambda x: metadata.get(str(x), ("Unassigned", ""))[0])
    out["ZoneInstance"] = out["PhysicalZone"].map(lambda x: metadata.get(str(x), ("Unassigned", ""))[1])
    return out


def functional_zone(zone_type: str, instance: str, target: str) -> str:
    if zone_type == "Unassigned":
        return "Unassigned"
    if zone_type == "Spawn":
        return "Spawn"
    if zone_type == "Platform":
        return "Platform"
    if zone_type == "StairEntrance":
        return "TargetSideStairEntrance" if target in instance else "NonTargetSideStairEntrance"
    if zone_type == "Stair":
        return "TargetSideStair" if target in instance else "NonTargetSideStair"
    if zone_type in {"HallInsideGate", "Gate", "HallOutsideGate"}:
        return zone_type
    if zone_type == "DecisionZone":
        if len(instance) == 4:
            return "TargetGroupDecision4" if target in instance else "NonTargetGroupDecision4"
        return "TargetPairDecision2" if target in instance else "NonTargetPairDecision2"
    if zone_type == "ExitCorridor":
        return "TargetExitCorridor" if target in instance else "NonTargetExitCorridor"
    if zone_type == "BranchCorridor":
        return "TargetBranchCorridor" if target == instance else "WrongBranchCorridor"
    if zone_type == "Exit":
        return "TargetExit" if target == instance else "WrongExit"
    return zone_type


def add_functional_and_time(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    target = str(row["TargetExit"])
    out["FunctionalZone"] = [
        functional_zone(t, i, target) for t, i in zip(out["ZoneType"], out["ZoneInstance"])
    ]
    start = pd.to_datetime(row["StartTime"], format=TIME_FORMAT)
    end = pd.to_datetime(row["EndTime"], format=TIME_FORMAT)
    duration = max((end - start).total_seconds(), 1e-6)
    progress = ((out["ParsedTime"] - start).dt.total_seconds() / duration).clip(0, 0.999999)
    out["GlobalTimePct"] = progress * 100.0
    out["GlobalTimeBin"] = np.floor(progress * BIN_N).astype(int) + 1
    out["GlobalTimeMidPct"] = (out["GlobalTimeBin"] - 0.5) * 100.0 / BIN_N
    return out


def read_position(row: pd.Series, zones: pd.DataFrame) -> pd.DataFrame:
    path = RAW / str(row["Scenario"]) / str(row["ParticipantID"]) / str(row["FileName"])
    frame = pd.read_csv(path, usecols=["TimeStamp", "PosX", "PosY", "PosZ", "AbsVelo"], low_memory=False)
    frame["ParsedTime"] = parse_time(frame["TimeStamp"])
    for col in ["PosX", "PosY", "PosZ", "AbsVelo"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["ParsedTime", "PosX", "PosY", "PosZ"]).sort_values("ParsedTime")
    start = pd.to_datetime(row["StartTime"], format=TIME_FORMAT)
    end = pd.to_datetime(row["EndTime"], format=TIME_FORMAT)
    frame = frame.loc[frame["ParsedTime"].between(start, end)].drop_duplicates("ParsedTime").reset_index(drop=True)
    frame = debounce_zone_ids(assign_zones(frame, zones), zones)
    return add_functional_and_time(frame, row)


def position_match(source_time: pd.Series, position: pd.DataFrame) -> pd.DataFrame:
    left = pd.DataFrame({"ParsedTime": source_time, "SourceIndex": np.arange(len(source_time))}).dropna().sort_values("ParsedTime")
    right = position[["ParsedTime", "PhysicalZone", "ZoneType", "ZoneInstance", "FunctionalZone", "GlobalTimeBin", "GlobalTimeMidPct"]].sort_values("ParsedTime")
    matched = pd.merge_asof(
        left, right, on="ParsedTime", direction="nearest",
        tolerance=pd.Timedelta(milliseconds=POSITION_TOLERANCE_MS),
    ).sort_values("SourceIndex")
    result = pd.DataFrame(index=np.arange(len(source_time)))
    for col in right.columns.drop("ParsedTime"):
        result.loc[matched["SourceIndex"].to_numpy(int), col] = matched[col].to_numpy()
    return result


def base_record(row: pd.Series) -> dict:
    return {
        **{key: row[key] for key in KEYS},
        "Guidance": row["Guidance"], "Smoke": row["Smoke"], "TargetExit": row["TargetExit"],
    }


def movement_and_visits(
    row: pd.Series,
    position: pd.DataFrame,
    node_ids: np.ndarray,
    tree,
    fields: dict,
) -> tuple[list[dict], list[dict]]:
    frame = temporal_subsample(position, 0.2).reset_index(drop=True)
    xyz = frame[["PosX", "PosY", "PosZ"]].to_numpy(float)
    times = frame["ParsedTime"].to_numpy()
    dt = np.r_[np.nan, np.diff(times).astype("timedelta64[ms]").astype(float) / 1000.0]
    step = np.r_[np.nan, np.linalg.norm(np.diff(xyz, axis=0), axis=1)]
    snap_dist, snap_idx = tree.query(xyz)
    matched = np.asarray(snap_dist <= 3.0)
    snapped = node_ids[np.asarray(snap_idx, int)]
    field = fields.get(str(row["TargetExit"]), {})
    geo = np.asarray([field.get(int(node), np.nan) for node in snapped], float)
    geo[~matched] = np.nan
    geo = pd.Series(geo).interpolate(limit=3).rolling(5, center=True, min_periods=1).median().to_numpy(float)
    contiguous = matched & np.r_[False, matched[:-1]] & (dt > 0) & (dt <= 1.0)
    geo_diff = np.r_[np.nan, np.diff(geo)]
    backtrack = np.where(contiguous & (geo_diff > 0.25), geo_diff, 0.0)
    gain = np.where(contiguous, np.maximum(-geo_diff, 0.0), np.nan)
    productive = np.minimum(step, gain)
    displacement = np.diff(xyz[:, [0, 2]], axis=0)
    norms = np.linalg.norm(displacement, axis=1)
    direction = np.full_like(displacement, np.nan)
    valid_dir = norms > 0.05
    direction[valid_dir] = displacement[valid_dir] / norms[valid_dir, None]
    turns = np.full(len(frame), np.nan)
    for i in range(2, len(frame)):
        if valid_dir[i - 1] and valid_dir[i - 2] and contiguous[i]:
            turns[i] = np.degrees(np.arccos(np.clip(np.dot(direction[i - 1], direction[i - 2]), -1, 1)))

    base = base_record(row)
    rows: list[dict] = []
    valid_zone = frame["PhysicalZone"].ne("Unassigned")
    group_cols = ["PhysicalZone", "ZoneType", "ZoneInstance", "FunctionalZone", "GlobalTimeBin", "GlobalTimeMidPct"]
    for key, idx in frame.loc[valid_zone].groupby(group_cols, observed=True).groups.items():
        chosen = np.zeros(len(frame), bool); chosen[np.asarray(idx, int)] = True
        valid_time = float(np.nansum(dt[chosen & contiguous]))
        observed = float(np.nansum(step[chosen]))
        progress = float(np.nansum(productive[chosen]))
        rows.append({
            **base, **dict(zip(group_cols, key)),
            "MovementObservedTime_s": valid_time,
            "MeanSpeed_m_s": float(np.nanmean(frame.loc[chosen, "AbsVelo"])) if chosen.any() else np.nan,
            "MovementTurnPathRate_deg_s": float(np.nansum(turns[chosen]) / valid_time) if valid_time > 0 else np.nan,
            "BacktrackIncrement_m": float(np.nansum(backtrack[chosen])),
            "ObservedPathIncrement_m": observed,
            "NonProgressPathRatio": 1.0 - progress / observed if observed > 0 else np.nan,
        })

    visit_frame = position.loc[position["PhysicalZone"].ne("Unassigned")].copy()
    next_time = visit_frame["ParsedTime"].shift(-1)
    weights = (next_time - visit_frame["ParsedTime"]).dt.total_seconds().clip(lower=0, upper=0.2)
    weights = weights.fillna(weights[weights.gt(0)].median() if weights.gt(0).any() else 0.02)
    visit_frame["Weight_s"] = weights
    visit_frame["NewVisit"] = visit_frame["PhysicalZone"].ne(visit_frame["PhysicalZone"].shift()).astype(int)
    visit_rows = []
    for key, group in visit_frame.groupby(["PhysicalZone", "ZoneType", "ZoneInstance", "FunctionalZone"], observed=True):
        visit_rows.append({
            **base, "PhysicalZone": key[0], "ZoneType": key[1], "ZoneInstance": key[2], "FunctionalZone": key[3],
            "Visited": 1,
            "FirstEntryTimePct": float(group["GlobalTimePct"].min()),
            "LastExitTimePct": float(group["GlobalTimePct"].max()),
            "ZoneDwellTime_s": float(group["Weight_s"].sum()),
            "ZoneVisitCount": int(group["NewVisit"].sum()),
        })
    return rows, visit_rows


def visual_bins(row: pd.Series, position: pd.DataFrame) -> list[dict]:
    eye_name = str(row["FileName"]).replace("_Subject_Simulation_Info_", "_EyeGaze_Simulation_Info_")
    path = RAW / str(row["Scenario"]) / str(row["ParticipantID"]) / eye_name
    if not path.exists():
        return []
    frame = read_eye(path)
    parsed = parse_time(frame["TimeStamp"])
    start = pd.to_datetime(row["StartTime"], format=TIME_FORMAT)
    end = pd.to_datetime(row["EndTime"], format=TIME_FORMAT)
    chosen = parsed.between(start, end).to_numpy()
    frame = frame.loc[chosen].reset_index(drop=True)
    parsed = parsed.loc[chosen].reset_index(drop=True)
    if len(frame) < 3:
        return []
    arr = derive_eye_arrays(frame)
    location = position_match(parsed, position)
    base = base_record(row)
    rows = []
    groups = location.dropna(subset=["PhysicalZone", "GlobalTimeBin"])
    groups = groups.loc[groups["PhysicalZone"].ne("Unassigned")]
    group_cols = ["PhysicalZone", "ZoneType", "ZoneInstance", "FunctionalZone", "GlobalTimeBin", "GlobalTimeMidPct"]
    for key, idx in groups.groupby(group_cols, observed=True).groups.items():
        mask = np.zeros(len(frame), bool); mask[np.asarray(idx, int)] = True
        metrics = metrics_for_selection(arr, mask)
        if metrics.get("ValidEyeDuration_s", 0) < 0.5:
            continue
        rows.append({**base, **dict(zip(group_cols, key)), **{m: metrics.get(m, np.nan) for m in VISUAL_METRICS}, "ValidEyeDuration_s": metrics["ValidEyeDuration_s"]})
    return rows


def aoi_bins(row: pd.Series, position: pd.DataFrame) -> list[dict]:
    if str(row["Guidance"]) == "StaticSign":
        return []
    path = integrated_path(row)
    if path is None:
        return []
    frame = pd.read_csv(path, low_memory=False)
    parsed = parse_time(frame["TimeStamp"])
    start = pd.to_datetime(row["StartTime"], format=TIME_FORMAT)
    end = pd.to_datetime(row["EndTime"], format=TIME_FORMAT)
    chosen = parsed.between(start, end).to_numpy()
    frame = frame.loc[chosen].reset_index(drop=True)
    parsed = parsed.loc[chosen].reset_index(drop=True)
    if len(frame) < 3:
        return []
    _, _, weights = timestamps(frame["TimeStamp"])
    valid_cols, gaze_col = aoi_columns(str(row["Guidance"]))
    valid = np.logical_or.reduce([to_bool(frame[col]) for col in valid_cols])
    gaze = to_bool(frame[gaze_col]) & valid
    location = position_match(parsed, position)
    base = base_record(row)
    rows = []
    group_cols = ["PhysicalZone", "ZoneType", "ZoneInstance", "FunctionalZone", "GlobalTimeBin", "GlobalTimeMidPct"]
    groups = location.dropna(subset=["PhysicalZone", "GlobalTimeBin"])
    groups = groups.loc[groups["PhysicalZone"].ne("Unassigned")]
    for key, idx in groups.groupby(group_cols, observed=True).groups.items():
        ids = np.asarray(idx, int)
        denom = float(weights[ids][valid[ids]].sum())
        numer = float(weights[ids][gaze[ids]].sum())
        if denom <= 0:
            continue
        rows.append({
            **base, **dict(zip(group_cols, key)), "AOIValidDuration_s": denom,
            "GuideDwellDuration_s": numer, "GuideDwellProportion": numer / denom,
            "GuideAcquiredInBin": int(numer >= 0.1),
        })
    return rows


def hrv_bins(row: pd.Series, position: pd.DataFrame, rr: pd.DataFrame) -> list[dict]:
    if rr.empty:
        return []
    part = rr.copy().sort_values("RRTime").reset_index(drop=True)
    location = position_match(pd.to_datetime(part["RRTime"]), position)
    for col in location:
        part[col] = location[col].to_numpy()
    part = part.dropna(subset=["PhysicalZone", "GlobalTimeBin"])
    part = part.loc[part["PhysicalZone"].ne("Unassigned")].copy()
    if part.empty:
        return []
    previous_same = (
        part["PhysicalZone"].eq(part["PhysicalZone"].shift())
        & part["GlobalTimeBin"].eq(part["GlobalTimeBin"].shift())
    )
    part["ZonePairValid"] = part["RRPrimaryValid"].astype(bool) & part["RRPrimaryValid"].shift(1, fill_value=False).astype(bool) & previous_same
    part["ZoneRRDiff_ms"] = part["RR_ms"].diff()
    base = base_record(row)
    rows = []
    group_cols = ["PhysicalZone", "ZoneType", "ZoneInstance", "FunctionalZone", "GlobalTimeBin", "GlobalTimeMidPct"]
    for key, group in part.groupby(group_cols, observed=True):
        valid = group["RRPrimaryValid"].astype(bool)
        hr = pd.to_numeric(group.loc[valid, "InstantHR_bpm"], errors="coerce")
        diffs = pd.to_numeric(group.loc[group["ZonePairValid"], "ZoneRRDiff_ms"], errors="coerce").dropna()
        span = (group["RRTime"].max() - group["RRTime"].min()).total_seconds() if len(group) > 1 else 0.0
        rmssd = float(np.sqrt(np.mean(diffs.to_numpy(float) ** 2))) if len(diffs) >= 10 and span >= 10 else np.nan
        rows.append({
            **base, **dict(zip(group_cols, key)),
            "RRValidN": int(valid.sum()), "HRVObservedSpan_s": float(span),
            "MeanHRBaselineDelta_bpm": float(hr.mean() - group["RestMeanHR_bpm"].iloc[0]) if len(hr) >= 3 else np.nan,
            "LogRMSSDBaselineDelta": float(np.log(rmssd) - group["RestLogRMSSD"].iloc[0]) if np.isfinite(rmssd) and rmssd > 0 else np.nan,
        })
    return rows


def extract_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = pd.read_csv(DATA / "analysis_trials.csv")
    zones = pd.read_csv(ZONE_DIR / "semantic_zones.csv")
    rr = pd.read_csv(DATA / "hrv_rr_events_assigned.csv", low_memory=False)
    rr["RRTime"] = pd.to_datetime(rr["RRTime"], errors="coerce")
    rr_groups = {key: part for key, part in rr.groupby(KEYS, sort=False)}
    node_ids, tree, fields = load_distance_fields()
    movement_rows: list[dict] = []
    visit_rows: list[dict] = []
    visual_rows: list[dict] = []
    aoi_rows: list[dict] = []
    hrv_rows: list[dict] = []
    for number, (_, row) in enumerate(trials.iterrows(), 1):
        position = read_position(row, zones)
        movement, visits = movement_and_visits(row, position, node_ids, tree, fields)
        movement_rows.extend(movement); visit_rows.extend(visits)
        visual_rows.extend(visual_bins(row, position))
        aoi_rows.extend(aoi_bins(row, position))
        hrv_rows.extend(hrv_bins(row, position, rr_groups.get(tuple(row[k] for k in KEYS), pd.DataFrame())))
        if number % 20 == 0:
            print(f"  space-time extraction {number}/{len(trials)}")

    movement = pd.DataFrame(movement_rows)
    visits = pd.DataFrame(visit_rows)
    visual = pd.DataFrame(visual_rows)
    aoi = pd.DataFrame(aoi_rows)
    hrv = pd.DataFrame(hrv_rows)
    join = KEYS + ["PhysicalZone", "ZoneType", "ZoneInstance", "FunctionalZone", "GlobalTimeBin", "GlobalTimeMidPct"]
    merged = movement.merge(visual, on=join + ["Guidance", "Smoke", "TargetExit"], how="outer", validate="one_to_one")
    merged = merged.merge(aoi, on=join + ["Guidance", "Smoke", "TargetExit"], how="outer", validate="one_to_one")
    merged = merged.merge(hrv, on=join + ["Guidance", "Smoke", "TargetExit"], how="outer", validate="one_to_one")
    controls = trials[KEYS + ["SpatialDifficultyRatio", "StandardizedTime_s", "ExitCorrect", "StartTime"]].copy()
    controls["StartParsed"] = pd.to_datetime(controls["StartTime"], format=TIME_FORMAT, errors="coerce")
    controls["TrialOrder"] = controls.groupby("ParticipantID")["StartParsed"].rank(method="first")
    merged = merged.merge(controls.drop(columns=["StartTime", "StartParsed"]), on=KEYS, how="left", validate="many_to_one")
    movement.to_csv(DATA / "zone_movement_time_bins.csv", index=False, encoding="utf-8-sig")
    visual.to_csv(DATA / "zone_visual_time_bins.csv", index=False, encoding="utf-8-sig")
    aoi.to_csv(DATA / "zone_aoi_time_bins.csv", index=False, encoding="utf-8-sig")
    hrv.to_csv(DATA / "zone_hrv_time_bins.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(DATA / "spatiotemporal_zone_bins.csv", index=False, encoding="utf-8-sig")
    visits.to_csv(DATA / "spatiotemporal_zone_visits.csv", index=False, encoding="utf-8-sig")
    return merged, visits


def zone_summaries(merged: pd.DataFrame, visits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = pd.read_csv(DATA / "analysis_trials.csv")
    visit_presence = visits.groupby(KEYS + ["Guidance", "Smoke", "FunctionalZone"], as_index=False).agg(
        Visited=("Visited", "max"), FirstEntryTimePct=("FirstEntryTimePct", "min"),
        ZoneDwellTime_s=("ZoneDwellTime_s", "sum"), ZoneVisitCount=("ZoneVisitCount", "sum"),
    )
    roles = sorted(visits["FunctionalZone"].dropna().unique())
    grid = trials[KEYS + ["Guidance", "Smoke"]].assign(_key=1).merge(pd.DataFrame({"FunctionalZone": roles, "_key": 1}), on="_key").drop(columns="_key")
    complete = grid.merge(visit_presence, on=KEYS + ["Guidance", "Smoke", "FunctionalZone"], how="left")
    complete["Visited"] = complete["Visited"].fillna(0)
    summary = complete.groupby(["Guidance", "Smoke", "FunctionalZone"], as_index=False, observed=True).agg(
        TrialN=("Scenario", "size"), VisitedN=("Visited", "sum"),
        MeanFirstEntryPct=("FirstEntryTimePct", "mean"), MedianDwellTime_s=("ZoneDwellTime_s", "median"),
        MeanVisitCount=("ZoneVisitCount", "mean"),
    )
    summary["VisitProportion"] = summary["VisitedN"] / summary["TrialN"]
    summary.to_csv(TABLE / "functional_zone_visit_summary.csv", index=False, encoding="utf-8-sig")

    agg = merged.groupby(KEYS + ["Guidance", "Smoke", "TargetExit", "FunctionalZone"], as_index=False, observed=True).agg(
        ZoneObservedBinN=("GlobalTimeBin", "nunique"),
        ZoneMovementTime_s=("MovementObservedTime_s", "sum"),
        MeanSpeed_m_s=("MeanSpeed_m_s", "mean"),
        MovementTurnPathRate_deg_s=("MovementTurnPathRate_deg_s", "mean"),
        BacktrackIncrement_m=("BacktrackIncrement_m", "sum"),
        NonProgressPathRatio=("NonProgressPathRatio", "mean"),
        HeadAngularPathRate_deg_s=("HeadAngularPathRate_deg_s", "mean"),
        FixationTimeProportion=("FixationTimeProportion", "mean"),
        GazeDirectionEntropy=("GazeDirectionEntropy", "mean"),
        MeanHRBaselineDelta_bpm=("MeanHRBaselineDelta_bpm", "mean"),
        LogRMSSDBaselineDelta=("LogRMSSDBaselineDelta", "mean"),
        GuideDwellProportion=("GuideDwellProportion", "mean"),
    )
    agg = agg.merge(
        visits.groupby(KEYS + ["FunctionalZone"], as_index=False).agg(
            FirstEntryTimePct=("FirstEntryTimePct", "min"), ZoneDwellTime_s=("ZoneDwellTime_s", "sum"),
            ZoneVisitCount=("ZoneVisitCount", "sum"),
        ), on=KEYS + ["FunctionalZone"], how="left", validate="one_to_one",
    )
    agg.to_csv(DATA / "spatiotemporal_trial_zone_features.csv", index=False, encoding="utf-8-sig")
    return summary, agg


def spline_tests(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = VISUAL_METRICS + HRV_METRICS + MOVEMENT_METRICS + AOI_METRICS
    for metric in metrics:
        if metric not in merged:
            continue
        source = merged.dropna(subset=[metric, "FunctionalZone", "GlobalTimeMidPct", "SpatialDifficultyRatio"]).copy()
        for role, data in source.groupby("FunctionalZone", observed=True):
            counts = data.groupby("Guidance")["ParticipantID"].nunique()
            valid_modes = counts[counts >= 6].index
            data = data.loc[data["Guidance"].isin(valid_modes)].copy()
            if len(data) < 50 or data["Guidance"].nunique() < 2 or data["GlobalTimeBin"].nunique() < 4:
                continue
            try:
                model = ols(
                    f"{metric} ~ C(Guidance) * bs(GlobalTimeMidPct, df=4, degree=3, include_intercept=False) + C(Smoke) + np.log(SpatialDifficultyRatio)",
                    data=data,
                ).fit(cov_type="cluster", cov_kwds={"groups": data["ParticipantID"]})
                names = model.model.exog_names
                interaction = [i for i, name in enumerate(names) if "C(Guidance)" in name and ":bs(" in name]
                guidance = [i for i, name in enumerate(names) if "C(Guidance)" in name and ":" not in name]
                def pvalue(indices: list[int]) -> float:
                    if not indices:
                        return np.nan
                    restriction = np.zeros((len(indices), len(names)))
                    for r, c in enumerate(indices): restriction[r, c] = 1.0
                    return float(model.wald_test(restriction, scalar=True).pvalue)
                rows.append({
                    "Metric": metric, "Layer": (
                        "visual_information" if metric in VISUAL_METRICS else "autonomic_load" if metric in HRV_METRICS
                        else "spatial_behavior" if metric in MOVEMENT_METRICS else "AR_information"
                    ),
                    "FunctionalZone": role, "N": len(data), "ParticipantN": data["ParticipantID"].nunique(),
                    "GuidanceModeN": data["Guidance"].nunique(), "TimeBinN": data["GlobalTimeBin"].nunique(),
                    "GuidanceMainP": pvalue(guidance), "GuidanceTimeInteractionP": pvalue(interaction),
                    "Status": "participant_clustered_spline_screen_GAMM_compatible_not_penalized_GAMM",
                })
            except Exception as exc:
                rows.append({"Metric": metric, "FunctionalZone": role, "N": len(data), "Status": f"fit_failed:{type(exc).__name__}"})
    out = pd.DataFrame(rows)
    out["GuidanceTimeInteractionFDR"] = np.nan
    okay = out["GuidanceTimeInteractionP"].notna() if "GuidanceTimeInteractionP" in out else pd.Series(False, index=out.index)
    if okay.any():
        out.loc[okay, "GuidanceTimeInteractionFDR"] = multipletests(out.loc[okay, "GuidanceTimeInteractionP"], method="fdr_bh")[1]
    out.to_csv(TABLE / "spatiotemporal_spline_tests.csv", index=False, encoding="utf-8-sig")
    return out


def pairwise_zone_time_contrasts(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = VISUAL_METRICS + ["MeanHRBaselineDelta_bpm"] + MOVEMENT_METRICS + AOI_METRICS
    for metric in metrics:
        if metric not in merged:
            continue
        source = merged.dropna(subset=[metric, "SpatialDifficultyRatio"]).copy()
        for (role, time_bin), cell in source.groupby(["FunctionalZone", "GlobalTimeBin"], observed=True):
            for reference, comparison in combinations(GUIDANCE_ORDER, 2):
                sub = cell.loc[cell["Guidance"].isin([reference, comparison])].copy()
                counts = sub.groupby("Guidance")["ParticipantID"].nunique()
                if len(counts) < 2 or counts.min() < 6:
                    continue
                sub["Treatment"] = sub["Guidance"].eq(comparison).astype(int)
                try:
                    fit = ols(
                        f"{metric} ~ Treatment + C(Smoke) + np.log(SpatialDifficultyRatio)", data=sub,
                    ).fit(cov_type="cluster", cov_kwds={"groups": sub["ParticipantID"]})
                    rows.append({
                        "Metric": metric, "FunctionalZone": role, "GlobalTimeBin": int(time_bin),
                        "GlobalTimeMidPct": (float(time_bin) - 0.5) * 10,
                        "Reference": reference, "Comparison": comparison,
                        "AdjustedDifference": float(fit.params.get("Treatment", np.nan)),
                        "SE": float(fit.bse.get("Treatment", np.nan)), "PValue": float(fit.pvalues.get("Treatment", np.nan)),
                        "N": len(sub), "ParticipantN": sub["ParticipantID"].nunique(),
                    })
                except Exception:
                    continue
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    if len(out):
        for metric, idx in out.groupby("Metric").groups.items():
            out.loc[idx, "PValueFDR"] = multipletests(out.loc[idx, "PValue"].fillna(1), method="fdr_bh")[1]
    out.to_csv(TABLE / "spatiotemporal_pairwise_guidance_contrasts.csv", index=False, encoding="utf-8-sig")
    return out


def wide_trial_features(trial_zone: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    roles = [
        "Spawn", "Platform", "HallInsideGate", "Gate", "HallOutsideGate",
        "TargetGroupDecision4", "TargetPairDecision2", "TargetExitCorridor", "TargetBranchCorridor",
    ]
    metrics = VISUAL_METRICS + ["MeanHRBaselineDelta_bpm", "LogRMSSDBaselineDelta"] + MOVEMENT_METRICS + AOI_METRICS
    chosen = trial_zone.loc[trial_zone["FunctionalZone"].isin(roles), KEYS + ["FunctionalZone"] + metrics].copy()
    wide_parts = []
    for metric in metrics:
        pivot = chosen.pivot_table(index=KEYS, columns="FunctionalZone", values=metric, aggfunc="mean")
        pivot.columns = [f"{role}__{metric}" for role in pivot.columns]
        wide_parts.append(pivot)
    wide = pd.concat(wide_parts, axis=1).reset_index() if wide_parts else pd.DataFrame(columns=KEYS)
    wrong = visits.assign(WrongBranch=lambda x: x["FunctionalZone"].eq("WrongBranchCorridor").astype(int)).groupby(KEYS, as_index=False).agg(
        WrongBranchVisited=("WrongBranch", "max"),
        WrongBranchDwell_s=("ZoneDwellTime_s", lambda x: float(x[visits.loc[x.index, "FunctionalZone"].eq("WrongBranchCorridor")].sum())),
    )
    trials = pd.read_csv(DATA / "analysis_trials.csv")
    trials["StartParsed"] = pd.to_datetime(trials["StartTime"], format=TIME_FORMAT, errors="coerce")
    trials["TrialOrder"] = trials.groupby("ParticipantID")["StartParsed"].rank(method="first")
    out = trials.merge(wide, on=KEYS, how="left", validate="one_to_one").merge(wrong, on=KEYS, how="left", validate="one_to_one")
    out[["WrongBranchVisited", "WrongBranchDwell_s"]] = out[["WrongBranchVisited", "WrongBranchDwell_s"]].fillna(0)
    out["SmokeHigh"] = out["Smoke"].eq("HighSmoke").astype(int)
    return out


def grouped_ebm(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = ["SpatialDifficultyRatio", "SmokeHigh", "TrialOrder"]
    # Primary prediction deliberately stops before the final directional
    # branch. Otherwise target/wrong-branch presence nearly encodes the exit
    # correctness outcome and creates an outcome-leakage interpretation.
    prechoice_roles = {"Spawn", "Platform", "HallInsideGate", "Gate", "HallOutsideGate"}
    process = [
        c for c in wide.columns
        if "__" in c
        and c.split("__", 1)[0] in prechoice_roles
        and not c.endswith("GuideDwellProportion")
    ]
    process = [c for c in process if wide[c].notna().sum() >= 150 and wide[c].nunique(dropna=True) > 1]
    rows = []
    importance_rows = []
    for scope, features in [("controls_only", controls), ("spatiotemporal_processes", controls + process)]:
        x = wide[features].apply(pd.to_numeric, errors="coerce")
        # Median imputation avoids letting EBM use modality missingness itself
        # as a proxy for the outcome. Missingness is audited separately.
        x = x.fillna(x.median(numeric_only=True))
        groups = wide["ParticipantID"]
        splitter = GroupKFold(n_splits=5)
        for outcome in ["StandardizedTime_s", "ExitCorrect"]:
            y = pd.to_numeric(wide[outcome], errors="coerce")
            pred = np.full(len(wide), np.nan)
            for train, test in splitter.split(x, y, groups):
                common = dict(max_bins=12, max_rounds=1500, outer_bags=4, interactions=3, random_state=SEED, n_jobs=1)
                model = ExplainableBoostingRegressor(**common) if outcome == "StandardizedTime_s" else ExplainableBoostingClassifier(**common)
                model.fit(x.iloc[train], y.iloc[train])
                pred[test] = model.predict(x.iloc[test]) if outcome == "StandardizedTime_s" else model.predict_proba(x.iloc[test])[:, 1]
            if outcome == "StandardizedTime_s":
                rows.append({"Scope": scope, "Outcome": outcome, "MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred), "AUC": np.nan, "Brier": np.nan, "FeatureN": len(features)})
            else:
                rows.append({"Scope": scope, "Outcome": outcome, "MAE": np.nan, "R2": np.nan, "AUC": roc_auc_score(y, pred), "Brier": brier_score_loss(y, pred), "FeatureN": len(features)})
            if scope == "spatiotemporal_processes":
                common = dict(max_bins=12, max_rounds=2000, outer_bags=6, interactions=3, random_state=SEED, n_jobs=1)
                full = ExplainableBoostingRegressor(**common) if outcome == "StandardizedTime_s" else ExplainableBoostingClassifier(**common)
                full.fit(x, y)
                for name, value in zip(full.term_names_, full.term_importances()):
                    importance_rows.append({"Outcome": outcome, "Term": name, "Importance": float(value)})
    cv = pd.DataFrame(rows)
    importance = pd.DataFrame(importance_rows).sort_values(["Outcome", "Importance"], ascending=[True, False])
    cv.to_csv(TABLE / "spatiotemporal_ebm_grouped_cv.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(TABLE / "spatiotemporal_ebm_importance.csv", index=False, encoding="utf-8-sig")
    return cv, importance


def build_mediators(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    def columns(roles: list[str], metrics: list[str]) -> list[str]:
        return [f"{role}__{metric}" for role in roles for metric in metrics if f"{role}__{metric}" in out]
    info_cols = columns(["TargetGroupDecision4", "TargetPairDecision2"], VISUAL_METRICS)
    autonomic_cols = columns(["TargetGroupDecision4", "TargetPairDecision2", "TargetExitCorridor"], HRV_METRICS)
    spatial_cols = columns(["TargetExitCorridor", "TargetBranchCorridor"], ["MovementTurnPathRate_deg_s", "BacktrackIncrement_m", "NonProgressPathRatio"])
    info_parts = []
    for col in info_cols:
        sign = -1 if col.endswith("FixationTimeProportion") else 1
        name = f"Z_{col}"; out[name] = sign * zscore(out[col]); info_parts.append(name)
    auto_parts = []
    for col in autonomic_cols:
        sign = -1 if col.endswith("LogRMSSDBaselineDelta") else 1
        name = f"Z_{col}"; out[name] = sign * zscore(out[col]); auto_parts.append(name)
    spatial_parts = []
    for col in spatial_cols:
        name = f"Z_{col}"; out[name] = zscore(out[col]); spatial_parts.append(name)
    out["Z_WrongBranchVisited"] = zscore(out["WrongBranchVisited"])
    spatial_parts.append("Z_WrongBranchVisited")
    out["DecisionInformationBurden"] = out[info_parts].mean(axis=1, skipna=True).where(out[info_parts].notna().sum(axis=1) >= 2)
    out["DecisionAutonomicLoad"] = out[auto_parts].mean(axis=1, skipna=True).where(out[auto_parts].notna().sum(axis=1) >= 1)
    out["DownstreamSpatialInefficiency"] = out[spatial_parts].mean(axis=1, skipna=True).where(out[spatial_parts].notna().sum(axis=1) >= 2)
    out.to_csv(DATA / "spatiotemporal_trial_mediators.csv", index=False, encoding="utf-8-sig")
    return out


def mediation_fit(data: pd.DataFrame, outcome: str, comparison: str, serial: bool) -> dict[str, float]:
    mediators = ["DecisionInformationBurden", "DecisionAutonomicLoad", "DownstreamSpatialInefficiency"]
    x = data.copy()
    a = x["Guidance"].eq(comparison).astype(float).to_numpy()
    controls = np.column_stack([
        x["Smoke"].eq("HighSmoke").astype(float), np.log(x["SpatialDifficultyRatio"].to_numpy(float)),
        x["TrialOrder"].to_numpy(float),
    ])
    m1, m2, m3 = [x[m].to_numpy(float) for m in mediators]
    y = np.log(x["StandardizedTime_s"].clip(lower=0.1).to_numpy(float)) if outcome == "StandardizedTime_s" else x["ExitCorrect"].to_numpy(float)
    base = np.column_stack([np.ones(len(x)), a, controls])
    total = np.linalg.lstsq(base, y, rcond=None)[0][1]
    if not serial:
        paths = []
        for m in (m1, m2, m3): paths.append(np.linalg.lstsq(base, m, rcond=None)[0][1])
        by = np.linalg.lstsq(np.column_stack([base, m1, m2, m3]), y, rcond=None)[0]
        specific = np.asarray(paths) * by[base.shape[1]:base.shape[1] + 3]
        return {
            "Total": total, "Direct": by[1], "JointIndirect": float(specific.sum()),
            "Specific:DecisionInformationBurden": specific[0],
            "Specific:DecisionAutonomicLoad": specific[1],
            "Specific:DownstreamSpatialInefficiency": specific[2],
        }
    b1 = np.linalg.lstsq(base, m1, rcond=None)[0]
    b2 = np.linalg.lstsq(np.column_stack([base, m1]), m2, rcond=None)[0]
    b3 = np.linalg.lstsq(np.column_stack([base, m1, m2]), m3, rcond=None)[0]
    by = np.linalg.lstsq(np.column_stack([base, m1, m2, m3]), y, rcond=None)[0]
    a1, a2, a3 = b1[1], b2[1], b3[1]
    d21, d31, d32 = b2[-1], b3[-2], b3[-1]
    y1, y2, y3 = by[-3], by[-2], by[-1]
    paths = {
        "A→Info→Y": a1 * y1, "A→Autonomic→Y": a2 * y2, "A→Spatial→Y": a3 * y3,
        "A→Info→Autonomic→Y": a1 * d21 * y2,
        "A→Info→Spatial→Y": a1 * d31 * y3,
        "A→Autonomic→Spatial→Y": a2 * d32 * y3,
        "A→Info→Autonomic→Spatial→Y": a1 * d21 * d32 * y3,
    }
    return {"Total": total, "Direct": by[1], "JointSerialIndirect": float(sum(paths.values())), **paths}


def scaled(value: float, outcome: str) -> float:
    return float((np.exp(value) - 1) * 100) if outcome == "StandardizedTime_s" else float(value * 100)


def bootstrap_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(2 * min((values <= 0).mean(), (values >= 0).mean())) if len(values) else np.nan


def spatial_mediation(mediators: pd.DataFrame) -> pd.DataFrame:
    mediator_cols = ["DecisionInformationBurden", "DecisionAutonomicLoad", "DownstreamSpatialInefficiency"]
    rows = []
    for reference, comparison in combinations(GUIDANCE_ORDER, 2):
        sub = mediators.loc[mediators["Guidance"].isin([reference, comparison])].dropna(
            subset=mediator_cols + ["SpatialDifficultyRatio", "TrialOrder", "StandardizedTime_s", "ExitCorrect"]
        ).copy()
        if sub["Guidance"].nunique() < 2:
            continue
        for col in mediator_cols: sub[col] = zscore(sub[col])
        participants = sub["ParticipantID"].unique()
        if len(participants) < 10:
            continue
        rows_by_id = {pid: np.flatnonzero(sub["ParticipantID"].to_numpy() == pid) for pid in participants}
        for model_name, serial in [("parallel_spatial_layers", False), ("serial_spatial_layers", True)]:
            for outcome in ["StandardizedTime_s", "ExitCorrect"]:
                estimate = mediation_fit(sub, outcome, comparison, serial)
                boot = {name: [] for name in estimate}
                for _ in range(BOOT_N):
                    sampled = RNG.choice(participants, len(participants), replace=True)
                    chosen = np.concatenate([rows_by_id[pid] for pid in sampled])
                    sample = sub.iloc[chosen]
                    if sample["Guidance"].nunique() < 2:
                        continue
                    try:
                        values = mediation_fit(sample, outcome, comparison, serial)
                        for name, value in values.items(): boot[name].append(value)
                    except Exception:
                        continue
                for effect, value in estimate.items():
                    values = np.asarray(boot[effect], float)
                    scaled_values = np.asarray([scaled(v, outcome) for v in values])
                    rows.append({
                        "Model": model_name, "Reference": reference, "Comparison": comparison,
                        "Outcome": outcome, "Effect": effect, "Estimate": scaled(value, outcome),
                        "CI95Low": float(np.quantile(scaled_values, 0.025)) if len(scaled_values) else np.nan,
                        "CI95High": float(np.quantile(scaled_values, 0.975)) if len(scaled_values) else np.nan,
                        "PValueBootstrap": bootstrap_p(values), "N": len(sub), "ParticipantN": len(participants),
                        "Controls": "smoke + log spatial difficulty + trial order",
                        "Status": "exploratory_spatially_ordered_mediation_cluster_bootstrap",
                    })
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    indirect = ~out["Effect"].isin(["Total", "Direct"])
    for (model, outcome), idx in out.loc[indirect].groupby(["Model", "Outcome"]).groups.items():
        out.loc[idx, "PValueFDR"] = multipletests(out.loc[idx, "PValueBootstrap"].fillna(1), method="fdr_bh")[1]
    out.to_csv(TABLE / "gcomputation_spatiotemporal_multifactor_mediation.csv", index=False, encoding="utf-8-sig")
    return out


def make_figures(
    visit_summary: pd.DataFrame,
    spline: pd.DataFrame,
    contrasts: pd.DataFrame,
    importance: pd.DataFrame,
    mediation: pd.DataFrame,
) -> None:
    role_order = [
        "Spawn", "Platform", "TargetSideStairEntrance", "TargetSideStair", "HallInsideGate", "Gate",
        "HallOutsideGate", "TargetGroupDecision4", "TargetExitCorridor", "TargetPairDecision2",
        "TargetBranchCorridor", "TargetExit", "WrongBranchCorridor", "WrongExit",
    ]
    visit = visit_summary.groupby(["Guidance", "FunctionalZone"], as_index=False)["VisitProportion"].mean()
    matrix = visit.pivot(index="FunctionalZone", columns="Guidance", values="VisitProportion").reindex(index=[r for r in role_order if r in visit["FunctionalZone"].unique()], columns=GUIDANCE_ORDER)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(matrix, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "访问比例"})
    ax.set_title("功能空间访问比例"); ax.set_xlabel("引导模式"); ax.set_ylabel("功能区域")
    savefig(fig, "14_functional_zone_visit_heatmap")

    okay = spline.dropna(subset=["GuidanceTimeInteractionFDR"]).copy()
    okay["Signal"] = -np.log10(okay["GuidanceTimeInteractionFDR"].clip(lower=1e-6))
    matrix = okay.pivot_table(index="FunctionalZone", columns="Metric", values="Signal", aggfunc="max")
    if len(matrix):
        fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(matrix))))
        sns.heatmap(matrix, cmap="YlOrBr", ax=ax, cbar_kws={"label": "-log10(FDR)"})
        ax.axvline(0, color="none"); ax.set_title("空间区域中的引导×时间动态差异强度")
        ax.set_xlabel("因素"); ax.set_ylabel("功能区域")
        ax.set_xticklabels([METRIC_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=35, ha="right")
        savefig(fig, "15_zone_time_spline_significance")

    sig = contrasts.loc[contrasts["PValueFDR"].lt(0.05)].copy()
    if len(sig):
        sig["Contrast"] = sig["Reference"] + "→" + sig["Comparison"]
        count = sig.groupby(["FunctionalZone", "GlobalTimeBin"], as_index=False).size()
        matrix = count.pivot(index="FunctionalZone", columns="GlobalTimeBin", values="size").fillna(0)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(matrix))))
        sns.heatmap(matrix, cmap="Oranges", annot=True, fmt=".0f", ax=ax, cbar_kws={"label": "FDR<0.05的模式两两差异数"})
        ax.set_title("差异集中在哪个空间与时间"); ax.set_xlabel("全程归一化时间格（1=0–10%）"); ax.set_ylabel("功能区域")
        savefig(fig, "16_zone_time_pairwise_difference_heatmap")

    top = importance.loc[~importance["Term"].str.contains(" & ", regex=False)].groupby("Outcome", group_keys=False).head(12).copy()
    if len(top):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, outcome in zip(axes, ["StandardizedTime_s", "ExitCorrect"]):
            data = top.loc[top["Outcome"].eq(outcome)].sort_values("Importance")
            ax.barh(data["Term"].str.replace("__", " / ", regex=False), data["Importance"], color="#0072B2" if outcome == "StandardizedTime_s" else "#D55E00")
            ax.set_title("标准化疏散时间" if outcome == "StandardizedTime_s" else "出口正确性")
            ax.set_xlabel("EBM全局重要性（预测贡献）")
        fig.suptitle("空间化EBM：哪个区域中的过程因素最能解释结果", y=1.01)
        fig.tight_layout(); savefig(fig, "17_spatiotemporal_ebm_importance")

    med = mediation.loc[
        mediation["Effect"].str.contains("Joint") & mediation["Model"].eq("serial_spatial_layers")
    ].copy()
    med = med.sort_values("PValueFDR").head(16)
    if len(med):
        med["Label"] = med["Reference"] + "→" + med["Comparison"] + " / " + med["Outcome"].map({"StandardizedTime_s": "时间", "ExitCorrect": "正确率"})
        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(med)); ax.errorbar(med["Estimate"], y, xerr=[med["Estimate"] - med["CI95Low"], med["CI95High"] - med["Estimate"]], fmt="o", color="#7A3E9D", ecolor="#B8A0C8", capsize=3)
        ax.axvline(0, color="#333333", lw=1); ax.set_yticks(y, med["Label"]); ax.invert_yaxis()
        ax.set_xlabel("联合空间序列间接效应（时间:%；正确率:百分点）")
        ax.set_title("空间有序多因素中介：决策信息→自主负荷→下游空间行为")
        fig.tight_layout(); savefig(fig, "18_spatiotemporal_serial_mediation")


def write_notes(summary: dict) -> None:
    text = f"""# 空间—时间增量分析说明

本轮只重跑受新空间区域或静息HRV基线影响的环节。原始总效果、空间难度标准化和眼动基础清洗未重跑。

- 分析单位：试次 × 功能空间区域 × 全程归一化10%时间格。
- 物理区域用于场景定位；功能区域用于跨目标出口对齐。
- 位置与眼动/AOI/RR使用最近时间戳匹配，容差{POSITION_TOLERANCE_MS} ms。
- 区域边界短摆动使用{DEBOUNCE_S:.1f}秒防抖。
- 静息HRV使用每位受试者独立RR文件，任务心率和log(RMSSD)改为相对静息基线变化。
- 动态检验使用受试者聚类稳健样条筛查；它回答GAMM同类问题，但不是带惩罚项的正式GAMM，论文阶段建议用R mgcv复核。
- 空间中介按“决策信息负担 → 决策/通道自主负荷 → 下游空间低效 → 结果”组织，并控制烟雾、空间难度和试次顺序。

## 数据规模

- 主试次：{summary['trial_n']}
- 空间—时间聚合行：{summary['zone_time_row_n']}
- 功能区域类型：{summary['functional_zone_n']}
- 静息基线可用受试者：{summary['rest_usable_participant_n']}
- 空间化中介完整试次：{summary['mediation_complete_trial_n']}
"""
    (DOC / "spatiotemporal_extension_method.md").write_text(text, encoding="utf-8")


def main(reuse_extracted: bool = False) -> None:
    if reuse_extracted and (DATA / "spatiotemporal_zone_bins.csv").exists() and (DATA / "spatiotemporal_zone_visits.csv").exists():
        print("A/7 Reusing completed zone-aligned multimodal bins")
        merged = pd.read_csv(DATA / "spatiotemporal_zone_bins.csv", low_memory=False)
        visits = pd.read_csv(DATA / "spatiotemporal_zone_visits.csv", low_memory=False)
    else:
        print("A/7 Extracting zone-aligned multimodal bins")
        merged, visits = extract_all()
    print("B/7 Building functional-zone summaries")
    visit_summary, trial_zone = zone_summaries(merged, visits)
    print("C/7 Fitting participant-clustered spline screens")
    spline = spline_tests(merged)
    print("D/7 Computing pairwise zone-time contrasts")
    contrasts = pairwise_zone_time_contrasts(merged)
    print("E/7 Fitting grouped-CV space-time EBM")
    wide = wide_trial_features(trial_zone, visits)
    cv, importance = grouped_ebm(wide)
    print("F/7 Running spatially ordered mediation")
    mediators = build_mediators(wide)
    mediation = spatial_mediation(mediators)
    print("G/7 Rendering reviewed figures and notes")
    make_figures(visit_summary, spline, contrasts, importance, mediation)
    rest = pd.read_csv(TABLE / "hrv_rest_baseline_quality_audit.csv")
    summary = {
        "generated_at": "2026-08-04", "trial_n": int(pd.read_csv(DATA / "analysis_trials.csv").shape[0]),
        "zone_time_row_n": int(len(merged)), "functional_zone_n": int(merged["FunctionalZone"].nunique()),
        "physical_zone_n": int(merged["PhysicalZone"].nunique()),
        "rest_usable_participant_n": int(rest.loc[rest["RestPrimaryUsable"].eq(1), "ParticipantID"].nunique()),
        "mediation_complete_trial_n": int(mediators[["DecisionInformationBurden", "DecisionAutonomicLoad", "DownstreamSpatialInefficiency"]].notna().all(axis=1).sum()),
        "spline_test_n": int(len(spline)), "spline_fdr_significant_n": int(spline["GuidanceTimeInteractionFDR"].lt(0.05).sum()),
        "pairwise_zone_time_test_n": int(len(contrasts)), "pairwise_zone_time_fdr_significant_n": int(contrasts["PValueFDR"].lt(0.05).sum()),
        "ebm_cv": cv.to_dict(orient="records"),
        "scope": "incremental_space_time_and_resting_HRV_baseline_only",
    }
    (DATA / "spatiotemporal_extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
