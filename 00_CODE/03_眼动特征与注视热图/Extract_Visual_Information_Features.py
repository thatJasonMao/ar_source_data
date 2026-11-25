"""Extract trial-level and phase-level visual information features.

All source VR logs are read-only. Outputs are written only under the dedicated
analysis directory. The raw eye CSV writer leaves a trailing comma on every
row; therefore the parser explicitly supplies a twentieth placeholder column
to prevent pandas from silently shifting TimeStamp into the index.

The raw log contains combined gaze direction and head Euler rotation, but no
vendor validity/confidence, pupil diameter, fixation id, or saccade id. Thus
fixations/saccades below are reproducible I-VT *proxies*, not device-native
clinical eye-event labels. The primary setting follows Tobii XR's recommended
I-VT configuration: 30 deg/s and 100 ms. Threshold sensitivity is retained.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DATA = OUT / "data"
TABLE = OUT / "tables"
NOTE = OUT / "notes"
for directory in (DATA, TABLE, NOTE):
    directory.mkdir(parents=True, exist_ok=True)

EYE_COLUMNS = [
    "TimeStamp", "HeadRotX", "HeadRotY", "HeadRotZ",
    "GazeDirX", "GazeDirY", "GazeDirZ",
    "Hit1_X", "Hit1_Y", "Hit1_Z", "Hit1_Name",
    "Hit2_X", "Hit2_Y", "Hit2_Z", "Hit2_Name",
    "Hit3_X", "Hit3_Y", "Hit3_Z", "Hit3_Name",
]
TIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
KEYS = ["Scenario", "ParticipantID", "FileName"]
PRIMARY_GAZE_VELOCITY = 30.0
PRIMARY_FIXATION_MIN_S = 0.100
PRIMARY_HEAD_SEARCH_VELOCITY = 60.0
MAX_CONTIGUOUS_GAP_S = 0.100
SMOOTH_WINDOW = 3
AOI_EXCLUDED = {"LAT_A2", "LXC_C1", "MC_B2"}


def read_eye(path: Path) -> pd.DataFrame:
    # Rows contain the 19 declared fields plus one empty trailing field.
    return pd.read_csv(
        path,
        names=EYE_COLUMNS + ["_Trailing"],
        skiprows=1,
        low_memory=False,
    )


def to_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0).to_numpy(bool)
    return (
        series.fillna("").astype(str).str.strip().str.lower()
        .isin({"1", "true", "yes", "y"}).to_numpy(bool)
    )


def timestamps(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parsed = pd.to_datetime(values.astype(str), format=TIME_FORMAT, errors="coerce")
    valid = parsed.notna().to_numpy()
    ns = parsed.astype("int64").to_numpy()
    if valid.sum() < 2:
        raise ValueError("fewer than two valid timestamps")
    first = ns[valid][0]
    elapsed = (ns - first) / 1e9
    elapsed[~valid] = np.nan
    dt = np.r_[np.nan, np.diff(elapsed)]
    positive = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive))
    weights = np.r_[np.diff(elapsed), median_dt]
    weights[~np.isfinite(weights) | (weights <= 0)] = median_dt
    weights = np.minimum(weights, MAX_CONTIGUOUS_GAP_S)
    return elapsed, dt, weights


def normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & (norm > 0.5) & (norm < 1.5)
    out = np.full_like(values, np.nan, dtype=float)
    out[valid] = values[valid] / norm[valid, None]
    return out, valid


def smooth_unit_vectors(vectors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full_like(vectors, np.nan, dtype=float)
    idx = np.flatnonzero(valid)
    if not len(idx):
        return out
    split = np.flatnonzero(np.diff(idx) > 1) + 1
    for group in np.split(idx, split):
        if not len(group):
            continue
        block = pd.DataFrame(vectors[group]).rolling(
            SMOOTH_WINDOW, center=True, min_periods=1
        ).mean().to_numpy(float)
        norm = np.linalg.norm(block, axis=1)
        okay = norm > 0
        out[group[okay]] = block[okay] / norm[okay, None]
    return out


def angular_steps(vectors: np.ndarray, valid: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.full(len(vectors), np.nan)
    pair = valid & np.r_[False, valid[:-1]] & np.isfinite(dt) & (dt > 0) & (dt <= MAX_CONTIGUOUS_GAP_S)
    indices = np.flatnonzero(pair)
    if len(indices):
        dot = np.sum(vectors[indices] * vectors[indices - 1], axis=1)
        angle[indices] = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    speed = angle / dt
    return angle, speed


def contiguous_events(mask: np.ndarray, elapsed: np.ndarray, weights: np.ndarray, min_duration: float) -> list[np.ndarray]:
    idx = np.flatnonzero(mask & np.isfinite(elapsed))
    if not len(idx):
        return []
    split = np.flatnonzero((np.diff(idx) > 1) | (np.diff(elapsed[idx]) > MAX_CONTIGUOUS_GAP_S)) + 1
    events = []
    for group in np.split(idx, split):
        if float(weights[group].sum()) >= min_duration:
            events.append(group)
    return events


def spherical_dispersion(vectors: np.ndarray, valid: np.ndarray) -> float:
    v = vectors[valid]
    if len(v) < 3:
        return np.nan
    center = v.mean(axis=0)
    norm = np.linalg.norm(center)
    if norm <= 1e-9:
        return 180.0
    center /= norm
    angles = np.degrees(np.arccos(np.clip(v @ center, -1, 1)))
    return float(np.quantile(angles, 0.90))


def direction_entropy(local_vectors: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> float:
    v = local_vectors[valid]
    w = weights[valid]
    if len(v) < 3 or w.sum() <= 0:
        return np.nan
    yaw = np.degrees(np.arctan2(v[:, 0], v[:, 2]))
    pitch = np.degrees(np.arcsin(np.clip(v[:, 1], -1, 1)))
    hist, _, _ = np.histogram2d(
        yaw, pitch,
        bins=[np.linspace(-180, 180, 13), np.linspace(-90, 90, 7)],
        weights=w,
    )
    p = hist.ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(72))


def event_metrics(
    local_speed: np.ndarray,
    local_angle: np.ndarray,
    elapsed: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    min_fixation_s: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    eligible = valid & np.isfinite(local_speed)
    fixation_events = contiguous_events(
        eligible & (local_speed < threshold), elapsed, weights, min_fixation_s
    )
    saccade_events = contiguous_events(
        eligible & (local_speed >= threshold), elapsed, weights, 0.020
    )
    fixation_mask = np.zeros(len(valid), bool)
    for event in fixation_events:
        fixation_mask[event] = True
    saccade_start = np.zeros(len(valid), bool)
    amplitudes = []
    for event in saccade_events:
        saccade_start[event[0]] = True
        amplitudes.append(float(np.nansum(local_angle[event])))
    valid_duration = float(weights[valid].sum())
    minute = valid_duration / 60 if valid_duration > 0 else np.nan
    fixation_duration = np.array([weights[e].sum() for e in fixation_events], float)
    return {
        "FixationRate_per_min": len(fixation_events) / minute if minute > 0 else np.nan,
        "MedianFixationDuration_ms": float(np.median(fixation_duration) * 1000) if len(fixation_duration) else np.nan,
        "FixationTimeProportion": float(weights[fixation_mask].sum() / valid_duration) if valid_duration > 0 else np.nan,
        "SaccadeRate_per_min": len(saccade_events) / minute if minute > 0 else np.nan,
        "MedianSaccadeAmplitude_deg": float(np.median(amplitudes)) if amplitudes else np.nan,
    }, fixation_mask, saccade_start


def derive_eye_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    elapsed, dt, weights = timestamps(frame["TimeStamp"])
    gaze_raw = frame[["GazeDirX", "GazeDirY", "GazeDirZ"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    world, gaze_valid = normalize_rows(gaze_raw)
    head = frame[["HeadRotX", "HeadRotY", "HeadRotZ"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = gaze_valid & np.isfinite(head).all(axis=1) & np.isfinite(elapsed)
    world = smooth_unit_vectors(world, valid)
    valid &= np.isfinite(world).all(axis=1)
    # Replace non-finite Euler rows only for Rotation construction; those rows
    # remain excluded by ``valid`` and never contribute to metrics.
    safe_head = np.where(np.isfinite(head), head, 0.0)
    rotations = Rotation.from_euler("zxy", safe_head[:, [2, 0, 1]], degrees=True)
    local = np.full_like(world, np.nan)
    if valid.any():
        local[valid] = rotations[valid].inv().apply(world[valid])
    local, local_valid = normalize_rows(local)
    valid &= local_valid
    local = smooth_unit_vectors(local, valid)
    world_angle, world_speed = angular_steps(world, valid, dt)
    local_angle, local_speed = angular_steps(local, valid, dt)
    # Quaternion increments avoid Euler wraparound/gimbal artefacts.
    forward = rotations.apply(np.tile([0.0, 0.0, 1.0], (len(frame), 1)))
    head_angle, head_speed = angular_steps(forward, valid, dt)
    alignment = np.full(len(frame), np.nan)
    alignment[valid] = np.degrees(
        np.arccos(np.clip(np.sum(forward[valid] * world[valid], axis=1), -1, 1))
    )
    return {
        "elapsed": elapsed, "dt": dt, "weights": weights, "valid": valid,
        "world": world, "local": local,
        "world_angle": world_angle, "world_speed": world_speed,
        "local_angle": local_angle, "local_speed": local_speed,
        "head_angle": head_angle, "head_speed": head_speed,
        "alignment": alignment,
        "raw_gaze_norm": np.linalg.norm(gaze_raw, axis=1),
    }


def metrics_for_selection(arr: dict[str, np.ndarray], selected: np.ndarray, threshold: float = PRIMARY_GAZE_VELOCITY, min_fixation_s: float = PRIMARY_FIXATION_MIN_S) -> dict[str, float]:
    valid = arr["valid"] & selected
    weights = arr["weights"]
    valid_duration = float(weights[valid].sum())
    ev, _, _ = event_metrics(
        arr["local_speed"], arr["local_angle"], arr["elapsed"], weights,
        valid, threshold, min_fixation_s,
    )
    transition = valid & np.isfinite(arr["dt"]) & (arr["dt"] > 0) & (arr["dt"] <= MAX_CONTIGUOUS_GAP_S)
    head_path = float(np.nansum(arr["head_angle"][transition]))
    eye_path = float(np.nansum(arr["local_angle"][transition]))
    world_path = float(np.nansum(arr["world_angle"][transition]))
    rapid_head = valid & np.isfinite(arr["head_speed"]) & (arr["head_speed"] >= PRIMARY_HEAD_SEARCH_VELOCITY)
    out = {
        "ValidEyeDuration_s": valid_duration,
        "HeadAngularPathRate_deg_s": head_path / valid_duration if valid_duration > 0 else np.nan,
        "RapidHeadSearchProportion": float(weights[rapid_head].sum() / valid_duration) if valid_duration > 0 else np.nan,
        "WorldGazeScanpathRate_deg_s": world_path / valid_duration if valid_duration > 0 else np.nan,
        "EyeGazeScanpathRate_deg_s": eye_path / valid_duration if valid_duration > 0 else np.nan,
        # Head-relative dispersion avoids labelling a normal corridor turn as
        # broad visual search merely because the whole world direction changed.
        "GazeP90Dispersion_deg": spherical_dispersion(arr["local"], valid),
        "GazeDirectionEntropy": direction_entropy(arr["local"], valid, weights),
        "EyeHeadCoordinationRatio": head_path / (head_path + eye_path) if (head_path + eye_path) > 0 else np.nan,
    }
    out.update(ev)
    return out


def integrated_path(row: pd.Series) -> Path | None:
    stamp = row["FileName"].replace("_Subject_Simulation_Info_", "|").replace(".csv", "").split("|", 1)[1]
    trial = row["FileName"].split("_", 1)[0]
    folder = ROOT / "Results_Guidance_GazeAOI_Integrated" / row["Scenario"] / row["ParticipantID"]
    matches = list(folder.glob(f"{trial}_GuidanceGazeIntegrated_{stamp}.csv"))
    return matches[0] if len(matches) == 1 else None


def aoi_columns(guidance: str) -> tuple[list[str], str]:
    if guidance == "ArrowRibbon":
        return ["ValidLightBlanketAOI"], "GazeOnLightBlanket"
    if guidance == "MiniMap":
        return ["ValidMiniMapAOI"], "GazeOnMiniMap"
    if guidance == "TextBar":
        return ["ValidTextAOI"], "GazeOnTextGuidance"
    if guidance == "Full":
        return ["ValidLightBlanketAOI", "ValidMiniMapAOI", "ValidTextAOI"], "AnyGuidanceGaze"
    raise ValueError(guidance)


def aoi_metrics(row: pd.Series, trial_duration: float) -> tuple[dict, list[dict]]:
    path = integrated_path(row)
    base = {**{k: row[k] for k in KEYS}, "Guidance": row["Guidance"], "Smoke": row["Smoke"]}
    if path is None:
        return {**base, "AOIFileMatched": 0, "AOIAvailable": 0}, []
    frame = pd.read_csv(path, low_memory=False)
    elapsed, _, weights = timestamps(frame["TimeStamp"])
    valid_columns, gaze_column = aoi_columns(str(row["Guidance"]))
    valid = np.logical_or.reduce([to_bool(frame[c]) for c in valid_columns]) & np.isfinite(elapsed)
    gaze = to_bool(frame[gaze_column]) & valid
    valid_duration = float(weights[valid].sum())
    gaze_duration = float(weights[gaze].sum())
    events = contiguous_events(gaze, elapsed, weights, 0.100)
    acquired = int(len(events) > 0)
    first_latency = float(elapsed[events[0][0]]) if events else np.nan
    duration = max(float(trial_duration), float(np.nanmax(elapsed)))
    prompt_score = max(0.0, 1.0 - first_latency / duration) if acquired else 0.0
    record = {
        **base,
        "AOIFileMatched": 1,
        "AOIAvailable": int(valid_duration > 0),
        "AOIValidTimeProportion": valid_duration / duration if duration > 0 else np.nan,
        "GuideAcquired": acquired if valid_duration > 0 else np.nan,
        "FirstGuideLatency_s": first_latency,
        "FirstGuideLatencyNormalized": first_latency / duration if acquired and duration > 0 else np.nan,
        "PromptGuideAcquisitionScore": prompt_score if valid_duration > 0 else np.nan,
        "GuideDwellDuration_s": gaze_duration,
        "GuideDwellProportion": gaze_duration / valid_duration if valid_duration > 0 else np.nan,
        "GuideVisitCount": len(events),
        "GuideVisitRate_per_min": len(events) / (valid_duration / 60) if valid_duration > 0 else np.nan,
        "MedianGuideVisitDuration_s": float(np.median([weights[e].sum() for e in events])) if events else np.nan,
    }
    phase = []
    normalized = np.clip(elapsed / duration, 0, 0.999999)
    bins = np.floor(normalized * 10).astype(int)
    for b in range(10):
        chosen = bins == b
        denom = float(weights[chosen & valid].sum())
        numer = float(weights[chosen & gaze].sum())
        phase.append({
            **base,
            "TrialProgressBin": b + 1,
            "TrialProgressMid": (b + 0.5) / 10,
            "AOIValidDuration_s": denom,
            "GuideDwellProportion": numer / denom if denom > 0 else np.nan,
        })
    return record, phase


def extract(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows, phase_rows, quality_rows, threshold_rows, head_threshold_rows = [], [], [], [], []
    aoi_rows, aoi_phase_rows = [], []
    for number, (_, row) in enumerate(trials.iterrows(), 1):
        eye_name = row["FileName"].replace("_Subject_Simulation_Info_", "_EyeGaze_Simulation_Info_")
        eye_path = RAW / row["Scenario"] / row["ParticipantID"] / eye_name
        if not eye_path.exists():
            quality_rows.append({**{k: row[k] for k in KEYS}, "EyeFileMatched": 0})
            continue
        frame = read_eye(eye_path)
        arr = derive_eye_arrays(frame)
        all_selected = np.ones(len(frame), bool)
        metrics = metrics_for_selection(arr, all_selected)
        base = {
            **{k: row[k] for k in KEYS},
            "Guidance": row["Guidance"], "Smoke": row["Smoke"],
            "TargetExit": row["TargetExit"],
        }
        feature_rows.append({**base, **metrics})
        elapsed = arr["elapsed"]
        duration = max(float(row["EvacuationTime_s"]), float(np.nanmax(elapsed)))
        bins = np.floor(np.clip(elapsed / duration, 0, 0.999999) * 10).astype(int)
        for b in range(10):
            chosen = bins == b
            phase_rows.append({
                **base,
                "TrialProgressBin": b + 1,
                "TrialProgressMid": (b + 0.5) / 10,
                **metrics_for_selection(arr, chosen),
            })
        valid = arr["valid"]
        positive_dt = arr["dt"][np.isfinite(arr["dt"]) & (arr["dt"] > 0)]
        quality_rows.append({
            **base,
            "EyeFileMatched": 1,
            "RawRowN": len(frame),
            "ValidEyeSampleN": int(valid.sum()),
            "ValidEyeSampleProportion": float(valid.mean()),
            "MedianSampleInterval_ms": float(np.median(positive_dt) * 1000),
            "P95SampleInterval_ms": float(np.quantile(positive_dt, 0.95) * 1000),
            "LargeGapProportion": float((positive_dt > MAX_CONTIGUOUS_GAP_S).mean()),
            "ValidDurationCoverage": metrics["ValidEyeDuration_s"] / duration,
            "MedianRawGazeNorm": float(np.nanmedian(arr["raw_gaze_norm"][valid])),
            "MedianHeadGazeAlignment_deg": float(np.nanmedian(arr["alignment"][valid])),
            "TrailingCommaParserApplied": 1,
        })
        for threshold in [30.0, 60.0, 100.0]:
            for min_fix in [0.080, 0.100, 0.120]:
                ev, _, _ = event_metrics(
                    arr["local_speed"], arr["local_angle"], arr["elapsed"], arr["weights"],
                    valid, threshold, min_fix,
                )
                threshold_rows.append({
                    **base, "VelocityThreshold_deg_s": threshold,
                    "MinimumFixationDuration_ms": min_fix * 1000, **ev,
                })
        for threshold in [30.0, 60.0, 90.0]:
            rapid = valid & np.isfinite(arr["head_speed"]) & (arr["head_speed"] >= threshold)
            valid_weight = float(arr["weights"][valid].sum())
            head_threshold_rows.append({
                **base, "HeadSearchThreshold_deg_s": threshold,
                "RapidHeadSearchProportion": float(arr["weights"][rapid].sum() / valid_weight) if valid_weight > 0 else np.nan,
            })
        if str(row["Guidance"]) != "StaticSign":
            aoi, phases = aoi_metrics(row, duration)
            aoi_rows.append(aoi)
            aoi_phase_rows.extend(phases)
        if number % 25 == 0:
            print(f"  eye feature extraction {number}/{len(trials)}")
    features = pd.DataFrame(feature_rows)
    phases = pd.DataFrame(phase_rows)
    quality = pd.DataFrame(quality_rows)
    thresholds = pd.DataFrame(threshold_rows)
    head_thresholds = pd.DataFrame(head_threshold_rows)
    aoi = pd.DataFrame(aoi_rows)
    aoi_phases = pd.DataFrame(aoi_phase_rows)
    return features, phases, quality, thresholds, head_thresholds, aoi, aoi_phases


def summarize_sensitivity(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    return frame.groupby(group_cols, as_index=False, observed=True).agg(
        TrialN=("Scenario", "size"),
        **{f"Mean_{m}": (m, "mean") for m in metrics},
    )


def feature_definitions() -> pd.DataFrame:
    rows = [
        ("HeadAngularPathRate_deg_s", "头部搜索路径率", "头朝向每秒累计转过的角度；越大表示更频繁地转头搜索。", "五模式"),
        ("RapidHeadSearchProportion", "快速头部搜索比例", "头部角速度>=60°/s的有效时长占比。", "五模式"),
        ("FixationRate_per_min", "凝视事件率", "I-VT代理：眼在头内角速度<100°/s并持续>=100ms的事件数/分钟。", "五模式"),
        ("MedianFixationDuration_ms", "凝视中位持续时间", "符合上述规则的凝视事件中位时长。", "五模式"),
        ("FixationTimeProportion", "凝视时间比例", "符合上述规则的凝视时长/有效眼动时长。", "五模式"),
        ("SaccadeRate_per_min", "快速眼跳事件率", "I-VT代理：眼在头内角速度>=100°/s且持续>=20ms的事件数/分钟。", "五模式"),
        ("MedianSaccadeAmplitude_deg", "眼跳中位幅度", "快速眼跳事件内累计眼在头内转动角度的中位数。", "五模式"),
        ("WorldGazeScanpathRate_deg_s", "扫视路径率", "世界坐标下注视方向每秒累计角路径；越大表示视觉扫描更繁复。", "五模式"),
        ("EyeGazeScanpathRate_deg_s", "眼在头内扫视路径率", "扣除头部朝向后，眼动方向每秒累计角路径。", "五模式"),
        ("GazeP90Dispersion_deg", "注视分散程度", "扣除头部朝向后，注视方向相对球面中心的90%角半径。", "五模式"),
        ("GazeDirectionEntropy", "视觉搜索熵", "眼在头内方向落入角度网格后的标准化熵；越大表示搜索更分散。", "五模式"),
        ("EyeHeadCoordinationRatio", "眼头协同比", "头部角路径/(头部角路径+眼在头内角路径)。", "五模式"),
        ("GuideAcquired", "是否获取引导", "有效AOI期间是否出现持续>=100ms的引导访问。", "四种AR"),
        ("FirstGuideLatency_s", "首次获取引导时间", "从试次开始到第一次持续引导访问的秒数；从未获取则缺失并由单独指示变量表达。", "四种AR"),
        ("PromptGuideAcquisitionScore", "及时获取得分", "获取者为1-首次时间/试次时长；从未获取为0。", "四种AR"),
        ("GuideDwellProportion", "引导停留比例", "AOI有效时长中注视引导的时间加权比例。", "四种AR"),
        ("GuideVisitRate_per_min", "引导访问率", "持续>=100ms的引导AOI访问事件数/有效分钟。", "四种AR"),
    ]
    return pd.DataFrame(rows, columns=["Feature", "中文概念", "操作定义", "适用范围"])


def main() -> None:
    trials = pd.read_csv(DATA / "analysis_trials_all_288.csv")
    features, phases, quality, thresholds, head_thresholds, aoi, aoi_phases = extract(trials)
    features.to_csv(DATA / "visual_information_trial_features_all_288.csv", index=False, encoding="utf-8-sig")
    phases.to_csv(DATA / "visual_information_progress_bins_all_288.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(TABLE / "eye_tracking_data_quality_audit.csv", index=False, encoding="utf-8-sig")
    aoi.to_csv(DATA / "ar_information_acquisition_features_all_288.csv", index=False, encoding="utf-8-sig")
    aoi_phases.to_csv(DATA / "ar_information_acquisition_progress_bins_all_288.csv", index=False, encoding="utf-8-sig")

    primary_features = features.loc[~features["ParticipantID"].isin(AOI_EXCLUDED)].copy()
    primary_phases = phases.loc[~phases["ParticipantID"].isin(AOI_EXCLUDED)].copy()
    primary_aoi = aoi.loc[~aoi["ParticipantID"].isin(AOI_EXCLUDED)].copy()
    primary_aoi_phases = aoi_phases.loc[~aoi_phases["ParticipantID"].isin(AOI_EXCLUDED)].copy()
    primary_features.to_csv(DATA / "visual_information_trial_features.csv", index=False, encoding="utf-8-sig")
    primary_phases.to_csv(DATA / "visual_information_progress_bins.csv", index=False, encoding="utf-8-sig")
    primary_aoi.to_csv(DATA / "ar_information_acquisition_features.csv", index=False, encoding="utf-8-sig")
    primary_aoi_phases.to_csv(DATA / "ar_information_acquisition_progress_bins.csv", index=False, encoding="utf-8-sig")

    threshold_summary = summarize_sensitivity(
        thresholds.loc[~thresholds["ParticipantID"].isin(AOI_EXCLUDED)],
        ["Guidance", "Smoke", "VelocityThreshold_deg_s", "MinimumFixationDuration_ms"],
        ["FixationRate_per_min", "MedianFixationDuration_ms", "FixationTimeProportion", "SaccadeRate_per_min", "MedianSaccadeAmplitude_deg"],
    )
    threshold_summary.to_csv(TABLE / "eye_event_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    head_summary = summarize_sensitivity(
        head_thresholds.loc[~head_thresholds["ParticipantID"].isin(AOI_EXCLUDED)],
        ["Guidance", "Smoke", "HeadSearchThreshold_deg_s"],
        ["RapidHeadSearchProportion"],
    )
    head_summary.to_csv(TABLE / "head_search_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    feature_definitions().to_csv(TABLE / "visual_information_feature_definitions.csv", index=False, encoding="utf-8-sig")

    common_summary = primary_features.groupby(["Guidance", "Smoke"], as_index=False, observed=True).agg(
        TrialN=("Scenario", "size"), ParticipantN=("ParticipantID", "nunique"),
        MeanHeadSearch=("HeadAngularPathRate_deg_s", "mean"),
        MeanFixationProportion=("FixationTimeProportion", "mean"),
        MeanSaccadeRate=("SaccadeRate_per_min", "mean"),
        MeanScanpathRate=("WorldGazeScanpathRate_deg_s", "mean"),
        MeanEyeScanpathRate=("EyeGazeScanpathRate_deg_s", "mean"),
        MeanDispersion=("GazeP90Dispersion_deg", "mean"),
        MeanEntropy=("GazeDirectionEntropy", "mean"),
    )
    common_summary.to_csv(TABLE / "five_mode_visual_information_summary.csv", index=False, encoding="utf-8-sig")
    ar_summary = primary_aoi.groupby(["Guidance", "Smoke"], as_index=False, observed=True).agg(
        TrialN=("Scenario", "size"), ParticipantN=("ParticipantID", "nunique"),
        AOIAvailableTrialN=("AOIAvailable", "sum"),
        AcquiredTrialN=("GuideAcquired", "sum"),
        GuideAcquisitionRate=("GuideAcquired", "mean"),
        MeanFirstGuideLatency_s=("FirstGuideLatency_s", "mean"),
        MeanPromptAcquisitionScore=("PromptGuideAcquisitionScore", "mean"),
        MeanGuideDwellProportion=("GuideDwellProportion", "mean"),
        MeanGuideVisitRate_per_min=("GuideVisitRate_per_min", "mean"),
        MeanAOIValidTimeProportion=("AOIValidTimeProportion", "mean"),
    )
    ar_summary.to_csv(TABLE / "ar_information_acquisition_summary.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "raw_eye_trial_expected": int(len(trials)),
        "raw_eye_trial_matched": int(quality["EyeFileMatched"].fillna(0).sum()),
        "primary_visual_trial_n": int(len(primary_features)),
        "primary_ar_aoi_trial_n": int(len(primary_aoi)),
        "primary_ar_aoi_available_trial_n": int(primary_aoi["AOIAvailable"].fillna(0).sum()),
        "primary_ar_never_acquired_trial_n": int(((primary_aoi["AOIAvailable"] == 1) & (primary_aoi["GuideAcquired"] == 0)).sum()),
        "median_valid_eye_sample_proportion": float(quality["ValidEyeSampleProportion"].median()),
        "median_sampling_interval_ms": float(quality["MedianSampleInterval_ms"].median()),
        "primary_ivt_velocity_deg_s": PRIMARY_GAZE_VELOCITY,
        "primary_fixation_min_ms": PRIMARY_FIXATION_MIN_S * 1000,
        "primary_head_search_velocity_deg_s": PRIMARY_HEAD_SEARCH_VELOCITY,
        "phase_axis": "normalized trial time in ten equal bins",
    }
    (DATA / "visual_information_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
