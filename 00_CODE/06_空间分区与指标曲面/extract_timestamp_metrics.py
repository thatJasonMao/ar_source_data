"""Build the formal timestamp-local spatial-metric input bundle.

No progress bins or position downsampling. Interval angles and exposure are
assigned to the interval midpoint. Saccades reproduce EyeIndicatorAssessment.py
once per complete trial and are assigned to the 2 m cell at event onset. The
head metric measures change in forward direction, not full 3-axis rotation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from SemanticZoneQuery import SemanticZoneQuery
from SpatialSlice_TrajPoint import VectorizedZoneClassifier

TIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
KEYS = ["Scenario", "ParticipantID", "FileName"]
CONFIG = dict(cell_m=2.0, max_gap_s=.1, max_position_step_m=3.0,
              gaze_smooth_samples=3,
              saccade_sample_rate_hz=45.0,
              saccade_min_velocity_deg_s=100.0,
              saccade_min_duration_s=.020,
              saccade_min_amplitude_deg=30.0,
              saccade_noise_multiplier=5.0,
              saccade_window_size=2,
              saccade_filter_cutoff_hz=10.0,
              saccade_filter_order=3,
              saccade_filter_enabled=False,
              min_cell_exposure_s=1.0, min_floor_exposure_s=5.0,
              eye_height_m=0.0, schema_version=4)
METRICS = {
    "head_search": ("HeadAngle_deg", "HeadTime_s", 1.0, "deg/s"),
    "gaze_scan_speed": ("EyeAngle_deg", "EyeTime_s", 1.0, "deg/s"),
    "gaze_depth": ("DepthIntegral_m_s", "DepthTime_s", 1.0, "m"),
    "saccade_rate": ("SaccadeCount", "SaccadeExposure_s", 60.0, "events/min"),
}


def read_raw(path: Path, required: list[str]):
    """Explicit columns prevent trailing CSV delimiters from shifting fields."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header, first = next(reader), next(reader)
    names = [name or f"_Empty_{i}" for i, name in enumerate(header)]
    names += [f"_Trailing_{i}" for i in range(max(0, len(first)-len(names)))]
    frame = pd.read_csv(path, names=names, skiprows=1, usecols=required,
                        low_memory=False, encoding="utf-8-sig")
    raw_n = len(frame)
    frame["ParsedTime"] = pd.to_datetime(frame.TimeStamp, format=TIME_FORMAT, errors="coerce")
    bad_time = int(frame.ParsedTime.isna().sum())
    frame = frame.dropna(subset=["ParsedTime"]).sort_values("ParsedTime")
    duplicates = int(frame.ParsedTime.duplicated(keep=False).sum())
    # Ambiguous timestamps are excluded rather than arbitrarily choosing a row.
    frame = frame.loc[~frame.ParsedTime.duplicated(keep=False)].copy()
    for col in required:
        if col != "TimeStamp":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.reset_index(drop=True), dict(raw_rows=raw_n, invalid_time=bad_time,
                                             duplicate_time_rows_excluded=duplicates)


def align_positions(eye_ns, pos_ns, positions, max_gap_s=.1):
    """Exact timestamp join, then bracketed interpolation only across short gaps."""
    n = len(eye_ns)
    result = np.full((n, 3), np.nan)
    method = np.zeros(n, dtype=np.int8)  # 0 missing, 1 exact, 2 interpolated
    right = np.searchsorted(pos_ns, eye_ns, side="left")
    possible = right < len(pos_ns)
    exact = possible.copy()
    exact[possible] = pos_ns[right[possible]] == eye_ns[possible]
    idx = np.flatnonzero(exact)
    okay = np.isfinite(positions[right[idx]]).all(axis=1)
    idx = idx[okay]
    result[idx] = positions[right[idx]]
    method[idx] = 1
    idx = np.flatnonzero(~exact & (right > 0) & (right < len(pos_ns)))
    left_index, right_index = right[idx]-1, right[idx]
    gap = (pos_ns[right_index]-pos_ns[left_index])/1e9
    okay = ((gap > 0) & (gap <= max_gap_s)
            & np.isfinite(positions[left_index]).all(axis=1)
            & np.isfinite(positions[right_index]).all(axis=1))
    idx, left_index, right_index = idx[okay], left_index[okay], right_index[okay]
    fraction = (eye_ns[idx]-pos_ns[left_index])/(pos_ns[right_index]-pos_ns[left_index])
    result[idx] = positions[left_index] + fraction[:, None]*(positions[right_index]-positions[left_index])
    method[idx] = 2
    return result, method


def unit_vectors(raw):
    norm = np.linalg.norm(raw, axis=1)
    valid = np.isfinite(raw).all(axis=1) & (norm > .5) & (norm < 1.5)
    result = np.full_like(raw, np.nan)
    result[valid] = raw[valid]/norm[valid, None]
    return result, valid


def smooth_vectors(vectors, valid, times, window=3):
    """Short local-vector smoothing never bridges invalid frames or time gaps."""
    out = np.full_like(vectors, np.nan)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return out
    split = np.flatnonzero((np.diff(indices) != 1)
                          | (np.diff(times[indices]) > CONFIG["max_gap_s"]))+1
    for run in np.split(indices, split):
        smooth = pd.DataFrame(vectors[run]).rolling(window, center=True, min_periods=1).mean().to_numpy()
        norms = np.linalg.norm(smooth, axis=1)
        good = norms > 1e-12
        out[run[good]] = smooth[good]/norms[good, None]
    return out


def interval_angles(vectors, valid, dt):
    angles = np.full(len(vectors), np.nan)
    pair = valid & np.r_[False, valid[:-1]] & (dt > 0) & (dt <= CONFIG["max_gap_s"])
    idx = np.flatnonzero(pair)
    angles[idx] = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", vectors[idx-1], vectors[idx]), -1, 1)))
    return angles


def detect_reference_saccades(local_hit_positions):
    """Reproduce EyeIndicatorAssessment.detect_saccades without corrections."""
    positions = np.asarray(local_hit_positions, dtype=float)
    valid_mask = (np.isfinite(positions) & (positions != 0)).all(axis=1)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 3:
        return [], {"baseline_noise_mad_deg_s": np.nan,
                    "velocity_threshold_deg_s": np.nan,
                    "valid_indices": valid_indices}

    # EyeIndicatorAssessment defines a Butterworth filter but comments out its
    # execution. Preserve the actual behavior by using the unfiltered points.
    filtered_positions = positions[valid_indices]
    sample_rate = float(CONFIG["saccade_sample_rate_hz"])
    dt = 1.0 / sample_rate
    k = int(CONFIG["saccade_window_size"])
    velocities = []
    for i in range(k, len(filtered_positions) - k):
        pos_forward = filtered_positions[i + k]
        pos_backward = filtered_positions[i - k]
        distance_eye = np.linalg.norm(pos_forward - pos_backward)
        distance = (np.linalg.norm(pos_forward) + np.linalg.norm(pos_backward)) / 2
        radian = np.arccos(
            (2 * distance ** 2 - distance_eye ** 2) / (2 * distance ** 2)
        )
        velocities.append(np.degrees(radian) / (2 * k * dt))
    if not velocities:
        return [], {"baseline_noise_mad_deg_s": np.nan,
                    "velocity_threshold_deg_s": np.nan,
                    "valid_indices": valid_indices}

    velocities = np.asarray(velocities)
    baseline_noise = np.median(
        np.abs(velocities - np.nanmedian(velocities))
    )
    if baseline_noise > 100:
        velocity_threshold = 300.0
    else:
        velocity_threshold = max(
            float(CONFIG["saccade_min_velocity_deg_s"]),
            float(CONFIG["saccade_noise_multiplier"]) * baseline_noise,
        )

    events = []
    in_saccade = False
    saccade_start = 0
    saccade_peak_velocity = 0.0
    for i, velocity in enumerate(velocities):
        if velocity > velocity_threshold:
            if not in_saccade:
                in_saccade = True
                saccade_start = i
                saccade_peak_velocity = velocity
            else:
                saccade_peak_velocity = max(saccade_peak_velocity, velocity)
        elif in_saccade:
            # These indices and the +1 duration intentionally reproduce the
            # reference implementation, including its first-below-threshold row.
            saccade_end = i
            duration = (saccade_end - saccade_start + 1) * dt
            if duration > float(CONFIG["saccade_min_duration_s"]):
                start_position = filtered_positions[saccade_start + k]
                end_position = filtered_positions[saccade_end + k]
                distance_eye = np.linalg.norm(start_position - end_position)
                distance_subject = (
                    np.linalg.norm(start_position) + np.linalg.norm(end_position)
                ) / 2
                amplitude = np.degrees(np.arccos(
                    (2 * distance_subject ** 2 - distance_eye ** 2)
                    / (2 * distance_subject ** 2)
                ))
                if amplitude >= float(CONFIG["saccade_min_amplitude_deg"]):
                    events.append({
                        "start_idx": int(saccade_start + k),
                        "end_idx": int(saccade_end + k),
                        "duration_s": float(duration),
                        "peak_velocity_deg_s": float(saccade_peak_velocity),
                        "amplitude_deg": float(amplitude),
                    })
            in_saccade = False

    if in_saccade:
        saccade_end = len(velocities) - 1
        duration = (saccade_end - saccade_start + 1) * dt
        if duration > float(CONFIG["saccade_min_duration_s"]):
            start_position = filtered_positions[saccade_start + k]
            end_position = filtered_positions[saccade_end + k]
            distance_eye = np.linalg.norm(start_position - end_position)
            distance_subject = (
                np.linalg.norm(start_position) + np.linalg.norm(end_position)
            ) / 2
            amplitude = np.degrees(np.arccos(
                (2 * distance_subject ** 2 - distance_eye ** 2)
                / (2 * distance_subject ** 2)
            ))
            if amplitude >= float(CONFIG["saccade_min_amplitude_deg"]):
                events.append({
                    "start_idx": int(saccade_start + k),
                    "end_idx": int(saccade_end + k),
                    "duration_s": float(duration),
                    "peak_velocity_deg_s": float(saccade_peak_velocity),
                    "amplitude_deg": float(amplitude),
                })
    return events, {
        "baseline_noise_mad_deg_s": float(baseline_noise),
        "velocity_threshold_deg_s": float(velocity_threshold),
        "valid_indices": valid_indices,
    }


def layer_of(zone_type):
    if zone_type in {"Platform", "Spawn", "StairEntrance"}:
        return "platform"
    return "stair" if zone_type == "Stair" else "hall"


def process_trial(trial, classifier):
    folder = ROOT / "RAW_DATA" / trial.Scenario / str(trial.SubjectFolder)
    if not (folder / trial.FileName).exists():
        folder = ROOT / "RAW_DATA" / trial.Scenario / str(trial.ParticipantID)
    subject_path = folder / trial.FileName
    eye_path = folder / trial.FileName.replace("_Subject_", "_EyeGaze_")
    pos, pq = read_raw(subject_path, ["TimeStamp", "PosX", "PosY", "PosZ"])
    eye, eq = read_raw(eye_path, ["TimeStamp", "HeadRotX", "HeadRotY", "HeadRotZ",
                                 "GazeDirX", "GazeDirY", "GazeDirZ", "Hit1_X", "Hit1_Y", "Hit1_Z"])
    if len(eye) < 2 or len(pos) < 2:
        raise ValueError(f"Too few samples: {eye_path}")
    ens = eye.ParsedTime.astype("int64").to_numpy()
    pns = pos.ParsedTime.astype("int64").to_numpy()
    origin = int(ens[0])
    times = (ens-origin)/1e9
    dt = np.r_[np.nan, np.diff(times)]
    positions, sync_method = align_positions(ens, pns, pos[["PosX", "PosY", "PosZ"]].to_numpy(float))
    start = (pd.to_datetime(trial.StartTime, format=TIME_FORMAT).value-origin)/1e9
    end = (pd.to_datetime(trial.EndTime, format=TIME_FORMAT).value-origin)/1e9
    if not end > start:
        raise ValueError("Invalid trial bounds")
    previous_time = np.r_[times[0], times[:-1]]
    overlap_start = np.maximum(previous_time, start)
    overlap_end = np.minimum(times, end)
    overlap = np.maximum(0, overlap_end-overlap_start)
    position_valid = np.isfinite(positions).all(axis=1)
    motion_valid = position_valid & np.r_[False, position_valid[:-1]]
    step_distance = np.r_[np.nan, np.linalg.norm(np.diff(positions, axis=0), axis=1)]
    motion_valid &= ((dt > 0) & (dt <= CONFIG["max_gap_s"])
                     & (step_distance <= CONFIG["max_position_step_m"]))
    overlap = np.where(motion_valid, overlap, 0.)

    head = eye[["HeadRotX", "HeadRotY", "HeadRotZ"]].to_numpy(float)
    head_valid = np.isfinite(head).all(axis=1)
    rotations = Rotation.from_euler("zxy", np.where(np.isfinite(head), head, 0)[:, [2, 0, 1]], degrees=True)
    forward = rotations.apply(np.tile([0., 0., 1.], (len(eye), 1)))
    head_angle = interval_angles(forward, head_valid, dt)

    gaze, gaze_valid = unit_vectors(eye[["GazeDirX", "GazeDirY", "GazeDirZ"]].to_numpy(float))
    local = rotations.inv().apply(np.nan_to_num(gaze))
    local_valid = gaze_valid & head_valid
    local = smooth_vectors(local, local_valid, times, CONFIG["gaze_smooth_samples"])
    eye_angle = interval_angles(local, np.isfinite(local).all(axis=1), dt)
    eye_speed = np.where(motion_valid, eye_angle/dt, np.nan)

    hit = eye[["Hit1_X", "Hit1_Y", "Hit1_Z"]].to_numpy(float)
    depth_valid = gaze_valid & position_valid & np.isfinite(hit).all(axis=1) & (hit != 0).any(axis=1)
    eye_position = positions.copy()
    eye_position[:, 1] += CONFIG["eye_height_m"]
    depth = np.linalg.norm(hit-eye_position, axis=1)
    depth[~depth_valid] = np.nan
    local_hit = np.full_like(hit, np.nan)
    local_hit_valid = (
        np.isfinite(hit).all(axis=1) & position_valid & head_valid
    )
    local_hit[local_hit_valid] = rotations[local_hit_valid].inv().apply(
        hit[local_hit_valid] - positions[local_hit_valid]
    )

    # Use the midpoint of the observed part of the interval, including edge clipping.
    alpha = np.divide((overlap_start+overlap_end)*.5-previous_time, dt,
                      out=np.zeros_like(dt), where=np.isfinite(dt) & (dt > 0))
    previous_pos = np.vstack([positions[0], positions[:-1]])
    midpoint = previous_pos + alpha[:, None]*(positions-previous_pos)
    idx = np.flatnonzero(overlap > 0)
    assigned = classifier.assign(midpoint[idx])
    row_zone = np.full(len(eye), -1, dtype=int)
    row_zone[idx] = assigned
    eligible = idx[assigned >= 0]
    keys = {key: str(getattr(trial, key)) for key in KEYS}
    frame = pd.DataFrame({"ZoneIndex": row_zone[eligible],
                          "GridCellX": np.floor(midpoint[eligible, 0]/CONFIG["cell_m"]+.5).astype(int),
                          "GridCellZ": np.floor(midpoint[eligible, 2]/CONFIG["cell_m"]+.5).astype(int)})
    hp = np.isfinite(head_angle[eligible])
    ep = np.isfinite(eye_angle[eligible])
    dp = depth_valid[eligible] & depth_valid[eligible-1]
    weight = overlap[eligible]
    fraction = weight/dt[eligible]
    frame["HeadAngle_deg"] = np.where(hp, head_angle[eligible]*fraction, 0.)
    frame["HeadTime_s"] = np.where(hp, weight, 0.)
    frame["EyeAngle_deg"] = np.where(ep, eye_angle[eligible]*fraction, 0.)
    frame["EyeTime_s"] = np.where(ep, weight, 0.)
    depth_mid = depth[eligible-1] + alpha[eligible]*(depth[eligible]-depth[eligible-1])
    frame["DepthIntegral_m_s"] = np.where(dp, depth_mid*weight, 0.)
    frame["DepthTime_s"] = np.where(dp, weight, 0.)
    frame["PositionTime_s"] = weight
    for axis, label in ((0, "X"), (1, "Y"), (2, "Z")):
        frame[f"Position{label}Integral"] = midpoint[eligible, axis]*weight
    cells = frame.groupby(["ZoneIndex", "GridCellX", "GridCellZ"], as_index=False).sum()

    # Detect once on the complete observed trial. Only after an event exists do
    # we assign it to the semantic 2 m cell containing its onset position.
    observed = (times >= start) & (times <= end)
    observed_indices = np.flatnonzero(observed)
    detected, metadata = detect_reference_saccades(local_hit[observed_indices])
    valid_original_indices = observed_indices[metadata["valid_indices"]]

    # Exposure remains spatially local: count valid Hit1 samples in the cell
    # occupied at that timestamp, then convert samples to seconds at 45 Hz.
    valid_zones = classifier.assign(positions[valid_original_indices])
    spatial_valid = valid_zones >= 0
    exposure_indices = valid_original_indices[spatial_valid]
    exposure_zones = valid_zones[spatial_valid].astype(int)
    exposure_rows = pd.DataFrame({
        "ZoneIndex": exposure_zones,
        "GridCellX": np.floor(
            positions[exposure_indices, 0] / CONFIG["cell_m"] + .5
        ).astype(int),
        "GridCellZ": np.floor(
            positions[exposure_indices, 2] / CONFIG["cell_m"] + .5
        ).astype(int),
    })
    if len(exposure_rows):
        counts = (
            exposure_rows.groupby(
                ["ZoneIndex", "GridCellX", "GridCellZ"], as_index=False
            )
            .size()
            .rename(columns={"size": "SaccadeValidHitSamples"})
        )
        counts["SaccadeExposure_s"] = (
            counts.SaccadeValidHitSamples
            / float(CONFIG["saccade_sample_rate_hz"])
        )
    else:
        counts = pd.DataFrame(columns=[
            "ZoneIndex", "GridCellX", "GridCellZ",
            "SaccadeValidHitSamples", "SaccadeExposure_s",
        ])

    event_rows = []
    for event in detected:
        start_original = valid_original_indices[event["start_idx"]]
        end_original = valid_original_indices[event["end_idx"]]
        point = positions[start_original]
        zone = int(classifier.assign(point[None, :])[0])
        if zone < 0:
            continue
        event_rows.append({
            **keys,
            "StartTime": eye.TimeStamp.iloc[start_original],
            "EndTime": eye.TimeStamp.iloc[end_original],
            "Duration_s": event["duration_s"],
            "PeakVelocity_deg_s": event["peak_velocity_deg_s"],
            "Amplitude_deg": event["amplitude_deg"],
            "BaselineNoiseMAD_deg_s": metadata["baseline_noise_mad_deg_s"],
            "VelocityThreshold_deg_s": metadata["velocity_threshold_deg_s"],
            "ZoneIndex": zone,
            "GridCellX": int(np.floor(point[0] / CONFIG["cell_m"] + .5)),
            "GridCellZ": int(np.floor(point[2] / CONFIG["cell_m"] + .5)),
        })
    events = pd.DataFrame(event_rows, columns=[
        *KEYS, "StartTime", "EndTime", "Duration_s", "PeakVelocity_deg_s",
        "Amplitude_deg", "BaselineNoiseMAD_deg_s", "VelocityThreshold_deg_s",
        "ZoneIndex", "GridCellX", "GridCellZ",
    ])
    if len(events):
        event_counts = (
            events.groupby(["ZoneIndex", "GridCellX", "GridCellZ"])
            .size()
            .rename("SaccadeCount")
            .reset_index()
        )
        counts = counts.merge(
            event_counts,
            on=["ZoneIndex", "GridCellX", "GridCellZ"],
            how="outer",
        )
    else:
        counts["SaccadeCount"] = 0
    if len(counts):
        cells = cells.merge(
            counts,
            on=["ZoneIndex", "GridCellX", "GridCellZ"],
            how="outer",
        )
    else:
        cells["SaccadeCount"] = 0
        cells["SaccadeValidHitSamples"] = 0
        cells["SaccadeExposure_s"] = 0.0
    cells["SaccadeCount"] = cells["SaccadeCount"].fillna(0).astype(int)
    cells["SaccadeValidHitSamples"] = cells["SaccadeValidHitSamples"].fillna(0).astype(int)
    cells["SaccadeExposure_s"] = cells["SaccadeExposure_s"].fillna(0.0)
    cells = cells.fillna(0)
    for key, value in keys.items():
        cells[key] = value
    cells["Layer"] = [layer_of(classifier.zone_list[int(i)].zone_type) for i in cells.ZoneIndex]
    cells["PhysicalZone"] = [classifier.zone_list[int(i)].zone_id for i in cells.ZoneIndex]
    cells["Guidance"], cells["Smoke"] = trial.Guidance, trial.Smoke
    # Independent conservation checks against interval and event contributions.
    for col in ["HeadAngle_deg", "HeadTime_s", "EyeAngle_deg", "EyeTime_s", "DepthIntegral_m_s", "DepthTime_s"]:
        np.testing.assert_allclose(cells[col].sum(), frame[col].sum(), rtol=1e-10, atol=1e-8)
    saccade_count = len(events)
    assert cells.SaccadeCount.sum() == saccade_count
    audit = {**keys, "TrialDuration_s": end-start, "EyeSamplesInTrial": int(observed.sum()),
             "ExactPositionSamples": int(((sync_method == 1) & observed).sum()),
             "InterpolatedPositionSamples": int(((sync_method == 2) & observed).sum()),
             "UnmatchedPositionSamples": int(((sync_method == 0) & observed).sum()),
             "PositionExposure_s": float(overlap.sum()),
             "ClassifiedExposure_s": float(weight.sum()),
             "UnclassifiedExposure_s": float(overlap.sum()-weight.sum()),
             "ValidHeadExposure_s": float(frame.HeadTime_s.sum()),
             "ValidEyeExposure_s": float(frame.EyeTime_s.sum()),
             "ValidDepthExposure_s": float(frame.DepthTime_s.sum()),
             "SaccadeDetectedWholeTrial": len(detected),
             "SaccadeUnassignedOutsideZones": len(detected)-saccade_count,
             "SaccadeCount": saccade_count,
             "SaccadeValidHitSamples": int(cells.SaccadeValidHitSamples.sum()),
             "SaccadeExposure_s": float(cells.SaccadeExposure_s.sum()),
             "MedianSampleInterval_ms": float(np.median(dt[observed & (dt > 0)])*1000),
             **{f"Position_{k}": v for k, v in pq.items()},
             **{f"Eye_{k}": v for k, v in eq.items()}}
    for layer in ["hall", "platform", "stair"]:
        audit[f"{layer}_EyeTime_s"] = float(cells.loc[cells.Layer.eq(layer), "EyeTime_s"].sum())
    inputs = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
               "mtime_ns": path.stat().st_mtime_ns} for path in [subject_path, eye_path]]
    return cells, events, audit, inputs


def metric_values(totals, minimum):
    parts = []
    identifiers = [c for c in ["Scenario", "Guidance", "Smoke", "ParticipantID", "Layer",
                              "GridCellX", "GridCellZ", "PosX", "PosY", "PosZ"] if c in totals]
    for metric, (num, den, multiplier, unit) in METRICS.items():
        result = totals[identifiers].copy()
        result["Metric"] = metric
        result["Numerator"] = totals[num]*multiplier
        result["Exposure_s"] = totals[den]
        result["Value"] = result.Numerator/result.Exposure_s.replace(0, np.nan)
        result["Supported"] = result.Exposure_s >= minimum
        result["Unit"] = unit
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Development smoke test; outputs go to smoke_test/")
    args = parser.parse_args()
    output = OUT / "smoke_test" if args.limit else OUT
    output.mkdir(exist_ok=True)
    trials = pd.read_csv(ROOT / "Results_Guidance_Spatial_Causal_Analysis/data/analysis_trials.csv")
    allowed = pd.read_csv(ROOT / "Results_Guidance_Spatial_Causal_Analysis/data/visual_information_trial_features.csv", usecols=KEYS)
    trials = trials.loc[trials.AnalysisIncluded.astype(str).str.lower().eq("true")]
    trials = trials.merge(allowed, on=KEYS, validate="one_to_one").sort_values(KEYS)
    if args.limit:
        trials = trials.head(args.limit)
    classifier = VectorizedZoneClassifier(SemanticZoneQuery())
    cell_parts, event_parts, audit_rows, inputs = [], [], [], []
    for number, trial in enumerate(trials.itertuples(index=False), 1):
        cells, events, audit, source_files = process_trial(trial, classifier)
        cell_parts.append(cells)
        event_parts.append(events)
        audit_rows.append(audit)
        inputs.extend(source_files)
        if number % 20 == 0 or number == len(trials):
            print(f"Extracted {number}/{len(trials)} trials", flush=True)
    zone_cells = pd.concat(cell_parts, ignore_index=True)
    zone_cells.to_csv(output / "trial_zone_cell_totals.csv.gz", index=False)
    events = pd.concat(event_parts, ignore_index=True)
    events.to_csv(output / "saccade_events.csv.gz", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "synchronization_audit.csv", index=False)
    totals_columns = ["HeadAngle_deg", "HeadTime_s", "EyeAngle_deg", "EyeTime_s",
                      "DepthIntegral_m_s", "DepthTime_s", "PositionTime_s", "PositionXIntegral",
                      "PositionYIntegral", "PositionZIntegral", "SaccadeCount",
                      "SaccadeValidHitSamples", "SaccadeExposure_s"]
    group_keys = ["Scenario", "Guidance", "Smoke", "ParticipantID", "Layer"]
    totals = zone_cells.groupby(group_keys+["GridCellX", "GridCellZ"], as_index=False)[totals_columns].sum()
    for axis in ["X", "Y", "Z"]:
        totals[f"Pos{axis}"] = totals[f"Position{axis}Integral"]/totals.PositionTime_s.replace(0, np.nan)
    totals.to_csv(output / "participant_cell_totals.csv.gz", index=False)
    values = metric_values(totals, CONFIG["min_cell_exposure_s"])
    values.to_csv(output / "participant_cell_metrics.csv.gz", index=False)
    floors = totals.groupby(group_keys, as_index=False)[totals_columns].sum()
    floors.to_csv(output / "participant_floor_totals.csv", index=False)
    floor_values = metric_values(floors, CONFIG["min_floor_exposure_s"])
    floor_values = floor_values.loc[floor_values.Supported & floor_values.Layer.isin(["hall", "platform"])]
    floor_values.to_csv(output / "participant_layer_means.csv", index=False)
    assert not values.duplicated(group_keys+["Metric", "GridCellX", "GridCellZ"]).any()
    assert np.isfinite(values.loc[values.Supported, "Value"]).all()
    manifest = dict(config=CONFIG,
                    metric_units={key: spec[3] for key, spec in METRICS.items()},
                    trial_count=len(trials), participants=trials.ParticipantID.nunique(),
                    samples_in_trial=int(audit.EyeSamplesInTrial.sum()),
                    exact_matches=int(audit.ExactPositionSamples.sum()),
                    interpolated=int(audit.InterpolatedPositionSamples.sum()),
                    unmatched=int(audit.UnmatchedPositionSamples.sum()),
                    cohort_counts=trials.groupby("Scenario").size().to_dict(),
                    extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    geometry_sha256=hashlib.sha256(classifier.query.geometry_path.read_bytes()).hexdigest()
                    if hasattr(classifier.query, "geometry_path") else None,
                    sources=inputs)
    (output / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"sources", "cohort_counts"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
