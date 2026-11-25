#!/usr/bin/env python3
"""
Reconstruct dynamic LightBlanket AOIs at every recorded eye-gaze timestamp.

Inputs per trial:
  - *_Log_Info_*.csv: timestamped VectorPath Points snapshots (about every 0.5 s)
  - *_EyeGaze_Simulation_Info_*.csv: gaze hit coordinates
  - *_Subject_Simulation_Info_*.csv: subject root position for range estimation

Primary output:
  One row per EyeGaze row with strict and accuracy-expanded LightBlanket AOI flags.

Important:
  - The Unity log coordinates already include the controller's +0.1 m Y offset.
  - Guidance.prefab renders the LightBlanket with a width of 0.5 m.
  - Between logged path snapshots, polylines are resampled by arc length and linearly
    interpolated. Discontinuous destination changes are not interpolated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Extract_DynamicLightBlanket_Path import (
    TIMESTAMP_FORMAT,
    infer_metadata,
    parse_log,
)


DEFAULT_CONDITIONS = (
    "ArrowRibbon-HighSmoke",
    "ArrowRibbon-LowSmoke",
    "Full-HighSmoke",
    "Full-LowSmoke",
)

LOG_FILE_RE = re.compile(
    r"^(?P<trial>.+?)_Log_Info_(?P<stamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.csv$"
)


@dataclass(frozen=True)
class Config:
    line_width_m: float = 0.5
    eye_accuracy_deg: float = 1.1
    max_accuracy_margin_m: float = 0.25
    eye_height_m: float = 1.65
    vertical_tolerance_m: float = 0.35
    resample_points: int = 160
    max_snapshot_interval_s: float = 0.75
    max_nearest_snapshot_age_s: float = 0.60
    max_endpoint_jump_m: float = 2.0


def read_csv_safely(path: Path) -> pd.DataFrame:
    # EyeGaze files have a trailing extra comma in data rows. index_col=False
    # prevents pandas from shifting all named columns by one position.
    return pd.read_csv(path, index_col=False, low_memory=False)


def parse_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=TIMESTAMP_FORMAT, errors="coerce")


def find_companion_files(log_path: Path) -> tuple[Path, Path, str, str]:
    match = LOG_FILE_RE.match(log_path.name)
    if match is None:
        raise ValueError(f"Unexpected log filename: {log_path.name}")

    trial = match.group("trial")
    stamp = match.group("stamp")
    eye_path = log_path.with_name(
        f"{trial}_EyeGaze_Simulation_Info_{stamp}.csv"
    )
    subject_path = log_path.with_name(
        f"{trial}_Subject_Simulation_Info_{stamp}.csv"
    )

    if not eye_path.is_file():
        raise FileNotFoundError(f"EyeGaze companion not found: {eye_path}")
    if not subject_path.is_file():
        raise FileNotFoundError(f"Subject companion not found: {subject_path}")

    return eye_path, subject_path, trial, stamp


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 0:
        return np.full((count, 3), np.nan, dtype=float)
    if len(points) == 1:
        return np.repeat(points.astype(float), count, axis=0)

    points = points.astype(float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])

    if total <= 1e-9:
        return np.repeat(points[:1], count, axis=0)

    targets = np.linspace(0.0, total, count)
    segment_indices = np.searchsorted(cumulative, targets, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(points) - 2)

    starts = points[segment_indices]
    ends = points[segment_indices + 1]
    local_lengths = segment_lengths[segment_indices]
    local_alpha = np.divide(
        targets - cumulative[segment_indices],
        local_lengths,
        out=np.zeros_like(targets),
        where=local_lengths > 1e-9,
    )
    return starts + local_alpha[:, None] * (ends - starts)


def prepare_paths(log_path: Path, config: Config) -> dict[str, Any]:
    snapshots = parse_log(log_path)
    if not snapshots:
        raise RuntimeError(f"No VectorPath snapshots in {log_path}")

    times = pd.to_datetime(
        [snapshot.timestamp_text for snapshot in snapshots],
        format=TIMESTAMP_FORMAT,
    )
    time_ns = times.astype("int64").to_numpy()

    raw_points = [
        np.asarray(
            [(point.x, point.y, point.z) for point in snapshot.points],
            dtype=float,
        )
        for snapshot in snapshots
    ]
    resampled = np.stack(
        [
            resample_polyline(points, config.resample_points)
            for points in raw_points
        ],
        axis=0,
    )
    endpoints = np.stack([points[-1] for points in raw_points], axis=0)
    starts = np.stack([points[0] for points in raw_points], axis=0)
    lengths = np.asarray([polyline_length(points) for points in raw_points])
    counts = np.asarray([len(points) for points in raw_points], dtype=int)

    return {
        "snapshots": snapshots,
        "times": times,
        "time_ns": time_ns,
        "raw_points": raw_points,
        "resampled": resampled,
        "endpoints": endpoints,
        "starts": starts,
        "lengths": lengths,
        "counts": counts,
    }


def path_at_time(
    timestamp_ns: int,
    paths: dict[str, Any],
    config: Config,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    path_times = paths["time_ns"]
    insertion = int(np.searchsorted(path_times, timestamp_ns, side="left"))

    if insertion < len(path_times) and path_times[insertion] == timestamp_ns:
        return paths["resampled"][insertion], {
            "mode": "exact_snapshot",
            "before_index": insertion,
            "after_index": insertion,
            "alpha": 0.0,
            "nearest_age_ms": 0.0,
        }

    before = insertion - 1
    after = insertion

    if before < 0:
        # Do not project a future path backwards into the pre-guidance interval.
        return None, {
            "mode": "before_first_snapshot",
            "before_index": -1,
            "after_index": after,
            "alpha": np.nan,
            "nearest_age_ms": (
                (path_times[after] - timestamp_ns) / 1e6
                if after < len(path_times)
                else np.nan
            ),
        }

    if after >= len(path_times):
        age_s = (timestamp_ns - path_times[before]) / 1e9
        if age_s <= config.max_nearest_snapshot_age_s:
            return paths["resampled"][before], {
                "mode": "hold_last_snapshot",
                "before_index": before,
                "after_index": -1,
                "alpha": 0.0,
                "nearest_age_ms": age_s * 1000.0,
            }
        return None, {
            "mode": "after_last_snapshot",
            "before_index": before,
            "after_index": -1,
            "alpha": np.nan,
            "nearest_age_ms": age_s * 1000.0,
        }

    interval_s = (path_times[after] - path_times[before]) / 1e9
    age_before_s = (timestamp_ns - path_times[before]) / 1e9
    age_after_s = (path_times[after] - timestamp_ns) / 1e9
    nearest_age_s = min(age_before_s, age_after_s)
    endpoint_jump = float(
        np.linalg.norm(paths["endpoints"][after] - paths["endpoints"][before])
    )

    can_interpolate = (
        interval_s <= config.max_snapshot_interval_s
        and endpoint_jump <= config.max_endpoint_jump_m
    )

    if can_interpolate:
        alpha = age_before_s / interval_s if interval_s > 0 else 0.0
        path = (
            (1.0 - alpha) * paths["resampled"][before]
            + alpha * paths["resampled"][after]
        )
        return path, {
            "mode": "interpolated",
            "before_index": before,
            "after_index": after,
            "alpha": alpha,
            "nearest_age_ms": nearest_age_s * 1000.0,
        }

    if nearest_age_s <= config.max_nearest_snapshot_age_s:
        nearest = before if age_before_s <= age_after_s else after
        return paths["resampled"][nearest], {
            "mode": "nearest_discontinuity",
            "before_index": before,
            "after_index": after,
            "alpha": 0.0 if nearest == before else 1.0,
            "nearest_age_ms": nearest_age_s * 1000.0,
        }

    return None, {
        "mode": "snapshot_gap",
        "before_index": before,
        "after_index": after,
        "alpha": np.nan,
        "nearest_age_ms": nearest_age_s * 1000.0,
    }


def point_to_polyline_xz(
    point: np.ndarray,
    path: np.ndarray,
) -> tuple[float, np.ndarray, int, float]:
    starts = path[:-1]
    ends = path[1:]
    vectors_xz = ends[:, (0, 2)] - starts[:, (0, 2)]
    point_offsets_xz = point[[0, 2]] - starts[:, (0, 2)]
    length_sq = np.einsum("ij,ij->i", vectors_xz, vectors_xz)
    alpha = np.divide(
        np.einsum("ij,ij->i", point_offsets_xz, vectors_xz),
        length_sq,
        out=np.zeros_like(length_sq),
        where=length_sq > 1e-12,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    nearest_points = starts + alpha[:, None] * (ends - starts)
    distances_xz = np.linalg.norm(
        nearest_points[:, (0, 2)] - point[[0, 2]],
        axis=1,
    )
    segment_index = int(np.argmin(distances_xz))
    nearest = nearest_points[segment_index]
    distance_xz = float(distances_xz[segment_index])
    distance_3d = float(np.linalg.norm(nearest - point))
    return distance_xz, nearest, segment_index, distance_3d


def subject_positions_by_timestamp(subject_path: Path) -> dict[int, np.ndarray]:
    subject = read_csv_safely(subject_path)
    required = ("TimeStamp", "PosX", "PosY", "PosZ")
    missing = [column for column in required if column not in subject.columns]
    if missing:
        raise ValueError(f"Missing Subject columns {missing}: {subject_path}")

    times = parse_timestamps(subject["TimeStamp"])
    positions = subject.loc[:, ["PosX", "PosY", "PosZ"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    result: dict[int, np.ndarray] = {}
    for timestamp, position in zip(times, positions.to_numpy(dtype=float)):
        if pd.isna(timestamp) or not np.isfinite(position).all():
            continue
        result[int(timestamp.value)] = position
    return result


def classify_trial(
    log_path: Path,
    data_root: Path,
    output_root: Path,
    config: Config,
) -> dict[str, Any]:
    eye_path, subject_path, trial_id, stamp = find_companion_files(log_path)
    metadata = infer_metadata(log_path, data_root)
    paths = prepare_paths(log_path, config)
    subject_positions = subject_positions_by_timestamp(subject_path)

    eye = read_csv_safely(eye_path)
    required_eye = (
        "TimeStamp",
        "Hit1_X",
        "Hit1_Y",
        "Hit1_Z",
        "Hit1_Name",
    )
    missing_eye = [column for column in required_eye if column not in eye.columns]
    if missing_eye:
        raise ValueError(f"Missing EyeGaze columns {missing_eye}: {eye_path}")

    eye_times = parse_timestamps(eye["TimeStamp"])
    hit_values = eye.loc[:, ["Hit1_X", "Hit1_Y", "Hit1_Z"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    hit_names = eye["Hit1_Name"].fillna("").astype(str).str.strip().to_numpy()

    records: list[dict[str, Any]] = []
    accuracy_angle_rad = math.radians(config.eye_accuracy_deg)
    half_width = config.line_width_m / 2.0

    for row_index, (timestamp, hit, hit_name) in enumerate(
        zip(eye_times, hit_values, hit_names)
    ):
        base: dict[str, Any] = {
            "OriginalRowIndex": row_index,
            "TimeStamp": (
                eye.iloc[row_index]["TimeStamp"]
                if row_index < len(eye)
                else ""
            ),
            "Condition": metadata["condition"],
            "GuidanceType": metadata["guidance_type"],
            "SmokeLevel": metadata["smoke_level"],
            "ParticipantFolder": metadata["participant_folder"],
            "TrialID": trial_id,
            "Hit1_Name": hit_name,
            "Hit1_X": hit[0],
            "Hit1_Y": hit[1],
            "Hit1_Z": hit[2],
            "GuideVisibleByCondition": 1,
            "ValidEyeHit": 0,
            "PathAvailable": 0,
            "PathMode": "",
            "PathBeforeIndex": -1,
            "PathAfterIndex": -1,
            "PathInterpolationAlpha": np.nan,
            "NearestPathSnapshotAgeMs": np.nan,
            "DistanceToCarpetCenterXZ_M": np.nan,
            "DistanceToCarpetCenter3D_M": np.nan,
            "VerticalDifferenceM": np.nan,
            "NearestCarpetX": np.nan,
            "NearestCarpetY": np.nan,
            "NearestCarpetZ": np.nan,
            "NearestCarpetSegment": -1,
            "ApproxEyeToHitDistanceM": np.nan,
            "AccuracyMarginM": np.nan,
            "CoreThresholdM": half_width,
            "ExpandedThresholdM": np.nan,
            "GazeOnLightBlanketCore": 0,
            "GazeOnLightBlanket": 0,
            "AOIConfidence": "unavailable",
            "AOIStatus": "",
        }

        valid_timestamp = not pd.isna(timestamp)
        valid_hit = (
            valid_timestamp
            and np.isfinite(hit).all()
            and hit_name not in ("", "Null", "nan", "None")
            and not np.allclose(hit, 0.0)
        )
        if not valid_hit:
            base["AOIStatus"] = "invalid_eye_hit"
            records.append(base)
            continue

        base["ValidEyeHit"] = 1
        timestamp_ns = int(timestamp.value)
        path, path_info = path_at_time(timestamp_ns, paths, config)
        base.update(
            {
                "PathMode": path_info["mode"],
                "PathBeforeIndex": path_info["before_index"],
                "PathAfterIndex": path_info["after_index"],
                "PathInterpolationAlpha": path_info["alpha"],
                "NearestPathSnapshotAgeMs": path_info["nearest_age_ms"],
            }
        )

        if path is None or len(path) < 2 or not np.isfinite(path).all():
            base["AOIStatus"] = "path_unavailable"
            records.append(base)
            continue

        base["PathAvailable"] = 1
        distance_xz, nearest, segment_index, distance_3d = (
            point_to_polyline_xz(hit, path)
        )
        vertical_difference = abs(float(hit[1] - nearest[1]))

        subject_position = subject_positions.get(timestamp_ns)
        if subject_position is not None:
            approximate_eye = subject_position.copy()
            approximate_eye[1] += config.eye_height_m
            eye_to_hit = float(np.linalg.norm(hit - approximate_eye))
        else:
            # Horizontal range remains a useful conservative fallback.
            eye_to_hit = float(
                np.linalg.norm(hit[[0, 2]] - path[0, [0, 2]])
            )

        accuracy_margin = min(
            eye_to_hit * math.tan(accuracy_angle_rad),
            config.max_accuracy_margin_m,
        )
        expanded_threshold = half_width + accuracy_margin
        vertical_ok = vertical_difference <= config.vertical_tolerance_m
        core = bool(distance_xz <= half_width and vertical_ok)
        expanded = bool(distance_xz <= expanded_threshold and vertical_ok)

        if core:
            confidence = "core"
            status = "on_light_blanket_core"
        elif expanded:
            confidence = "expanded_margin"
            status = "on_light_blanket_margin"
        else:
            confidence = "outside"
            status = (
                "outside_vertical_tolerance"
                if not vertical_ok
                else "outside_light_blanket"
            )

        base.update(
            {
                "DistanceToCarpetCenterXZ_M": distance_xz,
                "DistanceToCarpetCenter3D_M": distance_3d,
                "VerticalDifferenceM": vertical_difference,
                "NearestCarpetX": nearest[0],
                "NearestCarpetY": nearest[1],
                "NearestCarpetZ": nearest[2],
                "NearestCarpetSegment": segment_index,
                "ApproxEyeToHitDistanceM": eye_to_hit,
                "AccuracyMarginM": accuracy_margin,
                "ExpandedThresholdM": expanded_threshold,
                "GazeOnLightBlanketCore": int(core),
                "GazeOnLightBlanket": int(expanded),
                "AOIConfidence": confidence,
                "AOIStatus": status,
            }
        )
        records.append(base)

    result = pd.DataFrame.from_records(records)

    condition_dir = output_root / metadata["condition"]
    participant_dir = condition_dir / metadata["participant_folder"]
    participant_dir.mkdir(parents=True, exist_ok=True)
    output_path = participant_dir / (
        f"{trial_id}_LightBlanketGazeAOI_{stamp}.csv"
    )
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    valid_eye = result["ValidEyeHit"].eq(1)
    path_available = result["PathAvailable"].eq(1)
    analyzable = valid_eye & path_available
    core = result["GazeOnLightBlanketCore"].eq(1)
    expanded = result["GazeOnLightBlanket"].eq(1)

    parsed_times = eye_times
    deltas = parsed_times.shift(-1) - parsed_times
    delta_s = deltas.dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=0.1)

    summary: dict[str, Any] = {
        "Condition": metadata["condition"],
        "GuidanceType": metadata["guidance_type"],
        "SmokeLevel": metadata["smoke_level"],
        "ParticipantFolder": metadata["participant_folder"],
        "TrialID": trial_id,
        "SourceLogFile": str(log_path.resolve()),
        "SourceEyeFile": str(eye_path.resolve()),
        "SourceSubjectFile": str(subject_path.resolve()),
        "OutputFile": str(output_path.resolve()),
        "EyeRowCount": int(len(result)),
        "ValidEyeHitCount": int(valid_eye.sum()),
        "PathAvailableCount": int(path_available.sum()),
        "AnalyzableCount": int(analyzable.sum()),
        "CoreGazeCount": int((core & analyzable).sum()),
        "ExpandedGazeCount": int((expanded & analyzable).sum()),
        "CoreGazeProportion": (
            float((core & analyzable).sum() / analyzable.sum())
            if analyzable.any()
            else np.nan
        ),
        "ExpandedGazeProportion": (
            float((expanded & analyzable).sum() / analyzable.sum())
            if analyzable.any()
            else np.nan
        ),
        "CoreGazeDurationS_RawSamples": float(delta_s[core].sum()),
        "ExpandedGazeDurationS_RawSamples": float(delta_s[expanded].sum()),
        "PathSnapshotCount": int(len(paths["snapshots"])),
        "MedianPathSnapshotIntervalS": float(
            np.median(np.diff(paths["time_ns"]) / 1e9)
        )
        if len(paths["time_ns"]) > 1
        else np.nan,
        "MedianPathPointCount": float(np.median(paths["counts"])),
        "MedianPathStartSubjectXZErrorM": np.nan,
    }

    # Validate logged path starts against the nearest Subject sample.
    subject = read_csv_safely(subject_path)
    subject["__time"] = parse_timestamps(subject["TimeStamp"])
    start_frame = pd.DataFrame(
        {
            "__time": paths["times"],
            "StartX": paths["starts"][:, 0],
            "StartZ": paths["starts"][:, 2],
        }
    )
    matched = pd.merge_asof(
        start_frame.sort_values("__time"),
        subject.sort_values("__time")[
            ["__time", "PosX", "PosZ"]
        ].assign(
            PosX=lambda frame: pd.to_numeric(frame["PosX"], errors="coerce"),
            PosZ=lambda frame: pd.to_numeric(frame["PosZ"], errors="coerce"),
        ),
        on="__time",
        direction="nearest",
        tolerance=pd.Timedelta("50ms"),
    )
    start_error = np.sqrt(
        (matched["StartX"] - matched["PosX"]) ** 2
        + (matched["StartZ"] - matched["PosZ"]) ** 2
    )
    if start_error.notna().any():
        summary["MedianPathStartSubjectXZErrorM"] = float(
            start_error.median()
        )

    return summary


def discover_logs(data_root: Path, conditions: tuple[str, ...]) -> list[Path]:
    logs: list[Path] = []
    for condition in conditions:
        condition_dir = data_root / condition
        if not condition_dir.is_dir():
            continue
        logs.extend(condition_dir.rglob("*_Log_Info_*.csv"))
    return sorted(path.resolve() for path in logs)


def serializable_config(config: Config, conditions: tuple[str, ...]) -> dict[str, Any]:
    value = asdict(config)
    value["conditions"] = list(conditions)
    value["primary_flag"] = "GazeOnLightBlanket"
    value["strict_flag"] = "GazeOnLightBlanketCore"
    value["temporal_reconstruction"] = (
        "arc-length resampling plus linear interpolation between ~0.5 s "
        "VectorPath log snapshots"
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify LightBlanket gaze AOI at every EyeGaze timestamp."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "RAW_DATA",
        help="Raw data root containing the scenario folders "
             "(default: <project>/RAW_DATA).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "Results_LightBlanket_GazeAOI",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Process only one log file for validation.",
    )
    parser.add_argument("--line-width-m", type=float, default=0.5)
    parser.add_argument("--eye-accuracy-deg", type=float, default=1.1)
    parser.add_argument("--max-accuracy-margin-m", type=float, default=0.25)
    parser.add_argument("--eye-height-m", type=float, default=1.65)
    parser.add_argument("--vertical-tolerance-m", type=float, default=0.35)
    parser.add_argument("--resample-points", type=int, default=160)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_dir.resolve()
    conditions = tuple(args.conditions)
    config = Config(
        line_width_m=args.line_width_m,
        eye_accuracy_deg=args.eye_accuracy_deg,
        max_accuracy_margin_m=args.max_accuracy_margin_m,
        eye_height_m=args.eye_height_m,
        vertical_tolerance_m=args.vertical_tolerance_m,
        resample_points=args.resample_points,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(
        json.dumps(
            serializable_config(config, conditions),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logs = (
        [args.log_file.resolve()]
        if args.log_file is not None
        else discover_logs(data_root, conditions)
    )
    if not logs:
        raise RuntimeError("No matching LightBlanket experiment logs found.")

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, log_path in enumerate(logs, start=1):
        try:
            summary = classify_trial(
                log_path=log_path,
                data_root=data_root,
                output_root=output_root,
                config=config,
            )
            summaries.append(summary)
            print(
                f"[{index}/{len(logs)}] OK "
                f"{summary['Condition']}/"
                f"{summary['ParticipantFolder']}/"
                f"{summary['TrialID']}"
            )
        except Exception as exc:
            errors.append(
                {
                    "LogFile": str(log_path),
                    "ErrorType": type(exc).__name__,
                    "ErrorMessage": str(exc),
                }
            )
            print(
                f"[{index}/{len(logs)}] ERROR {log_path}: {exc}",
                file=sys.stderr,
            )

    summary_frame = pd.DataFrame(summaries)
    summary_path = output_root / "experiment_summary.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")

    error_frame = pd.DataFrame(
        errors,
        columns=("LogFile", "ErrorType", "ErrorMessage"),
    )
    error_path = output_root / "errors.csv"
    error_frame.to_csv(error_path, index=False, encoding="utf-8-sig")

    batch_qc = {
        "requested_log_count": len(logs),
        "successful_trial_count": len(summaries),
        "failed_trial_count": len(errors),
        "conditions": list(conditions),
        "total_eye_rows": (
            int(summary_frame["EyeRowCount"].sum())
            if not summary_frame.empty
            else 0
        ),
        "total_analyzable_rows": (
            int(summary_frame["AnalyzableCount"].sum())
            if not summary_frame.empty
            else 0
        ),
    }
    (output_root / "batch_qc.json").write_text(
        json.dumps(batch_qc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(batch_qc, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Errors: {error_path}")

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
