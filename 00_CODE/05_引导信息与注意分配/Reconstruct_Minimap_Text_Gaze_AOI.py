#!/usr/bin/env python3
"""
Reconstruct MiniMap and Text guidance gaze AOIs from existing VR experiment data.

The final AgentNEW.prefab overrides both UI canvases to World Space:
  - parent: AgentNEW / Guidance root
  - canvas local plane coordinate: z = 0.9 m from the AgentNEW root
  - camera local position: (0, 1.771, 0.2) m
  - camera-to-canvas forward separation: 0.9 - 0.2 = 0.7 m
  - canvas scale: 0.0008 m per UI unit

Therefore this script does not approximate a headset FOV. It reconstructs the
moving UI plane from Subject PosX/PosY/PosZ/RotY, intersects the world-space
gaze line with that plane, and tests the intersection against the final prefab
rectangles.
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

from Extract_DynamicLightBlanket_Path import TIMESTAMP_FORMAT, infer_metadata
from Reconstruct_LightBlanket_Gaze_AOI import (
    find_companion_files,
    read_csv_safely,
)


DEFAULT_CONDITIONS = (
    "MiniMap-HighSmoke",
    "MiniMap-LowSmoke",
    "TextBar-HighSmoke",
    "TextBar-LowSmoke",
    "Full-HighSmoke",
    "Full-LowSmoke",
)

LOG_TIMESTAMP_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{3})\]"
)


@dataclass(frozen=True)
class Rectangle:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class Config:
    # Both final canvases are at local z=0.9 m in the AgentNEW root frame.
    # The camera is at local z=0.2 m, so the forward camera-to-canvas
    # separation is 0.7 m. Do not replace canvas_plane_local_z_m with 0.7:
    # the two values use different origins.
    canvas_plane_local_z_m: float = 0.9
    camera_local_y_m: float = 1.771
    camera_local_z_m: float = 0.2
    canvas_scale_m_per_ui_unit: float = 0.0008

    # Final AgentNEW.prefab overrides:
    # 2DMap Canvas: local (0.164, 1.668, 0.9)
    # StaticMap: anchored (132, -153), size (400, 800)
    minimap_rect: Rectangle = Rectangle(
        x_min=0.1096,
        x_max=0.4296,
        y_min=1.2256,
        y_max=1.8656,
    )

    # Text Canvas: local (0, 1.699, 0.9), scale 0.0008
    # Direction Panel: anchored (0.5, 353), size (1000, 100)
    direction_panel_rect: Rectangle = Rectangle(
        x_min=-0.3996,
        x_max=0.4004,
        y_min=1.9414,
        y_max=2.0214,
    )

    # Distance Panel: anchored (0.5, 254), size (1000, 100)
    distance_panel_rect: Rectangle = Rectangle(
        x_min=-0.3996,
        x_max=0.4004,
        y_min=1.8622,
        y_max=1.9422,
    )

    # EvacUIController shows DistanceText content only above this root Y.
    distance_text_player_y_threshold_m: float = -11.8

    # Accuracy-expanded AOI. At the 0.7 m forward camera-to-UI separation,
    # 1.1 degrees is approximately 1.34 cm for a straight-ahead gaze.
    eye_accuracy_deg: float = 1.1
    max_accuracy_margin_m: float = 0.03
    subject_time_tolerance_ms: float = 40.0

    @property
    def camera_to_canvas_forward_m(self) -> float:
        return self.canvas_plane_local_z_m - self.camera_local_z_m


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=TIMESTAMP_FORMAT, errors="coerce")


def condition_has_minimap(condition: str) -> bool:
    return condition.startswith("MiniMap-") or condition.startswith("Full-")


def condition_has_text(condition: str) -> bool:
    return condition.startswith("TextBar-") or condition.startswith("Full-")


def parse_minimap_visibility_events(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    event_times: list[pd.Timestamp] = []
    event_states: list[int] = []

    for raw_line in log_path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).splitlines():
        if "[MinimapManager] Minimap display activated." in raw_line:
            state = 1
        elif "[MinimapManager] Minimap display deactivated." in raw_line:
            state = 0
        else:
            continue

        match = LOG_TIMESTAMP_RE.match(raw_line)
        if match is None:
            continue
        timestamp = pd.to_datetime(
            match.group("timestamp"),
            format=TIMESTAMP_FORMAT,
            errors="coerce",
        )
        if pd.isna(timestamp):
            continue
        event_times.append(timestamp)
        event_states.append(state)

    if not event_times:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int8)

    order = np.argsort(np.asarray([value.value for value in event_times]))
    times_ns = np.asarray(
        [event_times[index].value for index in order],
        dtype=np.int64,
    )
    states = np.asarray(
        [event_states[index] for index in order],
        dtype=np.int8,
    )
    return times_ns, states


def visibility_state_at_times(
    sample_time_ns: np.ndarray,
    event_time_ns: np.ndarray,
    event_states: np.ndarray,
) -> np.ndarray:
    if len(event_time_ns) == 0:
        return np.zeros(len(sample_time_ns), dtype=bool)

    indices = np.searchsorted(event_time_ns, sample_time_ns, side="right") - 1
    result = np.zeros(len(sample_time_ns), dtype=bool)
    valid = indices >= 0
    result[valid] = event_states[indices[valid]].astype(bool)
    return result


def inside_rectangle(
    local_x: np.ndarray,
    local_y: np.ndarray,
    rectangle: Rectangle,
    margin: np.ndarray | float = 0.0,
) -> np.ndarray:
    return (
        (local_x >= rectangle.x_min - margin)
        & (local_x <= rectangle.x_max + margin)
        & (local_y >= rectangle.y_min - margin)
        & (local_y <= rectangle.y_max + margin)
    )


def merge_eye_subject(
    eye_path: Path,
    subject_path: Path,
    config: Config,
) -> pd.DataFrame:
    eye = read_csv_safely(eye_path).copy()
    subject = read_csv_safely(subject_path).copy()

    eye["__EyeRowIndex"] = np.arange(len(eye), dtype=int)
    eye["__Time"] = parse_time(eye["TimeStamp"])
    subject["__Time"] = parse_time(subject["TimeStamp"])

    required_eye = (
        "GazeDirX",
        "GazeDirY",
        "GazeDirZ",
        "Hit1_X",
        "Hit1_Y",
        "Hit1_Z",
        "Hit1_Name",
    )
    required_subject = ("PosX", "PosY", "PosZ", "RotY")
    missing_eye = [column for column in required_eye if column not in eye.columns]
    missing_subject = [
        column for column in required_subject if column not in subject.columns
    ]
    if missing_eye:
        raise ValueError(f"Missing EyeGaze columns {missing_eye}: {eye_path}")
    if missing_subject:
        raise ValueError(
            f"Missing Subject columns {missing_subject}: {subject_path}"
        )

    subject_columns = ["__Time", *required_subject]
    subject_subset = subject.loc[:, subject_columns].copy()
    for column in required_subject:
        subject_subset[column] = pd.to_numeric(
            subject_subset[column],
            errors="coerce",
        )

    valid_eye_time = eye["__Time"].notna()
    invalid_time_rows = eye.loc[~valid_eye_time].copy()
    valid_eye_rows = eye.loc[valid_eye_time].sort_values("__Time")
    valid_subject_rows = subject_subset.loc[
        subject_subset["__Time"].notna()
    ].sort_values("__Time")

    merged_valid = pd.merge_asof(
        valid_eye_rows,
        valid_subject_rows,
        on="__Time",
        direction="nearest",
        tolerance=pd.Timedelta(
            milliseconds=config.subject_time_tolerance_ms
        ),
        suffixes=("", "_Subject"),
    )

    if not invalid_time_rows.empty:
        for column in required_subject:
            invalid_time_rows[column] = np.nan
        merged = pd.concat(
            [merged_valid, invalid_time_rows],
            ignore_index=True,
            sort=False,
        )
    else:
        merged = merged_valid

    return merged.sort_values("__EyeRowIndex").reset_index(drop=True)


def reconstruct_plane_intersections(
    frame: pd.DataFrame,
    config: Config,
) -> dict[str, np.ndarray]:
    numeric_columns = (
        "GazeDirX",
        "GazeDirY",
        "GazeDirZ",
        "Hit1_X",
        "Hit1_Y",
        "Hit1_Z",
        "PosX",
        "PosY",
        "PosZ",
        "RotY",
    )
    values: dict[str, np.ndarray] = {}
    for column in numeric_columns:
        values[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)

    root = np.column_stack(
        (values["PosX"], values["PosY"], values["PosZ"])
    )
    yaw = np.deg2rad(values["RotY"])
    right = np.column_stack(
        (np.cos(yaw), np.zeros(len(frame)), -np.sin(yaw))
    )
    forward = np.column_stack(
        (np.sin(yaw), np.zeros(len(frame)), np.cos(yaw))
    )

    gaze = np.column_stack(
        (
            values["GazeDirX"],
            values["GazeDirY"],
            values["GazeDirZ"],
        )
    )
    gaze_norm = np.linalg.norm(gaze, axis=1)
    valid_gaze = np.isfinite(gaze).all(axis=1) & (gaze_norm > 0.5)
    gaze_unit = np.divide(
        gaze,
        gaze_norm[:, None],
        out=np.full_like(gaze, np.nan),
        where=gaze_norm[:, None] > 1e-9,
    )

    hit = np.column_stack(
        (values["Hit1_X"], values["Hit1_Y"], values["Hit1_Z"])
    )
    hit_names = frame["Hit1_Name"].fillna("").astype(str).str.strip()
    valid_hit = (
        np.isfinite(hit).all(axis=1)
        & ~np.all(np.isclose(hit, 0.0), axis=1)
        & ~hit_names.isin(("", "Null", "nan", "None")).to_numpy()
    )

    valid_subject = np.isfinite(root).all(axis=1) & np.isfinite(yaw)
    # Camera is a child of AgentNEW. RotY rotates its local +Z offset with the
    # participant, while the local +Y offset remains world-up in this rig.
    approximate_eye = root + forward * config.camera_local_z_m
    approximate_eye[:, 1] += config.camera_local_y_m
    anchor = np.where(valid_hit[:, None], hit, approximate_eye)
    anchor_source = np.where(valid_hit, "Hit1Line", "ApproxEyeOrigin")

    plane_point = (
        root + forward * config.canvas_plane_local_z_m
    )
    denominator = np.einsum("ij,ij->i", gaze_unit, forward)
    numerator = np.einsum("ij,ij->i", plane_point - anchor, forward)
    line_parameter = np.divide(
        numerator,
        denominator,
        out=np.full(len(frame), np.nan),
        where=np.abs(denominator) > 1e-7,
    )
    intersection = anchor + line_parameter[:, None] * gaze_unit

    relative_to_root = intersection - root
    local_x = np.einsum("ij,ij->i", relative_to_root, right)
    local_y = relative_to_root[:, 1]
    local_z = np.einsum("ij,ij->i", relative_to_root, forward)

    camera_to_intersection = intersection - approximate_eye
    forward_projection = np.einsum(
        "ij,ij->i",
        camera_to_intersection,
        gaze_unit,
    )
    camera_to_intersection_distance = np.linalg.norm(
        camera_to_intersection,
        axis=1,
    )
    valid_intersection = (
        valid_gaze
        & valid_subject
        & np.isfinite(intersection).all(axis=1)
        & (forward_projection > 0.0)
        & (np.abs(local_z - config.canvas_plane_local_z_m) < 1e-4)
    )

    angular_margin = np.minimum(
        np.maximum(camera_to_intersection_distance, 0.0)
        * math.tan(math.radians(config.eye_accuracy_deg)),
        config.max_accuracy_margin_m,
    )
    angular_margin[~valid_intersection] = np.nan

    return {
        "root": root,
        "valid_subject": valid_subject,
        "valid_gaze": valid_gaze,
        "valid_hit": valid_hit,
        "gaze_norm": gaze_norm,
        "anchor_source": anchor_source,
        "intersection": intersection,
        "local_x": local_x,
        "local_y": local_y,
        "local_z": local_z,
        "forward_distance": camera_to_intersection_distance,
        "valid_intersection": valid_intersection,
        "accuracy_margin": angular_margin,
    }


def sample_durations_seconds(times: pd.Series) -> np.ndarray:
    delta = times.shift(-1) - times
    return (
        delta.dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0, upper=0.1)
        .to_numpy(dtype=float)
    )


def classify_trial(
    log_path: Path,
    data_root: Path,
    output_root: Path,
    config: Config,
) -> dict[str, Any]:
    eye_path, subject_path, trial_id, stamp = find_companion_files(log_path)
    metadata = infer_metadata(log_path, data_root)
    condition = metadata["condition"]
    has_minimap = condition_has_minimap(condition)
    has_text = condition_has_text(condition)

    frame = merge_eye_subject(eye_path, subject_path, config)
    geometry = reconstruct_plane_intersections(frame, config)
    valid_intersection = geometry["valid_intersection"]
    local_x = geometry["local_x"]
    local_y = geometry["local_y"]
    margin = geometry["accuracy_margin"]

    sample_time_ns = frame["__Time"].astype("int64").to_numpy()
    event_time_ns, event_states = parse_minimap_visibility_events(log_path)
    minimap_visible = (
        visibility_state_at_times(
            sample_time_ns,
            event_time_ns,
            event_states,
        )
        if has_minimap
        else np.zeros(len(frame), dtype=bool)
    )
    text_visible = np.full(len(frame), has_text, dtype=bool)

    subject_y = pd.to_numeric(frame["PosY"], errors="coerce").to_numpy(
        dtype=float
    )
    distance_text_content_visible = (
        text_visible
        & np.isfinite(subject_y)
        & (subject_y >= config.distance_text_player_y_threshold_m)
    )

    minimap_core = (
        valid_intersection
        & minimap_visible
        & inside_rectangle(
            local_x,
            local_y,
            config.minimap_rect,
        )
    )
    minimap_expanded = (
        valid_intersection
        & minimap_visible
        & inside_rectangle(
            local_x,
            local_y,
            config.minimap_rect,
            margin,
        )
    )

    direction_core = (
        valid_intersection
        & text_visible
        & inside_rectangle(
            local_x,
            local_y,
            config.direction_panel_rect,
        )
    )
    direction_expanded = (
        valid_intersection
        & text_visible
        & inside_rectangle(
            local_x,
            local_y,
            config.direction_panel_rect,
            margin,
        )
    )

    distance_panel_core = (
        valid_intersection
        & text_visible
        & inside_rectangle(
            local_x,
            local_y,
            config.distance_panel_rect,
        )
    )
    distance_panel_expanded = (
        valid_intersection
        & text_visible
        & inside_rectangle(
            local_x,
            local_y,
            config.distance_panel_rect,
            margin,
        )
    )
    distance_content_core = distance_panel_core & distance_text_content_visible
    distance_content_expanded = (
        distance_panel_expanded & distance_text_content_visible
    )

    # Primary Text guidance flag: the always-present direction panel plus the
    # distance panel only while its textual content is enabled.
    text_guidance_core = direction_core | distance_content_core
    text_guidance_expanded = (
        direction_expanded | distance_content_expanded
    )
    text_any_panel_core = direction_core | distance_panel_core
    text_any_panel_expanded = (
        direction_expanded | distance_panel_expanded
    )

    ui_label = np.full(len(frame), "None", dtype=object)
    ui_label[minimap_expanded] = "MiniMap"
    ui_label[text_guidance_expanded] = "Text"
    ui_label[minimap_expanded & text_guidance_expanded] = "MiniMap+Text"
    ui_label[~valid_intersection] = "Unavailable"

    output = pd.DataFrame(
        {
            "OriginalRowIndex": frame["__EyeRowIndex"].astype(int),
            "TimeStamp": frame["TimeStamp"],
            "Condition": condition,
            "GuidanceType": metadata["guidance_type"],
            "SmokeLevel": metadata["smoke_level"],
            "ParticipantFolder": metadata["participant_folder"],
            "TrialID": trial_id,
            "SubjectPosX": pd.to_numeric(frame["PosX"], errors="coerce"),
            "SubjectPosY": subject_y,
            "SubjectPosZ": pd.to_numeric(frame["PosZ"], errors="coerce"),
            "SubjectRotY": pd.to_numeric(frame["RotY"], errors="coerce"),
            "GazeDirX": pd.to_numeric(frame["GazeDirX"], errors="coerce"),
            "GazeDirY": pd.to_numeric(frame["GazeDirY"], errors="coerce"),
            "GazeDirZ": pd.to_numeric(frame["GazeDirZ"], errors="coerce"),
            "Hit1_X": pd.to_numeric(frame["Hit1_X"], errors="coerce"),
            "Hit1_Y": pd.to_numeric(frame["Hit1_Y"], errors="coerce"),
            "Hit1_Z": pd.to_numeric(frame["Hit1_Z"], errors="coerce"),
            "Hit1_Name": frame["Hit1_Name"],
            "ValidSubjectPose": geometry["valid_subject"].astype(int),
            "ValidGazeDirection": geometry["valid_gaze"].astype(int),
            "ValidHit1": geometry["valid_hit"].astype(int),
            "ValidCanvasIntersection": valid_intersection.astype(int),
            "RayAnchorSource": geometry["anchor_source"],
            "CanvasIntersectionX": geometry["intersection"][:, 0],
            "CanvasIntersectionY": geometry["intersection"][:, 1],
            "CanvasIntersectionZ": geometry["intersection"][:, 2],
            "CanvasLocalX_M": local_x,
            "CanvasLocalY_M": local_y,
            "CanvasLocalZ_M": geometry["local_z"],
            "EyeToCanvasDistanceApproxM": geometry["forward_distance"],
            "AOIMarginM": margin,
            "MiniMapVisible": minimap_visible.astype(int),
            "TextVisible": text_visible.astype(int),
            "DistanceTextContentVisible": (
                distance_text_content_visible.astype(int)
            ),
            "GazeOnMiniMapCore": minimap_core.astype(int),
            "GazeOnMiniMap": minimap_expanded.astype(int),
            "GazeOnTextDirectionCore": direction_core.astype(int),
            "GazeOnTextDirection": direction_expanded.astype(int),
            "GazeOnTextDistancePanelCore": (
                distance_panel_core.astype(int)
            ),
            "GazeOnTextDistancePanel": (
                distance_panel_expanded.astype(int)
            ),
            "GazeOnTextDistanceContentCore": (
                distance_content_core.astype(int)
            ),
            "GazeOnTextDistanceContent": (
                distance_content_expanded.astype(int)
            ),
            "GazeOnTextAnyPanelCore": text_any_panel_core.astype(int),
            "GazeOnTextAnyPanel": text_any_panel_expanded.astype(int),
            "GazeOnTextGuidanceCore": text_guidance_core.astype(int),
            "GazeOnTextGuidance": text_guidance_expanded.astype(int),
            "UIAOILabel": ui_label,
        }
    ).sort_values("OriginalRowIndex")

    condition_dir = output_root / condition
    participant_dir = condition_dir / metadata["participant_folder"]
    participant_dir.mkdir(parents=True, exist_ok=True)
    output_path = participant_dir / (
        f"{trial_id}_MinimapTextGazeAOI_{stamp}.csv"
    )
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    duration_s = sample_durations_seconds(frame["__Time"])
    analyzable = valid_intersection

    def count(flag: np.ndarray) -> int:
        return int(np.sum(flag & analyzable))

    def proportion(
        flag: np.ndarray,
        eligible: np.ndarray | None = None,
    ) -> float:
        if eligible is None:
            eligible = analyzable
        eligible = eligible & analyzable
        denominator = int(np.sum(eligible))
        numerator = int(np.sum(flag & eligible))
        return float(numerator / denominator) if denominator else np.nan

    def duration(flag: np.ndarray) -> float:
        return float(duration_s[flag & analyzable].sum())

    minimap_eligible = analyzable & minimap_visible
    text_eligible = analyzable & text_visible

    return {
        "Condition": condition,
        "GuidanceType": metadata["guidance_type"],
        "SmokeLevel": metadata["smoke_level"],
        "ParticipantFolder": metadata["participant_folder"],
        "TrialID": trial_id,
        "SourceLogFile": str(log_path.resolve()),
        "SourceEyeFile": str(eye_path.resolve()),
        "SourceSubjectFile": str(subject_path.resolve()),
        "OutputFile": str(output_path.resolve()),
        "EyeRowCount": int(len(output)),
        "ValidCanvasIntersectionCount": int(analyzable.sum()),
        "ValidCanvasIntersectionProportion": float(analyzable.mean()),
        "MiniMapVisibleSampleCount": int(minimap_visible.sum()),
        "MiniMapAnalyzableVisibleCount": int(minimap_eligible.sum()),
        "MiniMapGazeCoreCount": count(minimap_core),
        "MiniMapGazeCount": count(minimap_expanded),
        "MiniMapGazeProportion": proportion(
            minimap_expanded,
            minimap_eligible,
        ),
        "MiniMapGazeDurationS_RawSamples": duration(minimap_expanded),
        "TextVisibleSampleCount": int(text_visible.sum()),
        "TextAnalyzableVisibleCount": int(text_eligible.sum()),
        "DistanceTextContentVisibleSampleCount": int(
            distance_text_content_visible.sum()
        ),
        "TextDirectionGazeCount": count(direction_expanded),
        "TextDistancePanelGazeCount": count(distance_panel_expanded),
        "TextDistanceContentGazeCount": count(distance_content_expanded),
        "TextAnyPanelGazeCoreCount": count(text_any_panel_core),
        "TextAnyPanelGazeCount": count(text_any_panel_expanded),
        "TextAnyPanelGazeProportion": proportion(
            text_any_panel_expanded,
            text_eligible,
        ),
        "TextAnyPanelGazeDurationS_RawSamples": duration(
            text_any_panel_expanded
        ),
        "TextGuidanceGazeCoreCount": count(text_guidance_core),
        "TextGuidanceGazeCount": count(text_guidance_expanded),
        "TextGuidanceGazeProportion": proportion(
            text_guidance_expanded,
            text_eligible,
        ),
        "TextGuidanceGazeDurationS_RawSamples": duration(
            text_guidance_expanded
        ),
        "RayUsingHit1Count": int(np.sum(geometry["valid_hit"])),
        "RayUsingApproxEyeOriginCount": int(
            np.sum(~geometry["valid_hit"] & geometry["valid_gaze"])
        ),
        "MinimapActivationEventCount": int(
            np.sum(event_states == 1)
        ),
        "MinimapDeactivationEventCount": int(
            np.sum(event_states == 0)
        ),
    }


def discover_logs(
    data_root: Path,
    conditions: tuple[str, ...],
) -> list[Path]:
    logs: list[Path] = []
    for condition in conditions:
        folder = data_root / condition
        if folder.is_dir():
            logs.extend(folder.rglob("*_Log_Info_*.csv"))
    return sorted(path.resolve() for path in logs)


def config_to_json(
    config: Config,
    conditions: tuple[str, ...],
) -> dict[str, Any]:
    value = asdict(config)
    value["camera_to_canvas_forward_m"] = (
        config.camera_to_canvas_forward_m
    )
    value["conditions"] = list(conditions)
    value["primary_minimap_flag"] = "GazeOnMiniMap"
    value["primary_text_flag"] = "GazeOnTextGuidance"
    value["text_any_visible_panel_flag"] = "GazeOnTextAnyPanel"
    value["coordinate_method"] = (
        "world gaze line intersected with AgentNEW-yaw-oriented "
        "World Space Canvas plane"
    )
    return value


def build_region_negative_control(
    output_root: Path,
    conditions: tuple[str, ...],
    config: Config,
) -> pd.DataFrame:
    """Measure occupancy of both geometric regions regardless of visibility.

    This is a spatial negative control: for example, the MiniMap rectangle is
    also tested in TextBar trials where no MiniMap was displayed.
    """
    rows: list[dict[str, Any]] = []
    usecols = (
        "ValidCanvasIntersection",
        "CanvasLocalX_M",
        "CanvasLocalY_M",
        "AOIMarginM",
    )

    for condition in conditions:
        condition_dir = output_root / condition
        files = sorted(
            condition_dir.glob("*/*_MinimapTextGazeAOI_*.csv")
        )
        valid_count = 0
        minimap_count = 0
        text_count = 0

        for path in files:
            frame = pd.read_csv(path, usecols=usecols)
            valid = frame["ValidCanvasIntersection"].eq(1).to_numpy()
            local_x = frame["CanvasLocalX_M"].to_numpy(dtype=float)
            local_y = frame["CanvasLocalY_M"].to_numpy(dtype=float)
            margin = frame["AOIMarginM"].fillna(0.0).to_numpy(dtype=float)

            raw_minimap = valid & inside_rectangle(
                local_x,
                local_y,
                config.minimap_rect,
                margin,
            )
            raw_direction = valid & inside_rectangle(
                local_x,
                local_y,
                config.direction_panel_rect,
                margin,
            )
            raw_distance = valid & inside_rectangle(
                local_x,
                local_y,
                config.distance_panel_rect,
                margin,
            )

            valid_count += int(valid.sum())
            minimap_count += int(raw_minimap.sum())
            text_count += int((raw_direction | raw_distance).sum())

        rows.append(
            {
                "Condition": condition,
                "TrialFileCount": len(files),
                "ValidCanvasIntersectionCount": valid_count,
                "RawMiniMapRegionCount": minimap_count,
                "RawMiniMapRegionProportion": (
                    minimap_count / valid_count
                    if valid_count
                    else np.nan
                ),
                "RawTextAnyPanelRegionCount": text_count,
                "RawTextAnyPanelRegionProportion": (
                    text_count / valid_count
                    if valid_count
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct MiniMap and Text guidance gaze AOIs."
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
        default=Path.cwd() / "Results_MinimapText_GazeAOI",
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
        help="Process one trial for validation.",
    )
    parser.add_argument("--eye-accuracy-deg", type=float, default=1.1)
    parser.add_argument("--max-accuracy-margin-m", type=float, default=0.03)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_dir.resolve()
    conditions = tuple(args.conditions)
    config = Config(
        eye_accuracy_deg=args.eye_accuracy_deg,
        max_accuracy_margin_m=args.max_accuracy_margin_m,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "config.json").write_text(
        json.dumps(
            config_to_json(config, conditions),
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
        raise RuntimeError("No matching MiniMap/Text experiment logs found.")

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, log_path in enumerate(logs, start=1):
        try:
            summary = classify_trial(
                log_path,
                data_root,
                output_root,
                config,
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
    error_frame = pd.DataFrame(
        errors,
        columns=("LogFile", "ErrorType", "ErrorMessage"),
    )
    summary_path = output_root / "experiment_summary.csv"
    error_path = output_root / "errors.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
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
        "total_valid_canvas_intersections": (
            int(summary_frame["ValidCanvasIntersectionCount"].sum())
            if not summary_frame.empty
            else 0
        ),
    }
    (output_root / "batch_qc.json").write_text(
        json.dumps(batch_qc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    negative_control_path = output_root / "region_negative_control.csv"
    build_region_negative_control(
        output_root,
        conditions,
        config,
    ).to_csv(
        negative_control_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(json.dumps(batch_qc, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Errors: {error_path}")
    print(f"Region negative control: {negative_control_path}")

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
