#!/usr/bin/env python3
"""Validate the exported A* graph against historical LineRenderer paths.

For every VectorPath snapshot in ArrowRibbon and Full trials:
1. Parse the target exit from the historical Level log.
2. Snap the logged path start and target exit to the exported A* graph.
3. Recompute the shortest path on the exported directed graph.
4. Compare the recomputed polyline with the logged LineRenderer polyline.

The logged LineRenderer vertices contain a +0.1 m Y offset; it is removed before
comparison.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
SPATIAL_DIR = ROOT / "SpatialSceneExport"
OUTPUT_DIR = ROOT / "Results_Astar_LineRenderer_Validation"
DOC_DIR = ROOT / "Documentation_Guidance_Spatial_Causal_Analysis"

CONDITIONS = (
    "ArrowRibbon-HighSmoke",
    "ArrowRibbon-LowSmoke",
    "Full-HighSmoke",
    "Full-LowSmoke",
)

TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{3})\]"
    r"\s+\[Info\]VectorPath Points:\s*$"
)
POINT_RE = re.compile(
    r"^Point\s+(?P<index>\d+):\s*"
    r"\(\s*(?P<x>[-+]?\d+(?:\.\d+)?)\s*,"
    r"\s*(?P<y>[-+]?\d+(?:\.\d+)?)\s*,"
    r"\s*(?P<z>[-+]?\d+(?:\.\d+)?)\s*\)\s*$"
)
LEVEL_RE = re.compile(r"\[Info\]Level:\s*(?P<level>[1-8])\s+Show Line")

LINE_Y_OFFSET_M = 0.1
RESAMPLE_STEP_M = 0.5

# Operational reproducibility thresholds. Both strict and corridor-level
# criteria are reported; neither is used to discard data.
STRICT_MEAN_DISTANCE_M = 0.5
STRICT_LENGTH_REL_ERROR = 0.05
CORRIDOR_MEAN_DISTANCE_M = 1.0
CORRIDOR_LENGTH_REL_ERROR = 0.10
ENDPOINT_TOLERANCE_M = 0.75


@dataclass(frozen=True)
class Snapshot:
    timestamp_text: str
    timestamp: datetime
    points: np.ndarray


@dataclass
class TrialTrajectory:
    times_ns: np.ndarray
    positions: np.ndarray


def parse_log(log_path: Path) -> tuple[int | None, list[Snapshot]]:
    lines = log_path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()

    levels = {
        int(match.group("level"))
        for line in lines
        if (match := LEVEL_RE.search(line))
    }
    level = next(iter(levels)) if len(levels) == 1 else None

    snapshots: list[Snapshot] = []
    i = 0
    while i < len(lines):
        header = HEADER_RE.match(lines[i].strip())
        if header is None:
            i += 1
            continue

        timestamp_text = header.group("timestamp")
        points: list[tuple[float, float, float]] = []
        expected_index = 0
        i += 1

        while i < len(lines):
            match = POINT_RE.match(lines[i].strip())
            if match is None:
                break

            vertex_index = int(match.group("index"))
            if vertex_index != expected_index:
                points = []
                break

            points.append(
                (
                    float(match.group("x")),
                    float(match.group("y")) - LINE_Y_OFFSET_M,
                    float(match.group("z")),
                )
            )
            expected_index += 1
            i += 1

        if len(points) >= 2:
            snapshots.append(
                Snapshot(
                    timestamp_text=timestamp_text,
                    timestamp=datetime.strptime(
                        timestamp_text, TIMESTAMP_FORMAT
                    ),
                    points=np.asarray(points, dtype=float),
                )
            )

    return level, snapshots


def subject_path_for_log(log_path: Path) -> Path:
    expected_name = log_path.name.replace(
        "_Log_Info_", "_Subject_Simulation_Info_"
    )
    return log_path.with_name(expected_name)


def load_subject_trajectory(path: Path) -> TrialTrajectory | None:
    if not path.is_file():
        return None

    frame = pd.read_csv(path, usecols=["TimeStamp", "PosX", "PosY", "PosZ"])
    frame = frame.dropna(subset=["TimeStamp", "PosX", "PosY", "PosZ"])
    if frame.empty:
        return None

    timestamps = pd.to_datetime(
        frame["TimeStamp"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    valid = timestamps.notna().to_numpy()
    if not valid.any():
        return None

    times_ns = timestamps[valid].astype("int64").to_numpy()
    positions = frame.loc[valid, ["PosX", "PosY", "PosZ"]].to_numpy(
        dtype=float
    )
    order = np.argsort(times_ns)
    return TrialTrajectory(times_ns=times_ns[order], positions=positions[order])


def nearest_subject_sample(
    trajectory: TrialTrajectory | None, timestamp: datetime
) -> tuple[float, np.ndarray | None]:
    if trajectory is None or trajectory.times_ns.size == 0:
        return math.nan, None

    target_ns = pd.Timestamp(timestamp).value
    index = int(np.searchsorted(trajectory.times_ns, target_ns))
    candidates = [
        candidate
        for candidate in (index - 1, index)
        if 0 <= candidate < trajectory.times_ns.size
    ]
    best = min(
        candidates,
        key=lambda candidate: abs(
            int(trajectory.times_ns[candidate]) - target_ns
        ),
    )
    delta_ms = abs(int(trajectory.times_ns[best]) - target_ns) / 1_000_000
    return delta_ms, trajectory.positions[best]


def load_graph() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, int],
    list[list[tuple[int, float]]],
    list[list[tuple[int, float]]],
]:
    nodes = pd.read_csv(SPATIAL_DIR / "astar_nodes.csv")
    edges = pd.read_csv(SPATIAL_DIR / "astar_edges.csv")

    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    coordinates = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    id_to_index = {
        int(node_id): index for index, node_id in enumerate(node_ids)
    }

    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(len(node_ids))
    ]
    reverse_adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(len(node_ids))
    ]

    bad_edges = 0
    for row in edges.itertuples(index=False):
        from_id = int(row.from_node)
        to_id = int(row.to_node)
        if from_id not in id_to_index or to_id not in id_to_index:
            bad_edges += 1
            continue

        source = id_to_index[from_id]
        target = id_to_index[to_id]
        length = float(row.length_m)
        adjacency[source].append((target, length))
        reverse_adjacency[target].append((source, length))

    if bad_edges:
        raise RuntimeError(f"Exported graph has {bad_edges} orphan edges.")

    return (
        node_ids,
        coordinates,
        id_to_index,
        adjacency,
        reverse_adjacency,
    )


def reverse_dijkstra(
    reverse_adjacency: list[list[tuple[int, float]]], target: int
) -> tuple[np.ndarray, np.ndarray]:
    count = len(reverse_adjacency)
    distances = np.full(count, np.inf, dtype=float)
    next_hop = np.full(count, -1, dtype=np.int64)
    distances[target] = 0.0
    heap: list[tuple[float, int]] = [(0.0, target)]

    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue

        for predecessor, edge_length in reverse_adjacency[node]:
            candidate = distance + edge_length
            if candidate < distances[predecessor]:
                distances[predecessor] = candidate
                next_hop[predecessor] = node
                heapq.heappush(heap, (candidate, predecessor))

    return distances, next_hop


def reconstruct_indices(
    start: int, target: int, next_hop: np.ndarray
) -> list[int] | None:
    path = [start]
    current = start
    seen = {current}

    while current != target:
        current = int(next_hop[current])
        if current < 0 or current in seen:
            return None
        path.append(current)
        seen.add(current)

    return path


def deduplicate_polyline(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    return points[keep]


def build_recomputed_polyline(
    start_point: np.ndarray,
    exit_point: np.ndarray,
    start_index: int,
    target_index: int,
    next_hop: np.ndarray,
    node_coordinates: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    indices = reconstruct_indices(start_index, target_index, next_hop)
    if indices is None:
        return None, 0

    graph_points = node_coordinates[np.asarray(indices, dtype=int)]
    combined = np.vstack([start_point, graph_points, exit_point])
    return deduplicate_polyline(combined), len(indices)


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, step_m: float) -> np.ndarray:
    points = deduplicate_polyline(points)
    if len(points) < 2:
        return points.copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    total = cumulative[-1]
    if total <= 0:
        return points[:1].copy()

    distances = np.arange(0.0, total, step_m)
    if distances.size == 0 or not np.isclose(distances[-1], total):
        distances = np.r_[distances, total]

    output = np.empty((len(distances), 3), dtype=float)
    segment_index = np.searchsorted(
        cumulative, distances, side="right"
    ) - 1
    segment_index = np.clip(segment_index, 0, len(segment_lengths) - 1)

    local = distances - cumulative[segment_index]
    fractions = local / segment_lengths[segment_index]
    output[:] = (
        points[segment_index]
        + fractions[:, None]
        * (points[segment_index + 1] - points[segment_index])
    )
    return output


def point_to_polyline_distances(
    query_points: np.ndarray, polyline: np.ndarray
) -> np.ndarray:
    polyline = deduplicate_polyline(polyline)
    if len(polyline) == 1:
        return np.linalg.norm(query_points - polyline[0], axis=1)

    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared > 1e-12
    starts = starts[valid]
    vectors = vectors[valid]
    squared = squared[valid]

    if len(starts) == 0:
        return np.linalg.norm(query_points - polyline[0], axis=1)

    relative = query_points[:, None, :] - starts[None, :, :]
    fractions = np.einsum(
        "qsd,sd->qs", relative, vectors
    ) / squared[None, :]
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts[None, :, :] + fractions[:, :, None] * vectors
    distances = np.linalg.norm(
        query_points[:, None, :] - projections, axis=2
    )
    return distances.min(axis=1)


def compare_polylines(
    logged: np.ndarray, recomputed: np.ndarray
) -> dict[str, float]:
    logged_sample = resample_polyline(logged, RESAMPLE_STEP_M)
    recomputed_sample = resample_polyline(recomputed, RESAMPLE_STEP_M)

    logged_to_recomputed = point_to_polyline_distances(
        logged_sample, recomputed
    )
    recomputed_to_logged = point_to_polyline_distances(
        recomputed_sample, logged
    )
    combined = np.r_[logged_to_recomputed, recomputed_to_logged]

    logged_length = polyline_length(logged)
    recomputed_length = polyline_length(recomputed)
    absolute_length_error = abs(recomputed_length - logged_length)
    relative_length_error = (
        absolute_length_error / logged_length
        if logged_length > 1e-9
        else math.nan
    )

    return {
        "logged_length_m": logged_length,
        "recomputed_length_m": recomputed_length,
        "length_signed_difference_m": recomputed_length - logged_length,
        "length_absolute_error_m": absolute_length_error,
        "length_relative_error": relative_length_error,
        "logged_to_recomputed_mean_m": float(
            logged_to_recomputed.mean()
        ),
        "recomputed_to_logged_mean_m": float(
            recomputed_to_logged.mean()
        ),
        "symmetric_mean_distance_m": float(combined.mean()),
        "symmetric_p95_distance_m": float(np.quantile(combined, 0.95)),
        "hausdorff_distance_m": float(combined.max()),
        "coverage_within_0_5m": float((combined <= 0.5).mean()),
        "coverage_within_1m": float((combined <= 1.0).mean()),
        "coverage_within_2m": float((combined <= 2.0).mean()),
    }


def trial_id_from_log(log_path: Path) -> str:
    return log_path.name.split("_Log_Info_", 1)[0]


def summarize_trials(snapshot_frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "condition",
        "guidance_type",
        "smoke_level",
        "participant",
        "trial_id",
        "target_exit",
        "log_file",
    ]
    aggregations = {
        "snapshot_index": "count",
        "symmetric_mean_distance_m": "median",
        "symmetric_p95_distance_m": "median",
        "hausdorff_distance_m": "median",
        "length_relative_error": "median",
        "endpoint_error_m": "median",
        "start_node_snap_error_m": "median",
        "subject_start_error_m": "median",
        "subject_time_delta_ms": "median",
        "coverage_within_1m": "median",
        "strict_pass": "mean",
        "corridor_pass": "mean",
    }
    result = snapshot_frame.groupby(keys, dropna=False).agg(aggregations)
    result = result.reset_index().rename(
        columns={
            "snapshot_index": "snapshot_count",
            "symmetric_mean_distance_m": "median_symmetric_mean_distance_m",
            "symmetric_p95_distance_m": "median_symmetric_p95_distance_m",
            "hausdorff_distance_m": "median_hausdorff_distance_m",
            "length_relative_error": "median_length_relative_error",
            "endpoint_error_m": "median_endpoint_error_m",
            "start_node_snap_error_m": "median_start_node_snap_error_m",
            "subject_start_error_m": "median_subject_start_error_m",
            "subject_time_delta_ms": "median_subject_time_delta_ms",
            "coverage_within_1m": "median_coverage_within_1m",
            "strict_pass": "snapshot_strict_pass_rate",
            "corridor_pass": "snapshot_corridor_pass_rate",
        }
    )
    return result


def summarize_conditions(
    trial_frame: pd.DataFrame, snapshot_frame: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, trials in trial_frame.groupby("condition"):
        snapshots = snapshot_frame[snapshot_frame["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "trial_count": len(trials),
                "participant_count": trials["participant"].nunique(),
                "snapshot_count": len(snapshots),
                "median_trial_path_mean_distance_m": trials[
                    "median_symmetric_mean_distance_m"
                ].median(),
                "p90_trial_path_mean_distance_m": trials[
                    "median_symmetric_mean_distance_m"
                ].quantile(0.90),
                "median_trial_path_p95_distance_m": trials[
                    "median_symmetric_p95_distance_m"
                ].median(),
                "median_trial_length_relative_error": trials[
                    "median_length_relative_error"
                ].median(),
                "median_trial_endpoint_error_m": trials[
                    "median_endpoint_error_m"
                ].median(),
                "median_trial_coverage_within_1m": trials[
                    "median_coverage_within_1m"
                ].median(),
                "snapshot_strict_pass_rate": snapshots[
                    "strict_pass"
                ].mean(),
                "snapshot_corridor_pass_rate": snapshots[
                    "corridor_pass"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    snapshot_frame: pd.DataFrame,
    trial_frame: pd.DataFrame,
    graph_context: dict[str, object],
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    condition_order = [c for c in CONDITIONS if c in set(trial_frame.condition)]
    data = [
        trial_frame.loc[
            trial_frame.condition == condition,
            "median_symmetric_mean_distance_m",
        ].dropna()
        for condition in condition_order
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(data, tick_labels=condition_order, showfliers=False)
    ax.axhline(
        CORRIDOR_MEAN_DISTANCE_M,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label="1 m corridor threshold",
    )
    ax.set_ylabel("Trial median symmetric path distance (m)")
    ax.set_title("A* recomputation vs historical LineRenderer path")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "path_error_by_condition.png", dpi=200
    )
    plt.close(fig)

    sample = snapshot_frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["logged_length_m", "recomputed_length_m"]
    )
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for condition in condition_order:
        subset = sample[sample.condition == condition]
        ax.scatter(
            subset.logged_length_m,
            subset.recomputed_length_m,
            s=7,
            alpha=0.25,
            label=condition,
        )
    maximum = float(
        max(sample.logged_length_m.max(), sample.recomputed_length_m.max())
    )
    ax.plot([0, maximum], [0, maximum], color="black", linewidth=1)
    ax.set_xlabel("Historical LineRenderer length (m)")
    ax.set_ylabel("Recomputed A* length (m)")
    ax.set_title("Path-length agreement")
    ax.legend(frameon=False, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "path_length_agreement.png", dpi=200)
    plt.close(fig)

    valid = snapshot_frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["symmetric_mean_distance_m"]
    ).sort_values("symmetric_mean_distance_m")
    if len(valid) < 3:
        return

    selected_rows = [
        valid.iloc[0],
        valid.iloc[len(valid) // 2],
        valid.iloc[-1],
    ]
    labels = ["Best", "Median", "Worst"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, label, row in zip(axes, labels, selected_rows):
        logged, recomputed = geometry_for_result_row(row, graph_context)
        ax.plot(
            logged[:, 0],
            logged[:, 2],
            color="tab:blue",
            linewidth=2,
            label="Historical LineRenderer",
        )
        ax.plot(
            recomputed[:, 0],
            recomputed[:, 2],
            color="tab:orange",
            linewidth=1.5,
            label="Recomputed A*",
        )
        ax.scatter(
            [logged[0, 0], logged[-1, 0]],
            [logged[0, 2], logged[-1, 2]],
            color=["green", "red"],
            s=25,
            zorder=3,
        )
        ax.set_title(
            f"{label}: {row['symmetric_mean_distance_m']:.2f} m\n"
            f"{row['condition']} / {row['participant']}"
        )
        ax.set_xlabel("Unity X (m)")
        ax.set_ylabel("Unity Z (m)")
        ax.set_aspect("equal", adjustable="box")

    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_path_overlays.png", dpi=200)
    plt.close(fig)


def geometry_for_result_row(
    row: pd.Series, graph_context: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    log_path = Path(row["log_file"])
    level, snapshots = parse_log(log_path)
    snapshot = snapshots[int(row["snapshot_index"])]
    exit_letter = chr(64 + int(level))

    node_coordinates: np.ndarray = graph_context["node_coordinates"]
    tree: cKDTree = graph_context["tree"]
    exit_points: dict[str, np.ndarray] = graph_context["exit_points"]
    target_indices: dict[str, int] = graph_context["target_indices"]
    next_hops: dict[str, np.ndarray] = graph_context["next_hops"]

    _, start_index = tree.query(snapshot.points[0], k=1)
    recomputed, _ = build_recomputed_polyline(
        snapshot.points[0],
        exit_points[exit_letter],
        int(start_index),
        target_indices[exit_letter],
        next_hops[exit_letter],
        node_coordinates,
    )
    if recomputed is None:
        raise RuntimeError("Could not rebuild selected example path.")
    return snapshot.points, recomputed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        node_ids,
        node_coordinates,
        _,
        adjacency,
        reverse_adjacency,
    ) = load_graph()
    tree = cKDTree(node_coordinates)

    exits_frame = pd.read_csv(SPATIAL_DIR / "exits.csv")
    exit_points = {
        str(row.exit_id): np.asarray([row.x, row.y, row.z], dtype=float)
        for row in exits_frame.itertuples(index=False)
    }

    target_indices: dict[str, int] = {}
    target_snap_errors: dict[str, float] = {}
    distance_fields: dict[str, np.ndarray] = {}
    next_hops: dict[str, np.ndarray] = {}

    for exit_letter, exit_point in exit_points.items():
        snap_error, node_index = tree.query(exit_point, k=1)
        target_indices[exit_letter] = int(node_index)
        target_snap_errors[exit_letter] = float(snap_error)
        distances, next_hop = reverse_dijkstra(
            reverse_adjacency, int(node_index)
        )
        distance_fields[exit_letter] = distances
        next_hops[exit_letter] = next_hop

    log_paths = sorted(
        path
        for condition in CONDITIONS
        for path in (RAW / condition).glob("*/*_Log_Info_*.csv")
    )

    records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    parsed_trials = 0
    subject_file_trials = 0
    endpoint_target_matches = 0
    endpoint_target_checks = 0

    for log_number, log_path in enumerate(log_paths, start=1):
        condition = log_path.parents[1].name
        participant = log_path.parent.name
        trial_id = trial_id_from_log(log_path)
        guidance_type, smoke_level = condition.rsplit("-", 1)

        try:
            level, snapshots = parse_log(log_path)
            if level is None:
                raise RuntimeError("Could not infer one unique Level ID.")
            if not snapshots:
                raise RuntimeError("No valid VectorPath snapshots.")

            exit_letter = chr(64 + level)
            if exit_letter not in exit_points:
                raise RuntimeError(f"Exit {exit_letter} is not exported.")

            parsed_trials += 1
            subject_path = subject_path_for_log(log_path)
            trajectory = load_subject_trajectory(subject_path)
            if trajectory is not None:
                subject_file_trials += 1

            target_index = target_indices[exit_letter]
            next_hop = next_hops[exit_letter]

            for snapshot_index, snapshot in enumerate(snapshots):
                logged = snapshot.points
                start_snap_error, start_index = tree.query(logged[0], k=1)
                start_index = int(start_index)

                recomputed, graph_node_count = build_recomputed_polyline(
                    logged[0],
                    exit_points[exit_letter],
                    start_index,
                    target_index,
                    next_hop,
                    node_coordinates,
                )
                if recomputed is None:
                    raise RuntimeError(
                        f"No graph path for snapshot {snapshot_index}."
                    )

                metrics = compare_polylines(logged, recomputed)
                endpoint_error = float(
                    np.linalg.norm(logged[-1] - exit_points[exit_letter])
                )
                nearest_exit = min(
                    exit_points,
                    key=lambda letter: float(
                        np.linalg.norm(logged[-1] - exit_points[letter])
                    ),
                )
                endpoint_target_checks += 1
                endpoint_target_matches += int(nearest_exit == exit_letter)

                time_delta_ms, subject_position = nearest_subject_sample(
                    trajectory, snapshot.timestamp
                )
                subject_start_error = (
                    float(np.linalg.norm(logged[0] - subject_position))
                    if subject_position is not None
                    else math.nan
                )

                strict_pass = (
                    metrics["symmetric_mean_distance_m"]
                    <= STRICT_MEAN_DISTANCE_M
                    and metrics["length_relative_error"]
                    <= STRICT_LENGTH_REL_ERROR
                    and endpoint_error <= ENDPOINT_TOLERANCE_M
                )
                corridor_pass = (
                    metrics["symmetric_mean_distance_m"]
                    <= CORRIDOR_MEAN_DISTANCE_M
                    and metrics["length_relative_error"]
                    <= CORRIDOR_LENGTH_REL_ERROR
                    and endpoint_error <= ENDPOINT_TOLERANCE_M
                )

                records.append(
                    {
                        "condition": condition,
                        "guidance_type": guidance_type,
                        "smoke_level": smoke_level,
                        "participant": participant,
                        "trial_id": trial_id,
                        "target_exit": exit_letter,
                        "log_file": str(log_path.resolve()),
                        "subject_file": (
                            str(subject_path.resolve())
                            if subject_path.is_file()
                            else ""
                        ),
                        "snapshot_index": snapshot_index,
                        "timestamp": snapshot.timestamp_text,
                        "logged_vertex_count": len(logged),
                        "recomputed_graph_node_count": graph_node_count,
                        "start_node_id": int(node_ids[start_index]),
                        "target_node_id": int(node_ids[target_index]),
                        "start_node_snap_error_m": float(start_snap_error),
                        "target_node_snap_error_m": target_snap_errors[
                            exit_letter
                        ],
                        "endpoint_error_m": endpoint_error,
                        "endpoint_nearest_exit": nearest_exit,
                        "endpoint_matches_target": nearest_exit == exit_letter,
                        "subject_time_delta_ms": time_delta_ms,
                        "subject_start_error_m": subject_start_error,
                        "strict_pass": strict_pass,
                        "corridor_pass": corridor_pass,
                        **metrics,
                    }
                )

        except Exception as exc:
            errors.append(
                {
                    "condition": condition,
                    "participant": participant,
                    "trial_id": trial_id,
                    "log_file": str(log_path.resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if log_number % 20 == 0 or log_number == len(log_paths):
            print(
                f"[{log_number}/{len(log_paths)}] "
                f"snapshots={len(records)} errors={len(errors)}"
            )

    if not records:
        raise RuntimeError("No snapshot comparisons were produced.")

    snapshot_frame = pd.DataFrame(records)
    trial_frame = summarize_trials(snapshot_frame)
    condition_frame = summarize_conditions(trial_frame, snapshot_frame)
    error_frame = pd.DataFrame(errors)

    snapshot_frame.to_csv(
        OUTPUT_DIR / "snapshot_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trial_frame.to_csv(
        OUTPUT_DIR / "trial_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    condition_frame.to_csv(
        OUTPUT_DIR / "condition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    error_frame.to_csv(
        OUTPUT_DIR / "errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_context = {
        "node_coordinates": node_coordinates,
        "tree": tree,
        "exit_points": exit_points,
        "target_indices": target_indices,
        "next_hops": next_hops,
    }
    save_figures(snapshot_frame, trial_frame, graph_context)

    finite_subject = snapshot_frame["subject_start_error_m"].dropna()
    summary = {
        "question": (
            "Can the exported/re-scanned A* graph reproduce historical "
            "LineRenderer paths in ArrowRibbon and Full trials?"
        ),
        "conditions": list(CONDITIONS),
        "graph": {
            "node_count": int(len(node_ids)),
            "directed_edge_count": int(
                sum(len(neighbors) for neighbors in adjacency)
            ),
            "exit_target_node_snap_error_m": target_snap_errors,
        },
        "coverage": {
            "candidate_log_count": len(log_paths),
            "parsed_trial_count": parsed_trials,
            "failed_trial_count": len(errors),
            "trial_with_subject_trajectory_count": subject_file_trials,
            "snapshot_comparison_count": len(snapshot_frame),
            "endpoint_target_match_rate": (
                endpoint_target_matches / endpoint_target_checks
                if endpoint_target_checks
                else None
            ),
        },
        "overall_snapshot_metrics": {
            "median_symmetric_mean_distance_m": float(
                snapshot_frame["symmetric_mean_distance_m"].median()
            ),
            "p90_symmetric_mean_distance_m": float(
                snapshot_frame["symmetric_mean_distance_m"].quantile(0.90)
            ),
            "median_symmetric_p95_distance_m": float(
                snapshot_frame["symmetric_p95_distance_m"].median()
            ),
            "median_length_relative_error": float(
                snapshot_frame["length_relative_error"].median()
            ),
            "median_endpoint_error_m": float(
                snapshot_frame["endpoint_error_m"].median()
            ),
            "median_start_node_snap_error_m": float(
                snapshot_frame["start_node_snap_error_m"].median()
            ),
            "median_subject_start_error_m": (
                float(finite_subject.median())
                if len(finite_subject)
                else None
            ),
            "strict_pass_rate": float(snapshot_frame["strict_pass"].mean()),
            "corridor_pass_rate": float(
                snapshot_frame["corridor_pass"].mean()
            ),
        },
        "thresholds": {
            "strict": {
                "symmetric_mean_distance_m_max": STRICT_MEAN_DISTANCE_M,
                "length_relative_error_max": STRICT_LENGTH_REL_ERROR,
                "endpoint_error_m_max": ENDPOINT_TOLERANCE_M,
            },
            "corridor": {
                "symmetric_mean_distance_m_max": CORRIDOR_MEAN_DISTANCE_M,
                "length_relative_error_max": CORRIDOR_LENGTH_REL_ERROR,
                "endpoint_error_m_max": ENDPOINT_TOLERANCE_M,
            },
        },
        "known_method_difference": (
            "Historical vectorPath may include Seeker modifiers/smoothing, "
            "whereas validation recomputes shortest paths from exported graph "
            "nodes and Euclidean edge weights."
        ),
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    methodology = f"""# A* vs LineRenderer validation

- Candidate logs: {len(log_paths)}
- Parsed trials: {parsed_trials}
- Compared snapshots: {len(snapshot_frame)}
- Graph: {len(node_ids)} walkable nodes and
  {sum(len(x) for x in adjacency)} directed edges.
- Logged LineRenderer Y offset removed: {LINE_Y_OFFSET_M} m.
- Polyline resampling interval: {RESAMPLE_STEP_M} m.
- Target exit source: historical `Level: n Show Line` log.
- Start validation source: timestamp-matched Subject Simulation position.
- Distance metric: bidirectional point-to-polyline distance in Unity XYZ space.

The strict and corridor thresholds are operational reproducibility criteria,
not hypothesis-test p-value thresholds. Historical `vectorPath` may include
Seeker modifiers or smoothing that cannot be recovered from node/edge CSV alone.
"""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "README_Astar_LineRenderer_Validation.md").write_text(methodology, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
