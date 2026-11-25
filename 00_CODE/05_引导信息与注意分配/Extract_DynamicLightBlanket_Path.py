#!/usr/bin/env python3
"""
Extract timestamped LightBlanketController VectorPath snapshots from experiment logs.

The Unity controller writes blocks in the following form:

    [2026_06_04_17_18_42_494] [Info]VectorPath Points:
    Point 0: (-6.96, -15.67, -11.95)
    Point 1: (...)

The logged coordinates are already world-space LineRenderer coordinates and already
include the controller's +0.1 m vertical offset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable


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
TRIAL_RE = re.compile(r"^(?P<trial>.+?)_Log_Info_")


@dataclass(frozen=True)
class Point:
    index: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Snapshot:
    timestamp_text: str
    timestamp: datetime
    points: tuple[Point, ...]


def parse_log(path: Path) -> list[Snapshot]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    snapshots: list[Snapshot] = []
    i = 0

    while i < len(lines):
        header = HEADER_RE.match(lines[i].strip())
        if header is None:
            i += 1
            continue

        timestamp_text = header.group("timestamp")
        timestamp = datetime.strptime(timestamp_text, TIMESTAMP_FORMAT)
        points: list[Point] = []
        i += 1

        while i < len(lines):
            point_match = POINT_RE.match(lines[i].strip())
            if point_match is None:
                break

            points.append(
                Point(
                    index=int(point_match.group("index")),
                    x=float(point_match.group("x")),
                    y=float(point_match.group("y")),
                    z=float(point_match.group("z")),
                )
            )
            i += 1

        if points:
            snapshots.append(
                Snapshot(
                    timestamp_text=timestamp_text,
                    timestamp=timestamp,
                    points=tuple(points),
                )
            )

    return snapshots


def infer_metadata(log_path: Path, data_root: Path | None) -> dict[str, str]:
    condition = ""
    participant_folder = log_path.parent.name

    if data_root is not None:
        try:
            relative = log_path.resolve().relative_to(data_root.resolve())
            if len(relative.parts) >= 1:
                condition = relative.parts[0]
            if len(relative.parts) >= 2:
                participant_folder = relative.parts[1]
        except ValueError:
            pass

    trial_match = TRIAL_RE.match(log_path.name)
    trial_id = trial_match.group("trial") if trial_match else log_path.stem

    guidance_type = condition
    smoke_level = ""
    if "-" in condition:
        guidance_type, smoke_level = condition.rsplit("-", 1)

    return {
        "condition": condition,
        "guidance_type": guidance_type,
        "smoke_level": smoke_level,
        "participant_folder": participant_folder,
        "trial_id": trial_id,
        "log_file": str(log_path.resolve()),
    }


def write_vertices(
    output_path: Path,
    log_path: Path,
    snapshots: Iterable[Snapshot],
    metadata: dict[str, str],
    line_width_m: float,
) -> int:
    fieldnames = [
        "Condition",
        "GuidanceType",
        "SmokeLevel",
        "ParticipantFolder",
        "TrialID",
        "LogFile",
        "SnapshotIndex",
        "TimeStamp",
        "VertexIndex",
        "X",
        "Y",
        "Z",
        "LineWidthM",
    ]

    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for snapshot_index, snapshot in enumerate(snapshots):
            for point in snapshot.points:
                writer.writerow(
                    {
                        "Condition": metadata["condition"],
                        "GuidanceType": metadata["guidance_type"],
                        "SmokeLevel": metadata["smoke_level"],
                        "ParticipantFolder": metadata["participant_folder"],
                        "TrialID": metadata["trial_id"],
                        "LogFile": str(log_path.resolve()),
                        "SnapshotIndex": snapshot_index,
                        "TimeStamp": snapshot.timestamp_text,
                        "VertexIndex": point.index,
                        "X": f"{point.x:.6f}",
                        "Y": f"{point.y:.6f}",
                        "Z": f"{point.z:.6f}",
                        "LineWidthM": f"{line_width_m:.6f}",
                    }
                )
                row_count += 1

    return row_count


def write_snapshot_summary(
    output_path: Path,
    snapshots: list[Snapshot],
    metadata: dict[str, str],
) -> None:
    fieldnames = [
        "Condition",
        "ParticipantFolder",
        "TrialID",
        "SnapshotIndex",
        "TimeStamp",
        "PointCount",
        "StartX",
        "StartY",
        "StartZ",
        "EndX",
        "EndY",
        "EndZ",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for snapshot_index, snapshot in enumerate(snapshots):
            start = snapshot.points[0]
            end = snapshot.points[-1]
            writer.writerow(
                {
                    "Condition": metadata["condition"],
                    "ParticipantFolder": metadata["participant_folder"],
                    "TrialID": metadata["trial_id"],
                    "SnapshotIndex": snapshot_index,
                    "TimeStamp": snapshot.timestamp_text,
                    "PointCount": len(snapshot.points),
                    "StartX": f"{start.x:.6f}",
                    "StartY": f"{start.y:.6f}",
                    "StartZ": f"{start.z:.6f}",
                    "EndX": f"{end.x:.6f}",
                    "EndY": f"{end.y:.6f}",
                    "EndZ": f"{end.z:.6f}",
                }
            )


def build_qc(
    log_path: Path,
    snapshots: list[Snapshot],
    vertex_rows: int,
    line_width_m: float,
) -> dict[str, object]:
    intervals = [
        (current.timestamp - previous.timestamp).total_seconds()
        for previous, current in zip(snapshots, snapshots[1:])
    ]
    point_counts = [len(snapshot.points) for snapshot in snapshots]

    sequential_index_failures = sum(
        tuple(point.index for point in snapshot.points)
        != tuple(range(len(snapshot.points)))
        for snapshot in snapshots
    )

    return {
        "source_log": str(log_path.resolve()),
        "snapshot_count": len(snapshots),
        "vertex_row_count": vertex_rows,
        "line_width_m_from_guidance_prefab": line_width_m,
        "first_timestamp": snapshots[0].timestamp_text if snapshots else None,
        "last_timestamp": snapshots[-1].timestamp_text if snapshots else None,
        "median_snapshot_interval_s": median(intervals) if intervals else None,
        "min_snapshot_interval_s": min(intervals) if intervals else None,
        "max_snapshot_interval_s": max(intervals) if intervals else None,
        "min_point_count": min(point_counts) if point_counts else None,
        "median_point_count": median(point_counts) if point_counts else None,
        "max_point_count": max(point_counts) if point_counts else None,
        "nonsequential_vertex_index_snapshots": sequential_index_failures,
        "coordinates_are_world_space": True,
        "vertical_offset_already_in_log_m": 0.1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract timestamped dynamic LightBlanket LineRenderer paths."
    )
    parser.add_argument("log_file", type=Path, help="A *_Log_Info_*.csv file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Experiment data root, used to infer condition metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for extracted CSV and QC JSON files.",
    )
    parser.add_argument(
        "--line-width-m",
        type=float,
        default=0.5,
        help="Rendered LightBlanket width in metres. Guidance.prefab uses 0.5.",
    )
    args = parser.parse_args()

    log_path = args.log_file.resolve()
    output_dir = args.output_dir.resolve()

    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = parse_log(log_path)
    if not snapshots:
        raise RuntimeError(f"No VectorPath snapshots found in: {log_path}")

    metadata = infer_metadata(log_path, args.data_root)
    stem = log_path.stem.replace("_Log_Info_", "_")
    vertices_path = output_dir / f"{stem}_LightBlanketVertices.csv"
    summary_path = output_dir / f"{stem}_LightBlanketSnapshots.csv"
    qc_path = output_dir / f"{stem}_LightBlanketQC.json"

    vertex_rows = write_vertices(
        vertices_path,
        log_path,
        snapshots,
        metadata,
        args.line_width_m,
    )
    write_snapshot_summary(summary_path, snapshots, metadata)

    qc = build_qc(log_path, snapshots, vertex_rows, args.line_width_m)
    qc_path.write_text(
        json.dumps(qc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(qc, ensure_ascii=False, indent=2))
    print(f"Vertices: {vertices_path}")
    print(f"Snapshots: {summary_path}")
    print(f"QC: {qc_path}")


if __name__ == "__main__":
    main()
