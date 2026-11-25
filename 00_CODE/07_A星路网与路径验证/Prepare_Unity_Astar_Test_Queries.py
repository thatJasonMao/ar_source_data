#!/usr/bin/env python3
"""Prepare a stratified historical-path sample for exact Unity A* recalculation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import Validate_Astar_LineRenderer_Paths as validation


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Results_Astar_LineRenderer_Validation" / "unity_exact_test"
TRIALS_PER_CONDITION = 8
SNAPSHOTS_PER_TRIAL = 5


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exits = pd.read_csv(validation.SPATIAL_DIR / "exits.csv").set_index(
        "exit_id"
    )
    query_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    query_number = 0
    selected_trials = 0
    for condition in validation.CONDITIONS:
        logs = sorted(
            (validation.RAW / condition).glob("*/*_Log_Info_*.csv")
        )
        if len(logs) > TRIALS_PER_CONDITION:
            selected_indices = np.linspace(
                0, len(logs) - 1, TRIALS_PER_CONDITION
            ).round().astype(int)
            logs = [logs[index] for index in selected_indices]

        for log_path in logs:
            level, snapshots = validation.parse_log(log_path)
            if level is None or not snapshots:
                continue

            exit_id = chr(64 + level)
            exit_row = exits.loc[exit_id]
            if len(snapshots) > SNAPSHOTS_PER_TRIAL:
                snapshot_indices = np.linspace(
                    0, len(snapshots) - 1, SNAPSHOTS_PER_TRIAL
                ).round().astype(int)
            else:
                snapshot_indices = np.arange(len(snapshots), dtype=int)

            selected_trials += 1
            for snapshot_index in np.unique(snapshot_indices):
                snapshot = snapshots[int(snapshot_index)]
                query_id = f"Q{query_number:04d}"
                query_number += 1
                query_rows.append(
                    {
                        "query_id": query_id,
                        "condition": condition,
                        "participant": log_path.parent.name,
                        "trial_id": validation.trial_id_from_log(log_path),
                        "log_file": str(log_path.resolve()),
                        "snapshot_index": int(snapshot_index),
                        "timestamp": snapshot.timestamp_text,
                        "target_exit": exit_id,
                        "start_x": float(snapshot.points[0, 0]),
                        "start_y": float(snapshot.points[0, 1]),
                        "start_z": float(snapshot.points[0, 2]),
                        "end_x": float(exit_row["x"]),
                        "end_y": float(exit_row["y"]),
                        "end_z": float(exit_row["z"]),
                    }
                )
                for point_index, point in enumerate(snapshot.points):
                    point_rows.append(
                        {
                            "query_id": query_id,
                            "point_index": point_index,
                            "x": float(point[0]),
                            "y": float(point[1]),
                            "z": float(point[2]),
                        }
                    )

    queries = pd.DataFrame(query_rows)
    points = pd.DataFrame(point_rows)
    queries.to_csv(
        OUTPUT_DIR / "unity_test_queries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    points.to_csv(
        OUTPUT_DIR / "unity_test_logged_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "sampling": (
            f"Up to {TRIALS_PER_CONDITION} evenly spaced trials per "
            f"condition and {SNAPSHOTS_PER_TRIAL} temporal snapshots per trial."
        ),
        "condition_count": int(queries["condition"].nunique()),
        "selected_trial_count": selected_trials,
        "query_count": len(queries),
        "historical_point_count": len(points),
    }
    (OUTPUT_DIR / "query_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(OUTPUT_DIR / "unity_test_queries.csv")


if __name__ == "__main__":
    main()
