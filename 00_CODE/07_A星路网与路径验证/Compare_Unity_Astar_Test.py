#!/usr/bin/env python3
"""Compare exact Unity A* recalculation with historical LineRenderer paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Validate_Astar_LineRenderer_Paths as validation


ROOT = Path(__file__).resolve().parent
TEST_DIR = (
    ROOT
    / "Results_Astar_LineRenderer_Validation"
    / "unity_exact_test"
)


def geometry(
    points: pd.DataFrame, query_id: str
) -> np.ndarray:
    return (
        points[
            (points["query_id"] == query_id)
            & (points["point_index"] >= 0)
        ]
        .sort_values("point_index")[["x", "y", "z"]]
        .to_numpy(dtype=float)
    )


def main() -> None:
    queries = pd.read_csv(TEST_DIR / "unity_test_queries.csv")
    historical = pd.read_csv(TEST_DIR / "unity_test_logged_points.csv")
    recalculated = pd.read_csv(TEST_DIR / "unity_recomputed_points.csv")

    rows: list[dict[str, object]] = []
    for query in queries.itertuples(index=False):
        old = geometry(historical, query.query_id)
        new = geometry(recalculated, query.query_id)
        failure = recalculated[
            (recalculated["query_id"] == query.query_id)
            & recalculated["error"].notna()
        ]
        if len(old) < 2 or len(new) < 2:
            rows.append(
                {
                    **query._asdict(),
                    "comparison_error": (
                        str(failure["error"].iloc[0])
                        if not failure.empty
                        else "Missing path geometry"
                    ),
                }
            )
            continue

        metrics = validation.compare_polylines(old, new)
        rows.append(
            {
                **query._asdict(),
                "historical_vertex_count": len(old),
                "recalculated_vertex_count": len(new),
                **metrics,
                # Suitable for testing same-version numerical repeatability.
                "near_exact_match": (
                    metrics["symmetric_mean_distance_m"] <= 0.10
                    and metrics["length_relative_error"] <= 0.01
                    and metrics["hausdorff_distance_m"] <= 0.25
                ),
                # Suitable for using the route as a broad spatial corridor.
                "same_corridor": (
                    metrics["symmetric_mean_distance_m"] <= 1.0
                    and metrics["coverage_within_2m"] >= 0.90
                ),
                "major_route_divergence": (
                    metrics["hausdorff_distance_m"] > 5.0
                    or metrics["symmetric_mean_distance_m"] > 2.0
                ),
                "comparison_error": "",
            }
        )

    comparison = pd.DataFrame(rows)
    successful = comparison[
        comparison["comparison_error"].fillna("") == ""
    ].copy()
    comparison.to_csv(
        TEST_DIR / "unity_exact_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    condition_rows: list[dict[str, object]] = []
    for condition, group in successful.groupby("condition"):
        condition_rows.append(
            {
                "condition": condition,
                "selected_trial_count": group[
                    ["participant", "trial_id"]
                ].drop_duplicates().shape[0],
                "query_count": len(group),
                "median_symmetric_mean_distance_m": group[
                    "symmetric_mean_distance_m"
                ].median(),
                "p90_symmetric_mean_distance_m": group[
                    "symmetric_mean_distance_m"
                ].quantile(0.90),
                "median_symmetric_p95_distance_m": group[
                    "symmetric_p95_distance_m"
                ].median(),
                "median_length_relative_error": group[
                    "length_relative_error"
                ].median(),
                "median_coverage_within_1m": group[
                    "coverage_within_1m"
                ].median(),
                "near_exact_match_rate": group["near_exact_match"].mean(),
                "same_corridor_rate": group["same_corridor"].mean(),
                "major_route_divergence_rate": group[
                    "major_route_divergence"
                ].mean(),
            }
        )
    condition_summary = pd.DataFrame(condition_rows)
    condition_summary.to_csv(
        TEST_DIR / "unity_exact_condition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exit_summary = (
        successful.groupby("target_exit", as_index=False)
        .agg(
            query_count=("query_id", "size"),
            median_symmetric_mean_distance_m=(
                "symmetric_mean_distance_m",
                "median",
            ),
            p90_symmetric_mean_distance_m=(
                "symmetric_mean_distance_m",
                lambda values: values.quantile(0.90),
            ),
            median_length_relative_error=(
                "length_relative_error",
                "median",
            ),
            same_corridor_rate=("same_corridor", "mean"),
            major_route_divergence_rate=(
                "major_route_divergence",
                "mean",
            ),
        )
    )
    exit_summary.to_csv(
        TEST_DIR / "unity_exact_exit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "share_with_caveats",
        "test_design": {
            "condition_count": int(queries["condition"].nunique()),
            "selected_trial_count": int(
                queries[["condition", "participant", "trial_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "query_count": len(queries),
            "successful_recalculation_count": len(successful),
        },
        "overall": {
            "median_symmetric_mean_distance_m": float(
                successful["symmetric_mean_distance_m"].median()
            ),
            "p90_symmetric_mean_distance_m": float(
                successful["symmetric_mean_distance_m"].quantile(0.90)
            ),
            "median_symmetric_p95_distance_m": float(
                successful["symmetric_p95_distance_m"].median()
            ),
            "median_length_relative_error": float(
                successful["length_relative_error"].median()
            ),
            "median_coverage_within_1m": float(
                successful["coverage_within_1m"].median()
            ),
            "near_exact_match_rate": float(
                successful["near_exact_match"].mean()
            ),
            "same_corridor_rate": float(
                successful["same_corridor"].mean()
            ),
            "major_route_divergence_rate": float(
                successful["major_route_divergence"].mean()
            ),
        },
        "interpretation": (
            "The current rescanned Unity A* graph often follows the same broad "
            "corridor, but it does not reproduce the historical LineRenderer "
            "path exactly. It should not replace the historical logged path "
            "for frame-level light-blanket AOI reconstruction."
        ),
    }
    (TEST_DIR / "unity_exact_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Three representative overlays: nearest median, largest non-divergent,
    # and worst route divergence.
    ordered = successful.sort_values("symmetric_mean_distance_m")
    median_value = successful["symmetric_mean_distance_m"].median()
    median_index = (
        successful["symmetric_mean_distance_m"] - median_value
    ).abs().idxmin()
    selected = [
        successful.loc[median_index],
        ordered.iloc[int(0.90 * (len(ordered) - 1))],
        ordered.iloc[-1],
    ]
    labels = ["Median", "90th percentile", "Worst"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for axis, label, row in zip(axes, labels, selected):
        old = geometry(historical, row["query_id"])
        new = geometry(recalculated, row["query_id"])
        axis.plot(
            old[:, 0],
            old[:, 2],
            linewidth=2,
            label="Historical LineRenderer",
        )
        axis.plot(
            new[:, 0],
            new[:, 2],
            linewidth=1.5,
            label="Unity rescanned A*",
        )
        axis.scatter(
            [old[0, 0], old[-1, 0]],
            [old[0, 2], old[-1, 2]],
            c=["green", "red"],
            s=25,
            zorder=3,
        )
        axis.set_title(
            f"{label}: {row['symmetric_mean_distance_m']:.2f} m\n"
            f"{row['condition']} / {row['participant']}"
        )
        axis.set_xlabel("Unity X (m)")
        axis.set_ylabel("Unity Z (m)")
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(TEST_DIR / "unity_exact_path_overlays.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    order = list(validation.CONDITIONS)
    data = [
        successful.loc[
            successful["condition"] == condition,
            "symmetric_mean_distance_m",
        ]
        for condition in order
    ]
    axis.boxplot(data, tick_labels=order, showfliers=True)
    axis.axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    axis.set_ylabel("Symmetric route distance (m)")
    axis.set_title("Exact Unity A* rescan vs historical LineRenderer")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(TEST_DIR / "unity_exact_error_by_condition.png", dpi=200)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(condition_summary.to_string(index=False))
    print(exit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
