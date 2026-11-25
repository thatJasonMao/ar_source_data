#!/usr/bin/env python3
"""Correct exit labels and remove A* / spatial tails behind terminal exits.

This is a scene-specific, idempotent correction pipeline.  It always rebuilds
the corrected CSVs from ``raw_before_exit_correction`` so rerunning it cannot
progressively delete more of the network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "SpatialSceneExport"
BACKUP_DIR = DATA_DIR / "raw_before_exit_correction"
FINAL_NODE_FILENAME = "astar_nodes.csv"
FINAL_EDGE_FILENAME = "astar_edges.csv"
FINAL_FLOOR_FILENAME = "floors.csv"
FINAL_ZONE_FILENAME = "spatial_zones.csv"

SOURCE_FILES = (
    "astar_nodes.csv",
    "astar_edges.csv",
    "exits.csv",
    "floors.csv",
    "spatial_zones.csv",
)

# Bounds are the six obsolete side-corridor portions visible in the Unity
# export.  B and C here use the corrected labels.
REMOVED_XZ_BOUNDS = {
    "B": (36.64, 40.28, -35.280, -14.29),
    "C": (-40.28, -36.64, -99.13, -80.518),
    "D": (-40.28, -36.64, -35.282, -14.29),
    "E": (36.64, 40.28, -7.96, 11.515),
    "G": (-40.28, -36.64, -7.96, 11.515),
    "H": (-40.28, -36.64, 56.700, 77.47),
}
PRUNED_EXITS = tuple(REMOVED_XZ_BOUNDS)
EXIT_ORDER = ("A", "B", "C", "D", "E", "F", "G", "H")
WALKING_PLANE_TOLERANCE_M = 0.15
HALL_ENVELOPE_PADDING_M = 1.0


def ensure_backup() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        destination = BACKUP_DIR / filename
        if not destination.exists():
            shutil.copy2(DATA_DIR / filename, destination)


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        Path(filename).stem: pd.read_csv(BACKUP_DIR / filename)
        for filename in SOURCE_FILES
    }


def swap_b_c_exit_labels(
    exits: pd.DataFrame, zones: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected_exits = exits.copy()
    b_values = corrected_exits.loc[
        corrected_exits["exit_id"] == "B",
        corrected_exits.columns.difference(["exit_id"], sort=False),
    ].copy()
    c_values = corrected_exits.loc[
        corrected_exits["exit_id"] == "C",
        corrected_exits.columns.difference(["exit_id"], sort=False),
    ].copy()
    value_columns = corrected_exits.columns.difference(
        ["exit_id"], sort=False
    )
    corrected_exits.loc[
        corrected_exits["exit_id"] == "B", value_columns
    ] = c_values.to_numpy()
    corrected_exits.loc[
        corrected_exits["exit_id"] == "C", value_columns
    ] = b_values.to_numpy()

    corrected_zones = zones.copy()
    corrected_zones["zone_id"] = corrected_zones["zone_id"].replace(
        {
            "ExitZone_B": "__TEMP_EXIT_ZONE__",
            "ExitZone_C": "ExitZone_B",
            "__TEMP_EXIT_ZONE__": "ExitZone_C",
        }
    )
    return corrected_exits, corrected_zones


def nearest_node_id(
    nodes: pd.DataFrame, exit_row: pd.Series
) -> int:
    delta = nodes[["x", "y", "z"]].to_numpy(float) - exit_row[
        ["x", "y", "z"]
    ].to_numpy(float)
    return int(nodes.loc[np.square(delta).sum(axis=1).argmin(), "node_id"])


def identify_tail_nodes(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    exits: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, set[int]]]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(int))
    graph.add_edges_from(
        edges[["from_node", "to_node"]].astype(int).itertuples(
            index=False, name=None
        )
    )
    by_id = nodes.set_index("node_id")
    exit_table = exits.set_index("exit_id")
    anchors: dict[str, int] = {}
    tails: dict[str, set[int]] = {}

    for exit_id in EXIT_ORDER:
        anchor = nearest_node_id(nodes, exit_table.loc[exit_id])
        anchors[exit_id] = anchor
        tails[exit_id] = set()
        if exit_id not in PRUNED_EXITS:
            continue

        neighbors = list(graph.neighbors(anchor))
        graph_without_anchor = graph.copy()
        graph_without_anchor.remove_node(anchor)
        neighbor_components: list[set[int]] = []
        for neighbor in neighbors:
            component = set(
                nx.node_connected_component(graph_without_anchor, neighbor)
            )
            if component not in neighbor_components:
                neighbor_components.append(component)

        candidates = [
            component
            for component in neighbor_components
            if not (by_id.loc[list(component), "x"].abs() < 15.0).any()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Exit {exit_id}: expected one non-interior tail component, "
                f"found {len(candidates)}"
            )
        tails[exit_id] = candidates[0]
    return anchors, tails


def complement_intervals(
    lower: float,
    upper: float,
    blocked: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(blocked):
        start, end = max(lower, start), min(upper, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    allowed: list[tuple[float, float]] = []
    cursor = lower
    for start, end in merged:
        if start > cursor:
            allowed.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < upper:
        allowed.append((cursor, upper))
    return allowed


def nodes_in_stair_zones(
    nodes: pd.DataFrame, zones: pd.DataFrame
) -> pd.Series:
    """Return nodes in a stair footprint; their intermediate Y is preserved."""
    mask = np.zeros(len(nodes), dtype=bool)
    for stair in zones.loc[zones["zone_type"] == "Stair"].itertuples(
        index=False
    ):
        mask |= (
            (nodes["x"] >= stair.center_x - stair.size_x / 2)
            & (nodes["x"] <= stair.center_x + stair.size_x / 2)
            & (nodes["z"] >= stair.center_z - stair.size_z / 2)
            & (nodes["z"] <= stair.center_z + stair.size_z / 2)
        ).to_numpy()
    return pd.Series(mask, index=nodes.index)


def walking_plane_mask(
    nodes: pd.DataFrame,
    raw_floors: pd.DataFrame,
    zones: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    floor_y = raw_floors.set_index("floor_id")["center_y"].astype(float)
    on_hall = (
        nodes["y"].astype(float) - float(floor_y.loc["Hall"])
    ).abs() <= WALKING_PLANE_TOLERANCE_M
    on_platform = (
        nodes["y"].astype(float) - float(floor_y.loc["Platform"])
    ).abs() <= WALKING_PLANE_TOLERANCE_M
    in_stair = nodes_in_stair_zones(nodes, zones)
    return on_hall | on_platform | in_stair, in_stair


def build_regular_floor_envelopes(
    cleaned_nodes: pd.DataFrame,
    raw_floors: pd.DataFrame,
) -> pd.DataFrame:
    """Build one regular Hall cuboid from the retained Hall-plane nodes."""
    hall = raw_floors.loc[raw_floors["floor_id"] == "Hall"].iloc[0]
    platform = raw_floors.loc[
        raw_floors["floor_id"] == "Platform"
    ].iloc[0]
    hall_nodes = cleaned_nodes.loc[
        (
            cleaned_nodes["y"].astype(float)
            - float(hall.center_y)
        ).abs()
        <= WALKING_PLANE_TOLERANCE_M
    ]
    if hall_nodes.empty:
        raise RuntimeError("No retained Hall-plane nodes for Hall envelope")
    hall_x0 = float(hall_nodes["x"].min() - HALL_ENVELOPE_PADDING_M)
    hall_x1 = float(hall_nodes["x"].max() + HALL_ENVELOPE_PADDING_M)
    hall_z0 = float(hall_nodes["z"].min() - HALL_ENVELOPE_PADDING_M)
    hall_z1 = float(hall_nodes["z"].max() + HALL_ENVELOPE_PADDING_M)
    rows = [
        {
            "zone_id": "Hall",
            "zone_type": "Floor",
            "floor_id": "Hall",
            "center_x": (hall_x0 + hall_x1) / 2,
            "center_y": float(hall.center_y),
            "center_z": (hall_z0 + hall_z1) / 2,
            "size_x": hall_x1 - hall_x0,
            "size_y": float(hall.size_y),
            "size_z": hall_z1 - hall_z0,
        },
        {
            "zone_id": "Platform",
            "zone_type": "Floor",
            "floor_id": "Platform",
            "center_x": float(platform.center_x),
            "center_y": float(platform.center_y),
            "center_z": float(platform.center_z),
            "size_x": float(platform.size_x),
            "size_y": float(platform.size_y),
            "size_z": float(platform.size_z),
        },
    ]
    return pd.DataFrame(rows)


def connected_component_count(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> int:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(int))
    graph.add_edges_from(
        edges[["from_node", "to_node"]].astype(int).itertuples(
            index=False, name=None
        )
    )
    return nx.number_connected_components(graph)


def main() -> None:
    raw_sources_available = all(
        (BACKUP_DIR / filename).exists() for filename in SOURCE_FILES
    )
    final_outputs_available = all(
        (DATA_DIR / filename).exists()
        for filename in (
            FINAL_NODE_FILENAME,
            FINAL_EDGE_FILENAME,
            FINAL_FLOOR_FILENAME,
            FINAL_ZONE_FILENAME,
        )
    )
    if not raw_sources_available and final_outputs_available:
        print(
            "Original reconstruction CSVs were intentionally removed during "
            "final consolidation. Canonical outputs already exist in "
            f"{DATA_DIR}."
        )
        return
    ensure_backup()
    raw = load_raw()
    nodes = raw["astar_nodes"].copy()
    edges = raw["astar_edges"].copy()
    exits, zones = swap_b_c_exit_labels(
        raw["exits"], raw["spatial_zones"]
    )
    anchors, tails = identify_tail_nodes(nodes, edges, exits)
    removed_ids = set().union(*(tails[exit_id] for exit_id in PRUNED_EXITS))

    corrected_nodes = nodes.loc[
        ~nodes["node_id"].isin(removed_ids)
    ].copy()
    corrected_edges = edges.loc[
        ~edges["from_node"].isin(removed_ids)
        & ~edges["to_node"].isin(removed_ids)
    ].copy()
    on_walking_structure, in_stair = walking_plane_mask(
        corrected_nodes, raw["floors"], zones
    )
    off_plane_nodes = corrected_nodes.loc[~on_walking_structure].copy()
    off_plane_node_ids = set(off_plane_nodes["node_id"].astype(int))
    corrected_nodes = corrected_nodes.loc[on_walking_structure].copy()
    corrected_edges = corrected_edges.loc[
        ~corrected_edges["from_node"].isin(off_plane_node_ids)
        & ~corrected_edges["to_node"].isin(off_plane_node_ids)
    ].copy()
    linked_node_ids = set(corrected_edges["from_node"]).union(
        corrected_edges["to_node"]
    )
    isolated_nodes = corrected_nodes.loc[
        ~corrected_nodes["node_id"].isin(linked_node_ids)
    ].copy()
    isolated_node_ids = set(isolated_nodes["node_id"].astype(int))
    corrected_nodes = corrected_nodes.loc[
        ~corrected_nodes["node_id"].isin(isolated_node_ids)
    ].copy()
    corrected_floors = build_regular_floor_envelopes(
        corrected_nodes, raw["floors"]
    )

    non_floor_zones = zones.loc[zones["zone_type"] != "Floor"].copy()
    corrected_zones = pd.concat(
        [non_floor_zones, corrected_floors], ignore_index=True
    )

    by_id = nodes.set_index("node_id")
    removed_rows = []
    for exit_id in PRUNED_EXITS:
        for node_id in sorted(tails[exit_id]):
            row = by_id.loc[node_id]
            removed_rows.append(
                {
                    "exit_id": exit_id,
                    "anchor_node_id": anchors[exit_id],
                    "node_id": node_id,
                    "x": row.x,
                    "y": row.y,
                    "z": row.z,
                    "reason": "small_noninterior_component_behind_exit",
                }
            )
    removed_nodes = pd.DataFrame(removed_rows)
    isolated_nodes_log = isolated_nodes[
        ["node_id", "graph_index", "x", "y", "z", "tag", "walkable"]
    ].copy()
    isolated_nodes_log["reason"] = "degree_zero_no_incident_edge"
    floor_y = raw["floors"].set_index("floor_id")["center_y"].astype(float)
    off_plane_nodes_log = off_plane_nodes[
        ["node_id", "graph_index", "x", "y", "z", "tag", "walkable"]
    ].copy()
    off_plane_nodes_log["distance_to_hall_plane_m"] = (
        off_plane_nodes_log["y"] - float(floor_y.loc["Hall"])
    ).abs()
    off_plane_nodes_log["distance_to_platform_plane_m"] = (
        off_plane_nodes_log["y"] - float(floor_y.loc["Platform"])
    ).abs()
    off_plane_nodes_log["in_stair_footprint"] = False
    off_plane_nodes_log["reason"] = (
        "outside_walking_plane_tolerance_and_not_stair"
    )

    hall = raw["floors"].loc[
        raw["floors"]["floor_id"] == "Hall"
    ].iloc[0]
    min_y = float(hall.center_y - hall.size_y / 2)
    max_y = float(hall.center_y + hall.size_y / 2)
    region_rows = []
    for exit_id, (min_x, max_x, min_z, max_z) in REMOVED_XZ_BOUNDS.items():
        region_rows.append(
            {
                "zone_id": f"RemovedAfterExit_{exit_id}",
                "zone_type": "RemovedAfterExit",
                "exit_id": exit_id,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "min_z": min_z,
                "max_z": max_z,
                "removed_node_count": len(tails[exit_id]),
            }
        )
    removed_regions = pd.DataFrame(region_rows)

    corrected_nodes.to_csv(DATA_DIR / FINAL_NODE_FILENAME, index=False)
    corrected_edges.to_csv(DATA_DIR / FINAL_EDGE_FILENAME, index=False)
    exits.to_csv(DATA_DIR / "exits.csv", index=False)
    corrected_floors.to_csv(DATA_DIR / FINAL_FLOOR_FILENAME, index=False)
    corrected_zones.to_csv(DATA_DIR / FINAL_ZONE_FILENAME, index=False)
    removed_nodes.to_csv(
        DATA_DIR / "exit_pruning_removed_nodes.csv", index=False
    )
    isolated_nodes_log.to_csv(
        DATA_DIR / "astar_isolated_removed_nodes.csv", index=False
    )
    off_plane_nodes_log.to_csv(
        DATA_DIR / "astar_off_plane_removed_nodes.csv", index=False
    )
    removed_regions.to_csv(
        DATA_DIR / "removed_spatial_regions.csv", index=False
    )

    summary = {
        "label_correction": {
            "B": "physical location previously labelled C",
            "C": "physical location previously labelled B",
        },
        "tail_policy": {
            "kept_unchanged": ["A", "F"],
            "pruned": list(PRUNED_EXITS),
            "criterion": (
                "After removing the nearest exit anchor, prune the single "
                "small neighbor component containing no interior nodes."
            ),
        },
        "walking_plane_policy": {
            "hall_plane_y_m": float(floor_y.loc["Hall"]),
            "platform_plane_y_m": float(floor_y.loc["Platform"]),
            "vertical_tolerance_m": WALKING_PLANE_TOLERANCE_M,
            "stair_exception": (
                "Preserve every node inside either exported stair X-Z "
                "footprint regardless of Y elevation."
            ),
            "preserved_stair_footprint_node_count": int(in_stair.sum()),
        },
        "graph_before": {
            "node_count": len(nodes),
            "directed_edge_count": len(edges),
            "connected_component_count": connected_component_count(
                nodes, edges
            ),
        },
        "graph_after": {
            "node_count": len(corrected_nodes),
            "directed_edge_count": len(corrected_edges),
            "connected_component_count": connected_component_count(
                corrected_nodes, corrected_edges
            ),
        },
        "removed_tail_node_count": len(removed_ids),
        "removed_off_plane_node_count": len(off_plane_node_ids),
        "removed_isolated_node_count": len(isolated_node_ids),
        "total_removed_node_count": (
            len(removed_ids)
            + len(off_plane_node_ids)
            + len(isolated_node_ids)
        ),
        "removed_nodes_by_exit": {
            exit_id: len(tails[exit_id]) for exit_id in EXIT_ORDER
        },
        "spatial_structure": {
            "original_floor_box_count": len(raw["floors"]),
            "corrected_floor_box_count": len(corrected_floors),
            "removed_region_count": len(removed_regions),
            "hall_regular_cuboid": corrected_floors.loc[
                corrected_floors["floor_id"] == "Hall"
            ].iloc[0].to_dict(),
            "hall_envelope_padding_m": HALL_ENVELOPE_PADDING_M,
        },
        "backup_directory": str(BACKUP_DIR.resolve()),
    }
    (DATA_DIR / "exit_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
