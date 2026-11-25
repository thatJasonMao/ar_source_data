"""Network route reconstruction used only by the route-efficiency raincloud plot.

Map 5 Hz trajectory coordinates to the nearest reachable 3D graph node within
3 m. Sum graph shortest distances in temporal order, preserving returns to
previously visited nodes. Both efficiency distances use the first and last
matched trajectory nodes, rather than an exit-centre anchor.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SAMPLE_INTERVAL_S = 0.20
MAP_THRESHOLD_M = 3.0
LONG_GAP_S = 2.0
DISTANCE_TOLERANCE_M = 1e-8


class NavigationNetwork:
    """The original undirected network; duplicate edges use minimum length."""

    def __init__(self, spatial_dir: Path):
        nodes = pd.read_csv(spatial_dir / "astar_nodes.csv")
        edges = pd.read_csv(spatial_dir / "astar_edges.csv")
        exits = pd.read_csv(spatial_dir / "exits.csv")
        ids = nodes["node_id"].to_numpy(int)
        coords = nodes[["x", "y", "z"]].to_numpy(float)
        if len(set(ids)) != len(ids) or not np.isfinite(coords).all():
            raise ValueError("Network node IDs must be unique and coordinates finite")
        self.coordinates = dict(zip(ids.tolist(), coords))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(ids.tolist())
        for edge in edges.itertuples(index=False):
            u, v, length = int(edge.from_node), int(edge.to_node), float(edge.length_m)
            if not np.isfinite(length) or length < 0:
                raise ValueError(f"Invalid edge length: {length}")
            if u in self.graph and v in self.graph:
                if self.graph.has_edge(u, v):
                    self.graph[u][v]["weight"] = min(
                        self.graph[u][v]["weight"], length
                    )
                else:
                    self.graph.add_edge(u, v, weight=length)
        tree = cKDTree(coords)
        self.exit_nodes = {}
        for row in exits.itertuples(index=False):
            _, index = tree.query([row.x, row.y, row.z])
            self.exit_nodes[str(row.exit_id)] = int(ids[int(index)])
        self._distance_fields: dict[int, dict[int, float]] = {}
        self._component_trees: dict[int, tuple[np.ndarray, cKDTree]] = {}

    def distance(self, start: int, end: int) -> float:
        if start == end:
            return 0.0
        if start not in self._distance_fields:
            self._distance_fields[start] = nx.single_source_dijkstra_path_length(
                self.graph, start, weight="weight"
            )
        try:
            return float(self._distance_fields[start][end])
        except KeyError as exc:
            raise ValueError(f"Unreachable network transition: {start} -> {end}") from exc

    def mapping_tree(self, spawn_node: int) -> tuple[np.ndarray, cKDTree]:
        if spawn_node not in self._component_trees:
            component = sorted(nx.node_connected_component(self.graph, spawn_node))
            ids = np.array(component, dtype=int)
            tree = cKDTree(np.array([self.coordinates[node] for node in component]))
            for node in component:
                self._component_trees[node] = (ids, tree)
        return self._component_trees[spawn_node]


def read_sampled_trajectory(path: Path, interval_s: float) -> tuple[pd.DataFrame, dict]:
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("Sampling interval must be positive and finite")
    raw = pd.read_csv(path, usecols=["TimeStamp", "PosX", "PosY", "PosZ"])
    raw["RawCSVRow"] = np.arange(len(raw)) + 2  # 1-based, including header.
    for column in ("PosX", "PosY", "PosZ"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["ParsedTime"] = pd.to_datetime(
        raw["TimeStamp"].astype(str), format="%Y_%m_%d_%H_%M_%S_%f", errors="coerce"
    )
    xyz = raw[["PosX", "PosY", "PosZ"]].to_numpy(float)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (np.abs(xyz).sum(axis=1) > 0.1)
        & raw["ParsedTime"].notna().to_numpy()
    )
    clean = raw.loc[valid].copy()
    if len(clean) < 2:
        raise ValueError(f"Fewer than two valid trajectory points: {path}")
    if (clean["ParsedTime"].diff().dropna().dt.total_seconds() < 0).any():
        raise ValueError(f"Trajectory timestamps decrease: {path}")
    elapsed = (clean["ParsedTime"] - clean["ParsedTime"].iloc[0]).dt.total_seconds()
    bins = np.floor(elapsed.to_numpy(float) / interval_s)
    keep = np.r_[True, bins[1:] > bins[:-1]]
    keep[-1] = True  # Use the final observation, not the start of its sampling bin.
    sampled = clean.loc[keep].reset_index(drop=True)
    return sampled, {
        "NetworkRawPointCount": len(raw),
        "NetworkInvalidOrInitializationPointCount": int((~valid).sum()),
        "NetworkSampledPointCount": len(sampled),
    }


def reconstruct_trial(
    path: Path,
    spawn_node: int,
    chosen_exit: str,
    network: NavigationNetwork,
    interval_s: float = SAMPLE_INTERVAL_S,
) -> tuple[dict, list[dict]]:
    sampled, counts = read_sampled_trajectory(path, interval_s)
    ids, tree = network.mapping_tree(spawn_node)
    xyz = sampled[["PosX", "PosY", "PosZ"]].to_numpy(float)
    snap, indices = tree.query(xyz)
    valid = snap <= MAP_THRESHOLD_M
    matched_indices = np.flatnonzero(valid)
    if len(matched_indices) < 2:
        raise ValueError(f"Fewer than two matched trajectory points: {path}")
    points = sampled.loc[valid].reset_index(drop=True)
    points_xyz = xyz[valid]
    point_nodes = ids[np.asarray(indices, dtype=int)[valid]]
    point_snap = np.asarray(snap)[valid]
    times = points["ParsedTime"].to_numpy()
    gaps = np.diff(times).astype("timedelta64[ns]").astype(float) / 1e9
    unmatched_between = np.diff(matched_indices) - 1

    # Consecutive duplicate nodes contribute zero. Later returns remain in order.
    rows = []
    for i, (u, v) in enumerate(zip(point_nodes[:-1], point_nodes[1:])):
        u, v = int(u), int(v)
        long_gap = bool(gaps[i] > LONG_GAP_S)
        inferred_gap = bool(long_gap or unmatched_between[i] > 0)
        if u == v and not inferred_gap:
            continue
        length = network.distance(u, v)
        rows.append({
            "FromNode": u,
            "ToNode": v,
            "FromRawCSVRow": int(points["RawCSVRow"].iloc[i]),
            "ToRawCSVRow": int(points["RawCSVRow"].iloc[i + 1]),
            "FromTimeStamp": str(points["TimeStamp"].iloc[i]),
            "ToTimeStamp": str(points["TimeStamp"].iloc[i + 1]),
            "TimeGap_s": float(gaps[i]),
            "IsLongGap": long_gap,
            "IsInferredGap": inferred_gap,
            "UnmatchedSamplesBetween": int(unmatched_between[i]),
            "IsDirectEdge": bool(network.graph.has_edge(u, v)),
            "NetworkDistance_m": length,
            "NodeStraightDistance_m": float(np.linalg.norm(
                network.coordinates[v] - network.coordinates[u]
            )),
            "CoordinateStep_m": float(np.linalg.norm(points_xyz[i + 1] - points_xyz[i])),
            "FromSnapDistance_m": float(point_snap[i]),
            "ToSnapDistance_m": float(point_snap[i + 1]),
        })

    start, end = int(point_nodes[0]), int(point_nodes[-1])
    observed = math.fsum(row["NetworkDistance_m"] for row in rows)
    shortest = network.distance(start, end)
    if not np.isfinite(observed) or observed <= 0:
        raise ValueError(f"Non-positive reconstructed network distance: {path}")
    if shortest > observed + DISTANCE_TOLERANCE_M:
        raise ValueError(f"Shortest distance exceeds reconstructed distance: {path}")
    efficiency = shortest / observed
    # Only normalize machine-precision overshoot, never a substantive ratio > 1.
    if 1 < efficiency <= 1 + 1e-12:
        efficiency = 1.0
    metrics = {
        **counts,
        "NetworkMatchedPointCount": len(points),
        "NetworkMapMatchFraction3m": float(valid.mean()),
        "NetworkStartNode": start,
        "NetworkEndNode": end,
        "NetworkStartMatchesSpawn": start == spawn_node,
        "NetworkShortestStartToEnd_m": shortest,
        "NetworkObservedPathDistance_m": observed,
        "NetworkRouteEfficiency": efficiency,
        "NetworkTransitionCount": int((point_nodes[1:] != point_nodes[:-1]).sum()),
        "NetworkIndirectTransitionCount": sum(
            row["FromNode"] != row["ToNode"] and not row["IsDirectEdge"] for row in rows
        ),
        "NetworkLongGapCount": int((gaps > LONG_GAP_S).sum()),
        "NetworkInferredGapCount": int(((gaps > LONG_GAP_S) | (unmatched_between > 0)).sum()),
        "NetworkGapBridgeDistance_m": math.fsum(
            row["NetworkDistance_m"] for row in rows if row["IsInferredGap"]
        ),
        "NetworkMaxTimeGap_s": float(gaps.max()),
        "NetworkMaxTransitionDistance_m": max(row["NetworkDistance_m"] for row in rows),
        "NetworkMedianSnapDistance_m": float(np.median(point_snap)),
        "NetworkMaxSnapDistance_m": float(point_snap.max()),
        "NetworkLeadingUnmatchedSampleCount": int(matched_indices[0]),
        "NetworkTrailingUnmatchedSampleCount": int(len(sampled) - 1 - matched_indices[-1]),
        "NetworkFinalRecordedToMatchedDistance_m": float(np.linalg.norm(xyz[-1] - points_xyz[-1])),
        "NetworkEndToChosenExit_m": network.distance(end, network.exit_nodes[chosen_exit]),
    }
    return metrics, rows


def add_network_route_metrics(
    data: pd.DataFrame,
    spatial_dir: Path,
    raw_dir: Path,
    interval_s: float = SAMPLE_INTERVAL_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    network = NavigationNetwork(spatial_dir)
    data = data.copy()
    spawn = pd.to_numeric(data["SpawnNode"], errors="raise")
    if not np.isfinite(spawn).all() or not (spawn % 1 == 0).all():
        raise ValueError("SpawnNode must contain finite integer node IDs")
    data["SpawnNode"] = spawn.astype(int)
    for column in ("TargetExit", "ChosenExit"):
        data[column] = data[column].astype(str).str.strip()
        unknown = sorted(set(data[column]) - set(network.exit_nodes))
        if unknown:
            raise ValueError(f"Unknown {column} labels: {unknown}")
    target_distances = np.array([
        network.distance(int(row.SpawnNode), network.exit_nodes[row.TargetExit])
        for row in data.itertuples(index=False)
    ])
    if not np.allclose(
        target_distances, data["ShortestStartToTarget_m"], rtol=0, atol=DISTANCE_TOLERANCE_M
    ):
        raise ValueError("Spawn-to-target distances disagree with original source distances")
    data["ShortestStartToChosenExit_m"] = [
        network.distance(int(row.SpawnNode), network.exit_nodes[row.ChosenExit])
        for row in data.itertuples(index=False)
    ]

    metrics, transitions = [], []
    for i, row in enumerate(data.itertuples(index=False), start=1):
        path = raw_dir / row.Scenario / row.ParticipantID / row.FileName
        trial_metrics, trial_rows = reconstruct_trial(
            path, int(row.SpawnNode), row.ChosenExit, network, interval_s=interval_s
        )
        metrics.append(trial_metrics)
        identity = {"Scenario": row.Scenario, "ParticipantID": row.ParticipantID, "FileName": row.FileName}
        transitions.extend({**identity, "TransitionIndex": j, **segment}
                           for j, segment in enumerate(trial_rows, start=1))
        if i % 50 == 0 or i == len(data):
            print(f"Network route reconstruction: {i}/{len(data)}")
    result = pd.concat([data.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    return result, pd.DataFrame(transitions)
