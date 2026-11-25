#!/usr/bin/env python3
"""Visualize Unity spatial exports and the A* node-edge network."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "SpatialSceneExport"
OUTPUT_DIR = DATA_DIR / "visualization"

COLORS = {
    "hall": "#0072B2",
    "platform": "#E69F00",
    "connector": "#CC79A7",
    "edge": "#4D4D4D",
    "exit": "#D55E00",
    "stair": "#56B4E9",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_data() -> dict[str, pd.DataFrame]:
    filenames = {
        "astar_nodes": "astar_nodes.csv",
        "astar_edges": "astar_edges.csv",
        "exits": "exits.csv",
        "floors": "floors.csv",
        "spatial_zones": "spatial_zones.csv",
    }
    return {
        name: pd.read_csv(DATA_DIR / filename)
        for name, filename in filenames.items()
    }


def build_unique_edges(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    id_to_row = nodes.set_index("node_id")[["x", "y", "z"]]
    pairs = {
        tuple(sorted((int(row.from_node), int(row.to_node))))
        for row in edges.itertuples(index=False)
        if int(row.from_node) in id_to_row.index
        and int(row.to_node) in id_to_row.index
    }
    rows = []
    for start_id, end_id in sorted(pairs):
        start = id_to_row.loc[start_id]
        end = id_to_row.loc[end_id]
        rows.append(
            {
                "from_node": start_id,
                "to_node": end_id,
                "x1": start.x,
                "y1": start.y,
                "z1": start.z,
                "x2": end.x,
                "y2": end.y,
                "z2": end.z,
                "mean_y": (start.y + end.y) / 2.0,
            }
        )
    return pd.DataFrame(rows)


def classify_nodes(
    nodes: pd.DataFrame,
    floors: pd.DataFrame,
    zones: pd.DataFrame,
) -> pd.Series:
    hall_y = float(
        floors.loc[floors["floor_id"] == "Hall", "center_y"].iloc[0]
    )
    platform_y = float(
        floors.loc[
            floors["floor_id"] == "Platform", "center_y"
        ].iloc[0]
    )
    stair_zones = zones[zones["zone_type"] == "Stair"]
    connector = np.zeros(len(nodes), dtype=bool)
    for zone in stair_zones.itertuples(index=False):
        connector |= (
            (nodes["x"] >= zone.center_x - zone.size_x / 2)
            & (nodes["x"] <= zone.center_x + zone.size_x / 2)
            & (nodes["z"] >= zone.center_z - zone.size_z / 2)
            & (nodes["z"] <= zone.center_z + zone.size_z / 2)
        ).to_numpy()

    nearest_hall = (
        np.abs(nodes["y"].to_numpy() - hall_y)
        <= np.abs(nodes["y"].to_numpy() - platform_y)
    )
    labels = np.where(nearest_hall, "Hall", "Platform").astype(object)
    labels[connector] = "Stair connector"
    return pd.Series(labels, index=nodes.index, name="spatial_layer")


def add_xz_rectangle(
    axis: plt.Axes,
    row: object,
    *,
    edgecolor: str,
    facecolor: str = "none",
    alpha: float = 1.0,
    linewidth: float = 1.0,
    linestyle: str = "-",
    zorder: float = 1,
) -> None:
    axis.add_patch(
        Rectangle(
            (row.center_x - row.size_x / 2, row.center_z - row.size_z / 2),
            row.size_x,
            row.size_z,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    )


def top_down_panel(
    axis: plt.Axes,
    nodes: pd.DataFrame,
    unique_edges: pd.DataFrame,
    exits: pd.DataFrame,
    floors: pd.DataFrame,
    zones: pd.DataFrame,
    *,
    title: str,
    node_mask: np.ndarray | None = None,
    show_colorbar: bool = True,
    figure: plt.Figure | None = None,
) -> None:
    if node_mask is None:
        node_mask = np.ones(len(nodes), dtype=bool)
    selected_ids = set(nodes.loc[node_mask, "node_id"].astype(int))
    edge_mask = (
        unique_edges["from_node"].isin(selected_ids)
        & unique_edges["to_node"].isin(selected_ids)
    )
    selected_edges = unique_edges[edge_mask]

    for floor in floors.itertuples(index=False):
        color = (
            COLORS["hall"]
            if floor.floor_id == "Hall"
            else COLORS["platform"]
        )
        add_xz_rectangle(
            axis,
            floor,
            edgecolor=color,
            facecolor=color,
            alpha=0.055,
            linewidth=1.2,
            zorder=0,
        )

    for zone in zones[zones["zone_type"] == "Stair"].itertuples(index=False):
        add_xz_rectangle(
            axis,
            zone,
            edgecolor=COLORS["connector"],
            facecolor=COLORS["connector"],
            alpha=0.16,
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )

    if not selected_edges.empty:
        segments = np.stack(
            [
                selected_edges[["x1", "z1"]].to_numpy(float),
                selected_edges[["x2", "z2"]].to_numpy(float),
            ],
            axis=1,
        )
        line_collection = LineCollection(
            segments,
            colors=COLORS["edge"],
            linewidths=0.55,
            alpha=0.42,
            zorder=2,
        )
        axis.add_collection(line_collection)

    selected_nodes = nodes.loc[node_mask]
    scatter = axis.scatter(
        selected_nodes["x"],
        selected_nodes["z"],
        c=selected_nodes["y"],
        cmap="viridis",
        vmin=nodes["y"].min(),
        vmax=nodes["y"].max(),
        s=8,
        alpha=0.92,
        linewidths=0,
        zorder=3,
    )

    axis.scatter(
        exits["x"],
        exits["z"],
        marker="*",
        s=95,
        c=COLORS["exit"],
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    for exit_row in exits.itertuples(index=False):
        horizontal = 1.6 if exit_row.x < 0 else -1.6
        alignment = "left" if exit_row.x < 0 else "right"
        axis.text(
            exit_row.x + horizontal,
            exit_row.z,
            exit_row.exit_id,
            ha=alignment,
            va="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS["exit"],
            zorder=6,
        )

    axis.set_title(title)
    axis.set_xlabel("Unity X (m)")
    axis.set_ylabel("Unity Z (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.margins(0.025)
    axis.grid(True, color="#D9D9D9", linewidth=0.35, alpha=0.55)
    if show_colorbar and figure is not None:
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.015, fraction=0.03)
        colorbar.set_label("Node Y (m)")


def make_overview(
    nodes: pd.DataFrame,
    unique_edges: pd.DataFrame,
    exits: pd.DataFrame,
    floors: pd.DataFrame,
    zones: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1.25, 1]},
        constrained_layout=True,
    )
    top_down_panel(
        axes[0],
        nodes,
        unique_edges,
        exits,
        floors,
        zones,
        title="Top-down spatial structure and A* network",
        show_colorbar=True,
        figure=fig,
    )

    segments = np.stack(
        [
            unique_edges[["z1", "y1"]].to_numpy(float),
            unique_edges[["z2", "y2"]].to_numpy(float),
        ],
        axis=1,
    )
    axes[1].add_collection(
        LineCollection(
            segments,
            colors=COLORS["edge"],
            linewidths=0.55,
            alpha=0.36,
            zorder=1,
        )
    )
    layer_palette = {
        "Hall": COLORS["hall"],
        "Platform": COLORS["platform"],
        "Stair connector": COLORS["connector"],
    }
    for layer, group in nodes.groupby("spatial_layer"):
        axes[1].scatter(
            group["z"],
            group["y"],
            s=9,
            alpha=0.85,
            c=layer_palette[layer],
            label=f"{layer} nodes (n={len(group)})",
            linewidths=0,
            zorder=2,
        )
    for floor in floors.itertuples(index=False):
        color = (
            COLORS["hall"]
            if floor.floor_id == "Hall"
            else COLORS["platform"]
        )
        axes[1].add_patch(
            Rectangle(
                (
                    floor.center_z - floor.size_z / 2,
                    floor.center_y - floor.size_y / 2,
                ),
                floor.size_z,
                floor.size_y,
                edgecolor=color,
                facecolor=color,
                alpha=0.09,
                linewidth=1.1,
                zorder=0,
            )
        )
    axes[1].set_title("Longitudinal elevation view")
    axes[1].set_xlabel("Unity Z (m)")
    axes[1].set_ylabel("Unity Y elevation (m)")
    axes[1].grid(True, color="#D9D9D9", linewidth=0.35, alpha=0.55)
    axes[1].legend(loc="upper center", frameon=False, ncol=1)
    axes[1].margins(0.025)
    fig.suptitle(
        "Unity spatial export: floor envelopes, exits, and A* graph",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"spatial_astar_overview.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def make_before_after_comparison(
    nodes: pd.DataFrame,
    unique_edges: pd.DataFrame,
    exits: pd.DataFrame,
    floors: pd.DataFrame,
    zones: pd.DataFrame,
) -> bool:
    """Compare the immutable raw export with the corrected exit-terminal graph."""
    backup_dir = DATA_DIR / "raw_before_exit_correction"
    removed_nodes_path = DATA_DIR / "exit_pruning_removed_nodes.csv"
    isolated_nodes_path = DATA_DIR / "astar_isolated_removed_nodes.csv"
    off_plane_nodes_path = DATA_DIR / "astar_off_plane_removed_nodes.csv"
    removed_regions_path = DATA_DIR / "removed_spatial_regions.csv"
    required = (
        backup_dir / "astar_nodes.csv",
        backup_dir / "astar_edges.csv",
        backup_dir / "exits.csv",
        backup_dir / "floors.csv",
        backup_dir / "spatial_zones.csv",
        removed_nodes_path,
        removed_regions_path,
    )
    if not all(path.exists() for path in required):
        return False

    raw_nodes = pd.read_csv(backup_dir / "astar_nodes.csv")
    raw_edges = pd.read_csv(backup_dir / "astar_edges.csv")
    raw_exits = pd.read_csv(backup_dir / "exits.csv")
    raw_floors = pd.read_csv(backup_dir / "floors.csv")
    raw_zones = pd.read_csv(backup_dir / "spatial_zones.csv")
    raw_unique_edges = build_unique_edges(raw_nodes, raw_edges)
    removed_nodes = pd.read_csv(removed_nodes_path)
    isolated_nodes = (
        pd.read_csv(isolated_nodes_path)
        if isolated_nodes_path.exists()
        else pd.DataFrame(columns=["x", "z"])
    )
    off_plane_nodes = (
        pd.read_csv(off_plane_nodes_path)
        if off_plane_nodes_path.exists()
        else pd.DataFrame(columns=["x", "z"])
    )
    removed_regions = pd.read_csv(removed_regions_path)

    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 9.3), constrained_layout=True, sharex=True, sharey=True
    )
    top_down_panel(
        axes[0],
        raw_nodes,
        raw_unique_edges,
        raw_exits,
        raw_floors,
        raw_zones,
        title=(
            "Before correction\n"
            f"{len(raw_nodes)} nodes; B/C labels were reversed"
        ),
        show_colorbar=False,
    )
    top_down_panel(
        axes[1],
        nodes,
        unique_edges,
        exits,
        floors,
        zones,
        title=(
            "After correction\n"
            f"{len(nodes)} nodes; walking planes and stairs retained"
        ),
        show_colorbar=False,
    )

    for region in removed_regions.itertuples(index=False):
        for axis in axes:
            axis.add_patch(
                Rectangle(
                    (region.min_x, region.min_z),
                    region.max_x - region.min_x,
                    region.max_z - region.min_z,
                    edgecolor="#B2182B",
                    facecolor="#EF8A62",
                    alpha=0.13,
                    linewidth=1.0,
                    linestyle="--",
                    zorder=3,
                )
            )
    axes[0].scatter(
        removed_nodes["x"],
        removed_nodes["z"],
        marker="x",
        s=24,
        linewidths=1.2,
        c="#B2182B",
        label=f"Removed tail node (n={len(removed_nodes)})",
        zorder=7,
    )
    if not isolated_nodes.empty:
        axes[0].scatter(
            isolated_nodes["x"],
            isolated_nodes["z"],
            marker="P",
            s=34,
            linewidths=0.7,
            c="#2166AC",
            edgecolors="white",
            zorder=8,
        )
    if not off_plane_nodes.empty:
        axes[0].scatter(
            off_plane_nodes["x"],
            off_plane_nodes["z"],
            marker="^",
            s=32,
            linewidths=0.6,
            c="#7B3294",
            edgecolors="white",
            zorder=8,
        )
    legend = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=COLORS["exit"],
            markeredgecolor="none",
            markersize=10,
            label="Exit",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#B2182B",
            linestyle="none",
            markersize=6,
            label="Removed A* tail node",
        ),
        Line2D(
            [0],
            [0],
            marker="P",
            color="none",
            markerfacecolor="#2166AC",
            markeredgecolor="white",
            markersize=7,
            label="Removed isolated node",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#7B3294",
            markeredgecolor="white",
            markersize=7,
            label="Removed off-plane node",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            edgecolor="#B2182B",
            facecolor="#EF8A62",
            alpha=0.20,
            linestyle="--",
            label="Removed spatial region",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.045),
    )
    fig.suptitle(
        "Exit-terminal correction of the Unity spatial structure and A* graph",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"spatial_astar_before_after.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return True


def make_by_floor(
    nodes: pd.DataFrame,
    unique_edges: pd.DataFrame,
    exits: pd.DataFrame,
    floors: pd.DataFrame,
    zones: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6.6), constrained_layout=True
    )
    hall_mask = nodes["spatial_layer"].isin(["Hall", "Stair connector"])
    platform_mask = nodes["spatial_layer"].isin(
        ["Platform", "Stair connector"]
    )
    top_down_panel(
        axes[0],
        nodes,
        unique_edges,
        exits,
        floors,
        zones,
        title=f"Hall layer (including connectors; n={hall_mask.sum()})",
        node_mask=hall_mask.to_numpy(),
        show_colorbar=False,
    )
    top_down_panel(
        axes[1],
        nodes,
        unique_edges,
        exits.iloc[0:0],
        floors,
        zones,
        title=(
            "Platform layer (including connectors; "
            f"n={platform_mask.sum()})"
        ),
        node_mask=platform_mask.to_numpy(),
        show_colorbar=False,
    )
    platform_floor = floors[floors["floor_id"] == "Platform"].iloc[0]
    axes[1].set_xlim(
        platform_floor["center_x"] - platform_floor["size_x"] / 2 - 4,
        platform_floor["center_x"] + platform_floor["size_x"] / 2 + 4,
    )
    axes[1].set_ylim(
        platform_floor["center_z"] - platform_floor["size_z"] / 2 - 4,
        platform_floor["center_z"] + platform_floor["size_z"] / 2 + 4,
    )
    legend = [
        Line2D(
            [0],
            [0],
            color=COLORS["edge"],
            linewidth=1,
            label="A* edge",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#21918C",
            markeredgecolor="none",
            markersize=5,
            label="A* node (color = Y elevation)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=COLORS["exit"],
            markeredgecolor="none",
            markersize=9,
            label="Exit",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["connector"],
            linestyle="--",
            linewidth=2,
            label="Stair connector envelope",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "A* graph separated by spatial layer",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"spatial_astar_by_floor.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def box_edges(row: object) -> list[np.ndarray]:
    x0, x1 = row.center_x - row.size_x / 2, row.center_x + row.size_x / 2
    y0, y1 = row.center_y - row.size_y / 2, row.center_y + row.size_y / 2
    z0, z1 = row.center_z - row.size_z / 2, row.center_z + row.size_z / 2
    corners = {
        "000": (x0, z0, y0),
        "001": (x0, z0, y1),
        "010": (x0, z1, y0),
        "011": (x0, z1, y1),
        "100": (x1, z0, y0),
        "101": (x1, z0, y1),
        "110": (x1, z1, y0),
        "111": (x1, z1, y1),
    }
    pairs = (
        ("000", "001"),
        ("000", "010"),
        ("000", "100"),
        ("111", "110"),
        ("111", "101"),
        ("111", "011"),
        ("001", "011"),
        ("001", "101"),
        ("010", "011"),
        ("010", "110"),
        ("100", "101"),
        ("100", "110"),
    )
    return [
        np.asarray([corners[start], corners[end]], dtype=float)
        for start, end in pairs
    ]


def make_3d_static(
    nodes: pd.DataFrame,
    unique_edges: pd.DataFrame,
    exits: pd.DataFrame,
    floors: pd.DataFrame,
    zones: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(12, 6.8))
    axis = fig.add_subplot(111, projection="3d")
    axis.set_position([0.01, 0.02, 0.83, 0.88])
    edge_segments = np.stack(
        [
            unique_edges[["x1", "z1", "y1"]].to_numpy(float),
            unique_edges[["x2", "z2", "y2"]].to_numpy(float),
        ],
        axis=1,
    )
    axis.add_collection3d(
        Line3DCollection(
            edge_segments,
            colors=COLORS["edge"],
            linewidths=0.45,
            alpha=0.30,
        )
    )
    scatter = axis.scatter(
        nodes["x"],
        nodes["z"],
        nodes["y"],
        c=nodes["y"],
        cmap="viridis",
        s=7,
        alpha=0.90,
        depthshade=False,
    )
    for floor in floors.itertuples(index=False):
        color = (
            COLORS["hall"]
            if floor.floor_id == "Hall"
            else COLORS["platform"]
        )
        axis.add_collection3d(
            Line3DCollection(
                box_edges(floor),
                colors=color,
                linewidths=1.1,
                alpha=0.8,
            )
        )
    for zone in zones[zones["zone_type"] == "Stair"].itertuples(index=False):
        axis.add_collection3d(
            Line3DCollection(
                box_edges(zone),
                colors=COLORS["connector"],
                linewidths=1.0,
                alpha=0.8,
                linestyles="--",
            )
        )
    axis.scatter(
        exits["x"],
        exits["z"],
        exits["y"],
        marker="*",
        s=90,
        c=COLORS["exit"],
        edgecolors="white",
        linewidths=0.5,
        depthshade=False,
    )
    for row in exits.itertuples(index=False):
        axis.text(
            row.x,
            row.z,
            row.y + 0.18,
            row.exit_id,
            color=COLORS["exit"],
            fontsize=8,
            fontweight="bold",
        )
    axis.set_xlabel("Unity X (m)")
    axis.set_ylabel("Unity Z (m)")
    axis.set_zlabel("Unity Y elevation (m)")
    axis.set_title(
        "3D A* network and exported spatial envelopes\n"
        "(vertical scale exaggerated for visibility)"
    )
    axis.view_init(elev=24, azim=-58)
    axis.set_box_aspect((1.0, 2.0, 0.42))
    colorbar = fig.colorbar(
        scatter, ax=axis, pad=0.015, shrink=0.72, fraction=0.035
    )
    colorbar.set_label("Unity Y elevation (m)")
    fig.savefig(
        OUTPUT_DIR / "spatial_astar_3d.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def add_plotly_box(
    figure: go.Figure,
    row: object,
    *,
    name: str,
    color: str,
    dash: str = "solid",
    showlegend: bool = True,
) -> None:
    segments = box_edges(row)
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for segment in segments:
        x.extend([segment[0, 0], segment[1, 0], None])
        y.extend([segment[0, 1], segment[1, 1], None])
        z.extend([segment[0, 2], segment[1, 2], None])
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            line={"color": color, "width": 4, "dash": dash},
            hoverinfo="skip",
        )
    )


def make_3d_interactive(
    nodes: pd.DataFrame,
    unique_edges: pd.DataFrame,
    exits: pd.DataFrame,
    floors: pd.DataFrame,
    zones: pd.DataFrame,
) -> None:
    edge_x: list[float | None] = []
    edge_z: list[float | None] = []
    edge_y: list[float | None] = []
    for row in unique_edges.itertuples(index=False):
        edge_x.extend([row.x1, row.x2, None])
        edge_z.extend([row.z1, row.z2, None])
        edge_y.extend([row.y1, row.y2, None])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_z,
            z=edge_y,
            mode="lines",
            name=f"A* edges (undirected n={len(unique_edges)})",
            line={"color": "rgba(70,70,70,0.42)", "width": 2},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=nodes["x"],
            y=nodes["z"],
            z=nodes["y"],
            mode="markers",
            name=f"A* nodes (n={len(nodes)})",
            marker={
                "size": 3,
                "color": nodes["y"],
                "colorscale": "Viridis",
                "cmin": float(nodes["y"].min()),
                "cmax": float(nodes["y"].max()),
                "colorbar": {"title": "Unity Y<br>(m)"},
                "opacity": 0.92,
            },
            customdata=np.column_stack(
                [
                    nodes["node_id"],
                    nodes["spatial_layer"],
                    nodes["x"],
                    nodes["y"],
                    nodes["z"],
                ]
            ),
            hovertemplate=(
                "Node %{customdata[0]}<br>"
                "Layer: %{customdata[1]}<br>"
                "X=%{customdata[2]:.2f}<br>"
                "Y=%{customdata[3]:.2f}<br>"
                "Z=%{customdata[4]:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=exits["x"],
            y=exits["z"],
            z=exits["y"],
            mode="markers+text",
            name="Exits",
            marker={
                "size": 7,
                "symbol": "diamond",
                "color": COLORS["exit"],
                "line": {"color": "white", "width": 1},
            },
            text=exits["exit_id"],
            textposition="top center",
            customdata=exits[["exit_id", "x", "y", "z"]].to_numpy(),
            hovertemplate=(
                "Exit %{customdata[0]}<br>"
                "X=%{customdata[1]:.2f}<br>"
                "Y=%{customdata[2]:.2f}<br>"
                "Z=%{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )
    for index, floor in enumerate(floors.itertuples(index=False)):
        color = (
            COLORS["hall"]
            if floor.floor_id == "Hall"
            else COLORS["platform"]
        )
        add_plotly_box(
            figure,
            floor,
            name=f"{floor.floor_id} envelope",
            color=color,
            showlegend=True,
        )
    for index, zone in enumerate(
        zones[zones["zone_type"] == "Stair"].itertuples(index=False)
    ):
        add_plotly_box(
            figure,
            zone,
            name="Stair envelopes",
            color=COLORS["connector"],
            dash="dash",
            showlegend=index == 0,
        )

    figure.update_layout(
        title=(
            "Interactive Unity spatial structure and A* network"
            "<br><sup>Drag to rotate; scroll to zoom; hover nodes for coordinates</sup>"
        ),
        template="plotly_white",
        legend={"orientation": "h", "y": -0.10, "x": 0.0},
        margin={"l": 0, "r": 0, "t": 75, "b": 55},
        scene={
            "xaxis_title": "Unity X (m)",
            "yaxis_title": "Unity Z (m)",
            "zaxis_title": "Unity Y elevation (m)",
            "aspectmode": "manual",
            "aspectratio": {"x": 0.9, "y": 1.9, "z": 0.42},
            "camera": {"eye": {"x": 1.35, "y": -1.55, "z": 0.72}},
        },
    )
    figure.write_html(
        OUTPUT_DIR / "spatial_astar_3d_interactive.html",
        include_plotlyjs=True,
        full_html=True,
    )


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    nodes = data["astar_nodes"].copy()
    edges = data["astar_edges"].copy()
    exits = data["exits"].copy()
    floors = data["floors"].copy()
    zones = data["spatial_zones"].copy()

    nodes["spatial_layer"] = classify_nodes(nodes, floors, zones)
    unique_edges = build_unique_edges(nodes, edges)

    comparison_created = make_before_after_comparison(
        nodes, unique_edges, exits, floors, zones
    )
    make_overview(nodes, unique_edges, exits, floors, zones)
    make_by_floor(nodes, unique_edges, exits, floors, zones)
    make_3d_static(nodes, unique_edges, exits, floors, zones)
    make_3d_interactive(nodes, unique_edges, exits, floors, zones)

    summary = {
        "source_directory": str(DATA_DIR.resolve()),
        "node_count": len(nodes),
        "directed_edge_count": len(edges),
        "displayed_unique_undirected_edge_count": len(unique_edges),
        "exit_count": len(exits),
        "floor_envelope_count": len(floors),
        "zone_count": len(zones),
        "node_elevation_range_m": [
            float(nodes["y"].min()),
            float(nodes["y"].max()),
        ],
        "node_layer_counts": {
            str(key): int(value)
            for key, value in nodes["spatial_layer"].value_counts().items()
        },
        "coordinate_convention": {
            "horizontal": "Unity X-Z",
            "vertical": "Unity Y",
        },
        "before_after_comparison_created": comparison_created,
        "caveat": (
            "Floor and zone geometry are exported axis-aligned envelopes, "
            "not full architectural meshes. A* edges are displayed once per "
            "node pair even when the CSV stores both directions."
        ),
    }
    (OUTPUT_DIR / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
