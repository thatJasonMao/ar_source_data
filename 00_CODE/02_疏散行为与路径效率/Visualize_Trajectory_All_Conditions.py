"""Generate Fig. 4/Fig. 6-inspired trajectory figures for all 10 groups.

The visual grammar follows Gath-Morad et al. (Scientific Reports, 2024):
shared-scale path-density fields and reference paths coloured by within-path
progress. The former representative-cluster figure is replaced by globally
verified A* network shortest paths from each common start to its assigned exit.
All geometry is projected with the exact Unity orthographic camera defined by
``Visualize_Unity_Spatial_Partitions.py``.

The ten groups are the complete 5-guidance x 2-smoke experimental design.
All source data are read-only. Final 600 dpi PNG files are written only to
``Results_Trajectory_All_10_Groups`` beside this program.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

from Visualize_Unity_Spatial_Partitions import (
    DEFAULT_CAMERA,
    UnityOrthographicProjector,
    build_line_segments,
    load_zones,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
RESULT_DIR = HERE / "Results_Trajectory_All_10_Groups"
OUTPUT_DPI = 600
TRIALS_FILE = (
    ROOT
    / "Results_Guidance_Spatial_Causal_Analysis"
    / "data"
    / "analysis_trials.csv"
)
SPATIAL_DIR = ROOT / "SpatialSceneExport"

GUIDANCES = ("StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full")
SMOKE_LEVELS = ("LowSmoke", "HighSmoke")
SCENARIOS = tuple(
    f"{guidance}-{smoke}"
    for smoke in SMOKE_LEVELS
    for guidance in GUIDANCES
)
GUIDANCE_LABELS = {
    "StaticSign": "Static signage",
    "ArrowRibbon": "AR light carpet",
    "MiniMap": "Mini-map",
    "TextBar": "Text bar",
    "Full": "Full guidance",
}
SMOKE_LABELS = {"LowSmoke": "Low smoke", "HighSmoke": "High smoke"}
# 2.5D exploded display parameters adopted from Trajectory_2p5D_AllInOne.py.
# They alter presentation geometry only; raw trajectories remain unchanged.
PLATFORM_DISPLAY_Y = -36.0
HALL_DISPLAY_Y = 18.0
# Keep the stair and hall fixed while moving only the platform upward.
STAIR_DISPLAY_Y = -8.0
PLATFORM_BLOCK_HEIGHT = 2.5
STAIR_BLOCK_HEIGHT = 7.0
HALL_BLOCK_HEIGHT = 2.5
PLATFORM_TO_STAIR_Y = -15.20
STAIR_TO_HALL_Y = -12.35
ASTAR_ROUTE_CMAP_NAME = "RdYlBu_r"
ASTAR_ROUTE_LINEWIDTH = 1.05
ASTAR_ROUTE_HALO_LINEWIDTH = 1.65
ASTAR_ROUTE_SMOOTH_SIGMA = 3.25


@dataclass
class TrackRecord:
    scenario: str
    points: np.ndarray


@dataclass(frozen=True)
class AStarRoute:
    scenario: str
    target_exit: str
    start_snap_distance_m: float
    exit_snap_distance_m: float
    graph_length_m: float
    points: np.ndarray


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def scene_bounds(zones: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
    bounds = np.asarray([geometry.bounds for geometry in zones["geometry"]], dtype=float)
    return (
        float(bounds[:, 0].min() - 5.0),
        float(bounds[:, 2].max() + 5.0),
        float(zones["y_min"].min() - 2.0),
        float(zones["y_max"].max() + 2.0),
        float(bounds[:, 1].min() - 5.0),
        float(bounds[:, 3].max() + 5.0),
    )


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_path(points: np.ndarray, point_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 1:
        return np.repeat(points, point_count, axis=0)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-8])
    cumulative = cumulative[keep]
    points = points[keep]
    if cumulative[-1] <= 1e-8:
        return np.repeat(points[:1], point_count, axis=0)
    query = np.linspace(0.0, cumulative[-1], point_count)
    return np.column_stack(
        [np.interp(query, cumulative, points[:, axis]) for axis in range(3)]
    )


def load_tracks(zones: pd.DataFrame) -> list[TrackRecord]:
    trials = pd.read_csv(TRIALS_FILE, low_memory=False)
    trials = trials.loc[
        trials["Scenario"].isin(SCENARIOS)
        & trials["AnalysisIncluded"].fillna(False).astype(bool)
    ].copy()
    xmin, xmax, ymin, ymax, zmin, zmax = scene_bounds(zones)
    records: list[TrackRecord] = []
    for row in trials.itertuples():
        path = RAW / str(row.Scenario) / str(row.SubjectFolder) / str(row.FileName)
        raw = pd.read_csv(
            path,
            usecols=["TimeStamp", "PosX", "PosY", "PosZ"],
            low_memory=False,
        )
        time = raw["TimeStamp"].astype(str)
        part = raw.loc[
            time.ge(str(row.StartTime)) & time.le(str(row.EndTime)),
            ["PosX", "PosY", "PosZ"],
        ].apply(pd.to_numeric, errors="coerce")
        part = part.dropna()
        points = part.to_numpy(float)
        valid = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
            & (np.abs(points).sum(axis=1) > 0)
        )
        points = points[valid]
        if len(points) < 3:
            continue
        motion = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5])
        points = points[motion]
        if len(points) < 3:
            continue
        records.append(
            TrackRecord(
                scenario=str(row.Scenario),
                points=points,
            )
        )
    return records


def load_target_exits() -> dict[str, str]:
    trials = pd.read_csv(TRIALS_FILE, low_memory=False)
    trials = trials.loc[
        trials["Scenario"].isin(SCENARIOS)
        & trials["AnalysisIncluded"].fillna(False).astype(bool),
        ["Scenario", "TargetExit"],
    ].copy()
    grouped = trials.groupby("Scenario", observed=True)["TargetExit"].agg(
        lambda values: sorted(set(values.astype(str)))
    )
    invalid = grouped.loc[grouped.map(len).ne(1)]
    if not invalid.empty:
        raise ValueError(f"Each scenario must have one target exit: {invalid.to_dict()}")
    targets = {str(scenario): exits[0] for scenario, exits in grouped.items()}
    missing = sorted(set(SCENARIOS) - set(targets))
    if missing:
        raise ValueError(f"Missing target exits for scenarios: {missing}")
    return targets


def load_astar_graph():
    nodes = pd.read_csv(SPATIAL_DIR / "astar_nodes.csv")
    edges = pd.read_csv(SPATIAL_DIR / "astar_edges.csv")
    exits = pd.read_csv(SPATIAL_DIR / "exits.csv")
    node_ids = nodes["node_id"].astype(int).to_numpy()
    coordinates = nodes[["x", "y", "z"]].to_numpy(float)
    coordinate_by_node = {
        int(node_id): coordinates[index]
        for index, node_id in enumerate(node_ids)
    }
    graph = nx.Graph()
    graph.add_nodes_from(node_ids.tolist())
    valid_ids = set(node_ids.tolist())
    for row in edges.itertuples(index=False):
        source = int(row.from_node)
        target = int(row.to_node)
        if source not in valid_ids or target not in valid_ids:
            continue
        length = float(row.length_m)
        if graph.has_edge(source, target):
            graph[source][target]["weight"] = min(
                float(graph[source][target]["weight"]), length
            )
        else:
            graph.add_edge(source, target, weight=length)
    exit_coordinates = {
        str(row.exit_id): np.asarray([row.x, row.y, row.z], dtype=float)
        for row in exits.itertuples(index=False)
    }
    return graph, node_ids, coordinates, coordinate_by_node, exit_coordinates


def build_astar_routes(records: list[TrackRecord]) -> dict[str, AStarRoute]:
    targets = load_target_exits()
    (
        graph,
        node_ids,
        coordinates,
        coordinate_by_node,
        exit_coordinates,
    ) = load_astar_graph()
    tree = cKDTree(coordinates)
    routes: dict[str, AStarRoute] = {}

    for scenario in SCENARIOS:
        group = [record for record in records if record.scenario == scenario]
        if not group:
            raise ValueError(f"No tracks available for {scenario}")
        starts = np.vstack([record.points[0] for record in group])
        common_start = np.median(starts, axis=0)
        if float(np.linalg.norm(starts - common_start, axis=1).max()) > 0.10:
            raise ValueError(f"Observed starts are not common within {scenario}")

        target_exit = targets[scenario]
        exact_exit = exit_coordinates[target_exit]
        start_snap_distance, start_index = tree.query(common_start)
        exit_snap_distance, exit_index = tree.query(exact_exit)
        start_node = int(node_ids[int(start_index)])
        exit_node = int(node_ids[int(exit_index)])

        def heuristic(node_a: int, node_b: int) -> float:
            return float(
                np.linalg.norm(
                    coordinate_by_node[int(node_a)]
                    - coordinate_by_node[int(node_b)]
                )
            )

        node_path = nx.astar_path(
            graph,
            start_node,
            exit_node,
            heuristic=heuristic,
            weight="weight",
        )
        graph_length = float(nx.path_weight(graph, node_path, weight="weight"))
        verified_shortest_length = float(
            nx.shortest_path_length(
                graph,
                start_node,
                exit_node,
                weight="weight",
            )
        )
        if not np.isclose(
            graph_length,
            verified_shortest_length,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"A* route is not globally shortest for {scenario}: "
                f"A*={graph_length}, Dijkstra={verified_shortest_length}"
            )

        graph_points = np.vstack(
            [coordinate_by_node[int(node)] for node in node_path]
        )
        full_points = np.vstack([common_start, graph_points, exact_exit])
        keep = np.r_[
            True,
            np.linalg.norm(np.diff(full_points, axis=0), axis=1) > 1e-8,
        ]
        routes[scenario] = AStarRoute(
            scenario=scenario,
            target_exit=target_exit,
            start_snap_distance_m=float(start_snap_distance),
            exit_snap_distance_m=float(exit_snap_distance),
            graph_length_m=graph_length,
            points=full_points[keep],
        )
    return routes


def truncated_viridis() -> ListedColormap:
    return ListedColormap(mpl.colormaps["viridis"](np.linspace(0.03, 0.76, 256)))


def panel_setup(ax, *, screen_y_shift_px: float = 0.0) -> None:
    ax.set_xlim(0.0, DEFAULT_CAMERA.width_px)
    # Keep the Unity screen scale unchanged while allowing panel-specific
    # vertical translation inside the fixed orthographic viewport.
    ax.set_ylim(
        DEFAULT_CAMERA.height_px - screen_y_shift_px,
        -screen_y_shift_px,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def add_skeleton(ax, skeleton_segments, *, alpha: float = 0.18, linewidth: float = 0.40) -> None:
    ax.add_collection(
        LineCollection(
            skeleton_segments,
            colors=[mpl.colors.to_rgba("#75808A", alpha)],
            linewidths=linewidth,
            capstyle="round",
            joinstyle="round",
            zorder=6,
        )
    )


def density_bounds(zones: pd.DataFrame, records: list[TrackRecord]):
    xmin, xmax, _, _, zmin, zmax = scene_bounds(zones)
    points = np.vstack([explode_points(record.points) for record in records])
    return (
        (min(xmin, float(points[:, 0].min())), max(xmax, float(points[:, 0].max()))),
        (
            PLATFORM_DISPLAY_Y - PLATFORM_BLOCK_HEIGHT / 2.0 - 2.0,
            HALL_DISPLAY_Y + HALL_BLOCK_HEIGHT / 2.0 + 2.0,
        ),
        (min(zmin, float(points[:, 2].min())), max(zmax, float(points[:, 2].max()))),
    )


def explode_points(points: np.ndarray) -> np.ndarray:
    """Map Unity trajectories onto separated 2.5D display layers."""

    exploded = np.asarray(points, dtype=float).copy()
    unity_y = exploded[:, 1]
    platform = unity_y <= PLATFORM_TO_STAIR_Y
    hall = unity_y >= STAIR_TO_HALL_Y
    stair = ~(platform | hall)
    exploded[platform, 1] = PLATFORM_DISPLAY_Y
    exploded[hall, 1] = HALL_DISPLAY_Y
    stair_fraction = np.clip(
        (unity_y[stair] - PLATFORM_TO_STAIR_Y)
        / (STAIR_TO_HALL_Y - PLATFORM_TO_STAIR_Y),
        0.0,
        1.0,
    )
    stair_bottom = STAIR_DISPLAY_Y - STAIR_BLOCK_HEIGHT / 2.0
    exploded[stair, 1] = stair_bottom + stair_fraction * STAIR_BLOCK_HEIGHT
    return exploded


def exploded_zone_rows(zones: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy of zone slabs at exploded layer heights."""

    display = zones.copy()
    settings = {
        "platform": (PLATFORM_DISPLAY_Y, PLATFORM_BLOCK_HEIGHT),
        "stair": (STAIR_DISPLAY_Y, STAIR_BLOCK_HEIGHT),
        "hall": (HALL_DISPLAY_Y, HALL_BLOCK_HEIGHT),
    }
    for layer, (center, height) in settings.items():
        mask = display["asset_layer"].eq(layer)
        display.loc[mask, "y_min"] = center - height / 2.0
        display.loc[mask, "y_max"] = center + height / 2.0
    return display


def three_dimensional_density(
    records: list[TrackRecord], bounds, bins=(72, 56, 96)
):
    equal_weight_points = np.vstack(
        [explode_points(resample_path(record.points, 360)) for record in records]
    )
    density, edges = np.histogramdd(equal_weight_points, bins=bins, range=bounds)
    density /= max(len(records), 1)
    # A narrow, physical-unit bandwidth retains the trajectory-centred ridges.
    bin_widths = np.asarray(
        [(bound[1] - bound[0]) / count for bound, count in zip(bounds, bins)],
        dtype=float,
    )
    bandwidth_world = np.asarray([1.65, 1.45, 1.65], dtype=float)
    density = gaussian_filter(
        density,
        sigma=bandwidth_world / bin_widths,
        mode="constant",
    )
    return density, edges


def project_density_field(
    density: np.ndarray,
    edges,
    projector: UnityOrthographicProjector,
    raster_shape=(540, 960),
):
    centers = [(edge[:-1] + edge[1:]) * 0.5 for edge in edges]
    xx, yy, zz = np.meshgrid(*centers, indexing="ij")
    world = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = density.ravel()
    screen, _ = projector.project(world)
    height, width = raster_shape
    px = np.floor(screen[:, 0] / DEFAULT_CAMERA.width_px * width).astype(int)
    py = np.floor(screen[:, 1] / DEFAULT_CAMERA.height_px * height).astype(int)
    valid = (
        (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
        & np.isfinite(values)
        & (values > 0)
    )
    raster = np.zeros((height, width), dtype=float)
    # Maximum-intensity projection preserves narrow 3D KDE ridges. Ray-summing
    # the entire volume makes overlapping low-density voxels look like broad fog.
    np.maximum.at(raster, (py[valid], px[valid]), values[valid])
    return gaussian_filter(raster, sigma=2.0, mode="constant"), world, screen, values


def add_density_coloured_path(
    ax,
    display_path: np.ndarray,
    density: np.ndarray,
    edges,
    projector: UnityOrthographicProjector,
    cmap,
    common_field_max: float,
) -> None:
    """Colour a path centreline by its trilinearly sampled 3D KDE value."""

    centers = [(edge[:-1] + edge[1:]) * 0.5 for edge in edges]
    coordinates = np.vstack(
        [
            np.interp(
                display_path[:, axis],
                center,
                np.arange(len(center), dtype=float),
            )
            for axis, center in enumerate(centers)
        ]
    )
    local_density = map_coordinates(
        density,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    screen, _ = projector.project(display_path)
    segments = np.stack([screen[:-1], screen[1:]], axis=1)
    segment_density = 0.5 * (local_density[:-1] + local_density[1:])
    normalised = np.clip(segment_density / max(common_field_max, 1e-12), 0.0, 1.0)
    colours = cmap(normalised)
    colours[:, 3] = np.clip(0.055 + 0.86 * np.power(normalised, 0.62), 0.0, 0.94)
    ax.add_collection(
        LineCollection(
            segments,
            colors=colours,
            linewidths=0.52,
            capstyle="round",
            joinstyle="round",
            zorder=5,
        )
    )


def save_figure(fig, stem: str, dpi: int) -> dict[str, Path]:
    if dpi != OUTPUT_DPI:
        raise ValueError(f"This script exports PNG only at {OUTPUT_DPI} dpi, got {dpi}.")
    path = RESULT_DIR / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches=None, facecolor="white", edgecolor="none")
    plt.close(fig)
    return {"png": path}


def plot_density_comparison(
    zones: pd.DataFrame,
    records: list[TrackRecord],
    projector: UnityOrthographicProjector,
    dpi: int,
):
    bounds = density_bounds(zones, records)
    fields = {}
    projected = {}
    for scenario in SCENARIOS:
        group = [record for record in records if record.scenario == scenario]
        density, edges = three_dimensional_density(group, bounds)
        raster, _, _, _ = project_density_field(density, edges, projector)
        fields[scenario] = (density, edges)
        projected[scenario] = raster

    common_max = max(float(projected[scenario].max()) for scenario in SCENARIOS)
    common_field_max = max(float(fields[scenario][0].max()) for scenario in SCENARIOS)
    cmap = truncated_viridis()
    skeleton = build_line_segments(exploded_zone_rows(zones), projector, "wireframe")
    fig, axes = plt.subplots(2, 5, figsize=(16.0, 7.6), facecolor="white")
    fig.subplots_adjust(
        left=0.055,
        right=0.992,
        top=0.875,
        bottom=0.155,
        wspace=0.018,
        hspace=0.055,
    )
    for ax, scenario in zip(axes.flat, SCENARIOS):
        panel_setup(ax, screen_y_shift_px=200.0)
        raster = projected[scenario]
        normalised = raster / max(common_max, 1e-12)
        rgba = cmap(np.clip(normalised, 0.0, 1.0))
        rgba[..., 3] = np.where(
            normalised >= 0.012,
            np.clip(0.015 + 0.34 * np.power(normalised, 0.72), 0.0, 0.38),
            0.0,
        )
        ax.imshow(
            rgba,
            extent=[0, DEFAULT_CAMERA.width_px, DEFAULT_CAMERA.height_px, 0],
            interpolation="bilinear",
            zorder=2,
        )
        # Sample the 3D KDE back onto each trajectory. The resulting coloured
        # centrelines retain the reference figure's linear path grammar.
        group = [record for record in records if record.scenario == scenario]
        density, edges = fields[scenario]
        for record in group:
            display_path = explode_points(resample_path(record.points, 360))
            add_density_coloured_path(
                ax,
                display_path,
                density,
                edges,
                projector,
                cmap,
                common_field_max,
            )
        contour_levels = [0.06, 0.14, 0.30, 0.58]
        contour_x = np.linspace(0.0, DEFAULT_CAMERA.width_px, normalised.shape[1])
        contour_y = np.linspace(0.0, DEFAULT_CAMERA.height_px, normalised.shape[0])
        ax.contour(
            contour_x,
            contour_y,
            normalised,
            levels=contour_levels,
            colors=[cmap(level) for level in contour_levels],
            linewidths=[0.24, 0.30, 0.38, 0.50],
            alpha=0.32,
            zorder=4,
        )
        add_skeleton(ax, skeleton, alpha=0.18, linewidth=0.40)
        group_n = sum(record.scenario == scenario for record in records)
        guidance = scenario.rsplit("-", 1)[0]
        ax.set_title(f"{GUIDANCE_LABELS[guidance]}  (n={group_n})", pad=4)

    for row_index, smoke in enumerate(SMOKE_LEVELS):
        fig.text(
            0.018,
            0.695 if row_index == 0 else 0.355,
            SMOKE_LABELS[smoke],
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.2,
            fontweight="semibold",
            color="#30363D",
        )

    fig.suptitle(
        "Projected 3D evacuation-path density across all 10 experimental groups",
        x=0.5,
        y=0.965,
        fontsize=11.0,
        fontweight="semibold",
    )
    cax = fig.add_axes([0.375, 0.078, 0.25, 0.022])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["Low", "Medium", "High"])
    colorbar.set_label("Participant-normalised path density · shared scale", labelpad=3)
    colorbar.ax.xaxis.set_label_position("top")
    fig.text(
        0.5,
        0.010,
        "Exploded layer centres: Platform=-8, Stair=1, Hall=18. Each participant contributes 360 equidistant samples; density uses one shared scale across all groups.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#535B64",
    )
    return save_figure(fig, "01_all_10_groups_path_density_unity_600dpi", dpi)


def add_astar_progress_path(ax, route: AStarRoute, projector, cmap) -> None:
    sampled = resample_path(route.points, 420)
    display = explode_points(sampled)
    screen, _ = projector.project(display)
    endpoints = screen[[0, -1]].copy()
    screen = np.column_stack(
        [
            gaussian_filter1d(
                screen[:, axis],
                sigma=ASTAR_ROUTE_SMOOTH_SIGMA,
                mode="nearest",
            )
            for axis in range(2)
        ]
    )
    screen[0] = endpoints[0]
    screen[-1] = endpoints[1]
    segments = np.stack([screen[:-1], screen[1:]], axis=1)
    ax.add_collection(
        LineCollection(
            segments,
            colors=[mpl.colors.to_rgba("#36404B", 0.48)],
            linewidths=ASTAR_ROUTE_HALO_LINEWIDTH,
            capstyle="round",
            joinstyle="round",
            zorder=8,
        )
    )
    progress_line = LineCollection(
        segments,
        cmap=cmap,
        norm=Normalize(0, 1),
        linewidths=ASTAR_ROUTE_LINEWIDTH,
        alpha=0.98,
        capstyle="round",
        joinstyle="round",
        zorder=9,
    )
    progress_line.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(progress_line)
    ax.scatter(
        screen[0, 0],
        screen[0, 1],
        s=6,
        c=[cmap(0.0)],
        edgecolors="#F7F9FB",
        linewidths=0.30,
        zorder=10,
    )
    ax.scatter(
        screen[-1, 0],
        screen[-1, 1],
        s=8,
        marker="s",
        c=[cmap(1.0)],
        edgecolors="#F7F9FB",
        linewidths=0.30,
        zorder=10,
    )


def plot_astar_comparison(
    zones: pd.DataFrame,
    records: list[TrackRecord],
    projector: UnityOrthographicProjector,
    dpi: int,
):
    routes = build_astar_routes(records)
    skeleton = build_line_segments(exploded_zone_rows(zones), projector, "wireframe")
    cmap = mpl.colormaps[ASTAR_ROUTE_CMAP_NAME]
    fig, axes = plt.subplots(2, 5, figsize=(16.0, 7.6), facecolor="white")
    fig.subplots_adjust(
        left=0.055,
        right=0.992,
        top=0.875,
        bottom=0.155,
        wspace=0.018,
        hspace=0.055,
    )
    for ax, scenario in zip(axes.flat, SCENARIOS):
        panel_setup(ax, screen_y_shift_px=200.0)
        add_skeleton(ax, skeleton, alpha=0.14, linewidth=0.36)
        group = [record for record in records if record.scenario == scenario]
        for record in group:
            display = explode_points(resample_path(record.points, 220))
            screen, _ = projector.project(display)
            ax.plot(
                screen[:, 0],
                screen[:, 1],
                color="#6F7782",
                linewidth=0.38,
                alpha=0.13,
                zorder=5,
            )
        add_astar_progress_path(ax, routes[scenario], projector, cmap)
        guidance = scenario.rsplit("-", 1)[0]
        ax.set_title(
            f"{GUIDANCE_LABELS[guidance]}  "
            f"(n={len(group)}, exit {routes[scenario].target_exit})",
            pad=4,
        )

    for row_index, smoke in enumerate(SMOKE_LEVELS):
        fig.text(
            0.018,
            0.695 if row_index == 0 else 0.355,
            SMOKE_LABELS[smoke],
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.2,
            fontweight="semibold",
            color="#30363D",
        )

    fig.suptitle(
        "A* shortest evacuation paths across all 10 experimental groups",
        x=0.5,
        y=0.965,
        fontsize=11.0,
        fontweight="semibold",
    )
    cax = fig.add_axes([0.375, 0.078, 0.25, 0.022])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["Start", "50%", "Target exit"])
    colorbar.set_label("A* shortest-path progress", labelpad=3)
    colorbar.ax.xaxis.set_label_position("top")
    fig.text(
        0.5,
        0.010,
        "Thin gray lines are observed trajectories. Blue-yellow-red lines are smoothed display traces of globally verified A* network shortest paths from the common observed start to each group's assigned exit.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#535B64",
    )
    return save_figure(
        fig,
        "02_all_10_groups_astar_shortest_paths_unity_600dpi",
        dpi,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Fig. 4/Fig. 6-inspired trajectory figures for all 10 groups."
    )
    parser.add_argument(
        "--only",
        choices=["all", "density", "astar"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    ensure_dirs()
    zones = load_zones()
    records = load_tracks(zones)
    actual_counts = {
        scenario: sum(record.scenario == scenario for record in records)
        for scenario in SCENARIOS
    }
    expected = {
        "StaticSign-LowSmoke": 45,
        "ArrowRibbon-LowSmoke": 15,
        "MiniMap-LowSmoke": 15,
        "TextBar-LowSmoke": 15,
        "Full-LowSmoke": 45,
        "StaticSign-HighSmoke": 45,
        "ArrowRibbon-HighSmoke": 15,
        "MiniMap-HighSmoke": 15,
        "TextBar-HighSmoke": 15,
        "Full-HighSmoke": 45,
    }
    if actual_counts != expected:
        raise RuntimeError(f"Unexpected valid-track counts: {actual_counts}; expected {expected}")

    projector = UnityOrthographicProjector(DEFAULT_CAMERA)
    generated: dict[str, dict[str, str]] = {}
    if args.only in {"all", "density"}:
        paths = plot_density_comparison(zones, records, projector, OUTPUT_DPI)
        generated["density"] = {key: str(path.relative_to(HERE)) for key, path in paths.items()}
    if args.only in {"all", "astar"}:
        paths = plot_astar_comparison(zones, records, projector, OUTPUT_DPI)
        generated["astar_shortest_paths"] = {
            key: str(path.relative_to(HERE)) for key, path in paths.items()
        }
    print(
        json.dumps(
            {
                "generated": generated,
                "track_counts": actual_counts,
                "dpi": OUTPUT_DPI,
                "font": "Times New Roman",
                "typical_subjects_or_clusters": False,
                "astar_shortest_paths_verified": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
