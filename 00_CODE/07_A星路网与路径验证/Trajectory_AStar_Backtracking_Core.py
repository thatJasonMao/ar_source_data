"""A* target-distance backtracking core used by formal trajectory figures.

The diagnostic plot keeps the Unity orthographic projection and exploded layout,
panel order, spatial skeleton, and 2 x 5 comparison grammar used by
``Visualize_Trajectory_All_Conditions.py``.  The continuous colour field is not
trajectory density: it is weighted by local increases in A* network distance
to each trial's assigned target exit.

Trajectories are sampled every 0.2 s and map-matched to the scene A* graph.
After local median smoothing, a contiguous step is backtracking when its
shortest-path distance to the assigned target increases by more than 0.25 m.
Only those observed backtracking segments contribute to the line KDE. Importing
this module performs no plotting; direct execution writes one diagnostic PNG.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, PowerNorm
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Visualize_Unity_Spatial_Partitions import (  # noqa: E402
    DEFAULT_CAMERA,
    UnityOrthographicProjector,
    build_line_segments,
    load_zones,
)


RESULT_DIR = ROOT / "Results_Trajectory_All_10_Groups"
OUTPUT_DPI = 600
TIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
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
EXPECTED_COUNTS = {
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

# Match the formal trajectory program's display-only exploded geometry.
PLATFORM_DISPLAY_Y = -8.0
STAIR_DISPLAY_Y = 1.0
HALL_DISPLAY_Y = 18.0
PLATFORM_BLOCK_HEIGHT = 2.5
STAIR_BLOCK_HEIGHT = 7.0
HALL_BLOCK_HEIGHT = 2.5
PLATFORM_TO_STAIR_Y = -15.20
STAIR_TO_HALL_Y = -12.35

# A* target-distance backtracking definition used by the time-resolved analysis.
TEMPORAL_SAMPLE_INTERVAL_S = 0.2
MAX_SNAP_DISTANCE_M = 3.0
MAX_CONTIGUOUS_GAP_S = 1.0
BACKTRACK_THRESHOLD_M = 0.25


@dataclass
class TrackRecord:
    scenario: str
    participant_id: str
    points: np.ndarray
    backtrack_increment_m: np.ndarray


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


def parse_time(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str), format=TIME_FORMAT, errors="coerce")


def temporal_subsample(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= 2:
        return frame.copy()
    elapsed = (
        frame["ParsedTime"] - frame["ParsedTime"].iloc[0]
    ).dt.total_seconds().to_numpy(float)
    bins = np.floor(elapsed / TEMPORAL_SAMPLE_INTERVAL_S)
    keep = np.r_[True, bins[1:] > bins[:-1]]
    return frame.loc[keep].copy().reset_index(drop=True)


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


def hall_mask(points: np.ndarray) -> np.ndarray:
    """Select raw Unity positions belonging to the station-hall layer."""

    return np.asarray(points, dtype=float)[:, 1] >= STAIR_TO_HALL_Y


def hall_runs(points: np.ndarray) -> list[np.ndarray]:
    """Split hall positions into contiguous runs without bridging other layers."""

    points = np.asarray(points, dtype=float)
    mask = hall_mask(points)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        points[start:stop]
        for start, stop in changes.reshape(-1, 2)
        if stop - start >= 2
    ]


def load_distance_fields() -> tuple[np.ndarray, cKDTree, dict[str, dict[int, float]]]:
    """Build target-specific shortest-path distance fields on the A* graph."""

    nodes = pd.read_csv(SPATIAL_DIR / "astar_nodes.csv")
    edges = pd.read_csv(SPATIAL_DIR / "astar_edges.csv")
    exits = pd.read_csv(SPATIAL_DIR / "exits.csv")
    node_ids = nodes["node_id"].astype(int).to_numpy()
    coordinates = nodes[["x", "y", "z"]].to_numpy(float)
    tree = cKDTree(coordinates)
    graph = nx.Graph()
    graph.add_nodes_from(node_ids.tolist())
    valid_ids = set(node_ids.tolist())
    for row in edges.itertuples(index=False):
        source = int(row.from_node)
        target = int(row.to_node)
        length = float(row.length_m)
        if source not in valid_ids or target not in valid_ids:
            continue
        if graph.has_edge(source, target):
            graph[source][target]["weight"] = min(
                graph[source][target]["weight"], length
            )
        else:
            graph.add_edge(source, target, weight=length)

    fields: dict[str, dict[int, float]] = {}
    target_exits = exits.loc[exits["exit_id"].astype(str).isin(list("ABCDEF"))]
    for row in target_exits.itertuples(index=False):
        _, nearest_index = tree.query(np.array([row.x, row.y, row.z], float))
        target_node = int(node_ids[int(nearest_index)])
        fields[str(row.exit_id)] = nx.single_source_dijkstra_path_length(
            graph, target_node, weight="weight"
        )
    return node_ids, tree, fields


def compute_a_star_backtracking(
    frame: pd.DataFrame,
    target_exit: str,
    node_ids: np.ndarray,
    tree: cKDTree,
    fields: dict[str, dict[int, float]],
) -> np.ndarray:
    """Return target-distance increases at the observed backtracking steps."""

    points = frame[["PosX", "PosY", "PosZ"]].to_numpy(float)
    snap_distance, snap_index = tree.query(points)
    matched = np.asarray(snap_distance <= MAX_SNAP_DISTANCE_M)
    snapped_nodes = node_ids[np.asarray(snap_index, int)]
    target_field = fields.get(str(target_exit), {})
    geodesic = np.asarray(
        [target_field.get(int(node), np.nan) for node in snapped_nodes], dtype=float
    )
    geodesic[~matched] = np.nan
    geodesic_smooth = (
        pd.Series(geodesic)
        .interpolate(limit=3)
        .rolling(5, center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )
    times = frame["ParsedTime"].to_numpy()
    dt = np.r_[
        np.nan,
        np.diff(times).astype("timedelta64[ms]").astype(float) / 1000.0,
    ]
    contiguous = (
        matched
        & np.r_[False, matched[:-1]]
        & (dt > 0)
        & (dt <= MAX_CONTIGUOUS_GAP_S)
    )
    geodesic_change = np.r_[np.nan, np.diff(geodesic_smooth)]
    return np.where(
        contiguous & (geodesic_change > BACKTRACK_THRESHOLD_M),
        geodesic_change,
        0.0,
    )


def load_tracks(zones: pd.DataFrame) -> list[TrackRecord]:
    trials = pd.read_csv(TRIALS_FILE, low_memory=False)
    trials = trials.loc[
        trials["Scenario"].isin(SCENARIOS)
        & trials["AnalysisIncluded"].fillna(False).astype(bool)
    ].copy()
    node_ids, tree, fields = load_distance_fields()
    xmin, xmax, ymin, ymax, zmin, zmax = scene_bounds(zones)
    records: list[TrackRecord] = []

    for row in trials.itertuples(index=False):
        path = RAW / str(row.Scenario) / str(row.SubjectFolder) / str(row.FileName)
        frame = pd.read_csv(
            path,
            usecols=["TimeStamp", "PosX", "PosY", "PosZ"],
            low_memory=False,
        )
        frame["ParsedTime"] = parse_time(frame["TimeStamp"])
        for column in ("PosX", "PosY", "PosZ"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(
            subset=["ParsedTime", "PosX", "PosY", "PosZ"]
        ).copy()
        start = pd.to_datetime(row.StartTime, format=TIME_FORMAT)
        end = pd.to_datetime(row.EndTime, format=TIME_FORMAT)
        frame = frame.loc[frame["ParsedTime"].between(start, end)].copy()
        frame = temporal_subsample(frame)
        if len(frame) < 3:
            continue

        backtracking = compute_a_star_backtracking(
            frame,
            str(row.TargetExit),
            node_ids,
            tree,
            fields,
        )

        points = frame[["PosX", "PosY", "PosZ"]].to_numpy(float)
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
        backtracking = backtracking[valid]
        if len(points) < 3:
            continue
        records.append(
            TrackRecord(
                scenario=str(row.Scenario),
                participant_id=str(row.ParticipantID),
                points=points,
                backtrack_increment_m=backtracking,
            )
        )
    return records


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


def explode_points(points: np.ndarray) -> np.ndarray:
    """Map raw Unity trajectories onto the formal program's display layers."""

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


def heat_bounds(zones: pd.DataFrame, records: list[TrackRecord]):
    xmin, xmax, _, _, zmin, zmax = scene_bounds(zones)
    points = np.vstack(
        [
            explode_points(record.points[hall_mask(record.points)])
            for record in records
            if np.any(hall_mask(record.points))
        ]
    )
    return (
        (min(xmin, float(points[:, 0].min())), max(xmax, float(points[:, 0].max()))),
        (
            HALL_DISPLAY_Y - HALL_BLOCK_HEIGHT / 2.0 - 2.0,
            HALL_DISPLAY_Y + HALL_BLOCK_HEIGHT / 2.0 + 2.0,
        ),
        (min(zmin, float(points[:, 2].min())), max(zmax, float(points[:, 2].max()))),
    )


def three_dimensional_backtracking_heat(
    records: list[TrackRecord],
    bounds,
    bins=(72, 56, 96),
):
    """Line KDE of observed steps that increase A* target distance.

    Each positive A* distance increment is distributed along its actual
    trajectory segment.  Thus the field retains the original figure's
    line-centred density grammar, while colour mass represents backtracking
    distance rather than general path occupancy.
    """

    backtrack_points: list[np.ndarray] = []
    backtrack_weights: list[np.ndarray] = []
    samples_per_step = 18
    for record in records:
        in_hall = hall_mask(record.points)
        for index in np.flatnonzero(record.backtrack_increment_m > 0):
            if index <= 0 or not (in_hall[index - 1] and in_hall[index]):
                continue
            segment = record.points[int(index) - 1 : int(index) + 1]
            if len(segment) < 2:
                continue
            display_segment = explode_points(
                resample_path(segment, samples_per_step)
            )
            increment = float(record.backtrack_increment_m[int(index)])
            backtrack_points.append(display_segment)
            backtrack_weights.append(
                np.full(samples_per_step, increment / samples_per_step)
            )
    if not backtrack_points:
        return np.zeros(bins, dtype=float), [
            np.linspace(low, high, count + 1)
            for (low, high), count in zip(bounds, bins)
        ]

    heat, edges = np.histogramdd(
        np.vstack(backtrack_points),
        bins=bins,
        range=bounds,
        weights=np.concatenate(backtrack_weights),
    )
    # Participant normalisation makes groups with n=15 and n=45 comparable.
    heat /= max(len(records), 1)
    bin_widths = np.asarray(
        [(bound[1] - bound[0]) / count for bound, count in zip(bounds, bins)],
        dtype=float,
    )
    bandwidth_world = np.asarray([1.65, 1.45, 1.65], dtype=float)
    heat = gaussian_filter(
        heat,
        sigma=bandwidth_world / bin_widths,
        mode="constant",
    )
    return heat, edges


def project_heat_field(
    heat: np.ndarray,
    edges,
    projector: UnityOrthographicProjector,
    raster_shape=(540, 960),
):
    centers = [(edge[:-1] + edge[1:]) * 0.5 for edge in edges]
    xx, yy, zz = np.meshgrid(*centers, indexing="ij")
    world = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = heat.ravel()
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
    np.maximum.at(raster, (py[valid], px[valid]), values[valid])
    return gaussian_filter(raster, sigma=2.0, mode="constant")


def backtracking_colormap() -> ListedColormap:
    return ListedColormap(mpl.colormaps["magma"](np.linspace(0.08, 1.0, 256)))


def panel_setup(ax, *, screen_y_shift_px: float = 0.0) -> None:
    ax.set_xlim(0.0, DEFAULT_CAMERA.width_px)
    ax.set_ylim(DEFAULT_CAMERA.height_px - screen_y_shift_px, -screen_y_shift_px)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def add_skeleton(
    ax,
    skeleton_segments,
    *,
    alpha: float = 0.18,
    linewidth: float = 0.40,
) -> None:
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


def add_path_context(ax, records: list[TrackRecord], projector) -> None:
    all_segments: list[np.ndarray] = []
    for record in records:
        for run in hall_runs(record.points):
            point_count = max(24, min(360, len(run) * 3))
            display_path = explode_points(resample_path(run, point_count))
            screen, _ = projector.project(display_path)
            all_segments.append(np.stack([screen[:-1], screen[1:]], axis=1))
    if all_segments:
        ax.add_collection(
            LineCollection(
                np.concatenate(all_segments),
                colors=[mpl.colors.to_rgba("#7B8087", 0.050)],
                linewidths=0.30,
                capstyle="round",
                joinstyle="round",
                zorder=3,
            )
        )


def add_heat_coloured_path(
    ax,
    display_path: np.ndarray,
    heat: np.ndarray,
    edges,
    projector,
    cmap,
    common_field_max: float,
) -> None:
    """Colour a whole trajectory centreline by the A* backtracking KDE."""

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
    local_heat = map_coordinates(
        heat,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    screen, _ = projector.project(display_path)
    segments = np.stack([screen[:-1], screen[1:]], axis=1)
    segment_heat = 0.5 * (local_heat[:-1] + local_heat[1:])
    normalised = np.clip(
        segment_heat / max(common_field_max, 1e-12), 0.0, 1.0
    )
    colour_position = np.sqrt(normalised)
    colours = cmap(colour_position)
    colours[:, 3] = np.where(
        normalised >= 0.0015,
        np.clip(0.035 + 0.90 * np.power(normalised, 0.36), 0.0, 0.95),
        0.0,
    )
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


def plot_backtracking_comparison(
    zones: pd.DataFrame,
    records: list[TrackRecord],
    projector: UnityOrthographicProjector,
) -> Path:
    bounds = heat_bounds(zones, records)
    fields = {}
    projected = {}
    for scenario in SCENARIOS:
        group = [record for record in records if record.scenario == scenario]
        heat, edges = three_dimensional_backtracking_heat(group, bounds)
        fields[scenario] = (heat, edges)
        projected[scenario] = project_heat_field(heat, edges, projector)

    common_max = max(float(projected[scenario].max()) for scenario in SCENARIOS)
    common_field_max = max(float(fields[scenario][0].max()) for scenario in SCENARIOS)
    if common_max <= 0 or common_field_max <= 0:
        raise RuntimeError("No positive backtracking increments were found.")

    cmap = backtracking_colormap()
    hall_zones = zones.loc[zones["asset_layer"].eq("hall")].copy()
    skeleton = build_line_segments(
        exploded_zone_rows(hall_zones), projector, "wireframe"
    )
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
        group = [record for record in records if record.scenario == scenario]
        raster = projected[scenario]
        linear_normalised = raster / common_max
        colour_position = np.sqrt(np.clip(linear_normalised, 0.0, 1.0))
        rgba = cmap(colour_position)
        rgba[..., 3] = np.where(
            linear_normalised >= 0.0015,
            np.clip(0.035 + 0.52 * np.power(linear_normalised, 0.30), 0.0, 0.64),
            0.0,
        )
        ax.imshow(
            rgba,
            extent=[0, DEFAULT_CAMERA.width_px, DEFAULT_CAMERA.height_px, 0],
            interpolation="bilinear",
            zorder=2,
        )
        add_path_context(ax, group, projector)
        heat, edges = fields[scenario]
        for record in group:
            for run in hall_runs(record.points):
                point_count = max(24, min(360, len(run) * 3))
                display_path = explode_points(resample_path(run, point_count))
                add_heat_coloured_path(
                    ax,
                    display_path,
                    heat,
                    edges,
                    projector,
                    cmap,
                    common_field_max,
                )
        contour_levels = [0.02, 0.08, 0.24, 0.55]
        contour_x = np.linspace(0.0, DEFAULT_CAMERA.width_px, raster.shape[1])
        contour_y = np.linspace(0.0, DEFAULT_CAMERA.height_px, raster.shape[0])
        available_levels = [
            level for level in contour_levels if linear_normalised.max() > level
        ]
        if available_levels:
            ax.contour(
                contour_x,
                contour_y,
                linear_normalised,
                levels=available_levels,
                colors=[cmap(np.sqrt(level)) for level in available_levels],
                linewidths=np.linspace(0.26, 0.52, len(available_levels)),
                alpha=0.40,
                zorder=4,
            )
        add_skeleton(ax, skeleton, alpha=0.18, linewidth=0.40)
        guidance = scenario.rsplit("-", 1)[0]
        ax.set_title(f"{GUIDANCE_LABELS[guidance]}  (n={len(group)})", pad=4)

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
        "Projected 3D A* backtracking heat on the station-hall layer",
        x=0.5,
        y=0.965,
        fontsize=11.0,
        fontweight="semibold",
    )
    cax = fig.add_axes([0.375, 0.078, 0.25, 0.022])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=common_max),
            cmap=cmap,
        ),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_ticks([0.0, common_max * 0.25, common_max])
    colorbar.set_ticklabels(["Low", "Medium", "High"])
    colorbar.set_label(
        "Participant-normalised A* backtracking-distance heat · shared scale",
        labelpad=3,
    )
    colorbar.ax.xaxis.set_label_position("top")
    fig.text(
        0.5,
        0.010,
        "Station-hall layer only. Backtracking = >0.25 m increase in A* shortest-path distance to the assigned target between contiguous 0.2 s samples; only those hall segments enter the 3D line KDE.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#535B64",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "01_all_10_groups_hall_layer_a_star_backtracking_heat_unity_600dpi.png"
    fig.savefig(
        output,
        dpi=OUTPUT_DPI,
        bbox_inches=None,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    return output


def validate_records(records: list[TrackRecord]) -> dict:
    actual_counts = {
        scenario: sum(record.scenario == scenario for record in records)
        for scenario in SCENARIOS
    }
    if actual_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Unexpected valid-track counts: {actual_counts}; expected {EXPECTED_COUNTS}"
        )
    step_counts = {
        scenario: int(
            sum(
                np.count_nonzero(
                    (record.backtrack_increment_m > 0)
                    & hall_mask(record.points)
                    & np.r_[False, hall_mask(record.points)[:-1]]
                )
                for record in records
                if record.scenario == scenario
            )
        )
        for scenario in SCENARIOS
    }
    distance_totals = {
        scenario: float(
            sum(
                np.nansum(
                    np.where(
                        hall_mask(record.points)
                        & np.r_[False, hall_mask(record.points)[:-1]],
                        record.backtrack_increment_m,
                        0.0,
                    )
                )
                for record in records
                if record.scenario == scenario
            )
        )
        for scenario in SCENARIOS
    }
    if not any(distance_totals.values()):
        raise RuntimeError("No positive A* target-distance backtracking was detected.")
    return {
        "track_counts": actual_counts,
        "hall_layer_a_star_backtracking_step_counts": step_counts,
        "hall_layer_a_star_backtracking_distance_m": distance_totals,
        "definition": {
            "visualised_layer": "hall",
            "target_exit_used": True,
            "sample_interval_s": TEMPORAL_SAMPLE_INTERVAL_S,
            "maximum_a_star_snap_distance_m": MAX_SNAP_DISTANCE_M,
            "maximum_contiguous_gap_s": MAX_CONTIGUOUS_GAP_S,
            "backtracking_increment_threshold_m": BACKTRACK_THRESHOLD_M,
        },
    }


def main() -> None:
    configure_style()
    zones = load_zones()
    records = load_tracks(zones)
    audit = validate_records(records)
    projector = UnityOrthographicProjector(DEFAULT_CAMERA)
    output = plot_backtracking_comparison(zones, records, projector)
    print(
        json.dumps(
            {
                "generated": str(output.relative_to(ROOT)),
                **audit,
                "dpi": OUTPUT_DPI,
                "font": "Times New Roman",
                "typical_subjects_or_clusters": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
