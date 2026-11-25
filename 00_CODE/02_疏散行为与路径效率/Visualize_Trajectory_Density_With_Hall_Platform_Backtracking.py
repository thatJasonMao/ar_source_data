"""Visualize all 10 groups with trajectory density and A* backtracking overlays.

The trajectory structure reproduces ``Visualize_Trajectory_All_Conditions.py``:
three exploded spatial layers, participant-balanced 3D path KDE, Unity camera,
and the original experimental comparison. The top two rows show trajectories;
the bottom two show hall/platform A*-target-distance backtracking KDE with the
same display heights and layer spacing. Source coordinates and A* calculations
remain unchanged. No representative-subject or clustering figure is made.

Each guidance group has its own trajectory gradient from ``a5_colormaps.py``.
The shared backtracking and A* reference colour maps use the same library.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a5_colormaps import get_cmap as get_a5_cmap  # noqa: E402
import Visualize_Trajectory_All_Conditions as density_base  # noqa: E402
import Trajectory_AStar_Backtracking_Core as backtrack  # noqa: E402
from Visualize_Unity_Spatial_Partitions import (  # noqa: E402
    DEFAULT_CAMERA,
    UnityOrthographicProjector,
    build_line_segments,
    iter_polygons,
    load_zones,
)


RESULT_DIR = ROOT / "Results_Trajectory_All_10_Groups"
OUTPUT_DPI = 600

# ---------------------------------------------------------------------------
# USER COLOUR CONTROLS
# Raw scenario keys remain unchanged for data compatibility. Display labels and
# trajectory gradients are defined here so every panel uses one naming scheme.
# ---------------------------------------------------------------------------
GUIDANCE_LABELS = {
    "StaticSign": "StaticSign",
    "ArrowRibbon": "Arrow",
    "MiniMap": "MiniMap",
    "TextBar": "TextBar",
    "Full": "Multimodal",
}
TRAJECTORY_PALETTE_SPECS = {
    "StaticSign": (29, False),
    "ArrowRibbon": (45, True),
    "MiniMap": (28, False),
    "TextBar": (46, True),
    "Full": (27, False),
}
LEGACY_03_TRAJECTORY_PALETTE_SPECS = {
    guidance: (62, False) for guidance in GUIDANCE_LABELS
}
# Independent output switches. Disable either version to skip its rendering.
RENDER_03 = True
RENDER_04 = True
BACKTRACKING_HEAT_COLORMAP_ID = 39
ASTAR_SHORTEST_PATH_COLORMAP_ID = 59

# Compact publication layout with larger, final-size-readable typography.
FIGURE_SIZE_INCHES = (16.0, 11.0)
PANEL_TITLE_SIZE = 17.0
ROW_LABEL_SIZE = 16.0
LEGEND_TITLE_SIZE = 14.6
LEGEND_TEXT_SIZE = 13.6
EXIT_LABEL_SIZE = 15.0
PANEL_HORIZONTAL_PADDING_PX = 18.0
PANEL_VERTICAL_PADDING_PX = 20.0
TRAJECTORY_LINE_ALPHA_MIN = 0.48
TRAJECTORY_LINE_ALPHA_MAX = 0.98

# Shared display heights keep the trajectory and backtracking layers aligned.
OVERLAY_HALL_DISPLAY_Y = float(density_base.HALL_DISPLAY_Y)
OVERLAY_HALL_HEIGHT = density_base.HALL_BLOCK_HEIGHT
OVERLAY_PLATFORM_DISPLAY_Y = float(density_base.PLATFORM_DISPLAY_Y)
OVERLAY_PLATFORM_HEIGHT = density_base.PLATFORM_BLOCK_HEIGHT
SHORTEST_PATH_LINEWIDTH = 0.80
SHORTEST_PATH_HALO_LINEWIDTH = 1
SHORTEST_PATH_SMOOTH_SIGMA = 3.25
BASE_SKELETON_LINEWIDTH = 0.62
OVERLAY_SKELETON_LINEWIDTH = 0.62
# Spatial-frame colours: original trajectory-density structure versus the
# duplicated hall/platform backtracking structures.
TRAJECTORY_DENSITY_FRAME_COLOR = "#333333"
BACKTRACKING_DENSITY_FRAME_COLOR = "#17365D"
TRAJECTORY_DENSITY_FRAME_ALPHA = 0.52
BACKTRACKING_DENSITY_FRAME_ALPHA = 0.42
VERTICAL_RIB_EXCLUDED_ZONE_TYPES = frozenset({"HallOutsideGate"})
HALL_INSIDE_GATE_EXIT_SIDE_X_TOLERANCE = 0.05


def build_figure_colormaps(trajectory_palette_specs):
    """Build group-specific trajectories and the two shared colour maps."""

    return (
        {
            guidance: get_a5_cmap(
                palette_id,
                continuous=True,
                reverse=reverse,
            )
            for guidance, (palette_id, reverse) in trajectory_palette_specs.items()
        },
        get_a5_cmap(BACKTRACKING_HEAT_COLORMAP_ID, continuous=True),
        get_a5_cmap(ASTAR_SHORTEST_PATH_COLORMAP_ID, continuous=True),
    )


def configure_style() -> None:
    """Apply the shared style, then enlarge text for the compact 4x5 layout."""

    density_base.configure_style()
    mpl.rcParams.update(
        {
            "font.size": 14.0,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelsize": LEGEND_TITLE_SIZE,
            "xtick.labelsize": LEGEND_TEXT_SIZE,
            "ytick.labelsize": LEGEND_TEXT_SIZE,
        }
    )


def add_trajectory_density_palette_legend(
    fig,
    density_cmaps,
    *,
    title_y: float = 0.145,
    bar_y: float = 0.116,
) -> None:
    """Show all five Low-to-High group gradients in one compact legend row."""

    left = 0.075
    right = 0.925
    gap = 0.014
    width = (right - left - gap * 4) / 5
    bar_height = 0.012
    gradient = np.linspace(0.0, 1.0, 256)[None, :]

    fig.text(
        (left + right) / 2,
        title_y,
        "Observed path density · within-group scale (Low → High)",
        ha="center",
        va="center",
        fontsize=LEGEND_TITLE_SIZE,
        fontweight="semibold",
        color="#30363D",
    )
    for index, guidance in enumerate(density_base.GUIDANCES):
        x = left + index * (width + gap)
        cax = fig.add_axes([x, bar_y, width, bar_height])
        cax.imshow(
            gradient,
            aspect="auto",
            cmap=density_cmaps[guidance],
            interpolation="bilinear",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        cax.set_xticks([])
        cax.set_yticks([])
        for spine in cax.spines.values():
            spine.set_color("#59616A")
            spine.set_linewidth(0.55)
        fig.text(
            x + width / 2,
            bar_y - 0.007,
            GUIDANCE_LABELS[guidance],
            ha="center",
            va="top",
            fontsize=LEGEND_TEXT_SIZE,
            color="#30363D",
        )


def add_single_trajectory_density_legend(
    fig,
    density_cmap,
    bounds=(0.075, 0.050, 0.245, 0.014),
) -> None:
    """Draw the legacy 03 trajectory scale as one of three aligned legends."""

    density_cax = fig.add_axes(bounds)
    density_colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=density_cmap),
        cax=density_cax,
        orientation="horizontal",
    )
    density_colorbar.set_ticks([0, 0.5, 1])
    density_colorbar.set_ticklabels(["Low", "Medium", "High"])
    density_colorbar.set_label(
        "Observed path density · within-group scale",
        labelpad=3,
    )
    density_colorbar.ax.xaxis.set_label_position("top")
    density_colorbar.ax.tick_params(labelsize=LEGEND_TEXT_SIZE, pad=2)


def add_backtracking_legend(fig, backtrack_cmap, bounds) -> None:
    """Add one horizontal hall/platform backtracking colour scale."""

    backtrack_cax = fig.add_axes(bounds)
    backtrack_colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
            cmap=backtrack_cmap,
        ),
        cax=backtrack_cax,
        orientation="horizontal",
    )
    backtrack_colorbar.set_ticks([0.0, 0.25, 1.0])
    backtrack_colorbar.set_ticklabels(["Low", "Medium", "High"])
    backtrack_colorbar.set_label(
        "Hall/platform backtracking heat · within-group scale",
        labelpad=3,
    )
    backtrack_colorbar.ax.xaxis.set_label_position("top")
    backtrack_colorbar.ax.tick_params(labelsize=LEGEND_TEXT_SIZE, pad=2)


def add_shortest_path_legend(fig, shortest_path_cmap, bounds) -> None:
    """Add one horizontal shortest-path progress colour scale."""

    shortest_path_cax = fig.add_axes(bounds)
    shortest_path_colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=Normalize(0, 1),
            cmap=shortest_path_cmap,
        ),
        cax=shortest_path_cax,
        orientation="horizontal",
    )
    shortest_path_colorbar.set_ticks([0, 0.5, 1])
    shortest_path_colorbar.set_ticklabels(["Start", "50%", "Target exit"])
    shortest_path_colorbar.set_label("Shortest-path progress", labelpad=3)
    shortest_path_colorbar.ax.xaxis.set_label_position("top")
    shortest_path_colorbar.ax.tick_params(labelsize=LEGEND_TEXT_SIZE, pad=2)


def add_high_opacity_density_coloured_path(*args, **kwargs) -> None:
    """Draw the shared density-coloured path with only its alpha strengthened."""

    ax = args[0]
    collection_count = len(ax.collections)
    density_base.add_density_coloured_path(*args, **kwargs)
    if len(ax.collections) != collection_count + 1:
        raise RuntimeError("Expected one trajectory LineCollection to be added")
    path_collection = ax.collections[-1]
    colours = path_collection.get_colors()
    original_position = np.clip((colours[:, 3] - 0.055) / 0.86, 0.0, 1.0)
    colours[:, 3] = (
        TRAJECTORY_LINE_ALPHA_MIN
        + (TRAJECTORY_LINE_ALPHA_MAX - TRAJECTORY_LINE_ALPHA_MIN)
        * original_position
    )
    path_collection.set_color(colours)


def projected_zone_vertical_envelope(rows, projector):
    """Return the filled slab silhouette's top/bottom screen y per x pixel."""

    faces: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        y_bottom = float(row.y_min)
        y_top = float(row.y_max)
        for polygon in iter_polygons(row.geometry):
            coordinates = np.asarray(polygon.exterior.coords, dtype=float)
            for unity_y in (y_bottom, y_top):
                world = np.column_stack(
                    [
                        coordinates[:, 0],
                        np.full(len(coordinates), unity_y),
                        coordinates[:, 1],
                    ]
                )
                screen, _ = projector.project(world)
                faces.append(screen)
            for start, end in zip(coordinates[:-1], coordinates[1:]):
                world = np.asarray(
                    [
                        [start[0], y_bottom, start[1]],
                        [end[0], y_bottom, end[1]],
                        [end[0], y_top, end[1]],
                        [start[0], y_top, start[1]],
                    ],
                    dtype=float,
                )
                screen, _ = projector.project(world)
                faces.append(screen)
    if not faces:
        raise ValueError("Cannot resolve the hall silhouette from empty zone rows")

    y_origin = int(np.floor(min(float(face[:, 1].min()) for face in faces))) - 2
    y_limit = int(np.ceil(max(float(face[:, 1].max()) for face in faces))) + 2
    width = int(DEFAULT_CAMERA.width_px)
    height = y_limit - y_origin + 1
    silhouette = Image.new("1", (width, height), 0)
    drawer = ImageDraw.Draw(silhouette)
    for face in faces:
        drawer.polygon(
            [(float(point[0]), float(point[1] - y_origin)) for point in face],
            fill=1,
        )

    mask = np.asarray(silhouette, dtype=bool)
    top = np.full(width, np.nan, dtype=float)
    bottom = np.full(width, np.nan, dtype=float)
    for x in np.flatnonzero(mask.any(axis=0)):
        occupied_y = np.flatnonzero(mask[:, x])
        top[x] = float(occupied_y[0] + y_origin)
        bottom[x] = float(occupied_y[-1] + y_origin)
    return top, bottom


def overlay_hall_zones(zones):
    display = zones.loc[zones["asset_layer"].eq("hall")].copy()
    display.loc[:, "y_min"] = OVERLAY_HALL_DISPLAY_Y - OVERLAY_HALL_HEIGHT / 2.0
    display.loc[:, "y_max"] = OVERLAY_HALL_DISPLAY_Y + OVERLAY_HALL_HEIGHT / 2.0
    return display


def overlay_hall_points(points: np.ndarray) -> np.ndarray:
    display = np.asarray(points, dtype=float).copy()
    display[:, 1] = OVERLAY_HALL_DISPLAY_Y
    return display


def overlay_platform_zones(zones):
    display = zones.loc[zones["asset_layer"].eq("platform")].copy()
    display.loc[:, "y_min"] = (
        OVERLAY_PLATFORM_DISPLAY_Y - OVERLAY_PLATFORM_HEIGHT / 2.0
    )
    display.loc[:, "y_max"] = (
        OVERLAY_PLATFORM_DISPLAY_Y + OVERLAY_PLATFORM_HEIGHT / 2.0
    )
    return display


def overlay_platform_points(points: np.ndarray) -> np.ndarray:
    display = np.asarray(points, dtype=float).copy()
    display[:, 1] = OVERLAY_PLATFORM_DISPLAY_Y
    return display


def overlay_heat_bounds(base_bounds):
    return (
        base_bounds[0],
        (
            OVERLAY_HALL_DISPLAY_Y - OVERLAY_HALL_HEIGHT / 2.0 - 2.0,
            OVERLAY_HALL_DISPLAY_Y + OVERLAY_HALL_HEIGHT / 2.0 + 2.0,
        ),
        base_bounds[2],
    )


def three_dimensional_layer_overlay_backtracking_heat(
    records,
    bounds,
    layer_mask,
    display_points,
    bins=(72, 56, 96),
):
    """KDE of one layer's A* backtracking segments at its duplicate Y."""

    segment_points: list[np.ndarray] = []
    segment_weights: list[np.ndarray] = []
    samples_per_step = 18
    for record in records:
        in_layer = layer_mask(record.points)
        for index in np.flatnonzero(record.backtrack_increment_m > 0):
            if index <= 0 or not (
                in_layer[index - 1] and in_layer[index]
            ):
                continue
            segment = record.points[int(index) - 1 : int(index) + 1]
            display_segment = display_points(
                density_base.resample_path(segment, samples_per_step)
            )
            increment = float(record.backtrack_increment_m[int(index)])
            segment_points.append(display_segment)
            segment_weights.append(
                np.full(samples_per_step, increment / samples_per_step)
            )

    if not segment_points:
        return np.zeros(bins, dtype=float), [
            np.linspace(low, high, count + 1)
            for (low, high), count in zip(bounds, bins)
        ]

    heat, edges = np.histogramdd(
        np.vstack(segment_points),
        bins=bins,
        range=bounds,
        weights=np.concatenate(segment_weights),
    )
    heat /= max(len(records), 1)
    bin_widths = np.asarray(
        [(bound[1] - bound[0]) / count for bound, count in zip(bounds, bins)],
        dtype=float,
    )
    heat = gaussian_filter(
        heat,
        sigma=np.asarray([1.65, 1.45, 1.65], dtype=float) / bin_widths,
        mode="constant",
    )
    return heat, edges


def three_dimensional_overlay_backtracking_heat(
    records,
    bounds,
    bins=(72, 56, 96),
):
    return three_dimensional_layer_overlay_backtracking_heat(
        records,
        bounds,
        backtrack.hall_mask,
        overlay_hall_points,
        bins=bins,
    )


def three_dimensional_platform_overlay_backtracking_heat(
    records,
    bounds,
    bins=(72, 56, 96),
):
    return three_dimensional_layer_overlay_backtracking_heat(
        records,
        bounds,
        platform_mask,
        overlay_platform_points,
        bins=bins,
    )


def add_layer_overlay_path_context(
    ax,
    records,
    projector,
    runs_function,
    display_points,
) -> None:
    segments: list[np.ndarray] = []
    for record in records:
        for run in runs_function(record.points):
            point_count = max(24, min(360, len(run) * 3))
            display = display_points(
                density_base.resample_path(run, point_count)
            )
            screen, _ = projector.project(display)
            segments.append(np.stack([screen[:-1], screen[1:]], axis=1))
    if segments:
        ax.add_collection(
            LineCollection(
                np.concatenate(segments),
                colors=[mpl.colors.to_rgba("#75636D", 0.050)],
                linewidths=0.30,
                capstyle="round",
                joinstyle="round",
                zorder=7,
            )
        )


def add_overlay_path_context(ax, records, projector) -> None:
    add_layer_overlay_path_context(
        ax,
        records,
        projector,
        backtrack.hall_runs,
        overlay_hall_points,
    )


def add_platform_overlay_path_context(ax, records, projector) -> None:
    add_layer_overlay_path_context(
        ax,
        records,
        projector,
        platform_runs,
        overlay_platform_points,
    )


def add_skeleton(
    ax,
    segments,
    *,
    color: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    ax.add_collection(
        LineCollection(
            segments,
            colors=[mpl.colors.to_rgba(color, alpha)],
            linewidths=linewidth,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )


def backtracking_stair_connections(zones, projector) -> np.ndarray:
    """Project hall-to-platform connectors at the two stair locations."""
    stairs = zones.loc[zones["asset_layer"].eq("stair")].sort_values("center_z")
    if len(stairs) != 2:
        raise RuntimeError(f"Expected two stair regions, found {len(stairs)}")
    segments = []
    for row in stairs.itertuples(index=False):
        world = np.asarray(
            [
                [float(row.center_x), OVERLAY_HALL_DISPLAY_Y, float(row.center_z)],
                [float(row.center_x), OVERLAY_PLATFORM_DISPLAY_Y, float(row.center_z)],
            ]
        )
        screen, _ = projector.project(world)
        segments.append(screen)
    return np.asarray(segments)


def add_shortest_path_reference(ax, route, projector, cmap) -> None:
    """Draw the approved thin A* progress line on the original structure."""

    sampled = density_base.resample_path(route.points, 420)
    display = density_base.explode_points(sampled)
    screen, _ = projector.project(display)
    unsmoothed_endpoints = screen[[0, -1]].copy()
    screen = np.column_stack(
        [
            gaussian_filter1d(
                screen[:, axis],
                sigma=SHORTEST_PATH_SMOOTH_SIGMA,
                mode="nearest",
            )
            for axis in range(2)
        ]
    )
    screen[0] = unsmoothed_endpoints[0]
    screen[-1] = unsmoothed_endpoints[1]
    segments = np.stack([screen[:-1], screen[1:]], axis=1)
    ax.add_collection(
        LineCollection(
            segments,
            colors=[mpl.colors.to_rgba("#36404B", 0.48)],
            linewidths=SHORTEST_PATH_HALO_LINEWIDTH,
            capstyle="round",
            joinstyle="round",
            zorder=10,
        )
    )
    progress_line = LineCollection(
        segments,
        cmap=cmap,
        norm=Normalize(0, 1),
        linewidths=SHORTEST_PATH_LINEWIDTH,
        alpha=0.98,
        capstyle="round",
        joinstyle="round",
        zorder=11,
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
        zorder=11.5,
    )
    ax.scatter(
        screen[-1, 0],
        screen[-1, 1],
        s=8,
        marker="s",
        c=[cmap(1.0)],
        edgecolors="#F7F9FB",
        linewidths=0.30,
        zorder=11.5,
    )


def load_target_exit_locations():
    """Resolve one assigned target and its scene coordinate per scenario."""

    trials = pd.read_csv(backtrack.TRIALS_FILE, low_memory=False)
    trials = trials.loc[
        trials["Scenario"].isin(density_base.SCENARIOS)
        & trials["AnalysisIncluded"].fillna(False).astype(bool),
        ["Scenario", "TargetExit"],
    ].copy()
    target_counts = trials.groupby("Scenario")["TargetExit"].nunique()
    if not target_counts.eq(1).all():
        raise RuntimeError(
            f"Each scenario must have one target exit: {target_counts.to_dict()}"
        )
    target_by_scenario = (
        trials.drop_duplicates("Scenario")
        .set_index("Scenario")["TargetExit"]
        .astype(str)
        .to_dict()
    )
    exits = pd.read_csv(backtrack.SPATIAL_DIR / "exits.csv")
    exits["exit_id"] = exits["exit_id"].astype(str)
    exits = exits.set_index("exit_id")
    missing = sorted(set(target_by_scenario.values()) - set(exits.index))
    if missing:
        raise RuntimeError(f"Target exits missing from exits.csv: {missing}")
    return target_by_scenario, exits


def add_target_exit_markers(
    ax,
    scenario: str,
    target_by_scenario: dict[str, str],
    exits: pd.DataFrame,
    projector,
) -> None:
    """Mark the assigned exit once on the hall in the current metric panel."""

    exit_id = target_by_scenario[scenario]
    row = exits.loc[exit_id]
    world = np.asarray(
        [
            [float(row.x), density_base.HALL_DISPLAY_Y, float(row.z)],
            [float(row.x), OVERLAY_HALL_DISPLAY_Y, float(row.z)],
        ],
        dtype=float,
    )
    screen, _ = projector.project(np.unique(world, axis=0))
    ax.scatter(
        screen[:, 0],
        screen[:, 1],
        s=48,
        marker="*",
        facecolors="#FFFFFF",
        edgecolors="#20262E",
        linewidths=0.75,
        zorder=12,
    )
    for x, y in screen:
        place_label_left = x > np.mean(ax.get_xlim())
        ax.annotate(
            exit_id,
            xy=(x, y),
            xytext=(-6.0 if place_label_left else 6.0, -3.0),
            textcoords="offset points",
            ha="right" if place_label_left else "left",
            va="center",
            fontsize=EXIT_LABEL_SIZE,
            fontweight="bold",
            color="#20262E",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            zorder=13,
        )


def build_selective_wireframe(rows, projector) -> np.ndarray:
    """Keep the full wireframe except vertical ribs in selected zone types."""

    bottom_rows = rows.copy()
    bottom_rows.loc[:, "y_max"] = bottom_rows["y_min"]
    segments = [
        *build_line_segments(rows, projector, "outline"),
        *build_line_segments(bottom_rows, projector, "outline"),
    ]
    for row in rows.itertuples(index=False):
        zone_type = str(row.zone_type)
        if zone_type in VERTICAL_RIB_EXCLUDED_ZONE_TYPES:
            continue
        exit_side_x = (
            float(row.geometry.bounds[2])
            if zone_type == "HallInsideGate"
            else None
        )
        y_bottom = float(row.y_min)
        y_top = float(row.y_max)
        for polygon in iter_polygons(row.geometry):
            for loop in [polygon.exterior, *polygon.interiors]:
                for x, z in np.asarray(loop.coords, dtype=float)[:-1]:
                    if (
                        exit_side_x is not None
                        and x
                        >= exit_side_x
                        - HALL_INSIDE_GATE_EXIT_SIDE_X_TOLERANCE
                    ):
                        continue
                    screen, _ = projector.project(
                        np.asarray(
                            [[x, y_bottom, z], [x, y_top, z]],
                            dtype=float,
                        )
                    )
                    segments.append(screen)
    return np.asarray(segments)


def overlay_platform_heat_bounds(base_bounds):
    return (
        base_bounds[0],
        (
            OVERLAY_PLATFORM_DISPLAY_Y
            - OVERLAY_PLATFORM_HEIGHT / 2.0
            - 2.0,
            OVERLAY_PLATFORM_DISPLAY_Y
            + OVERLAY_PLATFORM_HEIGHT / 2.0
            + 2.0,
        ),
        base_bounds[2],
    )


def platform_mask(points: np.ndarray) -> np.ndarray:
    """Select raw Unity positions belonging to the station-platform layer."""

    return (
        np.asarray(points, dtype=float)[:, 1]
        <= density_base.PLATFORM_TO_STAIR_Y
    )


def platform_runs(points: np.ndarray) -> list[np.ndarray]:
    """Split platform positions into runs without bridging other layers."""

    points = np.asarray(points, dtype=float)
    mask = platform_mask(points)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        points[start:stop]
        for start, stop in changes.reshape(-1, 2)
        if stop - start >= 2
    ]


def projected_layout(zones, projector):
    """Use identical hall/platform geometry and screen limits for both metrics."""
    exploded_rows = density_base.exploded_zone_rows(zones)
    base_skeleton = build_selective_wireframe(exploded_rows, projector)
    hall_rows = overlay_hall_zones(zones)
    platform_rows = overlay_platform_zones(zones)
    overlay_skeleton = build_selective_wireframe(hall_rows, projector)
    overlay_platform_skeleton = build_selective_wireframe(platform_rows, projector)
    for layer, rows in (("hall", hall_rows), ("platform", platform_rows)):
        original = exploded_rows.loc[exploded_rows["asset_layer"].eq(layer)]
        np.testing.assert_allclose(
            rows[["y_min", "y_max"]].to_numpy(),
            original[["y_min", "y_max"]].to_numpy(),
            err_msg=f"{layer} backtracking and trajectory display heights differ",
        )
    _, hall_bottom = projected_zone_vertical_envelope(hall_rows, projector)
    platform_top, _ = projected_zone_vertical_envelope(platform_rows, projector)
    shared = np.isfinite(hall_bottom) & np.isfinite(platform_top)
    projected_gap = float(np.min(platform_top[shared] - hall_bottom[shared]))
    all_segments = np.concatenate(
        [base_skeleton, overlay_skeleton, overlay_platform_skeleton]
    )
    x_left = max(
        0.0,
        float(all_segments[:, :, 0].min() - PANEL_HORIZONTAL_PADDING_PX),
    )
    x_right = min(
        float(DEFAULT_CAMERA.width_px),
        float(all_segments[:, :, 0].max() + PANEL_HORIZONTAL_PADDING_PX),
    )
    y_top = float(
        all_segments[:, :, 1].min() - PANEL_VERTICAL_PADDING_PX
    )
    y_bottom = float(
        all_segments[:, :, 1].max() + PANEL_VERTICAL_PADDING_PX
    )
    return (
        base_skeleton, overlay_skeleton, overlay_platform_skeleton,
        projected_gap, projected_gap, x_left, x_right, y_top, y_bottom,
    )


def panel_setup(
    ax,
    *,
    x_left: float,
    x_right: float,
    y_top: float,
    y_bottom: float,
) -> None:
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def project_field_to_extended_view(
    field: np.ndarray,
    edges,
    projector,
    *,
    y_top: float,
    y_bottom: float,
    raster_width: int = 960,
):
    """Maximum-intensity project a 3D field onto the extended screen window."""

    centers = [(edge[:-1] + edge[1:]) * 0.5 for edge in edges]
    xx, yy, zz = np.meshgrid(*centers, indexing="ij")
    world = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = field.ravel()
    screen, _ = projector.project(world)
    scale = raster_width / DEFAULT_CAMERA.width_px
    raster_height = int(np.ceil((y_bottom - y_top) * scale))
    px = np.floor(screen[:, 0] * scale).astype(int)
    py = np.floor((screen[:, 1] - y_top) * scale).astype(int)
    valid = (
        (px >= 0)
        & (px < raster_width)
        & (py >= 0)
        & (py < raster_height)
        & np.isfinite(values)
        & (values > 0)
    )
    raster = np.zeros((raster_height, raster_width), dtype=float)
    np.maximum.at(raster, (py[valid], px[valid]), values[valid])
    return gaussian_filter(raster, sigma=2.0, mode="constant")


def add_panel_smoke_captions(fig, axes) -> None:
    """Label smoke beneath each equal-aspect panel without a left row caption."""
    caption_gap_in = 0.035
    for row_index, smoke in enumerate(density_base.SMOKE_LEVELS):
        for ax in axes[row_index]:
            position = ax.get_position()
            fig.text(
                (position.x0 + position.x1) / 2,
                position.y0 - caption_gap_in / fig.get_figheight(),
                density_base.SMOKE_LABELS[smoke],
                ha="center",
                va="top",
                fontsize=ROW_LABEL_SIZE,
                fontweight="normal",
                color="#30363D",
            )


def plot_combined(
    zones,
    records,
    projector,
    *,
    output_version: str,
    render_backtracking_only: bool,
    trajectory_palette_specs,
    output_filename: str,
    render_combined: bool = True,
) -> tuple[dict[str, Path], float, float]:
    global OVERLAY_HALL_DISPLAY_Y, OVERLAY_PLATFORM_DISPLAY_Y
    OVERLAY_HALL_DISPLAY_Y = float(density_base.HALL_DISPLAY_Y)
    OVERLAY_PLATFORM_DISPLAY_Y = float(density_base.PLATFORM_DISPLAY_Y)
    stair_connections = backtracking_stair_connections(zones, projector)
    base_bounds = density_base.density_bounds(zones, records)
    backtrack_bounds = overlay_heat_bounds(base_bounds)
    platform_backtrack_bounds = overlay_platform_heat_bounds(base_bounds)
    (
        base_skeleton,
        overlay_skeleton,
        overlay_platform_skeleton,
        projected_gap,
        platform_projected_gap,
        x_left,
        x_right,
        y_top,
        y_bottom,
    ) = projected_layout(zones, projector)
    target_by_scenario, exit_locations = load_target_exit_locations()
    shortest_path_records = density_base.load_tracks(zones)
    shortest_routes = density_base.build_astar_routes(shortest_path_records)
    density_fields = {}
    density_rasters = {}
    backtrack_fields = {}
    backtrack_rasters = {}
    platform_backtrack_fields = {}
    platform_backtrack_rasters = {}

    for scenario in density_base.SCENARIOS:
        group = [record for record in records if record.scenario == scenario]
        density, density_edges = density_base.three_dimensional_density(
            group, base_bounds
        )
        density_raster = project_field_to_extended_view(
            density,
            density_edges,
            projector,
            y_top=y_top,
            y_bottom=y_bottom,
        )
        backtrack_heat, backtrack_edges = three_dimensional_overlay_backtracking_heat(
            group, backtrack_bounds
        )
        backtrack_raster = project_field_to_extended_view(
            backtrack_heat,
            backtrack_edges,
            projector,
            y_top=y_top,
            y_bottom=y_bottom,
        )
        (
            platform_backtrack_heat,
            platform_backtrack_edges,
        ) = three_dimensional_platform_overlay_backtracking_heat(
            group, platform_backtrack_bounds
        )
        platform_backtrack_raster = project_field_to_extended_view(
            platform_backtrack_heat,
            platform_backtrack_edges,
            projector,
            y_top=y_top,
            y_bottom=y_bottom,
        )
        density_fields[scenario] = (density, density_edges)
        density_rasters[scenario] = density_raster
        backtrack_fields[scenario] = (backtrack_heat, backtrack_edges)
        backtrack_rasters[scenario] = backtrack_raster
        platform_backtrack_fields[scenario] = (
            platform_backtrack_heat,
            platform_backtrack_edges,
        )
        platform_backtrack_rasters[scenario] = platform_backtrack_raster

    # Colour scaling is resolved independently for every experimental group.
    # Density calculations are already participant-normalised; these per-group
    # display limits ensure each panel uses the complete Low-to-High colour span.
    density_raster_max = {
        scenario: float(density_rasters[scenario].max())
        for scenario in density_base.SCENARIOS
    }
    density_field_max = {
        scenario: float(density_fields[scenario][0].max())
        for scenario in density_base.SCENARIOS
    }
    # Hall and platform backtracking retain one common upper limit within each
    # group, so their relative strength remains comparable inside a panel.
    backtrack_raster_max = {
        scenario: max(
            float(backtrack_rasters[scenario].max()),
            float(platform_backtrack_rasters[scenario].max()),
        )
        for scenario in density_base.SCENARIOS
    }
    backtrack_field_max = {
        scenario: max(
            float(backtrack_fields[scenario][0].max()),
            float(platform_backtrack_fields[scenario][0].max()),
        )
        for scenario in density_base.SCENARIOS
    }
    all_display_maxima = [
        *density_raster_max.values(),
        *density_field_max.values(),
        *backtrack_raster_max.values(),
        *backtrack_field_max.values(),
    ]
    if min(all_display_maxima) <= 0:
        raise RuntimeError("A required density field is empty.")

    (
        density_cmaps,
        backtrack_cmap,
        shortest_path_cmap,
    ) = build_figure_colormaps(trajectory_palette_specs)
    single_trajectory_palette = len(set(trajectory_palette_specs.values())) == 1

    def draw_trajectory_panel(ax, scenario: str, *, show_title: bool) -> None:
        panel_setup(
            ax,
            x_left=x_left,
            x_right=x_right,
            y_top=y_top,
            y_bottom=y_bottom,
        )
        group = [record for record in records if record.scenario == scenario]
        guidance = scenario.rsplit("-", 1)[0]
        density_cmap = density_cmaps[guidance]
        density_raster = density_rasters[scenario]
        density_normalised = density_raster / density_raster_max[scenario]
        density_rgba = density_cmap(np.clip(density_normalised, 0.0, 1.0))
        density_rgba[..., 3] = np.where(
            density_normalised >= 0.012,
            np.clip(
                0.035 + 0.68 * np.power(density_normalised, 0.68),
                0.0,
                0.52,
            ),
            0.0,
        )
        ax.imshow(
            density_rgba,
            extent=[0, DEFAULT_CAMERA.width_px, y_bottom, y_top],
            interpolation="bilinear",
            zorder=2,
        )
        density, density_edges = density_fields[scenario]
        for record in group:
            display = density_base.explode_points(
                density_base.resample_path(record.points, 360)
            )
            add_high_opacity_density_coloured_path(
                ax,
                display,
                density,
                density_edges,
                projector,
                density_cmap,
                density_field_max[scenario],
            )
        density_levels = [0.06, 0.14, 0.30, 0.58]
        contour_x = np.linspace(
            0.0,
            DEFAULT_CAMERA.width_px,
            density_raster.shape[1],
        )
        contour_y = np.linspace(y_top, y_bottom, density_raster.shape[0])
        ax.contour(
            contour_x,
            contour_y,
            density_normalised,
            levels=density_levels,
            colors=[density_cmap(level) for level in density_levels],
            linewidths=[0.24, 0.30, 0.38, 0.50],
            alpha=0.40,
            zorder=4,
        )
        add_skeleton(
            ax,
            base_skeleton,
            color=TRAJECTORY_DENSITY_FRAME_COLOR,
            alpha=TRAJECTORY_DENSITY_FRAME_ALPHA,
            linewidth=BASE_SKELETON_LINEWIDTH,
            zorder=6,
        )
        add_shortest_path_reference(
            ax,
            shortest_routes[scenario],
            projector,
            shortest_path_cmap,
        )
        add_target_exit_markers(
            ax,
            scenario,
            target_by_scenario,
            exit_locations,
            projector,
        )
        if show_title:
            ax.set_title(
                f"{GUIDANCE_LABELS[guidance]}  (n={len(group)})",
                pad=5,
                fontsize=PANEL_TITLE_SIZE,
            )

    def draw_backtracking_panel(ax, scenario: str, *, show_title: bool) -> None:
        panel_setup(
            ax,
            x_left=x_left,
            x_right=x_right,
            y_top=y_top,
            y_bottom=y_bottom,
        )
        group = [record for record in records if record.scenario == scenario]
        guidance = scenario.rsplit("-", 1)[0]
        ax.add_collection(
            LineCollection(
                stair_connections,
                colors=[mpl.colors.to_rgba("#737D89", 0.78)],
                linewidths=0.65,
                linestyles=[(0, (3.0, 3.0))],
                capstyle="butt",
                zorder=6.5,
            )
        )
        backtrack_raster = backtrack_rasters[scenario]
        backtrack_normalised = (
            backtrack_raster / backtrack_raster_max[scenario]
        )
        backtrack_colour_position = np.sqrt(
            np.clip(backtrack_normalised, 0.0, 1.0)
        )
        backtrack_rgba = backtrack_cmap(backtrack_colour_position)
        backtrack_rgba[..., 3] = np.where(
            backtrack_normalised >= 0.0015,
            np.clip(
                0.035 + 0.52 * np.power(backtrack_normalised, 0.30),
                0.0,
                0.64,
            ),
            0.0,
        )
        ax.imshow(
            backtrack_rgba,
            extent=[0, DEFAULT_CAMERA.width_px, y_bottom, y_top],
            interpolation="bilinear",
            zorder=7,
        )
        add_overlay_path_context(ax, group, projector)
        backtrack_heat, backtrack_edges = backtrack_fields[scenario]
        for record in group:
            for run in backtrack.hall_runs(record.points):
                point_count = max(24, min(360, len(run) * 3))
                display = overlay_hall_points(
                    density_base.resample_path(run, point_count)
                )
                backtrack.add_heat_coloured_path(
                    ax,
                    display,
                    backtrack_heat,
                    backtrack_edges,
                    projector,
                    backtrack_cmap,
                    backtrack_field_max[scenario],
                )
        backtrack_levels = [0.02, 0.08, 0.24, 0.55]
        contour_x = np.linspace(
            0.0,
            DEFAULT_CAMERA.width_px,
            backtrack_raster.shape[1],
        )
        contour_y = np.linspace(y_top, y_bottom, backtrack_raster.shape[0])
        available_levels = [
            level
            for level in backtrack_levels
            if backtrack_normalised.max() > level
        ]
        if available_levels:
            ax.contour(
                contour_x,
                contour_y,
                backtrack_normalised,
                levels=available_levels,
                colors=[
                    backtrack_cmap(np.sqrt(level)) for level in available_levels
                ],
                linewidths=np.linspace(0.26, 0.52, len(available_levels)),
                alpha=0.40,
                zorder=8,
            )
        add_skeleton(
            ax,
            overlay_skeleton,
            color=BACKTRACKING_DENSITY_FRAME_COLOR,
            alpha=BACKTRACKING_DENSITY_FRAME_ALPHA,
            linewidth=OVERLAY_SKELETON_LINEWIDTH,
            zorder=9,
        )
        platform_backtrack_raster = platform_backtrack_rasters[scenario]
        platform_backtrack_normalised = (
            platform_backtrack_raster / backtrack_raster_max[scenario]
        )
        platform_backtrack_colour_position = np.sqrt(
            np.clip(platform_backtrack_normalised, 0.0, 1.0)
        )
        platform_backtrack_rgba = backtrack_cmap(
            platform_backtrack_colour_position
        )
        platform_backtrack_rgba[..., 3] = np.where(
            platform_backtrack_normalised >= 0.0015,
            np.clip(
                0.035
                + 0.52
                * np.power(platform_backtrack_normalised, 0.30),
                0.0,
                0.64,
            ),
            0.0,
        )
        ax.imshow(
            platform_backtrack_rgba,
            extent=[0, DEFAULT_CAMERA.width_px, y_bottom, y_top],
            interpolation="bilinear",
            zorder=7,
        )
        add_platform_overlay_path_context(ax, group, projector)
        (
            platform_backtrack_heat,
            platform_backtrack_edges,
        ) = platform_backtrack_fields[scenario]
        for record in group:
            for run in platform_runs(record.points):
                point_count = max(24, min(360, len(run) * 3))
                display = overlay_platform_points(
                    density_base.resample_path(run, point_count)
                )
                backtrack.add_heat_coloured_path(
                    ax,
                    display,
                    platform_backtrack_heat,
                    platform_backtrack_edges,
                    projector,
                    backtrack_cmap,
                    backtrack_field_max[scenario],
                )
        platform_available_levels = [
            level
            for level in backtrack_levels
            if platform_backtrack_normalised.max() > level
        ]
        if platform_available_levels:
            ax.contour(
                contour_x,
                contour_y,
                platform_backtrack_normalised,
                levels=platform_available_levels,
                colors=[
                    backtrack_cmap(np.sqrt(level))
                    for level in platform_available_levels
                ],
                linewidths=np.linspace(
                    0.26,
                    0.52,
                    len(platform_available_levels),
                ),
                alpha=0.40,
                zorder=8,
            )
        add_skeleton(
            ax,
            overlay_platform_skeleton,
            color=BACKTRACKING_DENSITY_FRAME_COLOR,
            alpha=BACKTRACKING_DENSITY_FRAME_ALPHA,
            linewidth=OVERLAY_SKELETON_LINEWIDTH,
            zorder=9,
        )
        add_target_exit_markers(
            ax,
            scenario,
            target_by_scenario,
            exit_locations,
            projector,
        )
        if show_title:
            ax.set_title(
                f"{GUIDANCE_LABELS[guidance]}  (n={len(group)})",
                pad=5,
                fontsize=PANEL_TITLE_SIZE,
            )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if render_combined:
        # A compact canvas is important here: equal-aspect projected panels leave
        # large internal blanks when the figure is unnecessarily tall.
        fig, axes = plt.subplots(
            4,
            5,
            figsize=FIGURE_SIZE_INCHES,
            facecolor="white",
        )
        fig.subplots_adjust(
            left=0.060,
            right=0.995,
            top=0.965,
            bottom=0.115 if single_trajectory_palette else 0.180,
            wspace=-0.045,
            hspace=0.015,
        )

        for index, scenario in enumerate(density_base.SCENARIOS):
            row, column = divmod(index, 5)
            ax = axes[row, column]
            backtrack_ax = axes[row + 2, column]
            panel_setup(
                ax,
                x_left=x_left,
                x_right=x_right,
                y_top=y_top,
                y_bottom=y_bottom,
            )
            panel_setup(
                backtrack_ax,
                x_left=x_left,
                x_right=x_right,
                y_top=y_top,
                y_bottom=y_bottom,
            )
            group = [record for record in records if record.scenario == scenario]
            guidance = scenario.rsplit("-", 1)[0]
            density_cmap = density_cmaps[guidance]

            density_raster = density_rasters[scenario]
            density_normalised = density_raster / density_raster_max[scenario]
            density_rgba = density_cmap(np.clip(density_normalised, 0.0, 1.0))
            density_rgba[..., 3] = np.where(
                density_normalised >= 0.012,
                np.clip(
                    0.035 + 0.68 * np.power(density_normalised, 0.68),
                    0.0,
                    0.52,
                ),
                0.0,
            )
            ax.imshow(
                density_rgba,
                extent=[0, DEFAULT_CAMERA.width_px, y_bottom, y_top],
                interpolation="bilinear",
                zorder=2,
            )
            density, density_edges = density_fields[scenario]
            for record in group:
                display = density_base.explode_points(
                    density_base.resample_path(record.points, 360)
                )
                add_high_opacity_density_coloured_path(
                    ax,
                    display,
                    density,
                    density_edges,
                    projector,
                    density_cmap,
                    density_field_max[scenario],
                )
            density_levels = [0.06, 0.14, 0.30, 0.58]
            contour_x = np.linspace(0.0, DEFAULT_CAMERA.width_px, density_raster.shape[1])
            contour_y = np.linspace(y_top, y_bottom, density_raster.shape[0])
            ax.contour(
                contour_x,
                contour_y,
                density_normalised,
                levels=density_levels,
                colors=[density_cmap(level) for level in density_levels],
                linewidths=[0.24, 0.30, 0.38, 0.50],
                alpha=0.40,
                zorder=4,
            )
            add_skeleton(
                ax,
                base_skeleton,
                color=TRAJECTORY_DENSITY_FRAME_COLOR,
                alpha=TRAJECTORY_DENSITY_FRAME_ALPHA,
                linewidth=BASE_SKELETON_LINEWIDTH,
                zorder=6,
            )
            add_shortest_path_reference(
                ax,
                shortest_routes[scenario],
                projector,
                shortest_path_cmap,
            )

            add_target_exit_markers(
                ax, scenario, target_by_scenario, exit_locations, projector
            )
            if row == 0:
                ax.set_title(
                    f"{GUIDANCE_LABELS[guidance]}  (n={len(group)})",
                    pad=5,
                    fontsize=PANEL_TITLE_SIZE,
                )
            ax = backtrack_ax
            ax.add_collection(
                LineCollection(
                    stair_connections,
                    colors=[mpl.colors.to_rgba("#737D89", 0.78)],
                    linewidths=0.65,
                    linestyles=[(0, (3.0, 3.0))],
                    capstyle="butt",
                    zorder=6.5,
                )
            )

            backtrack_raster = backtrack_rasters[scenario]
            backtrack_normalised = (
                backtrack_raster / backtrack_raster_max[scenario]
            )
            backtrack_colour_position = np.sqrt(
                np.clip(backtrack_normalised, 0.0, 1.0)
            )
            backtrack_rgba = backtrack_cmap(backtrack_colour_position)
            backtrack_rgba[..., 3] = np.where(
                backtrack_normalised >= 0.0015,
                np.clip(
                    0.035 + 0.52 * np.power(backtrack_normalised, 0.30),
                    0.0,
                    0.64,
                ),
                0.0,
            )
            ax.imshow(
                backtrack_rgba,
                extent=[0, DEFAULT_CAMERA.width_px, y_bottom, y_top],
                interpolation="bilinear",
                zorder=7,
            )
            add_overlay_path_context(ax, group, projector)
            backtrack_heat, backtrack_edges = backtrack_fields[scenario]
            for record in group:
                for run in backtrack.hall_runs(record.points):
                    point_count = max(24, min(360, len(run) * 3))
                    display = overlay_hall_points(
                        density_base.resample_path(run, point_count)
                    )
                    backtrack.add_heat_coloured_path(
                        ax,
                        display,
                        backtrack_heat,
                        backtrack_edges,
                        projector,
                        backtrack_cmap,
                        backtrack_field_max[scenario],
                    )
            backtrack_levels = [0.02, 0.08, 0.24, 0.55]
            available_levels = [
                level
                for level in backtrack_levels
                if backtrack_normalised.max() > level
            ]
            if available_levels:
                ax.contour(
                    contour_x,
                    contour_y,
                    backtrack_normalised,
                    levels=available_levels,
                    colors=[
                        backtrack_cmap(np.sqrt(level)) for level in available_levels
                    ],
                    linewidths=np.linspace(0.26, 0.52, len(available_levels)),
                    alpha=0.40,
                    zorder=8,
                )
            add_skeleton(
                ax,
                overlay_skeleton,
                color=BACKTRACKING_DENSITY_FRAME_COLOR,
                alpha=BACKTRACKING_DENSITY_FRAME_ALPHA,
                linewidth=OVERLAY_SKELETON_LINEWIDTH,
                zorder=9,
            )

            platform_backtrack_raster = platform_backtrack_rasters[scenario]
            platform_backtrack_normalised = (
                platform_backtrack_raster / backtrack_raster_max[scenario]
            )
            platform_backtrack_colour_position = np.sqrt(
                np.clip(platform_backtrack_normalised, 0.0, 1.0)
            )
            platform_backtrack_rgba = backtrack_cmap(
                platform_backtrack_colour_position
            )
            platform_backtrack_rgba[..., 3] = np.where(
                platform_backtrack_normalised >= 0.0015,
                np.clip(
                    0.035
                    + 0.52
                    * np.power(platform_backtrack_normalised, 0.30),
                    0.0,
                    0.64,
                ),
                0.0,
            )
            ax.imshow(
                platform_backtrack_rgba,
                extent=[0, DEFAULT_CAMERA.width_px, y_bottom, y_top],
                interpolation="bilinear",
                zorder=7,
            )
            add_platform_overlay_path_context(ax, group, projector)
            (
                platform_backtrack_heat,
                platform_backtrack_edges,
            ) = platform_backtrack_fields[scenario]
            for record in group:
                for run in platform_runs(record.points):
                    point_count = max(24, min(360, len(run) * 3))
                    display = overlay_platform_points(
                        density_base.resample_path(run, point_count)
                    )
                    backtrack.add_heat_coloured_path(
                        ax,
                        display,
                        platform_backtrack_heat,
                        platform_backtrack_edges,
                        projector,
                        backtrack_cmap,
                        backtrack_field_max[scenario],
                    )
            platform_available_levels = [
                level
                for level in backtrack_levels
                if platform_backtrack_normalised.max() > level
            ]
            if platform_available_levels:
                ax.contour(
                    contour_x,
                    contour_y,
                    platform_backtrack_normalised,
                    levels=platform_available_levels,
                    colors=[
                        backtrack_cmap(np.sqrt(level))
                        for level in platform_available_levels
                    ],
                    linewidths=np.linspace(
                        0.26,
                        0.52,
                        len(platform_available_levels),
                    ),
                    alpha=0.40,
                    zorder=8,
                )
            add_skeleton(
                ax,
                overlay_platform_skeleton,
                color=BACKTRACKING_DENSITY_FRAME_COLOR,
                alpha=BACKTRACKING_DENSITY_FRAME_ALPHA,
                linewidth=OVERLAY_SKELETON_LINEWIDTH,
                zorder=9,
            )
            add_target_exit_markers(
                ax,
                scenario,
                target_by_scenario,
                exit_locations,
                projector,
            )

        for row_index in range(4):
            smoke = density_base.SMOKE_LEVELS[row_index % 2]
            metric = "Trajectories" if row_index < 2 else "Backtracking"
            position = axes[row_index, 0].get_position()
            fig.text(
                0.024, (position.y0 + position.y1) / 2,
                f"{metric}\n{density_base.SMOKE_LABELS[smoke]}",
                ha="center", va="center", rotation=90,
                fontsize=ROW_LABEL_SIZE, fontweight="semibold", color="#30363D",
            )

        if single_trajectory_palette:
            add_single_trajectory_density_legend(
                fig,
                next(iter(density_cmaps.values())),
            )
            backtrack_cax_bounds = [0.3775, 0.050, 0.245, 0.014]
            shortest_path_cax_bounds = [0.680, 0.050, 0.245, 0.014]
        else:
            add_trajectory_density_palette_legend(fig, density_cmaps)
            backtrack_cax_bounds = [0.165, 0.047, 0.285, 0.014]
            shortest_path_cax_bounds = [0.550, 0.047, 0.285, 0.014]

        backtrack_cax = fig.add_axes(backtrack_cax_bounds)
        backtrack_colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
                cmap=backtrack_cmap,
            ),
            cax=backtrack_cax,
            orientation="horizontal",
        )
        backtrack_colorbar.set_ticks([0.0, 0.25, 1.0])
        backtrack_colorbar.set_ticklabels(["Low", "Medium", "High"])
        backtrack_colorbar.set_label(
            "Hall/platform backtracking heat · within-group scale",
            labelpad=3,
        )
        backtrack_colorbar.ax.xaxis.set_label_position("top")
        backtrack_colorbar.ax.tick_params(labelsize=LEGEND_TEXT_SIZE, pad=2)

        shortest_path_cax = fig.add_axes(shortest_path_cax_bounds)
        shortest_path_colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=Normalize(0, 1),
                cmap=shortest_path_cmap,
            ),
            cax=shortest_path_cax,
            orientation="horizontal",
        )
        shortest_path_colorbar.set_ticks([0, 0.5, 1])
        shortest_path_colorbar.set_ticklabels(["Start", "50%", "Target exit"])
        shortest_path_colorbar.set_label("Shortest-path progress", labelpad=3)
        shortest_path_colorbar.ax.xaxis.set_label_position("top")
        shortest_path_colorbar.ax.tick_params(labelsize=LEGEND_TEXT_SIZE, pad=2)

        output = RESULT_DIR / output_filename
        fig.savefig(
            output,
            dpi=OUTPUT_DPI,
            bbox_inches=None,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

    panel_smoke_captions = output_version == "03"
    trajectory_height = 6.6 if not single_trajectory_palette else 6.3
    trajectory_bottom = 0.235 if not single_trajectory_palette else 0.170
    trajectory_fig, trajectory_axes = plt.subplots(
        2,
        5,
        figsize=(FIGURE_SIZE_INCHES[0], trajectory_height),
        facecolor="white",
    )
    trajectory_fig.subplots_adjust(
        left=0.015 if panel_smoke_captions else 0.060,
        right=0.995,
        top=0.945,
        bottom=trajectory_bottom,
        wspace=-0.045,
        hspace=0.150 if panel_smoke_captions else 0.015,
    )
    for index, scenario in enumerate(density_base.SCENARIOS):
        row, column = divmod(index, 5)
        draw_trajectory_panel(
            trajectory_axes[row, column],
            scenario,
            show_title=row == 0,
        )
    if panel_smoke_captions:
        add_panel_smoke_captions(trajectory_fig, trajectory_axes)
    else:
        for row_index, smoke in enumerate(density_base.SMOKE_LEVELS):
            position = trajectory_axes[row_index, 0].get_position()
            trajectory_fig.text(
                0.024,
                (position.y0 + position.y1) / 2,
                f"Trajectories\n{density_base.SMOKE_LABELS[smoke]}",
                ha="center",
                va="center",
                rotation=90,
                fontsize=ROW_LABEL_SIZE,
                fontweight="semibold",
                color="#30363D",
            )
    if single_trajectory_palette:
        add_single_trajectory_density_legend(
            trajectory_fig,
            next(iter(density_cmaps.values())),
            bounds=(0.165, 0.052, 0.285, 0.018),
        )
        add_shortest_path_legend(
            trajectory_fig,
            shortest_path_cmap,
            bounds=(0.550, 0.052, 0.285, 0.018),
        )
    else:
        add_trajectory_density_palette_legend(
            trajectory_fig,
            density_cmaps,
            title_y=0.160,
            bar_y=0.132,
        )
        add_shortest_path_legend(
            trajectory_fig,
            shortest_path_cmap,
            bounds=(0.350, 0.042, 0.300, 0.016),
        )
    trajectory_output = RESULT_DIR / (
        f"{output_version}_all_10_groups_trajectory_density_only_"
        "unity_600dpi.png"
    )
    trajectory_fig.savefig(
        trajectory_output,
        dpi=OUTPUT_DPI,
        bbox_inches=None,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(trajectory_fig)

    outputs = {"trajectories": trajectory_output}
    if render_combined:
        outputs["combined"] = output
    if render_backtracking_only:
        backtracking_fig, backtracking_axes = plt.subplots(
            2,
            5,
            figsize=(FIGURE_SIZE_INCHES[0], 6.3),
            facecolor="white",
        )
        backtracking_fig.subplots_adjust(
            left=0.015,
            right=0.995,
            top=0.945,
            bottom=0.170,
            wspace=-0.045,
            hspace=0.150,
        )
        for index, scenario in enumerate(density_base.SCENARIOS):
            row, column = divmod(index, 5)
            draw_backtracking_panel(
                backtracking_axes[row, column],
                scenario,
                show_title=row == 0,
            )
        add_panel_smoke_captions(backtracking_fig, backtracking_axes)
        add_backtracking_legend(
            backtracking_fig,
            backtrack_cmap,
            bounds=(0.350, 0.052, 0.300, 0.018),
        )
        backtracking_output = RESULT_DIR / (
            "all_10_groups_hall_platform_backtracking_only_"
            "unity_600dpi.png"
        )
        backtracking_fig.savefig(
            backtracking_output,
            dpi=OUTPUT_DPI,
            bbox_inches=None,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(backtracking_fig)
        outputs["backtracking"] = backtracking_output
    return (
        outputs,
        projected_gap,
        platform_projected_gap,
    )


def platform_backtracking_audit(records) -> dict:
    step_counts = {}
    distance_totals = {}
    for scenario in density_base.SCENARIOS:
        steps = 0
        distance = 0.0
        for record in records:
            if record.scenario != scenario:
                continue
            mask = platform_mask(record.points)
            valid = (
                record.backtrack_increment_m > 0
            ) & mask & np.r_[False, mask[:-1]]
            steps += int(np.count_nonzero(valid))
            distance += float(
                np.nansum(record.backtrack_increment_m[valid])
            )
        step_counts[scenario] = steps
        distance_totals[scenario] = distance
    if not any(distance_totals.values()):
        raise RuntimeError("No platform A* backtracking was detected.")
    return {
        "platform_layer_a_star_backtracking_step_counts": step_counts,
        "platform_layer_a_star_backtracking_distance_m": distance_totals,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caption-panels-only", action="store_true",
        help="Render only the 03 density and standalone backtracking figures.",
    )
    args = parser.parse_args()
    configure_style()
    zones = load_zones()
    records = backtrack.load_tracks(zones)
    audit = backtrack.validate_records(records)
    platform_audit = platform_backtracking_audit(records)
    projector = UnityOrthographicProjector(DEFAULT_CAMERA)
    variants = {}
    if RENDER_04 and not args.caption_panels_only:
        variants["04"] = (
            TRAJECTORY_PALETTE_SPECS,
            "04_all_10_groups_trajectory_density_with_hall_platform_"
            "backtracking_unity_600dpi.png",
        )
    if RENDER_03 or args.caption_panels_only:
        variants["03"] = (
            LEGACY_03_TRAJECTORY_PALETTE_SPECS,
            "03_all_10_groups_trajectory_density_with_hall_platform_"
            "backtracking_unity_600dpi.png",
        )
    if not variants:
        raise ValueError("Enable at least one of RENDER_03 or RENDER_04")
    generated = {}
    projected_gap = None
    platform_projected_gap = None
    for variant_index, (
        version,
        (palette_specs, output_filename),
    ) in enumerate(variants.items()):
        outputs, projected_gap, platform_projected_gap = plot_combined(
            zones,
            records,
            projector,
            output_version=version,
            render_backtracking_only=variant_index == 0,
            trajectory_palette_specs=palette_specs,
            output_filename=output_filename,
            render_combined=not args.caption_panels_only,
        )
        generated[version] = {
            kind: str(path.relative_to(ROOT))
            for kind, path in outputs.items()
        }
    print(
        json.dumps(
            {
                "generated": generated,
                "colormap_ids": {
                    "backtracking_heat": BACKTRACKING_HEAT_COLORMAP_ID,
                    "astar_shortest_path": ASTAR_SHORTEST_PATH_COLORMAP_ID,
                },
                "trajectory_palettes": {
                    version: {
                        GUIDANCE_LABELS[guidance]: {
                            "a5_palette": palette_id,
                            "reversed": reverse,
                        }
                        for guidance, (
                            palette_id,
                            reverse,
                        ) in palette_specs.items()
                    }
                    for version, (palette_specs, _) in variants.items()
                },
                "guidance_display_labels": GUIDANCE_LABELS,
                "colour_scaling": {
                    "trajectory_density": "zero_to_group_maximum",
                    "backtracking_heat": (
                        "zero_to_group_hall_platform_combined_maximum"
                    ),
                    "participant_normalised_before_colour_scaling": True,
                },
                "trajectory_hall_platform_projected_gap_px": projected_gap,
                "overlay_hall_display_y": OVERLAY_HALL_DISPLAY_Y,
                "backtracking_hall_platform_projected_gap_px": platform_projected_gap,
                "layout": {
                    "combined": (
                        "compact 4x5: trajectory low/high smoke, "
                        "backtracking low/high smoke"
                    ),
                    "separate": "compact 2x5 per metric",
                },
                "trajectory_frame_color": TRAJECTORY_DENSITY_FRAME_COLOR,
                "overlay_platform_display_y": OVERLAY_PLATFORM_DISPLAY_Y,
                **audit,
                **platform_audit,
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
