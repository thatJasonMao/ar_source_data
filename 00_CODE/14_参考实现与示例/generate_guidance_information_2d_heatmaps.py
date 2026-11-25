"""Draw all available guidance-information acquisition heatmaps in 2D.

The figure contains the eight experimental conditions for which a comparable
guidance AOI exists (four guidance modes x two smoke levels).  Each condition
is shown on the station-hall and platform plans.  Heat is the participant-
balanced local proportion of valid AOI time spent looking at guidance.

Layer order (bottom to top): floor-plan image, heat field, semantic-zone frame.
Unity coordinates follow the original Hall/Platform floor plans so Exit A is at
the upper-left.  Background images are only aspect-fitted; they are never stretched.

Run with the requested environment:
    C:\\Users\\11274\\.conda\\envs\\yoyo\\python.exe \
        generate_guidance_information_2d_heatmaps.py
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import contains_xy


DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
AOI_DIR = ROOT / "Results_Guidance_GazeAOI_Integrated"
TRIALS_FILE = ROOT / "Results_Guidance_Spatial_Causal_Analysis" / "data" / "analysis_trials.csv"
ZONES_FILE = ROOT / "Results_Guidance_Spatial_Causal_Analysis" / "zone_annotation" / "semantic_zones.csv"
BACKGROUND_DIR = ROOT / "Background_Trajectory"
RESULT_DIR = DEMO_DIR / "Results_Guidance_Information_2D_Heatmaps"
OUTPUT_FILE = RESULT_DIR / "01_all_guidance_groups_hall_platform_2d_heatmaps.png"

TIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
GUIDANCE_ORDER = ["ArrowRibbon", "MiniMap", "TextBar", "Full"]
SMOKE_ORDER = ["LowSmoke", "HighSmoke"]
FLOOR_ORDER = ["Hall", "Platform"]

GUIDANCE_LABELS = {
    "ArrowRibbon": "Light carpet",
    "MiniMap": "Mini-map",
    "TextBar": "Text bar",
    "Full": "Full guidance",
}
ROW_LABELS = {
    ("LowSmoke", "Hall"): "Low smoke · Hall",
    ("LowSmoke", "Platform"): "Low smoke · Platform",
    ("HighSmoke", "Hall"): "High smoke · Hall",
    ("HighSmoke", "Platform"): "High smoke · Platform",
}

# These are the same semantic colours used by the formal spatial-partition
# renderer.  Keeping the mapping here makes this demo script self-contained.
ZONE_COLORS = {
    "Platform": "#8DBBD5",
    "Spawn": "#B45A8D",
    "StairEntrance": "#E3C14B",
    "Stair": "#E58B3A",
    "HallInsideGate": "#93DBE0",
    "Gate": "#4C9F70",
    "HallOutsideGate": "#C5C7C9",
    "ExitCorridor": "#4F83B6",
    "DecisionZone": "#D36A4A",
    "BranchCorridor": "#9B9D9F",
    "Exit": "#30343B",
}

ZONE_LABELS = {
    "Platform": "Platform",
    "Spawn": "Start area",
    "StairEntrance": "Stair entrance",
    "Stair": "Stair",
    "HallInsideGate": "Hall inside-gate",
    "Gate": "Gate",
    "HallOutsideGate": "Hall outside-gate",
    "DecisionZone": "Decision zone",
    "ExitCorridor": "Exit corridor",
    "BranchCorridor": "Branch corridor",
    "Exit": "Exit",
}

# Heat appearance.  Each condition gets its own robust upper bound across its
# Hall + Platform panels; the displayed colour is therefore within-group 0–1.
HEAT_CMAP = "turbo"
ROBUST_HIGH_QUANTILE = 0.98
MIN_GUIDANCE_EVENT_S = 0.100
MAX_CONTIGUOUS_GAP_S = 0.100
DEFAULT_BANDWIDTH_M = 2.0
GRID_WIDTH = 220

# Manually calibrated against Background_Trajectory/Hall.png and Platform.png.
# The zone frames and heat field stay in Unity map coordinates; only the
# background image is transformed.  Scale is uniform in X/Y, so the source
# image aspect ratio is preserved exactly.
BACKGROUND_CALIBRATION = {
    "Hall": {
        "scale": 1.0565,
        "offset_x": 0.0,
        "offset_y": 1.55,
        "rotation_deg": 0.0,
    },
    "Platform": {
        "scale": 0.943,
        "offset_x": 0.0,
        "offset_y": 0.3,
        "rotation_deg": 0.0,
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titlesize": 10.0,
        }
    )


def layer_from_zone_type(zone_type: str) -> str:
    if zone_type == "Stair":
        return "Stair"
    if zone_type in {"Platform", "Spawn", "StairEntrance"}:
        return "Platform"
    return "Hall"


def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(ZONES_FILE, low_memory=False)
    zones["asset_layer"] = zones["zone_type"].astype(str).map(layer_from_zone_type)
    zones["parts"] = zones["collider_parts_xz"].map(json.loads)

    # Display transform used by the original trajectory floor plans:
    # (Unity X, Unity Z) -> (-X, -Z), placing Exit A upper-left.
    def display_geometry(parts: list[dict]):
        rectangles = [
            box(-float(p["xmax"]), -float(p["zmax"]), -float(p["xmin"]), -float(p["zmin"]))
            for p in parts
        ]
        return unary_union(rectangles)

    zones["geometry"] = zones["parts"].map(display_geometry)
    zones["priority"] = pd.to_numeric(zones["priority"], errors="raise")
    zones["y_min"] = pd.to_numeric(zones["y_min"], errors="raise")
    zones["y_max"] = pd.to_numeric(zones["y_max"], errors="raise")
    return zones.sort_values(["priority", "canonical_order", "zone_id"], ascending=[False, True, True])


def image_path(floor: str) -> Path:
    return BACKGROUND_DIR / ("Hall.png" if floor == "Hall" else "Platform.png")


@lru_cache(maxsize=2)
def background_image(floor: str) -> np.ndarray:
    """Return a display-sized copy of the original image without distortion."""
    path = image_path(floor)
    if not path.exists():
        raise FileNotFoundError(f"Missing background image: {path}")
    with Image.open(path) as source:
        image = source.convert("RGBA")
        image.thumbnail((1200, 2400), Image.Resampling.LANCZOS)
        return np.asarray(image).copy()


def geometry_extent(zones: pd.DataFrame, floor: str) -> tuple[float, float, float, float]:
    """Aspect-fit the layer geometry to its source image; never stretch it."""
    part = zones.loc[zones["asset_layer"].eq(floor)]
    bounds = np.asarray([geom.bounds for geom in part["geometry"]], dtype=float)
    xmin, ymin = bounds[:, :2].min(axis=0)
    xmax, ymax = bounds[:, 2:].max(axis=0)
    xmargin = max((xmax - xmin) * 0.04, 1.0)
    ymargin = max((ymax - ymin) * 0.04, 1.0)
    xmin, xmax = xmin - xmargin, xmax + xmargin
    ymin, ymax = ymin - ymargin, ymax + ymargin

    image = background_image(floor)
    image_aspect = image.shape[1] / image.shape[0]
    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    width, height = xmax - xmin, ymax - ymin
    if width / height < image_aspect:
        width = height * image_aspect
        xmin, xmax = xc - width / 2.0, xc + width / 2.0
    else:
        height = width / image_aspect
        ymin, ymax = yc - height / 2.0, yc + height / 2.0
    return float(xmin), float(xmax), float(ymin), float(ymax)


def calibrated_background_extent(
    extent: tuple[float, float, float, float], floor: str
) -> tuple[float, float, float, float]:
    """Apply manual translation and uniform scale around the image centre."""
    xmin, xmax, ymin, ymax = extent
    calibration = BACKGROUND_CALIBRATION[floor]
    scale = float(calibration["scale"])
    centre_x = (xmin + xmax) / 2.0 + float(calibration["offset_x"])
    centre_y = (ymin + ymax) / 2.0 + float(calibration["offset_y"])
    half_width = (xmax - xmin) * scale / 2.0
    half_height = (ymax - ymin) * scale / 2.0
    return (
        centre_x - half_width,
        centre_x + half_width,
        centre_y - half_height,
        centre_y + half_height,
    )


def as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0).to_numpy(bool)
    return (
        series.fillna("").astype(str).str.strip().str.lower()
        .isin({"1", "true", "yes", "y"}).to_numpy(bool)
    )


def aoi_columns(guidance: str) -> tuple[list[str], str]:
    if guidance == "ArrowRibbon":
        return ["ValidLightBlanketAOI"], "GazeOnLightBlanket"
    if guidance == "MiniMap":
        return ["ValidMiniMapAOI"], "GazeOnMiniMap"
    if guidance == "TextBar":
        return ["ValidTextAOI"], "GazeOnTextGuidance"
    if guidance == "Full":
        return ["ValidLightBlanketAOI", "ValidMiniMapAOI", "ValidTextAOI"], "AnyGuidanceGaze"
    raise ValueError(f"No comparable guidance AOI for {guidance}")


def integrated_path(row) -> Path | None:
    filename = str(row.FileName)
    marker = "_Subject_Simulation_Info_"
    if marker not in filename:
        return None
    trial, stamp = filename.replace(marker, "|", 1).removesuffix(".csv").split("|", 1)
    folder = AOI_DIR / str(row.Scenario) / str(row.ParticipantID)
    matches = list(folder.glob(f"{trial}_GuidanceGazeIntegrated_{stamp}.csv"))
    return matches[0] if len(matches) == 1 else None


def sample_weights(parsed_time: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    valid_time = parsed_time.notna().to_numpy()
    ns = parsed_time.astype("int64").to_numpy()
    elapsed = np.full(len(parsed_time), np.nan, dtype=float)
    if valid_time.sum() < 2:
        return elapsed, np.zeros(len(parsed_time), dtype=float)
    first = ns[valid_time][0]
    elapsed[valid_time] = (ns[valid_time] - first) / 1e9
    positive = np.diff(elapsed[valid_time])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    median_dt = float(np.median(positive)) if len(positive) else 0.02
    weights = np.r_[np.diff(elapsed), median_dt]
    weights[~np.isfinite(weights) | (weights <= 0)] = median_dt
    weights = np.minimum(weights, MAX_CONTIGUOUS_GAP_S)
    return elapsed, weights


def retain_guidance_events(
    gaze: np.ndarray, elapsed: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Reject isolated AOI hits; retain gaze runs lasting at least 100 ms."""
    retained = np.zeros(len(gaze), dtype=bool)
    indices = np.flatnonzero(gaze & np.isfinite(elapsed))
    if not len(indices):
        return retained
    breaks = np.flatnonzero(
        (np.diff(indices) > 1)
        | (np.diff(elapsed[indices]) > MAX_CONTIGUOUS_GAP_S)
    ) + 1
    for event in np.split(indices, breaks):
        if float(weights[event].sum()) >= MIN_GUIDANCE_EVENT_S:
            retained[event] = True
    return retained


def assign_floor(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, zones: pd.DataFrame
) -> np.ndarray:
    """Assign the highest-priority semantic collider, then keep its layer."""
    layer = np.full(len(x), "Unassigned", dtype=object)
    for row in zones.itertuples(index=False):
        available = layer == "Unassigned"
        if not available.any():
            break
        xz_hit = np.zeros(len(x), dtype=bool)
        for part in row.parts:
            xz_hit |= (
                (x >= float(part["xmin"]))
                & (x <= float(part["xmax"]))
                & (z >= float(part["zmin"]))
                & (z <= float(part["zmax"]))
            )
        hit = available & xz_hit & (y >= row.y_min) & (y <= row.y_max)
        layer[hit] = row.asset_layer
    return layer


def read_trial_points(row, zones: pd.DataFrame) -> pd.DataFrame | None:
    aoi_path = integrated_path(row)
    movement_path = (
        ROOT
        / "RAW_DATA"
        / str(row.Scenario)
        / str(row.SubjectFolder)
        / str(row.FileName)
    )
    if aoi_path is None or not movement_path.exists():
        return None

    valid_columns, gaze_column = aoi_columns(str(row.Guidance))
    aoi = pd.read_csv(
        aoi_path,
        usecols=["TimeStamp", *valid_columns, gaze_column],
        low_memory=False,
    )
    aoi["ParsedTime"] = pd.to_datetime(aoi["TimeStamp"].astype(str), format=TIME_FORMAT, errors="coerce")
    aoi = aoi.sort_values("ParsedTime").drop_duplicates("ParsedTime").reset_index(drop=True)
    elapsed, weights = sample_weights(aoi["ParsedTime"])
    valid = np.logical_or.reduce([as_bool(aoi[column]) for column in valid_columns])
    gaze_raw = as_bool(aoi[gaze_column]) & valid
    gaze = retain_guidance_events(gaze_raw, elapsed, weights)

    movement = pd.read_csv(
        movement_path,
        usecols=["TimeStamp", "PosX", "PosY", "PosZ"],
        low_memory=False,
    )
    movement["ParsedTime"] = pd.to_datetime(
        movement["TimeStamp"].astype(str), format=TIME_FORMAT, errors="coerce"
    )
    for column in ["PosX", "PosY", "PosZ"]:
        movement[column] = pd.to_numeric(movement[column], errors="coerce")
    movement = (
        movement.dropna(subset=["ParsedTime", "PosX", "PosY", "PosZ"])
        .sort_values("ParsedTime")
        .drop_duplicates("ParsedTime")
    )
    if len(movement) < 2:
        return None

    query = aoi["ParsedTime"].astype("int64").to_numpy(dtype=np.int64)
    source = movement["ParsedTime"].astype("int64").to_numpy(dtype=np.int64)
    temporal = valid & aoi["ParsedTime"].notna().to_numpy() & (query >= source[0]) & (query <= source[-1])
    if not temporal.any():
        return None

    # Convert nanoseconds to local seconds before interpolation to retain precision.
    origin = source[0]
    source_s = (source - origin) / 1e9
    query_s = (query[temporal] - origin) / 1e9
    px = np.interp(query_s, source_s, movement["PosX"].to_numpy(float))
    py = np.interp(query_s, source_s, movement["PosY"].to_numpy(float))
    pz = np.interp(query_s, source_s, movement["PosZ"].to_numpy(float))
    floors = assign_floor(px, py, pz, zones)
    chosen_weights = weights[temporal]
    chosen_gaze = gaze[temporal]
    keep = np.isin(floors, FLOOR_ORDER) & np.isfinite(chosen_weights) & (chosen_weights > 0)
    if not keep.any():
        return None
    return pd.DataFrame(
        {
            "Scenario": str(row.Scenario),
            "Guidance": str(row.Guidance),
            "Smoke": str(row.Smoke),
            "ParticipantID": str(row.ParticipantID),
            "Floor": floors[keep],
            "PlotX": -px[keep],
            "PlotY": -pz[keep],
            "ValidDuration_s": chosen_weights[keep],
            "GazeDuration_s": chosen_weights[keep] * chosen_gaze[keep].astype(float),
        }
    )


def load_guidance_points(zones: pd.DataFrame) -> pd.DataFrame:
    trials = pd.read_csv(TRIALS_FILE, low_memory=False)
    trials = trials.loc[
        trials["Guidance"].isin(GUIDANCE_ORDER)
        & trials["Smoke"].isin(SMOKE_ORDER)
        & trials["AnalysisIncluded"].fillna(False).astype(bool)
    ].copy()
    expected = {f"{guidance}-{smoke}" for smoke in SMOKE_ORDER for guidance in GUIDANCE_ORDER}
    present = set(trials["Scenario"].astype(str))
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Missing experimental conditions in analysis_trials.csv: {missing}")

    parts: list[pd.DataFrame] = []
    missing_trials: list[str] = []
    total = len(trials)
    for number, row in enumerate(trials.itertuples(index=False), 1):
        points = read_trial_points(row, zones)
        if points is None:
            missing_trials.append(f"{row.Scenario}/{row.ParticipantID}/{row.FileName}")
        else:
            parts.append(points)
        if number % 30 == 0 or number == total:
            print(f"Mapped guidance AOI to movement positions: {number}/{total}")
    if not parts:
        raise RuntimeError("No guidance AOI samples could be mapped to movement positions")
    if missing_trials:
        print(f"Warning: {len(missing_trials)} trials were unavailable and skipped")
    return pd.concat(parts, ignore_index=True)


def load_target_exits() -> dict[str, str]:
    """Return the single experiment-defined target exit for every condition."""
    trials = pd.read_csv(TRIALS_FILE, low_memory=False)
    trials = trials.loc[
        trials["Guidance"].isin(GUIDANCE_ORDER)
        & trials["Smoke"].isin(SMOKE_ORDER)
        & trials["AnalysisIncluded"].fillna(False).astype(bool),
        ["Scenario", "TargetExit"],
    ].dropna()
    grouped = trials.groupby("Scenario", observed=True)["TargetExit"].agg(
        lambda values: sorted(set(values.astype(str)))
    )
    invalid = grouped.loc[grouped.map(len).ne(1)]
    if not invalid.empty:
        raise ValueError(f"Each condition must have exactly one target exit: {invalid.to_dict()}")
    return {str(scenario): exits[0] for scenario, exits in grouped.items()}


def floor_footprint(zones: pd.DataFrame, floor: str):
    return unary_union(zones.loc[zones["asset_layer"].eq(floor), "geometry"].tolist())


def participant_balanced_surface(
    data: pd.DataFrame,
    zones: pd.DataFrame,
    floor: str,
    extent: tuple[float, float, float, float],
    bandwidth_m: float,
) -> dict:
    """Average local gaze/valid-time ratios with equal participant influence."""
    xmin, xmax, ymin, ymax = extent
    nx = GRID_WIDTH
    ny = max(GRID_WIDTH, int(round(nx * (ymax - ymin) / (xmax - xmin))))
    y_edges = np.linspace(ymin, ymax, ny + 1)
    x_edges = np.linspace(xmin, xmax, nx + 1)
    cell_x = (xmax - xmin) / nx
    cell_y = (ymax - ymin) / ny
    sigma = (max(bandwidth_m / cell_y, 1.0), max(bandwidth_m / cell_x, 1.0))

    rate_sum = np.zeros((ny, nx), dtype=float)
    support_n = np.zeros((ny, nx), dtype=np.int16)
    participants = sorted(data["ParticipantID"].astype(str).unique())
    for participant in participants:
        part = data.loc[data["ParticipantID"].astype(str).eq(participant)]
        valid_hist, _, _ = np.histogram2d(
            part["PlotY"], part["PlotX"], bins=[y_edges, x_edges], weights=part["ValidDuration_s"]
        )
        gaze_hist, _, _ = np.histogram2d(
            part["PlotY"], part["PlotX"], bins=[y_edges, x_edges], weights=part["GazeDuration_s"]
        )
        valid_smooth = gaussian_filter(valid_hist, sigma=sigma, mode="constant")
        gaze_smooth = gaussian_filter(gaze_hist, sigma=sigma, mode="constant")
        local_support = valid_smooth > max(float(valid_smooth.max()) * 0.01, 1e-7)
        rate = np.divide(
            gaze_smooth,
            valid_smooth,
            out=np.zeros_like(gaze_smooth),
            where=local_support,
        )
        rate_sum[local_support] += np.clip(rate[local_support], 0.0, 1.0)
        support_n[local_support] += 1

    field = np.divide(rate_sum, support_n, out=np.full_like(rate_sum, np.nan), where=support_n > 0)
    min_support = max(2, int(math.ceil(len(participants) * 0.08)))
    field[support_n < min_support] = np.nan

    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    xx, yy = np.meshgrid(x_centres, y_centres)
    inside = contains_xy(floor_footprint(zones, floor).buffer(0.15), xx, yy)
    field[~inside] = np.nan
    return {
        "field": field,
        "support_n": support_n,
        "extent": extent,
        "participant_n": len(participants),
        "min_support": min_support,
    }


def build_surfaces(
    points: pd.DataFrame, zones: pd.DataFrame, bandwidth_m: float
) -> tuple[dict, dict]:
    surfaces: dict[tuple[str, str], dict] = {}
    scales: dict[str, float] = {}
    for smoke in SMOKE_ORDER:
        for guidance in GUIDANCE_ORDER:
            scenario = f"{guidance}-{smoke}"
            fields = []
            for floor in FLOOR_ORDER:
                part = points.loc[
                    points["Scenario"].eq(scenario) & points["Floor"].eq(floor)
                ].copy()
                surface = participant_balanced_surface(
                    part, zones, floor, geometry_extent(zones, floor), bandwidth_m
                )
                surfaces[(scenario, floor)] = surface
                finite = surface["field"][np.isfinite(surface["field"]) & (surface["field"] > 0)]
                if len(finite):
                    fields.append(finite)
            combined = np.concatenate(fields) if fields else np.asarray([1.0])
            high = float(np.quantile(combined, ROBUST_HIGH_QUANTILE))
            scales[scenario] = max(high, 1e-6)
            print(f"{scenario}: within-group P98={scales[scenario]:.4f}")
    return surfaces, scales


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms


def draw_zone_frames(ax: plt.Axes, zones: pd.DataFrame, floor: str) -> None:
    rows = zones.loc[zones["asset_layer"].eq(floor)].sort_values(
        ["priority", "canonical_order", "zone_id"]
    )
    for row in rows.itertuples(index=False):
        colour = ZONE_COLORS.get(str(row.zone_type), "#56616C")
        for polygon in polygon_parts(row.geometry):
            x, y = polygon.exterior.xy
            # A light halo keeps every semantic colour readable over hot areas.
            ax.plot(x, y, color="white", lw=2.60, alpha=0.92, zorder=4, solid_joinstyle="round")
            ax.plot(x, y, color=colour, lw=1.55, alpha=1.00, zorder=5, solid_joinstyle="round")
            for interior in polygon.interiors:
                hx, hy = interior.xy
                ax.plot(hx, hy, color="white", lw=2.60, alpha=0.92, zorder=4)
                ax.plot(hx, hy, color=colour, lw=1.55, alpha=1.00, zorder=5)


def draw_target_exit(
    ax: plt.Axes, zones: pd.DataFrame, floor: str, target_exit: str
) -> None:
    """Mark the physical target only where an exit exists: on the Hall plan."""
    if floor != "Hall":
        return
    target = zones.loc[zones["zone_id"].eq(f"Exit_{target_exit}")]
    if len(target) != 1:
        raise ValueError(f"Cannot resolve target-exit zone: Exit_{target_exit}")
    centroid = target.iloc[0]["geometry"].centroid
    ax.scatter(
        [centroid.x],
        [centroid.y],
        marker="*",
        s=88,
        facecolor="#FFD447",
        edgecolor="#18212A",
        linewidth=0.85,
        zorder=7,
        clip_on=False,
    )


def draw_panel(
    ax: plt.Axes,
    zones: pd.DataFrame,
    surface: dict,
    group_high: float,
    floor: str,
    target_exit: str,
) -> None:
    extent = surface["extent"]
    background_extent = calibrated_background_extent(extent, floor)
    background_artist = ax.imshow(
        background_image(floor),
        extent=background_extent,
        origin="upper",
        interpolation="lanczos",
        alpha=0.76,
        zorder=0,
    )
    rotation = float(BACKGROUND_CALIBRATION[floor]["rotation_deg"])
    if abs(rotation) > 1e-12:
        centre_x = (background_extent[0] + background_extent[1]) / 2.0
        centre_y = (background_extent[2] + background_extent[3]) / 2.0
        # CSS uses clockwise-positive screen rotation; Matplotlib data
        # coordinates use counter-clockwise-positive rotation.
        background_artist.set_transform(
            Affine2D().rotate_deg_around(centre_x, centre_y, -rotation) + ax.transData
        )
    field = surface["field"]
    relative = np.clip(field / group_high, 0.0, 1.0)
    support_fraction = surface["support_n"] / max(surface["participant_n"], 1)
    alpha = np.where(
        np.isfinite(relative),
        np.clip(np.power(np.nan_to_num(relative), 0.62) * 0.92, 0.0, 0.92)
        * np.clip(support_fraction / 0.25, 0.45, 1.0),
        0.0,
    )
    ax.imshow(
        relative,
        extent=extent,
        origin="lower",
        cmap=HEAT_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=alpha,
        interpolation="bilinear",
        zorder=2,
        rasterized=True,
    )
    draw_zone_frames(ax, zones, floor)
    draw_target_exit(ax, zones, floor, target_exit)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def legend_handles() -> list[Line2D]:
    legend_types = [
        "Platform",
        "Spawn",
        "StairEntrance",
        "HallInsideGate",
        "Gate",
        "HallOutsideGate",
        "DecisionZone",
        "ExitCorridor",
        "BranchCorridor",
        "Exit",
    ]
    handles = [
        Line2D([0], [0], color=ZONE_COLORS[name], lw=2.0, label=ZONE_LABELS[name])
        for name in legend_types
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#FFD447",
            markeredgecolor="#18212A",
            markersize=7.5,
            label="Target exit",
        )
    )
    return handles


def plot_all(
    points: pd.DataFrame,
    zones: pd.DataFrame,
    surfaces: dict,
    scales: dict,
    target_exits: dict[str, str],
    output: Path,
    dpi: int,
) -> Path:
    configure_style()
    # Both source plan images are approximately 1:2, so a tall page preserves
    # equal map scale while keeping all 16 panels legible.
    fig, axes = plt.subplots(4, 4, figsize=(10.8, 20.0), squeeze=False)
    row_keys = [
        ("LowSmoke", "Hall"),
        ("LowSmoke", "Platform"),
        ("HighSmoke", "Hall"),
        ("HighSmoke", "Platform"),
    ]
    panel_index = 0
    for row_index, (smoke, floor) in enumerate(row_keys):
        for column_index, guidance in enumerate(GUIDANCE_ORDER):
            ax = axes[row_index, column_index]
            scenario = f"{guidance}-{smoke}"
            target_exit = target_exits[scenario]
            draw_panel(
                ax,
                zones,
                surfaces[(scenario, floor)],
                scales[scenario],
                floor,
                target_exit,
            )
            panel_index += 1
            ax.text(
                0.02,
                0.985,
                f"({chr(96 + panel_index)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                fontweight="bold",
                color="#111827",
                zorder=8,
            )
            participant_n = int(
                points.loc[points["Scenario"].eq(scenario), "ParticipantID"].nunique()
            )
            if row_index in {0, 2}:
                title = GUIDANCE_LABELS[guidance] if row_index == 0 else ""
                if row_index == 0:
                    ax.set_title(title, pad=4, fontweight="bold")
            panel_note = f"n={participant_n} · Target {target_exit}" if floor == "Hall" else f"Target {target_exit}"
            ax.text(
                0.98,
                0.985,
                panel_note,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.8,
                color="#26313B",
                zorder=8,
            )
        axes[row_index, 0].text(
            -0.13,
            0.5,
            ROW_LABELS[(smoke, floor)],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color="#18212A",
        )

    cax = fig.add_axes([0.35, 0.045, 0.30, 0.009])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=HEAT_CMAP),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.ax.tick_params(labelsize=7, length=0, pad=2)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label("Within-group relative acquisition intensity", fontsize=7.5, labelpad=2)

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        ncol=6,
        frameon=False,
        fontsize=6.6,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.992, bottom=0.065, wspace=0.035, hspace=0.045)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compress_level": 7},
    )
    plt.close(fig)
    return output


def plot_individual_panels(
    points: pd.DataFrame,
    zones: pd.DataFrame,
    surfaces: dict,
    scales: dict,
    target_exits: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Export the 16 component panels as transparent-background PNG assets."""
    configure_style()
    outputs: list[Path] = []
    row_keys = [
        ("LowSmoke", "Hall"),
        ("LowSmoke", "Platform"),
        ("HighSmoke", "Hall"),
        ("HighSmoke", "Platform"),
    ]
    number = 2
    for smoke, floor in row_keys:
        for guidance in GUIDANCE_ORDER:
            scenario = f"{guidance}-{smoke}"
            target_exit = target_exits[scenario]
            participant_n = int(
                points.loc[points["Scenario"].eq(scenario), "ParticipantID"].nunique()
            )
            fig, ax = plt.subplots(figsize=(3.0, 6.0))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            draw_panel(
                ax,
                zones,
                surfaces[(scenario, floor)],
                scales[scenario],
                floor,
                target_exit,
            )
            panel_note = (
                f"n={participant_n} · Target {target_exit}"
                if floor == "Hall"
                else f"Target {target_exit}"
            )
            ax.text(
                0.98,
                0.992,
                panel_note,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#26313B",
                zorder=8,
            )
            fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
            output = output_dir / (
                f"{number:02d}_{guidance}_{smoke}_{floor}_Transparent.png"
            )
            fig.savefig(
                output,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.01,
                transparent=True,
                facecolor="none",
                edgecolor="none",
                pil_kwargs={"compress_level": 7},
            )
            plt.close(fig)
            outputs.append(output)
            number += 1
    return outputs


def plot_legend_svg(output: Path) -> Path:
    """Export the heat scale, semantic-zone keys and target symbol as SVG."""
    configure_style()
    fig = plt.figure(figsize=(10.8, 1.35))
    fig.patch.set_alpha(0.0)
    cax = fig.add_axes([0.35, 0.66, 0.30, 0.11])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=HEAT_CMAP),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.ax.tick_params(labelsize=8, length=0, pad=2)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label(
        "Within-group relative acquisition intensity", fontsize=8.5, labelpad=2
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=6,
        frameon=False,
        fontsize=8.0,
        handlelength=2.0,
        columnspacing=1.35,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600 dpi)")
    parser.add_argument(
        "--bandwidth-m",
        type=float,
        default=DEFAULT_BANDWIDTH_M,
        help="Gaussian spatial bandwidth in Unity metres (default: 2.0)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zones = load_zones()
    points = load_guidance_points(zones)
    print("Mapped samples by experimental condition and floor:")
    print(
        points.groupby(["Scenario", "Floor"], observed=True)
        .agg(
            participants=("ParticipantID", "nunique"),
            valid_duration_s=("ValidDuration_s", "sum"),
            gaze_duration_s=("GazeDuration_s", "sum"),
        )
        .round(3)
        .to_string()
    )
    surfaces, scales = build_surfaces(points, zones, args.bandwidth_m)
    target_exits = load_target_exits()
    print(f"Target exits: {target_exits}")
    output = plot_all(points, zones, surfaces, scales, target_exits, args.output, args.dpi)
    print(f"Saved: {output}")
    panel_outputs = plot_individual_panels(
        points,
        zones,
        surfaces,
        scales,
        target_exits,
        args.output.parent,
        args.dpi,
    )
    for panel_output in panel_outputs:
        print(f"Saved: {panel_output}")
    legend_output = plot_legend_svg(
        args.output.parent / "18_Guidance_Information_Acquisition_Legend.svg"
    )
    print(f"Saved: {legend_output}")


if __name__ == "__main__":
    main()
