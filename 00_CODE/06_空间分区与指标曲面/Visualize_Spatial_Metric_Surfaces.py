"""Render individual and overlaid timestamp-local 3D spatial metrics.

The program is a plotting consumer, not a feature-extraction pipeline. All four
metrics are read from the self-contained 2 m participant-cell table stored
under ``Results_Spatial_Metric_Surfaces/_timestamp_input``. Saccades use the
EyeIndicatorAssessment.py detector once per complete trial and are assigned
once to the 2 m cell occupied at event onset.

Edit ``METRIC_SWITCHES`` and the two render switches below for the simplest
screening workflow, or override them with ``--metrics`` and ``--render-mode``.
By default, three metrics (head search, gaze depth, and gaze scan speed)
are stacked into one persistent
All_Metrics_Overlay panel per scenario/layer, then every panel is assembled
into one final overview figure. Only hall and platform
panels are generated with continuous colour gradients; the stair layer and
discrete bands remain available through explicit command-line options. Only
600 dpi PNG images are written below ``Results_Spatial_Metric_Surfaces``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.patches import PathPatch, Polygon as PolygonPatch
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import shapely
from matplotlib.pyplot import autumn, winter
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from a5_colormaps import PALETTES as A5_PALETTES
from Visualize_Unity_Spatial_Partitions import (
    DEFAULT_CAMERA,
    UnityOrthographicProjector,
    compound_top_path,
    iter_polygons,
    load_zones,
    polygon_true_boundary_segments,
)


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
RESULT_ROOT = ROOT / "Results_Spatial_Metric_Surfaces"
TIMESTAMP_INPUT_ROOT = RESULT_ROOT / "_timestamp_input"
TIMESTAMP_PARTICIPANT_CELL_FILE = (
    TIMESTAMP_INPUT_ROOT / "participant_cell_metrics.csv.gz"
)
BASEMAP_ROOT = ROOT / "Assets_Spatial_Metric_Surfaces" / "Basemaps"
GAZE_DEPTH_POINT_ROOT = ROOT / "Results_SpatialSlice_GazeDepth"
VISUAL_INFORMATION_DATA = (
    ROOT / "Results_Guidance_Spatial_Causal_Analysis" / "data"
)
VISUAL_INFORMATION_PROGRESS_FILE = (
    VISUAL_INFORMATION_DATA / "visual_information_progress_bins.csv"
)
TIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
SCENE_BASEMAPS = {
    "hall": BASEMAP_ROOT / "Concourse_Unity_Camera.png",
    "platform": BASEMAP_ROOT / "Platform_Unity_Camera.png",
}
OUTPUT_DPI = 600
# Preserve the approved 1920×1080 layout at its original 300 dpi physical size;
# exporting at 600 dpi therefore produces a true 3840×2160 raster.
LAYOUT_DPI = 300

GUIDANCES = ("StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full")
SMOKE_LEVELS = ("LowSmoke", "HighSmoke")
SCENARIOS = tuple(
    f"{guidance}-{smoke}"
    for smoke in SMOKE_LEVELS
    for guidance in GUIDANCES
)
GUIDANCE_LABELS = {
    "StaticSign": "StaticSign",
    "ArrowRibbon": "Arrow",
    "MiniMap": "MiniMap",
    "TextBar": "TextBar",
    "Full": "Multimodal",
}
SMOKE_LABELS = {"LowSmoke": "Low smoke", "HighSmoke": "High smoke"}
SCENARIO_LABELS = {
    scenario: (
        f"{GUIDANCE_LABELS[scenario.rsplit('-', 1)[0]]} · "
        f"{SMOKE_LABELS[scenario.rsplit('-', 1)[1]].lower()}"
    )
    for scenario in SCENARIOS
}
SUPPORTED_LAYERS = ("platform", "stair", "hall")
DEFAULT_LAYERS = ("platform", "hall")
LAYER_LABELS = {"platform": "Platform", "stair": "Stair", "hall": "Hall"}

CELL_M = 2.0
GRID_M = 1.15

# ---------------------------------------------------------------------------
# SURFACE INTERPOLATION — Gaussian kernel regression preserves metric units.
# Increase the bandwidth for broader/smoother peaks; decrease it for sharper
# local peaks. Set SURFACE_METHOD to "idw" to reproduce the earlier method.
# ---------------------------------------------------------------------------
SURFACE_METHOD = "gaussian_kernel"
GAUSSIAN_KERNEL_BANDWIDTH_M = 4.0
GAUSSIAN_KERNEL_TRUNCATE_SIGMA = 3.0

IDW_POWER = 2.0
IDW_SMOOTH_M = 1.25
IDW_NEIGHBOURS = 12
MAX_SUPPORT_DISTANCE_M = {"platform": 11.0, "stair": 7.0, "hall": 12.0}
VIEWPORT_Y_SHIFT_PX = {"platform": 60.0, "stair": 60.0, "hall": 65.0}
# ---------------------------------------------------------------------------
# User-editable screening switches. Set a metric to False to suppress it.
# Command-line ``--metrics`` / ``--render-mode`` take precedence when supplied.
# ---------------------------------------------------------------------------
METRIC_SWITCHES = {
    "head_search": True,
    "gaze_depth": True,
    "gaze_scan_speed": True,
    "saccade_rate": False,
}
# Default run: stack three metrics; saccade remains an explicit opt-in.
RENDER_INDIVIDUAL_SURFACES = False
RENDER_ALL_METRICS_OVERLAY = True

# Colour limits are independent of data clipping and geometric height limits.
# Calibrate on all ten scenarios and both default floors, even for subset runs.
COLOUR_LIMIT_MODE = "percentile"  # percentile / max_fraction / original
COLOUR_UPPER_PERCENTILE = 98.0
COLOUR_MAX_FRACTION = 0.8  # fraction of the final surface maximum, per metric


COMBINED_RELIEF_HEIGHT_M = 8.0
BASEMAP_TO_FIRST_SURFACE_GAP_M = 2.0
COMBINED_PROJECTED_GAP_PX = 18.0
COMBINED_APPROVED_COMPACT_GAP_PX = 30.0
COMBINED_LAYER_OFFSET_SCALE = 0.5
COMBINED_SURFACE_ALPHA = 0.94
# Within each selected spatial layer, use the largest initially calculated
# interval from every panel for every adjacent pair of stacked surfaces.
UNIFORM_OVERLAY_INTERVAL_LAYERS = ("platform", "hall")
STAIR_OVERLAY_CONNECTOR_COUNT = 2
STAIR_OVERLAY_CONNECTOR_COLOUR = "#B2B8C0"
COMBINED_TOP_MARGIN_PX = 105.0
COMBINED_BOTTOM_MARGIN_PX = 90.0
COMBINED_MIN_CANVAS_HEIGHT_PX = 1080

# Final overview sheet: five guidance columns and one row per smoke/layer pair.
OVERLAY_SHEET_PANEL_WIDTH_PX = 1050
OVERLAY_SHEET_COLUMN_GAP_PX = 4
OVERLAY_SHEET_ROW_GAP_PX = 4
OVERLAY_SHEET_LEFT_MARGIN_PX = 330
OVERLAY_SHEET_RIGHT_MARGIN_PX = 30
OVERLAY_SHEET_TOP_MARGIN_PX = 155
OVERLAY_SHEET_BOTTOM_MARGIN_PX = 35
OVERLAY_SHEET_PANEL_CROP_PADDING_PX = 24
OVERLAY_SHEET_LEGEND_TOP_GAP_PX = 45
OVERLAY_SHEET_LEGEND_HEIGHT_PX = 215
OVERLAY_SHEET_LEGEND_COLUMN_GAP_PX = 55

# ---------------------------------------------------------------------------
# COLOUR-PALETTE INPUT — enter an a5_colormaps.py number from 1 to 183.
# Numbers 1–96 are the original/user palettes. Numbers 97–183 are the complete
# Matplotlib 3.9.2 base palette set; see MATPLOTLIB_NAME_TO_NUMBER in
# a5_colormaps.py for the name-to-number table. Use the switches below for
# reversed variants.
# ---------------------------------------------------------------------------
METRIC_PALETTE_NUMBERS = {
    "head_search": 162,
    "gaze_depth": 99,
    "gaze_scan_speed": 100,
    "saccade_rate": 157,
}
REVERSE_ALL_METRIC_PALETTES = False
METRIC_PALETTE_REVERSED = {
    "head_search": False,
    "gaze_depth": False,
    "gaze_scan_speed": False,
    "saccade_rate": False,
}


def a5_palette(number: int) -> tuple[str, ...]:
    """Resolve one numeric colour-palette selection from a5_colormaps.py."""
    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError(
            f"Palette selection must be an integer from 1 to {len(A5_PALETTES)}"
        )
    name = f"a5_{number:02d}"
    if name not in A5_PALETTES:
        available = ", ".join(item.removeprefix("a5_") for item in A5_PALETTES)
        raise KeyError(f"Palette {name!r} is unavailable; choose from: {available}")
    return tuple(A5_PALETTES[name])


ACTIVE_METRIC_PALETTES = {
    metric: (
        a5_palette(number)[::-1]
        if REVERSE_ALL_METRIC_PALETTES or METRIC_PALETTE_REVERSED[metric]
        else a5_palette(number)
    )
    for metric, number in METRIC_PALETTE_NUMBERS.items()
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    field: str
    source: str
    label: str
    unit: str
    value_min: float
    value_max: float
    value_ticks: tuple[float, ...]
    world_height_max: float
    palette_stops: tuple[str, ...]
    definition: str


METRIC_SPECS = {
    "head_search": MetricSpec(
        key="head_search",
        field="HeadAngularPathRate_deg_s",
        source="timestamp_participant_cells",
        label="Head search",
        unit="°/s",
        value_min=0.0,
        value_max=50.0,
        value_ticks=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
        world_height_max=12.0,
        palette_stops=ACTIVE_METRIC_PALETTES["head_search"],
        definition="Head-orientation angular path accumulated per valid second.",
    ),
    "gaze_depth": MetricSpec(
        key="gaze_depth",
        field="GazeDepth_m",
        source="timestamp_participant_cells",
        label="Gaze depth",
        unit="m",
        value_min=0.0,
        value_max=25.0,
        value_ticks=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
        world_height_max=12.0,
        palette_stops=ACTIVE_METRIC_PALETTES["gaze_depth"],
        definition="Distance from participant/eye position to the first valid gaze-ray hit.",
    ),
    "gaze_scan_speed": MetricSpec(
        key="gaze_scan_speed",
        field="EyeGazeScanpathRate_deg_s",
        source="timestamp_participant_cells",
        label="Eye scan speed",
        unit="°/s",
        value_min=0.0,
        value_max=75.0,
        value_ticks=(0.0, 15.0, 30.0, 45.0, 60.0, 75.0),
        world_height_max=12.0,
        palette_stops=ACTIVE_METRIC_PALETTES["gaze_scan_speed"],
        definition="Eye-in-head angular scanpath accumulated per valid second.",
    ),
    "saccade_rate": MetricSpec(
        key="saccade_rate",
        field="SaccadeRate_events_min",
        source="timestamp_participant_cells",
        label="Saccade rate",
        unit="events/min",
        value_min=0.0,
        value_max=240.0,
        value_ticks=(0.0, 60.0, 120.0, 180.0, 240.0),
        world_height_max=12.0,
        palette_stops=ACTIVE_METRIC_PALETTES["saccade_rate"],
        definition=(
            "EyeIndicatorAssessment.py Hit1 detector run once per complete trial; "
            "each event is assigned once to the participant's onset 2 m cell."
        ),
    ),
}


@lru_cache(maxsize=1)
def load_timestamp_participant_cells() -> pd.DataFrame:
    """Load and validate the self-contained timestamp-local cell interface."""
    if not TIMESTAMP_PARTICIPANT_CELL_FILE.exists():
        raise FileNotFoundError(
            "Missing timestamp-local metric table: "
            f"{TIMESTAMP_PARTICIPANT_CELL_FILE}. Recreate it with "
            "Results_Spatial_Metric_Surfaces/_timestamp_input/"
            "extract_timestamp_metrics.py or restore the complete input bundle."
        )
    frame = pd.read_csv(TIMESTAMP_PARTICIPANT_CELL_FILE)
    required = {
        "Scenario",
        "Layer",
        "ParticipantID",
        "GridCellX",
        "GridCellZ",
        "PosX",
        "PosY",
        "PosZ",
        "Metric",
        "Value",
        "Exposure_s",
        "Supported",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Timestamp-cell interface is missing columns: {missing}")
    frame["Supported"] = frame["Supported"].astype(str).str.lower().eq("true")
    return frame


def timestamp_participant_cells(
    spec: MetricSpec,
    zones: pd.DataFrame,
    scenarios: tuple[str, ...],
    layers: tuple[str, ...],
) -> pd.DataFrame:
    """Select supported participant cells in the formal surface schema."""
    del zones
    frame = load_timestamp_participant_cells()
    selected = frame.loc[
        frame["Metric"].eq(spec.key)
        & frame["Scenario"].isin(scenarios)
        & frame["Layer"].isin(layers)
        & frame["Supported"]
    ].copy()
    selected = selected.dropna(subset=["Value", "PosX", "PosY", "PosZ"])
    if selected.empty:
        raise ValueError(f"No supported timestamp-local cells for {spec.key}")
    identity = [
        "Scenario", "Layer", "ParticipantID", "GridCellX", "GridCellZ",
    ]
    if selected.duplicated(identity).any():
        raise ValueError(f"Duplicate participant cells for {spec.key}")
    numeric = selected[["Value", "PosX", "PosY", "PosZ"]]
    if not np.isfinite(numeric).all().all():
        raise ValueError(f"Non-finite values remain for {spec.key}")
    return selected[
        identity + ["PosX", "PosY", "PosZ", "Value"]
    ].copy()


def nice_metric_limits(values: pd.Series) -> tuple[float, float, tuple[float, ...]]:
    """Use a robust supported range with rounded endpoints and readable ticks."""
    finite = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values available for colour limits")
    observed_min = float(np.min(finite))
    observed_max = float(np.max(finite))
    lower_quantile, upper_quantile = np.quantile(finite, [0.01, 0.99])
    if upper_quantile <= lower_quantile:
        pad = max(abs(observed_max) * 0.05, 1.0)
        lower, upper = observed_min - pad, observed_max + pad
    else:
        lower = 0.0 if observed_min >= 0 else float(lower_quantile)
        span = float(upper_quantile) - lower
        step = (
            1.0 if span <= 10
            else 5.0 if span <= 50
            else 10.0 if span <= 150
            else 25.0
        )
        upper = float(np.ceil(float(upper_quantile) / step) * step)
    ticks = tuple(float(value) for value in np.linspace(lower, upper, 6))
    return lower, upper, ticks


def configure_timestamp_metric_specs(frame: pd.DataFrame) -> dict[str, MetricSpec]:
    """Derive shared metric limits from every supported participant cell."""
    configured: dict[str, MetricSpec] = {}
    for key, spec in METRIC_SPECS.items():
        values = frame.loc[frame["Metric"].eq(key) & frame["Supported"], "Value"]
        lower, upper, ticks = nice_metric_limits(values)
        configured[key] = replace(
            spec,
            value_min=lower,
            value_max=upper,
            value_ticks=ticks,
        )
    return configured


def calibrate_saccade_scale_from_visible_surfaces(
    zones: pd.DataFrame,
) -> dict[str, object]:
    """Calibrate the saccade legend from the participant-balanced 4 m surface."""
    global METRIC_SPECS
    spec = METRIC_SPECS["saccade_rate"]
    frame = load_timestamp_participant_cells()
    raw = frame.loc[
        frame["Metric"].eq("saccade_rate") & frame["Supported"], "Value"
    ]
    raw_max = float(pd.to_numeric(raw, errors="coerce").max())
    wide_spec = replace(
        spec,
        value_min=0.0,
        value_max=max(raw_max, 1.0),
        value_ticks=(0.0, max(raw_max, 1.0)),
    )
    _, surface, _ = build_runtime_surface_data(
        zones=zones,
        spec=wide_spec,
        scenarios=SCENARIOS,
        layers=DEFAULT_LAYERS,
        colour_scale_mode="continuous",
    )
    visible = pd.to_numeric(surface["Value"], errors="coerce")
    # Work in events/s, then scale the complete legend to events/min. This
    # preserves the approved appearance while presenting the requested unit.
    base_lower, base_upper, base_ticks = nice_metric_limits(visible / 60.0)
    lower = base_lower * 60.0
    upper = base_upper * 60.0
    ticks = tuple(value * 60.0 for value in base_ticks)
    METRIC_SPECS["saccade_rate"] = replace(
        spec,
        value_min=lower,
        value_max=upper,
        value_ticks=ticks,
    )
    finite = visible[np.isfinite(visible)]
    return {
        "basis": (
            "all finite nodes in final participant-balanced 4 m Gaussian "
            "rate surfaces"
        ),
        "finite_nodes": int(len(finite)),
        "visible_min": float(finite.min()),
        "visible_max": float(finite.max()),
        "visible_p99": float(finite.quantile(0.99)),
        "unit": spec.unit,
        "unit_scale_from_events_s": 60.0,
        "colour_min": lower,
        "colour_max": upper,
    }


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "metric"


def configure_surface_colour_limits(config, surface, *, mode, percentile, fraction):
    """Calibrate colour only; retain original values, heights and contour levels."""
    if mode not in {"percentile", "max_fraction", "original"}:
        raise ValueError(f"Unknown colour limit mode: {mode}")
    if not 0 < percentile <= 100 or not 0 < fraction <= 1:
        raise ValueError("Colour percentile must be in (0, 100]; fraction in (0, 1]")
    values = surface["Value"].to_numpy(float)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite surface values for colour calibration")
    lower = float(config["value_min"])
    observed_max = float(values.max())
    if mode == "original":
        upper = float(config["value_max"])
        ticks = list(config["value_ticks"])
    elif mode == "max_fraction":
        upper = observed_max * fraction
        if upper <= lower:
            raise ValueError("Fractional colour maximum must exceed the lower limit")
        ticks = np.linspace(lower, upper, 6).tolist()
    else:
        raw_upper = float(np.percentile(values, percentile))
        span = raw_upper - lower
        if span <= 0:
            upper = float(config["value_max"])
        else:
            # Round up to readable tick endpoints without excluding extra values.
            magnitude = 10.0 ** np.floor(np.log10(span))
            step = magnitude / 2.0
            upper = min(float(config["value_max"]),
                        lower + float(np.ceil(span / step) * step))
        ticks = np.linspace(lower, upper, 6).tolist()
    if not np.isfinite(upper) or upper <= lower:
        raise ValueError("Invalid calibrated colour range")
    config.update(colour_min=lower, colour_max=upper, colour_ticks=ticks,
                  colour_saturated=bool(observed_max > upper))
    panels = []
    for (scenario, layer), part in surface.groupby(["Scenario", "Layer"]):
        v = part["Value"].to_numpy(float)
        v = v[np.isfinite(v)]
        panels.append({"scenario": scenario, "layer": layer,
                       "maximum": float(v.max()) if v.size else None,
                       "saturated_fraction": float(np.mean(v > upper)) if v.size else None})
    return {"mode": mode, "percentile": percentile, "max_fraction": fraction,
            "reference": "all scenarios, platform and hall",
            "weighting": "each finite final surface grid vertex has equal weight",
            "colour_min": lower, "colour_max": upper,
            "original_value_max": float(config["value_max"]),
            "surface_max": observed_max,
            "surface_quantiles": {str(q): float(np.percentile(values, q))
                                  for q in (50, 90, 95, 98, 99)},
            "saturated_fraction": float(np.mean(values > upper)),
            "panels": panels}


def colour_tick_labels(ticks, saturated):
    labels = [f"{float(tick):.4g}" for tick in ticks]
    if saturated:
        labels[-1] = "≥" + labels[-1]
    return labels


def metric_cmap_norm(config: dict, name: str):
    """Build either continuous colours or bands aligned with the metric ticks."""
    boundaries = np.asarray(config.get("colour_ticks", config["value_ticks"]), dtype=float)
    if len(boundaries) < 2 or np.any(np.diff(boundaries) <= 0):
        raise ValueError(f"Invalid colour-band boundaries for {config['key']}")
    continuous = LinearSegmentedColormap.from_list(
        f"{name}_continuous",
        config["palette_stops"],
        N=256,
    )
    if config.get("colour_scale_mode", "banded") == "continuous":
        norm = Normalize(
            vmin=float(config.get("colour_min", config["value_min"])),
            vmax=float(config.get("colour_max", config["value_max"])),
            clip=True,
        )
        return continuous, norm, None
    value_min = float(boundaries[0])
    value_max = float(boundaries[-1])
    midpoint_values = 0.5 * (boundaries[:-1] + boundaries[1:])
    midpoint_positions = (midpoint_values - value_min) / (value_max - value_min)
    cmap = ListedColormap(
        continuous(midpoint_positions),
        name=f"{name}_banded",
    )
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    return cmap, norm, boundaries


def save_png_with_retry(fig, target: Path) -> None:
    """Retry transient Windows file-open failures while overwriting PNG output."""
    for attempt in range(3):
        try:
            fig.savefig(
                target,
                dpi=OUTPUT_DPI,
                transparent=False,
                facecolor="white",
                edgecolor="none",
                bbox_inches=None,
                pad_inches=0,
                metadata={"Software": "Visualize_Spatial_Metric_Surfaces.py"},
            )
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.35 * (attempt + 1))


def save_pil_png_with_retry(image: Image.Image, target: Path) -> None:
    """Encode beside the destination, then replace it with transient-lock retries."""
    temporary = target.with_name(f".{target.stem}.writing.png")
    metadata = PngInfo()
    for key, value in image.info.items():
        if isinstance(value, str):
            metadata.add_text(key, value)
    try:
        image.save(
            temporary,
            format="PNG",
            dpi=(OUTPUT_DPI, OUTPUT_DPI),
            optimize=True,
            pnginfo=metadata,
        )
        for attempt in range(5):
            try:
                temporary.replace(target)
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.4 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 13.0,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "path.simplify": False,
        }
    )


def layer_from_zone_type(zone_type: str) -> str:
    if zone_type == "Stair":
        return "stair"
    if zone_type in {"Platform", "Spawn", "StairEntrance"}:
        return "platform"
    return "hall"


def parse_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def load_direct_metric_points(
    metric: str,
    zones: pd.DataFrame,
    scenarios: tuple[str, ...],
) -> pd.DataFrame:
    """Attach Extract_Visual_Information_Features progress-bin values to positions."""
    progress_file = Path(VISUAL_INFORMATION_PROGRESS_FILE)
    trials_file = Path(VISUAL_INFORMATION_DATA) / "analysis_trials.csv"
    for source in (progress_file, trials_file):
        if not source.exists():
            raise FileNotFoundError(source)

    progress_header = pd.read_csv(progress_file, nrows=0)
    required_progress = {
        "Scenario",
        "ParticipantID",
        "FileName",
        "TrialProgressBin",
        metric,
    }
    missing = sorted(required_progress - set(progress_header.columns))
    if missing:
        raise ValueError(
            f"{progress_file.name} does not expose the requested metric/interface columns: {missing}"
        )
    keys = ["Scenario", "ParticipantID", "FileName"]
    lookup = pd.read_csv(
        progress_file,
        usecols=keys + ["TrialProgressBin", metric],
        low_memory=False,
    )
    lookup["TrialProgressBin"] = pd.to_numeric(
        lookup["TrialProgressBin"], errors="coerce"
    ).astype("Int64")
    lookup[metric] = pd.to_numeric(lookup[metric], errors="coerce")
    lookup = lookup.dropna(subset=["TrialProgressBin", metric]).copy()
    lookup = lookup.loc[lookup["Scenario"].isin(scenarios)].copy()
    if lookup.duplicated(keys + ["TrialProgressBin"]).any():
        raise ValueError("Visual-information progress interface contains duplicate trial/bin rows")

    trials = pd.read_csv(trials_file, low_memory=False)
    required_trials = {
        *keys,
        "StartTime",
        "EndTime",
        "SubjectFolder",
        "AnalysisIncluded",
    }
    missing = sorted(required_trials - set(trials.columns))
    if missing:
        raise ValueError(f"{trials_file.name} is missing columns: {missing}")
    trials = trials.loc[parse_bool(trials["AnalysisIncluded"])].copy()
    trial_keys = lookup[keys].drop_duplicates()
    trials = trials.merge(trial_keys, on=keys, how="inner", validate="one_to_one")
    absent = sorted(set(scenarios) - set(trials["Scenario"].astype(str)))
    if absent:
        raise ValueError(f"Direct visual-information interface lacks scenarios: {absent}")

    zone_records = []
    for zone in zones.sort_values("priority", ascending=False).itertuples(index=False):
        parts = json.loads(str(zone.collider_parts_xz))
        zone_records.append(
            (
                str(zone.zone_id),
                layer_from_zone_type(str(zone.zone_type)),
                float(zone.y_min),
                float(zone.y_max),
                tuple(
                    (
                        float(part["xmin"]),
                        float(part["xmax"]),
                        float(part["zmin"]),
                        float(part["zmax"]),
                    )
                    for part in parts
                ),
            )
        )

    position_parts: list[pd.DataFrame] = []
    missing_trajectories: list[str] = []
    for trial in trials.itertuples(index=False):
        scenario = str(trial.Scenario)
        participant = str(trial.ParticipantID)
        subject_folder = str(trial.SubjectFolder)
        filename = str(trial.FileName)
        path = RAW / scenario / subject_folder / filename
        if not path.exists():
            fallback = RAW / scenario / participant / filename
            path = fallback if fallback.exists() else path
        if not path.exists():
            missing_trajectories.append(str(path))
            continue

        pos = pd.read_csv(
            path,
            usecols=["TimeStamp", "PosX", "PosY", "PosZ"],
            low_memory=False,
        )
        pos["ParsedTime"] = pd.to_datetime(
            pos["TimeStamp"], format=TIME_FORMAT, errors="coerce"
        )
        for column in ("PosX", "PosY", "PosZ"):
            pos[column] = pd.to_numeric(pos[column], errors="coerce")
        start = pd.to_datetime(str(trial.StartTime), format=TIME_FORMAT, errors="coerce")
        end = pd.to_datetime(str(trial.EndTime), format=TIME_FORMAT, errors="coerce")
        if pd.isna(start) or pd.isna(end) or end <= start:
            raise ValueError(f"Invalid trial interval: {scenario}/{participant}/{filename}")
        pos = pos.dropna(subset=["ParsedTime", "PosX", "PosY", "PosZ"])
        pos = (
            pos.loc[pos["ParsedTime"].between(start, end)]
            .sort_values("ParsedTime")
            .drop_duplicates("ParsedTime")
        )
        if pos.empty:
            continue
        elapsed = (pos["ParsedTime"] - start).dt.total_seconds()
        second = np.floor(elapsed).astype(np.int64)
        pos = pos.loc[second.ne(second.shift())].copy()
        duration = max((end - start).total_seconds(), 1e-6)
        fraction = ((pos["ParsedTime"] - start).dt.total_seconds() / duration).clip(
            0.0, 0.999999
        )
        pos["TrialProgressBin"] = (np.floor(fraction * 10).astype(int) + 1)

        px = pos["PosX"].to_numpy(float)
        py = pos["PosY"].to_numpy(float)
        pz = pos["PosZ"].to_numpy(float)
        zone_ids = np.full(len(pos), "Unassigned", dtype=object)
        layers = np.full(len(pos), "Unassigned", dtype=object)
        for zone_id, layer, y_min, y_max, rectangles in zone_records:
            xz_hit = np.zeros(len(pos), dtype=bool)
            for x_min, x_max, z_min, z_max in rectangles:
                xz_hit |= (
                    (px >= x_min)
                    & (px <= x_max)
                    & (pz >= z_min)
                    & (pz <= z_max)
                )
            hit = (
                (zone_ids == "Unassigned")
                & xz_hit
                & (py >= y_min)
                & (py <= y_max)
            )
            zone_ids[hit] = zone_id
            layers[hit] = layer
        pos["PhysicalZone"] = zone_ids
        pos["Layer"] = layers
        pos = pos.loc[pos["PhysicalZone"].ne("Unassigned")].copy()
        pos["Scenario"] = scenario
        pos["ParticipantID"] = participant
        pos["FileName"] = filename
        position_parts.append(
            pos[
                keys
                + [
                    "TrialProgressBin",
                    "PhysicalZone",
                    "Layer",
                    "PosX",
                    "PosY",
                    "PosZ",
                ]
            ]
        )

    if missing_trajectories:
        preview = "\n".join(missing_trajectories[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_trajectories)} raw trajectory files; first paths:\n{preview}"
        )
    if not position_parts:
        raise ValueError("No spatial positions could be matched to the visual-information trials")
    points = pd.concat(position_parts, ignore_index=True)
    points = points.merge(
        lookup,
        on=keys + ["TrialProgressBin"],
        how="inner",
        validate="many_to_one",
    )
    points = points.dropna(subset=[metric]).copy()
    return points


def progress_metric_participant_cells(
    spec: MetricSpec,
    zones: pd.DataFrame,
    scenarios: tuple[str, ...],
    layers: tuple[str, ...],
) -> pd.DataFrame:
    data = load_direct_metric_points(spec.field, zones, scenarios)
    data = data.loc[data["Layer"].isin(layers)].copy()
    data["GridCellX"] = np.floor(data["PosX"] / CELL_M + 0.5).astype(int)
    data["GridCellZ"] = np.floor(data["PosZ"] / CELL_M + 0.5).astype(int)
    return (
        data.groupby(
            ["Scenario", "Layer", "ParticipantID", "GridCellX", "GridCellZ"],
            as_index=False,
            observed=True,
        )
        .agg(
            PosX=("PosX", "mean"),
            PosY=("PosY", "mean"),
            PosZ=("PosZ", "mean"),
            Value=(spec.field, "mean"),
        )
    )


def gaze_depth_participant_cells(
    spec: MetricSpec,
    zones: pd.DataFrame,
    scenarios: tuple[str, ...],
    layers: tuple[str, ...],
) -> pd.DataFrame:
    """Read formal gaze-depth points and reduce them to participant/cell means."""
    if not GAZE_DEPTH_POINT_ROOT.exists():
        raise FileNotFoundError(GAZE_DEPTH_POINT_ROOT)
    progress_file = Path(VISUAL_INFORMATION_PROGRESS_FILE)
    if not progress_file.exists():
        raise FileNotFoundError(progress_file)
    allowed = pd.read_csv(
        progress_file,
        usecols=["Scenario", "ParticipantID"],
        low_memory=False,
    ).drop_duplicates()
    allowed = allowed.loc[allowed["Scenario"].isin(scenarios)].copy()
    allowed_by_scenario = {
        scenario: set(
            allowed.loc[allowed["Scenario"].eq(scenario), "ParticipantID"].astype(str)
        )
        for scenario in scenarios
    }
    zone_layers = {
        str(row.zone_id): layer_from_zone_type(str(row.zone_type))
        for row in zones.itertuples(index=False)
    }
    parts: list[pd.DataFrame] = []
    for scenario in scenarios:
        scenario_dir = GAZE_DEPTH_POINT_ROOT / scenario
        if not scenario_dir.is_dir():
            raise FileNotFoundError(scenario_dir)
        allowed_participants = allowed_by_scenario[scenario]
        for source in sorted(scenario_dir.glob("points_*.csv")):
            zone_id = source.stem.removeprefix("points_")
            layer = zone_layers.get(zone_id)
            if layer not in layers:
                continue
            frame = pd.read_csv(
                source,
                usecols=["x", "y", "z", "gaze_depth", "subject"],
                low_memory=False,
            )
            frame = frame.loc[
                frame["subject"].astype(str).isin(allowed_participants)
            ].copy()
            if frame.empty:
                continue
            for column in ("x", "y", "z", "gaze_depth"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["x", "y", "z", "gaze_depth"])
            if frame.empty:
                continue
            frame["GridCellX"] = np.floor(frame["x"] / CELL_M + 0.5).astype(int)
            frame["GridCellZ"] = np.floor(frame["z"] / CELL_M + 0.5).astype(int)
            frame["ValueSum"] = frame["gaze_depth"]
            reduced = (
                frame.groupby(
                    ["subject", "GridCellX", "GridCellZ"],
                    as_index=False,
                    observed=True,
                )
                .agg(
                    PosXSum=("x", "sum"),
                    PosYSum=("y", "sum"),
                    PosZSum=("z", "sum"),
                    ValueSum=("ValueSum", "sum"),
                    SampleN=("gaze_depth", "size"),
                )
                .rename(columns={"subject": "ParticipantID"})
            )
            reduced.insert(0, "Layer", layer)
            reduced.insert(0, "Scenario", scenario)
            parts.append(reduced)
    if not parts:
        raise ValueError("No gaze-depth points matched the selected scenarios/layers")
    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(
            ["Scenario", "Layer", "ParticipantID", "GridCellX", "GridCellZ"],
            as_index=False,
            observed=True,
        )
        .agg(
            PosXSum=("PosXSum", "sum"),
            PosYSum=("PosYSum", "sum"),
            PosZSum=("PosZSum", "sum"),
            ValueSum=("ValueSum", "sum"),
            SampleN=("SampleN", "sum"),
        )
    )
    combined["PosX"] = combined["PosXSum"] / combined["SampleN"]
    combined["PosY"] = combined["PosYSum"] / combined["SampleN"]
    combined["PosZ"] = combined["PosZSum"] / combined["SampleN"]
    combined["Value"] = combined["ValueSum"] / combined["SampleN"]
    return combined[
        [
            "Scenario",
            "Layer",
            "ParticipantID",
            "GridCellX",
            "GridCellZ",
            "PosX",
            "PosY",
            "PosZ",
            "Value",
        ]
    ]


def participant_balanced_cells(
    participant_cells: pd.DataFrame,
    value_min: float,
    value_max: float,
) -> pd.DataFrame:
    cells = (
        participant_cells.groupby(
            ["Scenario", "Layer", "GridCellX", "GridCellZ"],
            as_index=False,
            observed=True,
        )
        .agg(
            PosX=("PosX", "mean"),
            PosZ=("PosZ", "mean"),
            Value=("Value", "mean"),
            ParticipantSupport=("ParticipantID", "nunique"),
        )
    )
    cells["Value"] = cells["Value"].clip(value_min, value_max)
    return cells


def layer_base_y(
    zones: pd.DataFrame,
    participant_cells: pd.DataFrame,
    layer: str,
) -> float:
    part = zones.loc[zones["asset_layer"].eq(layer)]
    if layer == "stair":
        stair_y = participant_cells.loc[
            participant_cells["Layer"].eq(layer), "PosY"
        ].dropna()
        if not stair_y.empty:
            return float(stair_y.median())
        return float(((part["y_min"] + part["y_max"]) * 0.5).median())
    return float(part["y_max"].median())


def interpolation_grid(footprint):
    """Build the common spatial grid and its footprint mask."""
    min_x, min_z, max_x, max_z = footprint.bounds
    xs = np.arange(min_x, max_x + GRID_M * 0.5, GRID_M)
    zs = np.arange(min_z, max_z + GRID_M * 0.5, GRID_M)
    x_grid, z_grid = np.meshgrid(xs, zs)
    flat = np.column_stack([x_grid.ravel(), z_grid.ravel()])
    inside = shapely.intersects_xy(footprint.buffer(0.10), flat[:, 0], flat[:, 1])
    return x_grid, z_grid, flat, inside


def interpolate_idw(cells: pd.DataFrame, footprint, layer: str):
    """Interpolate the earlier inverse-distance-weighted comparison surface."""
    x_grid, z_grid, flat, inside = interpolation_grid(footprint)
    tree = cKDTree(cells[["PosX", "PosZ"]].to_numpy(float))
    neighbour_count = min(IDW_NEIGHBOURS, len(cells))
    distances, indices = tree.query(flat, k=neighbour_count)
    if neighbour_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    supported = distances[:, 0] <= MAX_SUPPORT_DISTANCE_M[layer]
    weights = 1.0 / np.power(distances + IDW_SMOOTH_M, IDW_POWER)
    local_values = cells["Value"].to_numpy(float)[indices]
    estimate = np.sum(weights * local_values, axis=1) / np.sum(weights, axis=1)
    estimate[~(inside & supported)] = np.nan
    return x_grid, z_grid, estimate.reshape(x_grid.shape)


def interpolate_gaussian_kernel(cells: pd.DataFrame, footprint, layer: str):
    """Estimate a smooth metric field with Nadaraya-Watson Gaussian kernels."""
    bandwidth = float(GAUSSIAN_KERNEL_BANDWIDTH_M)
    truncate_sigma = float(GAUSSIAN_KERNEL_TRUNCATE_SIGMA)
    if bandwidth <= 0:
        raise ValueError("GAUSSIAN_KERNEL_BANDWIDTH_M must be greater than zero")
    if truncate_sigma <= 0:
        raise ValueError("GAUSSIAN_KERNEL_TRUNCATE_SIGMA must be greater than zero")

    x_grid, z_grid, flat, inside = interpolation_grid(footprint)
    sample_points = cells[["PosX", "PosZ"]].to_numpy(float)
    sample_values = cells["Value"].to_numpy(float)
    tree = cKDTree(sample_points)
    kernel_radius = min(
        float(MAX_SUPPORT_DISTANCE_M[layer]),
        truncate_sigma * bandwidth,
    )
    neighbourhoods = tree.query_ball_point(flat, r=kernel_radius)
    estimate = np.full(len(flat), np.nan, dtype=float)
    inverse_two_bandwidth_squared = 0.5 / (bandwidth * bandwidth)

    for grid_index in np.flatnonzero(inside):
        neighbour_indices = neighbourhoods[grid_index]
        if not neighbour_indices:
            continue
        neighbour_indices = np.asarray(neighbour_indices, dtype=int)
        offsets = sample_points[neighbour_indices] - flat[grid_index]
        squared_distances = np.einsum("ij,ij->i", offsets, offsets)
        weights = np.exp(-squared_distances * inverse_two_bandwidth_squared)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            estimate[grid_index] = float(
                np.dot(weights, sample_values[neighbour_indices]) / weight_sum
            )

    return x_grid, z_grid, estimate.reshape(x_grid.shape)


def interpolate_surface(cells: pd.DataFrame, footprint, layer: str):
    """Dispatch to the user-selected metric-surface interpolation method."""
    if SURFACE_METHOD == "gaussian_kernel":
        return interpolate_gaussian_kernel(cells, footprint, layer)
    if SURFACE_METHOD == "idw":
        return interpolate_idw(cells, footprint, layer)
    raise ValueError(
        f"Unknown SURFACE_METHOD {SURFACE_METHOD!r}; choose 'gaussian_kernel' or 'idw'"
    )


def select_postpoints(cells: pd.DataFrame, layer: str) -> pd.DataFrame:
    max_points = {"platform": 28, "stair": 12, "hall": 34}[layer]
    min_spacing = {"platform": 4.0, "stair": 2.0, "hall": 4.0}[layer]
    ranked = cells.sort_values(["ParticipantSupport", "Value"], ascending=False)
    chosen: list[tuple[np.ndarray, object]] = []
    for row in ranked.itertuples():
        point = np.asarray([row.PosX, row.PosZ], dtype=float)
        if chosen and min(np.linalg.norm(point - item[0]) for item in chosen) < min_spacing:
            continue
        chosen.append((point, row))
        if len(chosen) >= max_points:
            break
    if not chosen:
        return pd.DataFrame(columns=["PosX", "PosZ", "Value", "ParticipantSupport"])
    return pd.DataFrame([item[1]._asdict() for item in chosen])


def build_runtime_surface_data(
    zones: pd.DataFrame,
    spec: MetricSpec,
    scenarios: tuple[str, ...],
    layers: tuple[str, ...],
    colour_scale_mode: str,
):
    if spec.source == "timestamp_participant_cells":
        participant_cells = timestamp_participant_cells(
            spec, zones, scenarios, layers
        )
        source_interface = str(TIMESTAMP_PARTICIPANT_CELL_FILE)
    elif spec.source == "visual_progress_bins":
        participant_cells = progress_metric_participant_cells(
            spec, zones, scenarios, layers
        )
        source_interface = str(VISUAL_INFORMATION_PROGRESS_FILE)
    elif spec.source == "gaze_depth_points":
        participant_cells = gaze_depth_participant_cells(
            spec, zones, scenarios, layers
        )
        source_interface = str(GAZE_DEPTH_POINT_ROOT / "<Scenario>" / "points_<Zone>.csv")
    else:
        raise ValueError(f"Unsupported metric source: {spec.source}")
    cells = participant_balanced_cells(
        participant_cells,
        spec.value_min,
        spec.value_max,
    )
    base_y_by_layer = {
        layer: layer_base_y(zones, participant_cells, layer)
        for layer in layers
    }
    surface_parts: list[pd.DataFrame] = []
    post_parts: list[pd.DataFrame] = []
    participant_counts: dict[str, dict[str, int]] = {}
    for scenario in scenarios:
        participant_counts[scenario] = {}
        for layer in layers:
            part = cells.loc[
                cells["Scenario"].eq(scenario) & cells["Layer"].eq(layer)
            ].copy()
            if part.empty:
                raise ValueError(f"No directly matched cells for {scenario}/{layer}")
            footprint = layer_footprint(zones, layer)
            x_grid, z_grid, values = interpolate_surface(part, footprint, layer)
            row_index, column_index = np.indices(values.shape)
            surface_parts.append(
                pd.DataFrame(
                    {
                        "Scenario": scenario,
                        "Layer": layer,
                        "GridRow": row_index.ravel(),
                        "GridColumn": column_index.ravel(),
                        "PosX": x_grid.ravel(),
                        "PosZ": z_grid.ravel(),
                        "Value": values.ravel(),
                        "BaseY": base_y_by_layer[layer],
                    }
                )
            )
            posts = select_postpoints(part, layer)[
                ["PosX", "PosZ", "Value", "ParticipantSupport"]
            ].copy()
            posts.insert(0, "Layer", layer)
            posts.insert(0, "Scenario", scenario)
            posts["BaseY"] = base_y_by_layer[layer]
            post_parts.append(posts)
            participant_counts[scenario][layer] = int(
                participant_cells.loc[
                    participant_cells["Scenario"].eq(scenario)
                    & participant_cells["Layer"].eq(layer),
                    "ParticipantID",
                ].nunique()
            )
    config = {
        "key": spec.key,
        "source_interface": source_interface,
        "source_kind": spec.source,
        "metric": spec.field,
        "label": spec.label,
        "unit": spec.unit,
        "definition": spec.definition,
        "value_min": spec.value_min,
        "value_max": spec.value_max,
        "value_ticks": list(spec.value_ticks),
        "world_height_max": spec.world_height_max,
        "scenarios": list(scenarios),
        "scenario_labels": SCENARIO_LABELS,
        "layers": list(layers),
        "layer_labels": LAYER_LABELS,
        "participant_counts": participant_counts,
        "surface_method": SURFACE_METHOD,
        "gaussian_kernel_bandwidth_m": GAUSSIAN_KERNEL_BANDWIDTH_M,
        "gaussian_kernel_truncate_sigma": GAUSSIAN_KERNEL_TRUNCATE_SIGMA,
        "viewport_y_shift_px": VIEWPORT_Y_SHIFT_PX,
        "palette_stops": list(spec.palette_stops),
        "colour_scale_mode": colour_scale_mode,
        "scene_basemaps": {key: str(value) for key, value in SCENE_BASEMAPS.items()},
        "scene_basemap_alpha": 0.82,
    }
    return (
        config,
        pd.concat(surface_parts, ignore_index=True),
        pd.concat(post_parts, ignore_index=True),
    )


def layer_footprint(zones: pd.DataFrame, layer: str):
    part = zones.loc[zones["asset_layer"].eq(layer)]
    return unary_union(part["geometry"].tolist()).buffer(0)


def grid_arrays(surface_part: pd.DataFrame):
    ordered = surface_part.sort_values(["GridRow", "GridColumn"])
    row_count = int(ordered["GridRow"].max()) + 1
    column_count = int(ordered["GridColumn"].max()) + 1
    if len(ordered) != row_count * column_count:
        raise ValueError("Prepared grid is not a complete rectangular row/column lattice")
    x_grid = ordered["PosX"].to_numpy(float).reshape(row_count, column_count)
    z_grid = ordered["PosZ"].to_numpy(float).reshape(row_count, column_count)
    values = ordered["Value"].to_numpy(float).reshape(row_count, column_count)
    base_values = ordered["BaseY"].dropna().unique()
    if len(base_values) != 1:
        raise ValueError(f"Prepared grid must contain one BaseY value, got {base_values}")
    return x_grid, z_grid, values, float(base_values[0])


def make_canvas(height_px: int | None = None):
    camera = DEFAULT_CAMERA
    canvas_height = int(height_px or camera.height_px)
    fig = plt.figure(
        figsize=(camera.width_px / LAYOUT_DPI, canvas_height / LAYOUT_DPI),
        dpi=OUTPUT_DPI,
        frameon=True,
    )
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, camera.width_px)
    ax.set_ylim(canvas_height, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.patch.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    return fig, ax


@lru_cache(maxsize=12)
def prepared_basemap(
    source_text: str,
    y_shift_px: int,
    alpha_fraction: float,
    canvas_height_px: int,
) -> np.ndarray:
    source = Path(source_text)
    if not source.exists():
        raise FileNotFoundError(source)
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        scale = DEFAULT_CAMERA.height_px / rgba.height
        width = int(round(rgba.width * scale))
        resized = rgba.resize((width, DEFAULT_CAMERA.height_px), Image.Resampling.LANCZOS)
    alpha = resized.getchannel("A").point(
        lambda value: int(round(value * alpha_fraction))
    )
    resized.putalpha(alpha)
    canvas = Image.new(
        "RGBA",
        (DEFAULT_CAMERA.width_px, int(canvas_height_px)),
        (0, 0, 0, 0),
    )
    x_offset = int(round((DEFAULT_CAMERA.width_px - width) / 2.0))
    canvas.alpha_composite(resized, dest=(x_offset, int(y_shift_px)))
    return np.asarray(canvas)


def draw_screenshot_basemap(
    ax,
    layer: str,
    config: dict,
    canvas_height_px: int | None = None,
) -> bool:
    source = config.get("scene_basemaps", {}).get(layer)
    if not source:
        return False
    shift = int(round(float(config["viewport_y_shift_px"][layer])))
    alpha = float(config.get("scene_basemap_alpha", 0.82))
    canvas_height = int(canvas_height_px or DEFAULT_CAMERA.height_px)
    image = prepared_basemap(str(source), shift, alpha, canvas_height)
    ax.imshow(
        image,
        extent=(0, DEFAULT_CAMERA.width_px, canvas_height, 0),
        interpolation="none",
        zorder=-20,
    )
    return True


class ViewportTranslatedProjector:
    def __init__(self, y_shift_px: float = 0.0):
        self.base = UnityOrthographicProjector(DEFAULT_CAMERA)
        self.y_shift_px = float(y_shift_px)

    def project(self, world_points):
        screen, depth = self.base.project(world_points)
        screen = screen.copy()
        screen[:, 1] += self.y_shift_px
        return screen, depth


def draw_base(ax, footprint, base_y: float, projector) -> None:
    all_segments = []
    for polygon in iter_polygons(footprint):
        path, _ = compound_top_path(polygon, base_y, projector)
        if path is not None:
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=(0.90, 0.93, 0.95, 0.22),
                    edgecolor="none",
                )
            )
        all_segments.extend(
            polygon_true_boundary_segments(polygon, base_y, projector)
        )
    if all_segments:
        ax.add_collection(
            LineCollection(
                all_segments,
                colors=[(0.30, 0.34, 0.39, 0.62)],
                linewidths=0.75,
                capstyle="round",
                joinstyle="round",
            )
        )


def draw_surface(
    ax,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    values: np.ndarray,
    base_y: float,
    projector,
    cmap,
    norm,
    height_scale: float,
    height_offset: float,
    alpha: float = 1.0,
    footprint=None,
    value_min: float = 0.0,
) -> int:
    # Display-only minimum-colour floor: preserve NaNs and supported heights.
    # Exact polygon paths retain holes and fill gaps at the grid boundary.
    if footprint is not None:
        minimum_y = base_y + height_offset + height_scale * value_min
        for polygon in iter_polygons(footprint):
            path, _ = compound_top_path(polygon, minimum_y, projector)
            if path is not None:
                ax.add_patch(PathPatch(
                    path, facecolor=mpl.colors.to_rgba(cmap(0.0), alpha=alpha),
                    edgecolor="none", linewidth=0.0,
                ))
    faces = []
    rows, columns = values.shape
    for row in range(rows - 1):
        for column in range(columns - 1):
            local_values = np.asarray(
                [
                    values[row, column],
                    values[row, column + 1],
                    values[row + 1, column + 1],
                    values[row + 1, column],
                ]
            )
            if not np.isfinite(local_values).all():
                continue
            world = np.asarray(
                [
                    [x_grid[row, column], base_y + height_offset + height_scale * local_values[0], z_grid[row, column]],
                    [x_grid[row, column + 1], base_y + height_offset + height_scale * local_values[1], z_grid[row, column + 1]],
                    [x_grid[row + 1, column + 1], base_y + height_offset + height_scale * local_values[2], z_grid[row + 1, column + 1]],
                    [x_grid[row + 1, column], base_y + height_offset + height_scale * local_values[3], z_grid[row + 1, column]],
                ]
            )
            screen, depth = projector.project(world)
            faces.append(
                (
                    float(np.mean(depth)),
                    PolygonPatch(
                        screen,
                        closed=True,
                        facecolor=mpl.colors.to_rgba(
                            cmap(norm(float(np.mean(local_values)))),
                            alpha=alpha,
                        ),
                        edgecolor="none",
                        linewidth=0.0,
                        antialiased=False,
                    ),
                )
            )
    for _, patch in sorted(faces, key=lambda item: item[0], reverse=True):
        ax.add_patch(patch)
    return len(faces)


def contour_segments(x_grid, z_grid, values, levels):
    temporary_figure, temporary_axes = plt.subplots(figsize=(2, 2))
    contour_set = temporary_axes.contour(
        x_grid,
        z_grid,
        np.ma.masked_invalid(values),
        levels=levels,
    )
    segments = [
        (float(level), segment.copy())
        for level, group in zip(contour_set.levels, contour_set.allsegs)
        for segment in group
    ]
    plt.close(temporary_figure)
    return segments


def draw_contours(
    ax,
    x_grid,
    z_grid,
    values,
    levels,
    base_y,
    projector,
    height_scale,
    height_offset,
    colour=(0.13, 0.15, 0.18, 0.52),
) -> None:
    projected = []
    for level, segment in contour_segments(x_grid, z_grid, values, levels):
        if len(segment) < 2:
            continue
        world = np.column_stack(
            [
                segment[:, 0],
                np.full(len(segment), base_y + height_offset + height_scale * level + 0.05),
                segment[:, 1],
            ]
        )
        screen, _ = projector.project(world)
        projected.append(screen)
    if projected:
        ax.add_collection(
            LineCollection(
                projected,
                colors=[colour],
                linewidths=0.65,
            )
        )


def draw_postpoints(
    ax,
    posts: pd.DataFrame,
    base_y: float,
    projector,
    cmap,
    norm,
    height_scale: float,
    height_offset: float,
) -> int:
    if posts.empty:
        return 0
    stems = []
    tops = []
    colours = []
    sizes = []
    for row in posts.itertuples():
        value = float(row.Value)
        world = np.asarray(
            [
                [row.PosX, base_y, row.PosZ],
                [row.PosX, base_y + height_offset + height_scale * value, row.PosZ],
            ]
        )
        screen, _ = projector.project(world)
        stems.append(screen)
        tops.append(screen[1])
        colours.append(cmap(norm(value)))
        sizes.append(10.0 + 2.0 * np.sqrt(float(row.ParticipantSupport)))
    ax.add_collection(
        LineCollection(
            stems,
            colors=[(0.24, 0.27, 0.31, 0.50)],
            linewidths=0.55,
        )
    )
    tops_array = np.asarray(tops)
    ax.scatter(
        tops_array[:, 0],
        tops_array[:, 1],
        s=sizes,
        c=colours,
        edgecolors=(1.0, 1.0, 1.0, 0.88),
        linewidths=0.45,
        zorder=9,
    )
    return len(posts)


def axis_anchor(footprint, base_y: float, projector):
    min_x, min_z, max_x, max_z = footprint.bounds
    padding = 4.0
    candidates = np.asarray(
        [
            [min_x - padding, base_y, min_z - padding],
            [min_x - padding, base_y, max_z + padding],
            [max_x + padding, base_y, min_z - padding],
            [max_x + padding, base_y, max_z + padding],
        ]
    )
    screen, _ = projector.project(candidates)
    viable = np.where(
        (screen[:, 0] > 60)
        & (screen[:, 0] < DEFAULT_CAMERA.width_px - 320)
        & (screen[:, 1] > 250)
        & (screen[:, 1] < 850)
    )[0]
    if len(viable) == 0:
        viable = np.arange(len(candidates))
    return candidates[viable[np.argmin(screen[viable, 0])]]


def draw_vertical_axis(
    ax,
    footprint,
    base_y: float,
    projector,
    ticks,
    height_scale: float,
    height_offset: float,
    label: str,
) -> None:
    anchor = axis_anchor(footprint, base_y, projector)
    worlds = np.asarray(
        [
            [anchor[0], base_y + height_offset + height_scale * value, anchor[2]]
            for value in ticks
        ]
    )
    screen, _ = projector.project(worlds)
    ax.plot(screen[:, 0], screen[:, 1], color="#20252B", linewidth=1.05, zorder=12)
    for index, (value, point) in enumerate(zip(ticks, screen)):
        ax.plot(
            [point[0] - 7, point[0] + 7],
            [point[1], point[1]],
            color="#20252B",
            linewidth=0.8,
        )
        if index in {0, len(ticks) - 1}:
            ax.text(
                point[0] - 12,
                point[1],
                f"{value:g}",
                ha="right",
                va="center",
                fontsize=11.0,
                color="#20252B",
            )
    middle = screen[len(screen) // 2]
    ax.text(
        middle[0] - 105,
        middle[1],
        label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#20252B",
    )


def add_colourbar(fig, cmap, norm, ticks, label: str, boundaries, saturated=False) -> None:
    colourbar_axes = fig.add_axes([0.875, 0.25, 0.018, 0.50])
    colourbar_kwargs = {}
    if boundaries is not None:
        colourbar_kwargs.update(boundaries=boundaries, spacing="proportional")
    colourbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colourbar_axes,
        **colourbar_kwargs,
    )
    colourbar.set_ticks(ticks, labels=colour_tick_labels(ticks, saturated))
    colourbar.ax.tick_params(labelsize=11.0, length=4.0, width=0.9)
    colourbar.outline.set_linewidth(0.8)
    colourbar.set_label(label, fontsize=12.5, fontweight="bold", labelpad=8)


def add_overlay_colourbars(
    fig,
    configs: list[dict],
    canvas_height_px: int,
) -> None:
    """Draw readable shared colour legends for the five-column overview sheet."""
    top_px = 175.0
    step_px = 210.0
    bar_height_px = 28.0
    for index, config in enumerate(configs):
        # Surface 1 is the bottom layer and surface 4 is the top layer, so the
        # legend uses the same physical bottom-to-top order.
        screen_y = top_px + (len(configs) - index - 1) * step_px
        y_pos = 1.0 - (screen_y + bar_height_px) / canvas_height_px
        bar_height = bar_height_px / canvas_height_px
        cmap, norm, boundaries = metric_cmap_norm(
            config,
            f"overlay_{config['key']}",
        )
        colourbar_axes = fig.add_axes([0.665, y_pos, 0.270, bar_height])
        colourbar_axes.set_facecolor((1.0, 1.0, 1.0, 0.92))
        colourbar_kwargs = {}
        if boundaries is not None:
            colourbar_kwargs.update(boundaries=boundaries, spacing="proportional")
        colourbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=colourbar_axes,
            orientation="horizontal",
            **colourbar_kwargs,
        )
        ticks = config.get("colour_ticks", config["value_ticks"])
        colourbar.set_ticks(ticks, labels=colour_tick_labels(
            ticks, config.get("colour_saturated", False)))
        colourbar.ax.tick_params(labelsize=12.0, length=4.0, width=0.9, pad=2.0)
        colourbar.outline.set_linewidth(0.9)
        fig.text(
            0.660,
            1.0 - (screen_y - 42.0) / canvas_height_px,
            f"{index + 1}. {config['label']} ({config['unit']})",
            ha="left",
            va="bottom",
            fontsize=13.0,
            fontweight="bold",
            color="#20252B",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
                "pad": 1.5,
            },
        )


def render_one(
    scenario: str,
    layer: str,
    surface_part: pd.DataFrame,
    post_part: pd.DataFrame,
    zones: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> Path:
    x_grid, z_grid, values, base_y = grid_arrays(surface_part)
    footprint = layer_footprint(zones, layer)
    value_min = float(config["value_min"])
    value_max = float(config["value_max"])
    ticks = np.asarray(config["value_ticks"], dtype=float)
    colour_ticks = config.get("colour_ticks", config["value_ticks"])
    cmap, norm, boundaries = metric_cmap_norm(
        config,
        "prepared_metric_palette",
    )
    height_scale = float(config["world_height_max"]) / (value_max - value_min)
    height_offset = (
        BASEMAP_TO_FIRST_SURFACE_GAP_M
        - height_scale * value_min
    )
    axis_label = f"{config['label']} ({config['unit']})"

    fig, ax = make_canvas()
    shift = float(config["viewport_y_shift_px"][layer])
    projector = ViewportTranslatedProjector(shift)
    has_basemap = draw_screenshot_basemap(ax, layer, config)
    if not has_basemap:
        draw_base(ax, footprint, base_y, projector)
    draw_surface(
        ax,
        x_grid,
        z_grid,
        values,
        base_y,
        projector,
        cmap,
        norm,
        height_scale,
        height_offset,
        footprint=footprint,
        value_min=value_min,
    )
    interior_ticks = ticks[1:-1]
    if len(interior_ticks):
        draw_contours(
            ax,
            x_grid,
            z_grid,
            values,
            interior_ticks,
            base_y,
            projector,
            height_scale,
            height_offset,
        )
    draw_postpoints(
        ax,
        post_part,
        base_y,
        projector,
        cmap,
        norm,
        height_scale,
        height_offset,
    )
    draw_vertical_axis(
        ax,
        footprint,
        base_y,
        projector,
        ticks,
        height_scale,
        height_offset,
        axis_label,
    )
    add_colourbar(fig, cmap, norm, colour_ticks, axis_label, boundaries,
                  saturated=config.get("colour_saturated", False))

    title = (
        f"{config['scenario_labels'][scenario]} · "
        f"{config['layer_labels'][layer]}"
    )
    ax.text(
        DEFAULT_CAMERA.width_px * 0.5,
        40,
        title,
        ha="center",
        va="center",
        fontsize=18.0,
        fontweight="bold",
        color="#171A1F",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_stem = safe_name(scenario.lower().replace("-", "__"))
    filename = f"{safe_name(config['metric'])}__{scenario_stem}__{safe_name(layer)}.png"
    target = output_dir / filename
    save_png_with_retry(fig, target)
    plt.close(fig)
    return target


def fallback_overlay_connector_indices(
    prepared: list[dict],
    count: int = STAIR_OVERLAY_CONNECTOR_COUNT,
) -> list[tuple[int, int]]:
    """Choose shared grid locations distributed along the footprint's main axis."""
    if not prepared or count <= 0:
        return []
    reference_shape = prepared[0]["values"].shape
    if any(item["values"].shape != reference_shape for item in prepared):
        return []
    common_valid = np.logical_and.reduce(
        [np.isfinite(item["values"]) for item in prepared]
    )
    row_indices, column_indices = np.where(common_valid)
    if len(row_indices) < count:
        return []
    x_grid = prepared[0]["x_grid"]
    z_grid = prepared[0]["z_grid"]
    coordinates = np.column_stack(
        [x_grid[common_valid], z_grid[common_valid]]
    )
    centre = np.mean(coordinates, axis=0)
    centred = coordinates - centre
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    main_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    along = centred @ main_axis
    across = np.abs(
        centred @ np.asarray([-main_axis[1], main_axis[0]])
    )
    along_scale = max(float(np.ptp(along)), 1e-9)
    across_scale = max(float(np.percentile(across, 75)), 1e-9)
    selected: list[tuple[int, int]] = []
    for quantile in np.linspace(0.28, 0.72, count):
        target = float(np.quantile(along, quantile))
        score = np.abs(along - target) / along_scale + 0.35 * across / across_scale
        for candidate in np.argsort(score):
            index = (int(row_indices[candidate]), int(column_indices[candidate]))
            if index not in selected:
                selected.append(index)
                break
    return selected


def stair_region_overlay_connector_indices(
    prepared: list[dict],
    zones: pd.DataFrame,
    layer: str,
    count: int = STAIR_OVERLAY_CONNECTOR_COUNT,
) -> list[tuple[int, int]]:
    """Snap two semantic stair-region centres to common valid metric-grid cells."""
    if not prepared or count <= 0:
        return []
    reference_shape = prepared[0]["values"].shape
    if any(item["values"].shape != reference_shape for item in prepared):
        return []
    common_valid = np.logical_and.reduce(
        [np.isfinite(item["values"]) for item in prepared]
    )
    row_indices, column_indices = np.where(common_valid)
    if len(row_indices) < count:
        return []

    anchor_type = "StairEntrance" if layer == "platform" else "Stair"
    anchor_rows = zones.loc[zones["zone_type"].eq(anchor_type)].copy()
    if len(anchor_rows) < count:
        return fallback_overlay_connector_indices(prepared, count)
    anchor_rows = anchor_rows.sort_values(
        ["canonical_order", "priority", "zone_id"]
    ).head(count)

    x_grid = prepared[0]["x_grid"]
    z_grid = prepared[0]["z_grid"]
    valid_coordinates = np.column_stack(
        [x_grid[common_valid], z_grid[common_valid]]
    )
    selected: list[tuple[int, int]] = []
    for zone in anchor_rows.itertuples(index=False):
        centre = zone.geometry.centroid
        squared_distance = (
            (valid_coordinates[:, 0] - float(centre.x)) ** 2
            + (valid_coordinates[:, 1] - float(centre.y)) ** 2
        )
        for candidate in np.argsort(squared_distance):
            index = (int(row_indices[candidate]), int(column_indices[candidate]))
            if index not in selected:
                selected.append(index)
                break
    if len(selected) < count:
        return fallback_overlay_connector_indices(prepared, count)
    return selected


def draw_stair_region_overlay_connectors(
    ax,
    prepared: list[dict],
    zones: pd.DataFrame,
    layer: str,
    base_y: float,
    projector,
) -> int:
    """Draw two straight dashed correspondence lines through all metric surfaces."""
    connector_indices = stair_region_overlay_connector_indices(
        prepared,
        zones,
        layer,
    )
    for row, column in connector_indices:
        world_points = []
        x = float(prepared[0]["x_grid"][row, column])
        z = float(prepared[0]["z_grid"][row, column])
        for item in prepared:
            config = item["config"]
            value = float(item["values"][row, column])
            value_min = float(config["value_min"])
            value_max = float(config["value_max"])
            normalized = float(
                np.clip((value - value_min) / (value_max - value_min), 0.0, 1.0)
            )
            world_points.append(
                [
                    x,
                    base_y + float(item["world_offset"]) + COMBINED_RELIEF_HEIGHT_M * normalized,
                    z,
                ]
            )
        screen, _ = projector.project(np.asarray(world_points, dtype=float))
        ax.plot(
            screen[:, 0],
            screen[:, 1],
            color=STAIR_OVERLAY_CONNECTOR_COLOUR,
            linewidth=1.05,
            linestyle=(0, (5.0, 4.0)),
            alpha=0.96,
            dash_capstyle="round",
            zorder=20,
        )
    return len(connector_indices)


def projected_surface_vertical_envelope(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    normalized: np.ndarray,
    base_y: float,
    projector,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a surface silhouette and return its top/bottom y per x pixel."""
    width = DEFAULT_CAMERA.width_px
    height = DEFAULT_CAMERA.height_px
    silhouette = Image.new("1", (width, height), 0)
    drawer = ImageDraw.Draw(silhouette)
    rows, columns = normalized.shape
    for row in range(rows - 1):
        for column in range(columns - 1):
            local = np.asarray(
                [
                    normalized[row, column],
                    normalized[row, column + 1],
                    normalized[row + 1, column + 1],
                    normalized[row + 1, column],
                ],
                dtype=float,
            )
            if not np.isfinite(local).all():
                continue
            world = np.asarray(
                [
                    [x_grid[row, column], base_y + COMBINED_RELIEF_HEIGHT_M * local[0], z_grid[row, column]],
                    [x_grid[row, column + 1], base_y + COMBINED_RELIEF_HEIGHT_M * local[1], z_grid[row, column + 1]],
                    [x_grid[row + 1, column + 1], base_y + COMBINED_RELIEF_HEIGHT_M * local[2], z_grid[row + 1, column + 1]],
                    [x_grid[row + 1, column], base_y + COMBINED_RELIEF_HEIGHT_M * local[3], z_grid[row + 1, column]],
                ],
                dtype=float,
            )
            screen, _ = projector.project(world)
            drawer.polygon(
                [(float(point[0]), float(point[1])) for point in screen],
                fill=1,
            )
    mask = np.asarray(silhouette, dtype=bool)
    top = np.full(width, np.nan, dtype=float)
    bottom = np.full(width, np.nan, dtype=float)
    for x in np.flatnonzero(mask.any(axis=0)):
        occupied_y = np.flatnonzero(mask[:, x])
        top[x] = float(occupied_y[0])
        bottom[x] = float(occupied_y[-1])
    return top, bottom


def apply_overlay_world_interval(
    prepared: list[dict],
    uniform_interval: float,
    projected_y_per_world_m: float,
) -> float:
    """Apply one shared adjacent base-plane interval to an overlay stack."""
    if not np.isfinite(uniform_interval) or uniform_interval <= 0:
        raise ValueError("Overlay world interval must be a positive finite value")
    if not prepared:
        raise ValueError("At least one prepared surface is required")
    first_offset = float(prepared[0]["world_offset"])
    for index, item in enumerate(prepared):
        world_offset = first_offset + index * uniform_interval
        upward_screen_shift = -projected_y_per_world_m * world_offset
        item["world_offset"] = float(world_offset)
        item["screen_min_y"] = float(
            item["base_screen_min_y"] - upward_screen_shift
        )
        item["screen_max_y"] = float(
            item["base_screen_max_y"] - upward_screen_shift
        )
        item["column_top_y"] = (
            item["base_column_top_y"] - upward_screen_shift
        )
        item["column_bottom_y"] = (
            item["base_column_bottom_y"] - upward_screen_shift
        )
        item["uniform_world_interval_m"] = uniform_interval
    return uniform_interval


def uniformize_overlay_world_offsets(
    prepared: list[dict],
    layer: str,
    projected_y_per_world_m: float,
) -> float | None:
    """Make adjacent base-plane intervals equal to their current maximum."""
    if layer not in UNIFORM_OVERLAY_INTERVAL_LAYERS or len(prepared) < 2:
        return None
    current_offsets = np.asarray(
        [float(item["world_offset"]) for item in prepared],
        dtype=float,
    )
    current_intervals = np.diff(current_offsets)
    if not np.isfinite(current_intervals).all() or np.any(current_intervals <= 0):
        raise ValueError("Overlay world offsets must increase from bottom to top")
    return apply_overlay_world_interval(
        prepared,
        float(np.max(current_intervals)),
        projected_y_per_world_m,
    )


def prepare_overlay_geometry(
    scenario: str,
    layer: str,
    metric_payloads: list[tuple[dict, pd.DataFrame, pd.DataFrame]],
    zones: pd.DataFrame,
) -> dict:
    """Prepare one panel without fixing its final layer-wide viewport."""
    if not metric_payloads:
        raise ValueError("At least one metric is required for an overlay")
    footprint = layer_footprint(zones, layer)
    reference_config = metric_payloads[0][0]
    base_y_reference = grid_arrays(metric_payloads[0][1])[3]
    base_projector = UnityOrthographicProjector(DEFAULT_CAMERA)
    vertical_probe, _ = base_projector.project(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    projected_y_per_world_m = float(vertical_probe[1, 1] - vertical_probe[0, 1])
    if projected_y_per_world_m >= 0:
        raise ValueError("Unity projection must move higher world Y upward on screen")
    basemap_gap_screen_shift = float(
        -projected_y_per_world_m * BASEMAP_TO_FIRST_SURFACE_GAP_M
    )

    prepared = []
    for config, surface_part, post_part in metric_payloads:
        x_grid, z_grid, values, _ = grid_arrays(surface_part)
        value_min = float(config["value_min"])
        value_max = float(config["value_max"])
        normalized = np.clip(
            (values - value_min) / (value_max - value_min),
            0.0,
            1.0,
        )
        valid = np.isfinite(normalized)
        world = np.column_stack(
            [
                x_grid[valid],
                base_y_reference + COMBINED_RELIEF_HEIGHT_M * normalized[valid],
                z_grid[valid],
            ]
        )
        screen, _ = base_projector.project(world)
        column_top_y, column_bottom_y = projected_surface_vertical_envelope(
            x_grid,
            z_grid,
            normalized,
            base_y_reference,
            base_projector,
        )
        prepared.append(
            {
                "config": config,
                "surface_part": surface_part,
                "post_part": post_part,
                "x_grid": x_grid,
                "z_grid": z_grid,
                "values": values,
                "base_screen_min_y": float(np.min(screen[:, 1])),
                "base_screen_max_y": float(np.max(screen[:, 1])),
                "base_column_top_y": column_top_y,
                "base_column_bottom_y": column_bottom_y,
            }
        )

    # Start from the configured scaled placement. The silhouette pass below
    # expands only a surface whose candidate would collide.
    previous_bbox_top_y = None
    for index, item in enumerate(prepared):
        if index == 0:
            touch_world_offset = 0.0
        else:
            desired_bottom_y = (
                previous_bbox_top_y - COMBINED_APPROVED_COMPACT_GAP_PX
            )
            touch_world_offset = (
                desired_bottom_y - item["base_screen_max_y"]
            ) / projected_y_per_world_m
        item["compact_world_offset"] = float(
            touch_world_offset * COMBINED_LAYER_OFFSET_SCALE
        )
        previous_bbox_top_y = float(
            item["base_screen_min_y"]
            + projected_y_per_world_m * touch_world_offset
        )

    # Check actual projected silhouettes rather than global bounding boxes. The
    # hall is diagonal in screen space, so this avoids both hidden overlap
    # and excessive whitespace.
    preserve_approved_compact_layout = (
        scenario.startswith(("StaticSign-", "TextBar-")) or layer == "platform"
    )
    previous_column_top_y = None
    for index, item in enumerate(prepared):
        compact_upward_shift = float(
            -projected_y_per_world_m * item["compact_world_offset"]
        )
        if index == 0:
            upward_screen_shift = basemap_gap_screen_shift
            world_offset = BASEMAP_TO_FIRST_SURFACE_GAP_M
        elif preserve_approved_compact_layout:
            upward_screen_shift = (
                basemap_gap_screen_shift + compact_upward_shift
            )
            world_offset = (
                BASEMAP_TO_FIRST_SURFACE_GAP_M
                + item["compact_world_offset"]
            )
        else:
            shared = np.isfinite(previous_column_top_y) & np.isfinite(
                item["base_column_bottom_y"]
            )
            if not np.any(shared):
                raise ValueError("Metric surface silhouettes do not share screen columns")
            required_shift = (
                item["base_column_bottom_y"][shared]
                - previous_column_top_y[shared]
                + COMBINED_PROJECTED_GAP_PX
            )
            upward_screen_shift = max(
                basemap_gap_screen_shift + compact_upward_shift,
                basemap_gap_screen_shift,
                float(np.max(required_shift)),
            )
            world_offset = -upward_screen_shift / projected_y_per_world_m
        item["world_offset"] = float(world_offset)
        item["screen_min_y"] = float(
            item["base_screen_min_y"] - upward_screen_shift
        )
        item["screen_max_y"] = float(
            item["base_screen_max_y"] - upward_screen_shift
        )
        item["column_top_y"] = item["base_column_top_y"] - upward_screen_shift
        item["column_bottom_y"] = (
            item["base_column_bottom_y"] - upward_screen_shift
        )
        if index > 0 and not preserve_approved_compact_layout:
            shared = np.isfinite(previous_column_top_y) & np.isfinite(
                item["column_bottom_y"]
            )
            actual_gap = float(
                np.min(
                    previous_column_top_y[shared]
                    - item["column_bottom_y"][shared]
                )
            )
            if actual_gap < COMBINED_PROJECTED_GAP_PX - 1e-6:
                raise ValueError(
                    "Metric surfaces overlap after silhouette-aware placement"
                )
        previous_column_top_y = item["column_top_y"]

    uniform_world_interval = uniformize_overlay_world_offsets(
        prepared,
        layer,
        projected_y_per_world_m,
    )
    return {
        "scenario": scenario,
        "layer": layer,
        "footprint": footprint,
        "reference_config": reference_config,
        "base_y_reference": base_y_reference,
        "projected_y_per_world_m": projected_y_per_world_m,
        "prepared": prepared,
        "uniform_world_interval_m": uniform_world_interval,
    }


def standardize_overlay_geometries_by_layer(
    geometries: dict[tuple[str, str], dict],
) -> dict[str, dict]:
    """Use one spacing, projection shift, and canvas size for each spatial layer."""
    summaries: dict[str, dict] = {}
    layer_names = tuple(dict.fromkeys(key[1] for key in geometries))
    for layer in layer_names:
        layer_geometries = [
            geometry
            for (_, geometry_layer), geometry in geometries.items()
            if geometry_layer == layer
        ]
        if not layer_geometries:
            continue

        interval_candidates = [
            float(geometry["uniform_world_interval_m"])
            for geometry in layer_geometries
            if geometry["uniform_world_interval_m"] is not None
        ]
        layer_interval = max(interval_candidates) if interval_candidates else None
        if layer_interval is not None:
            for geometry in layer_geometries:
                if len(geometry["prepared"]) < 2:
                    continue
                apply_overlay_world_interval(
                    geometry["prepared"],
                    layer_interval,
                    float(geometry["projected_y_per_world_m"]),
                )
                geometry["uniform_world_interval_m"] = layer_interval

        raw_min_y = min(
            float(item["screen_min_y"])
            for geometry in layer_geometries
            for item in geometry["prepared"]
        )
        raw_max_y = max(
            float(item["screen_max_y"])
            for geometry in layer_geometries
            for item in geometry["prepared"]
        )
        common_shift = float(COMBINED_TOP_MARGIN_PX - raw_min_y)
        content_bottom = max(
            raw_max_y + common_shift,
            DEFAULT_CAMERA.height_px + common_shift,
        )
        canvas_height = max(
            COMBINED_MIN_CANVAS_HEIGHT_PX,
            int(np.ceil(content_bottom + COMBINED_BOTTOM_MARGIN_PX)),
        )
        for geometry in layer_geometries:
            geometry["common_shift_px"] = common_shift
            geometry["canvas_height_px"] = canvas_height

        summaries[layer] = {
            "uniform_world_interval_m": layer_interval,
            "common_shift_px": common_shift,
            "canvas_width_px": DEFAULT_CAMERA.width_px,
            "canvas_height_px": canvas_height,
            "panel_count": len(layer_geometries),
            "stacked_metric_count": len(layer_geometries[0]["prepared"]),
        }
    return summaries


def render_overlay(
    scenario: str,
    layer: str,
    metric_payloads: list[tuple[dict, pd.DataFrame, pd.DataFrame]],
    zones: pd.DataFrame,
    output_dir: Path,
    show_legend: bool = False,
    geometry: dict | None = None,
) -> Path:
    """Render a prepared overlay using its layer-wide standardized viewport."""
    if geometry is None:
        geometry = prepare_overlay_geometry(
            scenario,
            layer,
            metric_payloads,
            zones,
        )
        standardize_overlay_geometries_by_layer({(scenario, layer): geometry})
    if geometry["scenario"] != scenario or geometry["layer"] != layer:
        raise ValueError("Prepared overlay geometry does not match the panel key")

    footprint = geometry["footprint"]
    reference_config = geometry["reference_config"]
    base_y_reference = float(geometry["base_y_reference"])
    prepared = geometry["prepared"]
    common_shift = float(geometry["common_shift_px"])
    canvas_height = int(geometry["canvas_height_px"])

    projector = ViewportTranslatedProjector(common_shift)
    fig, ax = make_canvas(canvas_height)
    basemap_config = dict(reference_config)
    basemap_config["viewport_y_shift_px"] = dict(
        reference_config["viewport_y_shift_px"]
    )
    basemap_config["viewport_y_shift_px"][layer] = common_shift
    has_basemap = draw_screenshot_basemap(
        ax,
        layer,
        basemap_config,
        canvas_height_px=canvas_height,
    )
    if not has_basemap:
        draw_base(ax, footprint, base_y_reference, projector)

    configs: list[dict] = []
    for item in prepared:
        config = item["config"]
        x_grid = item["x_grid"]
        z_grid = item["z_grid"]
        values = item["values"]
        value_min = float(config["value_min"])
        value_max = float(config["value_max"])
        cmap, norm, _ = metric_cmap_norm(
            config,
            f"overlay_surface_{config['key']}",
        )
        height_scale = COMBINED_RELIEF_HEIGHT_M / (value_max - value_min)
        height_offset = float(item["world_offset"]) - height_scale * value_min
        draw_surface(
            ax,
            x_grid,
            z_grid,
            values,
            base_y_reference,
            projector,
            cmap,
            norm,
            height_scale,
            height_offset,
            alpha=COMBINED_SURFACE_ALPHA,
            footprint=footprint,
            value_min=value_min,
        )
        interior_ticks = np.asarray(config["value_ticks"], dtype=float)[1:-1]
        if len(interior_ticks):
            line_colour = mpl.colors.to_rgba(config["palette_stops"][-1], 0.72)
            draw_contours(
                ax,
                x_grid,
                z_grid,
                values,
                interior_ticks,
                base_y_reference,
                projector,
                height_scale,
                height_offset,
                colour=line_colour,
            )
        configs.append(config)

    if layer in {"platform", "stair", "hall"}:
        draw_stair_region_overlay_connectors(
            ax,
            prepared,
            zones,
            layer,
            base_y_reference,
            projector,
        )

    if show_legend:
        add_overlay_colourbars(fig, configs, canvas_height)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_stem = safe_name(scenario.lower().replace("-", "__"))
    filename = f"all_metrics_overlay__{scenario_stem}__{safe_name(layer)}.png"
    target = output_dir / filename
    save_png_with_retry(fig, target)
    plt.close(fig)
    return target


def overlay_sheet_font(size: int, *, bold: bool = False):
    """Load a consistent Windows serif font, with a portable fallback."""
    filename = "timesbd.ttf" if bold else "times.ttf"
    candidates = (
        Path("C:/Windows/Fonts") / filename,
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def overlay_panel_content_bbox(source: Path, padding_px: int) -> tuple[int, int, int, int]:
    """Find a padded non-white panel extent so the overview can be compact."""
    with Image.open(source) as image:
        original_bbox = image.info.get("OverlayUnlabelledBBox")
        if original_bbox:
            left, top, right, bottom = json.loads(original_bbox)
            return (max(0, left - padding_px), max(0, top - padding_px),
                    min(image.width, right + padding_px),
                    min(image.height, bottom + padding_px))
        rgb = np.asarray(image.convert("RGB"))
    non_white = np.any(rgb < 252, axis=2)
    if not np.any(non_white):
        return 0, 0, int(rgb.shape[1]), int(rgb.shape[0])
    rows, columns = np.where(non_white)
    return (
        max(0, int(columns.min()) - padding_px),
        max(0, int(rows.min()) - padding_px),
        min(int(rgb.shape[1]), int(columns.max()) + padding_px + 1),
        min(int(rgb.shape[0]), int(rows.max()) + padding_px + 1),
    )


def label_overlay_panel(source: Path, label: str, crop_box: tuple[int, int, int, int]) -> None:
    """Label the retained right-hand whitespace without changing crop geometry."""
    original_bbox = overlay_panel_content_bbox(source, 0)
    with Image.open(source) as image:
        existing = image.info.get("OverlaySubplot")
        if existing == label:
            return
        if existing:
            raise ValueError(f"Panel {source} already has a different subplot label: {existing}")
        info = dict(image.info)
        panel = image.convert("RGB")
    font_path = Path("C:/Windows/Fonts/timesbd.ttf")
    font = ImageFont.truetype(str(font_path), size=323)
    draw = ImageDraw.Draw(panel)
    position = (crop_box[2] - 48, crop_box[1] + 180)
    bounds = draw.textbbox(position, label, font=font, anchor="rm")
    if not (crop_box[0] <= bounds[0] < bounds[2] <= crop_box[2]
            and crop_box[1] <= bounds[1] < bounds[3] <= crop_box[3]):
        raise ValueError(f"Subplot label does not fit the original crop: {source}")
    if np.any(np.asarray(panel.crop(bounds)) < 252):
        raise ValueError(f"Subplot label would overlap existing content: {source}")
    draw.text(position, label, fill="#000000", font=font, anchor="rm")
    panel.info.update(info)
    panel.info["OverlaySubplot"] = label
    panel.info["OverlayUnlabelledBBox"] = json.dumps(original_bbox)
    panel.info["OverlayLabelBBox"] = json.dumps(bounds)
    save_pil_png_with_retry(panel, source)


def draw_overlay_sheet_legends(
    sheet: Image.Image,
    configs: list[dict],
    top_px: int,
) -> None:
    """Distribute only the selected metric legends across the panel grid width."""
    if not configs:
        return
    draw = ImageDraw.Draw(sheet)
    left = OVERLAY_SHEET_LEFT_MARGIN_PX
    right = sheet.width - OVERLAY_SHEET_RIGHT_MARGIN_PX
    gap = OVERLAY_SHEET_LEGEND_COLUMN_GAP_PX
    column_width = int((right - left - gap * (len(configs) - 1)) / len(configs))
    label_font = overlay_sheet_font(58, bold=True)
    tick_font = overlay_sheet_font(46)
    bar_top = top_px + 78
    bar_height = 58
    sample_positions = np.linspace(0.0, 1.0, max(column_width, 2))

    for index, config in enumerate(configs):
        x0 = left + index * (column_width + gap)
        x1 = x0 + column_width
        draw.text(
            ((x0 + x1) // 2, top_px),
            f"{index + 1}. {config['label']} ({config['unit']})",
            fill="#20252B",
            font=label_font,
            anchor="mt",
        )
        cmap, norm, _ = metric_cmap_norm(config, f"sheet_legend_{config['key']}")
        colour_min = float(config.get("colour_min", config["value_min"]))
        colour_max = float(config.get("colour_max", config["value_max"]))
        sample_values = colour_min + sample_positions * (colour_max - colour_min)
        colours = np.asarray(cmap(norm(sample_values))[:, :3] * 255, dtype=np.uint8)
        gradient = np.repeat(colours[np.newaxis, :, :], bar_height, axis=0)
        sheet.paste(Image.fromarray(gradient, mode="RGB"), (x0, bar_top))
        draw.rectangle(
            (x0, bar_top, x1 - 1, bar_top + bar_height - 1),
            outline="#20252B",
            width=2,
        )
        ticks = config.get("colour_ticks", config["value_ticks"])
        labels = colour_tick_labels(ticks, config.get("colour_saturated", False))
        for tick, tick_label in zip(ticks, labels):
            fraction = (float(tick) - colour_min) / (colour_max - colour_min)
            x = int(round(x0 + fraction * (column_width - 1)))
            draw.line(
                (x, bar_top + bar_height, x, bar_top + bar_height + 10),
                fill="#20252B",
                width=2,
            )
            draw.text(
                (x, bar_top + bar_height + 13),
                tick_label,
                fill="#20252B",
                font=tick_font,
                anchor="lt" if tick == ticks[0] else "rt" if tick == ticks[-1] else "mt",
            )


def combine_overlay_panels(
    panel_paths: dict[tuple[str, str], Path],
    scenarios: tuple[str, ...],
    layers: tuple[str, ...],
    legend_configs: list[dict],
    output_dir: Path,
) -> Path:
    """Assemble every scenario/layer overlay into one semantic overview sheet."""
    if not panel_paths:
        raise ValueError("No All_Metrics_Overlay panels are available to combine")

    scenario_parts = {tuple(scenario.rsplit("-", 1)) for scenario in scenarios}
    guidances = tuple(
        guidance
        for guidance in GUIDANCES
        if any(part[0] == guidance for part in scenario_parts)
    )
    smoke_levels = tuple(
        smoke
        for smoke in SMOKE_LEVELS
        if any(part[1] == smoke for part in scenario_parts)
    )
    row_keys = tuple(
        (smoke, layer)
        for smoke in smoke_levels
        for layer in layers
        if any(
            (f"{guidance}-{smoke}", layer) in panel_paths
            for guidance in guidances
        )
    )
    if not guidances or not row_keys:
        raise ValueError("Overlay panels cannot be arranged by guidance/smoke/layer")

    source_sizes: dict[tuple[str, str], tuple[int, int]] = {}
    for panel_key, source in panel_paths.items():
        with Image.open(source) as image:
            source_sizes[panel_key] = image.size
    for layer in layers:
        layer_sizes = {
            size
            for (scenario, panel_layer), size in source_sizes.items()
            if panel_layer == layer and scenario in scenarios
        }
        if len(layer_sizes) > 1:
            details = {
                scenario: source_sizes[(scenario, layer)]
                for scenario in scenarios
                if (scenario, layer) in source_sizes
            }
            raise ValueError(
                f"Same-layer overlay panels must have identical dimensions: "
                f"{layer} -> {details}"
            )

    crop_boxes: dict[str, tuple[int, int, int, int]] = {}
    for layer in layers:
        boxes = [
            overlay_panel_content_bbox(source, OVERLAY_SHEET_PANEL_CROP_PADDING_PX)
            for (scenario, panel_layer), source in panel_paths.items()
            if panel_layer == layer and scenario in scenarios
        ]
        if boxes:
            crop_boxes[layer] = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
    # All panels originate from the same Unity camera. A shared horizontal
    # crop width therefore preserves the platform/hall scene-size ratio;
    # only their unused vertical whitespace is trimmed independently.
    if crop_boxes:
        shared_left = min(box[0] for box in crop_boxes.values())
        shared_right = max(box[2] for box in crop_boxes.values())
        crop_boxes = {
            layer: (shared_left, box[1], shared_right, box[3])
            for layer, box in crop_boxes.items()
        }

    # A-E follow the guidance columns; 1-4 follow low-platform, low-hall,
    # high-platform and high-hall. Compute all original crops before labelling.
    for row_index, (smoke, layer) in enumerate(row_keys, start=1):
        for column, guidance in enumerate(guidances):
            source = panel_paths.get((f"{guidance}-{smoke}", layer))
            if source is not None:
                label_overlay_panel(source, f"{chr(65 + column)}{row_index}", crop_boxes[layer])

    panel_width = OVERLAY_SHEET_PANEL_WIDTH_PX
    row_heights: dict[tuple[str, str], int] = {}
    for row_key in row_keys:
        smoke, layer = row_key
        scaled_heights = []
        for guidance in guidances:
            source = panel_paths.get((f"{guidance}-{smoke}", layer))
            if source is None:
                continue
            crop_box = crop_boxes[layer]
            width = crop_box[2] - crop_box[0]
            height = crop_box[3] - crop_box[1]
            scaled_heights.append(max(1, round(height * panel_width / width)))
        if not scaled_heights:
            raise ValueError(f"No panels are available for overview row {row_key}")
        row_heights[row_key] = max(scaled_heights)

    sheet_width = (
        OVERLAY_SHEET_LEFT_MARGIN_PX
        + len(guidances) * panel_width
        + max(0, len(guidances) - 1) * OVERLAY_SHEET_COLUMN_GAP_PX
        + OVERLAY_SHEET_RIGHT_MARGIN_PX
    )
    sheet_height = (
        OVERLAY_SHEET_TOP_MARGIN_PX
        + sum(row_heights.values())
        + max(0, len(row_keys) - 1) * OVERLAY_SHEET_ROW_GAP_PX
        + OVERLAY_SHEET_LEGEND_TOP_GAP_PX
        + OVERLAY_SHEET_LEGEND_HEIGHT_PX
        + OVERLAY_SHEET_BOTTOM_MARGIN_PX
    )
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    header_font = overlay_sheet_font(76, bold=True)
    row_font = overlay_sheet_font(72, bold=True)

    for column, guidance in enumerate(guidances):
        x_center = (
            OVERLAY_SHEET_LEFT_MARGIN_PX
            + column * (panel_width + OVERLAY_SHEET_COLUMN_GAP_PX)
            + panel_width // 2
        )
        draw.text(
            (x_center, OVERLAY_SHEET_TOP_MARGIN_PX - 78),
            GUIDANCE_LABELS[guidance],
            fill="#20252B",
            font=header_font,
            anchor="mm",
        )

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    y_cursor = OVERLAY_SHEET_TOP_MARGIN_PX
    for row_index, (smoke, layer) in enumerate(row_keys):
        row_height = row_heights[(smoke, layer)]
        for column, guidance in enumerate(guidances):
            source = panel_paths.get((f"{guidance}-{smoke}", layer))
            if source is None:
                continue
            with Image.open(source) as source_image:
                source_rgb = source_image.convert("RGB").crop(crop_boxes[layer])
                scaled_height = max(
                    1,
                    round(source_rgb.height * panel_width / source_rgb.width),
                )
                resized = source_rgb.resize(
                    (panel_width, scaled_height),
                    resample=resampling,
                )
            x = (
                OVERLAY_SHEET_LEFT_MARGIN_PX
                + column * (panel_width + OVERLAY_SHEET_COLUMN_GAP_PX)
            )
            y = y_cursor + (row_height - scaled_height) // 2
            sheet.paste(resized, (x, y))
        draw.multiline_text(
            (OVERLAY_SHEET_LEFT_MARGIN_PX + 38, y_cursor + row_height // 2),
            f"{SMOKE_LABELS[smoke]}\n{LAYER_LABELS[layer]}",
            fill="#20252B",
            font=row_font,
            anchor="rm",
            align="right",
            spacing=8,
        )
        y_cursor += row_height
        if row_index < len(row_keys) - 1:
            y_cursor += OVERLAY_SHEET_ROW_GAP_PX

    legend_top = y_cursor + OVERLAY_SHEET_LEGEND_TOP_GAP_PX
    draw_overlay_sheet_legends(sheet, legend_configs, legend_top)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "all_metrics_overlay__combined.png"
    save_pil_png_with_retry(sheet, target)
    return target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        help=(
            "Comma-separated registry keys or 'all'. Available: "
            + ", ".join(METRIC_SPECS)
        ),
    )
    parser.add_argument(
        "--metric",
        help="Backward-compatible single metric registry key or exported field name.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("individual", "overlay", "both"),
        help="Override the two user-editable render switches.",
    )
    parser.add_argument("--label", help="Optional label override for --metric.")
    parser.add_argument("--unit", help="Optional unit override for --metric.")
    parser.add_argument("--value-min", type=float)
    parser.add_argument("--value-max", type=float)
    parser.add_argument("--value-ticks")
    parser.add_argument("--world-height-max", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--layer", default="all")
    parser.add_argument(
        "--colour-scale",
        choices=("banded", "continuous"),
        default="continuous",
        help="Use discrete tick-aligned colour bands or a continuous gradient.",
    )
    parser.add_argument("--colour-limit-mode", choices=("percentile", "max_fraction", "original"),
                        default=COLOUR_LIMIT_MODE)
    parser.add_argument("--colour-upper-percentile", type=float, default=COLOUR_UPPER_PERCENTILE)
    parser.add_argument("--colour-max-fraction", type=float, default=COLOUR_MAX_FRACTION)
    return parser.parse_args()


def parse_ticks(value: str) -> list[float]:
    ticks = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(ticks) < 2:
        raise ValueError("--value-ticks requires at least two comma-separated numbers")
    return ticks


def resolve_metric_specs(args) -> list[MetricSpec]:
    if args.metrics and args.metric:
        raise ValueError("Use either --metrics or --metric, not both")
    if args.metric:
        matches = [
            spec
            for spec in METRIC_SPECS.values()
            if args.metric in {spec.key, spec.field}
        ]
        if not matches:
            raise ValueError(
                f"Unknown --metric {args.metric!r}; choose a registry key or field"
            )
        spec = matches[0]
        ticks = (
            tuple(parse_ticks(args.value_ticks))
            if args.value_ticks
            else spec.value_ticks
        )
        spec = replace(
            spec,
            label=args.label or spec.label,
            unit=args.unit or spec.unit,
            value_min=args.value_min if args.value_min is not None else spec.value_min,
            value_max=args.value_max if args.value_max is not None else spec.value_max,
            value_ticks=ticks,
            world_height_max=(
                args.world_height_max
                if args.world_height_max is not None
                else spec.world_height_max
            ),
        )
        specs = [spec]
    elif args.metrics:
        names = (
            list(METRIC_SPECS)
            if args.metrics.strip().lower() == "all"
            else [item.strip() for item in args.metrics.split(",") if item.strip()]
        )
        unknown = [name for name in names if name not in METRIC_SPECS]
        if unknown:
            raise ValueError(
                f"Unknown metric keys {unknown}; choose from {tuple(METRIC_SPECS)}"
            )
        specs = [METRIC_SPECS[name] for name in names]
    else:
        specs = [
            METRIC_SPECS[name]
            for name, enabled in METRIC_SWITCHES.items()
            if enabled
        ]
    if not specs:
        raise ValueError("No metrics are enabled")
    for spec in specs:
        if spec.value_max <= spec.value_min:
            raise ValueError(f"Invalid value range for {spec.key}")
        if (
            spec.value_ticks[0] < spec.value_min
            or spec.value_ticks[-1] > spec.value_max
        ):
            raise ValueError(f"Value ticks fall outside the range for {spec.key}")
    return specs


def resolve_render_modes(args) -> tuple[bool, bool]:
    if args.render_mode:
        return (
            args.render_mode in {"individual", "both"},
            args.render_mode in {"overlay", "both"},
        )
    return RENDER_INDIVIDUAL_SURFACES, RENDER_ALL_METRICS_OVERLAY


def main() -> None:
    global METRIC_SPECS
    args = parse_args()
    configure_style()
    timestamp_input = load_timestamp_participant_cells()
    METRIC_SPECS = configure_timestamp_metric_specs(timestamp_input)
    zones = load_zones()
    specs = resolve_metric_specs(args)
    saccade_surface_scale = None
    if any(spec.key == "saccade_rate" for spec in specs):
        saccade_surface_scale = calibrate_saccade_scale_from_visible_surfaces(zones)
        specs = resolve_metric_specs(args)
    render_individual, render_overlay_layers = resolve_render_modes(args)
    if not render_individual and not render_overlay_layers:
        raise ValueError("Both render modes are disabled")
    scenarios = SCENARIOS
    layers = DEFAULT_LAYERS
    if args.scenario != "all":
        if args.scenario not in scenarios:
            raise ValueError(f"Unknown scenario {args.scenario!r}; choose from {scenarios}")
        scenarios = (args.scenario,)
    if args.layer != "all":
        if args.layer not in SUPPORTED_LAYERS:
            raise ValueError(
                f"Unknown layer {args.layer!r}; choose from {SUPPORTED_LAYERS}"
            )
        layers = (args.layer,)
    output_root = (
        args.output_dir.resolve()
        if args.output_dir
        else RESULT_ROOT
    )
    output_root.mkdir(parents=True, exist_ok=True)
    colour_scale_audit = {}
    payloads: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]] = {}
    for spec in specs:
        print(f"loading metric: {spec.key} <- {spec.field}")
        payloads[spec.key] = build_runtime_surface_data(
            zones=zones,
            spec=spec,
            scenarios=SCENARIOS,
            layers=tuple(dict.fromkeys(DEFAULT_LAYERS + layers)),
            colour_scale_mode=args.colour_scale,
        )
        config, surface, _ = payloads[spec.key]
        reference = surface.loc[surface["Layer"].isin(DEFAULT_LAYERS)]
        colour_scale_audit[spec.key] = configure_surface_colour_limits(
            config, reference, mode=args.colour_limit_mode,
            percentile=args.colour_upper_percentile, fraction=args.colour_max_fraction)
        print(f"colour range: {spec.key} = {config['colour_min']:g}..{config['colour_max']:g}")

    generated_individual: list[str] = []
    generated_overlay: list[str] = []
    overlay_panel_count = 0
    overlay_layer_layouts: dict[str, dict] = {}
    if render_individual:
        for spec in specs:
            config, surface, posts = payloads[spec.key]
            output_dir = output_root / safe_name(spec.field)
            for scenario in scenarios:
                for layer in layers:
                    surface_part = surface.loc[
                        surface["Scenario"].eq(scenario)
                        & surface["Layer"].eq(layer)
                    ].copy()
                    post_part = posts.loc[
                        posts["Scenario"].eq(scenario)
                        & posts["Layer"].eq(layer)
                    ].copy()
                    target = render_one(
                        scenario,
                        layer,
                        surface_part,
                        post_part,
                        zones,
                        config,
                        output_dir,
                    )
                    generated_individual.append(str(target))
                    print(f"generated: {target}")

    if render_overlay_layers:
        panel_paths: dict[tuple[str, str], Path] = {}
        panel_payloads: dict[
            tuple[str, str],
            list[tuple[dict, pd.DataFrame, pd.DataFrame]],
        ] = {}
        panel_geometries: dict[tuple[str, str], dict] = {}
        overlay_dir = output_root / "All_Metrics_Overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for scenario in scenarios:
            for layer in layers:
                selected_payloads = []
                for spec in specs:
                    config, surface, posts = payloads[spec.key]
                    selected_payloads.append(
                        (
                            config,
                            surface.loc[
                                surface["Scenario"].eq(scenario)
                                & surface["Layer"].eq(layer)
                            ].copy(),
                            posts.loc[
                                posts["Scenario"].eq(scenario)
                                & posts["Layer"].eq(layer)
                            ].copy(),
                        )
                    )
                panel_key = (scenario, layer)
                panel_payloads[panel_key] = selected_payloads
                panel_geometries[panel_key] = prepare_overlay_geometry(
                    scenario,
                    layer,
                    selected_payloads,
                    zones,
                )

        overlay_layer_layouts = standardize_overlay_geometries_by_layer(
            panel_geometries
        )
        for scenario in scenarios:
            for layer in layers:
                panel_key = (scenario, layer)
                panel_target = render_overlay(
                    scenario,
                    layer,
                    panel_payloads[panel_key],
                    zones,
                    overlay_dir,
                    show_legend=False,
                    geometry=panel_geometries[panel_key],
                )
                panel_paths[panel_key] = panel_target
                generated_overlay.append(str(panel_target))
                overlay_panel_count += 1
                print(f"generated overlay panel: {panel_target}")
        combined_target = combine_overlay_panels(
            panel_paths,
            scenarios,
            layers,
            [payloads[spec.key][0] for spec in specs],
            overlay_dir,
        )
        generated_overlay.append(str(combined_target))
        print(f"generated combined overlay: {combined_target}")
    audit = {
        "input": str(TIMESTAMP_PARTICIPANT_CELL_FILE),
        "input_kind": "timestamp-local participant 2.0 m cell metrics",
        "saccade_input_metric": "saccade_rate",
        "saccade_method": (
            "literal EyeIndicatorAssessment.py detector run once per complete "
            "trial; each event assigned once to its onset 2 m cell; 60 * raw "
            "event count / (valid Hit1 samples / 45 Hz)"
        ),
        "saccade_surface_scale": saccade_surface_scale,
        "data_driven_limits": {
            key: {
                "min": spec.value_min,
                "max": spec.value_max,
                "ticks": list(spec.value_ticks),
                "observed_min": float(
                    timestamp_input.loc[
                        timestamp_input["Metric"].eq(key)
                        & timestamp_input["Supported"],
                        "Value",
                    ].min()
                ),
                "observed_max": float(
                    timestamp_input.loc[
                        timestamp_input["Metric"].eq(key)
                        & timestamp_input["Supported"],
                        "Value",
                    ].max()
                ),
                "limit_method": (
                    "final visible surface 1st-99th percentile, rounded; "
                    "lower fixed at zero"
                    if key == "saccade_rate"
                    else "overall supported participant-cell 1st-99th "
                    "percentile, rounded; lower fixed at zero for "
                    "non-negative metrics"
                ),
            }
            for spec in specs
            for key in (spec.key,)
        },
        "metrics": [spec.key for spec in specs],
        "metric_fields": [spec.field for spec in specs],
        "colour_scale_calibration": colour_scale_audit,
        "metric_value_ranges": {
            spec.key: [spec.value_min, spec.value_max] for spec in specs
        },
        "palette_stops": {
            spec.key: list(spec.palette_stops) for spec in specs
        },
        "palette_numbers": {
            spec.key: METRIC_PALETTE_NUMBERS[spec.key] for spec in specs
        },
        "guidance_display_names": GUIDANCE_LABELS,
        "layer_display_names": LAYER_LABELS,
        "individual_png": len(generated_individual),
        "overlay_panel_count": overlay_panel_count,
        "overlay_png": len(generated_overlay),
        "overlay_output_mode": "scenario_layer_panels_and_combined_sheet",
        "missing_data_display": "minimum cmap colour on exact footprint at minimum surface height; data unchanged; holes preserved",
        "generated_png": len(generated_individual) + len(generated_overlay),
        "dpi": OUTPUT_DPI,
        "font": "Times New Roman",
        "colour_scale": args.colour_scale,
        "reverse_all_metric_palettes": REVERSE_ALL_METRIC_PALETTES,
        "metric_palette_reversed": METRIC_PALETTE_REVERSED,
        "surface_method": SURFACE_METHOD,
        "gaussian_kernel_bandwidth_m": GAUSSIAN_KERNEL_BANDWIDTH_M,
        "gaussian_kernel_truncate_sigma": GAUSSIAN_KERNEL_TRUNCATE_SIGMA,
        "cell_m": CELL_M,
        "grid_m": GRID_M,
        "uniform_overlay_interval_layers": list(
            UNIFORM_OVERLAY_INTERVAL_LAYERS
        ),
        "overlay_layer_layouts": overlay_layer_layouts,
        "overlay_legend": {
            "location": "one horizontal board below all overview panels",
            "bar_height_px": 58,
            "tick_fontsize_px": 36,
            "label_fontsize_px": 46,
        },
        "combined_relief_height_m": COMBINED_RELIEF_HEIGHT_M,
        "combined_layer_offset_scale": COMBINED_LAYER_OFFSET_SCALE,
        "output_dir": str(output_root),
        "no_intermediate_data_written": True,
    }
    audit_path = output_root / "render_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
