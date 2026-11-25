"""Generate Unity-camera-matched transparent spatial-partition assets.

The renderer reproduces a Unity orthographic camera directly instead of using
Matplotlib's approximate 3D camera.  Polygon tops are drawn as complete paths
with holes, so there are no triangulation seams inside a zone.

All source data are read-only. Every generated file is written below
``Results_Unity_Spatial_Partition_Assets`` beside this program.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon as PolygonPatch
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


HERE = Path(__file__).resolve().parent
ROOT = HERE
OUTPUT_DIR = HERE / "Results_Unity_Spatial_Partition_Assets"
COMPOSITE_DIR = OUTPUT_DIR / "composite"
GROUPED_TYPE_DIR = OUTPUT_DIR / "grouped_by_type"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
ZONES_FILE = (
    ROOT
    / "Results_Guidance_Spatial_Causal_Analysis"
    / "zone_annotation"
    / "semantic_zones.csv"
)


@dataclass(frozen=True)
class UnityCamera:
    """Unity camera parameters copied from the Inspector screenshot.

    Unity uses world coordinates (X right, Y up, Z forward).  ``rotation`` is
    interpreted in Unity's Z-X-Y Euler application order.  Orthographic Size is
    half of the vertical view span in world units.
    """

    position_x: float = 81.0
    position_y: float = 71.2
    position_z: float = 69.1
    rotation_x: float = 35.264
    rotation_y: float = 225.0
    rotation_z: float = 0.0
    orthographic_size: float = 53.0
    near_clip: float = 0.3
    far_clip: float = 1000.0
    width_px: int = 1920
    height_px: int = 1080


DEFAULT_CAMERA = UnityCamera()
LAYER_ORDER = ("platform", "stair", "hall")

SEMANTIC_COLORS = {
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

PALETTE_COLORS = {
    "white": "#FFFFFF",
    "black": "#101318",
    "cyan": "#65D5E6",
}

# Coplanar semantic faces use this explicit paint order in composite assets.
# Larger values are drawn later and therefore remain visible.  These overrides
# mirror the formal partition priorities while making the requested hierarchy
# independent of canonical CSV row order.
SEMANTIC_RENDER_PRIORITY = {
    "HallOutsideGate": 30.0,
    "HallInsideGate": 50.0,
    "Gate": 60.0,
}

# Default output per layer/zone: four translucent fills, four opaque fills,
# and four line-only assets. The canvas itself always remains transparent.
DEFAULT_JOBS = (
    ("volume", "semantic", True, "translucent"),
    ("volume", "semantic", False, "translucent"),
    ("flat", "semantic", True, "translucent"),
    ("flat", "semantic", False, "translucent"),
    ("volume", "semantic", True, "opaque"),
    ("volume", "semantic", False, "opaque"),
    ("flat", "semantic", True, "opaque"),
    ("flat", "semantic", False, "opaque"),
    ("wireframe", "white", True, "line-only"),
    ("wireframe", "black", True, "line-only"),
    ("outline", "white", True, "line-only"),
    ("outline", "black", True, "line-only"),
)


@dataclass
class Face:
    depth: float
    priority: float
    patch: object


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
            "path.simplify": False,
        }
    )


@lru_cache(maxsize=128)
def geometry_from_parts(parts_json: str):
    parts = json.loads(parts_json)
    rectangles = [
        box(float(part["xmin"]), float(part["zmin"]), float(part["xmax"]), float(part["zmax"]))
        for part in parts
    ]
    return unary_union(rectangles).buffer(0)


def layer_from_zone_type(zone_type: str) -> str:
    if zone_type in {"Platform", "Spawn", "StairEntrance"}:
        return "platform"
    if zone_type == "Stair":
        return "stair"
    return "hall"


def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(ZONES_FILE, low_memory=False)
    zones["geometry"] = zones["collider_parts_xz"].map(geometry_from_parts)
    zones["asset_layer"] = zones["zone_type"].map(layer_from_zone_type)
    zones["y_min"] = pd.to_numeric(zones["y_min"], errors="raise")
    zones["y_max"] = pd.to_numeric(zones["y_max"], errors="raise")
    zones["priority"] = pd.to_numeric(zones["priority"], errors="raise")
    zones["render_priority"] = zones["zone_type"].map(SEMANTIC_RENDER_PRIORITY)
    zones["render_priority"] = zones["render_priority"].fillna(zones["priority"])
    return zones.sort_values(["canonical_order", "priority", "zone_id"]).reset_index(drop=True)


def iter_polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            if isinstance(item, Polygon):
                yield item


def rotation_matrix_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_matrix_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_matrix_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class UnityOrthographicProjector:
    """Project Unity world points into exact orthographic image pixels."""

    def __init__(self, camera: UnityCamera):
        self.camera = camera
        # Unity Quaternion.Euler applies Z, then X, then Y rotations.
        rotation = (
            rotation_matrix_y(camera.rotation_y)
            @ rotation_matrix_x(camera.rotation_x)
            @ rotation_matrix_z(camera.rotation_z)
        )
        self.right = rotation @ np.asarray([1.0, 0.0, 0.0])
        self.up = rotation @ np.asarray([0.0, 1.0, 0.0])
        self.forward = rotation @ np.asarray([0.0, 0.0, 1.0])
        self.position = np.asarray(
            [camera.position_x, camera.position_y, camera.position_z], dtype=float
        )
        self.aspect = camera.width_px / camera.height_px

    def project(self, world_points: Sequence[Sequence[float]]):
        points = np.asarray(world_points, dtype=float)
        relative = points - self.position
        camera_x = relative @ self.right
        camera_y = relative @ self.up
        depth = relative @ self.forward
        half_height = self.camera.orthographic_size
        half_width = half_height * self.aspect
        pixel_x = (camera_x / half_width + 1.0) * 0.5 * self.camera.width_px
        pixel_y = (1.0 - (camera_y / half_height + 1.0) * 0.5) * self.camera.height_px
        return np.column_stack([pixel_x, pixel_y]), depth

    def basis_dict(self):
        return {
            "right": self.right.tolist(),
            "up": self.up.tolist(),
            "forward": self.forward.tolist(),
            "aspect": self.aspect,
        }


def resolve_colour(zone_type: str, palette: str) -> str:
    if palette == "semantic":
        return SEMANTIC_COLORS.get(zone_type, "#98A1AA")
    return PALETTE_COLORS[palette]


def with_alpha(colour: str, alpha: float):
    r, g, b, _ = to_rgba(colour)
    return r, g, b, float(np.clip(alpha, 0.0, 1.0))


def darken(colour: str, factor: float):
    r, g, b, a = to_rgba(colour)
    return max(r * factor, 0.0), max(g * factor, 0.0), max(b * factor, 0.0), a


def world_loop(loop, unity_y: float):
    return [(float(x), float(unity_y), float(z)) for x, z in loop.coords]


def compound_top_path(poly: Polygon, unity_y: float, projector: UnityOrthographicProjector):
    # Enforce exterior/interior winding so holes remain holes under nonzero fill.
    poly = orient(poly, sign=1.0)
    vertices = []
    codes = []
    depths = []
    for loop in [poly.exterior, *poly.interiors]:
        screen, depth = projector.project(world_loop(loop, unity_y))
        if len(screen) < 3:
            continue
        vertices.extend(screen.tolist())
        codes.extend([MplPath.MOVETO, *([MplPath.LINETO] * (len(screen) - 2)), MplPath.CLOSEPOLY])
        depths.extend(depth.tolist())
    if not vertices:
        return None, np.nan
    return MplPath(np.asarray(vertices), np.asarray(codes)), float(np.mean(depths))


def polygon_true_boundary_segments(
    poly: Polygon, unity_y: float, projector: UnityOrthographicProjector
):
    segments = []
    for loop in [poly.exterior, *poly.interiors]:
        screen, _ = projector.project(world_loop(loop, unity_y))
        segments.extend([[p0, p1] for p0, p1 in zip(screen[:-1], screen[1:])])
    return segments


def side_faces_for_polygon(
    poly: Polygon,
    y_bottom: float,
    y_top: float,
    projector: UnityOrthographicProjector,
    colour: str,
    alpha: float,
    priority: float,
):
    faces = []
    for loop in [poly.exterior, *poly.interiors]:
        coords = [(float(x), float(z)) for x, z in loop.coords]
        for (x0, z0), (x1, z1) in zip(coords[:-1], coords[1:]):
            world = [(x0, y_bottom, z0), (x1, y_bottom, z1), (x1, y_top, z1), (x0, y_top, z0)]
            screen, depths = projector.project(world)
            patch = PolygonPatch(
                screen,
                closed=True,
                facecolor=with_alpha(
                    darken(colour, 0.82),
                    1.0 if np.isclose(alpha, 1.0) else alpha * 0.82,
                ),
                edgecolor="none",
                linewidth=0.0,
                antialiased=False,
            )
            faces.append(Face(float(np.mean(depths)), priority, patch))
    return faces


def build_filled_faces(
    rows: pd.DataFrame,
    projector: UnityOrthographicProjector,
    mode: str,
    palette: str,
    fill_alpha: float,
):
    faces = []
    for row in rows.itertuples():
        colour = resolve_colour(str(row.zone_type), palette)
        priority = float(row.render_priority)
        y_top = float(row.y_max)
        y_bottom = float(row.y_min)
        for polygon in iter_polygons(row.geometry):
            if mode == "volume":
                faces.extend(
                    side_faces_for_polygon(
                        polygon,
                        y_bottom,
                        y_top,
                        projector,
                        colour,
                        fill_alpha,
                        priority,
                    )
                )
            path, depth = compound_top_path(polygon, y_top, projector)
            if path is not None:
                faces.append(
                    Face(
                        depth,
                        priority,
                        PathPatch(
                            path,
                            facecolor=with_alpha(colour, fill_alpha),
                            edgecolor="none",
                            linewidth=0.0,
                            antialiased=False,
                        ),
                    )
                )
    # Low-priority zones are painted first. Higher-priority semantic zones then
    # replace their projected overlaps; camera depth only orders faces within
    # the same semantic priority.
    return sorted(faces, key=lambda item: (item.priority, -item.depth))


def build_line_segments(
    rows: pd.DataFrame,
    projector: UnityOrthographicProjector,
    mode: str,
):
    segments = []
    for row in rows.itertuples():
        y_top = float(row.y_max)
        y_bottom = float(row.y_min)
        for polygon in iter_polygons(row.geometry):
            segments.extend(polygon_true_boundary_segments(polygon, y_top, projector))
            if mode == "wireframe":
                segments.extend(polygon_true_boundary_segments(polygon, y_bottom, projector))
                for loop in [polygon.exterior, *polygon.interiors]:
                    coords = [(float(x), float(z)) for x, z in loop.coords[:-1]]
                    for x, z in coords:
                        screen, _ = projector.project([(x, y_bottom, z), (x, y_top, z)])
                        segments.append([screen[0], screen[1]])
    return segments


def add_true_boundaries(
    ax,
    rows: pd.DataFrame,
    projector: UnityOrthographicProjector,
    colour: str,
    alpha: float,
    line_width: float,
    volume: bool,
):
    segments = build_line_segments(rows, projector, "wireframe" if volume else "outline")
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=[with_alpha(colour, alpha)],
                linewidths=line_width,
                antialiaseds=True,
                capstyle="round",
                joinstyle="round",
            )
        )


def transparent_canvas(camera: UnityCamera, dpi: int):
    figure = plt.figure(
        figsize=(camera.width_px / dpi, camera.height_px / dpi), dpi=dpi, frameon=False
    )
    figure.patch.set_alpha(0.0)
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_xlim(0.0, camera.width_px)
    axes.set_ylim(camera.height_px, 0.0)
    axes.set_aspect("equal", adjustable="box")
    axes.set_axis_off()
    axes.patch.set_alpha(0.0)
    return figure, axes


def safe_name(value: str):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")


def output_name(stem: str, mode: str, palette: str, border: bool, opacity: str):
    border_tag = "border" if border else "noborder"
    if mode in {"wireframe", "outline"}:
        border_tag = "lines"
    if opacity == "opaque" and mode in {"volume", "flat"}:
        return f"{safe_name(stem)}__{mode}__{palette}__opaque__{border_tag}.png"
    return f"{safe_name(stem)}__{mode}__{palette}__{border_tag}.png"


def render_asset(
    rows: pd.DataFrame,
    target_dir: Path,
    stem: str,
    projector: UnityOrthographicProjector,
    camera: UnityCamera,
    mode: str,
    palette: str,
    border: bool,
    opacity: str,
    fill_alpha: float,
    edge_alpha: float,
    line_width: float,
    dpi: int,
):
    figure, axes = transparent_canvas(camera, dpi)
    if mode in {"volume", "flat"}:
        for face in build_filled_faces(rows, projector, mode, palette, fill_alpha):
            axes.add_patch(face.patch)
        if border:
            edge_colour = "#FFFFFF" if palette == "semantic" else resolve_colour("", palette)
            add_true_boundaries(
                axes,
                rows,
                projector,
                edge_colour,
                edge_alpha,
                line_width,
                volume=mode == "volume",
            )
    else:
        line_colour = resolve_colour("", palette)
        add_true_boundaries(
            axes,
            rows,
            projector,
            line_colour,
            edge_alpha,
            line_width,
            volume=mode == "wireframe",
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / output_name(stem, mode, palette, border, opacity)
    figure.savefig(
        path,
        dpi=dpi,
        transparent=True,
        facecolor=(0.0, 0.0, 0.0, 0.0),
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
        metadata={"Software": "Visualize_Unity_Spatial_Partitions.py / Unity orthographic renderer"},
    )
    plt.close(figure)
    return path


def alpha_corner_audit(path: Path):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        corners = [
            rgba.getpixel((0, 0))[3],
            rgba.getpixel((rgba.width - 1, 0))[3],
            rgba.getpixel((0, rgba.height - 1))[3],
            rgba.getpixel((rgba.width - 1, rgba.height - 1))[3],
        ]
        return rgba.width, rgba.height, all(value == 0 for value in corners)


def build_jobs(mode: str, palette: str, border: str, opacity: str):
    if (mode, palette, border, opacity) == ("all", "auto", "both", "both"):
        return list(DEFAULT_JOBS)
    modes = ["volume", "flat", "wireframe", "outline"] if mode == "all" else [mode]
    jobs = []
    for selected_mode in modes:
        if palette == "auto":
            palettes = ["semantic"] if selected_mode in {"volume", "flat"} else ["white", "black"]
        else:
            palettes = [palette]
        borders = [True, False] if border == "both" and selected_mode in {"volume", "flat"} else [border != "off"]
        if selected_mode in {"wireframe", "outline"}:
            borders = [True]
            opacities = ["line-only"]
        else:
            opacities = ["translucent", "opaque"] if opacity == "both" else [opacity]
        jobs.extend(
            (selected_mode, selected_palette, selected_border, selected_opacity)
            for selected_palette in palettes
            for selected_border in borders
            for selected_opacity in opacities
        )
    return jobs


def effective_alphas(opacity: str, fill_alpha: float, edge_alpha: float):
    if opacity == "opaque":
        return 1.0, 1.0
    return float(np.clip(fill_alpha, 0.0, 1.0)), float(np.clip(edge_alpha, 0.0, 1.0))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render transparent partition assets with a Unity orthographic camera."
    )
    parser.add_argument(
        "--scope",
        choices=["all", "both", "composite", "grouped", "individual"],
        default="all",
        help="'all' includes layer composites, semantic-type groups and individual zones; 'both' keeps the legacy composite+individual behavior.",
    )
    parser.add_argument("--layer", choices=["all", *LAYER_ORDER], default="all")
    parser.add_argument("--zone", help="Optional exact zone_id; implies individual output.")
    parser.add_argument(
        "--zone-type",
        help="Optional exact zone_type; implies grouped-by-type output.",
    )
    parser.add_argument("--mode", choices=["all", "volume", "flat", "wireframe", "outline"], default="all")
    parser.add_argument("--palette", choices=["auto", "semantic", "white", "black", "cyan"], default="auto")
    parser.add_argument("--border", choices=["both", "on", "off"], default="both")
    parser.add_argument(
        "--opacity",
        choices=["both", "translucent", "opaque"],
        default="both",
        help="Filled-region opacity variants; the PNG canvas remains transparent.",
    )
    parser.add_argument("--camera-x", type=float, default=DEFAULT_CAMERA.position_x)
    parser.add_argument("--camera-y", type=float, default=DEFAULT_CAMERA.position_y)
    parser.add_argument("--camera-z", type=float, default=DEFAULT_CAMERA.position_z)
    parser.add_argument("--rotation-x", type=float, default=DEFAULT_CAMERA.rotation_x)
    parser.add_argument("--rotation-y", type=float, default=DEFAULT_CAMERA.rotation_y)
    parser.add_argument("--rotation-z", type=float, default=DEFAULT_CAMERA.rotation_z)
    parser.add_argument("--ortho-size", type=float, default=DEFAULT_CAMERA.orthographic_size)
    parser.add_argument("--near", type=float, default=DEFAULT_CAMERA.near_clip)
    parser.add_argument("--far", type=float, default=DEFAULT_CAMERA.far_clip)
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA.width_px)
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA.height_px)
    parser.add_argument("--fill-alpha", type=float, default=0.62)
    parser.add_argument("--edge-alpha", type=float, default=0.95)
    parser.add_argument("--line-width", type=float, default=1.35)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def camera_from_args(args):
    return UnityCamera(
        position_x=args.camera_x,
        position_y=args.camera_y,
        position_z=args.camera_z,
        rotation_x=args.rotation_x,
        rotation_y=args.rotation_y,
        rotation_z=args.rotation_z,
        orthographic_size=args.ortho_size,
        near_clip=args.near,
        far_clip=args.far,
        width_px=args.width,
        height_px=args.height,
    )


def main():
    args = parse_args()
    configure_matplotlib()
    zones = load_zones()
    if args.zone and args.zone_type:
        raise ValueError("Use either --zone or --zone-type, not both.")
    if args.zone:
        zones = zones.loc[zones["zone_id"].eq(args.zone)].copy()
        if zones.empty:
            raise ValueError(f"Unknown zone_id: {args.zone}")
        args.scope = "individual"
    if args.zone_type:
        zones = zones.loc[zones["zone_type"].eq(args.zone_type)].copy()
        if zones.empty:
            raise ValueError(f"Unknown zone_type: {args.zone_type}")
        args.scope = "grouped"
    if args.layer != "all":
        zones = zones.loc[zones["asset_layer"].eq(args.layer)].copy()

    camera = camera_from_args(args)
    if camera.width_px <= 0 or camera.height_px <= 0 or camera.orthographic_size <= 0:
        raise ValueError("Camera width, height and orthographic size must be positive.")
    projector = UnityOrthographicProjector(camera)
    jobs = build_jobs(args.mode, args.palette, args.border, args.opacity)
    index_rows = []

    layers = [layer for layer in LAYER_ORDER if layer in set(zones["asset_layer"])]
    if args.scope in {"all", "both", "composite"}:
        for layer in layers:
            layer_rows = zones.loc[zones["asset_layer"].eq(layer)].copy()
            for mode, palette, border, opacity in jobs:
                job_fill_alpha, job_edge_alpha = effective_alphas(
                    opacity, args.fill_alpha, args.edge_alpha
                )
                path = render_asset(
                    layer_rows,
                    COMPOSITE_DIR / layer,
                    layer,
                    projector,
                    camera,
                    mode,
                    palette,
                    border,
                    opacity,
                    job_fill_alpha,
                    job_edge_alpha,
                    max(args.line_width, 0.05),
                    args.dpi,
                )
                width, height, corners_clear = alpha_corner_audit(path)
                index_rows.append(
                    {
                        "scope": "composite",
                        "layer": layer,
                        "zone_id": "",
                        "zone_type": "",
                        "mode": mode,
                        "palette": palette,
                        "border": border if mode in {"volume", "flat"} else "line-only",
                        "opacity": opacity,
                        "width_px": width,
                        "height_px": height,
                        "transparent_corners": corners_clear,
                        "path": str(path.relative_to(OUTPUT_DIR)),
                    }
                )
                print(f"generated: {path}")

    if args.scope in {"all", "grouped"}:
        for zone_type, type_rows in zones.groupby("zone_type", sort=True):
            target = GROUPED_TYPE_DIR / safe_name(str(zone_type))
            type_layers = sorted(set(type_rows["asset_layer"]), key=LAYER_ORDER.index)
            layer_tag = "+".join(type_layers)
            for mode, palette, border, opacity in jobs:
                job_fill_alpha, job_edge_alpha = effective_alphas(
                    opacity, args.fill_alpha, args.edge_alpha
                )
                path = render_asset(
                    type_rows.copy(),
                    target,
                    str(zone_type),
                    projector,
                    camera,
                    mode,
                    palette,
                    border,
                    opacity,
                    job_fill_alpha,
                    job_edge_alpha,
                    max(args.line_width, 0.05),
                    args.dpi,
                )
                width, height, corners_clear = alpha_corner_audit(path)
                index_rows.append(
                    {
                        "scope": "grouped_by_type",
                        "layer": layer_tag,
                        "zone_id": "",
                        "zone_type": str(zone_type),
                        "mode": mode,
                        "palette": palette,
                        "border": border if mode in {"volume", "flat"} else "line-only",
                        "opacity": opacity,
                        "width_px": width,
                        "height_px": height,
                        "transparent_corners": corners_clear,
                        "path": str(path.relative_to(OUTPUT_DIR)),
                    }
                )
                print(f"generated: {path}")

    if args.scope in {"all", "both", "individual"}:
        for row in zones.itertuples():
            one_zone = zones.loc[zones["zone_id"].eq(row.zone_id)].copy()
            target = INDIVIDUAL_DIR / str(row.asset_layer) / safe_name(str(row.zone_id))
            for mode, palette, border, opacity in jobs:
                job_fill_alpha, job_edge_alpha = effective_alphas(
                    opacity, args.fill_alpha, args.edge_alpha
                )
                path = render_asset(
                    one_zone,
                    target,
                    str(row.zone_id),
                    projector,
                    camera,
                    mode,
                    palette,
                    border,
                    opacity,
                    job_fill_alpha,
                    job_edge_alpha,
                    max(args.line_width, 0.05),
                    args.dpi,
                )
                width, height, corners_clear = alpha_corner_audit(path)
                index_rows.append(
                    {
                        "scope": "individual",
                        "layer": str(row.asset_layer),
                        "zone_id": str(row.zone_id),
                        "zone_type": str(row.zone_type),
                        "mode": mode,
                        "palette": palette,
                        "border": border if mode in {"volume", "flat"} else "line-only",
                        "opacity": opacity,
                        "width_px": width,
                        "height_px": height,
                        "transparent_corners": corners_clear,
                        "path": str(path.relative_to(OUTPUT_DIR)),
                    }
                )
                print(f"generated: {path}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(index_rows)
    index.to_csv(MANIFEST_DIR / "asset_index.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "source": str(ZONES_FILE.relative_to(ROOT)),
        "camera": asdict(camera),
        "unity_euler_order": "Z-X-Y",
        "projection": "Orthographic",
        "camera_basis": projector.basis_dict(),
        "no_text_or_axes": True,
        "transparent_background": True,
        "filled_opacity_variants": {
            "translucent": float(np.clip(args.fill_alpha, 0.0, 1.0)),
            "opaque": 1.0,
        },
        "polygon_top_method": "compound path with true holes; no triangulation",
        "internal_triangulation_seams": False,
        "semantic_render_priority": SEMANTIC_RENDER_PRIORITY,
        "face_paint_order": "ascending semantic priority; far-to-near depth within equal priority",
        "zone_count": int(zones["zone_id"].nunique()),
        "zone_type_count": int(zones["zone_type"].nunique()),
        "asset_count": int(len(index)),
        "scope_asset_counts": {
            str(scope): int(count)
            for scope, count in index["scope"].value_counts().sort_index().items()
        },
        "all_corners_transparent": bool(index["transparent_corners"].all()) if len(index) else False,
        "jobs": [
            {
                "mode": mode,
                "palette": palette,
                "border": border,
                "opacity": opacity,
            }
            for mode, palette, border, opacity in jobs
        ],
    }
    (MANIFEST_DIR / "unity_camera_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"index: {MANIFEST_DIR / 'asset_index.csv'}")
    print(f"manifest: {MANIFEST_DIR / 'unity_camera_manifest.json'}")


if __name__ == "__main__":
    main()
