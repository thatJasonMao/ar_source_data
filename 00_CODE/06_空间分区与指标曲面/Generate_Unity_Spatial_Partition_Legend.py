"""Generate transparent voxel legends for Unity spatial-partition assets.

The palette and default translucent alpha are shared with
``Visualize_Unity_Spatial_Partitions.py`` so the legend stays visually aligned
with the exported semantic volume assets.
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

# Keep Matplotlib's runtime cache in a writable temporary location.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatial-partition-legend-mpl")
)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
from PIL import Image

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "Results_Unity_Spatial_Partition_Assets" / "legends"
OUTPUT_STEM = "semantic_legend__volume__translucent"

FILL_ALPHA = 0.62
SIDE_ALPHA = FILL_ALPHA * 0.82


def load_semantic_colors() -> dict[str, str]:
    """Read the canonical palette without importing the heavy renderer stack."""
    source_path = HERE / "Visualize_Unity_Spatial_Partitions.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SEMANTIC_COLORS" for target in node.targets)
        ):
            palette = ast.literal_eval(node.value)
            if not isinstance(palette, dict) or not palette:
                break
            return {str(key): str(value) for key, value in palette.items()}
    raise RuntimeError(f"SEMANTIC_COLORS not found in {source_path}")


SEMANTIC_COLORS = load_semantic_colors()

LEGEND_LABELS = {
    "Platform": "Platform",
    "Spawn": "Spawn",
    "StairEntrance": "Stair entrance",
    "Stair": "Stair",
    "HallInsideGate": "Inside-gate hall",
    "Gate": "Gate",
    "HallOutsideGate": "Outside-gate hall",
    "ExitCorridor": "Exit corridor",
    "DecisionZone": "Decision zone",
    "BranchCorridor": "Branch corridor",
    "Exit": "Exit",
}

ITEM_ORDER = tuple(SEMANTIC_COLORS)


def darken(colour: str, factor: float) -> tuple[float, float, float]:
    return tuple(channel * factor for channel in to_rgb(colour))


def draw_voxel(ax, x: float, y: float, colour: str) -> None:
    """Draw a compact isometric voxel using the renderer's face treatment."""
    top = [(x, y + 0.14), (x + 0.27, y), (x + 0.54, y + 0.14), (x + 0.27, y + 0.28)]
    left = [(x, y + 0.14), (x + 0.27, y + 0.28), (x + 0.27, y + 0.66), (x, y + 0.51)]
    right = [
        (x + 0.27, y + 0.28),
        (x + 0.54, y + 0.14),
        (x + 0.54, y + 0.51),
        (x + 0.27, y + 0.66),
    ]

    face_specs = (
        (left, darken(colour, 0.74), SIDE_ALPHA),
        (right, darken(colour, 0.84), SIDE_ALPHA),
        (top, to_rgb(colour), FILL_ALPHA),
    )
    for vertices, face_colour, alpha in face_specs:
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=(*face_colour, alpha),
                edgecolor="none",
                linewidth=0.0,
                joinstyle="round",
                antialiased=False,
            )
        )


def render() -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10.0,
            "svg.fonttype": "none",
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
        }
    )

    columns = 4
    rows = 3
    cell_width = 2.34
    cell_height = 0.93
    fig = plt.figure(figsize=(9.2, 2.65), dpi=300, frameon=False)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.012, 0.025, 0.976, 0.95])
    ax.patch.set_alpha(0.0)
    ax.set_xlim(0.0, columns * cell_width)
    ax.set_ylim(rows * cell_height, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for index, key in enumerate(ITEM_ORDER):
        row, column = divmod(index, columns)
        x = column * cell_width + 0.08
        y = row * cell_height + 0.10
        draw_voxel(ax, x, y, SEMANTIC_COLORS[key])
        ax.text(
            x + 0.69,
            y + 0.34,
            LEGEND_LABELS[key],
            ha="left",
            va="center",
            color="#20242A",
            fontsize=10.0,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    save_options = {
        "transparent": True,
        "facecolor": (0.0, 0.0, 0.0, 0.0),
        "edgecolor": "none",
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    fig.savefig(svg_path, **save_options)
    fig.savefig(png_path, dpi=300, **save_options)
    plt.close(fig)
    return svg_path, png_path


def audit_png(path: Path) -> tuple[int, int, bool]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        corners = (
            rgba.getpixel((0, 0))[3],
            rgba.getpixel((rgba.width - 1, 0))[3],
            rgba.getpixel((0, rgba.height - 1))[3],
            rgba.getpixel((rgba.width - 1, rgba.height - 1))[3],
        )
        return rgba.width, rgba.height, all(alpha == 0 for alpha in corners)


def main() -> None:
    svg_path, png_path = render()
    width, height, transparent_corners = audit_png(png_path)
    print(svg_path.resolve())
    print(png_path.resolve())
    print(f"PNG: {width}x{height}; transparent corners: {transparent_corners}")


if __name__ == "__main__":
    main()
