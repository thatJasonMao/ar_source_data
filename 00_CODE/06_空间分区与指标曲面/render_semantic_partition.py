#!/usr/bin/env python3
"""绘制正式语义区域划分图。

用途说明：读取正式区域几何与审计结果，按照当前方向约定、统一字体和配色，
生成带完整区域文字标注的科研风高分辨率 PNG；它只绘图，不修改区域数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


# Nature-style editable vector output and compact journal typography.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.65
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.55
plt.rcParams["ytick.major.width"] = 0.55
plt.rcParams["xtick.major.size"] = 2.5
plt.rcParams["ytick.major.size"] = 2.5


# Restrained, low-saturation palette. Repeated spatial roles stay in one family.
PALETTE = {
    "Platform": "#B8D4E3",
    "Spawn": "#C9A0B8",
    "StairEntrance": "#E8DFA6",
    "Stair": "#D6A85F",
    "HallInsideGate": "#93DBE0",
    "Gate": "#4E9B8A",
    "HallOutsideGate": "#D9DDE0",
    "ExitCorridor": "#547FA3",
    "DecisionZone": "#C47A5A",
    "BranchCorridor": "#A7A7A7",
    "Exit": "#3F3F3F",
}

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


# Manual positions are used only when a geometric centroid is not visually useful.
LABEL_POSITIONS = {
    ("Hall", "HallInsideGate_Main"): (5.6, -14.5),
    ("Hall", "HallOutsideGate_Main"): (-6.7, -8.0),
    ("Platform", "Platform_Main"): (0.0, -8.0),
    ("Platform", "StairEntrance_EFGH"): (0.0, 22.5),
    ("Platform", "StairEntrance_ABCD"): (0.0, -45.0),
}

ZONE_LABEL_SIZE = 5.0


def flip_coordinates(x, z):
    """Rotate the plan 180 degrees and express it in the flipped X-Z frame."""
    return -x, -z


def zone_label(zone):
    """Return an explicit, unambiguous, wrapped label for every zone."""
    zone_id = zone["zone_id"]
    fixed = {
        "HallInsideGate_Main": "Inside-gate\nhall",
        "HallOutsideGate_Main": "Outside-gate\nhall",
        "Platform_Main": "Platform\nmain",
    }
    if zone_id in fixed:
        return fixed[zone_id]
    prefix_rules = [
        ("StairEntrance_", "Stair\nentrance"),
        ("DecisionZone_", "Decision\nzone"),
        ("ExitCorridor_", "Exit\ncorridor"),
        ("BranchCorridor_", "Branch\ncorridor"),
        ("Stair_", "Stair"),
        ("Gate_0", "Gate"),
        ("Spawn_0", "Spawn"),
        ("Exit_", "Exit"),
    ]
    for prefix, title in prefix_rules:
        if zone_id.startswith(prefix):
            suffix = zone_id[len(prefix):]
            return f"{title}\n{suffix}"
    return zone_id.replace("_", "\n")


def draw_zone(ax, zone):
    """Draw collision rectangles, then the formal polygon boundary."""
    color = PALETTE[zone["zone_type"]]
    for rect in zone["collider_parts"]:
        xmin, zmin = flip_coordinates(rect["xmax"], rect["zmax"])
        xmax, zmax = flip_coordinates(rect["xmin"], rect["zmin"])
        ax.add_patch(
            Rectangle(
                (xmin, zmin),
                xmax - xmin,
                zmax - zmin,
                facecolor=color,
                edgecolor="none",
                alpha=0.88,
                zorder=1 + zone["priority"] / 100,
            )
        )
    for loop in zone["boundary_loops"]:
        flipped = [flip_coordinates(point[0], point[1]) for point in loop]
        xs = [point[0] for point in flipped] + [flipped[0][0]]
        zs = [point[1] for point in flipped] + [flipped[0][1]]
        ax.plot(xs, zs, color="#4D4D4D", linewidth=0.42, zorder=5)


def add_direct_label(ax, zone, panel):
    x, z = LABEL_POSITIONS.get(
        (panel, zone["zone_id"]),
        (zone["center_x"], zone["center_z"]),
    )
    x, z = flip_coordinates(x, z)
    ax.text(
        x,
        z,
        zone_label(zone),
        ha="center",
        va="center",
        multialignment="center",
        linespacing=0.88,
        fontsize=ZONE_LABEL_SIZE,
        fontweight="normal",
        color="#202020",
        zorder=10,
        bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.82},
    )


def style_panel(ax, limits, panel_label, title, xticks, zticks):
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.set_xlabel("X (m)", labelpad=2)
    ax.set_ylabel("Z (m)", labelpad=2)
    ax.tick_params(labelsize=6, colors="#4D4D4D")
    ax.set_title(title, fontsize=7.2, fontweight="bold", pad=4)
    ax.text(
        -0.08,
        1.015,
        panel_label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.65)
    ax.grid(False)


def render(zones, audit, output_png):
    if audit.get("overlap_count") != 0:
        raise ValueError("Refusing publication export: formal partition still contains overlaps")
    if audit.get("zone_count") != len(zones):
        raise ValueError("Geometry/audit zone counts do not match")

    hall_zones = [zone for zone in zones if "Hall" in zone["floor_id"]]
    platform_zones = [
        zone for zone in zones
        if "Platform" in zone["floor_id"] or zone["zone_type"] == "Spawn"
    ]
    hall_height = next(
        zone["size_y"] for zone in zones
        if zone["floor_id"] == "Hall" and zone["zone_type"] != "Stair"
    )
    platform_height = next(
        zone["size_y"] for zone in zones
        if zone["floor_id"] == "Platform" and zone["zone_type"] != "Stair"
    )

    # Double-column width (183 mm); tall spatial geometry is preserved at equal scale.
    fig = plt.figure(figsize=(183 / 25.4, 190 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 0.085],
        width_ratios=[3.7, 1.05],
        left=0.075,
        right=0.985,
        top=0.965,
        bottom=0.045,
        wspace=0.23,
        hspace=0.10,
    )
    hall_ax = fig.add_subplot(grid[0, 0])
    platform_ax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[1, :])

    for zone in sorted(hall_zones, key=lambda item: item["priority"]):
        draw_zone(hall_ax, zone)
    for zone in sorted(hall_zones, key=lambda item: item["priority"], reverse=True):
        add_direct_label(hall_ax, zone, "Hall")

    for zone in sorted(platform_zones, key=lambda item: item["priority"]):
        draw_zone(platform_ax, zone)
    for zone in sorted(platform_zones, key=lambda item: item["priority"], reverse=True):
        add_direct_label(platform_ax, zone, "Platform")

    style_panel(
        hall_ax,
        (-45, 45, -65, 90),
        "a",
        f"Hall level ({hall_height:.3g} m)",
        [-40, -20, 0, 20, 40],
        [-40, 0, 40, 80],
    )
    style_panel(
        platform_ax,
        (-12, 12, -43, 67),
        "b",
        f"Platform level ({platform_height:.4g} m)",
        [-10, 0, 10],
        [-40, 0, 40, 60],
    )

    legend_ax.axis("off")
    handles = [
        Patch(facecolor=PALETTE[key], edgecolor="#4D4D4D", linewidth=0.35, label=LEGEND_LABELS[key])
        for key in PALETTE
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=6,
        fontsize=6,
        handlelength=1.25,
        handleheight=0.8,
        columnspacing=1.25,
        labelspacing=0.65,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    save_common = {"bbox_inches": "tight", "pad_inches": 0.025, "facecolor": "white"}
    fig.savefig(output_png, dpi=600, **save_common)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()

    zones = json.loads(args.geometry.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    render(zones, audit, args.output_png)
    print(args.output_png.resolve())


if __name__ == "__main__":
    main()
