#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialSlice_Traj
=================

绘制各功能分区内各组别（A / B / C）的受试者轨迹（纯白背景，不叠底图）。

与 SpatialSlice_TrajPoint.py 高度同源，直接复用其核心组件：
  - TrajectoryPointCollector（场景/受试者遍历 + Subject CSV 逐帧读取）；
  - VectorizedZoneClassifier（39 个正式语义分区的向量化归属，
    与 UnifiedZoneQuery.classify_point 语义完全一致）。
分组显示约定（颜色 / 标签 / 图例样式）复用 Extract_Trajectory.py。

绘制策略：
直接使用 Unity 世界坐标 XZ（不做坐标变换、不叠底图、纯白背景），
每张分区图的视口（xlim/ylim）取该分区 collider 包围盒外扩后的范围；
轨迹按"受试者 × 分区"切连续段，只绘制落在该分区内的部分
（受试者进出分区处自动断开，不产生跨段连线）；分区边界以红色线框标出。
跨楼层楼梯分区（Hall-Platform）内受试者连续穿行的轨迹保持连续，
不按楼层人为切断。

产物组织（与 SpatialSlice_TrajPoint 对齐）：
  SLICE/Results_SpatialSlice_Traj/<场景>/
      traj_<zone_id>.png   —— 每个正式分区固定一张图，共 39 张；
                              无轨迹点的分区输出 "No Data" 图
      scene_meta.json      —— 场景元信息（受试者数 / 点数 / 覆盖分区数）

说明：
  - 组内颜色仅在混合场景（StaticSign-* / Full-*，含 A/B/C 全部受试者）
    中呈现多色；引导场景（ArrowRibbon-* / MiniMap-* / TextBar-*）
    各只含本组受试者，图内为单色，图例也只显示实际存在的组；
  - X/Z 轴比例尺一致（set_aspect equal）；图内文字全为英文，
    Times New Roman。

用法：
  python SpatialSlice_Traj.py [--root DIR] [--output DIR]
                              [--scenarios S1,S2] [--dpi N] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS, GROUP_COLORS, GROUP_LABELS, get_subject_group,
)
from Extract_Trajectory import OUTPUT_DPI  # noqa: E402
from UnifiedZoneQuery import UnifiedZoneQuery, VectorizedZoneClassifier  # noqa: E402
from SpatialSlice_TrajPoint import (  # noqa: E402
    TrajectoryPointCollector,
)

DEFAULT_OUTPUT_NAME = "Results_SpatialSlice_Traj"
DEFAULT_DPI = 300

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

GROUP_ORDER = ("A", "B", "C")
GROUP_INDEX = {g: i for i, g in enumerate(GROUP_ORDER)}

# 分区视口外扩（Unity 米）
ZONE_MARGIN_RATIO = 0.05
ZONE_MARGIN_MIN_M = 0.5


def collect_scenario_points(root: Path, scenario: str):
    """读取一个场景下全部受试者的轨迹点。

    返回 dict：
        pts      (N, 3) float64 —— Unity 世界坐标 (x, y, z)
        groups   (N,)  int8     —— 组别编码（A=0, B=1, C=2）
        subjects (N,)  int64    —— 受试者编号（用于分段边界）
        n_subjects int
    无法识别组别的受试者（如 TEST）跳过，与 Extract_Trajectory 行为一致。
    """
    collector = TrajectoryPointCollector(root)
    pts_parts, grp_parts, subj_parts = [], [], []
    subject_id = 0

    for subject, csv_path in tqdm(
        list(collector.iter_subject_files(scenario)),
        desc="Load [{}]".format(scenario),
    ):
        group = get_subject_group(str(csv_path))
        if group is None:
            continue
        xs, ys, zs, _ = collector.read_points(csv_path)
        if len(xs) == 0:
            continue
        pts_parts.append(np.column_stack([xs, ys, zs]))
        grp_parts.append(np.full(len(xs), GROUP_INDEX[group], dtype=np.int8))
        subj_parts.append(np.full(len(xs), subject_id, dtype=np.int64))
        subject_id += 1

    if not pts_parts:
        return None
    return {
        "pts": np.concatenate(pts_parts),
        "groups": np.concatenate(grp_parts),
        "subjects": np.concatenate(subj_parts),
        "n_subjects": subject_id,
    }


def build_zone_segments(data: dict, classifier: VectorizedZoneClassifier) -> dict:
    """把场景轨迹切分为（分区, 组别）二级结构下的连续轨迹段。

    分段规则：同一受试者、同一分区的连续帧为一段；受试者切换 /
    分区切换 / 落到分区外（Null）均断开。不按楼层切断——跨楼层
    楼梯分区（Hall-Platform）内受试者连续穿行的轨迹保持完整。
    轨迹段保留 Unity 原始 (x, z) 坐标。

    返回 {zone_id: {group: [seg_xz (M,2), ...]}}，仅含有数据的分区。
    """
    pts, groups, subjects = data["pts"], data["groups"], data["subjects"]
    n = len(pts)
    codes = classifier.assign(pts)

    valid = codes >= 0
    # 唯一键：受试者 * 1000 + 分区码；无效点取互不相同的负值强制断开
    keys = np.where(valid, subjects * 1000 + codes,
                    -(np.arange(n, dtype=np.int64) + 2))
    bounds = np.nonzero(np.diff(keys))[0] + 1

    zone_by_code = classifier.zone_list
    segments = {}
    for seg in np.split(np.arange(n), bounds):
        if len(seg) < 2:
            continue
        i0 = int(seg[0])
        if not valid[i0]:
            continue
        zone_id = zone_by_code[int(codes[i0])].zone_id
        group = GROUP_ORDER[int(groups[i0])]
        seg_xz = pts[seg][:, (0, 2)]
        segments.setdefault(
            zone_id, {g: [] for g in GROUP_ORDER}
        )[group].append(seg_xz)
    return segments


def zone_viewport(zone):
    """分区 collider 包围盒的视口（Unity 米，外扩后）。

    返回 (x_min, x_max, z_min, z_max)。
    """
    xs = [p.xmin for p in zone.collider_parts] + [p.xmax for p in zone.collider_parts]
    zs = [p.zmin for p in zone.collider_parts] + [p.zmax for p in zone.collider_parts]
    x0, x1 = min(xs), max(xs)
    z0, z1 = min(zs), max(zs)
    xm = max(ZONE_MARGIN_MIN_M, (x1 - x0) * ZONE_MARGIN_RATIO)
    zm = max(ZONE_MARGIN_MIN_M, (z1 - z0) * ZONE_MARGIN_RATIO)
    return x0 - xm, x1 + xm, z0 - zm, z1 + zm


def figure_size_for_viewport(viewport):
    """按视口纵横比决定图幅：长边固定 10 英寸，短边下限 2.2 英寸。"""
    vx0, vx1, vz0, vz1 = viewport
    aspect = (vz1 - vz0) / max(vx1 - vx0, 1e-9)  # 高(z) / 宽(x)
    if aspect >= 1.0:
        fh, fw = 10.0, max(10.0 / aspect, 2.2)
    else:
        fw, fh = 10.0, max(10.0 * aspect, 2.2)
    return fw, fh


def make_group_legend_handles(groups_present):
    """按实际存在的组生成图例元素（样式与 Extract_Trajectory 一致）。"""
    return [
        Line2D(
            [0], [0],
            marker="o",
            color=GROUP_COLORS[g],
            label=GROUP_LABELS.get(g, g),
            markersize=5,
            linestyle="",
        )
        for g in groups_present
    ]


def draw_zone_traj(zone, scenario: str, group_segments: dict,
                   out_path: Path, dpi: int) -> None:
    """绘制单分区轨迹图：纯白背景 + 分区视口 + 各组轨迹段。

    无轨迹点的分区输出 "No Data" 图（与 SpatialSlice_TrajPoint 对齐）；
    图例仅包含该图实际有数据的组。
    """
    viewport = zone_viewport(zone)
    fw, fh = figure_size_for_viewport(viewport)

    fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    groups_present = [g for g in GROUP_ORDER if group_segments.get(g)]
    if not groups_present:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                ha="center", va="center", fontsize=16)
    else:
        for group in groups_present:
            for seg in group_segments[group]:
                ax.plot(
                    seg[:, 0], seg[:, 1],
                    marker="o",
                    color=GROUP_COLORS[group],
                    markersize=0.8,
                    linewidth=0.3,
                    zorder=2,
                )

    # 分区 collider 线框（Unity 原始坐标）
    for p in zone.collider_parts:
        ax.add_patch(Rectangle(
            (p.xmin, p.zmin), p.xmax - p.xmin, p.zmax - p.zmin,
            facecolor="none", edgecolor="red", linewidth=1.5, zorder=3,
        ))

    ax.set_xlim(viewport[0], viewport[1])
    ax.set_ylim(viewport[2], viewport[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("{} - {}".format(zone.zone_id, scenario), fontsize=12)

    if groups_present:
        fig.legend(
            handles=make_group_legend_handles(groups_present),
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=11,
            frameon=False,
        )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_scenario(scenario: str, data: dict, classifier: VectorizedZoneClassifier,
                     args) -> int:
    """处理单个场景：分段 -> 遍历全部 39 个分区绘图。返回输出图数量。"""
    out_dir = args.output / scenario
    marker = out_dir / "scene_meta.json"
    if marker.exists() and not args.force:
        print("skip (output exists, use --force to redo):", scenario)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    segments = build_zone_segments(data, classifier)

    count = 0
    zones_with_traj = 0
    for zone in classifier.zone_list:
        gdict = segments.get(zone.zone_id)
        if gdict is None:
            gdict = {g: [] for g in GROUP_ORDER}
        else:
            zones_with_traj += 1
        draw_zone_traj(
            zone, scenario, gdict,
            out_dir / "traj_{}.png".format(zone.zone_id),
            args.dpi,
        )
        count += 1

    meta = {
        "scenario": scenario,
        "n_subjects": data["n_subjects"],
        "n_points": int(len(data["pts"])),
        "n_zones_total": len(classifier.zone_list),
        "n_zones_with_traj": zones_with_traj,
        "n_figures": count,
    }
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("done:", scenario, "-> {} figures ({} zones with trajectories) in {}".format(
        count, zones_with_traj, out_dir))
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Draw per-zone, per-group evacuation trajectories "
                    "(white background, Unity XZ coordinates, no background map).")
    parser.add_argument("--root", type=str, default=RAW_DATA_DIR,
                        help="raw data root containing the 10 scenario folders "
                             "(default: <project>/RAW_DATA)")
    parser.add_argument("--output", type=str, default=None,
                        help="output root (default: <project root>/SLICE/Results_SpatialSlice_Traj)")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIO_FOLDERS),
                        help="comma-separated scenario names (default: all 10)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help="output DPI (default: {})".format(DEFAULT_DPI))
    parser.add_argument("--force", action="store_true",
                        help="redo scenarios whose output already exists")
    args = parser.parse_args()

    args.root = Path(args.root)
    args.output = Path(args.output) if args.output else Path(PROJECT_ROOT) / "SLICE" / DEFAULT_OUTPUT_NAME

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name in scenarios:
        if name not in SCENARIO_FOLDERS:
            parser.error("unknown scenario: {} (expected one of {})".format(
                name, ", ".join(SCENARIO_FOLDERS)))

    print("loading semantic zones ...")
    query = UnifiedZoneQuery()
    classifier = VectorizedZoneClassifier(query)
    print("zones loaded: {}".format(len(query.zones)))

    args.output.mkdir(parents=True, exist_ok=True)
    total = 0
    for scenario in tqdm(scenarios, desc="Scenario"):
        data = collect_scenario_points(args.root, scenario)
        if data is None:
            print("[skip] no trajectory data:", scenario)
            continue
        total += process_scenario(scenario, data, classifier, args)

    print("all done. {} figures in {}".format(total, args.output))


if __name__ == "__main__":
    main()
