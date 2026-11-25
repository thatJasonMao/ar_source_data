#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialSlice_TrajPoint
======================

按"场景 = 组别"的分组策略（10 个实验场景 = 10 个实验组别），提取每个场景
下全部受试者的**全量轨迹点**（Subject_Simulation_Info CSV 的每一帧），
并基于既有空间分区能力（UnifiedZoneQuery + semantic_zones.json, 39 分区）
把每个轨迹点归入其所属空间分区，最终输出：

  SLICE/Results_SpatialSlice_TrajPoint/<场景>/
      points_<zone_id>.csv    —— 落在该分区内的全部轨迹点 (x, y, z, frame, subject, src_file)
      heatmap_<zone_id>.png   —— 该分区轨迹点的原始计数热力图（hist2d，非 KDE）
      scatter_<zone_id>.png   —— 该分区轨迹点散点图（热力图的校正参照）
      summary.csv             —— 场景级分区统计汇总表
      scene_meta.json         —— 场景元信息（受试者数 / 文件数 / 点数）

设计说明
--------
1. 轨迹点收集（步骤 1）：全部在内存中完成（数据容器），不产生中间实体文件；
   遍历场景目录 -> 受试者子目录 -> *Subject_Simulation_Info*.csv，逐行取
   (PosX, PosY, PosZ)（Unity 世界坐标，直接可喂给分区查询）。
2. 分区归属（步骤 2）：复用 UnifiedZoneQuery 的统一分区语义（分类时修正地板高度漂移）
   （Y 定楼层 Platform/Hall -> XZ collider 包含 -> priority/面积/zone_id 仲裁），
   但以 numpy 向量化实现（分块广播），与 UnifiedZoneQuery.classify_point
   输出完全一致；不在任何分区内的点归入 "Null"（仅统计，不画图）。
3. 热力图（步骤 3）：np.hist2d 原始计数（非 KDE），plasma 色图，红色分区
   collider 线框，坐标范围为分区包围盒外扩 5% 余量；网格默认自适应
   （目标每 bin 平均约 10 个点，clamp 到 [10, 150]，--bins 可固定覆盖）；
   色标 vmax 默认取计数的 99 分位（--vmax-percentile 100 恢复为真实最大值），
   仅截断显示范围、不改变计数值；图内所有文字均为英文（Times New Roman），
   含坐标轴刻度值标注。
4. 幂等：场景输出目录中 summary.csv 已存在时跳过（--force 强制重算）。

用法
----
  python SpatialSlice_TrajPoint.py [--root DIR] [--output DIR]
                                   [--scenarios S1,S2] [--bins N]
                                   [--dpi N] [--vmax-percentile P] [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# 确保可从任意工作目录 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import SCENARIO_FOLDERS, PROJECT_ROOT, RAW_DATA_DIR  # noqa: E402
from UnifiedZoneQuery import UnifiedZoneQuery, VectorizedZoneClassifier  # noqa: E402

DEFAULT_OUTPUT_NAME = "Results_SpatialSlice_TrajPoint"
SUBJECT_FILE_KEYWORD = "Subject_Simulation_Info"
NULL_ZONE = "Null"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


class TrajectoryPointCollector:
    """从场景目录收集全部受试者的全量轨迹点（内存数据容器，不落盘）。

    返回的数组：
        pts      (N, 3) float64 —— Unity 世界坐标 (x, y, z)
        frames   (N,)  int64    —— 源 CSV 内的数据行号（1-based，表头行除外）
        subjects (N,)  str      —— 受试者目录名
        srcs     (N,)  str      —— 源文件名（含时间戳，可回查）
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def iter_subject_files(self, scenario: str):
        """产出 (受试者目录名, Subject CSV 路径)。"""
        scenario_dir = self.root / scenario
        if not scenario_dir.is_dir():
            return
        for subject_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            for csv_path in sorted(subject_dir.glob("*{}*.csv".format(SUBJECT_FILE_KEYWORD))):
                yield subject_dir.name, csv_path

    @staticmethod
    def read_points(csv_path: Path):
        """逐行读取 Subject CSV，返回 (x, y, z, frame) 数组。

        CSV 列为 TimeStamp, PosX, PosY, PosZ, ...（行尾带尾随逗号，
        csv.reader 自动处理）；表头行因 float() 转换失败而被自然跳过。
        """
        xs, ys, zs, frames = [], [], [], []
        with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            for row_idx, row in enumerate(reader):
                if len(row) <= 3:
                    continue
                try:
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                except (ValueError, IndexError):
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(z)
                frames.append(row_idx)
        return (np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
                np.asarray(zs, dtype=np.float64),
                np.asarray(frames, dtype=np.int64))

    def collect(self, scenario: str):
        """收集某场景下全部受试者的全量轨迹点。

        返回 (pts, frames, subjects, srcs, n_subjects, n_files)。
        """
        x_parts, y_parts, z_parts, f_parts = [], [], [], []
        subj_parts, src_parts = [], []
        subject_set = set()
        n_files = 0

        for subject, csv_path in tqdm(
            list(self.iter_subject_files(scenario)),
            desc="Collect [{}]".format(scenario),
        ):
            xs, ys, zs, frames = self.read_points(csv_path)
            if len(xs) == 0:
                continue
            x_parts.append(xs)
            y_parts.append(ys)
            z_parts.append(zs)
            f_parts.append(frames)
            subj_parts.extend([subject] * len(xs))
            src_parts.extend([csv_path.name] * len(xs))
            subject_set.add(subject)
            n_files += 1

        if not x_parts:
            empty = np.zeros((0, 3), dtype=np.float64)
            return empty, np.zeros(0, dtype=np.int64), [], [], 0, 0

        pts = np.column_stack([
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            np.concatenate(z_parts),
        ])
        frames = np.concatenate(f_parts)
        return pts, frames, subj_parts, src_parts, len(subject_set), n_files


# 兼容旧导入路径：分类器实现已迁入根目录 UnifiedZoneQuery.py。


def write_points_csv(out_path: Path, pts, frames, subjects, srcs):
    """写入某分区内的全部轨迹点（散点值实体 CSV）。"""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "z", "frame", "subject", "src_file"])
        for i in range(len(pts)):
            writer.writerow([
                "{:.6f}".format(pts[i, 0]),
                "{:.6f}".format(pts[i, 1]),
                "{:.6f}".format(pts[i, 2]),
                int(frames[i]),
                subjects[i],
                srcs[i],
            ])


def compute_bins(n_points, xspan, zspan, bins):
    """计算正方形网格 (nx, nz)：nx / nz = xspan / zspan，保证每个 bin 为正方形。

    bins 为 None 时按点数自适应（目标每 bin 平均约 10 个点）；
    否则以 bins 作为每轴平均格数（总格数约 bins^2）。
    两轴格数按跨度比例分配，避免长条分区（如站台 10.5m x 111.1m）
    在等格数下产生扁长 bin。
    """
    if xspan <= 0 or zspan <= 0:
        return 20, 20
    if bins is None:
        per_axis = int(np.clip(round(np.sqrt(max(n_points, 1) / 10.0)), 10, 150))
    else:
        per_axis = int(bins)
    n_total = per_axis ** 2
    ratio = xspan / zspan
    nx = int(np.clip(round(np.sqrt(n_total * ratio)), 5, 400))
    nz = int(np.clip(round(np.sqrt(n_total / ratio)), 5, 400))
    return max(nx, 1), max(nz, 1)


def zone_bounds(zone):
    """分区 collider 包围盒 (xmin, xmax, zmin, zmax)。"""
    xs = [p.xmin for p in zone.collider_parts] + [p.xmax for p in zone.collider_parts]
    zs = [p.zmin for p in zone.collider_parts] + [p.zmax for p in zone.collider_parts]
    return min(xs), max(xs), min(zs), max(zs)


def draw_zone_heatmap(zone, pts, scenario, out_path, bins, dpi, vmax_percentile):
    """绘制单分区轨迹点原始计数热力图（hist2d，非 KDE）。

    坐标范围为分区 collider 包围盒外扩 5%（最小 0.5 m）余量；
    热力值为每个网格单元内的原始轨迹点数（无 KDE 平滑）；
    网格为正方形（nx/nz = xspan/zspan，每个 bin 边长 1:1），
    bins 为 None 时按点数自适应（目标每 bin 平均约 10 个点），
    否则以 bins 作为每轴平均格数；
    vmax 取计数分布 vmax_percentile 分位（100 表示真实最大值），
    仅截断显示色标范围，计数值本身不变；
    X/Z 坐标轴比例尺相等（set_aspect equal）；
    红色线框为分区 collider 边界；图内全部文字为英文。
    """
    xmin, xmax, zmin, zmax = zone_bounds(zone)
    xmargin = max(0.5, (xmax - xmin) * 0.05)
    zmargin = max(0.5, (zmax - zmin) * 0.05)
    extent = [xmin - xmargin, xmax + xmargin, zmin - zmargin, zmax + zmargin]

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(pts) == 0:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                ha="center", va="center", fontsize=16)
    else:
        xspan = extent[1] - extent[0]
        zspan = extent[3] - extent[2]
        nx, nz = compute_bins(len(pts), xspan, zspan, bins)
        # pts 列为 (x, y, z)，Unity 水平面为 XZ：取第 0、2 列
        counts, _, _, im = ax.hist2d(
            pts[:, 0], pts[:, 2],
            bins=(nx, nz),
            range=[[extent[0], extent[1]], [extent[2], extent[3]]],
            cmap="plasma",
            vmin=0,
        )
        if vmax_percentile < 100.0:
            im.set_clim(0, float(np.percentile(counts, vmax_percentile)))
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20)
        cbar.set_label("Point Count")

    for p in zone.collider_parts:
        ax.add_patch(Rectangle(
            (p.xmin, p.zmin), p.xmax - p.xmin, p.zmax - p.zmin,
            facecolor="none", edgecolor="red", linewidth=2.0, zorder=4,
        ))

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title("{} - {}".format(zone.zone_id, scenario))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_zone_scatter(zone, pts, scenario, out_path, dpi):
    """绘制单分区全量轨迹点散点图（XZ 平面，作为热力图的校正参照）。

    每个轨迹点以橙色、不透明散点呈现（相对像素点增大 50%），
    可见点云的真实形态与密度；X/Z 坐标轴比例尺相等；
    红色线框为分区 collider 边界；坐标范围与热力图一致；图内全部文字为英文。
    """
    xmin, xmax, zmin, zmax = zone_bounds(zone)
    xmargin = max(0.5, (xmax - xmin) * 0.05)
    zmargin = max(0.5, (zmax - zmin) * 0.05)
    extent = [xmin - xmargin, xmax + xmargin, zmin - zmargin, zmax + zmargin]

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(pts) == 0:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                ha="center", va="center", fontsize=16)
    else:
        # pts 列为 (x, y, z)，Unity 水平面为 XZ：取第 0、2 列
        # 橙色、不透明散点（s=1.5 相对像素点增大 50%），栅格化加速渲染
        ax.scatter(pts[:, 0], pts[:, 2], s=1.5, marker=".", color="orange",
                   alpha=1.0, linewidths=0, rasterized=True)

    for p in zone.collider_parts:
        ax.add_patch(Rectangle(
            (p.xmin, p.zmin), p.xmax - p.xmin, p.zmax - p.zmin,
            facecolor="none", edgecolor="red", linewidth=2.0, zorder=4,
        ))

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title("{} - {} Scatter".format(zone.zone_id, scenario))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_scenario(scenario, args, classifier):
    """处理单个场景：收集 -> 分区归属 -> 写 CSV -> 画热力图。"""
    output_dir = args.output / scenario
    summary_path = output_dir / "summary.csv"
    if summary_path.exists() and not args.force:
        print("skip (output exists, use --force to redo):", scenario)
        return

    collector = TrajectoryPointCollector(args.root)
    pts, frames, subjects, srcs, n_subjects, n_files = collector.collect(scenario)
    total = len(pts)
    print("{}: subjects={}, files={}, points={}".format(
        scenario, n_subjects, n_files, total))

    zone_code = classifier.assign(pts)  # 全局 zone 索引，-1 = Null

    output_dir.mkdir(parents=True, exist_ok=True)

    # 按分区分组（zone_code 排序后切连续段），建立 code -> 索引段 映射
    order = np.argsort(zone_code, kind="stable")
    codes_sorted = zone_code[order]
    boundaries = np.nonzero(np.diff(codes_sorted))[0] + 1
    segments = np.split(order, boundaries)
    seg_by_code = {int(zone_code[seg[0]]): seg for seg in segments}

    summary_rows = []
    # 遍历全部正式分区：0 点分区同样输出空 CSV 与 "No Data" 图，
    # 保证每个场景的分区产物与 39 个功能分区一一对应、可审计
    for code in tqdm(range(len(classifier.zone_list)), desc="Write [{}]".format(scenario)):
        zone = classifier.zone_list[code]
        seg = seg_by_code.get(code)
        if seg is not None and len(seg) > 0:
            seg_pts = pts[seg]
            seg_frames = frames[seg]
            seg_subjects = [subjects[i] for i in seg]
            seg_srcs = [srcs[i] for i in seg]
        else:
            seg_pts = np.zeros((0, 3), dtype=np.float64)
            seg_frames = np.zeros(0, dtype=np.int64)
            seg_subjects = []
            seg_srcs = []

        write_points_csv(output_dir / "points_{}.csv".format(zone.zone_id),
                         seg_pts, seg_frames, seg_subjects, seg_srcs)
        draw_zone_heatmap(
            zone, seg_pts, scenario,
            output_dir / "heatmap_{}.png".format(zone.zone_id),
            args.bins, args.dpi, args.vmax_percentile,
        )
        draw_zone_scatter(
            zone, seg_pts, scenario,
            output_dir / "scatter_{}.png".format(zone.zone_id),
            args.dpi,
        )
        summary_rows.append([
            zone.zone_id, zone.floor_id, len(seg_pts),
            "{:.6f}".format(len(seg_pts) / total if total else 0.0),
            "{:.4f}".format(zone.area_xz),
        ])

    # Null 统计行（不属于任何分区，仅统计不画图）
    null_seg = seg_by_code.get(-1)
    null_n = int(null_seg.size) if null_seg is not None else 0
    summary_rows.append([
        NULL_ZONE, "-", null_n,
        "{:.6f}".format(null_n / total if total else 0.0),
        "0.0000",
    ])

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["zone_id", "floor_id", "point_count", "share_of_total", "area_xz"])
        writer.writerows(summary_rows)

    meta = {
        "scenario": scenario,
        "n_subjects": n_subjects,
        "n_files": n_files,
        "n_points": total,
        "n_null_points": null_n,
        "n_zones_with_points": sum(1 for row in summary_rows if row[0] != NULL_ZONE and row[2] > 0),
    }
    with open(output_dir / "scene_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("done:", scenario, "->", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-scenario full trajectory points, assign them to "
                    "semantic zones, and export zone-wise point CSVs and raw-count heatmaps.")
    parser.add_argument("--root", type=str, default=RAW_DATA_DIR,
                        help="raw data root containing the 10 scenario folders "
                             "(default: <project>/RAW_DATA)")
    parser.add_argument("--output", type=str, default=None,
                        help="output root (default: <project root>/SLICE/Results_SpatialSlice_TrajPoint)")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIO_FOLDERS),
                        help="comma-separated scenario names (default: all 10)")
    parser.add_argument("--bins", type=int, default=None,
                        help="average grid cells per axis (default: adaptive, "
                             "target ~10 points/bin; grid cells are always square)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="heatmap DPI (default: 300)")
    parser.add_argument("--vmax-percentile", type=float, default=99.0,
                        help="colorbar vmax as counts percentile; 100 = true max "
                             "(default: 99, display-only truncation, counts unchanged)")
    parser.add_argument("--force", action="store_true",
                        help="recompute scenarios whose output already exists")
    args = parser.parse_args()

    args.root = Path(args.root)
    args.output = Path(args.output) if args.output else Path(PROJECT_ROOT) / "SLICE" / DEFAULT_OUTPUT_NAME
    args.output.mkdir(parents=True, exist_ok=True)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name in scenarios:
        if name not in SCENARIO_FOLDERS:
            parser.error("unknown scenario: {} (expected one of {})".format(
                name, ", ".join(SCENARIO_FOLDERS)))

    print("loading semantic zones ...")
    query = UnifiedZoneQuery()
    print("zones loaded: {}".format(len(query.zones)))
    classifier = VectorizedZoneClassifier(query)

    for scenario in tqdm(scenarios, desc="Scenario"):
        process_scenario(scenario, args, classifier)

    print("all done. output root:", args.output)


if __name__ == "__main__":
    main()
