#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialSlice_GazeDepth
======================

在各功能分区平面上绘制"受试者站立位置 → 注视深度 (gaze depth)"的热力分布。
思路与 C1 工程 Compare_GazeDepthTopo.py 一致：先逐帧计算受试者站在每个
位置点时的 gaze depth，再投影到平面网格上求均值，输出"受试者站在哪里时
他的注视深度是多少"的地形热力分布；区别是本脚本按 A5 的 39 个正式语义
分区逐区输出（复用 SpatialSlice_TrajPoint 的分区切片框架）。

gaze depth 定义（逐帧）：
    depth = || Hit1 - EyePos ||
  其中 Hit1 为 EyeGaze CSV 的视线射线首击点世界坐标（Hit1_X/Y/Z 列），
  EyePos 取 Subject CSV 的受试者位置 PosX/PosY/PosZ —— VR 环境中受试者
  位置即头显/相机位置（≈眼位），默认不做眼高偏移（--eye-height 可调）。
  语义：受试者站在某处时，其视线落在多远处（距离越长 = 看得越远）。

数据对齐：
  Subject 与 EyeGaze CSV 在 A5 工程中为 43Hz 逐帧同步（行数一致、时间戳
  逐帧一致），按行索引对齐（防御性截取较短者）；EyeGaze 中 Hit1 全 0
  （射线无命中）或非有限值的帧视为无效帧，整帧剔除，不参与任何统计与绘图。

分区归属：
  复用 UnifiedZoneQuery.VectorizedZoneClassifier（39 个正式语义分区，
  判定语义与 UnifiedZoneQuery 完全一致），以受试者站立位置 (x, y, z) 归属
  分区；不在任何分区内的有效点归入 Null（仅统计，不画图）。

热力图（原始峰值图，无 KDE / 无平滑）：
  在分区 collider 包围盒外扩 5%（最小 0.5 m）的网格内按站立位置分 bin
  （正方形网格，固定 bin 尺寸 0.4 m × 0.4 m，与其他切片脚本对齐），
  每个 bin 的热力值 = 该 bin 内全部有效帧 gaze depth 的算术均值
  （np.histogram2d 加权求和 / 计数，imshow interpolation="nearest"，
  不叠加任何平滑）。每分区输出全体受试者（All）与 A/B/C 各组热力图
  （仅输出该分区实际有数据的分组）；同一分区内 All 与各组图共用同一网格
  与色标范围，可直接对比。色标 vmax 默认取真实最大值（--vmax-percentile
  可下调为分位数截断），仅截断显示范围、不改变数值。

产物组织（与 SpatialSlice_TrajPoint 对齐）：
  SLICE/Results_SpatialSlice_GazeDepth/<场景>/
      points_<zone_id>.csv       —— 分区内逐帧有效数据
                                    (x, y, z, gaze_depth, group, frame, subject, src_file)
      heatmap_<zone_id>.png      —— 全体受试者均值 gaze depth 热力图
      heatmap_<zone_id>_A.png    —— A 组热力图（该分区无该组数据则不输出）
      heatmap_<zone_id>_B.png    —— B 组热力图
      heatmap_<zone_id>_C.png    —— C 组热力图
      summary.csv                —— 场景级分区统计汇总表（全体 + 各组）
      scene_meta.json            —— 场景元信息（受试者数 / 文件数 / 点数）

说明：
  - 无组别的受试者（如 TEST）跳过，与 SpatialSlice_Traj.py 行为一致；
  - 引导场景（ArrowRibbon-* / MiniMap-* / TextBar-*）只含本组受试者，
    其 per-group 图与 All 图数值相同；
  - X/Z 坐标轴比例尺一致（set_aspect equal）；图内文字全为英文，
    Times New Roman；调色板默认 plasma。

用法：
  python SpatialSlice_GazeDepth.py [--root DIR] [--output DIR]
                                   [--scenarios S1,S2]
                                   [--dpi N] [--vmax-percentile P]
                                   [--cmap NAME] [--eye-height M] [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from tqdm import tqdm

# 确保可从任意工作目录 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import SCENARIO_FOLDERS, PROJECT_ROOT, RAW_DATA_DIR, get_subject_group  # noqa: E402
from UnifiedZoneQuery import UnifiedZoneQuery, VectorizedZoneClassifier  # noqa: E402
from SpatialSlice_TrajPoint import (  # noqa: E402
    TrajectoryPointCollector,
    zone_bounds,
)

DEFAULT_OUTPUT_NAME = "Results_SpatialSlice_GazeDepth"
SUBJECT_FILE_KEYWORD = "Subject_Simulation_Info"
GAZE_FILE_KEYWORD = "EyeGaze_Simulation_Info"
NULL_ZONE = "Null"
GROUP_ORDER = ("A", "B", "C")
GROUP_INDEX = {g: i for i, g in enumerate(GROUP_ORDER)}

# VR 头显/相机位置即眼位（Subject Pos），无需额外眼高偏移；如确需修正可 --eye-height
EYE_HEIGHT_M = 0.0

# 热力图网格 bin 尺寸（米，正方形），与其他切片脚本对齐
BIN_SIZE_M = 0.4

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


class GazeDepthPointCollector:
    """逐受试者读取 Subject + EyeGaze CSV，帧对齐后逐帧计算 gaze depth。"""

    def __init__(self, root: Path):
        self.root = Path(root)

    def iter_subject_pairs(self, scenario: str):
        """产出 (受试者目录名, Subject CSV 路径, EyeGaze CSV 路径)。

        Subject 与 EyeGaze 按去掉类型关键字后的文件名基准配对
        （同一试次的两类文件共享同一时间戳命名）；目录下缺任一文件、
        无法配对的试次跳过。
        """
        scenario_dir = self.root / scenario
        if not scenario_dir.is_dir():
            return
        for subject_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            subject_csvs = sorted(
                subject_dir.glob("*{}*.csv".format(SUBJECT_FILE_KEYWORD)))
            gaze_csvs = sorted(
                subject_dir.glob("*{}*.csv".format(GAZE_FILE_KEYWORD)))
            if not subject_csvs or not gaze_csvs:
                continue
            used = set()
            for subj_csv in subject_csvs:
                gaze_csv = self._match_gaze_csv(subj_csv, gaze_csvs, used)
                if gaze_csv is None and len(gaze_csvs) == 1 and id(gaze_csvs[0]) not in used:
                    gaze_csv = gaze_csvs[0]  # 单文件兜底（防文件名基准差异）
                if gaze_csv is not None:
                    used.add(id(gaze_csv))
                    yield subject_dir.name, subj_csv, gaze_csv

    @staticmethod
    def _match_gaze_csv(subj_csv, gaze_csvs, used):
        """按文件名基准（去除类型关键字后）配对 EyeGaze 文件。"""
        base = subj_csv.name.replace(SUBJECT_FILE_KEYWORD, "")
        for gaze_csv in gaze_csvs:
            if id(gaze_csv) in used:
                continue
            if gaze_csv.name.replace(GAZE_FILE_KEYWORD, "") == base:
                return gaze_csv
        return None

    @staticmethod
    def read_gaze_hits(csv_path: Path):
        """逐行读取 EyeGaze CSV 的 Hit1 世界坐标（列 7/8/9）。

        返回 (hx, hy, hz)；表头行与缺列/解析失败行以 NaN 占位，
        保持与 Subject CSV 的行序对齐。
        """
        hxs, hys, hzs = [], [], []
        with open(csv_path, "r", newline="", encoding="utf-8",
                  errors="replace") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) <= 9:
                    hxs.append(np.nan)
                    hys.append(np.nan)
                    hzs.append(np.nan)
                    continue
                try:
                    hx, hy, hz = float(row[7]), float(row[8]), float(row[9])
                except (ValueError, IndexError):
                    hx = hy = hz = np.nan
                hxs.append(hx)
                hys.append(hy)
                hzs.append(hz)
        return (np.asarray(hxs, dtype=np.float64),
                np.asarray(hys, dtype=np.float64),
                np.asarray(hzs, dtype=np.float64))

    def compute_subject_depths(self, subj_csv: Path, gaze_csv: Path,
                               eye_height: float):
        """读取一对 Subject/EyeGaze CSV，返回帧对齐后的数组。

        返回 (xs, ys, zs, depths, frames)：
            depths = ||Hit1 - (Pos + 眼高偏移)||，无效帧（位置非有限 /
            射线无命中 / Hit 非有限）为 NaN。
        """
        xs, ys, zs, frames = TrajectoryPointCollector.read_points(subj_csv)
        hxs, hys, hzs = self.read_gaze_hits(gaze_csv)
        n = min(len(xs), len(hxs))
        xs, ys, zs, frames = xs[:n], ys[:n], zs[:n], frames[:n]
        hxs, hys, hzs = hxs[:n], hys[:n], hzs[:n]

        pos_ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        hit_ok = (np.isfinite(hxs) & np.isfinite(hys) & np.isfinite(hzs)
                  & ((hxs != 0.0) | (hys != 0.0) | (hzs != 0.0)))
        valid = pos_ok & hit_ok

        depths = np.sqrt((hxs - xs) ** 2
                         + (hys - (ys + eye_height)) ** 2
                         + (hzs - zs) ** 2)
        depths = np.where(valid, depths, np.nan)
        return xs, ys, zs, depths, frames


def collect_scenario_depths(root: Path, scenario: str, eye_height: float):
    """读取一个场景下全部受试者的帧对齐 gaze depth 数据。

    返回 dict（无任何有效数据时返回 None）：
        pts             (N, 4) float64 —— (x, y, z, gaze_depth)，仅有效帧
        groups          (N,)  int8     —— 组别编码（A=0, B=1, C=2）
        frames          (N,)  int64    —— 源 Subject CSV 数据行号
        subjects        (N,)  str      —— 受试者目录名
        srcs            (N,)  str      —— Subject 源文件名（可回查）
        n_subjects      int
        n_files         int            —— 成功配对的 (Subject, EyeGaze) 对数
        n_frames_loaded int            —— 帧对齐总行数（含无效帧）
    无组别受试者（如 TEST）跳过，与 SpatialSlice_Traj 行为一致。
    """
    collector = GazeDepthPointCollector(root)
    x_parts, y_parts, z_parts, d_parts, f_parts = [], [], [], [], []
    grp_parts, subj_parts, src_parts = [], [], []
    subject_set = set()
    n_files = 0
    n_frames_loaded = 0

    for subject, subj_csv, gaze_csv in tqdm(
        list(collector.iter_subject_pairs(scenario)),
        desc="Collect [{}]".format(scenario),
    ):
        group = get_subject_group(str(subj_csv))
        if group is None:
            continue
        xs, ys, zs, depths, frames = collector.compute_subject_depths(
            subj_csv, gaze_csv, eye_height)
        if len(xs) == 0:
            continue
        n_frames_loaded += len(xs)
        valid = np.isfinite(depths)
        if not valid.any():
            continue
        x_parts.append(xs[valid])
        y_parts.append(ys[valid])
        z_parts.append(zs[valid])
        d_parts.append(depths[valid])
        f_parts.append(frames[valid])
        grp_parts.append(np.full(int(valid.sum()), GROUP_INDEX[group],
                                 dtype=np.int8))
        subj_parts.extend([subject] * int(valid.sum()))
        src_parts.extend([subj_csv.name] * int(valid.sum()))
        subject_set.add(subject)
        n_files += 1

    if not x_parts:
        return None

    pts = np.column_stack([np.concatenate(x_parts),
                           np.concatenate(y_parts),
                           np.concatenate(z_parts),
                           np.concatenate(d_parts)])
    return {
        "pts": pts,
        "groups": np.concatenate(grp_parts),
        "frames": np.concatenate(f_parts),
        "subjects": subj_parts,
        "srcs": src_parts,
        "n_subjects": len(subject_set),
        "n_files": n_files,
        "n_frames_loaded": n_frames_loaded,
    }


def grid_mean_gaze_depth(xs, zs, depths, x_edges, z_edges):
    """每 bin 内 gaze depth 算术均值；无样本 bin 为 NaN。形状 (nx, nz)。"""
    sum_grid, _, _ = np.histogram2d(xs, zs, bins=[x_edges, z_edges],
                                    weights=depths)
    cnt_grid, _, _ = np.histogram2d(xs, zs, bins=[x_edges, z_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan)
    return mean_grid


def zone_edges(zone):
    """分区网格边缘：包围盒外扩 5%（最小 0.5 m），正方形网格。

    网格由固定 bin 尺寸 BIN_SIZE_M（0.4 m）决定，All 与各组共用同一网格；
    末边补齐到覆盖整个 extent（nx = ceil(span / BIN)），不丢点。
    """
    xmin, xmax, zmin, zmax = zone_bounds(zone)
    xmargin = max(0.5, (xmax - xmin) * 0.05)
    zmargin = max(0.5, (zmax - zmin) * 0.05)
    xspan = (xmax - xmin) + 2 * xmargin
    zspan = (zmax - zmin) + 2 * zmargin
    nx = int(np.ceil(xspan / BIN_SIZE_M))
    nz = int(np.ceil(zspan / BIN_SIZE_M))
    x_edges = (xmin - xmargin) + np.arange(nx + 1) * BIN_SIZE_M
    z_edges = (zmin - zmargin) + np.arange(nz + 1) * BIN_SIZE_M
    return x_edges, z_edges


def draw_zone_gaze_depth_heatmap(zone, scenario, label, xs, zs, depths,
                                 x_edges, z_edges, vmin, vmax, cmap,
                                 out_path, dpi):
    """绘制单分区均值 gaze depth 热力图（原始峰值图，无 KDE / 无平滑）。

    label 为 "All" / "Group A" / "Group B" / "Group C"，写入标题；
    热力值 = 每 bin 内 gaze depth 算术均值，interpolation="nearest"
    直接显示原始网格值；无样本 bin 留白（NaN，不填充）；
    红色线框为分区 collider 边界；坐标范围与网格边缘一致。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    grid = grid_mean_gaze_depth(xs, zs, depths, x_edges, z_edges)
    masked = np.ma.masked_invalid(grid)
    if masked.mask.all():
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                ha="center", va="center", fontsize=16)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        im = ax.imshow(
            masked.T,
            extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20)
        cbar.set_label("Mean Gaze Depth (m)")

    for p in zone.collider_parts:
        ax.add_patch(Rectangle(
            (p.xmin, p.zmin), p.xmax - p.xmin, p.zmax - p.zmin,
            facecolor="none", edgecolor="red", linewidth=2.0, zorder=4,
        ))

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(z_edges[0], z_edges[-1])
    ax.set_aspect("equal")
    ax.set_title("{} - {} - {}".format(zone.zone_id, scenario, label))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_points_csv(out_path, pts, groups, frames, subjects, srcs):
    """写入某分区内的逐帧有效数据（含 gaze depth 与组别）。"""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "z", "gaze_depth", "group",
                         "frame", "subject", "src_file"])
        for i in range(len(pts)):
            writer.writerow([
                "{:.6f}".format(pts[i, 0]),
                "{:.6f}".format(pts[i, 1]),
                "{:.6f}".format(pts[i, 2]),
                "{:.6f}".format(pts[i, 3]),
                GROUP_ORDER[int(groups[i])],
                int(frames[i]),
                subjects[i],
                srcs[i],
            ])


def fmt_mean(value):
    """均值格式化：NaN 输出空串，其余保留 4 位小数。"""
    if not np.isfinite(value):
        return ""
    return "{:.4f}".format(float(value))


def process_scenario(scenario, args, classifier):
    """处理单个场景：收集 -> 分区归属 -> 写 CSV -> 画热力图。"""
    output_dir = args.output / scenario
    summary_path = output_dir / "summary.csv"
    if summary_path.exists() and not args.force:
        print("skip (output exists, use --force to redo):", scenario)
        return

    data = collect_scenario_depths(args.root, scenario, args.eye_height)
    if data is None:
        print("[skip] no valid gaze depth data:", scenario)
        return

    pts, groups = data["pts"], data["groups"]
    total = len(pts)
    print("{}: subjects={}, files={}, frames_loaded={}, valid_points={}".format(
        scenario, data["n_subjects"], data["n_files"],
        data["n_frames_loaded"], total))

    zone_code = classifier.assign(pts[:, :3])  # 全局 zone 索引，-1 = Null

    output_dir.mkdir(parents=True, exist_ok=True)

    # 按分区分组（zone_code 排序后切连续段），建立 code -> 索引段 映射
    order = np.argsort(zone_code, kind="stable")
    codes_sorted = zone_code[order]
    boundaries = np.nonzero(np.diff(codes_sorted))[0] + 1
    segments = np.split(order, boundaries)
    seg_by_code = {int(zone_code[seg[0]]): seg for seg in segments}

    summary_rows = []
    for code in tqdm(range(len(classifier.zone_list)), desc="Write [{}]".format(scenario)):
        zone = classifier.zone_list[code]
        seg = seg_by_code.get(code)
        if seg is None or len(seg) == 0:
            # 空分区：写空 CSV + "No Data" 图（All），不输出各组图
            empty = np.zeros((0, 4), dtype=np.float64)
            write_points_csv(output_dir / "points_{}.csv".format(zone.zone_id),
                             empty, np.zeros(0, dtype=np.int8),
                             np.zeros(0, dtype=np.int64), [], [])
            x_edges, z_edges = zone_edges(zone)
            draw_zone_gaze_depth_heatmap(
                zone, scenario, "All", empty[:, 0], empty[:, 2], empty[:, 3],
                x_edges, z_edges, 0.0, 1.0, args.cmap,
                output_dir / "heatmap_{}.png".format(zone.zone_id), args.dpi)
            summary_rows.append([zone.zone_id, zone.floor_id, 0, "",
                                 "0", "", "0", "", "0", ""])
            continue

        idx = seg
        zpts = pts[idx]
        zgroups = groups[idx]
        xs, zs, depths = zpts[:, 0], zpts[:, 2], zpts[:, 3]
        n = len(idx)

        # 全体网格（All 与各组共用同一网格与色标范围）
        x_edges, z_edges = zone_edges(zone)
        grid_all = grid_mean_gaze_depth(xs, zs, depths, x_edges, z_edges)
        valid_grid = grid_all[np.isfinite(grid_all)]
        if valid_grid.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.min(valid_grid))
            vmax = float(np.max(valid_grid))
            if args.vmax_percentile < 100.0:
                vmax = float(np.percentile(valid_grid, args.vmax_percentile))
            if vmax <= vmin:
                vmax = vmin + 1.0

        write_points_csv(output_dir / "points_{}.csv".format(zone.zone_id),
                         zpts, zgroups, data["frames"][idx],
                         [data["subjects"][i] for i in idx],
                         [data["srcs"][i] for i in idx])
        draw_zone_gaze_depth_heatmap(
            zone, scenario, "All", xs, zs, depths,
            x_edges, z_edges, vmin, vmax, args.cmap,
            output_dir / "heatmap_{}.png".format(zone.zone_id), args.dpi)

        # 各组热力图（仅输出该分区实际有数据的分组）
        group_stats = {}
        for gi, gname in enumerate(GROUP_ORDER):
            gmask = zgroups == gi
            if not gmask.any():
                group_stats[gname] = (0, np.nan)
                continue
            gxs, gzs, gd = xs[gmask], zs[gmask], depths[gmask]
            draw_zone_gaze_depth_heatmap(
                zone, scenario, "Group {}".format(gname), gxs, gzs, gd,
                x_edges, z_edges, vmin, vmax, args.cmap,
                output_dir / "heatmap_{}_{}.png".format(zone.zone_id, gname),
                args.dpi)
            group_stats[gname] = (int(gmask.sum()), float(np.mean(gd)))

        summary_rows.append([
            zone.zone_id, zone.floor_id, n,
            fmt_mean(float(np.mean(depths))),
            str(group_stats["A"][0]), fmt_mean(group_stats["A"][1]),
            str(group_stats["B"][0]), fmt_mean(group_stats["B"][1]),
            str(group_stats["C"][0]), fmt_mean(group_stats["C"][1]),
        ])

    # Null 统计行（不属于任何分区，仅统计不画图）
    null_seg = seg_by_code.get(-1)
    if null_seg is not None and len(null_seg) > 0:
        null_pts = pts[null_seg]
        null_n = len(null_pts)
        null_mean = float(np.mean(null_pts[:, 3]))
    else:
        null_n = 0
        null_mean = np.nan
    summary_rows.append([NULL_ZONE, "-", null_n, fmt_mean(null_mean),
                         "0", "", "0", "", "0", ""])

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["zone_id", "floor_id", "n_points", "mean_gaze_depth",
                         "n_A", "mean_A", "n_B", "mean_B", "n_C", "mean_C"])
        writer.writerows(summary_rows)

    meta = {
        "scenario": scenario,
        "n_subjects": data["n_subjects"],
        "n_files": data["n_files"],
        "n_frames_loaded": data["n_frames_loaded"],
        "n_points": total,
        "n_null_points": null_n,
        "n_zones_with_points": sum(1 for row in summary_rows
                                   if row[0] != NULL_ZONE and row[2] > 0),
    }
    with open(output_dir / "scene_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("done:", scenario, "->", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Per-zone mean gaze depth heatmaps on the ground plane "
                    "(subject standing position -> gaze depth), pooled and "
                    "per group (A/B/C).")
    parser.add_argument("--root", type=str, default=RAW_DATA_DIR,
                        help="raw data root containing the 10 scenario folders "
                             "(default: <project>/RAW_DATA)")
    parser.add_argument("--output", type=str, default=None,
                        help="output root (default: <project root>/SLICE/Results_SpatialSlice_GazeDepth)")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIO_FOLDERS),
                        help="comma-separated scenario names (default: all 10)")

    parser.add_argument("--dpi", type=int, default=300,
                        help="heatmap DPI (default: 300)")
    parser.add_argument("--vmax-percentile", type=float, default=100.0,
                        help="colorbar vmax as grid-means percentile; 100 = true "
                             "max (default: 100, display-only truncation, "
                             "values unchanged)")
    parser.add_argument("--cmap", type=str, default="plasma",
                        help="matplotlib colormap (default: plasma; alias heat/heatmap "
                             "-> hot)")
    parser.add_argument("--eye-height", type=float, default=EYE_HEIGHT_M,
                        help="eye height above Subject PosY in meters "
                             "(default: 0.0, VR camera position == eye)")
    parser.add_argument("--force", action="store_true",
                        help="recompute scenarios whose output already exists")
    args = parser.parse_args()

    args.root = Path(args.root)
    args.output = Path(args.output) if args.output else Path(PROJECT_ROOT) / "SLICE" / DEFAULT_OUTPUT_NAME
    args.output.mkdir(parents=True, exist_ok=True)

    # 兼容别名：heat/heatmap -> hot
    args.cmap = {"heat": "hot", "heatmap": "hot"}.get(args.cmap.lower(), args.cmap)
    try:
        plt.get_cmap(args.cmap)
    except ValueError:
        parser.error("unknown colormap: {}".format(args.cmap))

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
