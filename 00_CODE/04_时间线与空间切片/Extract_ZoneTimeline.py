# -*- coding: utf-8 -*-
"""
Extract_ZoneTimeline
====================

受试者关键空间节点进出时间线提取。

试次唯一来源为 common.load_trials()，固定45人/270试次；禁止自行扫描或纳入重复会话。

对每个场景 × 每个受试者，基于 Subject_Simulation_Info 逐帧轨迹
（Unity 坐标系，XZ 水平面、Y 垂直轴；位置列为 CSV 的 B/C/D 列
PosX/PosY/PosZ）与正式语义分区（semantic_zones.json，39 分区，
复用 UnifiedZoneQuery + VectorizedZoneClassifier 的归属判定），
提取以下 6 个事件的时间戳：

  a. 离开 Spawn           —— Spawn_01 / Spawn_02 任一，最后一次在区内之后
                             的首个有效帧（即最终离开时刻）；
  b. 离开 StairEntrance   —— StairEntrance_ABCD / StairEntrance_EFGH 任一，
                             同上（最终离开时刻）；
  c. 进入 HallInsideGate  —— HallInsideGate_Main 的首个命中帧；
  d. 离开 HallInsideGate  —— HallInsideGate_Main 最后一次在区内之后的
                             首个有效帧；
  e. 离开 DecisionZone    —— DecisionZone_ABCD / DecisionZone_EFGH 任一，
                             最终离开时刻；
  f. 进入 Exit            —— Exit_A ~ Exit_H 任一的首个命中帧。

时间戳约定
----------
所有时间戳一律取源 CSV A 列 TimeStamp 的原始字符串
（形如 2026_06_04_20_16_17_020），不做任何换算或插值，
保证可直接反查源数据；同时输出 frame（源 CSV 内数据行号，
1-based，表头行除外）与实际命中的 zone_id（ABCD/EFGH 二选一
的事件报告最后命中的具体分区）。

数据清洗
--------
全零帧（PosX=PosY=PosZ=0，为记录启动前的无效原点帧，且原点
(0,0,0) 恰落在 HallInsideGate_Main 几何内，会造成假进入）
在归属判定前整体剔除；frame 编号仍按源 CSV 行号记录，剔除不
影响反查。

产物组织
--------
  TIMELINE/subject_zone_timeline.csv
      每行 = 一个场景 × 一个受试者 × 一个源文件：
      scenario, subject, group, src_file, n_valid_frames,
      各事件的 <event>_ts / <event>_frame / <event>_zone 三元组。
      缺失事件的 ts 写为 Null+最后有效位置的 zone_id，frame/zone 留空；
      此标记不是时间戳，也不表示在终点发生了该事件。
      terminal_ts/frame/zone/x/y/z 记录最后有效帧（XYZ 为吸附前原始值）。
      timeline_order 检查已捕获事件的原始时间戳：strict / equal /
      reversed / insufficient；n_missing_events 表示缺失数。
      缺失节点不参与比较；strict 且 n_missing_events=0 才代表六节点严格递增。

用法
----
  python Extract_ZoneTimeline.py [--root DIR] [--output DIR] [--scenarios ...]
"""

import argparse
import csv
from datetime import datetime
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TimelineTrialInput import (canonical_trials, load_timeline)
from common import (  # noqa: E402
    SCENARIO_FOLDERS, PROJECT_ROOT, RAW_DATA_DIR,
)
from UnifiedZoneQuery import UnifiedZoneQuery, VectorizedZoneClassifier  # noqa: E402

DEFAULT_OUTPUT_NAME = "TIMELINE"
SUBJECT_FILE_KEYWORD = "Subject_Simulation_Info"

# ---------------------------------------------------------------------------
# 事件定义：name -> (mode, zone_id 集合)
#   enter 模式：首个命中帧
#   leave 模式：最后一次命中之后的首个有效帧（最终离开时刻）
# ---------------------------------------------------------------------------
SPAWN_ZONES = ("Spawn_01", "Spawn_02")
STAIR_ENTRANCE_ZONES = ("StairEntrance_ABCD", "StairEntrance_EFGH")
HALL_INSIDE_GATE_ZONES = ("HallInsideGate_Main",)
DECISION_ZONES = ("DecisionZone_ABCD", "DecisionZone_EFGH")

EVENTS = [
    ("leave_spawn", "leave", SPAWN_ZONES),
    ("leave_stair_entrance", "leave", STAIR_ENTRANCE_ZONES),
    ("enter_hall_inside_gate", "enter", HALL_INSIDE_GATE_ZONES),
    ("leave_hall_inside_gate", "leave", HALL_INSIDE_GATE_ZONES),
    ("leave_decision_zone", "leave", DECISION_ZONES),
    ("enter_exit", "enter", None),  # Exit_* 动态收集
]


def read_subject_frames(csv_path: Path):
    """逐行读取 Subject CSV。

    返回 (timestamps, pts, frames)：
        timestamps  (N,) str      —— A 列 TimeStamp 原始字符串
        pts         (N, 3) float  —— B/C/D 列 (PosX, PosY, PosZ)
        frames      (N,) int      —— 源 CSV 数据行号（1-based，表头除外）

    剔除规则：
        - 长度不足 / 数值解析失败的行（含表头）；
        - 全零帧（x == y == z == 0，记录启动前的无效原点帧）。
    frame 编号始终按源行号记录，剔除不影响反查。
    """
    stamps, pts, frames = [], [], []
    n_zero = 0
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
            if x == 0.0 and y == 0.0 and z == 0.0:
                n_zero += 1
                continue
            stamps.append(row[0].strip())
            pts.append((x, y, z))
            frames.append(row_idx)
    return stamps, np.asarray(pts, dtype=np.float64), np.asarray(frames, dtype=np.int64), n_zero


def extract_events(stamps, zone_ids_per_frame, exit_zone_set, warnings, tag):
    """按 EVENTS 定义提取 6 个事件。

    zone_ids_per_frame: (N,) 每帧归属的 zone_id 字符串（"Null" = 无归属）。
    返回 dict: event_name -> (timestamp, frame_idx_in_arrays, zone_id) 或 None。
    """
    n = len(stamps)
    frames_arr = None  # frame 由调用方在写行时换算
    results = {}

    for name, mode, zone_set in EVENTS:
        if zone_set is None:  # enter_exit：全部 Exit_* 分区
            members = exit_zone_set
        else:
            members = set(zone_set)
        mask = np.fromiter(
            (zid in members for zid in zone_ids_per_frame), dtype=bool, count=n
        )
        hit_idx = np.nonzero(mask)[0]
        if len(hit_idx) == 0:
            results[name] = None
            warnings.append("{}: 事件 {} 无命中帧".format(tag, name))
            continue
        if mode == "enter":
            i = int(hit_idx[0])
            results[name] = (stamps[i], i, zone_ids_per_frame[i])
        else:  # leave：最后一次命中之后的首个有效帧
            last = int(hit_idx[-1])
            if last + 1 >= n:
                results[name] = None
                warnings.append(
                    "{}: 事件 {} 记录结束时受试者仍在 {} 内，无离开帧".format(
                        tag, name, zone_ids_per_frame[last])
                )
                continue
            i = last + 1
            results[name] = (stamps[i], i, zone_ids_per_frame[last])
    return results



def timeline_order(events):
    """只检查真实事件，不排序、不修改时间戳；倒序优先于相等。"""
    stamps = [datetime.strptime(events[name][0], "%Y_%m_%d_%H_%M_%S_%f")
              for name, _, _ in EVENTS if events[name] is not None]
    if len(stamps) < 2:
        return "insufficient"
    if any(a > b for a, b in zip(stamps, stamps[1:])):
        return "reversed"
    if any(a == b for a, b in zip(stamps, stamps[1:])):
        return "equal"
    return "strict"


def main():
    parser = argparse.ArgumentParser(description="Extract per-subject zone enter/leave timeline")
    parser.add_argument("--root", type=str, default=RAW_DATA_DIR,
                        help="raw data root containing the 10 scenario folders "
                             "(default: <project>/RAW_DATA)")
    parser.add_argument("--output", type=str, default=None,
                        help="output dir (default: <project root>/TIMELINE)")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIO_FOLDERS),
                        help="comma-separated scenario names (default: all 10)")
    args = parser.parse_args()

    args.root = Path(args.root)
    args.output = Path(args.output) if args.output else Path(PROJECT_ROOT) / DEFAULT_OUTPUT_NAME
    trials = canonical_trials(args.root)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if len(scenarios) != len(SCENARIO_FOLDERS) or set(scenarios) != set(SCENARIO_FOLDERS):
        parser.error("Canonical timeline requires all 10 scenarios (270 trials)")
    args.output.mkdir(parents=True, exist_ok=True)

    query = UnifiedZoneQuery()
    classifier = VectorizedZoneClassifier(query)
    zone_id_of = [z.zone_id for z in classifier.zone_list]
    exit_zone_set = {z.zone_id for z in classifier.zone_list if z.zone_type == "Exit"}

    header = ["scenario", "subject", "group", "src_file", "n_valid_frames", "n_snapped_frames"]
    for name, _mode, _zs in EVENTS:
        header += ["{}_ts".format(name), "{}_frame".format(name), "{}_zone".format(name)]

    header += ["terminal_ts", "terminal_frame", "terminal_zone",
               "terminal_x", "terminal_y", "terminal_z",
               "timeline_order", "n_missing_events"]

    out_path = args.output / "subject_zone_timeline.csv"
    warnings = []
    n_rows = 0
    n_zero_total = 0
    n_snapped_total = 0

    staged_path = out_path.with_suffix(".csv.tmp")
    with open(staged_path, "w", newline="", encoding="utf-8-sig") as fh_out:
        writer = csv.writer(fh_out)
        writer.writerow(header)

        for scenario in scenarios:
            subjects = [(r['subject'], Path(r['subject_file'])) for r in trials if r['scenario'] == scenario]
            if not subjects:
                warnings.append("场景 {} 不存在或无受试者数据".format(scenario))
                continue
            for subject, csv_path in tqdm(subjects, desc="Timeline [{}]".format(scenario)):
                stamps, pts, frames, n_zero = read_subject_frames(csv_path)
                n_zero_total += n_zero
                tag = "{}/{}".format(scenario, subject)
                if len(pts) == 0:
                    raise ValueError("{}: canonical trial has no valid Subject frames".format(tag))

                terminal_xyz = pts[-1].copy()  # 保留吸附前原始终点坐标

                # 统一接口在分类副本上修正高度；这里只统计受影响帧数。
                n_snapped = query.count_floor_drift(pts)
                n_snapped_total += n_snapped

                zone_idx = classifier.assign(pts)
                zone_ids_per_frame = [
                    zone_id_of[i] if i >= 0 else "Null" for i in zone_idx
                ]

                events = extract_events(
                    stamps, zone_ids_per_frame, exit_zone_set, warnings, tag
                )

                group = next(r['group'] for r in trials if r['scenario'] == scenario and r['subject'] == subject and r['src_file'] == csv_path.name)
                row = [scenario, subject, group, csv_path.name, len(pts), n_snapped]
                for name, _mode, _zs in EVENTS:
                    ev = events.get(name)
                    if ev is None:
                        row += ["Null+" + zone_ids_per_frame[-1], "", ""]
                    else:
                        ts, i, zid = ev
                        row += [ts, int(frames[i]), zid]
                order = timeline_order(events)
                missing = sum(events[name] is None for name, _, _ in EVENTS)
                row += [stamps[-1], int(frames[-1]), zone_ids_per_frame[-1],
                        *terminal_xyz.tolist(), order, missing]
                if order in ("equal", "reversed"):
                    warnings.append("{}: timeline_order={}".format(tag, order))
                writer.writerow(row)
                n_rows += 1

    load_timeline(staged_path, args.root)
    staged_path.replace(out_path)
    print("\n完成：{} 行标准试次时间线 -> {}".format(n_rows, out_path))
    print("剔除全零无效帧合计：{}".format(n_zero_total))
    print("站厅地板漂移吸附帧合计：{}".format(n_snapped_total))
    if warnings:
        print("\n警告（{} 条）：".format(len(warnings)))
        for w in warnings:
            print("  -", w)


if __name__ == "__main__":
    main()
