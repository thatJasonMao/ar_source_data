#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Timeline P1-P5 speed evolution, normalized time only; see output METHODS.md.

python Analyze_TimelineSpeed.py [--bins 50] [--output DIR] [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from TimelineFinalAssets import plot_final_assets

from TimelineTrialInput import (load_timeline, cohort_metadata, require_fresh_frame_cohort)
from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS, GUIDANCE_SMOKE_COLORS

EVENTS = ['leave_spawn', 'leave_stair_entrance', 'enter_hall_inside_gate',
          'leave_hall_inside_gate', 'leave_decision_zone', 'enter_exit']
PERIODS = [
    ('P1', 'Spawn departure to stair-entrance departure'),
    ('P2', 'Stair-entrance departure to inside-gate hall entry'),
    ('P3', 'Inside-gate hall entry to departure'),
    ('P4', 'Inside-gate hall departure to decision-zone departure'),
    ('P5', 'Decision-zone departure to exit entry'),
]
MODES = ['StaticSign', 'ArrowRibbon', 'MiniMap', 'TextBar', 'Full']
# Shared with graph_0911: low smoke light, high smoke dark.
SCENARIO_PALETTE = GUIDANCE_SMOKE_COLORS
SCENARIO_COLORS = {f'{mode}-{smoke}': color for mode, shades in SCENARIO_PALETTE.items() for smoke, color in shades.items()}
COLORS = {mode: shades['HighSmoke'] for mode, shades in SCENARIO_PALETTE.items()}


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                             else v for k, v in row.items()})


def mean_curves(curves):
    a = np.asarray(curves, dtype=float)
    n = np.isfinite(a).sum(axis=0)
    return np.divide(np.nansum(a, axis=0), n, out=np.full(a.shape[1], np.nan), where=n > 0)


def parse_stamp(s):
    return datetime.strptime(s.strip(), '%Y_%m_%d_%H_%M_%S_%f')


def bin_speed(times, speeds, start, stop, bins):
    """Sample means in half-open bins; final bin includes period endpoint."""
    u = (times - start) / (stop - start)
    keep = np.isfinite(speeds) & (u >= 0) & (u <= 1)
    index = np.minimum((u[keep] * bins).astype(int), bins - 1)
    n = np.bincount(index, minlength=bins)
    total = np.bincount(index, weights=speeds[keep], minlength=bins)
    return np.divide(total, n, out=np.full(bins, np.nan), where=n > 0), n


def load_trial(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if rows[0][4:7] != ['VeloX', 'VeloY', 'VeloZ']:
        raise ValueError(f'Unexpected E:G columns: {path}')
    return rows


def extract_period(rows, trial, k, bins):
    start_event, end_event = EVENTS[k:k + 2]
    a, b = trial[start_event + '_ts'], trial[end_event + '_ts']
    if not a or not b or a.startswith('Null') or b.startswith('Null'):
        return None, {}, 'missing_boundary'
    ia, ib = int(trial[start_event + '_frame']), int(trial[end_event + '_frame'])
    if not 0 < ia <= ib < len(rows):
        raise ValueError(f'Invalid boundary frames: {trial["src_file"]}, {ia}, {ib}')
    if rows[ia][0].strip() != a or rows[ib][0].strip() != b:
        raise ValueError(f'Timeline/source timestamp mismatch: {trial["src_file"]}')
    ta, tb = parse_stamp(a), parse_stamp(b)
    duration = (tb - ta).total_seconds()
    meta = dict(start_ts=a, end_ts=b, start_frame=ia, end_frame=ib, duration_s=duration)
    if duration <= 0:
        return None, meta, 'nonpositive_duration'
    times, speeds = [], []
    invalid = 0
    for row in rows[ia:ib + 1]:
        try:
            t = (parse_stamp(row[0]) - ta).total_seconds()
            vx, vz = float(row[4]), float(row[6])
            pos = np.array([float(v) for v in row[1:4]])
            speed = np.hypot(vx, vz)
            if not np.isfinite(pos).all() or np.all(pos == 0) or not np.isfinite(speed):
                invalid += 1
                continue
        except (ValueError, IndexError):
            invalid += 1
            continue
        times.append(t)
        speeds.append(speed)
    times, speeds = np.asarray(times), np.asarray(speeds)
    if np.any(np.diff(times) < 0):
        return None, meta, 'nonmonotonic_source_time'
    curve, counts = bin_speed(times, speeds, 0, duration, bins)
    meta.update(n_source_frames=ib - ia + 1, n_valid_frames=int(counts.sum()),
                n_invalid_frames=invalid, n_nonempty_bins=int((counts > 0).sum()))
    if not counts.any():
        return None, meta, 'no_valid_speed'
    return (curve, counts), meta, 'included'


def selected_trial_curves(trials, curves, pid, level, name):
    """Return each included trial separately, with its actual guidance and smoke-condition colour."""
    result = []
    for i, trial in enumerate(trials):
        if (i, pid) not in curves:
            continue
        scenario = trial['scenario']
        mode = scenario.rsplit('-', 1)[0]
        if level == 'scenario' and scenario != name:
            continue
        if level == 'modality' and mode != name:
            continue
        result.append((i, scenario, curves[i, pid]))
    return result


def draw_trial_backgrounds(ax, centers, trial_lines, *, linewidth=.7):
    for _, scenario, curve in trial_lines:
        ax.plot(centers, curve, color=SCENARIO_COLORS[scenario], alpha=.30,
                lw=linewidth, ls='-', zorder=1)


def trial_legend_handles(trial_lines):
    present = {scenario for _, scenario, _ in trial_lines}
    order = [f'{mode}-{smoke}' for mode in MODES for smoke in ('LowSmoke', 'HighSmoke')]
    return [Line2D([0], [0], color=SCENARIO_COLORS[scenario], alpha=.30, lw=1,
                   ls='-', label=scenario) for scenario in order if scenario in present]


def grouping(trials, curves, pid):
    """Participant curves for equal-weight means and exports; sessions -> scenario -> mode -> person."""
    by_scenario = defaultdict(list)
    for i, r in enumerate(trials):
        if (i, pid) in curves:
            by_scenario[(r['scenario'], r['subject'])].append(curves[i, pid])
    if any(len(v) != 1 for v in by_scenario.values()):
        raise ValueError('Duplicate participant-condition in canonical timeline')
    scenario_person = {key: v[0] for key, v in by_scenario.items()}
    by_mode = defaultdict(list)
    for (scenario, subject), curve in scenario_person.items():
        by_mode[(scenario.rsplit('-', 1)[0], subject)].append(curve)
    mode_person = {key: mean_curves(v) for key, v in by_mode.items()}
    by_person = defaultdict(list)
    for (_, subject), curve in mode_person.items():
        by_person[subject].append(curve)
    all_person = {s: mean_curves(v) for s, v in by_person.items()}
    result = []
    for scenario in SCENARIO_FOLDERS:
        result.append(('scenario', scenario, {s: v for (sc, s), v in scenario_person.items() if sc == scenario}))
    for mode in MODES:
        result.append(('modality', mode, {s: v for (m, s), v in mode_person.items() if m == mode}))
    result.append(('all', 'AllParticipants', all_person))
    return result



def draw_selection_means(ax, centers, groups, level, name, linewidth=3):
    """AllParticipants shows five smoke-pooled, participant-equal guidance means."""
    handles = []
    for group_level, group_name, people in groups:
        if not people:
            continue
        if level == 'all':
            if group_level != 'modality':
                continue
        elif (group_level, group_name) != (level, name):
            continue
        color = COLORS[group_name] if group_level == 'modality' else SCENARIO_COLORS[group_name]
        label = group_name if level == 'all' else 'Participant mean'
        line, = ax.plot(centers, mean_curves(list(people.values())), color=color,
                        lw=linewidth, zorder=3, label=label)
        handles.append(line)
    return handles


def guidance_legend_handles():
    return [Line2D([0], [0], color=COLORS[mode], lw=3, label=mode)
            for mode in MODES]


def load_cached_curves(output, metric, value_column):
    """Read existing bins without modifying data or re-extracting raw frames."""
    with (output / f'trial_{metric}_bins.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    bins = max(int(row['bin']) for row in rows)
    centers = (np.arange(bins) + .5) / bins
    trials_by_id, curves, seen = {}, {}, set()
    for row in rows:
        trial_id = int(row['trial_id']) - 1
        metadata = {key: row[key] for key in ('scenario', 'subject', 'group', 'src_file')}
        if trials_by_id.setdefault(trial_id, metadata) != metadata:
            raise ValueError('Inconsistent cached trial identity')
        pid, j = row['period'], int(row['bin']) - 1
        key = (trial_id, pid)
        if (trial_id, pid, j) in seen or pid not in dict(PERIODS):
            raise ValueError('Invalid or duplicate cached bin')
        seen.add((trial_id, pid, j))
        if not np.isclose(float(row['time_normalized']), centers[j]):
            raise ValueError('Cached time grid mismatch')
        curves.setdefault(key, np.full(bins, np.nan))[j] = float(row[value_column]) if row[value_column] else np.nan
    if len(seen) != len(curves) * bins:
        raise ValueError('Incomplete cached curves')
    ids = sorted(trials_by_id)
    remap = {old: new for new, old in enumerate(ids)}
    trials = [trials_by_id[i] for i in ids]
    return trials, {(remap[i], pid): v for (i, pid), v in curves.items()}, centers


def plot_merges(trials,curves,output,centers,dpi, *, all_only=False):
    """Five panels + legend panel for the all-participant speed overview."""
    groups={pid:grouping(trials,curves,pid) for pid,_ in PERIODS}
    for group_index in tqdm(range(15,16),desc='Merge P1-P5'):
        level,name,_=groups['P1'][group_index]
        maximum=max((np.nanmax(v) for pid,_ in PERIODS for _,_,v in selected_trial_curves(trials,curves,pid,level,name)),default=1.)
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        for ax,(pid,label) in zip(axes.flat,PERIODS):
            people=groups[pid][group_index][2]
            trial_lines=selected_trial_curves(trials,curves,pid,level,name)
            draw_trial_backgrounds(ax,centers,trial_lines)
            if people:
                draw_selection_means(ax, centers, groups[pid], level, name, linewidth=2.8)
            ax.set_title(f'{pid} | {name}\n{label}',fontsize=9)
            ax.set(xlim=(0,1),xlabel='Normalized time',ylabel='Mean planar speed (m/s)')
            ax.set_ylim(0,max(1.,maximum*1.08))
            ax.tick_params(labelbottom=True)
            ax.spines[['top','right']].set_visible(False)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.04,.95,'P1-P5 planar speed evolution\n\nThin solid: individual trials (alpha=0.30)\nBold: equal-weight participant mean\n\n'+str(len(centers))+' time bins per period\nOnly time is normalized\nOriginal subject timestamps\nNo speed smoothing or gap filling',
                          va='top',fontsize=9 if level=='all' else 10,linespacing=1.1 if level=='all' else 1.3,transform=axes.flat[-1].transAxes)
        if level=='all':
            note = axes.flat[-1].texts[0]
            note.set_text(note.get_text().replace('Bold: equal-weight participant mean',
                'Bold: five guidance means\nHigh + low smoke pooled within participant'))
        all_trial_lines=[t for pid,_ in PERIODS for t in selected_trial_curves(trials,curves,pid,level,name)]
        axes.flat[-1].legend(handles=guidance_legend_handles() if level=='all' else trial_legend_handles(all_trial_lines),loc='lower left',frameon=False,fontsize=8 if level=='all' else 9,ncol=2)
        fig.tight_layout()
        for ext in (['png'] if all_only else ['png','svg']):
            if level=='all':
                fig.savefig(output/f'MERGE.{ext}',dpi=dpi,bbox_inches='tight')
        plt.close(fig)




def plot_results(trials, curves, output, centers, dpi, *, images_only=False, all_only=False):
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                         'font.size':12,'axes.labelsize':14,'axes.titlesize':14,
                         'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
                         'axes.grid':True,'grid.alpha':.18})
    subject_rows, mean_rows, manifest = [], [], []
    for pid, label in tqdm(PERIODS, desc='Plot periods (16 each)'):
        directory = output/pid
        directory.mkdir(exist_ok=True)
        groups = grouping(trials, curves, pid)
        maximum = max((np.nanmax(v) for _, _, v in selected_trial_curves(trials, curves, pid, 'all', 'AllParticipants')), default=1.)
        for level, name, people in groups:
            if all_only and level != 'all':
                continue
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            trial_lines = selected_trial_curves(trials, curves, pid, level, name)
            draw_trial_backgrounds(ax, centers, trial_lines)
            matrix = np.asarray(list(people.values()))
            if len(people):
                mean = mean_curves(matrix)
                n = np.isfinite(matrix).sum(axis=0)
                for subject, curve in people.items():
                    for j, u in enumerate(centers):
                        subject_rows.append(dict(period=pid, level=level, selection=name,
                                                 subject=subject, bin=j+1, time_normalized=u,
                                                 speed_m_s=curve[j]))
                mean_handles = draw_selection_means(ax, centers, groups, level, name)
                ax.legend(handles=mean_handles if level=='all' else trial_legend_handles(trial_lines) + mean_handles,
                          loc='upper right',frameon=False,ncol=2,fontsize=8)
                for j, u in enumerate(centers):
                    mean_rows.append(dict(period=pid,level=level,selection=name,bin=j+1,
                                          time_normalized=u,speed_m_s=mean[j],n_subjects=int(n[j])))
            else:
                ax.text(.5,.5,'No valid periods',ha='center',transform=ax.transAxes)
            ax.set(xlim=(0,1), ylim=(0,max(1.,maximum*1.12)),
                   xlabel='Normalized time within period',ylabel='Mean planar speed (m/s)')
            ax.set_title(f'{pid} | {name}\n{label}',pad=14)
            ax.text(.01,.98,f'n = {len(people)} participants | {len(trial_lines)} trials',va='top',transform=ax.transAxes,fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout()
            stem = f'{level}_{name}'
            for ext in (['png'] if all_only else ['png','svg']):
                fig.savefig(directory/(stem+'.'+ext),dpi=dpi,bbox_inches='tight')
            plt.close(fig)
            manifest.append(dict(period=pid,level=level,selection=name,n_subjects=len(people),
                                 png=f'{pid}/{stem}.png',svg=f'{pid}/{stem}.svg'))
    if not images_only:
        write_csv(output/'subject_speed_curves.csv',
                  ['period','level','selection','subject','bin','time_normalized','speed_m_s'],subject_rows)
        write_csv(output/'mean_speed_curves.csv',
                  ['period','level','selection','bin','time_normalized','speed_m_s','n_subjects'],mean_rows)
        write_csv(output/'figure_manifest.csv',
                  ['period','level','selection','n_subjects','png','svg'],manifest)
    plot_merges(trials,curves,output,centers,dpi,all_only=all_only)
    plot_final_assets(trials, curves, output, centers, dpi, 'speed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(RAW_DATA_DIR))
    parser.add_argument('--timeline', type=Path, default=Path(PROJECT_ROOT) / 'TIMELINE/subject_zone_timeline.csv')
    parser.add_argument('--output', type=Path, default=Path(PROJECT_ROOT) / 'TIMELINE/Speed')
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--dpi', type=int, default=300)
    redraw = parser.add_mutually_exclusive_group()
    redraw.add_argument('--final-assets-only', action='store_true',
                        help='Rebuild only the five packaged final PNGs from existing bins')
    redraw.add_argument('--all-participants-only', action='store_true',
                        help='Redraw only P1-P5 and merged AllParticipants PNGs (including overview aliases) from existing bins; preserve other outputs')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.final_assets_only:
        trials, curves, centers = load_cached_curves(args.output, 'speed', 'speed_m_s')
        plot_final_assets(trials, curves, args.output, centers, args.dpi, 'speed')
        return
    if args.all_participants_only:
        trials, curves, centers = load_cached_curves(args.output, 'speed', 'speed_m_s')
        plot_results(trials, curves, args.output, centers, args.dpi,
                     images_only=True, all_only=True)
        return
    if args.bins < 2:
        parser.error('--bins must be at least 2')
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        parser.error('Output is nonempty; use --force to overwrite generated files')
    trials = load_timeline(args.timeline, args.root)
    require_fresh_frame_cohort(args.output, trials)
    args.output.mkdir(parents=True, exist_ok=True)
    curves, audits, trial_rows = {}, [], []
    centers = (np.arange(args.bins) + .5) / args.bins
    for i, r in enumerate(tqdm(trials, desc='Read speed periods')):
        path = Path(r['subject_file'])
        group = r['group']
        if r['scenario'] not in SCENARIO_FOLDERS:
            raise ValueError(f'Unknown scenario: {r["scenario"]}')
        if group and r['group'] != group:
            raise ValueError(f'Group mismatch: {path}')
        rows = load_trial(path) if group else None
        for k, (pid, _) in enumerate(PERIODS):
            base = dict(trial_id=i + 1, scenario=r['scenario'], subject=r['subject'],
                        group=group or '', src_file=r['src_file'], period=pid)
            if not group:
                audits.append(dict(base, status='excluded_test_or_unknown_group'))
                continue
            value, meta, status = extract_period(rows, r, k, args.bins)
            audits.append(dict(base, **meta, status=status))
            if value is None:
                continue
            curve, counts = value
            curves[i, pid] = curve
            for j, u in enumerate(centers):
                trial_rows.append(dict(base, bin=j + 1, time_normalized=u,
                                       speed_m_s=curve[j], n_samples=int(counts[j])))
    audit_fields = ['trial_id','scenario','subject','group','src_file','period','status',
                    'start_ts','end_ts','start_frame','end_frame','duration_s',
                    'n_source_frames','n_valid_frames','n_invalid_frames','n_nonempty_bins']
    write_csv(args.output/'period_audit.csv', audit_fields, audits)
    write_csv(args.output/'trial_speed_bins.csv',
              ['trial_id','scenario','subject','group','src_file','period','bin',
               'time_normalized','speed_m_s','n_samples'], trial_rows)
    manifest = plot_results(trials, curves, args.output, centers, args.dpi)
    summary = {**cohort_metadata(trials), 'timeline':str(args.timeline),'raw_data':str(args.root),'bins':args.bins,
               'timeline_rows':len(trials),'figures':len(manifest),
               'period_status_counts':dict(Counter(a['status'] for a in audits)),
               'included_by_period':dict(Counter(a['period'] for a in audits if a['status']=='included'))}
    (args.output/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    methods = '''# P1–P5 平面速度演化

## 周期和时间
P1：离开 Spawn → 离开 StairEntrance。
P2：离开 StairEntrance → 进入 HallInsideGate。
P3：进入 HallInsideGate → 离开 HallInsideGate。
P4：离开 HallInsideGate → 离开 DecisionZone。
P5：离开 DecisionZone → 进入 Exit。

这是事件间的连续时间段，包含期间折返；不是仅保留某个几何区域内的点。
使用 timeline 的原始边界时间戳及源行号，逐个与 Subject CSV 核对。
u = (t - t_start) / (t_end - t_start)，只归一化时间。
每 P 使用 BINS 个等宽时间窗，横坐标为窗中心。除最后一窗外均左闭右开。
P 两端帧均纳入，因此公共边界帧可分别出现在相邻两个 P 中。

## 速度
v = sqrt(VeloX**2 + VeloZ**2)，即 CSV E/G 列，单位 m/s。
每窗速度是该窗有效原始帧速度的算术均值（采样均值，不是时间积分加权均值）。
不做速度归一化、标准化、平滑或插值；保留零速度，不裁剪极值。
无效坐标、全零位置帧、无效速度/时间排除；空窗保持缺失，不补零。
时间分箱提供平均速度曲线，原始瞬时峰值可能被窗内平均降低，但速度量纲不变。

## 人员与平均权重
仅纳入 common.load_trials() 的45人/270试次；时间线必须完整、唯一匹配标准试次，队列外记录直接报错。
缺失任一边界、零时长或倒序时间的 P 不使用，其他有效 P 保留；不以终点替代缺失事件。
同一人同一场景仅保留 common 指定的唯一会话；重复身份直接报错，trial_speed_bins.csv 保留标准试次。
10个场景图：每人该场景的曲线为一条受试者汇总曲线，粗线对人等权平均。
5个模态图：先对每人的高/低烟雾场景曲线等权平均，再对人等权平均。
全量图：先对每人在各模态的曲线等权平均，再对人等权平均。
合并时使用每个时间窗的可用值；各人可用条件可能不同。缺失在审计表中保留。
每张图的 n 是至少有一个有效窗的人数，逐窗有效人数在 mean_speed_curves.csv。
每个 P 的16张图共用纵轴范围；共80张图，每张保存 PNG（300dpi默认）和 SVG。

## 产物
- period_audit.csv：每条timeline记录×每P的纳入状态、原始边界、时长与有效帧数。
- trial_speed_bins.csv：每会话每P每窗的速度及原始采样数。
- subject_speed_curves.csv：各图每名受试者的受试者汇总曲线数据。
- mean_speed_curves.csv：各图粗线及每窗有效人数。
- figure_manifest.csv：80张图索引。
- run_summary.json：参数与纳入计数。

重新运行根目录 Analyze_TimelineSpeed.py；默认输出 TIMELINE/Speed，覆盖时用 --force。
'''.replace('BINS',str(args.bins))
    methods += '\n背景为每个有效 trial 的独立曲线，按引导方式×烟雾条件的10色情景配色着色（低烟浅色、高烟深色），alpha=0.30 的细实线；粗线仍为受试者等权均值。subject_*_curves.csv 仅保存汇总数据，背景线来自 trial_*_bins.csv。\n'
    (args.output/'METHODS.md').write_text(methods,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
