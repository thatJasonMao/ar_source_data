#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame-level gaze depth followed by timeline P1-P5 plotting.

python Analyze_TimelineGazeDepth.py [--bins 50] [--output DIR] [--force]
Depth is computed before plotting bins. Reference alignment/validity is reused.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from TimelineFinalAssets import plot_final_assets
from TimelineTrialInput import (load_timeline, cohort_metadata, require_fresh_frame_cohort)
from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS
from Analyze_TimelineSpeed import draw_selection_means, guidance_legend_handles, load_cached_curves, selected_trial_curves, draw_trial_backgrounds, trial_legend_handles, EVENTS, PERIODS, COLORS, SCENARIO_COLORS, grouping, mean_curves, write_csv, bin_speed, parse_stamp

# Shared display range for every single-period and merged timeline plot.
GAZE_DEPTH_YMAX_M = 40.0

REFERENCE = Path(PROJECT_ROOT) / 'Results_Spatial_Metric_Surfaces/_timestamp_input/extract_timestamp_metrics.py'
spec = importlib.util.spec_from_file_location('gaze_depth_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def source_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))


def load_trial(subject_path):
    """Read actual eye frames; no spatial assignment, exposure, or metric bins."""
    eye_path = subject_path.with_name(subject_path.name.replace('_Subject_', '_EyeGaze_'))
    raw_pos = source_rows(subject_path)
    if not eye_path.exists():
        return dict(raw_pos=raw_pos, missing_eye=True, eye_path=eye_path)
    pos, pq = reference.read_raw(subject_path, ['TimeStamp', 'PosX', 'PosY', 'PosZ'])
    eye, eq = reference.read_raw(eye_path, ['TimeStamp', 'GazeDirX','GazeDirY','GazeDirZ',
                                                   'Hit1_X','Hit1_Y','Hit1_Z'])
    ens = eye.ParsedTime.astype('datetime64[ns]').astype('int64').to_numpy()
    pns = pos.ParsedTime.astype('datetime64[ns]').astype('int64').to_numpy()
    aligned, sync = reference.align_positions(ens, pns, pos[['PosX','PosY','PosZ']].to_numpy(float))
    _, gaze_valid = reference.unit_vectors(eye[['GazeDirX','GazeDirY','GazeDirZ']].to_numpy(float))
    hit = eye[['Hit1_X','Hit1_Y','Hit1_Z']].to_numpy(float)
    position_valid = np.isfinite(aligned).all(axis=1)
    hit_valid = np.isfinite(hit).all(axis=1) & (hit != 0).any(axis=1)
    valid = gaze_valid & position_valid & hit_valid
    eye_position = aligned.copy()
    eye_position[:,1] += reference.CONFIG['eye_height_m']
    depth = np.linalg.norm(hit - eye_position, axis=1)
    depth[~valid] = np.nan
    # read_raw excludes ambiguous timestamps; source rows stay recoverable by timestamp.
    raw_eye = source_rows(eye_path)
    eye_row = {r[0]: i for i,r in enumerate(raw_eye[1:],1) if r}
    frame = pd.DataFrame({'eye_frame':[eye_row[s] for s in eye.TimeStamp],
                          'timestamp':eye.TimeStamp.to_numpy(),
                          'gaze_depth_m':depth,'depth_valid':valid.astype(int),
                          'gaze_valid':gaze_valid.astype(int),'position_valid':position_valid.astype(int),
                          'hit_valid':hit_valid.astype(int),
                          'position_sync':np.array(['missing','exact','interpolated'])[sync]})
    for j,axis in enumerate('xyz'):
        frame['subject_'+axis] = aligned[:,j]
        frame['hit1_'+axis] = hit[:,j]
    return dict(raw_pos=raw_pos,missing_eye=False,eye_path=eye_path,
                ns=ens,depth=depth,frame=frame,pos_audit=pq,eye_audit=eq)


def bounds(data, trial, k):
    a,b = [trial[n+'_ts'] for n in EVENTS[k:k+2]]
    if not a or not b or a.startswith('Null') or b.startswith('Null'):
        return None, {}, 'missing_boundary'
    ia,ib = [int(trial[n+'_frame']) for n in EVENTS[k:k+2]]
    raw=data['raw_pos']
    if not 0 < ia <= ib < len(raw) or raw[ia][0].strip()!=a or raw[ib][0].strip()!=b:
        raise ValueError('Timeline/source mismatch: '+trial['src_file'])
    ta,tb=pd.Timestamp(parse_stamp(a)).value,pd.Timestamp(parse_stamp(b)).value
    meta=dict(start_ts=a,end_ts=b,start_frame=ia,end_frame=ib,duration_s=(tb-ta)/1e9)
    if tb<=ta:
        return None,meta,'nonpositive_duration'
    return (ta,tb),meta,'valid_bounds'


def export_frames(data, trial, output):
    """Full cleaned eye stream, invalid depth retained as missing, no downsampling."""
    if data['missing_eye']:
        return dict(scenario=trial['scenario'],subject=trial['subject'],src_file=trial['src_file'],
                    status='missing_eye_file')
    frame=data['frame']
    for k,(pid,_) in enumerate(PERIODS):
        boundary,_,_=bounds(data,trial,k)
        u=np.full(len(frame),np.nan)
        if boundary is not None:
            a,b=boundary
            mask=(data['ns']>=a)&(data['ns']<=b)
            u[mask]=(data['ns'][mask]-a)/(b-a)
        frame[pid+'_normalized_time']=u
    target=output/'frames'/trial['scenario']/trial['subject']/(data['eye_path'].stem+'_depth.csv.gz')
    target.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(target,index=False,encoding='utf-8-sig',compression='gzip')
    return dict(scenario=trial['scenario'],subject=trial['subject'],src_file=trial['src_file'],
                eye_file=data['eye_path'].name,status='extracted',output=str(target.relative_to(output)),
                n_eye_frames=len(frame),n_depth_valid=int(frame.depth_valid.sum()),
                n_exact=int((frame.position_sync=='exact').sum()),
                n_interpolated=int((frame.position_sync=='interpolated').sum()),
                n_unmatched=int((frame.position_sync=='missing').sum()),
                subject_cleaning=data['pos_audit'],eye_cleaning=data['eye_audit'])


def extract_period(data,trial,k,bins):
    boundary,meta,status=bounds(data,trial,k)
    if boundary is None:
        return None,meta,status
    if data['missing_eye']:
        return None,meta,'missing_eye_file'
    a,b=boundary
    selected=(data['ns']>=a)&(data['ns']<=b)
    times=(data['ns'][selected]-a)/1e9
    depth=data['depth'][selected]
    curve,counts=bin_speed(times,depth,0.,(b-a)/1e9,bins)
    meta.update(n_source_frames=int(selected.sum()),n_valid_frames=int(counts.sum()),
                n_invalid_frames=int((~np.isfinite(depth)).sum()),n_nonempty_bins=int((counts>0).sum()))
    if not counts.any():
        return None,meta,'no_valid_gaze_depth'
    return (curve,counts),meta,'included'


def plot_merges(trials,curves,output,centers,dpi, *, all_only=False):
    """Five panels + legend panel, matching Speed/MERGE.png; one per selection."""
    groups={pid:grouping(trials,curves,pid) for pid,_ in PERIODS}
    merge_dir=output/'MERGE'
    merge_dir.mkdir(exist_ok=True)
    for group_index in tqdm(range(15,16) if all_only else range(16),desc='Merge P1-P5'):
        level,name,_=groups['P1'][group_index]
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        for ax,(pid,label) in zip(axes.flat,PERIODS):
            people=groups[pid][group_index][2]
            trial_lines=selected_trial_curves(trials,curves,pid,level,name)
            draw_trial_backgrounds(ax,centers,trial_lines)
            if people:
                draw_selection_means(ax, centers, groups[pid], level, name, linewidth=2.8)
            ax.set_title(f'{pid} | {name}\n{label}',fontsize=9)
            ax.set(xlim=(0,1),xlabel='Normalized time',ylabel='Mean gaze depth (m)')
            ax.set_ylim(0,GAZE_DEPTH_YMAX_M)
            ax.tick_params(labelbottom=True)
            ax.spines[['top','right']].set_visible(False)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.04,.95,'P1-P5 gaze depth evolution\n\nThin solid: individual trials (alpha=0.30)\nBold: equal-weight participant mean\n\n'+str(len(centers))+' time bins per period\nOnly time is normalized\nDepth uses original eye timestamps\nNo depth smoothing or gap filling',
                          va='top',fontsize=9 if level=='all' else 10,linespacing=1.1 if level=='all' else 1.3,transform=axes.flat[-1].transAxes)
        if level=='all':
            note = axes.flat[-1].texts[0]
            note.set_text(note.get_text().replace('Bold: equal-weight participant mean',
                'Bold: five guidance means\nHigh + low smoke pooled within participant'))
        all_trial_lines=[t for pid,_ in PERIODS for t in selected_trial_curves(trials,curves,pid,level,name)]
        axes.flat[-1].legend(handles=guidance_legend_handles() if level=='all' else trial_legend_handles(all_trial_lines),loc='lower left',frameon=False,fontsize=8 if level=='all' else 9,ncol=2)
        fig.tight_layout()
        for ext in (['png'] if all_only else ['png','svg']):
            fig.savefig(merge_dir/f'{level}_{name}.{ext}',dpi=dpi,bbox_inches='tight')
            if level=='all':
                fig.savefig(output/f'MERGE.{ext}',dpi=dpi,bbox_inches='tight')
        plt.close(fig)


def plot_results(trials, curves, output, centers, dpi, *, images_only=False, all_only=False):
    """Render all timeline figures; optionally preserve existing numerical tables."""
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                         'font.size':12,'axes.labelsize':14,'axes.titlesize':14,
                         'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
                         'axes.grid':True,'grid.alpha':.18})
    subject_rows, mean_rows, manifest = [], [], []
    for pid, label in tqdm(PERIODS, desc='Plot periods (16 each)'):
        directory = output/pid
        directory.mkdir(exist_ok=True)
        groups = grouping(trials, curves, pid)
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
                                                 gaze_depth_m=curve[j]))
                mean_handles = draw_selection_means(ax, centers, groups, level, name)
                ax.legend(handles=mean_handles if level=='all' else trial_legend_handles(trial_lines) + mean_handles,
                          loc='upper right',frameon=False,ncol=2,fontsize=8)
                for j, u in enumerate(centers):
                    mean_rows.append(dict(period=pid,level=level,selection=name,bin=j+1,
                                          time_normalized=u,gaze_depth_m=mean[j],n_subjects=int(n[j])))
            else:
                ax.text(.5,.5,'No valid periods',ha='center',transform=ax.transAxes)
            ax.set(xlim=(0,1), ylim=(0,GAZE_DEPTH_YMAX_M),
                   xlabel='Normalized time within period',ylabel='Mean gaze depth (m)')
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
        write_csv(output/'subject_gaze_depth_curves.csv',
                  ['period','level','selection','subject','bin','time_normalized','gaze_depth_m'],subject_rows)
        write_csv(output/'mean_gaze_depth_curves.csv',
                  ['period','level','selection','bin','time_normalized','gaze_depth_m','n_subjects'],mean_rows)
        write_csv(output/'figure_manifest.csv',
                  ['period','level','selection','n_subjects','png','svg'],manifest)
    plot_merges(trials,curves,output,centers,dpi,all_only=all_only)
    plot_final_assets(trials, curves, output, centers, dpi, 'gaze_depth')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(RAW_DATA_DIR))
    parser.add_argument('--timeline', type=Path, default=Path(PROJECT_ROOT) / 'TIMELINE/subject_zone_timeline.csv')
    parser.add_argument('--output', type=Path, default=Path(PROJECT_ROOT) / 'TIMELINE/GazeDepth')
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
        trials, curves, centers = load_cached_curves(args.output, 'gaze_depth', 'gaze_depth_m')
        plot_final_assets(trials, curves, args.output, centers, args.dpi, 'gaze_depth')
        return
    if args.all_participants_only:
        trials, curves, centers = load_cached_curves(args.output, 'gaze_depth', 'gaze_depth_m')
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
    frame_audits = []
    centers = (np.arange(args.bins) + .5) / args.bins
    for i, r in enumerate(tqdm(trials, desc='Read gaze_depth periods')):
        path = Path(r['subject_file'])
        group = r['group']
        if r['scenario'] not in SCENARIO_FOLDERS:
            raise ValueError(f'Unknown scenario: {r["scenario"]}')
        if group and r['group'] != group:
            raise ValueError(f'Group mismatch: {path}')
        rows = load_trial(path) if group else None
        if group:
            frame_audits.append(export_frames(rows, r, args.output))
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
                                       gaze_depth_m=curve[j], n_samples=int(counts[j])))
    audit_fields = ['trial_id','scenario','subject','group','src_file','period','status',
                    'start_ts','end_ts','start_frame','end_frame','duration_s',
                    'n_source_frames','n_valid_frames','n_invalid_frames','n_nonempty_bins']
    write_csv(args.output/'period_audit.csv', audit_fields, audits)
    write_csv(args.output/'trial_gaze_depth_bins.csv',
              ['trial_id','scenario','subject','group','src_file','period','bin',
               'time_normalized','gaze_depth_m','n_samples'], trial_rows)
    manifest = plot_results(trials, curves, args.output, centers, args.dpi)
    (args.output/'frame_extraction_audit.json').write_text(json.dumps(frame_audits,indent=2),encoding='utf-8')
    summary = {**cohort_metadata(trials), 'timeline':str(args.timeline),'raw_data':str(args.root),'bins':args.bins,
               'timeline_rows':len(trials),'figures':len(manifest),'merge_figures':16,
               'reference':str(REFERENCE),'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
               'eye_height_m':reference.CONFIG['eye_height_m'],
               'n_eye_frames':sum(r.get('n_eye_frames',0) for r in frame_audits),
               'n_valid_depth_frames':sum(r.get('n_depth_valid',0) for r in frame_audits),
               'period_status_counts':dict(Counter(a['status'] for a in audits)),
               'included_by_period':dict(Counter(a['period'] for a in audits if a['status']=='included'))}
    (args.output/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    methods = '# Gaze depth：逐帧提取与 P1–P5 演化\n\n## 逐帧计算（与指定参考脚本对齐）\n参考：Results_Spatial_Metric_Surfaces/_timestamp_input/extract_timestamp_metrics.py。\n复用 read_raw、align_positions、unit_vectors；不调用其 process_trial 或 main。\n按实际 EyeGaze 时间戳获取 Subject XYZ：优先精确匹配，无精确匹配时仅在两侧位置时间间隔<=0.1秒时线性插值，不外推。\ngaze_depth = ||Hit1 - (Subject位置 + [0, eye_height_m, 0])||，eye_height_m 使用参考 CONFIG，当前为0，单位米。\n有效条件：GazeDir有限且长度严格在(0.5,1.5)内；Subject位置有限；Hit1有限且不全零。\n不额外加入全零Subject位置过滤，以遵守参考的depth_valid定义。所有原始坐标用于距离，无楼层吸附。\n时间戳解析失败或重复的歧义记录按参考read_raw排除；清洗数量写入 frame_extraction_audit.json。\n不采用参考的空间网格、motion_valid区间权重、区间深度积分、任何其他指标平滑。\n\n## 连续数据\nframes/<场景>/<受试者>/*_depth.csv.gz 保存整条trial清洗后的逐眼动帧数据，未分箱、未降采样。\n包括原始眼动timestamp和eye_frame（表头除外1-based）、深度、有效标志、位置对齐方法、Subject与Hit1坐标。\n无效深度保留空值；位置插值只用于匹配参考方案，不插补深度曲线、不新造眼动时间戳。\nP1_normalized_time至P5_normalized_time标记帧所属P和归一化时间；不属于该P或边界无效则留空。\n这是gzip压缩CSV，可用pandas.read_csv直接读取，不需要先解压。\n\n## 时间与绘图\nP1:离开Spawn→离开StairEntrance；P2:离开StairEntrance→进入HallInsideGate；\nP3:进入HallInsideGate→离开HallInsideGate；P4:离开HallInsideGate→离开DecisionZone；P5:离开DecisionZone→进入Exit。\n用timeline原始边界时间戳和源行号验证Subject文件；将期间所有EyeGaze帧按真实时间纳入，不按行号硬拼。\nu=(t-t_start)/(t_end-t_start)，每P绘图时分为BINS个等宽窗，窗内取有效逐帧深度算术均值。\n时间归一化，深度不归一化、不裁剪、不额外平滑；空窗不补零、不填充。\n窗左闭右开，末窗含右端点；相邻P可共享边界时刻。P表示事件间连续时间段，不按几何区域二次筛选。\n\n## 分组与平均（同Speed）\n仅纳入 common.load_trials() 的45人/270试次，严格核验完整时间线身份。无有效眼动深度的P自然排除。\n缺失边界、零时长只排除对应P，其他P保留；不以终点替代缺失事件。\n试次由 common.load_trials() 唯一确定，同人同场景仅一条；模态图再按每人高低烟雾平均；全量图再按每人可用模态平均；最后跨人等权平均。\n每张图受试者汇总曲线为该层级的个人曲线，粗线为逐窗有效人的平均。空窗贡献人数见mean_gaze_depth_curves.csv。\n80张单P图（每P16张），16张P1-P5合并图在MERGE/；根目录MERGE.png/svg为全量受试者合并图。\n\n## 其他产物\nperiod_audit.csv：trial×P边界、眼动帧数量、纳入和排除原因。\ntrial_gaze_depth_bins.csv：每会话绘图分箱数据和样本量；subject_gaze_depth_curves.csv：个人曲线；mean_gaze_depth_curves.csv：粗线。\nfigure_manifest.csv：80张单P图索引。run_summary.json：参数、参考脚本哈希、纳入计数。\n所有时间戳均保持源数据字符串；frame_extraction_audit.json保存原始文件与逐帧产物的对应关系。\n'
    methods += '\n背景为每个有效 trial 的独立曲线，按引导方式×烟雾条件的10色情景配色着色（低烟浅色、高烟深色），alpha=0.30 的细实线；粗线仍为受试者等权均值。subject_*_curves.csv 仅保存汇总数据，背景线来自 trial_*_bins.csv。\n'
    (args.output/'METHODS.md').write_text(methods.replace('BINS',str(args.bins)),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
