#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference eye scan speed angular rate in timeline P1-P5 (deg/s).
python Analyze_TimelineScanSpeed.py [--bins 50] [--force]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from TimelineFinalAssets import plot_final_assets
from TimelineTrialInput import (load_timeline, cohort_metadata, require_fresh_frame_cohort)
from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS
from Analyze_TimelineSpeed import draw_selection_means, guidance_legend_handles, load_cached_curves, selected_trial_curves, draw_trial_backgrounds, trial_legend_handles, EVENTS, PERIODS, COLORS, SCENARIO_COLORS, grouping, mean_curves, write_csv
from Analyze_TimelineGazeDepth import reference, REFERENCE, source_rows, bounds


def scan_intervals(head, gaze_raw, positions, ns):
    """Reference gaze_scan_speed: inverse head rotation, 3-sample smoothing, angle/dt."""
    n=len(ns)
    if not n:
        empty=np.array([]);flags=np.array([],dtype=bool)
        return empty,empty,empty,empty,flags,flags,np.empty((0,3)),np.empty((0,3)),flags
    times=(ns-ns[0])/1e9
    dt=np.r_[np.nan,np.diff(times)]
    head_valid=np.isfinite(head).all(axis=1)
    rot=Rotation.from_euler('zxy',np.where(np.isfinite(head),head,0)[:,[2,0,1]],degrees=True)
    gaze,gaze_valid=reference.unit_vectors(gaze_raw)
    local=rot.inv().apply(np.nan_to_num(gaze))
    local_valid=gaze_valid & head_valid
    smooth=reference.smooth_vectors(local,local_valid,times,reference.CONFIG['gaze_smooth_samples'])
    angle=reference.interval_angles(smooth,np.isfinite(smooth).all(axis=1),dt)
    pos_valid=np.isfinite(positions).all(axis=1)
    step=np.r_[np.nan,np.linalg.norm(np.diff(positions,axis=0),axis=1)]
    motion=pos_valid & np.r_[False,pos_valid[:-1]]
    motion &= (dt>0)&(dt<=reference.CONFIG['max_gap_s'])&(step<=reference.CONFIG['max_position_step_m'])
    valid=motion & np.isfinite(angle)
    rate=np.divide(angle,dt,out=np.full(n,np.nan),where=valid)
    local[~local_valid]=np.nan
    return angle,dt,rate,step,valid,motion,local,smooth,local_valid


def load_trial(subject_path):
    eye_path=subject_path.with_name(subject_path.name.replace('_Subject_','_EyeGaze_'))
    raw_pos=source_rows(subject_path)
    if not eye_path.exists():
        return dict(raw_pos=raw_pos,missing_eye=True,eye_path=eye_path)
    pos,pq=reference.read_raw(subject_path,['TimeStamp','PosX','PosY','PosZ'])
    eye,eq=reference.read_raw(eye_path,['TimeStamp','HeadRotX','HeadRotY','HeadRotZ','GazeDirX','GazeDirY','GazeDirZ'])
    ens=eye.ParsedTime.astype('datetime64[ns]').astype('int64').to_numpy()
    pns=pos.ParsedTime.astype('datetime64[ns]').astype('int64').to_numpy()
    aligned,sync=reference.align_positions(ens,pns,pos[['PosX','PosY','PosZ']].to_numpy(float))
    head=eye[['HeadRotX','HeadRotY','HeadRotZ']].to_numpy(float)
    gaze_raw=eye[['GazeDirX','GazeDirY','GazeDirZ']].to_numpy(float)
    angle,dt,rate,step,valid,motion,local,smooth,local_valid=scan_intervals(head,gaze_raw,aligned,ens)
    raw_eye=source_rows(eye_path)
    row_map={r[0]:i for i,r in enumerate(raw_eye[1:],1) if r}
    rows=np.array([row_map[s] for s in eye.TimeStamp],dtype=int)
    stamps=eye.TimeStamp.to_numpy()
    previous_ns=np.r_[ens[0],ens[:-1]] if len(ens) else ens.copy()
    frame=pd.DataFrame({'eye_frame':rows,'timestamp':stamps,
                        'previous_eye_frame':np.r_[0,rows[:-1]] if len(rows) else rows,
                        'previous_timestamp':np.concatenate([np.array([''],dtype=object),stamps[:-1]]) if len(stamps) else stamps,
                        'interval_s':dt,'eye_angle_deg':angle,'scan_speed_deg_s':rate,
                        'scan_interval_valid':valid.astype(int),'motion_valid':motion.astype(int),
                        'position_step_m':step,'position_sync':np.array(['missing','exact','interpolated'])[sync]})
    for j,axis in enumerate('xyz'):
        frame['head_rot_'+axis]=head[:,j]
        frame['subject_'+axis]=aligned[:,j]
        frame['gaze_world_'+axis]=gaze_raw[:,j]
        frame['gaze_local_'+axis]=local[:,j]
        frame['gaze_local_smoothed_'+axis]=smooth[:,j]
    frame['local_direction_valid']=local_valid.astype(int)
    return dict(raw_pos=raw_pos,missing_eye=False,eye_path=eye_path,ns=ens,previous_ns=previous_ns,
                angle=angle,dt=dt,rate=rate,valid=valid,frame=frame,pos_audit=pq,eye_audit=eq)


def bin_intervals(left,right,rate,a,b,bins):
    """Clip original intervals to each normalized bin; angle/time, not sample mean."""
    curve=np.full(bins,np.nan);counts=np.zeros(bins,dtype=int);exposure=np.zeros(bins)
    # Relative seconds avoid rounding large epoch nanoseconds at fractional bin edges.
    x=(left-a)/1e9;y=(right-a)/1e9
    edges=np.linspace(0,(b-a)/1e9,bins+1)
    for j in range(bins):
        overlap=np.maximum(0,np.minimum(y,edges[j+1])-np.maximum(x,edges[j]))
        keep=(overlap>0)&np.isfinite(rate)
        counts[j]=keep.sum();exposure[j]=overlap[keep].sum()
        if exposure[j]>0:
            curve[j]=np.sum(rate[keep]*overlap[keep])/exposure[j]
    return curve,counts,exposure


def extract_period(data,trial,k,bins):
    boundary,meta,status=bounds(data,trial,k)
    if boundary is None:return None,meta,status
    if data['missing_eye']:return None,meta,'missing_eye_file'
    a,b=boundary
    selected=(data['previous_ns']<b)&(data['ns']>a)
    curve,counts,exposure=bin_intervals(data['previous_ns'][selected],data['ns'][selected],data['rate'][selected],a,b,bins)
    meta.update(n_source_frames=int(selected.sum()),n_valid_frames=int((selected&data['valid']).sum()),
                n_invalid_frames=int((selected&~data['valid']).sum()),n_nonempty_bins=int((counts>0).sum()),
                valid_time_s=float(exposure.sum()))
    if not counts.any():return None,meta,'no_valid_scan_speed'
    return (curve,counts,exposure),meta,'included'


def export_frames(data,trial,output):
    if data['missing_eye']:
        return dict(scenario=trial['scenario'],subject=trial['subject'],src_file=trial['src_file'],status='missing_eye_file')
    frame=data['frame']
    for k,(pid,_) in enumerate(PERIODS):
        boundary,_,_=bounds(data,trial,k)
        u=np.full(len(frame),np.nan);exposure=np.zeros(len(frame))
        if boundary is not None:
            a,b=boundary
            left=np.maximum(data['previous_ns'],a);right=np.minimum(data['ns'],b)
            selected=right>left
            u[selected]=((left[selected]-a)+(right[selected]-a))*.5/(b-a)
            exposure[selected & data['valid']]=(right[selected & data['valid']]-left[selected & data['valid']])/1e9
        frame[pid+'_normalized_midpoint']=u
        frame[pid+'_valid_time_s']=exposure
    target=output/'frames'/trial['scenario']/trial['subject']/(data['eye_path'].stem+'_scan_speed.csv.gz')
    target.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(target,index=False,encoding='utf-8-sig',compression='gzip')
    return dict(scenario=trial['scenario'],subject=trial['subject'],src_file=trial['src_file'],
                eye_file=data['eye_path'].name,status='extracted',output=str(target.relative_to(output)),
                n_eye_frames=len(frame),n_scan_valid=int(frame.scan_interval_valid.sum()),
                n_exact=int((frame.position_sync=='exact').sum()),
                n_interpolated=int((frame.position_sync=='interpolated').sum()),
                n_unmatched=int((frame.position_sync=='missing').sum()),
                subject_cleaning=data['pos_audit'],eye_cleaning=data['eye_audit'])


def plot_merges(trials,curves,output,centers,dpi, *, all_only=False):
    """Five panels + legend panel, matching Speed/MERGE.png; one per selection."""
    groups={pid:grouping(trials,curves,pid) for pid,_ in PERIODS}
    merge_dir=output/'MERGE'
    merge_dir.mkdir(exist_ok=True)
    for group_index in tqdm(range(15,16) if all_only else range(16),desc='Merge P1-P5'):
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
            ax.set(xlim=(0,1),xlabel='Normalized time',ylabel='Mean eye scan speed (deg/s)')
            ax.set_ylim(0,125. if level=='all' else max(1.,maximum*1.08))
            ax.tick_params(labelbottom=True)
            ax.spines[['top','right']].set_visible(False)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.04,.95,'P1-P5 eye scan speed evolution\n\nThin solid: individual trials (alpha=0.30)\nBold: equal-weight participant mean\n\n'+str(len(centers))+' time bins per period\nOnly time is normalized\nOriginal timestamp intervals\n3-frame direction smoothing\nAngle / valid time; no extra smoothing',
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
                                                 scan_speed_deg_s=curve[j]))
                mean_handles = draw_selection_means(ax, centers, groups, level, name)
                ax.legend(handles=mean_handles if level=='all' else trial_legend_handles(trial_lines) + mean_handles,
                          loc='upper right',frameon=False,ncol=2,fontsize=8)
                for j, u in enumerate(centers):
                    mean_rows.append(dict(period=pid,level=level,selection=name,bin=j+1,
                                          time_normalized=u,scan_speed_deg_s=mean[j],n_subjects=int(n[j])))
            else:
                ax.text(.5,.5,'No valid periods',ha='center',transform=ax.transAxes)
            ax.set(xlim=(0,1), ylim=(0,125. if level=='all' else max(1.,maximum*1.12)),
                   xlabel='Normalized time within period',ylabel='Mean eye scan speed (deg/s)')
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
        write_csv(output/'subject_scan_speed_curves.csv',
                  ['period','level','selection','subject','bin','time_normalized','scan_speed_deg_s'],subject_rows)
        write_csv(output/'mean_scan_speed_curves.csv',
                  ['period','level','selection','bin','time_normalized','scan_speed_deg_s','n_subjects'],mean_rows)
        write_csv(output/'figure_manifest.csv',
                  ['period','level','selection','n_subjects','png','svg'],manifest)
    plot_merges(trials,curves,output,centers,dpi,all_only=all_only)
    plot_final_assets(trials, curves, output, centers, dpi, 'scan_speed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(RAW_DATA_DIR))
    parser.add_argument('--timeline', type=Path, default=Path(PROJECT_ROOT) / 'TIMELINE/subject_zone_timeline.csv')
    parser.add_argument('--output', type=Path, default=Path(PROJECT_ROOT) / 'TIMELINE/ScanSpeed')
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
        trials, curves, centers = load_cached_curves(args.output, 'scan_speed', 'scan_speed_deg_s')
        plot_final_assets(trials, curves, args.output, centers, args.dpi, 'scan_speed')
        return
    if args.all_participants_only:
        trials, curves, centers = load_cached_curves(args.output, 'scan_speed', 'scan_speed_deg_s')
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
    for i, r in enumerate(tqdm(trials, desc='Read scan_speed periods')):
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
            curve, counts, exposure = value
            curves[i, pid] = curve
            for j, u in enumerate(centers):
                trial_rows.append(dict(base, bin=j + 1, time_normalized=u,
                                       scan_speed_deg_s=curve[j], n_samples=int(counts[j]),valid_time_s=exposure[j]))
    audit_fields = ['trial_id','scenario','subject','group','src_file','period','status',
                    'start_ts','end_ts','start_frame','end_frame','duration_s',
                    'n_source_frames','n_valid_frames','n_invalid_frames','n_nonempty_bins','valid_time_s']
    write_csv(args.output/'period_audit.csv', audit_fields, audits)
    write_csv(args.output/'trial_scan_speed_bins.csv',
              ['trial_id','scenario','subject','group','src_file','period','bin',
               'time_normalized','scan_speed_deg_s','n_samples','valid_time_s'], trial_rows)
    manifest = plot_results(trials, curves, args.output, centers, args.dpi)
    (args.output/'frame_extraction_audit.json').write_text(json.dumps(frame_audits,indent=2),encoding='utf-8')
    summary = {**cohort_metadata(trials), 'timeline':str(args.timeline),'raw_data':str(args.root),'bins':args.bins,
               'timeline_rows':len(trials),'figures':len(manifest),'merge_figures':16,
               'reference':str(REFERENCE),'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
               'gaze_smooth_samples':reference.CONFIG['gaze_smooth_samples'],'max_gap_s':reference.CONFIG['max_gap_s'],'max_position_step_m':reference.CONFIG['max_position_step_m'],
               'n_eye_frames':sum(r.get('n_eye_frames',0) for r in frame_audits),
               'n_valid_scan_intervals':sum(r.get('n_scan_valid',0) for r in frame_audits),
               'period_status_counts':dict(Counter(a['status'] for a in audits)),
               'included_by_period':dict(Counter(a['period'] for a in audits if a['status']=='included'))}
    (args.output/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    methods = "# Eye Scan Speed Timeline\n\n## 指标及参考口径\n指标对应extract_timestamp_metrics.py的gaze_scan_speed（EyeAngle_deg/EyeTime_s），单位deg/s；不是人的运动速度，也不是眼跳事件率。\n读取EyeGaze的HeadRotX/Y/Z及GazeDirX/Y/Z。GazeDir有限且长度在(0.5,1.5)内才有效，并归一化为单位向量。\n按Rotation.from_euler('zxy',head[:,[2,0,1]],degrees=True)建立头部旋转，用逆旋转把世界注视方向转换到头部局部坐标。\n局部方向还要求HeadRot有限。复用参考smooth_vectors，在连续有效段内进行3帧居中移动均值（边缘min_periods=1），然后重新单位化。\n方向平滑不跨越无效帧或>0.1s时间间隔。在完整trial上计算，不在每个P边界重新启动平滑。\n相邻平滑后局部方向的夹角/真实dt即逐间隔scan speed。缺失不填零、不补曲线；这是参考算法自带的方向平滑，绘图不额外平滑。\n保留参考motion_valid条件：相邻Subject位置有限、0<dt<=0.1s、相邻位置位移<=3m。无需Hit1有效性。\n位置对齐复用参考align_positions：精确时间戳优先，否则仅在<=0.1s位置间隔内插值；内部时间显式使用纳秒。\n不使用空间网格、楼层吸附或空间区划过滤，参考脚本原文件未修改。\n\n## 逐间隔与绘图数据\nframes/内csv.gz保留完整清洗后眼动序列：原始两端时间戳、源行号、dt、eye_angle_deg、scan_speed_deg_s及有效标志。\n同时保存世界注视方向、头部Euler角、原始局部方向和平滑后局部方向，便于反查计算。首帧无前一间隔，速度为空。\n时间解析失败和重复歧义时间戳按参考read_raw规则排除，数量在frame_extraction_audit.json。\nP1–P5边界、分组、50个归一化时间窗与HeadSearch一致。区间跨P或窗边界时按实际重叠时长拆分角度贡献。\n每窗=sum(angle*overlap/dt)/sum(overlap)，指标保持deg/s，仅时间归一化。空窗不插值、不补零、不裁剪极值。\nP_normalized_midpoint是区间与P交集的派生归一化中点，不是原始时间戳。\n每窗贡献间隔数和有效时长见trial_scan_speed_bins.csv；各P纳入/排除见period_audit.csv。\n\n## 分组与图版\n试次由 common.load_trials() 唯一确定，同人同场景仅一条；模态图再按每人的烟雾条件平均；全量图再按每人的可用模态平均；最后跨人等权平均。\n半透明同色个人曲线、彩色粗均值线；无有效值时不纳入该窗均值，人数见mean_scan_speed_curves.csv。\n标准入口为 common.load_trials() 的45人/270试次，无有效扫描速度的P据实排除；缺失边界或零时长仅影响对应P。\n每P16图共80张PNG+SVG，MERGE/下16张P1–P5合并图，根目录MERGE.png/svg为全量版本。\nsubject_scan_speed_curves.csv、mean_scan_speed_curves.csv、trial_scan_speed_bins.csv保存全部曲线数据。\nrun_summary.json记录参考脚本哈希和参数。重新运行Analyze_TimelineScanSpeed.py覆盖时使用--force。\n"
    methods += '\n背景为每个有效 trial 的独立曲线，按引导方式×烟雾条件的10色情景配色着色（低烟浅色、高烟深色），alpha=0.30 的细实线；粗线仍为受试者等权均值。subject_*_curves.csv 仅保存汇总数据，背景线来自 trial_*_bins.csv。\n'
    (args.output/'METHODS.md').write_text(methods,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
