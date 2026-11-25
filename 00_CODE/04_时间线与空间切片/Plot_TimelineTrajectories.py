#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Display-negated Unity XZ trajectories in timeline P1-P5; 80 plots +16 merges.
python Plot_TimelineTrajectories.py [--force] [--dpi 300]
"""
from __future__ import annotations
import argparse,csv,json,gzip,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tqdm import tqdm
from TimelineTrialInput import (load_timeline, cohort_metadata, validate_metadata, validate_trial_table)
from common import PROJECT_ROOT,RAW_DATA_DIR,SCENARIO_FOLDERS,get_scenario_color,GUIDANCE_SMOKE_COLORS
from Analyze_TimelineSpeed import EVENTS,PERIODS,MODES,write_csv,parse_stamp
from UnifiedZoneQuery import UnifiedZoneQuery,VectorizedZoneClassifier

FLOORS={0:'Platform',1:'Hall',-1:'Outside height range'}


def geometry(path,query):
    """Draw CSV boundary loops; verify they describe the classifier's geometry."""
    with path.open(encoding='utf-8-sig',newline='') as f: zones=list(csv.DictReader(f))
    index={z.zone_id:z for z in query.zones}
    if set(index)!={z['zone_id'] for z in zones}:raise ValueError('CSV/JSON zone IDs differ')
    for z in zones:
        q=index[z['zone_id']]
        parts=json.loads(z['collider_parts_xz'])
        actual=[dict(xmin=p.xmin,xmax=p.xmax,zmin=p.zmin,zmax=p.zmax) for p in q.collider_parts]
        if len(parts)!=len(actual):raise ValueError('CSV/JSON collider mismatch')
        for a,b in zip(parts,actual):
            if not all(np.isclose(a[k],b[k],atol=1e-6,rtol=0) for k in b):raise ValueError('CSV/JSON geometry mismatch')
        if z['floor_id']!=q.floor_id:raise ValueError('CSV/JSON floor mismatch')
        for key in ['y_min','y_max','priority']:
            if not np.isclose(float(z[key]),getattr(q,key),atol=1e-6,rtol=0):raise ValueError('CSV/JSON height mismatch')
        z['loops']=[np.asarray(loop,float) for loop in json.loads(z['boundary_loops_xz'])]
    return zones


def split_segments(frames,times,pts,floors,max_gap=.1,max_step=3.):
    """Break links, never remove valid points or bridge files/floors/gaps/teleports."""
    if not len(pts):return np.array([],dtype=int)
    breaks=(np.diff(frames)!=1)|(np.diff(times)>max_gap)|(np.diff(times)<=0)
    breaks|=np.linalg.norm(np.diff(pts,axis=0),axis=1)>max_step
    breaks|=np.diff(floors)!=0
    return np.r_[0,np.cumsum(breaks)]


def hall_rectangle(zones):
    """Main hall includes the two end decision zones, but not side corridors."""
    ids={'HallOutsideGate_Main','DecisionZone_ABCD','DecisionZone_EFGH'}
    selected=[z for z in zones if z['zone_id'] in ids]
    if len(selected)!=3:raise ValueError('Missing main hall geometry')
    parts=[p for z in selected for p in json.loads(z['collider_parts_xz'])]
    return np.array([min(p['xmin'] for p in parts),min(p['zmin'] for p in parts)]),np.array([max(p['xmax'] for p in parts),max(p['zmax'] for p in parts)])


def hall_prefix(pts,frames,floors,entry_frame,bounds):
    lo,hi=bounds
    inside=np.all((pts[:,[0,2]]>=lo)&(pts[:,[0,2]]<=hi),axis=1)&(floors==1)
    exits=np.flatnonzero((frames>=entry_frame)&~inside)
    cutoff=int(frames[exits[0]]) if len(exits) else None
    return inside & ((frames<cutoff) if cutoff is not None else True),cutoff


def first_zone_visit(frames,zones,entry_frame,zone_index):
    """Retain the initial visit only; the first outside frame is excluded."""
    inside=zones==zone_index
    exits=np.flatnonzero((frames>=entry_frame)&~inside)
    cutoff=int(frames[exits[0]]) if len(exits) else None
    return inside & (frames>=entry_frame) & ((frames<cutoff) if cutoff is not None else True),cutoff


def load_periods(args,classifier,hall_bounds):
    trials=load_timeline(args.timeline,args.root)
    data={pid:[] for pid,_ in PERIODS};audit=[]
    fields=['trial_id','scenario','subject','group','src_file','period','frame','timestamp',
            'normalized_time','x','y','z','floor','zone_id','segment_id']
    handles={};writers={}
    for pid,_ in PERIODS:
        directory=args.output/pid;directory.mkdir(exist_ok=True)
        handles[pid]=gzip.open(directory/'trajectory_points.csv.gz','wt',encoding='utf-8-sig',newline='')
        writers[pid]=csv.writer(handles[pid]);writers[pid].writerow(fields)
    try:
        for trial_id,r in enumerate(tqdm(trials,desc='Extract original P trajectories'),1):
            path=Path(r['subject_file']);group=r['group']
            if r['scenario'] not in SCENARIO_FOLDERS:raise ValueError('Unknown scenario')
            base=dict(trial_id=trial_id,scenario=r['scenario'],subject=r['subject'],group=group or '',src_file=r['src_file'])
            if not group:
                audit.extend(dict(base,period=pid,status='excluded_test_or_unknown_group') for pid,_ in PERIODS);continue
            if group!=r['group']:raise ValueError('Timeline group mismatch')
            with path.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.reader(f))
            frames=[];stamps=[];points=[]
            for i,row in enumerate(raw[1:],1):
                try:p=np.asarray(row[1:4],float)
                except ValueError:continue
                if len(p)!=3 or not np.isfinite(p).all() or np.all(p==0):continue
                frames.append(i);stamps.append(row[0].strip());points.append(p)
            pts=np.asarray(points).reshape(-1,3);frames=np.asarray(frames)
            parsed=pd.to_datetime(stamps,format='%Y_%m_%d_%H_%M_%S_%f',errors='coerce')
            ok=~parsed.isna();pts=pts[ok];frames=frames[ok];stamps=np.asarray(stamps)[ok]
            ns=parsed[ok].astype('datetime64[ns]').astype('int64').to_numpy()
            zones=classifier.assign(pts);floors=classifier.floor_of(pts[:,1])
            entry=r['enter_hall_inside_gate_frame']
            hall_keep,exit_frame=hall_prefix(pts,frames,floors,int(entry),hall_bounds) if entry and not entry.startswith('Null') else (np.zeros(len(pts),dtype=bool),None)
            gate_keep,gate_exit=first_zone_visit(frames,zones,int(entry),classifier.global_index['HallInsideGate_Main']) if entry and not entry.startswith('Null') else (np.zeros(len(pts),dtype=bool),None)
            for k,(pid,_) in enumerate(PERIODS):
                info=dict(base,period=pid)
                a,b=[r[n+'_ts'] for n in EVENTS[k:k+2]]
                if not a or not b or a.startswith('Null') or b.startswith('Null'):
                    audit.append(dict(info,status='missing_boundary'));continue
                ia,ib=[int(r[n+'_frame']) for n in EVENTS[k:k+2]]
                if not 0<ia<=ib<len(raw) or raw[ia][0].strip()!=a or raw[ib][0].strip()!=b:raise ValueError('Timeline/source mismatch')
                ta,tb=pd.Timestamp(parse_stamp(a)).value,pd.Timestamp(parse_stamp(b)).value
                info.update(start_ts=a,end_ts=b,start_frame=ia,end_frame=ib,duration_s=(tb-ta)/1e9)
                if tb<=ta:audit.append(dict(info,status='nonpositive_duration'));continue
                keep=(frames>=ia)&(frames<=ib)&(ns>=ta)&(ns<=tb)
                before=int(keep.sum())
                if pid=='P1':keep &= floors==0
                if pid in ('P3','P4'):keep &= hall_keep
                if pid=='P3':keep &= gate_keep
                info.update(n_points_before_filter=before,n_removed=before-int(keep.sum()),
                            first_gate_exit_frame=gate_exit if pid=='P3' else '',
                            first_gate_exit_ts=raw[gate_exit][0].strip() if pid=='P3' and gate_exit is not None else '',
                            first_hall_exit_frame=exit_frame if pid in ('P3','P4') else '',
                            first_hall_exit_ts=raw[exit_frame][0].strip() if pid in ('P3','P4') and exit_frame is not None else '')
                p=pts[keep];fr=frames[keep];z=zones[keep];fl=floors[keep];st=stamps[keep];t=(ns[keep]-ta)/1e9
                if not len(p):audit.append(dict(info,status='no_points_after_spatial_filter' if before else 'no_valid_points'));continue
                if np.any(np.diff(t)<0):raise ValueError('Source time order invalid')
                seg=split_segments(fr,t,p,fl);u=(ns[keep]-ta)/(tb-ta)
                item=dict(info,points=p,frames=fr,zones=z,floors=fl,segments=seg)
                data[pid].append(item)
                audit.append(dict(info,status='included',n_points=len(p),n_segments=int(seg[-1])+1,
                                  n_null_zone=int((z<0).sum()),n_unknown_floor=int((fl<0).sum())))
                for j in range(len(p)):
                    writers[pid].writerow([trial_id,r['scenario'],r['subject'],group,r['src_file'],pid,
                                          int(fr[j]),st[j],u[j],*p[j],FLOORS[int(fl[j])],
                                          classifier.zone_list[z[j]].zone_id if z[j]>=0 else 'Null',int(seg[j])])
    finally:
        for f in handles.values():f.close()
    write_csv(args.output/'period_audit.csv',
              ['trial_id','scenario','subject','group','src_file','period','status','start_ts','end_ts',
               'start_frame','end_frame','duration_s','n_points','n_segments','n_null_zone','n_unknown_floor',
               'n_points_before_filter','n_removed','first_hall_exit_frame','first_hall_exit_ts',
               'first_gate_exit_frame','first_gate_exit_ts'],audit)
    return data,audit


def layout_plan(items,zones,classifier,pid,hall_bounds):
    """Same extent/outline for all 16 selections within each P, including excursions."""
    result={}
    if pid in ('P3','P4'):
        lo,hi=hall_bounds
        rectangle=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
        selected=[z for z in zones if 'Hall' in z['floor_id'].split('-') and all(np.all((loop>=lo)&(loop<=hi)) for loop in z['loops'])]
        selected=[dict(zone_id='MainHall_Rectangle',loops=[rectangle])]+selected
        pad=np.maximum((hi-lo)*.04,1.)
        return {1:dict(zones=selected,extent=(lo-pad,hi+pad))}
    for code,floor in FLOORS.items():
        samples=[r['points'][r['floors']==code][:,[0,2]] for r in items if np.any(r['floors']==code)]
        if not samples:continue
        visited={classifier.zone_list[i].zone_id for r in items for i in np.unique(r['zones'][r['floors']==code]) if i>=0}
        selected=[z for z in zones if z['zone_id'] in visited and (code<0 or floor in z['floor_id'].split('-'))]
        # Include the outline of every actually visited zone. Unknown-zone points stay visible.
        cloud=np.concatenate(samples+[loop for z in selected for loop in z['loops']])
        lo=cloud.min(axis=0);hi=cloud.max(axis=0);pad=np.maximum((hi-lo)*.04,1.)
        result[code]=dict(zones=selected,extent=(lo-pad,hi+pad))
    return result


def draw_tracks(ax,items,code,compact=False):
    """Shared trajectory layer: preserve points, breaks, colors and draw order."""
    # Shuffle deterministically so no modality is systematically painted last.
    for index in np.random.default_rng(20260916).permutation(len(items)):
        r=items[index];keep=r['floors']==code;p=-r['points'][keep][:,[0,2]]
        if not len(p):continue
        color=get_scenario_color(r['scenario']);segments=r['segments'][keep]
        runs=np.split(p,np.flatnonzero(np.diff(segments)!=0)+1)
        lines=[run for run in runs if len(run)>1]
        if lines:ax.add_collection(LineCollection(lines,colors=color,linewidths=.45 if compact else .6,alpha=.55,rasterized=True,zorder=2))
        ax.scatter(p[:,0],p[:,1],s=.22 if compact else .4,c=color,alpha=.5,linewidths=0,rasterized=True,zorder=3)


def draw(fig,spec,pid,items,plan,compact=False):
    codes=list(plan)
    widths=[max(.28,(plan[c]['extent'][1][0]-plan[c]['extent'][0][0])/(plan[c]['extent'][1][1]-plan[c]['extent'][0][1])) for c in codes]
    nested=spec.subgridspec(1,len(codes),width_ratios=widths,wspace=.32)
    for j,code in enumerate(codes):
        ax=fig.add_subplot(nested[0,j]);cfg=plan[code]
        for z in cfg['zones']:
            for loop in z['loops']:
                closed=-np.vstack([loop,loop[0]])  # Display only: (X, Z) -> (-X, -Z).
                ax.plot(closed[:,0],closed[:,1],color='#737A80',lw=.55 if compact else .8,alpha=.7,zorder=0)
        draw_tracks(ax,items,code,compact)
        lo,hi=cfg['extent'];ax.set(xlim=(-hi[0],-lo[0]),ylim=(-hi[1],-lo[1]),xlabel='-Unity X (m)',ylabel='-Unity Z (m)')
        ax.set_aspect('equal',adjustable='box');ax.set_title(f'{pid} | {FLOORS[code]}',fontsize=9 if compact else 13)
        ax.tick_params(labelsize=7 if compact else 10);ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False)
        if not any(np.any(r['floors']==code) for r in items):ax.text(.5,.5,'No samples',ha='center',transform=ax.transAxes)


def load_cached_periods(args,classifier,trials):
    """Rebuild drawing inputs from approved points without rerunning extraction."""
    summary=json.loads((args.output/'run_summary.json').read_text(encoding='utf-8'))
    validate_metadata(summary,trials)
    for name in ('timeline','geometry'):
        if summary[name+'_sha256']!=hashlib.sha256(getattr(args,name).read_bytes()).hexdigest():
            raise ValueError(f'{name} changed; regenerate trajectories first')
    if 'first HallInsideGate_Main visit' not in summary.get('trajectory_rule',''):
        raise ValueError('Regenerate trajectories with the first gate-exit rule first')
    floor_codes={name:code for code,name in FLOORS.items()};data={}
    for pid,_ in PERIODS:
        table=pd.read_csv(args.output/pid/'trajectory_points.csv.gz',keep_default_na=False)
        validate_trial_table(table,trials)
        if not table.period.eq(pid).all():raise ValueError('Cached period mismatch')
        if len(table)!=summary['periods'][pid]['n_points']:raise ValueError('Cached point count mismatch')
        data[pid]=[]
        for tid,frame in table.groupby('trial_id',sort=False):
            r=frame.iloc[0]
            data[pid].append(dict(trial_id=int(tid),scenario=r.scenario,subject=r.subject,
                                 points=frame[['x','y','z']].to_numpy(float),
                                 floors=np.asarray([floor_codes[f] for f in frame.floor]),
                                 zones=np.asarray([classifier.global_index[z] if z!='Null' else -1 for z in frame.zone_id]),
                                 segments=frame.segment_id.to_numpy(int)))
        if len(data[pid])!=summary['periods'][pid]['n_trials']:raise ValueError('Cached trial count mismatch')
    return data


def render_traj_only(data,plans,out,dpi):
    """Fixed-size transparent AllTrials exports with identical metres per pixel."""
    lo=np.min([cfg['extent'][0] for p in plans.values() for cfg in p.values()],axis=0)
    hi=np.max([cfg['extent'][1] for p in plans.values() for cfg in p.values()],axis=0)
    span=hi-lo;width,height=int(round(9*dpi)),int(round(8*dpi));gap=int(round(.2*dpi))
    max_floors=max(len(plan) for plan in plans.values())
    scale=min((width-gap*(max_floors-1))/(max_floors*span[0]),height/span[1])
    panel_width,panel_height=span*scale;rows=[]
    for pid,_ in tqdm(PERIODS,desc='Render AllTrials trajectory-only'):
        codes=list(plans[pid]);occupied=len(codes)*panel_width+(len(codes)-1)*gap
        left=(width-occupied)/2;bottom=(height-panel_height)/2
        fig=plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi,frameon=False);panels=[]
        try:
            for j,code in enumerate(codes):
                rect=[(left+j*(panel_width+gap))/width,bottom/height,panel_width/width,panel_height/height]
                ax=fig.add_axes(rect,frameon=False)
                draw_tracks(ax,data[pid],code)
                ax.set(xlim=(-hi[0],-lo[0]),ylim=(-hi[1],-lo[1]))
                ax.set_aspect('equal');ax.set_axis_off()
                panels.append(dict(floor=FLOORS[code],axes_fraction=rect))
            filename=f'{pid}/all_AllTrials_traj_only.png'
            (out/pid).mkdir(parents=True,exist_ok=True)
            with plt.rc_context({'savefig.bbox':None,'savefig.pad_inches':0}):
                fig.savefig(out/filename,dpi=dpi,transparent=True,bbox_inches=None,pad_inches=0)
            rows.append(dict(period=pid,png=filename,panels=panels,n_trials=len(data[pid]),
                             n_points=sum(len(r['points']) for r in data[pid])))
        finally:
            plt.close(fig)
    metadata=dict(images=rows,pixel_size=[width,height],dpi=dpi,pixels_per_metre=float(scale),
                  metres_per_pixel=float(1/scale),display_xlim=[float(-hi[0]),float(-lo[0])],
                  display_ylim=[float(-hi[1]),float(-lo[1])],background='transparent',
                  layout='Original floor order; shared viewport and scale; single floor centred',
                  palette=GUIDANCE_SMOKE_COLORS)
    (out/'traj_only_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(RAW_DATA_DIR))
    parser.add_argument('--timeline',type=Path,default=Path(PROJECT_ROOT)/'TIMELINE/subject_zone_timeline.csv')
    parser.add_argument('--geometry',type=Path,default=Path(PROJECT_ROOT)/'Results_Guidance_Spatial_Causal_Analysis/zone_annotation/semantic_zones.csv')
    parser.add_argument('--output',type=Path,default=Path(PROJECT_ROOT)/'TIMELINE/Traj_By_P')
    parser.add_argument('--dpi',type=int,default=300);parser.add_argument('--force',action='store_true')
    parser.add_argument('--traj-only',action='store_true',help='Export five undecorated AllTrials PNGs from existing trajectory point caches')
    args=parser.parse_args()
    if args.dpi<=0:parser.error('--dpi must be positive')
    if args.traj_only:
        targets=[args.output/pid/'all_AllTrials_traj_only.png' for pid,_ in PERIODS]+[args.output/'traj_only_manifest.json']
        if any(path.exists() for path in targets) and not args.force:parser.error('Trajectory-only output exists; use --force')
    elif args.output.exists() and any(args.output.iterdir()) and not args.force:parser.error('Nonempty output; use --force')
    trials=load_timeline(args.timeline,args.root)  # preflight before creating outputs
    args.output.mkdir(parents=True,exist_ok=True)
    classifier=VectorizedZoneClassifier(UnifiedZoneQuery(args.geometry.with_suffix('.json')))
    zones=geometry(args.geometry,classifier.query);hall_bounds=hall_rectangle(zones)
    if args.traj_only:data=load_cached_periods(args,classifier,trials)
    else:data,audit=load_periods(args,classifier,hall_bounds)
    plans={pid:layout_plan(data[pid],zones,classifier,pid,hall_bounds) for pid,_ in PERIODS}
    if any(not p for p in plans.values()):raise ValueError('A period has no trajectory data')
    pure=render_traj_only(data,plans,args.output,args.dpi)
    if args.traj_only:
        print(f'{len(pure["images"])} trajectory-only PNGs; {pure["pixel_size"]} pixels',flush=True)
        return
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':11,'svg.fonttype':'none'})
    selections=[('scenario',s) for s in SCENARIO_FOLDERS]+[('modality',m) for m in MODES]+[('all','AllTrials')]
    def choose(pid,level,name):return [r for r in data[pid] if level=='all' or (r['scenario']==name if level=='scenario' else r['scenario'].rsplit('-',1)[0]==name)]
    def legend(items):
        present={r['scenario'] for r in items}
        return [Line2D([0],[0],color=get_scenario_color(s),lw=2,label=s) for s in SCENARIO_FOLDERS if s in present]
    manifest=[]
    for pid,label in tqdm(PERIODS,desc='Plot 80 period trajectories'):
        for level,name in selections:
            items=choose(pid,level,name);fig=plt.figure(figsize=(10,8));spec=fig.add_gridspec(1,1,left=.09,right=.96,bottom=.20,top=.88)
            draw(fig,spec[0,0],pid,items,plans[pid]);fig.suptitle(f'{pid}: {label}\n{name} | {len({r["subject"] for r in items})} participants, {len(items)} trials',fontsize=12)
            handles=legend(items)
            if handles:fig.legend(handles=handles,loc='lower center',ncol=min(3,len(handles)),fontsize=9,frameon=False,bbox_to_anchor=(.5,.01))
            stem=f'{pid}/{level}_{name}'
            for ext in ['png','svg']:fig.savefig(args.output/(stem+'.'+ext),dpi=args.dpi,bbox_inches='tight')
            plt.close(fig)
            manifest.append(dict(period=pid,level=level,selection=name,n_subjects=len({r['subject'] for r in items}),n_trials=len(items),n_points=sum(len(r['points']) for r in items),png=stem+'.png',svg=stem+'.svg'))
    merge=args.output/'MERGE';merge.mkdir(exist_ok=True)
    for level,name in tqdm(selections,desc='Merge P1-P5 trajectories'):
        fig=plt.figure(figsize=(18,12));grid=fig.add_gridspec(2,3,left=.05,right=.98,bottom=.06,top=.93,hspace=.30,wspace=.27)
        allitems=[]
        for k,(pid,_) in enumerate(PERIODS):
            items=choose(pid,level,name);allitems.extend(items);draw(fig,grid[k//3,k%3],pid,items,plans[pid],compact=True)
        ax=fig.add_subplot(grid[1,2]);ax.axis('off')
        ax.text(0,1,'Display coordinates: (-X, -Z)\nP1-P5 | '+name+'\n\nP1: platform only\nP3: inside gate, stop at first exit\nP4: main hall, stop at first hall exit\nNo re-entry after each cutoff\nColors: guidance modality and smoke',va='top',transform=ax.transAxes,fontsize=11,linespacing=1.5)
        ax.legend(handles=legend(allitems),loc='lower left',frameon=False,fontsize=9,ncol=1)
        fig.suptitle('Timeline trajectories | '+name,fontsize=17)
        for ext in ['png','svg']:
            fig.savefig(merge/f'{level}_{name}.{ext}',dpi=args.dpi,bbox_inches='tight')
            if level=='all':fig.savefig(args.output/f'MERGE.{ext}',dpi=args.dpi,bbox_inches='tight')
        plt.close(fig)
    write_csv(args.output/'figure_manifest.csv',['period','level','selection','n_subjects','n_trials','n_points','png','svg'],manifest)
    config={**cohort_metadata(load_timeline(args.timeline,args.root)), 'timeline':str(args.timeline),'geometry':str(args.geometry),'palette':GUIDANCE_SMOKE_COLORS,
            'main_hall_bounds_xz':[v.tolist() for v in hall_bounds], 'display_coordinates':'(-Unity X, -Unity Z); metres; raw data unchanged',
            'trajectory_rule':'P1 platform only; P3 first HallInsideGate_Main visit only; P3/P4 also share first main-hall exit; P2/P4/P5 otherwise unchanged',
            'timeline_sha256':hashlib.sha256(args.timeline.read_bytes()).hexdigest(),
            'geometry_sha256':hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
            'status_counts':dict(Counter(r['status'] for r in audit)),'figures':80,'merges':16,
            'traj_only_figures':len(pure['images']),'traj_only_manifest':'traj_only_manifest.json',
            'periods':{pid:{'floors':[FLOORS[c] for c in plans[pid]],'n_trials':len(data[pid]),'n_points':sum(len(r['points']) for r in data[pid]),'zones':{FLOORS[c]:[z['zone_id'] for z in plans[pid][c]['zones']] for c in plans[pid]}} for pid,_ in PERIODS}}
    (args.output/'run_summary.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'METHODS.md').write_text('''# P1–P5 平面运动轨迹
按timeline六个相邻事件划分五个P，两端帧均纳入。边界时间戳及行号均与原始Subject CSV核对。
取B/C/D列原始XYZ，仅绘图时将X和Z各取一次负，横轴为-Unity X、纵轴为-Unity Z，轨迹、分区边界和坐标范围同步变换，等比例米制。保存原始坐标和原始timestamp；不插值、不平滑、不分箱、不平均轨迹、不降采样。
P1仅保留Platform层，不显示Hall及Outside height range子图。P2/P5沿用原规则。
P3/P4仅保留主站厅矩形内的Hall层轨迹：矩形由HallOutsideGate_Main和DecisionZone_ABCD/EFGH的collider边界共同确定，包含两端决策区，不包含侧向过道。
P3额外限定为HallInsideGate_Main的首次连续访问：从enter_hall_inside_gate帧开始，在统一分区接口首次判为其他分区或Null时截断；离开帧与后续所有返回点均不纳入。Gate_01~04不属于HallInsideGate_Main，与timeline事件定义一致。P4仍沿用原timeline起点，不前移至P3的新展示终点。
从enter_hall_inside_gate帧起查找首次离开矩形或Hall层的有效帧，P3/P4共用该截断帧，该帧及之后所有点均不纳入，包括返回站厅的点；不会在P4起点重新计数。
原timeline时间戳、四类指标时间演化结果均不修改，归一化时间仍相对于原P起止时间计算。原始无效位置或无法解析时间的帧跳过；TEST不参与绘图，缺失边界/零时长仅排除对应P。
分组为10场景、5模态合并烟雾、全部trial。仅保留 common.load_trials() 的45人/270试次，同人同场景唯一会话；颜色来自common.GUIDANCE_SMOKE_COLORS/get_scenario_color。
分层只用于防止XZ投影重叠，使用统一UnifiedZoneQuery的高度漂移容差；原始Y不改变。楼梯按各帧所在高度分层，跨层不连线。
线框来自用户指定semantic_zones.csv，已与分类器JSON几何核对。P3/P4绘制主站厅矩形与完全位于矩形内的分区边界；其余P绘制所保留轨迹实际经过分区。16张图共享该P的分层、范围和线框。
不同trial不连线；无效/缺帧、时间间隔>0.1秒、时间不递增、位移>3米或换层处断线；空间筛选和首次离开截断后保留的点不再降采样。
P*/trajectory_points.csv.gz保留所有绘图点及原始帧号、时间戳、XYZ、归一化时间、floor/zone_id/segment_id，segment_id在每个trial×P内编号。
period_audit.csv记录纳入和排除、过滤前点数、去除点数、首次离厅以及P3首次离开gate内部的原始帧/时间戳；figure_manifest.csv索引80张单P图；MERGE/为16张合并图；根目录MERGE.png/svg为全部trial合成。
PNG默认300dpi。SVG线框/文字为矢量，密集轨迹点线栅格化以控制文件体积，未减少样本。
各P额外输出all_AllTrials_traj_only.png，仅包含原色轨迹点线，去掉标题、坐标轴、图例、网格、分区线框及文字，面外透明；P2仍合并两个楼层且保留原顺序。P1–P5统一9×8英寸（300dpi时2700×2400像素）、全局空间范围及米/像素比例，单楼层居中不放大。traj_only_manifest.json记录文件、尺寸、比例尺、各P人数/点数及布局。--traj-only复用现有逐帧轨迹缓存补画纯图，不重新提取轨迹或覆盖原图。
''',encoding='utf-8')
    print(json.dumps({k:config[k] for k in ['status_counts','figures','merges']},indent=2))


if __name__=='__main__':main()
