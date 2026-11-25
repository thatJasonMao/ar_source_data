#!/usr/bin/env python3
"""Four frame-level metric heatmaps on the approved P1-P5 trajectory support."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import shapely
from shapely.geometry import box
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from tqdm import tqdm
from TimelineTrialInput import (load_timeline, cohort_metadata, validate_metadata, validate_trial_table)
from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS
from Analyze_TimelineSpeed import PERIODS, MODES
from Analyze_TimelineGazeDepth import load_trial as load_depth
from Analyze_TimelineHeadSearch import load_trial as load_head
from Analyze_TimelineScanSpeed import load_trial as load_scan
from UnifiedZoneQuery import UnifiedZoneQuery, VectorizedZoneClassifier
from a5_colormaps import get_cmap
import Visualize_Spatial_Metric_Surfaces as surfaces

ROOT=Path(PROJECT_ROOT)
METRICS={
    'Gazedepth':('Gaze depth','m',99,'gaze_depth_m',load_depth),
    'HeadSearch':('Head angular speed','deg/s',162,'head_search_deg_s',load_head),
    'ScanSpeed':('Eye scan speed','deg/s',100,'scan_speed_deg_s',load_scan),
    'Speed':('Planar movement speed','m/s',83,None,None),
}
SELECTIONS=[('scenario',s) for s in SCENARIO_FOLDERS]+[('modality',s) for s in MODES]+[('all','AllTrials')]
FLOORS={'Platform':0,'Hall':1}
TIME_FORMAT='%Y_%m_%d_%H_%M_%S_%f'


def ns_of(stamps):
    return pd.to_datetime(stamps,format=TIME_FORMAT).astype('datetime64[ns]').astype('int64').to_numpy()


def selected(df,level,name):
    if level=='all':return df
    return df[df.scenario.eq(name)] if level=='scenario' else df[df.scenario.str.startswith(name+'-')]


def support_mask(ns,trajectory,previous=None):
    """Only exact samples or interpolation within the same retained trajectory run.

    Angular intervals must have both timestamps inside one retained run; no
    previous point from outside the selected P/domain can contribute a turn.
    """
    keep=np.zeros(len(ns),bool)
    for _,run in trajectory.groupby('segment_id',sort=False):
        t=ns_of(run.timestamp)
        hit=(ns>=t[0])&(ns<=t[-1])
        if previous is not None:hit &= previous>=t[0]
        keep |= hit
    return keep


def load_support(traj):
    trials=load_timeline(ROOT/'TIMELINE/subject_zone_timeline.csv')
    summary=json.loads((traj/'run_summary.json').read_text(encoding='utf-8'))
    validate_metadata(summary,trials)
    if 'first HallInsideGate_Main visit' not in summary.get('trajectory_rule',''):
        raise ValueError('Regenerate Traj_By_P with the first gate-exit rule first')
    if summary['timeline_sha256']!=hashlib.sha256((ROOT/'TIMELINE/subject_zone_timeline.csv').read_bytes()).hexdigest():
        raise ValueError('Trajectory timeline has changed')
    tables={pid:pd.read_csv(traj/pid/'trajectory_points.csv.gz',keep_default_na=False) for pid,_ in PERIODS}
    for table in tables.values():validate_trial_table(table,trials)
    assert tables['P3'].zone_id.eq('HallInsideGate_Main').all()
    return tables,summary


def footprints(tables,summary,geometry):
    zones=pd.read_csv(geometry)
    polys={r.zone_id:unary_union([box(p['xmin'],p['zmin'],p['xmax'],p['zmax']) for p in json.loads(r.collider_parts_xz)]) for r in zones.itertuples()}
    lo,hi=summary['main_hall_bounds_xz'];hall=box(*lo,*hi)
    result={}
    for pid,_ in PERIODS:
        result[pid]={}
        for floor,df in tables[pid].groupby('floor',sort=False):
            if floor not in FLOORS:raise ValueError('Unexpected floor')
            ids=set(df.zone_id)-{'Null'}
            shape=unary_union([polys[z] for z in ids])
            if pid=='P3':shape=polys['HallInsideGate_Main'].intersection(hall)
            if pid=='P4':shape=hall.difference(polys['HallInsideGate_Main'])
            result[pid][floor]=shape
    return result


def extract(args,tables,domains,classifier,outputs):
    trials=pd.DataFrame(load_timeline(ROOT/'TIMELINE/subject_zone_timeline.csv'))
    grouped={pid:{int(t):d for t,d in frame.groupby('trial_id',sort=False)} for pid,frame in tables.items()}
    cells={m:[] for m in METRICS};audits=[];handles={}
    for m,out in outputs.items():
        (out/'points').mkdir(parents=True,exist_ok=True)
        for pid,_ in PERIODS:
            h=gzip.open(out/'points'/f'{pid}.csv.gz','wt',encoding='utf-8-sig',newline='')
            handles[m,pid]=(h,False)
    try:
        for tid,r in enumerate(tqdm(trials.itertuples(index=False),total=len(trials),desc='Compute four frame metrics'),1):
            path=Path(r.subject_file)
            group=r.group
            if not group:continue
            with path.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.reader(f))
            loaded={m:spec[4](path) for m,spec in METRICS.items() if spec[4]}
            for pid,_ in PERIODS:
                tr=grouped[pid].get(tid)
                if tr is None:continue
                for metric,spec in METRICS.items():
                    meta=dict(trial_id=tid,scenario=r.scenario,subject=r.subject,group=group,src_file=r.src_file,period=pid,metric=metric)
                    if metric=='Speed':
                        source=[raw[int(i)] for i in tr.frame]
                        velocity=pd.DataFrame([[s[4],s[6]] for s in source]).apply(pd.to_numeric,errors='coerce').to_numpy(float)
                        value=np.linalg.norm(velocity,axis=1)
                        pts=tr[['x','y','z']].to_numpy(float);stamps=tr.timestamp.to_numpy();frames=tr.frame.to_numpy()
                        weight=np.ones(len(tr));keep=np.isfinite(value);sync=np.full(len(tr),'exact');previous=np.full(len(tr),'')
                    else:
                        d=loaded[metric]
                        if d['missing_eye']:
                            audits.append(dict(meta,status='missing_eye_file',n_selected=0));continue
                        f=d['frame'];value=f[spec[3]].to_numpy(float);pts=f[['subject_x','subject_y','subject_z']].to_numpy(float)
                        prev=d.get('previous_ns');keep=support_mask(d['ns'],tr,prev)
                        keep &= np.isfinite(value)&np.isfinite(pts).all(axis=1)
                        weight=f.interval_s.to_numpy(float) if prev is not None else np.ones(len(f))
                        keep &= np.isfinite(weight)&(weight>0)
                        stamps=f.timestamp.to_numpy();frames=f.eye_frame.to_numpy();sync=f.position_sync.to_numpy()
                        previous=f.previous_timestamp.to_numpy() if prev is not None else np.full(len(f),'')
                    before=int(keep.sum())
                    fl=classifier.floor_of(pts[:,1]);zi=classifier.assign(pts)
                    domain_mask=np.zeros(len(pts),bool)
                    for floor,shape in domains[pid].items():
                        domain_mask |= (fl==FLOORS[floor])&shapely.intersects_xy(shape,pts[:,0],pts[:,2])
                    if pid=='P3':domain_mask &= zi==classifier.global_index['HallInsideGate_Main']
                    if pid=='P4':domain_mask &= zi!=classifier.global_index['HallInsideGate_Main']
                    keep &= domain_mask
                    df=pd.DataFrame(dict(timestamp=stamps[keep],source_frame=frames[keep],previous_timestamp=previous[keep],
                                         x=pts[keep,0],y=pts[keep,1],z=pts[keep,2],value=value[keep],weight=weight[keep],
                                         floor=np.where(fl[keep]==0,'Platform','Hall'),position_sync=sync[keep],
                                         zone_id=[classifier.zone_list[i].zone_id if i>=0 else 'Null' for i in zi[keep]]))
                    for k,v in meta.items():df[k]=v
                    h,written=handles[metric,pid];df.to_csv(h,index=False,header=not written);handles[metric,pid]=(h,True)
                    audits.append(dict(meta,status='included' if len(df) else 'no_valid_metric',n_before_domain=before,n_selected=len(df),
                                       n_outside_domain=before-len(df),n_interpolated=int((df.position_sync=='interpolated').sum())))
                    if not len(df):continue
                    df['cell_x']=np.floor(df.x/args.cell).astype(int);df['cell_z']=np.floor(df.z/args.cell).astype(int)
                    df['wv']=df.value*df.weight;df['wx']=df.x*df.weight;df['wz']=df.z*df.weight
                    c=df.groupby(['floor','cell_x','cell_z'],as_index=False).agg(weight=('weight','sum'),wv=('wv','sum'),wx=('wx','sum'),wz=('wz','sum'),n=('value','size'))
                    c['Value']=c.wv/c.weight;c['PosX']=c.wx/c.weight;c['PosZ']=c.wz/c.weight
                    for k,v in meta.items():c[k]=v
                    cells[metric].append(c.drop(columns=['wv','wx','wz']))
    finally:
        for h,_ in handles.values():h.close()
    for metric,out in outputs.items():
        pd.DataFrame([r for r in audits if r['metric']==metric]).to_csv(out/'sample_audit.csv',index=False,encoding='utf-8-sig')
        cells[metric]=pd.concat(cells[metric],ignore_index=True)
        cells[metric].to_csv(out/'trial_cells.csv.gz',index=False,compression='gzip')
    return cells


def balanced(cells):
    # Repeated sessions/conditions first combine within person, then each person
    # contributes one equally weighted estimate to the group cell.
    keys=['floor','cell_x','cell_z']
    p=cells.groupby(keys+['subject'],as_index=False)[['Value','PosX','PosZ']].mean()
    return p.groupby(keys,as_index=False).agg(Value=('Value','mean'),PosX=('PosX','mean'),PosZ=('PosZ','mean'),ParticipantSupport=('subject','nunique'))


def polygon_path(shape):
    polygons=[shape] if shape.geom_type=='Polygon' else list(shape.geoms)
    verts=[];codes=[]
    for polygon in polygons:
        polygon=shapely.geometry.polygon.orient(polygon,sign=1.)
        for ring in [polygon.exterior,*polygon.interiors]:
            xy=np.asarray(ring.coords);verts.extend(xy);codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(xy)-2)+[MplPath.CLOSEPOLY])
    return MplPath(np.asarray(verts),codes)


def participant_surface(cells,shape,metric,args):
    """Gaussian numerator/denominator per person, then equal-person mean.

    Support is measured after kernel weighting, not by an arbitrary minimum
    residence time in each 0.4 m cell. All weights keep their original units.
    """
    x0,z0,x1,z1=shape.bounds
    xs=x0+np.arange(int(np.ceil((x1-x0)/args.grid))+1)*args.grid
    zs=z0+np.arange(int(np.ceil((z1-z0)/args.grid))+1)*args.grid
    x,z=np.meshgrid(xs,zs);flat=np.column_stack([x.ravel(),z.ravel()])
    inside=shapely.intersects_xy(shape,flat[:,0],flat[:,1]);indices=np.flatnonzero(inside)
    targets=flat[inside];tree=cKDTree(targets)
    sums=np.zeros(len(targets));counts=np.zeros(len(targets),dtype=np.int32)
    exposure=np.zeros(len(targets));minimum=args.min_seconds if metric in ('HeadSearch','ScanSpeed') else args.min_samples
    for subject,c in cells.groupby('subject',sort=False):
        xy=c[['PosX','PosZ']].to_numpy(float);sample_tree=cKDTree(xy)
        kernel=tree.sparse_distance_matrix(sample_tree,3*args.bandwidth,output_type='coo_matrix')
        kernel.data=np.exp(-.5*(kernel.data/args.bandwidth)**2);kernel=kernel.tocsr()
        weight=c.weight.to_numpy(float);den=kernel@weight
        num=kernel@(weight*c.Value.to_numpy(float))
        nearest=sample_tree.query(targets,k=1)[0]
        supported=(den>=minimum)&(nearest<=args.bandwidth)
        sums[supported]+=num[supported]/den[supported]
        counts[supported]+=1;exposure[supported]+=den[supported]
    values=np.full(len(flat),np.nan);good=counts>=args.min_participants
    values[indices[good]]=sums[good]/counts[good]
    support=np.zeros(len(flat),dtype=np.int32);support[indices]=counts
    total_exposure=np.zeros(len(flat));total_exposure[indices]=exposure
    return x,z,values.reshape(x.shape),support.reshape(x.shape),total_exposure.reshape(x.shape)


def compute_surfaces(cells,domains,out,metric,args):
    maps={};manifest=[]
    for level,name in tqdm(SELECTIONS,desc='Gaussian kernel surfaces'):
        chosen=selected(cells,level,name)
        for pid,_ in PERIODS:
            p=chosen[chosen.period.eq(pid)]
            for floor,shape in domains[pid].items():
                c=p[p.floor.eq(floor)]
                x,z,v,count,exposure=participant_surface(c,shape,metric,args)
                inside=shapely.intersects_xy(shape,x,z)
                assert not np.isfinite(v[~inside]).any()
                finite=v[np.isfinite(v)]
                assert not len(finite) or (finite.min()>=c.Value.min()-1e-9 and finite.max()<=c.Value.max()+1e-9)
                maps[pid,level,name,floor]=(x,z,v)
                folder=out/pid;folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{level}_{name}_{floor}_surface.npz',x=x,z=z,value=v,
                                    participant_count=count,kernel_support=exposure,domain_mask=inside)
            manifest.append(dict(period=pid,level=level,selection=name,n_subjects=p.subject.nunique(),n_trials=p.trial_id.nunique(),n_points=int(p.n.sum()),png=f'{pid}/{level}_{name}.png'))
    return maps,manifest


def panel(fig,spec,pid,domains,maps,level,name,cmap,norm,compact=False):
    floors=list(domains[pid]);grid=spec.subgridspec(1,len(floors),wspace=.36)
    for j,floor in enumerate(floors):
        ax=fig.add_subplot(grid[0,j]);shape=domains[pid][floor];x,z,v=maps[pid,level,name,floor]
        # Fill the exact domain at the color minimum without changing missing data.
        path=polygon_path(shape)
        path=MplPath(-path.vertices,path.codes)  # Display only; preserve raw geometry and grids.
        ax.add_patch(PathPatch(path,facecolor=cmap(0.0),edgecolor='none',zorder=0))
        artist=ax.pcolormesh(-x,-z,np.ma.masked_invalid(v),shading='nearest',cmap=cmap,norm=norm,rasterized=True)
        artist.set_clip_path(PathPatch(path,transform=ax.transData))
        ax.add_patch(PathPatch(path,facecolor='none',edgecolor='#68717A',lw=.6))
        x0,z0,x1,z1=shape.bounds;pad=max(1.,.035*max(x1-x0,z1-z0))
        ax.set(xlim=(-x1-pad,-x0+pad),ylim=(-z1-pad,-z0+pad),xlabel='-Unity X (m)',ylabel='-Unity Z (m)')
        ax.set_aspect('equal');ax.set_title(f'{pid} | {floor}',fontsize=10 if compact else 13)
        ax.tick_params(labelsize=7 if compact else 9);ax.spines[['top','right']].set_visible(False)
        if not np.isfinite(v).any():ax.text(.5,.5,'No valid samples',ha='center',transform=ax.transAxes,fontsize=8)


def render_heatmap_only(domains,maps,out,cmap,norm,dpi):
    """One transparent AllTrials image per P, with a shared pixel/metre scale.

    Preserve each P's existing floor order (including both floors in P2).
    Every floor uses the same world viewport and pixel size; a single floor
    is centred, never enlarged to fill the two-floor canvas. No tight crop.
    """
    bounds=np.asarray([shape.bounds for floors in domains.values() for shape in floors.values()])
    x0,z0=bounds[:,:2].min(axis=0);x1,z1=bounds[:,2:].max(axis=0)
    width,height=int(round(9*dpi)),int(round(8*dpi))
    max_floors=max(len(floors) for floors in domains.values())
    gap=int(round(.2*dpi))
    scale=min((width-gap*(max_floors-1))/(max_floors*(x1-x0)),height/(z1-z0))
    panel_width=(x1-x0)*scale;panel_height=(z1-z0)*scale
    rows=[]
    for pid,_ in tqdm(PERIODS,desc='Render AllTrials heatmap-only'):
        floors=list(domains[pid]);occupied=len(floors)*panel_width+(len(floors)-1)*gap
        left=(width-occupied)/2;bottom=(height-panel_height)/2
        fig=plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi,frameon=False)
        panels=[]
        try:
            for j,floor in enumerate(floors):
                rect=[(left+j*(panel_width+gap))/width,bottom/height,panel_width/width,panel_height/height]
                ax=fig.add_axes(rect,frameon=False)
                shape=domains[pid][floor];x,z,v=maps[pid,'all','AllTrials',floor]
                path=polygon_path(shape);path=MplPath(-path.vertices,path.codes)
                ax.add_patch(PathPatch(path,facecolor=cmap(0.0),edgecolor='none',zorder=0))
                artist=ax.pcolormesh(-x,-z,np.ma.masked_invalid(v),shading='nearest',cmap=cmap,norm=norm,rasterized=True)
                artist.set_clip_path(PathPatch(path,transform=ax.transData))
                ax.set(xlim=(-x1,-x0),ylim=(-z1,-z0));ax.set_aspect('equal');ax.set_axis_off()
                panels.append(dict(floor=floor,axes_fraction=rect))
            filename=f'{pid}/all_AllTrials_heatmap_only.png'
            (out/pid).mkdir(parents=True,exist_ok=True)
            # rc_context prevents a user's savefig.bbox='tight' from changing size.
            with plt.rc_context({'savefig.bbox':None,'savefig.pad_inches':0}):
                fig.savefig(out/filename,dpi=dpi,transparent=True,bbox_inches=None,pad_inches=0)
            rows.append(dict(period=pid,png=filename,panels=panels))
        finally:
            plt.close(fig)
    metadata=dict(images=rows,pixel_size=[width,height],dpi=dpi,
                  pixels_per_metre=float(scale),metres_per_pixel=float(1/scale),
                  display_xlim=[float(-x1),float(-x0)],display_ylim=[float(-z1),float(-z0)],
                  color_limits=[float(norm.vmin),float(norm.vmax)],background='transparent',
                  layout='Existing floor order; common viewport and scale; single floor centred',
                  missing_data_display='Minimum color inside domain; outside and holes transparent')
    (out/'heatmap_only_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def load_heatmap_only_cache(out,domains,trials,trajectory):
    """Reuse the exact saved surfaces and their original color calibration."""
    old=json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
    validate_metadata(old,trials)
    maps={}
    for pid,_ in PERIODS:
        digest=hashlib.sha256((trajectory/pid/'trajectory_points.csv.gz').read_bytes()).hexdigest()
        if old['trajectory_hashes'][pid]!=digest:
            raise ValueError('Trajectory changed; regenerate metric surfaces first')
        for floor,shape in domains[pid].items():
            with np.load(out/pid/f'all_AllTrials_{floor}_surface.npz',allow_pickle=False) as data:
                x,z,v=data['x'],data['z'],data['value']
                if not (x.shape==z.shape==v.shape):raise ValueError('Cached surface shape mismatch')
                if not np.array_equal(data['domain_mask'],shapely.intersects_xy(shape,x,z)):
                    raise ValueError('Cached surface domain changed; regenerate surfaces first')
                maps[pid,'all','AllTrials',floor]=(x,z,v)
    return maps,old


def render_colorbar(metric,out,palette,limits,unit,dpi):
    """Export the exact shared metric scale for reuse beside P1-P5 panels."""
    lower,upper=map(float,limits)
    if not np.isfinite([lower,upper]).all() or lower>=upper:
        raise ValueError('Invalid color limits for '+metric)
    label=METRICS[metric][0]
    with plt.rc_context({'font.family':'serif','font.serif':['Times New Roman'],'font.size':11}):
        fig=plt.figure(figsize=(7.2,1.35))
        ax=fig.add_axes([.07,.60,.86,.22])
        sm=plt.cm.ScalarMappable(norm=Normalize(lower,upper),cmap=get_cmap(palette))
        ticks=np.linspace(lower,upper,5)
        bar=fig.colorbar(sm,cax=ax,orientation='horizontal',extend='both',ticks=ticks)
        bar.set_ticklabels([f'{value:.2f}' for value in ticks])
        bar.set_label(f'{label} ({unit})')
        for ext in ('png','svg'):
            fig.savefig(out/f'colorbar.{ext}',dpi=dpi,transparent=True)
        plt.close(fig)
    metadata=dict(metric=metric,label=label,unit=unit,palette=palette,
                  color_limits=[lower,upper],ticks=ticks.tolist(),extend='both',
                  orientation='horizontal',files=['colorbar.png','colorbar.svg'])
    (out/'colorbar_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def render(metric,cells,domains,out,args):
    label,unit,palette,_,_=METRICS[metric];cmap=get_cmap(palette)
    maps,manifest=compute_surfaces(cells,domains,out,metric,args)
    # Calibrate only on the ten disjoint scenario selections, not duplicated
    # modality/all selections. One scale per metric across every P and group.
    calibration=np.concatenate([v[np.isfinite(v)] for (pid,level,name,floor),(_,_,v) in maps.items() if level=='scenario'])
    if not len(calibration):raise ValueError('No supported scenario surfaces for '+metric)
    lower,upper=np.quantile(calibration,[.01,.99])
    if upper<=lower:lower=max(0.,lower-.5);upper+=.5
    norm=Normalize(lower,upper);sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    render_colorbar(metric,out,palette,[lower,upper],unit,args.dpi)
    pure=render_heatmap_only(domains,maps,out,cmap,norm,args.dpi)
    for row in manifest:
        if row['level']=='all':row['heatmap_only_png']=f'{row["period"]}/all_AllTrials_heatmap_only.png'
    for row in tqdm(manifest,desc=f'Render {metric} 80 P maps'):
        pid,level,name=row['period'],row['level'],row['selection']
        fig=plt.figure(figsize=(9,8));spec=fig.add_gridspec(1,1,left=.1,right=.80,bottom=.1,top=.88)
        panel(fig,spec[0,0],pid,domains,maps,level,name,cmap,norm)
        cax=fig.add_axes([.85,.23,.022,.52]);fig.colorbar(sm,cax=cax,label=f'{label} ({unit})',extend='both')
        fig.suptitle(f'{label} | {name}\n{row["n_subjects"]} participants, {row["n_trials"]} trials',fontsize=13)
        fig.savefig(out/row['png'],dpi=args.dpi);plt.close(fig)
    (out/'MERGE').mkdir(exist_ok=True)
    for level,name in tqdm(SELECTIONS,desc=f'Merge {metric}'):
        fig=plt.figure(figsize=(18,12));grid=fig.add_gridspec(2,3,left=.05,right=.96,bottom=.07,top=.92,hspace=.28,wspace=.28)
        for k,(pid,_) in enumerate(PERIODS):panel(fig,grid[k//3,k%3],pid,domains,maps,level,name,cmap,norm,True)
        ax=fig.add_subplot(grid[1,2]);ax.axis('off');ax.text(.05,.95,f'{label}\n{name}\n\nEqual-weight participant surfaces\nGaussian bandwidth: {args.bandwidth:g} m | grid: {args.grid:g} m\nAt least {args.min_participants} supported participants\nShared color scale: scenario P1-P99\nNo data / low support: minimum color\nBlank: outside domain / holes',va='top',transform=ax.transAxes,fontsize=11,linespacing=1.6)
        cax=ax.inset_axes([.08,.13,.75,.055]);fig.colorbar(sm,cax=cax,orientation='horizontal',label=f'{label} ({unit})',extend='both')
        fig.suptitle(f'P1-P5 spatial metric heatmaps | {label} | {name}',fontsize=16)
        fig.savefig(out/'MERGE'/f'{level}_{name}.png',dpi=args.dpi)
        if level=='all':fig.savefig(out/'MERGE.png',dpi=args.dpi)
        plt.close(fig)
    pd.DataFrame(manifest).to_csv(out/'figure_manifest.csv',index=False,encoding='utf-8-sig')
    all_values=np.concatenate([v[np.isfinite(v)] for _,_,v in maps.values()])
    return dict(metric=metric,unit=unit,palette=palette,color_limits=[lower,upper],
                display_coordinates='(-Unity X, -Unity Z); metres; raw data unchanged',
                color_method='1st-99th percentile of supported scenario surfaces across P1-P5; values not clipped',
                surface_range=[float(all_values.min()),float(all_values.max())],
                color_saturation_fraction=float(((all_values<lower)|(all_values>upper)).mean()),
                min_participants=args.min_participants,min_kernel_samples=args.min_samples,min_kernel_seconds=args.min_seconds,
                max_nearest_distance_m=args.bandwidth,figures=80,merges=16,n_samples=int(cells.n.sum()),
                heatmap_only_figures=len(pure['images']),heatmap_only_manifest='heatmap_only_manifest.json')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trajectory',type=Path,default=ROOT/'TIMELINE/Traj_By_P')
    ap.add_argument('--output',type=Path,default=ROOT/'TIMELINE')
    ap.add_argument('--cell',type=float,default=.4);ap.add_argument('--dpi',type=int,default=300)
    ap.add_argument('--grid',type=float,default=.4);ap.add_argument('--bandwidth',type=float,default=1.)
    ap.add_argument('--min-participants',type=int,default=3)
    ap.add_argument('--min-samples',type=float,default=5.)
    ap.add_argument('--min-seconds',type=float,default=.1)
    ap.add_argument('--force',action='store_true')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--colorbar-only',action='store_true',help='Export standalone PNG/SVG colorbars using existing run_summary palettes, limits and units')
    mode.add_argument('--render-only',action='store_true')
    mode.add_argument('--heatmap-only',action='store_true',help='Export only five undecorated AllTrials PNGs per metric from existing surface NPZs and color limits')
    args=ap.parse_args()
    if args.dpi<=0:ap.error('--dpi must be positive')
    if min(args.cell,args.grid,args.bandwidth,args.min_participants,args.min_samples,args.min_seconds)<=0:ap.error('All spatial/support parameters must be positive')
    outputs={m:args.output/f'0_Heatmap_By_P_{m}' for m in METRICS}
    if args.colorbar_only:
        trials=load_timeline(ROOT/'TIMELINE/subject_zone_timeline.csv')
        cached={}
        for metric,out in outputs.items():
            old=json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
            validate_metadata(old,trials)
            if old['metric']!=metric or old['unit']!=METRICS[metric][1]:
                raise ValueError('Cached metric/unit mismatch: '+metric)
            limits=np.asarray(old['color_limits'],dtype=float)
            if limits.shape!=(2,) or not np.isfinite(limits).all() or limits[0]>=limits[1]:
                raise ValueError('Invalid cached color limits: '+metric)
            get_cmap(old['palette'])
            if any((out/name).exists() for name in ('colorbar.png','colorbar.svg','colorbar_manifest.json')) and not args.force:
                ap.error('Colorbar output exists; use --force')
            cached[metric]=old
        for metric,out in outputs.items():
            old=cached[metric]
            result=render_colorbar(metric,out,old['palette'],old['color_limits'],old['unit'],args.dpi)
            print(json.dumps(result),flush=True)
        return
    for out in outputs.values():
        if args.heatmap_only:
            targets=[out/pid/'all_AllTrials_heatmap_only.png' for pid,_ in PERIODS]+[out/'heatmap_only_manifest.json']
            if any(path.exists() for path in targets) and not args.force:ap.error('Heatmap-only output exists; use --force')
        elif out.exists() and any(out.iterdir()) and not args.force:ap.error('Nonempty output; use --force')
    trials=load_timeline(ROOT/'TIMELINE/subject_zone_timeline.csv')
    tables,summary=load_support(args.trajectory)
    geom=ROOT/'Results_Guidance_Spatial_Causal_Analysis/zone_annotation/semantic_zones.csv'
    if hashlib.sha256(geom.read_bytes()).hexdigest()!=summary['geometry_sha256']:raise ValueError('Trajectory geometry has changed')
    domains=footprints(tables,summary,geom)
    if args.heatmap_only:
        # Validate all four caches before writing any new images. Numerical CLI
        # defaults are deliberately ignored: the cached surfaces are authoritative.
        cached={metric:load_heatmap_only_cache(out,domains,trials,args.trajectory) for metric,out in outputs.items()}
        for metric,out in outputs.items():
            maps,old=cached[metric]
            result=render_heatmap_only(domains,maps,out,get_cmap(old['palette']),Normalize(*old['color_limits']),args.dpi)
            print(f'{metric}: {len(result["images"])} heatmap-only PNGs; {result["pixel_size"]} pixels',flush=True)
        return
    classifier=VectorizedZoneClassifier(UnifiedZoneQuery(geom.with_suffix('.json')))
    for out in outputs.values():out.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        for out in outputs.values():
            old=json.loads((out/'run_summary.json').read_text())
            validate_metadata(old,trials)
            if old['cell_m']!=args.cell:raise ValueError('Cached cell size differs; rerun extraction')
            for pid,_ in PERIODS:
                if old['trajectory_hashes'][pid]!=hashlib.sha256((args.trajectory/pid/'trajectory_points.csv.gz').read_bytes()).hexdigest():
                    raise ValueError('Trajectory changed; rerun extraction')
    cells={m:pd.read_csv(out/'trial_cells.csv.gz') for m,out in outputs.items()} if args.render_only else extract(args,tables,domains,classifier,outputs)
    for table in cells.values():validate_trial_table(table,trials)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10})
    for metric,out in outputs.items():
        result=render(metric,cells[metric],domains,out,args)
        result.update(**cohort_metadata(trials), cell_m=args.cell,grid_m=args.grid,bandwidth_m=args.bandwidth,
                      truncate_sigma=3.,trajectory_rule=summary['trajectory_rule'],
                      trajectory_hashes={pid:hashlib.sha256((args.trajectory/pid/'trajectory_points.csv.gz').read_bytes()).hexdigest() for pid,_ in PERIODS},
                      source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['Analyze_TimelineSpeed.py','Analyze_TimelineGazeDepth.py','Analyze_TimelineHeadSearch.py','Analyze_TimelineScanSpeed.py','Visualize_Spatial_Metric_Surfaces.py','a5_colormaps.py']})
        (out/'run_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        (out/'METHODS.md').write_text('''# P1-P5 指标空间热力图
试次唯一来源为 common.load_trials() 的45人/270试次；时间线、Traj_By_P及复用格子均校验身份和队列摘要。以最新Traj_By_P导出的逐帧轨迹和连续段作为时间支持，不修改原timeline。仅绘图时将X和Z各取一次负，横轴为-Unity X、纵轴为-Unity Z；热力网格、区域底色、边界、裁剪路径和坐标范围同步变换。CSV/NPZ保留原始Unity坐标及指标值。
P1仅Platform；P3仅HallInsideGate_Main首次访问，离开帧和重入点不计；P4保留当前轨迹规则且不纳入gate内部；P2/P5沿用轨迹时间支持。未知空间不绘制。
运动速度由Subject原始E/G列模长计算；另三指标直接调用相应Analyze_Timeline*的load_trial，沿用其帧级定义、同步及有效性规则，不使用时间分箱曲线。
眼动时间戳按参考方法映射受试者站位，非注视命中位置。只接受保留轨迹连续段内部的时间；角速度区间两端必须在同一保留段中，不跨P/离区边界。原始时间戳不改写。
基础格子默认0.4m，保留试次均值及其分母：速度/深度为有效样本数，头部/眼部角速度为有效区间时长。每名受试者内对全部保留格子的累计量和分母分别施加高斯核，二者相除得到个人局部指标；同一选择集内该人的不同条件/重复会话按有效样本或时长合并。最后对有支持的参与者局部估计等权平均。
沿用参考脚本Nadaraya-Watson高斯核回归形式，默认带宽1m、截断3sigma，输出网格0.4m。个人在每节点的加权有效支持至少5个样本（速度/深度）或0.1秒（角速度），且最近样本格子质心不超过1个带宽；每节点至少3名有支持参与者，否则数值保留为缺失，绘图以色标最小值颜色填充。参数可通过CLI配置，实际值见run_summary.json。这些为展示支持规则，不是统计显著性标准。保留原单位，不做概率密度归一化，不裁剪指标值。
各P和楼层分别平滑。覆盖范围严格按面域遮罩，取消参考脚本0.1m外扩，栅格绘制再用精确多边形裁剪，孔洞也保留；面域内无样本支持的位置以色标最小值颜色填充，仅作显示；网格缺失值保持不变，面域外与孔洞仍为空白。平滑是欧氏距离核，不表示墙体感知传播。
同一指标全部P和16组共用色标：依据10个场景的全部P有效热力值的1%与99%分位数校准，不重复计入模态/全量图。超范围仅颜色饱和，色条两端三角标示；npz保存完整未裁剪数值。四个指标分别使用a5_colormaps的99/162/100/87号。
points/P*.csv.gz为全部有效绘图输入及原始帧/时间戳、站位、数值、权重、分区；trial_cells.csv.gz为试次格子统计；sample_audit.csv为样本筛选审计；P*/*_surface.npz为热力网格及participant_count/kernel_support/domain_mask；figure_manifest.csv索引80张单P图，MERGE/保存16张合并图。
每个指标根目录另存colorbar.png/svg：透明背景横向独立色条，标注指标名称、数值和单位，沿用该指标全部P共用的色图与上下限；两端三角表示超范围颜色饱和。--colorbar-only读取现有run_summary单独生成色条，colorbar_manifest.json记录精确上下限及刻度。
各P另存all_AllTrials_heatmap_only.png：只绘制热力面，无标题、坐标轴、色条、轮廓线或文字；P2保留原楼层顺序并合在一张图。全部指标/P共用9×8英寸画布（300dpi时2700×2400像素）、全局XZ范围和米/像素比例；单楼层居中且不放大，面域外透明；同一指标沿用原色标。heatmap_only_manifest.json记录像素尺寸、空间范围、比例尺和文件列表。--heatmap-only直接复用已有NPZ及run_summary色标补画纯图，不重算热力面或重绘原图。
''',encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
