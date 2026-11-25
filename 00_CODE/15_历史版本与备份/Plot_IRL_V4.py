"""V4: unchanged 0.4 m spatial rule computation; Gaussian smoothing only in drawing."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from Spatial_IRL_Rule import SpatialIRLRule, GUIDANCE
from Plot_IRL_Weakness import build_spatial_grid, CELL, SMOKE, PlanarIsovist, NextDecisionDistance, _outline
from gaussian_field_v3 import smooth_display, continuous_display


def draw_field(frame,spatial,guidance,norm,target,dpi,sigma_m,upsample,model_label="V4 full-frame IRL"):
    fig,axes=plt.subplots(2,2,figsize=(12,16),gridspec_kw={'width_ratios':[3,1]})
    fig.subplots_adjust(left=.08,right=.82,bottom=.07,top=.9,hspace=.24,wspace=.3)
    try:
        for row,smoke in enumerate(SMOKE):
            for col,floor in enumerate(['Hall','Platform']):
                ax=axes[row,col];domain=spatial.domains[floor]
                xmin,zmin,xmax,zmax=domain.bounds
                x0,z0=np.floor(np.array([xmin,zmin])/CELL).astype(int)
                x1,z1=np.ceil(np.array([xmax,zmax])/CELL).astype(int)
                xe=np.arange(x0,x1+1)*CELL;ze=np.arange(z0,z1+1)*CELL
                raw=np.full((z1-z0,x1-x0),np.nan)
                part=frame.loc[frame.smoke.eq(smoke)&frame.floor.eq(floor)]
                ix=np.rint(part.x.to_numpy()/CELL-.5).astype(int)-x0
                iz=np.rint(part.z.to_numpy()/CELL-.5).astype(int)-z0
                raw[iz,ix]=part.relative_information_utility
                walkable=domain.difference(spatial.exclusions[floor])
                # This is the ONLY place smoothing is applied. frame/raw remain unchanged.
                display=smooth_display(raw,(xe[:-1]+xe[1:])/2,(ze[:-1]+ze[1:])/2,
                                       walkable,sigma_m,CELL)
                display,fx,fz=continuous_display(display,xe,ze,walkable,upsample)
                ax.pcolormesh(-fx[::-1],-fz[::-1],np.ma.masked_invalid(display[::-1,::-1]),
                              cmap='plasma',norm=norm,shading='flat',rasterized=True)
                _outline(ax,domain,color='red',linewidth=.65)
                _outline(ax,spatial.exclusions[floor],color='gray',linewidth=.65,linestyle='--')
                ax.set_aspect('equal');ax.grid(False);ax.set_title(f'{smoke} | {floor}')
                ax.set_xlabel('Display X = -Unity X (m)');ax.set_ylabel('Display Z = -Unity Z (m)')
                ax.ticklabel_format(useOffset=False,style='plain')
        cax=fig.add_axes([.87,.16,.025,.65])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='plasma'),cax=cax,
                     label='Relative guidance-information demand (IRL reward contrast)')
        fig.suptitle(f'{guidance} | Global relative guidance-information demand\n{model_label}; Gaussian smoothing for display only',fontsize=13)
        fig.text(.08,.025,f'0.4 m raw rule grid; display sigma = {sigma_m:g} m. CSV and point queries are unsmoothed.',fontsize=9)
        fig.savefig(target,dpi=dpi,bbox_inches='tight')
    finally:plt.close(fig)


def plot_run(run,*,dpi=300,force=False,sigma_m=None,upsample=None):
    run=Path(run).resolve()
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    if config.get('method')!='budgeted_full_frame_spatial_IRL_v4': raise ValueError('Requires a completed V4 run')
    sigma_m=config.get('plot_sigma_m',.6) if sigma_m is None else sigma_m
    upsample=config.get('plot_upsample',4) if upsample is None else upsample
    if dpi<50 or not np.isfinite(sigma_m) or sigma_m<=0 or not 1<=upsample<=16: raise ValueError('Invalid display options')
    rule=SpatialIRLRule(run)
    out=run/'Results_Global_Spatial_Field_V4'
    if out.exists() and any(out.iterdir()) and not force: raise FileExistsError('Output exists; use --force')
    spatial=PlanarIsovist();decision=NextDecisionDistance(spatial.geometry_path)
    geometry_sha=hashlib.sha256(spatial.geometry_path.read_bytes()).hexdigest()
    if config.get('geometry_sha256')!=geometry_sha: raise ValueError('Geometry changed since training')
    frame=build_spatial_grid(spatial,decision)
    fields=[]
    for guidance in GUIDANCE:
        part=frame.copy();part['guidance']=guidance
        part['relative_information_utility']=rule.query_features(frame,guidance)
        fields.append(part)
    field=pd.concat(fields,ignore_index=True)
    vmin=float(field.relative_information_utility.min());vmax=float(field.relative_information_utility.max())
    if vmax<=vmin: vmin-=.5;vmax+=.5
    norm=Normalize(vmin,vmax)
    out.mkdir(parents=True,exist_ok=True)
    # Numerical spatial reward contrast is written before display smoothing.
    csv=out/'global_spatial_rule_field.csv.gz'
    field.to_csv(csv,index=False,compression='gzip')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10})
    files=[]
    model_label="V3 optimized spatial IRL" if config.get("optimization") else "V4 full-frame IRL"
    for guidance in tqdm(GUIDANCE,desc="Render four spatial fields"):
        name=guidance+'_spatial_demand.png'
        draw_field(field.loc[field.guidance.eq(guidance)],spatial,guidance,norm,out/name,dpi,sigma_m,upsample,model_label)
        files.append(name)
    manifest=dict(status='complete',method=config['method'],optimization=config.get('optimization'),reward_basis={g:json.loads((run/relative/'config.json').read_text(encoding='utf-8')).get('basis','linear') for g,relative in config['model_bank'].items()} if config.get('model_bank') else config.get('basis','linear'),field='relative_information_utility',
        interpretation='Behavior-inferred relative guidance-information demand, not performance improvement or probability',
        cell_m=CELL,grid_rows=len(frame),field_rows=len(field),files=files,png_count=4,pdf_count=0,
        raw_csv_smoothed=False,query_smoothed=False,training_smoothed=False,trajectory_input=False,
        display_smoothing=dict(kernel='normalized Gaussian on signed rule values',sigma_m=sigma_m,
            support_sigma=3,wall_policy='center-to-center visibility within walkable geometry',
            upsample=upsample,display_cell_m=CELL/upsample,interpolation='bilinear display only'),
        shared_color_scale=[vmin,vmax],color_scale_source='raw unsmoothed global values',
        geometry_sha256=geometry_sha,raw_csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        reward_sha256=({g:hashlib.sha256((run/rel/'reward.pt').read_bytes()).hexdigest() for g,rel in config['model_bank'].items()} if config.get('model_bank') else hashlib.sha256((run/'reward.pt').read_bytes()).hexdigest()),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exclusions=['stairs','outside geometry','unassigned decision zones','task completed at exit'],
        limitation='Spatial-rule extrapolation remains unvalidated in unobserved feature combinations')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    config['plot_status']='complete';config.pop('plot_error',None)
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--sigma-m',type=float,default=None);p.add_argument('--upsample',type=int,default=None)
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    print(json.dumps(plot_run(args.run,dpi=args.dpi,force=args.force,sigma_m=args.sigma_m,upsample=args.upsample),indent=2))


if __name__=='__main__': main()
