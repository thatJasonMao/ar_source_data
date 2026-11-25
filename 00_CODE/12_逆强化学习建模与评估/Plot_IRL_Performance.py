"""Four global performance-benefit PNGs after performance IRL; no PDF output."""
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

from irl_performance import PerformanceIRLQuery, GUIDANCE, METHOD
from Plot_IRL_Weakness import build_spatial_grid, PlanarIsovist, NextDecisionDistance, CELL, SMOKE, _outline


def draw_field(frame,spatial,guidance,norm,target,dpi,window_s):
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
                values=np.full((z1-z0,x1-x0),np.nan)
                part=frame.loc[frame.smoke.eq(smoke)&frame.floor.eq(floor)]
                ix=np.rint(part.x.to_numpy()/CELL-.5).astype(int)-x0
                iz=np.rint(part.z.to_numpy()/CELL-.5).astype(int)-z0
                values[iz,ix]=part.performance_benefit_s
                ax.pcolormesh(-xe[::-1],-ze[::-1],np.ma.masked_invalid(values[::-1,::-1]),
                              cmap='plasma',norm=norm,shading='flat',rasterized=True)
                _outline(ax,domain,color='red',linewidth=.65)
                _outline(ax,spatial.exclusions[floor],color='gray',linewidth=.65,linestyle='--')
                ax.set_aspect('equal');ax.grid(False);ax.set_title(f'{smoke} | {floor}')
                ax.set_xlabel('Display X = -Unity X (m)');ax.set_ylabel('Display Z = -Unity Z (m)')
                ax.ticklabel_format(useOffset=False,style='plain')
        cax=fig.add_axes([.87,.16,.025,.65])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='plasma'),cax=cax,
                     label='Predicted slowdown loss: no look minus look (equivalent full-stop seconds)')
        fig.suptitle(f'{guidance} | Global performance benefit of looking\n{window_s:g} s comparison; positive = lower modeled slowdown loss when looking',fontsize=13)
        fig.text(.08,.025,'0.4 m geometry queries | Shared dynamic references | Model estimates, not randomized causal effects',fontsize=9)
        fig.savefig(target,dpi=dpi,bbox_inches='tight')
    finally:plt.close(fig)


def plot_run(run,*,dpi=300,force=False):
    run=Path(run).resolve()
    if dpi<50:raise ValueError('dpi must be at least 50')
    query=PerformanceIRLQuery(run);config=query.config
    out=run/'Results_Global_Performance_Field'
    if out.exists() and any(out.iterdir()) and not force:raise FileExistsError('Output exists; use --force')
    spatial=PlanarIsovist();decision=NextDecisionDistance(spatial.geometry_path)
    geometry_sha=hashlib.sha256(spatial.geometry_path.read_bytes()).hexdigest()
    if config['geometry_sha256']!=geometry_sha:raise ValueError('Geometry differs from training')
    # This interface computes geometry everywhere, never loads trial coordinates.
    grid=build_spatial_grid(spatial,decision);frames=[]
    for guidance in tqdm(GUIDANCE,desc='Query global performance rule'):
        scores=query.query_features(grid,guidance)
        frame=pd.concat([grid.reset_index(drop=True),scores],axis=1);frame['guidance']=guidance
        frames.append(frame)
    field=pd.concat(frames,ignore_index=True)
    values=field.performance_benefit_s.dropna()
    if not len(values):raise ValueError('No global estimates; inspect saved reference coverage')
    limit=float(np.abs(values).max())
    if limit<1e-9:limit=1.
    norm=Normalize(-limit,limit)
    out.mkdir(parents=True,exist_ok=True)
    field.to_csv(out/'global_performance_field.csv.gz',index=False,compression='gzip')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10})
    files=[];window=config['horizon']*config['step_s']
    for guidance in tqdm(GUIDANCE,desc='Render four performance maps'):
        name=guidance+'_performance_benefit.png'
        draw_field(field.loc[field.guidance.eq(guidance)],spatial,guidance,norm,out/name,dpi,window)
        files.append(name)
    model_files=['reward.pt','encoder.json','mdp.npz','conditional_dynamics.npz','reference_bank.json']
    manifest=dict(status='complete',method=METHOD,field='performance_benefit_s',
        definition='mean_reference(expected_slowdown_cost_no_look - expected_slowdown_cost_look)',
        units='equivalent full-stop seconds',window_s=window,cell_m=CELL,dpi=dpi,
        initial_action='forced 0 versus forced 1 for one step; remaining steps follow the same fitted policy',
        reference='same train-only joint dynamic reference bank per floor/stage across all modes and smoke conditions',
        geometry_sha256=geometry_sha,model_sha256={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in model_files},
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
                       ['Plot_IRL_Performance.py','irl_performance.py','Plot_IRL_Weakness.py']},
        shared_color_scale=[-limit,limit],png_count=4,pdf_count=0,files=files,
        grid_rows=len(grid),field_rows=len(field),no_reference_rows=int(field.performance_benefit_s.isna().sum()),
        mean_reference_supported_fraction=float(field.reference_supported_fraction.mean()),
        trajectory_input=False,smoothing=False,
        exclusions=['stairs','outside geometry','unassigned decision stage','task completed at exit','no trained reference for floor/stage'],
        limitation='Observational model estimates. Low reference-supported fractions indicate extrapolation; no causal identification claimed.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    config['plot_status']='complete';config.pop('plot_error',None)
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--dpi',type=int,default=300);p.add_argument('--force',action='store_true')
    args=p.parse_args();result=plot_run(args.run,dpi=args.dpi,force=args.force)
    print(json.dumps({k:result[k] for k in ['status','png_count','pdf_count','grid_rows']},indent=2))


if __name__=='__main__':main()
