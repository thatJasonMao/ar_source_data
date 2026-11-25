"""Four global spatial-rule PNGs, one per guidance mode; no trajectory/PDF input."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from Spatial_IRL_Rule import SpatialIRLRule, SPATIAL, GUIDANCE

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'0_RL_SOURCE'))
from spatial_isovist_core import PlanarIsovist, _outline
from Analyze_NextDecisionDistance import NextDecisionDistance
CELL=.4
SMOKE={'HighSmoke':7.5,'LowSmoke':12.5}


def build_spatial_grid(spatial, decision):
    """Geometry defines every query point, independently of observed positions."""
    rows=[]
    for floor,domain in spatial.domains.items():
        xmin,zmin,xmax,zmax=domain.bounds
        xe=np.arange(int(np.floor(xmin/CELL)),int(np.ceil(xmax/CELL))+1)*CELL
        ze=np.arange(int(np.floor(zmin/CELL)),int(np.ceil(zmax/CELL))+1)*CELL
        xx,zz=np.meshgrid((xe[:-1]+xe[1:])/2,(ze[:-1]+ze[1:])/2)
        valid=shapely.intersects_xy(domain,xx,zz)&~shapely.intersects_xy(spatial.exclusions[floor],xx,zz)
        for iz,ix in tqdm(np.argwhere(valid),desc='Global geometry: '+floor):
            x,z=float(xx[iz,ix]),float(zz[iz,ix])
            d=decision.query(x,None,z,floor=floor)
            if d.status!='ok' or d.stage=='arrived_exit':continue
            for smoke,radius in SMOKE.items():
                s=spatial.query(x,None,z,radius,floor=floor)
                if s.status!='ok':raise ValueError('Invalid isovist at a valid global grid point')
                row={name:getattr(s,name) for name in SPATIAL[:4]}
                row.update(x=x,z=z,floor=floor,smoke=smoke,stage=d.stage,
                           next_decision_distance=d.next_decision_distance)
                rows.append(row)
    if not rows:raise ValueError('No valid global spatial cells')
    return pd.DataFrame(rows)


def draw_field(frame,spatial,guidance,norm,target,dpi):
    # One PNG = one rule; smoke is a controlled input, not an undocumented average.
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
                values[iz,ix]=part.relative_information_utility
                ax.pcolormesh(-xe[::-1],-ze[::-1],np.ma.masked_invalid(values[::-1,::-1]),
                              cmap='plasma',norm=norm,shading='flat',rasterized=True)
                _outline(ax,domain,color='red',linewidth=.65)
                _outline(ax,spatial.exclusions[floor],color='gray',linewidth=.65,linestyle='--')
                ax.set_aspect('equal');ax.grid(False)
                ax.set_title(f'{smoke} | {floor}')
                ax.set_xlabel('Display X = -Unity X (m)');ax.set_ylabel('Display Z = -Unity Z (m)')
                ax.ticklabel_format(useOffset=False,style='plain')
        cax=fig.add_axes([.87,.16,.025,.65])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='plasma'),cax=cax,
                     label='Relative information-acquisition utility (higher = greater modeled incentive)')
        fig.suptitle(f'{guidance} | Global spatial guidance-weakness index\nIRL spatial reward contrast; candidate indicator, not failure probability',fontsize=13)
        fig.text(.08,.025,'0.4 m geometry queries; no trajectory projection or smoothing. Unobserved conditions are extrapolations.',fontsize=9)
        fig.savefig(target,dpi=dpi,bbox_inches='tight')
    finally:plt.close(fig)


def plot_run(run,*,dpi=300,force=False):
    run=Path(run).resolve()
    if dpi<50:raise ValueError('dpi must be at least 50')
    rule=SpatialIRLRule(run)
    out=run/'Results_Global_Spatial_Field'
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError('Global output exists; use --force to regenerate')
    spatial=PlanarIsovist();decision=NextDecisionDistance(spatial.geometry_path)
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    geometry_sha=hashlib.sha256(spatial.geometry_path.read_bytes()).hexdigest()
    if config.get('geometry_sha256') and config['geometry_sha256']!=geometry_sha:
        raise ValueError('Geometry differs from training geometry')
    # No opening of dataset, predictions, trial roster, or trajectory-field CSVs.
    frame=build_spatial_grid(spatial,decision)
    frames=[]
    for guidance in GUIDANCE:
        part=frame.copy();part['guidance']=guidance
        part['relative_information_utility']=rule.query_features(frame,guidance)
        frames.append(part)
    field=pd.concat(frames,ignore_index=True)
    vmin=float(field.relative_information_utility.min());vmax=float(field.relative_information_utility.max())
    if vmax<=vmin:vmin-=.5;vmax+=.5
    norm=Normalize(vmin,vmax)
    out.mkdir(parents=True,exist_ok=True)
    field.to_csv(out/'global_spatial_rule_field.csv.gz',index=False,compression='gzip')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10})
    files=[]
    for guidance in tqdm(GUIDANCE,desc='Render four global fields'):
        name=guidance+'_spatial_weakness.png'
        draw_field(field.loc[field.guidance.eq(guidance)],spatial,guidance,norm,out/name,dpi)
        files.append(name)
    manifest=dict(status='complete',field='relative_information_utility',reward_scope=config['reward_scope'],
        interpretation='Spatial information-acquisition incentive as a candidate guidance-weakness index; not calibrated failure probability',
        input='Full geometry grid, four isovist metrics, NextDecisionDistance, spatial stage, floor, smoke, guidance',
        cell_m=CELL,smoke_visibility_m=SMOKE,dpi=dpi,files=files,png_count=4,pdf_count=0,
        shared_color_scale=[vmin,vmax],geometry_sha256=geometry_sha,
        rule_sha256=hashlib.sha256((run/'reward.pt').read_bytes()).hexdigest(),
        encoder_sha256=hashlib.sha256((run/'encoder.json').read_bytes()).hexdigest(),
        rule_code_sha256=hashlib.sha256((HERE/'Spatial_IRL_Rule.py').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trajectory_input=False,smoothing=False,grid_rows=len(frame),field_rows=len(field),
        exclusions=['stairs','outside geometry','unassigned decision zones','task completed at exit'],
        limitation='Rule predictions at unobserved feature combinations are unvalidated extrapolations')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    config['plot_status']='complete';config.pop('plot_error',None)
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    m=plot_run(args.run,dpi=args.dpi,force=args.force)
    print(json.dumps({key:m[key] for key in ['status','png_count','pdf_count','grid_rows']},indent=2))


if __name__=='__main__':main()
