"""Direct XZ Euclidean distance to the nearest next-stage target region."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.ops import unary_union, nearest_points
from spatial_isovist_core import (
    PlanarIsovist, HERE, DEFAULT_GEOMETRY_PATH, SMOKE, STAIRS, _outline,
    HIGH_SMOKE_VISIBILITY_M, LOW_SMOKE_VISIBILITY_M,
)

@dataclass(frozen=True)
class NextDecisionResult:
    status: str
    floor: str | None
    zone_id: str | None = None
    stage: str | None = None
    next_decision_distance: float = float('nan')
    target_zone: str | None = None
    target_x: float | None = None
    target_z: float | None = None
    tied_targets: tuple[str, ...] = ()


class NextDecisionDistance:
    """Only fixed geometry is cached. Every query uses the original Unity XZ.

    Distances are Euclidean distance to target polygon sets, in metres;
    they may cross walls and are not a walkable-path distance or visibility.
    Smoke does not change the value. Y selects a floor with UnifiedZoneQuery.
    """
    def __init__(self, geometry_path=DEFAULT_GEOMETRY_PATH):
        self.spatial = PlanarIsovist(geometry_path)
        self.geometry_path = self.spatial.geometry_path
        self.domains, self.exclusions = self.spatial.domains, self.spatial.exclusions
        self.zone_query = self.spatial.zone_query
        self.zone_shapes = {
            z.zone_id: unary_union([box(p.xmin, p.zmin, p.xmax, p.zmax) for p in z.collider_parts])
            for z in self.zone_query.zones
        }
        self.groups = {
            'to_stair': tuple(sorted(STAIRS)),
            'to_gate': tuple(f'Gate_{i:02}' for i in range(1,5)),
            'to_exit_corridor': tuple('ExitCorridor_'+s for s in ('AB','CD','EF','GH')),
            'to_branch_corridor': tuple('BranchCorridor_'+s for s in 'ABCDEFGH'),
            'to_exit': tuple('Exit_'+s for s in 'ABCDEFGH'),
        }
        for names in self.groups.values():
            for name in names:
                if name not in self.zone_shapes:
                    raise ValueError(f'Missing required target zone: {name}')
        self.zones_by_floor = {
            floor: sorted([z for z in self.zone_query.zones if z.belongs_to_floor(floor)],
                          key=lambda z: (-z.priority,z.area_xz,z.zone_id))
            for floor in self.domains
        }
        self.rules = {}
        for floor,zones in self.zones_by_floor.items():
            for z in zones:
                name=z.zone_id
                if name in STAIRS:
                    continue
                if floor=='Platform':stage='to_stair'
                elif name=='HallInsideGate_Main' or name.startswith('Gate_'):stage='to_gate'
                elif name in ('HallOutsideGate_Main','DecisionZone_ABCD','DecisionZone_EFGH'):stage='to_exit_corridor'
                elif name.startswith('ExitCorridor_') or name in ('DecisionZone_AB','DecisionZone_CD','DecisionZone_EF','DecisionZone_GH'):stage='to_branch_corridor'
                elif name.startswith('BranchCorridor_'):stage='to_exit'
                elif name.startswith('Exit_'):stage='arrived_exit'
                else:raise ValueError(f'Undefined stage for {floor}/{name}')
                self.rules[floor,name]=stage

    def query(self,x,y,z,*,floor=None):
        floor,status=self.spatial.locate(x,y,z,floor=floor)
        if status!='ok':return NextDecisionResult(status,floor)
        point=Point(float(x),float(z))
        zone=next((q for q in self.zones_by_floor[floor] if self.zone_shapes[q.zone_id].covers(point)),None)
        if zone is None:return NextDecisionResult('unassigned_zone',floor)
        stage=self.rules[floor,zone.zone_id]
        if stage=='arrived_exit':
            return NextDecisionResult('ok',floor,zone.zone_id,stage,0.0,zone.zone_id,float(x),float(z),(zone.zone_id,))
        names=self.groups[stage]
        distances=np.array([point.distance(self.zone_shapes[n]) for n in names])
        idx=int(np.argmin(distances))
        distance=float(distances[idx])
        target=names[idx]
        closest=nearest_points(point,self.zone_shapes[target])[1]
        ties=tuple(n for n,d in zip(names,distances) if abs(float(d)-distance)<=1e-9)
        return NextDecisionResult('ok',floor,zone.zone_id,stage,distance,target,float(closest.x),float(closest.y),ties)

    def query_high_smoke(self,x,y,z,*,floor=None):
        return self.query(x,y,z,floor=floor)

    def query_low_smoke(self,x,y,z,*,floor=None):
        return self.query(x,y,z,floor=floor)


@lru_cache(maxsize=1)
def get_engine():
    return NextDecisionDistance()


def query_high_smoke(x,y,z,*,floor=None):
    """Scalar metres; NaN for excluded stairs or invalid/outside positions."""
    return get_engine().query_high_smoke(x,y,z,floor=floor).next_decision_distance


def query_low_smoke(x,y,z,*,floor=None):
    """Identical to high smoke: no visibility truncation."""
    return get_engine().query_low_smoke(x,y,z,floor=floor).next_decision_distance


def render_heatmaps(engine,output,*,cell=0.4,dpi=300,force=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from tqdm import tqdm
    from collections import Counter
    if not np.isfinite(cell) or cell<=0:raise ValueError('cell must be positive and finite')
    output=Path(output)
    files=[output/f'next_decision_distance_{s}.png' for s in SMOKE]
    manifest_path=output/'manifest_next_decision_distance.json'
    if not force and any(p.exists() for p in files+[manifest_path]):
        raise FileExistsError('Output exists; use --force')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); grids={}; counts={}; stages=Counter(); maximum=0.
    for floor,g in engine.domains.items():
        xmin,zmin,xmax,zmax=g.bounds
        xe=xmin+np.arange(int(np.ceil((xmax-xmin)/cell))+1)*cell
        ze=zmin+np.arange(int(np.ceil((zmax-zmin)/cell))+1)*cell
        xx,zz=np.meshgrid((xe[:-1]+xe[1:])/2,(ze[:-1]+ze[1:])/2)
        keep=shapely.intersects_xy(g,xx,zz)&~shapely.intersects_xy(engine.exclusions[floor],xx,zz)
        values=np.full(xx.shape,np.nan)
        for iz,ix in tqdm(np.argwhere(keep),desc=f'{floor}: next target distance'):
            result=engine.query(xx[iz,ix],None,zz[iz,ix],floor=floor)
            if result.status!='ok':raise RuntimeError(str(result))
            values[iz,ix]=result.next_decision_distance
            stages[result.stage]+=1
        counts[floor]=int(keep.sum())
        maximum=max(maximum,float(np.nanmax(values)))
        grids[floor]=(xe,ze,values)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10})
    for smoke in SMOKE:
        fig,axes=plt.subplots(1,2,figsize=(10,12),layout='constrained',gridspec_kw={'width_ratios':[3,1]})
        for ax,floor in zip(axes,('Hall','Platform')):
            xe,ze,values=grids[floor]
            im=ax.pcolormesh(-xe[::-1],-ze[::-1],values[::-1,::-1],cmap='plasma',norm=Normalize(0,maximum),shading='flat',rasterized=True)
            _outline(ax,engine.domains[floor],color='red',linewidth=0.7)
            _outline(ax,engine.exclusions[floor],color='gray',linewidth=0.7,linestyle='--')
            ax.set_aspect('equal');ax.set_title(floor)
            ax.set_xlabel('Display X = -Unity X (m)');ax.set_ylabel('Display Z = -Unity Z (m)')
            ax.ticklabel_format(useOffset=False,style='plain')
        fig.colorbar(im,ax=axes,shrink=0.75,label='Distance to next target region (m)')
        fig.suptitle(f'Next decision distance | {smoke.capitalize()} smoke\nXZ Euclidean distance; independent of smoke visibility')
        fig.savefig(output/f'next_decision_distance_{smoke}.png',dpi=dpi)
        plt.close(fig)
    manifest={
        'geometry_path':str(engine.geometry_path),
        'geometry_sha256':hashlib.sha256(engine.geometry_path.read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metric':'next_decision_distance','distance_type':'XZ Euclidean point-to-polygon distance in metres',
        'smoke_affects_distance':False,'smoke_visibility_m_for_interface_compatibility':SMOKE,
        'floor_split_y':engine.zone_query.split_y,'floor_drift_tolerance_m':engine.zone_query.floor_drift_tolerance,
        'stage_rules':{f'{f}/{z}':s for (f,z),s in engine.rules.items()},'target_groups':engine.groups,
        'floor_samples':counts,'stage_samples':dict(stages),'cell_m':cell,'dpi':dpi,
        'shared_color_scale':[0,maximum],'sampling':'direct cell-center evaluation, no smoothing, never used by query APIs',
        'excluded_query_zones':sorted(STAIRS),'files':[str(p) for p in files],
        'elapsed_seconds':time.perf_counter()-started,
    }
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'files':manifest['files'],'floor_samples':counts,'stage_samples':dict(stages),'elapsed_seconds':manifest['elapsed_seconds']},indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument('--output',type=Path,default=HERE/'Results_Planar_Isovists')
    parser.add_argument('--cell',type=float,default=0.4)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--point',nargs=3,type=float,metavar=('X','Y','Z'))
    parser.add_argument('--floor',choices=['Hall','Platform'])
    args=parser.parse_args();engine=NextDecisionDistance(args.geometry)
    if args.point:
        for smoke in SMOKE:
            r=engine.query(*args.point,floor=args.floor)
            payload={'smoke':smoke,**asdict(r)}
            if not np.isfinite(r.next_decision_distance):payload['next_decision_distance']=None
            print(json.dumps(payload,ensure_ascii=False,allow_nan=False))
    else:render_heatmaps(engine,args.output,cell=args.cell,dpi=args.dpi,force=args.force)

if __name__=='__main__':main()
