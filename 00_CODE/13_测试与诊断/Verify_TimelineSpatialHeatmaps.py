"""Independent artifact checks for the four P-space heatmaps."""
import sys,json,csv
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root))
import Analyze_TimelineSpatialHeatmaps as h
tables,summary=h.load_support(root/'TIMELINE/Traj_By_P')
domains=h.footprints(tables,summary,root/'Results_Guidance_Spatial_Causal_Analysis/zone_annotation/semantic_zones.csv')
from TimelineTrialInput import load_timeline, validate_trial_table, validate_metadata
timeline=pd.DataFrame(load_timeline(root/'TIMELINE/subject_zone_timeline.csv'))
total=0
for metric,spec in h.METRICS.items():
    out=root/'TIMELINE'/f'0_Heatmap_By_P_{metric}'
    config=json.loads((out/'run_summary.json').read_text())
    manifest=pd.read_csv(out/'figure_manifest.csv')
    assert len(manifest)==80 and manifest.groupby('period').size().eq(16).all()
    assert len(list((out/'MERGE').glob('*.png')))==16
    validate_metadata(config,timeline.to_dict('records'))
    cells=pd.read_csv(out/'trial_cells.csv.gz')
    validate_trial_table(cells,timeline.to_dict('records'))
    sampled={}
    for pid,_ in h.PERIODS:
        d=pd.read_csv(out/'points'/f'{pid}.csv.gz',keep_default_na=False)
        assert np.isfinite(d[['x','y','z','value','weight']]).all().all() and (d.weight>0).all()
        if pid=='P3':assert d.zone_id.eq('HallInsideGate_Main').all()
        if pid=='P4':assert not d.zone_id.eq('HallInsideGate_Main').any()
        for floor,g in d.groupby('floor'):
            assert shapely.intersects_xy(domains[pid][floor],g.x,g.z).all()
        supporting={tid:g for tid,g in tables[pid].groupby('trial_id')}
        for tid,g in d.groupby('trial_id'):
            ns=h.ns_of(g.timestamp)
            previous=h.ns_of(g.previous_timestamp) if metric in ['HeadSearch','ScanSpeed'] else None
            assert h.support_mask(ns,supporting[tid],previous).all()
            # Check whole retained streams for representative trials, including re-entry cases.
            if tid in [1,73,90,169,261]:
                r=timeline.iloc[tid-1];path=root/'RAW_DATA'/r.scenario/r.subject/r.src_file
                if tid not in sampled:
                    if metric=='Speed':
                        with path.open(encoding='utf-8-sig',newline='') as f:sampled[tid]=list(csv.reader(f))
                    else:sampled[tid]=spec[4](path)['frame'].set_index('timestamp')
                if metric=='Speed':
                    raw=sampled[tid];src=[raw[int(i)] for i in g.source_frame]
                    value=np.linalg.norm(np.array([[s[4],s[6]] for s in src],float),axis=1)
                    assert g.timestamp.tolist()==[s[0].strip() for s in src]
                    assert np.allclose(g[['x','y','z']],np.array([s[1:4] for s in src],float),rtol=0,atol=0)
                else:
                    source=sampled[tid].loc[g.timestamp]
                    value=source[spec[3]].to_numpy()
                    assert np.allclose(g[['x','y','z']],source[['subject_x','subject_y','subject_z']],rtol=0,atol=1e-12)
                assert np.allclose(g.value,value,rtol=1e-12,atol=1e-12)
        for row in manifest[manifest.period.eq(pid)].itertuples():
            chosen=h.selected(d,row.level,row.selection)
            assert len(chosen)==row.n_points and chosen.subject.nunique()==row.n_subjects and chosen.trial_id.nunique()==row.n_trials
            assert (out/row.png).stat().st_size>1000
            for floor,shape in domains[pid].items():
                a=np.load(out/pid/f'{row.level}_{row.selection}_{floor}_surface.npz')
                inside=shapely.intersects_xy(shape,a['x'],a['z']);v=a['value']
                assert np.isnan(v[~inside]).all()
                assert np.array_equal(np.isfinite(v),inside & (a['participant_count']>=config['min_participants']))
                assert np.allclose(np.diff(a['x'][0]),config['grid_m'])
                # Independent scalar recomputation at a supported node.
                if row.level=='all' and np.isfinite(v).any():
                    finite_indices=np.flatnonzero(np.isfinite(v));index=finite_indices[len(finite_indices)//2]
                    target=np.array([a['x'].ravel()[index],a['z'].ravel()[index]])
                    source_cells=cells[(cells.period==pid)&(cells.floor==floor)]
                    people=[]
                    for person,c in source_cells.groupby('subject'):
                        distances=np.linalg.norm(c[['PosX','PosZ']].to_numpy()-target,axis=1)
                        within=distances<=3*config['bandwidth_m']
                        weights=np.exp(-.5*(distances[within]/config['bandwidth_m'])**2)*c.weight.to_numpy()[within]
                        threshold=config['min_kernel_seconds'] if metric in ['HeadSearch','ScanSpeed'] else config['min_kernel_samples']
                        if weights.sum()>=threshold and distances.min()<=config['max_nearest_distance_m']:
                            people.append(np.average(c.Value.to_numpy()[within],weights=weights))
                    assert len(people)==a['participant_count'].ravel()[index]
                    assert np.isclose(np.mean(people),v.ravel()[index],rtol=1e-12,atol=1e-12)
                finite=v[np.isfinite(v)]
                assert not len(finite) or (finite.min()>=0 and finite.max()<=config['surface_range'][1]+1e-9)
        total+=len(d)
    assert int(cells.n.sum())==config['n_samples']
    # Independently verify time weighting and cell assignment for a representative stream.
    d=pd.read_csv(out/'points/P3.csv.gz');d=d[d.trial_id.eq(1)]
    d['cell_x']=np.floor(d.x/config['cell_m']).astype(int);d['cell_z']=np.floor(d.z/config['cell_m']).astype(int)
    for key,g in d.groupby(['floor','cell_x','cell_z']):
        c=cells[(cells.trial_id==1)&(cells.period=='P3')&(cells.floor==key[0])&(cells.cell_x==key[1])&(cells.cell_z==key[2])]
        assert len(c)==1 and np.isclose(c.iloc[0].Value,np.average(g.value,weights=g.weight))
    print(metric,'PASS:',config['n_samples'],'samples; 80 plots, 16 merges; strict masks and original metric checks.',flush=True)
print('TOTAL VERIFIED',total,flush=True)
