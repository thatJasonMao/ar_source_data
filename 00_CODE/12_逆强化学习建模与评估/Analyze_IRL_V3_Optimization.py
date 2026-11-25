"""Post-selection evidence: clustered bootstrap and raw-field seed stability."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from irl_search_v4 import write_json
from Spatial_IRL_Rule import SpatialIRLRule
from irl_data_v3 import GUIDANCE

def participant_bootstrap(y,score,person,repeats=500,seed=20260926):
    # Sort/group scores once. Each bootstrap samples participants, never frames.
    _,inverse=np.unique(score,return_inverse=True)
    people=np.unique(person);counts=np.zeros((len(people),inverse.max()+1,2),np.float64)
    for i,p in enumerate(people):
        ix=person==p;np.add.at(counts[i],(inverse[ix],y[ix].astype(int)),1.)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        mult=np.bincount(rng.integers(0,len(people),len(people)),minlength=len(people))
        c=(counts*mult[:,None,None]).sum(0);neg,pos=c[:,0],c[:,1]
        if not pos.sum() or not neg.sum():continue
        values.append(float((pos*(np.cumsum(neg)-.5*neg)).sum()/(pos.sum()*neg.sum())))
    return dict(participants=len(people),replicates=len(values),auc_ci95=np.quantile(values,[.025,.975]).tolist(),resampling_unit='participant, keeping all within-person frames')

def main():
    p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True);args=p.parse_args()
    out=args.search;model=out/'Best_Model';eval=json.loads((model/'evaluation.json').read_text())
    rows=[]
    with np.load(model/'evaluation_predictions.npz',allow_pickle=False) as f:
        for split in ['val','test']:
            y=f[split+'_action'];person=f[split+'_person'];guide=f[split+'_guidance']
            for i,g in enumerate(GUIDANCE):
                ix=guide==i
                for mode,column in [('policy','probability'),('spatial_reward','utility')]:
                    r=participant_bootstrap(y[ix],f[split+'_'+column][ix],person[ix]);r.update(split=split,guidance=g,mode=mode)
                    rows.append(r)
    write_json(out/'participant_bootstrap.json',dict(rows=rows,test_used_for_selection=False,
        limitation='One fixed participant split; validation intervals do not adjust for search selection'))
    # Compare the same observed test rows using spatial rewards alone, rather
    # than mixing legacy state-policy AUC with a new spatial-field AUC.
    from irl_frame_optimization import load_frames
    from sklearn.metrics import roc_auc_score
    test_reference=out/'Final_Test_Cache'
    if not test_reference.exists():test_reference=Path(json.loads((out/'test_cache_reference.json').read_text(encoding='utf-8'))['folder'])
    d=load_frames(test_reference,'test')
    current_rule=SpatialIRLRule(model);baseline_rows=[]
    result_root=Path(__file__).resolve().parent/'Results_IRL'
    for label,relative in [('V3_2s','20260927_174152_043499'),('V4_initial','20260927_185236_303154/Best_Model')]:
        legacy=SpatialIRLRule(result_root/relative)
        if not np.allclose(legacy.encoder.median,current_rule.encoder.median) or not np.allclose(legacy.encoder.scale,current_rule.encoder.scale):raise ValueError('Baseline encoding differs')
        score=(d['x']*legacy.weights[d['guidance']][:,legacy.mask.astype(bool)]).sum(1)
        for i,g in enumerate(GUIDANCE):
            ix=d['guidance']==i;baseline_rows.append(dict(model=label,guidance=g,spatial_reward_auc=float(roc_auc_score(d['action'][ix],score[ix]))))
    for g,r in eval['test']['by_guidance'].items():baseline_rows.append(dict(model='V3_optimized',guidance=g,spatial_reward_auc=r['spatial_reward_auc']))
    write_json(out/'spatial_reward_comparison.json',dict(rows=baseline_rows,same_test_rows=True,test_used_for_selection=False))
    field=model/'Results_Global_Spatial_Field_V4/global_spatial_rule_field.csv.gz'
    if not field.exists():return
    selected=json.loads((out/'selection.json').read_text())['candidate']
    rule=SpatialIRLRule(model);frame=pd.read_csv(field)
    checks=json.loads((out/'seed_checks.json').read_text()) if (out/'seed_checks.json').exists() else []
    stability=[]
    for c in checks:
        if c['status']!='complete':continue
        folder=Path(c['output'])
        # Diagnostic workers save rewards/basis; install compatible metadata only.
        for name in ['encoder.json']:
            if not (folder/name).exists():
                import shutil
                shutil.copy2(Path(c['cache'])/name,folder/name)
        if not (folder/'config.json').exists():
            config=json.loads((model/'config.json').read_text());config.pop('model_bank',None);config.update(c['config'],candidate=True)
            write_json(folder/'config.json',config)
        other=SpatialIRLRule(folder)
        for g,part in frame.groupby('guidance'):
            if c.get('guidance_component') and c['guidance_component']!=g:continue
            a=part.relative_information_utility.to_numpy();b=other.query_features(part,g);n=max(1,int(np.ceil(.1*len(a))))
            aa=set(np.argsort(a)[-n:]);bb=set(np.argsort(b)[-n:])
            stability.append(dict(seed=c['config']['seed'],guidance=g,rank_correlation=float(spearmanr(a,b).statistic),top10_overlap=len(aa&bb)/n,mean_absolute_difference=float(np.abs(a-b).mean())))
    write_json(out/'seed_stability.json',dict(rows=stability,grid='same unsmoothed 0.4m geometry grid',test_used=False))
    # Whole-domain extrapolation audit, without suppressing any queried grid cell.
    training=np.load(Path(selected['frames'])/'train_x.npy',mmap_mode='r');metadata=json.loads((Path(selected['frames'])/'metadata.json').read_text())
    names=metadata['spatial_input_names'];mask=rule.mask.astype(bool)
    from irl_data_v3 import NUMERIC
    query=frame.copy()
    for n in NUMERIC[5:]:query[n]=0.
    query['previous_action']=-1
    x=rule.encoder.transform(query)[:,mask]
    numeric=[names.index('scaled_'+n) for n in NUMERIC[:5]]
    lo=training[:,numeric].min(0);hi=training[:,numeric].max(0)
    outside=((x[:,numeric]<lo-1e-6)|(x[:,numeric]>hi+1e-6)).any(1)
    write_json(out/'spatial_support_audit.json',dict(grid_rows=len(frame),outside_training_numeric_range_fraction=float(outside.mean()),
        meaning='Range check only, not joint-density coverage or confidence. Global rule extrapolation remains an assumption.'))
    print('POST_SELECTION_ANALYSIS='+str(out))

if __name__=='__main__':main()
