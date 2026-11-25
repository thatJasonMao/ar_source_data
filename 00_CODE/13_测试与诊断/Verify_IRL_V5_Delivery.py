"""Verify a completed V5 export and the immutable three-guidance data filter."""
from pathlib import Path
import argparse,json,sys,tempfile,shutil
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent.parent/'0_RL_FRAMEWORK'
sys.path.insert(0,str(ROOT))
from Spatial_IRL_Rule import SpatialIRLRule
from Train_IRL_V5 import GUIDANCE,select_by_guidance
from irl_search_v4 import digest,load_cache,write_json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run;best=run/'Best_Model';cfg=read(best/'config.json');dataset=Path(cfg['dataset']);m=read(dataset/'manifest.json')
    parent=read(Path(m['parent_dataset'])/'manifest.json');source={r['trial_id']:r for r in parent['trials']}
    assert len(m['trials'])==88 and m['cohort_participants']==44
    assert cfg['active_guidance']==GUIDANCE and cfg['test_previously_inspected']
    for r in m['trials']:
        assert r['guidance'] in GUIDANCE and r==source[r['trial_id']]
        assert digest(dataset/r['file'])==r['sha256']
        assert m['participant_splits'][r['participant_id']]==parent['participant_splits'][r['participant_id']]
    runs=pd.read_csv(dataset/'action_runs.csv');assert set(runs.trial_id)==set(r['trial_id'] for r in m['trials'])
    assert digest(Path(m['parent_dataset'])/'manifest.json')==cfg['parent_manifest_sha256']
    assert digest(dataset/'manifest.json')==cfg['manifest_sha256']
    for name,h in cfg['source_code_sha256'].items():assert digest(ROOT/name)==h
    records=read(run/'search_records.json');winners,_=select_by_guidance({r['id']:r for r in records})
    selected=read(run/'selection.json')['by_guidance'];assert all(winners[g]['id']==selected[g]['id'] for g in GUIDANCE)
    rule=SpatialIRLRule(best)
    for g,relative in cfg['model_bank'].items():
        child=best/relative;state=torch.load(child/'reward.pt',map_location='cpu',weights_only=True)
        assert state['deviation'].shape[0]==3
        enc=read(child/'encoder.json');assert all(k.split('/')[0] in GUIDANCE for k in enc['contexts'])
    # No trial/MDP files are required by a serialized spatial rule.
    with tempfile.TemporaryDirectory(prefix='v5_rule_only_') as temp:
        portable=Path(temp);shutil.copy2(best/'config.json',portable/'config.json')
        for relative in cfg['model_bank'].values():
            target=portable/relative;target.mkdir(parents=True)
            for name in ['config.json','encoder.json','reward_basis.json','reward.pt']:shutil.copy2(best/relative/name,target/name)
        query=SpatialIRLRule(portable)
        field=best/'Results_Global_Spatial_Field_V4';frame=pd.read_csv(field/'global_spatial_rule_field.csv.gz')
        assert len(list(field.glob('*.png')))==3 and not list(best.rglob('*.pdf'))
        errors={}
        for g in GUIDANCE:
            part=frame.loc[frame.guidance.eq(g)]
            np.testing.assert_allclose(rule.query_features(part,g),query.query_features(part,g),atol=0,rtol=0)
            errors[g]=float(np.abs(query.query_features(part,g)-part.relative_information_utility.to_numpy()).max())
            assert errors[g]<1e-5
        try:query.query_features(frame.iloc[:1],'Full')
        except ValueError:pass
        else:raise AssertionError('Full was accepted')
    # Validate all complete state caches and exposure consistency, not only the winning state resolution.
    all_records=records+read(run/'seed_checks.json')
    caches={r['cache'] for r in all_records if r['status']=='complete'}
    for cache in caches:
        mdp,_,_=load_cache(cache,dataset)
        assert all(k.split('/')[0] in GUIDANCE for k in mdp['contexts'])
        record=next(r for r in all_records if r.get('cache')==cache and r['status']=='complete')
        frames=Path(record['frames']);s=np.load(frames/'train_state.npy');a=np.load(frames/'train_action.npy').astype(int);w=np.load(frames/'train_weight.npy')
        count=np.zeros_like(mdp['counts'],dtype=np.float64);np.add.at(count,(s,a),w)
        assert np.allclose(count,mdp['counts'],atol=1e-6)
    result=dict(status='passed',trial_bytes_unchanged=True,split_map_unchanged=True,full_frames=0,reward_branches=3,
        selection_recomputed=True,portable_query_without_mdp=True,raw_query_csv_max_error=errors,cache_variants_verified=len(caches),
        png_count=3,pdf_count=0,parent_manifest_unchanged=True,source_code_unchanged_since_export=True)
    write_json(run/'verification_final.json',result);print(json.dumps(result))

if __name__=='__main__':main()
