"""Choose an IRL rule per guidance using validation only; retain all sources.

This is model selection, not a new reward or policy optimizer. The inspected
test set is reused for exploratory reporting, not claimed as a fresh blind test.
"""
from pathlib import Path
from datetime import datetime
import argparse,json,math,shutil,time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from irl_search_v4 import write_json,load_cache,digest
from irl_data_v3 import GUIDANCE
from irl_frame_optimization import load_frames,prepare_frames,frame_report,frame_logit
from irl_core_v3 import LinearReward,soft_policy
from Spatial_IRL_Rule import SpatialIRLRule,reward_mask,SCOPE
from spatial_reward_basis import SpatialRewardBasis
from IRL_V4_Worker import configure_device
from Train_IRL_V3_Optimize import launch_batch

HERE=Path(__file__).resolve().parent

def contrast(rule,x,g):
    j=GUIDANCE.index(g)
    return (rule.basis.transform(x)@rule.weights[j]) if rule.basis else x@rule.weights[j,rule.mask.astype(bool)]

def choose(search):
    records=json.loads((search/'search_records.json').read_text(encoding='utf-8'))
    records=[r for r in records if r['status']=='complete']
    # Only validation values enter this pool. Historical baselines are eligible.
    reference=load_frames(records[0]['frames'],'val')
    expected_hash=json.loads((Path(records[0]['frames'])/'metadata.json').read_text(encoding='utf-8'))['manifest_sha256']
    for label,run in [('V3_2s','20260927_174152_043499'),('V4_initial','20260927_185236_303154/Best_Model')]:
        folder=HERE/'Results_IRL'/run
        if not (folder/'config.json').exists() or not (folder/'evaluation.json').exists():continue
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'));evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        if label=='V4_initial':
            parent=folder.parent;selected=json.loads((parent/'selection.json').read_text(encoding='utf-8'))['candidate']
            cache=selected['cache']
        else:
            # V3 direct output has the same on-disk MDP/encoder but not V4 cache metadata.
            # Its validation loss is much worse for all modes here; include for audit,
            # but require a compatible cache if it were ever selected.
            cache=None
        # Historical candidates must have a compatible data provenance cache.
        if cache is None:continue
        if json.loads((Path(cache)/'metadata.json').read_text(encoding='utf-8'))['manifest_sha256']!=expected_hash:continue
        rule=SpatialIRLRule(folder);val=dict(evaluation['val'])
        val['by_guidance']={g:dict(d) for g,d in val['by_guidance'].items()}
        for i,g in enumerate(GUIDANCE):
            ix=reference['guidance']==i;val['by_guidance'][g]['spatial_reward_auc']=float(roc_auc_score(reference['action'][ix],contrast(rule,reference['x'][ix],g)))
        if not all('participant_macro_nll' in d for d in val['by_guidance'].values()):continue
        records.append(dict(id=label,status='complete',output=str(folder),cache=cache,frames=records[0]['frames'],config=config,validation=val,legacy=True))
    winners={};selection={}
    for g in GUIDANCE:
        minimum=min(r['validation']['by_guidance'][g]['participant_macro_nll'] for r in records)
        pool=[r for r in records if r['validation']['by_guidance'][g]['participant_macro_nll']<=minimum+.005]
        ordered=sorted(pool,key=lambda r:(-r['validation']['by_guidance'][g]['spatial_reward_auc'],r['validation']['by_guidance'][g]['participant_macro_nll'],r['id']))
        winners[g]=ordered[0]
        selection[g]=dict(id=ordered[0]['id'],source=ordered[0]['output'],validation=ordered[0]['validation']['by_guidance'][g],minimum_validation_person_nll=minimum,eligible=[r['id'] for r in ordered])
    return winners,selection

def predictions(record,dataset,testfolder,device):
    mdp,_,_=load_cache(record['cache'],dataset)
    prepare_frames(dataset,record['cache'],testfolder,splits=('test',))
    src=Path(record['output']);rule=SpatialIRLRule(src)
    state=torch.load(src/'reward.pt',map_location='cpu',weights_only=True)
    model=LinearReward(len(rule.basis.names) if rule.basis else len(rule.encoder.names)).to(device);model.load_state_dict(state)
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(np.array(a,copy=True),dtype=dtype,device=device)
    trans=tensor(mdp['transitions']);dt=tensor(mdp['mean_dt']);mdp['guide_tensor']=tensor(mdp['guidance_index'],torch.long)
    c=record['config'];horizon=max(1,int(math.ceil(c['horizon_seconds']/mdp['step_s'])))
    if record.get('legacy'):
        legacy_frames=testfolder.parent/(record['id']+'_train_val')
        prepare_frames(dataset,record['cache'],legacy_frames)
        record=dict(record,frames=str(legacy_frames))
    if not record.get('legacy'):
        mean=tensor(np.load(src/'state_reward_features.npy'))
    else:
        with torch.no_grad():lp,_=soft_policy(model(tensor(mdp['centers']*reward_mask(mdp['encoder'].names)),mdp['guide_tensor']),trans,horizon,c['gamma'],mdp['terminal_index'],mean_dt=dt)
    result={}
    with torch.no_grad():
        for split in ['train','val','test']:
            d=load_frames(testfolder if split=='test' else record['frames'],split)
            if record.get('legacy'):
                p=lp.exp().cpu().numpy()[d['state'],1]
                w=rule.weights[d['guidance']][:,rule.mask.astype(bool)];u=(d['x']*w).sum(1)
            else:
                z,u=frame_logit(model,tensor(rule.basis.transform(d['x'])),tensor(d['guidance'],torch.long),tensor(d['state'],torch.long),mean,mdp,trans,dt,horizon,c['gamma'])
                p=torch.sigmoid(z).cpu().numpy();u=u.cpu().numpy()
            result[split]=dict(data=d,probability=p,utility=u)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--search',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--jobs',type=int,default=2);p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=16)
    p.add_argument('--budget-hours',type=float,default=.5);p.add_argument('--skip-plots',action='store_true');p.add_argument('--reuse-checks',type=Path)
    args=p.parse_args()
    if args.jobs<1 or args.epochs<1 or args.budget_hours<=0:p.error('Positive jobs, epochs and budget are required')
    device=configure_device(args.device);torch.set_num_threads(2);started=time.time()
    out=HERE/'Results_IRL'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_V3OptimizeBank');out.mkdir()
    for n in ['Tasks','Trials','Caches','Frame_Caches']:(out/n).mkdir()
    winners,selection=choose(args.search)
    # Freeze before evaluating any component on test. The test set is already
    # known from earlier runs; explicitly record this exploratory limitation.
    first=next(iter(winners.values()))
    write_json(out/'selection.json',dict(by_guidance=selection,candidate=first,
        frozen_at=datetime.now().isoformat(),test_used_for_selection=False,test_previously_inspected=True,
        selection_rule='Per-guidance max spatial AUC within minimum validation participant NLL + 0.005',
        limitation='Exploratory reuse of a previously inspected test split; independent replication still required'))
    print('FROZEN',json.dumps({g:r['id'] for g,r in winners.items()}),flush=True)
    previous_checks=json.loads((args.reuse_checks/'seed_checks.json').read_text(encoding='utf-8')) if args.reuse_checks else []
    checks=[];deadline=time.time()+args.budget_hours*3600-120;caches={};frames={}
    for g,r in winners.items():
        reused=[q for q in previous_checks if q.get('status')=='complete' and q.get('guidance_component')==g and all(r['config'].get(k)==v for k,v in q['config'].items() if k!='seed')]
        reused=[q for q in reused if Path(q['output']).joinpath('reward.pt').exists() and json.loads((Path(q['cache'])/'metadata.json').read_text(encoding='utf-8'))['manifest_sha256']==digest(args.dataset/'manifest.json')]
        if len(reused)>=2:
            checks.extend(reused);continue
        if r.get('legacy'):
            # Existing complete repeats of the same V4 baseline are reused.
            previous=HERE/'Results_IRL/20260927_190858_823193/seed_checks.json'
            for q in json.loads(previous.read_text(encoding='utf-8')):
                if {k:v for k,v in q['config'].items() if k!='seed'}=={k:r['config'][k] for k in q['config'] if k!='seed'}:
                    checks.append(dict(q,guidance_component=g))
            continue
        check_records={};configs=[dict(r['config'],seed=20260926+11*i) for i in [1,2]]
        launch_batch(configs,args,out,deadline,check_records,caches,frames)
        checks.extend(dict(q,guidance_component=g) for q in check_records.values())
    write_json(out/'seed_checks.json',checks)
    folder=out/'Best_Model';folder.mkdir();(folder/'Models').mkdir();bank={};all_predictions={};reference_data={};test_caches={}
    for g,r in winners.items():
        if r['cache'] is None:raise RuntimeError('Selected historical model lacks compatible cache')
        child=folder/'Models'/g;child.mkdir();source=Path(r['output'])
        for n in ['reward.pt','reward_basis.json','state_reward_features.npy','training_history.csv','config.json','encoder.json']:
            if (source/n).exists():shutil.copy2(source/n,child/n)
        if not (child/'encoder.json').exists():shutil.copy2(Path(r['cache'])/'encoder.json',child/'encoder.json')
        if not (child/'config.json').exists():
            manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
            from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
            write_json(child/'config.json',dict(r['config'],status='complete',method='budgeted_full_frame_spatial_IRL_v4',reward_scope=SCOPE,geometry_sha256=manifest['source_hashes'][str(Path(DEFAULT_GEOMETRY_PATH).resolve())]))
        # Keep the exact empirical MDP for audit/replay, although point queries do not need it.
        for name in ['mdp.npz','metadata.json']:
            shutil.copy2(Path(r['cache'])/name,child/('mdp_metadata.json' if name=='metadata.json' else name))
        bank[g]=str(Path('Models')/g)
        testfolder=out/'Test_Caches'/r['id'];pred=predictions(dict(r,output=str(child)),args.dataset,testfolder,device);test_caches[g]=str(testfolder)
        for split,d in pred.items():
            if split not in all_predictions:
                reference_data[split]=d['data'];all_predictions[split]=dict(probability=np.zeros(len(d['data']['action']),np.float32),utility=np.zeros(len(d['data']['action']),np.float32))
            assert np.array_equal(reference_data[split]['action'],d['data']['action'])
            ix=d['data']['guidance']==GUIDANCE.index(g)
            for n in ['probability','utility']:all_predictions[split][n][ix]=d[n][ix]
    evaluation={s:frame_report(reference_data[s],d['probability'],d['utility']) for s,d in all_predictions.items()}
    saved={}
    for s in ['val','test']:
        for n,v in dict(**all_predictions[s],**{n:reference_data[s][n] for n in ['action','person','guidance']}).items():saved[s+'_'+n]=v
    np.savez_compressed(folder/'evaluation_predictions.npz',**saved)
    evaluation.update(trained=True,test_used_for_selection=False,test_previously_inspected=True,
        score_meaning='Policy uses dynamics; spatial_reward_auc and final fields use spatial-only immediate contrast',by_guidance_selection=selection)
    write_json(folder/'evaluation.json',evaluation)
    config=json.loads((folder/'Models'/GUIDANCE[0]/'config.json').read_text(encoding='utf-8'))
    config.update(status='complete',method='budgeted_full_frame_spatial_IRL_v4',reward_scope=SCOPE,optimization='v3_per_guidance_validated_rule_bank',model_bank=bank,
        dataset=str(args.dataset),search_run=str(out),plot_status='pending',plot_sigma_m=.6,plot_upsample=4,plot_dpi=300,
        source_search=str(args.search),test_used_for_selection=False,test_previously_inspected=True)
    write_json(folder/'config.json',config);write_json(out/'test_caches.json',test_caches)
    rules=[]
    for g,relative in bank.items():
        rule=SpatialIRLRule(folder/relative)
        names=rule.basis.names if rule.basis else rule.encoder.names
        for name,value in zip(names,rule.weights[GUIDANCE.index(g)]):
            rules.append(dict(guidance=g,feature=name,relative_utility_coefficient=float(value),basis=rule.basis.spec['kind'] if rule.basis else 'linear'))
    pd.DataFrame(rules).to_csv(folder/'reward_rules.csv',index=False)
    # Alias only for post-selection audit scripts, never required by point queries.
    write_json(out/'test_cache_reference.json',dict(folder=test_caches[GUIDANCE[0]]))
    if not args.skip_plots:
        from Plot_IRL_V4 import plot_run
        plot_run(folder)
    write_json(out/'completion.json',dict(status='complete',best_model=str(folder),total_seconds=time.time()-started,test_used_for_selection=False,
        seed_checks_completed=sum(r['status']=='complete' for r in checks),png_count=0 if args.skip_plots else 4))
    print('BEST_MODEL='+str(folder));print('OUTPUT='+str(out))

if __name__=='__main__':main()
