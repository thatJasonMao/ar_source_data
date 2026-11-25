"""V5: bounded full-frame spatial IRL for ArrowRibbon, MiniMap and TextBar.

Full is excluded before encoding, state estimation, basis fitting and selection.
No forward RL, new packages or behavior penalties. Original inputs are immutable.
"""
from pathlib import Path
from datetime import datetime
import argparse,json,itertools,sys,shutil,time,math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from irl_data_v3 import validate_manifest_scope,audit_dataset,NUMERIC
from irl_search_v4 import write_json,digest,load_cache
from irl_frame_optimization import prepare_frames,load_frames,frame_report,frame_logit
from spatial_reward_basis import SpatialRewardBasis
from irl_core_v3 import LinearReward
from Spatial_IRL_Rule import SpatialIRLRule,SCOPE
from IRL_V4_Worker import configure_device
from Train_IRL_V3_Optimize import launch_batch

HERE=Path(__file__).resolve().parent
GUIDANCE=['ArrowRibbon','MiniMap','TextBar']
SCOPE_V5='three_single_guidance_exclude_full_v5'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def build_dataset(source,target):
    """Copy only selected immutable frame files; retain IDs, labels and splits."""
    source=Path(source).resolve();target=Path(target)
    m=read(source/'manifest.json');validate_manifest_scope(m)
    if target.exists():raise FileExistsError(target)
    # The source cohort was built through common.load_trials(); recheck that
    # canonical selection today agrees instead of enumerating RAW_DATA folders.
    project=HERE.parent
    if str(project) not in sys.path:sys.path.insert(0,str(project))
    from common import load_trials
    from IRL_Cohort import select_irl_trials
    canonical,_=select_irl_trials(load_trials())
    expected={r.trial_id:(r.ParticipantID,r.Guidance,r.Smoke) for r in canonical.itertuples() if r.Guidance in GUIDANCE}
    kept=[r for r in m['trials'] if r['guidance'] in GUIDANCE]
    actual={r['trial_id']:(r['participant_id'],r['guidance'],r['smoke']) for r in kept}
    if actual!=expected:raise ValueError('V5 source does not match current common.load_trials cohort')
    if not kept or set(r['guidance'] for r in kept)!=set(GUIDANCE):raise ValueError('Missing V5 guidance')
    target.mkdir(parents=True);(target/'trials').mkdir()
    nrows=nvalid=one=zero=mixed=0
    for r in tqdm(kept,desc='Copy V5 full-frame inputs'):
        src=(source/r['file']).resolve()
        if not src.is_relative_to(source):raise ValueError('Invalid trial path')
        if digest(src)!=r['sha256']:raise ValueError('Source trial changed')
        dest=(target/r['file']).resolve()
        if not dest.is_relative_to(target.resolve()):raise ValueError('Invalid output path')
        shutil.copy2(src,dest)
        d=pd.read_csv(dest,usecols=['action','valid']);v=d.valid.eq(1)
        nrows+=len(d);nvalid+=int(v.sum());one+=int((v&d.action.eq(1)).sum());zero+=int((v&d.action.eq(0)).sum())
        mixed+=int(d.loc[v,'action'].nunique()==2)
    people={r['participant_id'] for r in kept};removed=[r for r in m['trials'] if r['guidance'] not in GUIDANCE]
    runs=pd.read_csv(source/'action_runs.csv');runs=runs.loc[runs.trial_id.isin(actual)].copy()
    runs.to_csv(target/'action_runs.csv',index=False)
    derived=dict(m,trials=kept,cohort_trials=len(kept),cohort_participants=len(people),created_at=datetime.now().isoformat(),
        active_guidance=GUIDANCE,training_scope=SCOPE_V5,parent_dataset=str(source),parent_manifest_sha256=digest(source/'manifest.json'),
        excluded_trials=m['excluded_trials']+len(removed),exclusion_counts=dict(m['exclusion_counts'],full_outside_v5_scope=len(removed)),
        participant_splits={p:s for p,s in m['participant_splits'].items() if p in people},
        sampled_rows=nrows,valid_rows=nvalid,action_one=one,action_zero=zero,mixed_action_trials=mixed)
    derived['config']=dict(m['config'],scope=SCOPE_V5)
    # Median nominal interval stays identical to V3 for a matched protocol;
    # actual timestamp durations always drive transitions and discounting.
    write_json(target/'manifest.json',derived)
    write_json(target/'filter_audit.json',dict(parent_trials=len(m['trials']),selected_trials=len(kept),excluded_full=len(removed),
        selected_participants=len(people),original_trial_bytes_unchanged=True,split_map_preserved=True,cohort_entry='common.load_trials()',
        excluded_full_trial_ids=[r['trial_id'] for r in removed]))
    return audit_dataset(target)

def select_by_guidance(records):
    complete=[r for r in records.values() if r['status']=='complete']
    if not complete:raise RuntimeError('No V5 candidate completed')
    winners={};summary={}
    for g in GUIDANCE:
        minimum=min(r['validation']['by_guidance'][g]['participant_macro_nll'] for r in complete)
        pool=[r for r in complete if r['validation']['by_guidance'][g]['participant_macro_nll']<=minimum+.005]
        ordered=sorted(pool,key=lambda r:(-r['validation']['by_guidance'][g]['spatial_reward_auc'],r['validation']['by_guidance'][g]['participant_macro_nll'],r['id']))
        winners[g]=ordered[0]
        summary[g]=dict(id=ordered[0]['id'],source=ordered[0]['output'],validation=ordered[0]['validation']['by_guidance'][g],minimum_person_nll=minimum,eligible=[r['id'] for r in ordered])
    return winners,summary

def predictions(record,dataset,testfolder,device):
    mdp,_,_=load_cache(record['cache'],dataset)
    if not (testfolder/'metadata.json').exists():prepare_frames(dataset,record['cache'],testfolder,splits=('test',))
    src=Path(record['output']);basis=SpatialRewardBasis(read(src/'reward_basis.json'))
    model=LinearReward(len(basis.names),n_guidance=3).to(device)
    model.load_state_dict(torch.load(src/'reward.pt',map_location='cpu',weights_only=True))
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(np.array(a,copy=True),dtype=dtype,device=device)
    mean=tensor(np.load(src/'state_reward_features.npy'));trans=tensor(mdp['transitions']);dt=tensor(mdp['mean_dt'])
    mdp['guide_tensor']=tensor(mdp['guidance_index'],torch.long);c=record['config'];horizon=max(1,int(math.ceil(c['horizon_seconds']/mdp['step_s'])))
    result={}
    with torch.no_grad():
        for split in ['train','val','test']:
            d=load_frames(testfolder if split=='test' else record['frames'],split)
            z,u=frame_logit(model,tensor(basis.transform(d['x'])),tensor(d['guidance'],torch.long),tensor(d['state'],torch.long),mean,mdp,trans,dt,horizon,c['gamma'])
            result[split]=dict(data=d,probability=torch.sigmoid(z).cpu().numpy(),utility=u.cpu().numpy())
    return result

def export(args,out,winners,selection):
    folder=out/'Best_Model';folder.mkdir();(folder/'Models').mkdir()
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    manifest=read(args.dataset/'manifest.json');geo=Path(DEFAULT_GEOMETRY_PATH).resolve()
    if digest(geo)!=manifest['source_hashes'][str(geo)]:raise ValueError('Geometry changed')
    bank={};combined={};reference={};testfolders={};rules=[];computed={}
    for i,g in enumerate(GUIDANCE):
        r=winners[g];child=folder/'Models'/g;child.mkdir();src=Path(r['output']);cache=Path(r['cache'])
        for n in ['reward.pt','reward_basis.json','state_reward_features.npy','training_history.csv']:shutil.copy2(src/n,child/n)
        for n in ['encoder.json','mdp.npz']:shutil.copy2(cache/n,child/n)
        shutil.copy2(cache/'metadata.json',child/'mdp_metadata.json')
        config=dict(r['config'],status='complete',version=5,method='budgeted_full_frame_spatial_IRL_v4',reward_scope=SCOPE,
            training_scope=SCOPE_V5,active_guidance=GUIDANCE,geometry_sha256=digest(geo),dataset=str(args.dataset),source_candidate=r['id'])
        write_json(child/'config.json',config);bank[g]=str(child.relative_to(folder))
        rule=SpatialIRLRule(child)
        for n,v in zip(rule.basis.names,rule.weights[i]):rules.append(dict(guidance=g,feature=n,relative_utility_coefficient=float(v),basis=r['config']['basis']))
        if r['id'] not in computed:
            testfolder=out/'Test_Caches'/r['id'];computed[r['id']]=predictions(r,args.dataset,testfolder,args.device)
        testfolders[g]=str(out/'Test_Caches'/r['id'])
        for split,values in computed[r['id']].items():
            d=values['data']
            if split not in combined:
                reference[split]=d;combined[split]=dict(probability=np.zeros(len(d['action']),np.float32),utility=np.zeros(len(d['action']),np.float32))
            for n in ['action','person','guidance']:assert np.array_equal(reference[split][n],d[n]),'Observation order differs'
            ix=d['guidance']==i
            for n in ['probability','utility']:combined[split][n][ix]=values[n][ix]
    evaluations={s:frame_report(reference[s],v['probability'],v['utility']) for s,v in combined.items()}
    pd.DataFrame(rules).to_csv(folder/'reward_rules.csv',index=False)
    arrays={}
    for split in ['val','test']:
        for n,v in dict(**combined[split],**{n:reference[split][n] for n in ['action','person','guidance']}).items():arrays[split+'_'+n]=v
    np.savez_compressed(folder/'evaluation_predictions.npz',**arrays)
    evaluations.update(trained=True,test_used_for_selection=False,test_previously_inspected=True,active_guidance=GUIDANCE,
        score_meaning='Policy uses dynamics; spatial_reward_auc and final fields use spatial-only immediate reward contrast')
    write_json(folder/'evaluation.json',evaluations)
    write_json(folder/'config.json',dict(status='complete',version=5,method='budgeted_full_frame_spatial_IRL_v4',reward_scope=SCOPE,
        optimization='v5_three_guidance_validated_rule_bank',training_scope=SCOPE_V5,active_guidance=GUIDANCE,model_bank=bank,
        dataset=str(args.dataset),parent_dataset=str(args.source_dataset),search_run=str(out),geometry_sha256=digest(geo),
        manifest_sha256=digest(args.dataset/'manifest.json'),parent_manifest_sha256=digest(args.source_dataset/'manifest.json'),
        reward_basis_by_guidance={g:winners[g]['config']['basis'] for g in GUIDANCE},plot_sigma_m=.6,plot_upsample=4,plot_dpi=300,plot_status='pending',
        test_used_for_selection=False,test_previously_inspected=True,
        source_code_sha256={n:digest(HERE/n) for n in ['Train_IRL_V5.py','irl_frame_optimization.py','spatial_reward_basis.py','Spatial_IRL_Rule.py','irl_data_v3.py','irl_data_v4.py','irl_state_distance.py']}))
    write_json(out/'test_caches.json',testfolders)
    write_json(out/'test_cache_reference.json',dict(folder=testfolders[GUIDANCE[0]]))
    if not args.skip_plots:
        from Plot_IRL_V4 import plot_run
        plot_run(folder)
    return folder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True,help='Original V3 full-frame dataset, not a previous V5 subset')
    p.add_argument('--train',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--budget-hours',type=float,default=1.);p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=16)
    p.add_argument('--horizon-seconds',type=float,default=2.);p.add_argument('--families',default='linear,spline,interaction,rbf32,rbf64')
    p.add_argument('--seed-checks',type=int,default=2);p.add_argument('--skip-plots',action='store_true')
    args=p.parse_args();args.dataset=args.dataset.resolve();args.source_dataset=args.dataset
    families=args.families.split(',')
    if any(b not in ['linear','spline','interaction','rbf32','rbf64'] for b in families):p.error('Unknown reward family')
    if args.jobs<1 or args.epochs<5 or args.patience<1 or args.seed_checks<0 or args.budget_hours<=.05 or args.horizon_seconds<=0:p.error('Invalid budgets or stopping settings')
    base=dict(lr=.02,l2=.001,deviation_ratio=1.,clusters=16,run_cap_s=10.,prior=.5,seed=20260926,horizon_seconds=args.horizon_seconds,gamma=.98,basis='linear',active_guidance=GUIDANCE)
    configs=[dict(base,clusters=k,basis=b,l2=l) for k,b,l in itertools.product([16,24],families,[.0001,.001,.01])]
    if not args.train:
        print(json.dumps(dict(trained=False,version=5,active_guidance=GUIDANCE,excluded_guidance=['Full'],configs=configs,test_used_for_selection=False),indent=2));return
    configure_device(args.device);torch.set_num_threads(2)
    started=time.time();deadline=started+args.budget_hours*3600-180
    out=HERE/'Results_IRL'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_V5');out.mkdir()
    for n in ['Tasks','Trials','Caches','Frame_Caches']:(out/n).mkdir()
    write_json(out/'config.json',dict(status='running',version=5,active_guidance=GUIDANCE,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},test_used_for_selection=False))
    args.dataset=out/'Dataset_V5';audit=build_dataset(args.source_dataset,args.dataset);write_json(out/'dataset_audit.json',audit)
    write_json(out/'plan.json',dict(configs=configs,extension='Each guidance best two by validation person NLL: clusters48, horizon1/5, lr.005/.05',
        selection_rule='Per-guidance max spatial AUC within minimum validation participant NLL +0.005',excluded_guidance=['Full'],test_previously_inspected=True))
    records={};caches={};frames={};launch_batch(configs,args,out,deadline,records,caches,frames)
    complete=[r for r in records.values() if r['status']=='complete'];extras=[]
    # Focus refinement on each included guidance rather than a four-mode macro score.
    for g in GUIDANCE:
        ordered=sorted(complete,key=lambda r:r['validation']['by_guidance'][g]['participant_macro_nll'])
        for r in ordered[:2]:
            for update in [dict(clusters=48),dict(horizon_seconds=1.),dict(horizon_seconds=5.),dict(lr=.005),dict(lr=.05)]:
                c=dict(r['config'],**update)
                if c not in extras:extras.append(c)
    write_json(out/'extension_plan.json',extras);launch_batch(extras,args,out,deadline,records,caches,frames)
    winners,selection=select_by_guidance(records)
    write_json(out/'selection.json',dict(by_guidance=selection,candidate=winners[GUIDANCE[0]],frozen_at=datetime.now().isoformat(),
        test_used_for_selection=False,test_previously_inspected=True,active_guidance=GUIDANCE,
        selection_rule='Per-guidance max spatial AUC within minimum validation participant NLL +0.005'))
    rows=[]
    for r in records.values():
        row=dict(id=r['id'],status=r['status'],**r['config']);row['active_guidance']=','.join(GUIDANCE)
        if r['status']=='complete':
            row.update(score=r['score'],best_epoch=r['best_epoch'],epochs_run=r['epochs_run'])
            for g,d in r['validation']['by_guidance'].items():
                for n in ['auc','participant_macro_nll','spatial_reward_auc']:row[g+'_'+n]=d[n]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'search_summary.csv',index=False)
    checks={};seed_configs=[]
    for r in winners.values():
        for i in range(1,args.seed_checks+1):
            c=dict(r['config'],seed=20260926+11*i)
            if c not in seed_configs:seed_configs.append(c)
    write_json(out/'seed_plan.json',seed_configs);launch_batch(seed_configs,args,out,deadline,checks,caches,frames)
    diagnostics=[]
    for g,r in winners.items():
        for q in checks.values():
            if all(q['config'][n]==v for n,v in r['config'].items() if n!='seed'):diagnostics.append(dict(q,guidance_component=g))
    write_json(out/'seed_checks.json',diagnostics);write_json(out/'search_records.json',list(records.values()))
    folder=export(args,out,winners,selection)
    from Analyze_IRL_V5 import analyze
    analyze(out)
    write_json(out/'completion.json',dict(status='complete',version=5,best_model=str(folder),total_seconds=time.time()-started,
        completed=sum(r['status']=='complete' for r in records.values()),failed=sum(r['status']!='complete' for r in records.values()),
        seed_checks_completed=sum(r['status']=='complete' for r in diagnostics),png_count=0 if args.skip_plots else 3,pdf_count=0,
        test_used_for_selection=False,test_previously_inspected=True))
    cfg=read(out/'config.json');cfg.update(status='complete',best_model=str(folder),derived_dataset=str(args.dataset));write_json(out/'config.json',cfg)
    print('BEST_MODEL='+str(folder));print('OUTPUT='+str(out))

if __name__=='__main__':main()
