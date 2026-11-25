"""Bounded V6 shared-reward IRL for four channels across five conditions.

No RL policy optimization, no new packages, no trial identity in the reward.
Test observations are loaded once after validation selection is frozen.
"""
from pathlib import Path
from datetime import datetime
import argparse,json,math,os,shutil,subprocess,sys,time,itertools
import numpy as np
import pandas as pd
import torch
from irl_data_v6 import audit_dataset,GUIDANCE,validate_manifest_scope
from irl_search_v6 import write_json,digest,config_id,prepare_cache,load_cache,state_code_hashes
from irl_frame_optimization_v6 import prepare_frames,load_frames,frame_report,frame_logit,training_code_hashes
from spatial_reward_basis import SpatialRewardBasis
from Spatial_IRL_Rule_V6 import reward_mask,SCOPE
from irl_core_v3 import LinearReward
from IRL_V4_Worker import configure_device
from Guidance_Channel_Contract import CONDITIONS

HERE=Path(__file__).resolve().parent

def cache_for(args,out,clusters,cap,seed):
    spec=dict(clusters=clusters,run_cap_s=cap,seed=seed,prior=.5)
    for run in sorted((HERE/'Results_IRL').glob('*_V6'),reverse=True):
        for meta in run.glob('Caches/*/metadata.json'):
            m=json.loads(meta.read_text())
            expected=state_code_hashes()
            if m.get('state')==spec and m.get('status')=='complete' and m.get('state_code_sha256')==expected:
                try:load_cache(meta.parent,args.dataset)
                except (ValueError,OSError):continue
                return meta.parent
    folder=out/'Caches'/config_id(spec);prepare_cache(args.dataset,folder,spec);return folder

def launch_batch(configs,args,out,deadline,records,caches,frames):
    pending=[c for c in configs if config_id(c) not in records];active={}
    try:
        while (pending or active) and time.time()<deadline:
            for pid,t in list(active.items()):
                proc=t['process']
                if proc.poll() is None:continue
                t['log'].close();folder=t['folder']
                if proc.returncode==0 and (folder/'result.json').exists():
                    r=json.loads((folder/'result.json').read_text());r.update(id=t['id'],output=str(folder),cache=str(t['cache']),frames=str(t['frames']))
                else:r=dict(id=t['id'],config=t['config'],status='failed',output=str(folder),returncode=proc.returncode)
                records[t['id']]=r;del active[pid]
                write_json(out/'search_records.json',list(records.values()))
                print('FINISHED',t['id'],r['status'],r.get('score'),flush=True)
            while pending and len(active)<args.jobs:
                c=pending.pop(0);cid=config_id(c);key=(c['clusters'],c['run_cap_s'],c['seed'])
                if key not in caches:
                    caches[key]=cache_for(args,out,*key);frames[key]=out/'Frame_Caches'/config_id(dict(key=list(key)))
                    print('PREPARE FRAMES',key,flush=True);prepare_frames(args.dataset,caches[key],frames[key])
                folder=out/'Trials'/cid;folder.mkdir(parents=True)
                job=dict(config=c,output=str(folder),cache=str(caches[key]),frames=str(frames[key]),dataset=str(args.dataset),
                    device=args.device,threads=2,epochs=args.epochs,patience=args.patience,deadline=deadline)
                jobpath=out/'Tasks'/f'{cid}.json';write_json(jobpath,job)
                log=(folder/'worker.log').open('w',encoding='utf-8');env=os.environ.copy()
                for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]='2'
                process=subprocess.Popen([sys.executable,'-B',str(HERE/'irl_frame_optimization_v6.py'),'--job',str(jobpath)],stdout=log,stderr=subprocess.STDOUT,env=env)
                active[process.pid]=dict(process=process,log=log,folder=folder,id=cid,config=c,cache=caches[key],frames=frames[key])
                print('TRAIN',cid,c,flush=True)
            time.sleep(.5)
        if active:
            for t in active.values():(t['folder']/'STOP').touch()
            for t in active.values():
                try:t['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:t['process'].terminate();t['process'].wait(timeout=10)
                t['log'].close()
                if t['process'].returncode==0 and (t['folder']/'result.json').exists():
                    r=json.loads((t['folder']/'result.json').read_text());r.update(id=t['id'],output=str(t['folder']),cache=str(t['cache']),frames=str(t['frames']));records[t['id']]=r
            active.clear()
    finally:
        for t in active.values():
            if t['process'].poll() is None:t['process'].terminate();t['process'].wait(timeout=10)
            t['log'].close()
        write_json(out/'search_records.json',list(records.values()))

def rank(records):
    return sorted([r for r in records.values() if r['status']=='complete'],key=lambda r:r['score'])

def select(records):
    ordered=rank(records)
    if not ordered:raise RuntimeError('No candidate completed')
    # Fixed before search: reward spatial discrimination is the target, within
    # 0.005 of the best validated equal-guidance log loss.
    pool=[r for r in ordered if r['score']<=ordered[0]['score']+.005]
    return sorted(pool,key=lambda r:(-r['validation']['spatial_macro_auc'],r['score'],r['id']))[0]

def plot_best_model(folder, *, skip_plots=False):
    """Export/reuse a verified raw field, then render its four channels in 3D."""
    if skip_plots:
        return
    folder=Path(folder).resolve()
    config_path=folder/'config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('plot_status')=='complete':
        field=folder/'Results_Global_Spatial_Field_V6'
        manifest=json.loads((field/'manifest.json').read_text(encoding='utf-8'))
        expected=[g+'_spatial_demand.png' for g in GUIDANCE]
        if (manifest.get('status')!='complete' or manifest.get('method')!=config.get('method')
                or manifest.get('files')!=expected
                or any(not (field/n).is_file() or (field/n).stat().st_size==0 for n in expected)
                or manifest.get('raw_csv_sha256')!=digest(field/'global_spatial_rule_field.csv.gz')
                or manifest.get('reward_sha256')!=digest(folder/'reward.pt')
                or manifest.get('geometry_sha256')!=config.get('geometry_sha256')):
            raise ValueError('Completed 2D spatial field failed integrity checks; repair it before resuming 3D plots')
        print('REUSE COMPLETED 2D SPATIAL FIELD '+str(field),flush=True)
    else:
        subprocess.run([sys.executable,'-B',str(HERE/'Plot_IRL_V6.py'),'--run',str(folder)],check=True)
        config=json.loads(config_path.read_text(encoding='utf-8'))
    config['plot_3d_status']='running'
    write_json(config_path,config)
    try:
        subprocess.run([sys.executable,'-B',str(HERE/'Plot_Spatial_Rule_3D.py'),'--run',str(folder)],check=True)
    except Exception as exc:
        config['plot_3d_status']='failed';config['plot_3d_error']=str(exc)
        write_json(config_path,config)
        raise
    config['plot_3d_status']='complete';config.pop('plot_3d_error',None)
    config['plot_3d_directory']='Results_Global_Spatial_Field_V6/Results_3D'
    write_json(config_path,config)


def export(args,out,selected):
    folder=out/'Best_Model';folder.mkdir()
    source=Path(selected['output']);cache=Path(selected['cache']);mdp,_,_=load_cache(cache,args.dataset)
    for name in ['reward.pt','reward_basis.json','state_reward_features.npy','training_history.csv']:shutil.copy2(source/name,folder/name)
    for name in ['encoder.json','mdp.npz']:shutil.copy2(cache/name,folder/name)
    c=selected['config'];basis=SpatialRewardBasis(json.loads((source/'reward_basis.json').read_text()))
    model=LinearReward(len(basis.names),n_guidance=len(GUIDANCE));model.load_state_dict(torch.load(source/'reward.pt',map_location='cpu',weights_only=True))
    device=configure_device(args.device);model.to(device);torch.set_num_threads(2)
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(np.array(a,copy=True),dtype=dtype,device=device)
    mean=tensor(np.load(source/'state_reward_features.npy'));trans=tensor(mdp['transitions']);dt=tensor(mdp['mean_dt']);mdp['guide_tensor']=tensor(mdp['guidance_index'],torch.long)
    horizon=max(1,int(math.ceil(c['horizon_seconds']/mdp['step_s'])))
    # Selection is already frozen. Test frames enter only this final report.
    testfolder=out/'Final_Test_Cache';prepare_frames(args.dataset,cache,testfolder,splits=('test',))
    evaluations={};predictions={}
    with torch.no_grad():
        for split in ['train','val','test']:
            d=load_frames(testfolder if split=='test' else selected['frames'],split)
            z,u=frame_logit(model,tensor(basis.transform(d['x'])),tensor(d['guidance'],torch.long),tensor(d['state'],torch.long),mean,mdp,trans,dt,horizon,c['gamma'])
            probabilities=torch.sigmoid(z).cpu().numpy();utility=u.cpu().numpy()
            evaluations[split]=frame_report(d,probabilities,utility)
            if split in ['val','test']:
                for key,value in dict(probability=probabilities,utility=utility,action=d['action'],person=d['person'],guidance=d['guidance'],condition=d['condition']).items():predictions[split+'_'+key]=value
    np.savez_compressed(folder/'evaluation_predictions.npz',**predictions)
    write_json(folder/'evaluation_identity.json',dict(channels=GUIDANCE,conditions=CONDITIONS,
        people={s:json.loads((Path(testfolder if s=='test' else selected['frames'])/'metadata.json').read_text())['splits'][s]['people']
                for s in ['train','val','test']}))
    state=torch.load(source/'reward.pt',map_location='cpu',weights_only=True);dev=state['deviation'].numpy();weights=state['shared'].numpy()[None,:]+dev-dev.mean(0)
    pd.DataFrame([dict(guidance=g,feature=n,contrast_weight=float(weights[i,j])) for i,g in enumerate(GUIDANCE) for j,n in enumerate(basis.names)]).to_csv(folder/'reward_rules.csv',index=False)
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    manifest=json.loads((args.dataset/'manifest.json').read_text());geo=Path(DEFAULT_GEOMETRY_PATH).resolve()
    if digest(geo)!=manifest['source_hashes'][str(geo)]:raise ValueError('Geometry changed')
    write_json(folder/'config.json',dict(c,status='complete',method='four_channel_full_frame_spatial_irl_v6',version=6,reward_scope=SCOPE,
        optimization='v6_four_channel_joint_shared_reward',dataset=str(args.dataset),search_run=str(out),
        experimental_conditions=CONDITIONS,action_model='per-channel binary composite likelihood',
        observation_weighting='equal channel / condition / participant / trial; capped duration within trial',
        geometry_sha256=digest(geo),dataset_manifest_sha256=digest(args.dataset/'manifest.json'),
        horizon=horizon,nominal_step_s=mdp['step_s'],device=args.device,plot_sigma_m=.6,plot_upsample=4,plot_dpi=300,
        plot_status='skipped' if args.skip_plots else 'pending',plot_3d_status='skipped' if args.skip_plots else 'pending',
        source_sha256={n:digest(HERE/n) for n in ['irl_frame_optimization_v6.py','Train_IRL_V6.py','spatial_reward_basis.py','Spatial_IRL_Rule_V6.py','irl_core_v3.py','irl_core.py','irl_data_v6.py']}))
    write_json(folder/'evaluation.json',dict(**evaluations,trained=True,states=len(mdp['centers']),
        selection_metric='spatial_macro_auc within best guidance_macro_nll + 0.005',test_used_for_selection=False,
        score_meaning='policy probabilities use the observed dynamic state; spatial_reward_auc and exported fields use spatial-only immediate reward contrast'))
    plot_best_model(folder,skip_plots=args.skip_plots)
    return folder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--train',action='store_true');p.add_argument('--device',default='cuda')
    p.add_argument('--budget-hours',type=float,default=1.);p.add_argument('--jobs',type=int,default=2);p.add_argument('--epochs',type=int,default=180);p.add_argument('--patience',type=int,default=16)
    p.add_argument('--horizon-seconds',type=float,default=2.);p.add_argument('--skip-plots',action='store_true')
    p.add_argument('--families',default='linear,spline,interaction,rbf32,rbf64')
    p.add_argument('--reuse-search',type=Path,help='Reuse validated candidates from a compatible earlier optimization search')
    p.add_argument('--seed-checks',type=int,default=2,help='Extra clustering seeds, diagnostic only')
    args=p.parse_args();args.dataset=args.dataset.resolve()
    if any(b not in ['linear','spline','interaction','rbf32','rbf64'] for b in args.families.split(',')):p.error('Unknown reward family')
    if args.patience<1 or args.seed_checks<0:p.error('Invalid stopping/seed settings')
    if args.jobs<1 or args.epochs<5 or args.budget_hours<=0 or args.horizon_seconds<=0:p.error('Invalid positive budgets')
    base=dict(lr=.02,l2=.001,deviation_ratio=1.,clusters=16,run_cap_s=10.,prior=.5,seed=20260926,horizon_seconds=args.horizon_seconds,gamma=.98,basis='linear',active_guidance=GUIDANCE)
    configs=[dict(base,clusters=k,basis=b,l2=l) for k,b,l in itertools.product([16,24],args.families.split(','),[.0001,.001,.01])]
    if not args.train:print(json.dumps(dict(trained=False,version=6,channels=GUIDANCE,configs=configs,test_used_for_selection=False),indent=2));return
    configure_device(args.device);validate_manifest_scope(json.loads((args.dataset/'manifest.json').read_text()))
    started=time.time();deadline=started+args.budget_hours*3600-180
    if deadline<=started:p.error('Budget must exceed three minutes')
    out=HERE/'Results_IRL'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_V6');out.mkdir()
    for name in ['Tasks','Trials','Caches','Frame_Caches']:(out/name).mkdir()
    write_json(out/'config.json',dict(status='running',version=6,active_guidance=GUIDANCE,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},test_used_for_selection=False))
    write_json(out/'dataset_audit.json',audit_dataset(args.dataset));write_json(out/'plan.json',dict(configs=configs,selection_rule='Max spatial macro AUC among models within 0.005 of best validation guidance macro NLL',seed=20260926,test_used=False))
    records={};caches={};frames={}
    if args.reuse_search:
        prior_config=json.loads((args.reuse_search/'config.json').read_text())
        if prior_config.get('version')!=6 or prior_config.get('active_guidance')!=GUIDANCE:raise ValueError('Only compatible V6 searches can be reused')
        for r in json.loads((args.reuse_search/'search_records.json').read_text()):
            if r.get('status')!='complete':continue
            cm=json.loads((Path(r['cache'])/'metadata.json').read_text())
            expected=state_code_hashes()
            if cm.get('state_code_sha256')!=expected:continue
            load_cache(r['cache'],args.dataset)
            fm=json.loads((Path(r['frames'])/'metadata.json').read_text())
            if fm['manifest_sha256']!=digest(args.dataset/'manifest.json'):raise ValueError('Reused data differs')
            if not (Path(r['output'])/'reward.pt').exists():raise ValueError('Reused checkpoint missing')
            if r['config'].get('active_guidance')!=GUIDANCE:raise ValueError('Reused channel scope differs')
            load_frames(r['frames'],'val')
            result=json.loads((Path(r['output'])/'result.json').read_text())
            if result.get('source_sha256')!=training_code_hashes():raise ValueError('Reused training implementation differs')
            r=dict(r,reused_from=str(args.reuse_search));records[r['id']]=r
            key=(r['config']['clusters'],r['config']['run_cap_s'],r['config']['seed'])
            caches[key]=Path(r['cache']);frames[key]=Path(r['frames'])
        write_json(out/'reused_candidates.json',list(records.values()))
    launch_batch(configs,args,out,deadline,records,caches,frames)
    best=rank(records)
    # Bounded, predefined local continuation: higher state resolution and horizon/LR checks.
    extra=[]
    for r in best[:2]:
        for k in [48]:extra.append(dict(r['config'],clusters=k))
        for h in [1.,5.]:extra.append(dict(r['config'],horizon_seconds=h))
        for lr in [.005,.05]:extra.append(dict(r['config'],lr=lr))
    write_json(out/'extension_plan.json',extra);launch_batch(extra,args,out,deadline,records,caches,frames)
    chosen=select(records)
    write_json(out/'selection.json',dict(candidate=chosen,frozen_at=datetime.now().isoformat(),test_used_for_selection=False,
        selection_rule='Max spatial macro AUC among models within 0.005 of best validation guidance macro NLL'))
    rows=[]
    for r in records.values():
        row=dict(id=r['id'],status=r['status'],**r['config'])
        if r['status']=='complete':
            row.update(score=r['score'],val_auc=r['validation']['auc'],spatial_macro_auc=r['validation']['spatial_macro_auc'],best_epoch=r['best_epoch'],epochs_run=r['epochs_run'])
            for g,d in r['validation']['by_guidance'].items():
                for metric in ['auc','nll','spatial_reward_auc']:row[g+'_'+metric]=d[metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'search_summary.csv',index=False)
    # Fixed selection; repeated seeds never enter selection and never use test.
    checks={}
    check_configs=[dict(chosen['config'],seed=20260926+11*i) for i in range(1,args.seed_checks+1)]
    write_json(out/'seed_plan.json',check_configs)
    launch_batch(check_configs,args,out,deadline,checks,caches,frames)
    write_json(out/'seed_checks.json',list(checks.values()))
    # Restore the main records after diagnostic-only workers wrote progress.
    write_json(out/'search_records.json',list(records.values()))
    folder=export(args,out,chosen)
    write_json(out/'completion.json',dict(status='complete',best_model=str(folder),total_seconds=time.time()-started,
        completed=sum(r['status']=='complete' for r in records.values()),failed=sum(r['status']!='complete' for r in records.values()),test_used_for_selection=False))
    print('BEST_MODEL='+str(folder));print('OUTPUT='+str(out))

if __name__=='__main__':main()
