"""Budgeted V3-based IRL search. Without --train: inspect the plan, never fit."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from irl_data_v3 import audit_dataset, iter_trials, GUIDANCE, validate_manifest_scope
from irl_core_v3 import LinearReward,soft_policy
from Spatial_IRL_Rule import reward_mask,SCOPE
from irl_search_v4 import (METHOD,write_json,digest,cache_key,config_id,coarse_configs,explore_configs,
    rank_records,load_cache,pack_observations,run_duration_table,summarize_policy,baseline_reports)

HERE=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--train',action='store_true');p.add_argument('--check-data',action='store_true')
    p.add_argument('--report',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--devices',help='Comma separated devices, e.g. cuda:0,cuda:1')
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--budget-hours',type=float,default=2.)
    p.add_argument('--reserve-minutes',type=float,default=10.,help='Reserve part of budget for final evaluation/plots')
    p.add_argument('--max-trials',type=int,default=64)
    p.add_argument('--screen-epochs',type=int,default=100)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--patience',type=int,default=40)
    p.add_argument('--lr-patience',type=int,default=8)
    p.add_argument('--min-delta',type=float,default=1e-6)
    p.add_argument('--min-epochs',type=int,default=30)
    p.add_argument('--horizon-seconds',type=float,default=2.)
    p.add_argument('--gamma',type=float,default=.98)
    p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--finalists',type=int,default=3)
    p.add_argument('--refit-seeds',type=int,default=3,help='Total clustering seeds for winner checks, including original')
    p.add_argument('--include-lbfgs',action='store_true')
    p.add_argument('--selection-metric',choices=['participant_macro_nll','guidance_macro_nll','full_participant_macro_nll'],default='participant_macro_nll')
    p.add_argument('--plot-dpi',type=int,default=300)
    p.add_argument('--plot-sigma-m',type=float,default=.6)
    p.add_argument('--plot-upsample',type=int,default=4)
    p.add_argument('--skip-plots',action='store_true',help='Optional numerical-only delivery')
    return p


def validate_args(args):
    for name in ['jobs','max_trials','screen_epochs','epochs','patience','lr_patience','min_epochs','threads','finalists','refit_seeds']:
        if getattr(args,name)<1:raise ValueError(name+' must be positive')
    for name in ['budget_hours','horizon_seconds','min_delta','plot_sigma_m']:
        value=getattr(args,name)
        if not np.isfinite(value) or value<=0:raise ValueError(name+' must be finite and positive')
    if args.epochs<args.screen_epochs:raise ValueError('--epochs must be >= --screen-epochs')
    if not np.isfinite(args.reserve_minutes) or args.reserve_minutes<0 or args.reserve_minutes*60>=args.budget_hours*3600:raise ValueError('Reserve must be smaller than total budget')
    if not 0<=args.gamma<=1 or args.plot_dpi<50 or not 1<=args.plot_upsample<=16:raise ValueError('Invalid discount/display settings')
    if args.train and args.check_data:raise ValueError('--train and --check-data are mutually exclusive')
    if args.report and args.report.exists():raise FileExistsError(args.report)


def devices_for(args,validate_cuda=False):
    devices=[x.strip() for x in args.devices.split(',')] if args.devices else [args.device]
    if not devices or any(not d for d in devices):raise ValueError('Empty device list')
    for d in devices:
        device=torch.device(d)
        if validate_cuda and device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
            index=0 if device.index is None else device.index
            if index>=torch.cuda.device_count():raise ValueError('Unavailable GPU '+d)
    return devices


def terminate_process(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)


class Search:
    def __init__(self,args,out,deadline):
        self.args=args;self.out=Path(out);self.deadline=deadline
        self.records={};self.cache_status={};self.active={};self.task_number=0
        self.data_hash=digest(args.dataset/'manifest.json');self.runs_hash=digest(args.dataset/'action_runs.csv')
        self.devices=devices_for(args,True)
        for name in ['Caches','Trials','Tasks']:(self.out/name).mkdir()

    def cache_for(self,c):
        return self.out/'Caches'/cache_key(c,self.data_hash,self.runs_hash)

    def launch(self,kind,c,phase,epochs,resume=False,selection_excluded=False):
        self.task_number+=1;cid=config_id(c);folder=self.out/'Trials'/cid
        task_out=folder if kind=='train' else self.out/'Tasks'/f'prepare_{self.task_number:04d}'
        task_out.mkdir(parents=True,exist_ok=True)
        if kind=='train' and (task_out/'STOP').exists():(task_out/'STOP').unlink()
        cache=self.cache_for(c)
        # Available slots have stable device assignments; supports multiple GPUs.
        used={v['slot'] for v in self.active.values()};slot=next(i for i in range(self.args.jobs) if i not in used)
        device=self.devices[slot%len(self.devices)]
        job=dict(kind=kind,id=cid,config=c,dataset=str(self.args.dataset),cache=str(cache),output=str(task_out),
            deadline=self.deadline,device=device,threads=self.args.threads,epochs=epochs,resume=resume,
            patience=self.args.patience,lr_patience=self.args.lr_patience,min_delta=self.args.min_delta,
            min_epochs=min(self.args.min_epochs,epochs),selection_metric=self.args.selection_metric)
        jobpath=self.out/'Tasks'/f'job_{self.task_number:04d}.json';write_json(jobpath,job)
        logpath=task_out/(f'{phase}_{self.task_number:04d}.log');log=logpath.open('w',encoding='utf-8')
        env=os.environ.copy()
        for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]=str(self.args.threads)
        process=subprocess.Popen([sys.executable,'-B',str(HERE/'IRL_V4_Worker.py'),'--job',str(jobpath)],stdout=log,stderr=subprocess.STDOUT,env=env)
        self.active[process.pid]=dict(process=process,log=log,kind=kind,config=c,id=cid,cache=cache,
            output=task_out,phase=phase,slot=slot,selection_excluded=selection_excluded,resume=resume,started=time.time())
        if kind=='prepare':self.cache_status[str(cache)]='building'
        else:
            self.records[cid]=dict(id=cid,config=c,status='running',phase=phase,output=str(folder),
                selection_excluded=selection_excluded,score=None,cache=str(cache))
        print(f'{phase}: {kind} {cid} on {device}',flush=True)

    def collect(self,force=False):
        changed=False
        for pid,task in list(self.active.items()):
            process=task['process']
            if force:terminate_process(process)
            if process.poll() is None:continue
            task['log'].close();changed=True
            if task['kind']=='prepare':
                ready=(task['cache']/'metadata.json').exists()
                self.cache_status[str(task['cache'])]='ready' if ready and process.returncode==0 else 'failed'
            else:
                folder=task['output'];result=folder/'result.json'
                record=dict(self.records[task['id']])
                if result.exists():
                    result_data=json.loads(result.read_text(encoding='utf-8'))
                    # A saved best checkpoint remains usable if a budget interrupts later work.
                    if (folder/'reward.pt').exists() and (process.returncode==0 or force or task['resume']):
                        record.update(result_data,status='complete',phase=task['phase'],selection_excluded=task['selection_excluded'])
                        if force:record['reason']='time_budget_interruption'
                        elif process.returncode!=0:record['reason']='resume_failed_previous_checkpoint_retained'
                    else:record.update(status='failed',returncode=process.returncode)
                else:record.update(status='budget_stopped' if force else 'failed',returncode=process.returncode)
                if (folder/'error.json').exists():record['error']=json.loads((folder/'error.json').read_text(encoding='utf-8'))
                self.records[task['id']]=record
            del self.active[pid]
        if changed:self.summarize()

    def summarize(self):
        records=list(self.records.values());write_json(self.out/'search_records.json',records)
        rows=[]
        for r in records:
            row=dict(id=r['id'],phase=r['phase'],status=r['status'],selection_excluded=r.get('selection_excluded',False),**r['config'])
            row.update(score=r.get('score'),best_epoch=r.get('best_epoch'),epochs_run=r.get('epochs_run'),reason=r.get('reason'))
            if r.get('validation'):
                v=r['validation']
                for k in ['nll','auc','brier','participant_macro_nll','guidance_macro_nll']:row['val_'+k]=v[k]
                for g,d in v['by_guidance'].items():
                    for k in ['nll','auc','brier','observed_fraction','predicted_fraction']:row[g+'_'+k]=d[k]
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.out/'search_summary.csv',index=False)

    def phase(self,configs,name,epochs,phase_deadline,resume=False,selection_excluded=False):
        pending=[]
        new_slots=self.args.max_trials-len(self.records)
        for c in configs:
            cid=config_id(c)
            if cid in self.records and not resume:continue
            if cid not in self.records:
                if new_slots<=0:break
                new_slots-=1
            pending.append(c)
        bar=tqdm(total=len(pending),desc=name);completed=0
        try:
            while (pending or self.active) and time.time()<min(phase_deadline,self.deadline):
                self.collect()
                for c in list(pending):
                    cid=config_id(c);cache=self.cache_for(c);status=self.cache_status.get(str(cache))
                    if status=='failed':
                        self.records[cid]=dict(id=cid,config=c,phase=name,status='cache_failed',score=None,selection_excluded=selection_excluded)
                        pending.remove(c);completed+=1;bar.update(1);continue
                    if len(self.active)>=self.args.jobs:break
                    if status is None:
                        self.launch('prepare',c,name,epochs,selection_excluded=selection_excluded)
                    elif status=='ready':
                        do_resume=resume and (self.out/'Trials'/cid/'checkpoint.pt').exists()
                        self.launch('train',c,name,epochs,resume=do_resume,selection_excluded=selection_excluded)
                        pending.remove(c)
                    # Other candidates sharing a building cache wait, never duplicate-fit it.
                finished=sum(r.get('phase')==name and r.get('status')!='running' for r in self.records.values())
                if finished>completed:bar.update(finished-completed);completed=finished
                time.sleep(.5)
            if self.active:
                # Phase resource allocation is bounded. Workers receive a graceful stop window.
                for task in self.active.values():
                    (task['output']/'STOP').touch()
                limit=time.time()+15
                while self.active and time.time()<limit:self.collect();time.sleep(.25)
                if self.active:self.collect(force=True)
            self.summarize()
        finally:bar.close()

    def ranked(self):
        return rank_records([r for r in self.records.values() if not r.get('selection_excluded')])

    def stop(self):
        self.collect(force=True)


def full_diagnostics(dataset):
    rows=[]
    for rec,df in iter_trials(dataset,{'train','val'}):
        if rec['guidance']!='Full':continue
        df=df.loc[df.valid.eq(1)]
        for stage,p in df.groupby('stage'):
            types=p.gaze_types.fillna('')
            rows.append(dict(split=rec['split'],smoke=rec['smoke'],stage=stage,n=len(p),any=int(p.action.sum()),
                arrow=int(types.str.contains('LightBlanket',regex=False).sum()),
                minimap=int(types.str.contains('MiniMap',regex=False).sum()),text=int(types.str.contains('Text',regex=False).sum())))
    if not rows:return dict(rows=[],interpretation='No supported Full records')
    table=pd.DataFrame(rows).groupby(['split','smoke','stage'],as_index=False).sum()
    for k in ['any','arrow','minimap','text']:table[k+'_fraction']=table[k]/table.n
    return dict(rows=table.to_dict('records'),input_splits=['train','val'],
        interpretation='Descriptive frame proportions; Full action is the union of three channels. Not overload or causal evidence.')


def export_best(args,out,selected):
    """Selection must be frozen before test observations are loaded here."""
    out=Path(out);folder=out/'Best_Model';folder.mkdir()
    cache=Path(selected['cache']);mdp,obs,keys=load_cache(cache,args.dataset)
    c=selected['config'];horizon=max(1,int(math.ceil(c['horizon_seconds']/mdp['step_s'])))
    model=LinearReward(len(mdp['encoder'].names))
    state=torch.load(Path(selected['output'])/'reward.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state);torch.set_num_threads(args.threads)
    mask=reward_mask(mdp['encoder'].names)
    with torch.no_grad():
        lp,_=soft_policy(model(torch.as_tensor(mdp['centers']*mask),torch.as_tensor(mdp['guidance_index'],dtype=torch.long)),
            torch.as_tensor(mdp['transitions']),horizon,c['gamma'],mdp['terminal_index'],mean_dt=torch.as_tensor(mdp['mean_dt']))
        policy=lp.exp().numpy()
    test,test_keys=pack_observations(args.dataset,mdp,['test'],c['run_cap_s'],run_duration_table(args.dataset))
    if test_keys!=keys:raise ValueError('Test context encoding differs')
    combined=dict(obs,**test)
    evaluation={sp:summarize_policy(combined[sp],policy,mdp['guidance_index']) for sp in ['train','val','test']}
    evaluation.update(states=len(mdp['centers']),train_rows=mdp['train_rows'],trained=True,
        selected_candidate=selected['id'],selection_metric=args.selection_metric,
        baselines={sp:baseline_reports(obs['train'],combined[sp],keys) for sp in ['val','test']},
        stopping=dict(reason=selected['reason'],best_epoch=selected['best_epoch'],epochs_run=selected['epochs_run']))
    torch.save(state,folder/'reward.pt')
    for name in ['encoder.json','mdp.npz']:shutil.copy2(cache/name,folder/name)
    history=Path(selected['output'])/'training_history.csv'
    if history.exists():shutil.copy2(history,folder/history.name)
    deviations=state['deviation'].numpy();weights=state['shared'].numpy()[None,:]+deviations-deviations.mean(0,keepdims=True)
    rows=[dict(guidance=g,feature=name,contrast_weight=float(weights[i,j])) for i,g in enumerate(GUIDANCE)
          for j,name in enumerate(mdp['encoder'].names) if mask[j]]
    pd.DataFrame(rows).to_csv(folder/'reward_rules.csv',index=False)
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    geometry=Path(DEFAULT_GEOMETRY_PATH).resolve();geometry_hash=manifest['source_hashes'][str(geometry)]
    if digest(geometry)!=geometry_hash:raise ValueError('Geometry changed since dataset build')
    config=dict(c,status='complete',method=METHOD,reward_scope=SCOPE,dataset=str(args.dataset),
        search_run=str(out),selected_candidate=selected['id'],horizon=horizon,
        nominal_step_s=mdp['step_s'],planning_window_approx_s=horizon*mdp['step_s'],discount_reference_s=.2,
        geometry_sha256=geometry_hash,dataset_manifest_sha256=digest(args.dataset/'manifest.json'),
        plot_sigma_m=args.plot_sigma_m,plot_upsample=args.plot_upsample,plot_dpi=args.plot_dpi,
        plot_status='skipped' if args.skip_plots else 'pending',device=selected.get('device'),
        source_sha256={n:digest(HERE/n) for n in ['Train_IRL_V4.py','IRL_V4_Worker.py','irl_search_v4.py','irl_data_v4.py','irl_core_v3.py','Spatial_IRL_Rule.py','Plot_IRL_V4.py','gaussian_field_v3.py']})
    write_json(folder/'config.json',config);write_json(folder/'evaluation.json',evaluation)
    return folder


def seed_stability(out,folder,selected,checks):
    """No additional maps: compare repeated-seed rules on the one final raw grid."""
    from Spatial_IRL_Rule import SpatialIRLRule
    from scipy.stats import spearmanr
    field=Path(folder)/'Results_Global_Spatial_Field_V4/global_spatial_rule_field.csv.gz'
    if not field.exists():
        write_json(Path(out)/'seed_stability.json',dict(status='not_computed',reason='No final raw field available'))
        return
    reference=pd.read_csv(field)
    signature={k:v for k,v in selected['config'].items() if k!='seed'}
    rows=[]
    for record in checks:
        if {k:v for k,v in record['config'].items() if k!='seed'}!=signature:continue
        rule=SpatialIRLRule(record['output'])
        for guidance,part in reference.groupby('guidance'):
            a=part.relative_information_utility.to_numpy();b=rule.query_features(part,guidance)
            count=max(1,int(np.ceil(.1*len(a))))
            ia=set(np.argsort(a)[-count:]);ib=set(np.argsort(b)[-count:])
            rho=spearmanr(a,b).statistic
            rows.append(dict(seed=record['config']['seed'],guidance=guidance,
                rank_correlation=float(rho) if np.isfinite(rho) else None,
                top10_overlap=len(ia&ib)/count,mean_absolute_difference=float(np.mean(np.abs(a-b)))))
    write_json(Path(out)/'seed_stability.json',dict(status='complete' if rows else 'no_matching_seed_checks',rows=rows,
        reference=selected['id'],grid='unsmoothed final 0.4 m global grid',test_used=False,
        limitation='Seed checks change train-only clustering, not participant splits; stable fields do not establish causal demand'))


def run_search(args):
    started=time.time();total_deadline=started+args.budget_hours*3600
    search_deadline=total_deadline-args.reserve_minutes*60
    devices_for(args,True)
    out=HERE/'Results_IRL'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(method=METHOD,status='auditing',budget_started_at=datetime.now().isoformat(),test_used_for_selection=False)
    write_json(out/'config.json',config)
    audit=audit_dataset(args.dataset,verify=True);write_json(out/'dataset_audit.json',audit)
    write_json(out/'full_channel_diagnostics.json',full_diagnostics(args.dataset))
    config['status']='searching';write_json(out/'config.json',config)
    search=Search(args,out,search_deadline)
    try:
        available=max(0.,search_deadline-time.time());phase_start=time.time()
        coarse=coarse_configs(args);write_json(out/'coarse_plan.json',coarse)
        search.phase(coarse,'coarse',args.screen_epochs,phase_start+available*.35)
        ranked=search.ranked()
        if not ranked:raise RuntimeError('No validated candidate completed; inspect Tasks/Trials logs and increase budget')
        explore=explore_configs([r['config'] for r in ranked[:2]])
        if args.include_lbfgs:
            explore += [dict(ranked[0]['config'],optimizer='lbfgs',lr=lr) for lr in [.5,1.]]
        write_json(out/'explore_plan.json',explore)
        search.phase(explore,'explore',args.screen_epochs,phase_start+available*.65)
        ranked=search.ranked();finalists=ranked[:args.finalists]
        write_json(out/'promotions.json',[r['id'] for r in finalists])
        search.phase([r['config'] for r in finalists],'refine',args.epochs,phase_start+available*.85,resume=True)
        ranked=search.ranked()
        # Seeds change clustering only; fixed participant splits remain unchanged.
        repeats=[]
        for r in ranked[:2]:
            for i in range(1,args.refit_seeds):repeats.append(dict(r['config'],seed=args.seed+11*i))
        search.phase(repeats,'seed_check',args.epochs,search_deadline,selection_excluded=True)
        ranked=search.ranked();selected=ranked[0]
        write_json(out/'selection.json',dict(candidate=selected,selection_metric=args.selection_metric,
            frozen_at=datetime.now().isoformat(),test_used_for_selection=False,
            seed_checks_used_for_selection=False,selection_rule='Minimum selected validation NLL; Brier then id for exact ties'))
        folder=export_best(args,out,selected)
        # Seed checks are diagnostic; do not cherry-pick the seed with the highest score.
        checks=[r for r in search.records.values() if r.get('selection_excluded') and r.get('status')=='complete']
        write_json(out/'seed_checks.json',checks)
        config.update(status='complete',best_model=str(folder),selected_candidate=selected['id'],
            search_seconds=time.time()-started,candidates_completed=len(search.ranked()),plot_status='skipped' if args.skip_plots else 'pending')
        write_json(out/'config.json',config)
        print('BEST_MODEL='+str(folder),flush=True)
        if not args.skip_plots:
            try:
                subprocess.run([sys.executable,'-B',str(HERE/'Plot_IRL_V4.py'),'--run',str(folder),'--dpi',str(args.plot_dpi)],check=True)
                config['plot_status']='complete'
            except Exception as exc:
                config['plot_status']='failed';config['plot_error']=str(exc)
                best=json.loads((folder/'config.json').read_text(encoding='utf-8'));best.update(plot_status='failed',plot_error=str(exc));write_json(folder/'config.json',best)
                raise RuntimeError('Model saved; plotting failed. Replay Plot_IRL_V4.py --run Best_Model --force') from exc
            finally:write_json(out/'config.json',config)
        config['total_seconds']=time.time()-started;write_json(out/'config.json',config)
        seed_stability(out,folder,selected,checks)
        print('OUTPUT='+str(out),flush=True)
        return out
    except BaseException as exc:
        search.stop();config.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        write_json(out/'config.json',config);raise


def main():
    p=parser();args=p.parse_args()
    try:validate_args(args)
    except (ValueError,FileExistsError) as exc:p.error(str(exc))
    args.dataset=args.dataset.resolve()
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'));validate_manifest_scope(manifest)
    if args.train:run_search(args)
    elif args.check_data:
        report=audit_dataset(args.dataset)
        if args.report:write_json(args.report,report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    else:
        plan=dict(trained=False,budget_hours=args.budget_hours,jobs=args.jobs,devices=devices_for(args),
            dataset=str(args.dataset),horizon_seconds=args.horizon_seconds,selection_metric=args.selection_metric,
            coarse_configs=coarse_configs(args),next_phases=['cap/clusters/deviation regularization','promote finalists','clustering seed checks'],
            test_used_for_selection=False)
        if args.report:write_json(args.report,plan)
        print(json.dumps(plan,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
