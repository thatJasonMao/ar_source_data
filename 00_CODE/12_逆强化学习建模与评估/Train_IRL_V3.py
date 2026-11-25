"""A5 reward inversion. Default invocation only audits data; --train is explicit."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import hashlib
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss,balanced_accuracy_score
from tqdm import trange

from irl_data_v3 import (GUIDANCE,StateEncoder,iter_trials,audit_dataset,build_mdp,assign_states)
from irl_core_v3 import LinearReward,soft_policy,weighted_nll
from Spatial_IRL_Rule import reward_mask, SCOPE

HERE = Path(__file__).resolve().parent


def trigger_plots(out, config, dpi=300):
    """Separate CPU plot process; a plot failure never removes trained artifacts."""
    config['plot_status']='running'
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    try:
        subprocess.run([sys.executable,'-B',str(HERE/'Plot_IRL_V3.py'),
                        '--run',str(out),'--dpi',str(dpi),'--sigma-m',str(config['plot_sigma_m']),'--upsample',str(config['plot_upsample'])],check=True)
    except Exception as exc:
        config.update(plot_status='failed',plot_error=str(exc))
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        raise RuntimeError('Training artifacts are saved; plotting failed. Replay Plot_IRL_V3.py with --run and --force: '+str(out)) from exc
    config['plot_status']='complete'
    config.pop('plot_error',None)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


def metrics(y,p):
    if not len(y): return dict(n=0)
    return dict(n=len(y),nll=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),
        auc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
        balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)) if len(np.unique(y)) == 2 else None,
        observed_fraction=float(np.mean(y)),predicted_fraction=float(np.mean(p)))


def cache_evaluation(dataset, mdp):
    """Encode all observed frames once; no repeated gzip/feature work per epoch."""
    result={s:[] for s in ['train','val','test']}
    for rec,df in iter_trials(dataset):
        df=df.loc[df.valid.eq(1)]
        if df.empty: continue
        state,_=assign_states(df,mdp['encoder'].transform(df),mdp['contexts'],mdp['centers'])
        if (state<0).any(): raise ValueError('Unseen V3 state context')
        result[rec['split']].append((rec['participant_id'],rec['guidance'],df.action.to_numpy(int),state))
    return result

def evaluate_cached(cache,policy,split):
    ys=[];ps=[];people={};guides={}
    for pid,guidance,y,state in cache[split]:
        p=policy[state,1];ys.append(y);ps.append(p)
        for target,key in [(people,pid),(guides,guidance)]:
            entry=target.setdefault(key,[[],[]]);entry[0].append(y);entry[1].append(p)
    if not ys: raise ValueError('No supported observations for '+split)
    report=metrics(np.concatenate(ys),np.concatenate(ps))
    report['unknown_context_rows']=0
    report['by_guidance']={g:metrics(np.concatenate(v[0]),np.concatenate(v[1])) for g,v in guides.items()}
    report['participant_macro_nll']=float(np.mean([metrics(np.concatenate(v[0]),np.concatenate(v[1]))['nll'] for v in people.values()]))
    return report


def train(args):
    source_manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    from irl_data_v3 import validate_manifest_scope
    validate_manifest_scope(source_manifest)
    nominal=float(source_manifest['config']['step_s'])
    if not np.isfinite(args.horizon_seconds) or args.horizon_seconds<=0: raise ValueError('Invalid horizon seconds')
    if args.horizon is None: args.horizon=max(1,int(np.ceil(args.horizon_seconds/nominal)))
    if not np.isfinite(args.plot_sigma_m) or args.plot_sigma_m<=0 or not 1<=args.plot_upsample<=16: raise ValueError('Invalid display smoothing')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    if args.trajectory_diagnostics: raise ValueError('V3 exports spatial demand only; trajectory diagnostics are not supported')
    if args.epochs<1 or args.clusters<1 or args.horizon<1 or not 0<=args.gamma<=1 or args.plot_dpi<50:
        raise ValueError('Invalid training configuration')
    if not np.isfinite(args.lr) or args.lr<=0 or not np.isfinite(args.l2) or args.l2<0 or args.threads<1 or args.patience<1 or args.min_support<1:
        raise ValueError('Invalid learning rate, regularization, threads, patience or support')
    audit_dataset(args.dataset,verify=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    out=HERE/'Results_IRL'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(status='initializing',method='full_frame_spatial_IRL_v3',sampling_mode='all_unique_subject_timestamps',nominal_step_s=nominal,planning_window_approx_s=args.horizon*nominal,discount_reference_s=.2,reward_scope=SCOPE,
        dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest())
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    geometry=Path(DEFAULT_GEOMETRY_PATH).resolve()
    source_manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    config['geometry_sha256']=source_manifest['source_hashes'][str(geometry)]
    if hashlib.sha256(geometry.read_bytes()).hexdigest()!=config['geometry_sha256']:
        raise ValueError('Geometry changed since data construction')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    mdp=build_mdp(args.dataset,args.clusters,args.seed)
    observations=cache_evaluation(args.dataset,mdp)
    device=torch.device(args.device)
    tensors={k:torch.as_tensor(mdp[k],dtype=torch.float32,device=device) for k in ['centers','transitions','counts','event_cost','mean_dt']}
    spatial_features=tensors['centers']*torch.as_tensor(reward_mask(mdp['encoder'].names),device=device)
    config['reward_feature_names']=[name for name,keep in zip(mdp['encoder'].names,reward_mask(mdp['encoder'].names)) if keep]
    g=torch.as_tensor(mdp['guidance_index'],dtype=torch.long,device=device)
    model=LinearReward(tensors['centers'].shape[1]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf');best_state=None;history=[];stale=0;best_epoch=None
    for epoch in trange(args.epochs,desc='IRL reward inversion'):
        optimizer.zero_grad()
        reward=model(spatial_features,g)
        logp,_=soft_policy(reward,tensors['transitions'],args.horizon,args.gamma,mdp['terminal_index'],mean_dt=tensors['mean_dt'])
        loss=weighted_nll(logp,tensors['counts'])+args.l2*(model.shared.square().mean()+model.deviation.square().mean())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite IRL objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        with torch.no_grad():
            logp,_=soft_policy(model(spatial_features,g),tensors['transitions'],args.horizon,args.gamma,mdp['terminal_index'],mean_dt=tensors['mean_dt'])
            policy=logp.exp().cpu().numpy()
        val=evaluate_cached(observations,policy,'val')
        score=val['participant_macro_nll']
        if score is None or not np.isfinite(score):raise ValueError('No finite validation score')
        history.append(dict(epoch=epoch+1,loss=float(loss.detach()),val_macro_nll=score))
        if score<best-1e-5:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch+1
        else:stale+=1
        if stale>=args.patience:break
    model.load_state_dict(best_state)
    with torch.no_grad():
        reward=model(spatial_features,g)
        logp,q=soft_policy(reward,tensors['transitions'],args.horizon,args.gamma,mdp['terminal_index'],mean_dt=tensors['mean_dt'])
        policy=logp.exp()
    policy_np=policy.cpu().numpy()
    report={split:evaluate_cached(observations,policy_np,split) for split in ['train','val','test']}
    report.update(states=len(mdp['centers']),train_rows=mdp['train_rows'],
        unsupported_state_actions=int((mdp['transition_counts']<args.min_support).sum()),
        selected_val_macro_nll=best,trained=True,
        stopping=dict(reason='validation_patience' if stale>=args.patience else 'max_epochs',
                      epochs_run=len(history),best_epoch=best_epoch,patience=args.patience,min_delta=1e-5,
                      monitor='val_participant_macro_nll',mode='min'))
    # Serialize numerical state only; no joblib/pickle required for encoding/clusters.
    np.savez_compressed(out/'mdp.npz',**{k:mdp[k] for k in ['mean_dt','centers','transitions','raw_counts','transition_counts','guidance_index','event_cost','radius','terminal_index']})
    (out/'encoder.json').write_text(json.dumps(dict(encoder=mdp['encoder'].as_dict(),contexts=mdp['contexts'],step_s=mdp['step_s']),indent=2),encoding='utf-8')
    torch.save(best_state,out/'reward.pt')
    deviations=best_state['deviation'].numpy();deviations-=deviations.mean(0,keepdims=True)
    rules=[]
    for i,name in enumerate(GUIDANCE):
        for feature,weight in zip(mdp['encoder'].names,best_state['shared'].numpy()+deviations[i]):
            if feature in config['reward_feature_names']:
                rules.append(dict(guidance=name,feature=feature,contrast_weight=float(weight)))
    pd.DataFrame(rules).to_csv(out/'reward_rules.csv',index=False)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    # Default scientific output is the global spatial rule, not trajectory fields.
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    config['source_sha256']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
        ['Train_IRL_V3.py','irl_data_v3.py','irl_core_v3.py','Spatial_IRL_Rule.py','Plot_IRL_V3.py','gaussian_field_v3.py']}
    config['status']='complete';(out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('MODEL_OUTPUT='+str(out),flush=True)
    trigger_plots(out,config,args.plot_dpi)
    print('OUTPUT='+str(out))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--train',action='store_true',help='Explicit authorization to fit IRL rewards')
    p.add_argument('--check-data',action='store_true',help='Audit only; cannot combine with --train')
    p.add_argument('--report',type=Path)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--clusters',type=int,default=24)
    p.add_argument('--horizon',type=int,default=None,help='Optional explicit native-frame planning steps')
    p.add_argument('--horizon-seconds',type=float,default=5.,help='Convert to steps using train nominal frame interval')
    p.add_argument('--plot-sigma-m',type=float,default=.6)
    p.add_argument('--plot-upsample',type=int,default=4)
    p.add_argument('--gamma',type=float,default=.98,help='Discount per 0.2 s; automatically scaled by state-action duration')
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--l2',type=float,default=.01)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--min-support',type=int,default=5)
    p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--plot-dpi',type=int,default=300,help='Automatic post-training figure resolution')
    p.add_argument('--trajectory-diagnostics',action='store_true',help='Reserved V1 compatibility flag; V3 rejects this option (spatial output only)')
    args=p.parse_args()
    if args.train and args.check_data:p.error('--train and --check-data are mutually exclusive')
    if args.report and args.report.exists():p.error('Report already exists; choose a new path')
    if args.train:
        train(args)
    else:
        report=audit_dataset(args.dataset)
        print(json.dumps(report,indent=2))
        if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
