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

from irl_data import (GUIDANCE,StateEncoder,iter_trials,audit_dataset,build_mdp,assign_states)
from irl_core import LinearReward,soft_policy,weighted_nll,expected_burden
from Spatial_IRL_Rule import reward_mask, SCOPE

HERE = Path(__file__).resolve().parent


def aggregate_weakness(df, keys):
    """Equal person means at the requested resolution, using original predictions."""
    individual=df.groupby(keys+['participant_id'],observed=True).agg(
        trap_probability=('trap_probability','mean'),trap_burden_s=('trap_burden_s','mean'),
        observed_trap=('trap_observed','mean'),supported_fraction=('supported','mean'),samples=('supported','size')).reset_index()
    return individual.groupby(keys,observed=True).agg(trap_probability=('trap_probability','mean'),
        trap_burden_s=('trap_burden_s','mean'),observed_trap=('observed_trap','mean'),
        supported_fraction=('supported_fraction','mean'),participants=('participant_id','nunique'),samples=('samples','sum')).reset_index()


def trigger_plots(out, config, dpi=300):
    """Separate CPU plot process; a plot failure never removes trained artifacts."""
    config['plot_status']='running'
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    try:
        subprocess.run([sys.executable,'-B',str(HERE/'Plot_IRL_Weakness.py'),
                        '--run',str(out),'--dpi',str(dpi)],check=True)
    except Exception as exc:
        config.update(plot_status='failed',plot_error=str(exc))
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        raise RuntimeError('Training artifacts are saved; plotting failed. Replay Plot_IRL_Weakness.py with --run and --force: '+str(out)) from exc
    config['plot_status']='complete'
    config.pop('plot_error',None)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


def metrics(y,p):
    if not len(y): return dict(n=0)
    return dict(n=len(y),nll=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),
        auc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
        balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)) if len(np.unique(y)) == 2 else None,
        observed_fraction=float(np.mean(y)),predicted_fraction=float(np.mean(p)))


def evaluate(dataset,mdp,policy,splits):
    ys,ps=[],[]; by_person={}; by_guidance={}; missing=0
    for rec,df in iter_trials(dataset,set(splits)):
        df=df.loc[df.valid.eq(1)]
        if df.empty: continue
        x=mdp['encoder'].transform(df)
        state,_=assign_states(df,x,mdp['contexts'],mdp['centers'])
        valid=state>=0; missing+=int((~valid).sum())
        y=df.action.to_numpy(int)[valid]; p=policy[state[valid],1]
        ys.extend(y); ps.extend(p)
        for group,key in [(by_person,rec['participant_id']),(by_guidance,rec['guidance'])]:
            entry=group.setdefault(key,[[],[]]); entry[0].extend(y);entry[1].extend(p)
    result=metrics(np.asarray(ys),np.asarray(ps));result['unknown_context_rows']=missing
    result['by_guidance']={k:metrics(np.asarray(v[0]),np.asarray(v[1])) for k,v in by_guidance.items()}
    person=[metrics(np.asarray(v[0]),np.asarray(v[1])) for v in by_person.values()]
    result['participant_macro_nll']=float(np.mean([m['nll'] for m in person])) if person else None
    return result


def export_predictions(dataset,mdp,policy,reward,q,prob,burden,output,min_support):
    rows=[]
    (output/'predictions').mkdir()
    for rec,df in iter_trials(dataset):
        df=df.loc[df.valid.eq(1)].copy()
        if df.empty: continue
        x=mdp['encoder'].transform(df)
        state,distance=assign_states(df,x,mdp['contexts'],mdp['centers'])
        supported_context=state>=0
        safe=np.maximum(state,0)
        support=mdp['transition_counts'][safe].min(1)
        supported=supported_context & (support>=min_support) & (distance<=mdp['radius'][safe]*1.5)
        df['state_id']=state; df['state_distance']=distance
        df['transition_support_min']=support; df['supported']=supported.astype(int)
        for name,values in dict(p_look=policy[:,1],reward_contrast=reward[:,1]-reward[:,0],
                q_contrast=q[:,1]-q[:,0],trap_probability=prob,trap_burden_s=burden).items():
            df[name]=np.where(supported_context,values[safe],np.nan)
        df.loc[~supported,['trap_probability','trap_burden_s']]=np.nan
        df.to_csv(output/'predictions'/(rec['trial_id']+'.csv.gz'),index=False,compression='gzip')
        rows.append(df[['participant_id','split','guidance','smoke','floor','stage','x','z','elapsed_s','trap_probability','trap_burden_s','trap_observed','supported']])
    if not rows:return
    df=pd.concat(rows,ignore_index=True)
    df['cell_x']=np.floor(df.x/.4)*.4+.2;df['cell_z']=np.floor(df.z/.4)*.4+.2
    df['elapsed_bin_s']=np.floor(df.elapsed_s/10)*10
    keys=['split','guidance','smoke','floor','stage','cell_x','cell_z','elapsed_bin_s']
    # Equal person means per visited cell; no smoothing or interpolation.
    field=aggregate_weakness(df,keys)
    field.to_csv(output/'spatiotemporal_weakness_field.csv',index=False)
    overview=aggregate_weakness(df,['split','guidance','smoke','floor','cell_x','cell_z'])
    overview.to_csv(output/'spatial_weakness_overview.csv',index=False)


def train(args):
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
    config.update(status='initializing',method='finite_horizon_soft_policy_likelihood_IRL',reward_scope=SCOPE,
        dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest())
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    geometry=Path(DEFAULT_GEOMETRY_PATH).resolve()
    source_manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    config['geometry_sha256']=source_manifest['source_hashes'][str(geometry)]
    if hashlib.sha256(geometry.read_bytes()).hexdigest()!=config['geometry_sha256']:
        raise ValueError('Geometry changed since data construction')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    mdp=build_mdp(args.dataset,args.clusters,args.seed)
    device=torch.device(args.device)
    tensors={k:torch.as_tensor(mdp[k],dtype=torch.float32,device=device) for k in ['centers','transitions','counts','event_cost']}
    spatial_features=tensors['centers']*torch.as_tensor(reward_mask(mdp['encoder'].names),device=device)
    config['reward_feature_names']=[name for name,keep in zip(mdp['encoder'].names,reward_mask(mdp['encoder'].names)) if keep]
    g=torch.as_tensor(mdp['guidance_index'],dtype=torch.long,device=device)
    model=LinearReward(tensors['centers'].shape[1]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf');best_state=None;history=[];stale=0;best_epoch=None
    for epoch in trange(args.epochs,desc='IRL reward inversion'):
        optimizer.zero_grad()
        reward=model(spatial_features,g)
        logp,_=soft_policy(reward,tensors['transitions'],args.horizon,args.gamma,mdp['terminal_index'])
        loss=weighted_nll(logp,tensors['counts'])+args.l2*(model.shared.square().mean()+model.deviation.square().mean())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite IRL objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        with torch.no_grad():
            logp,_=soft_policy(model(spatial_features,g),tensors['transitions'],args.horizon,args.gamma,mdp['terminal_index'])
            policy=logp.exp().cpu().numpy()
        val=evaluate(args.dataset,mdp,policy,['val'])
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
        logp,q=soft_policy(reward,tensors['transitions'],args.horizon,args.gamma,mdp['terminal_index'])
        policy=logp.exp()
        if args.trajectory_diagnostics:
            prob,burden=expected_burden(policy,tensors['transitions'],tensors['event_cost'],args.horizon,mdp['step_s'])
    policy_np=policy.cpu().numpy()
    report={split:evaluate(args.dataset,mdp,policy_np,[split]) for split in ['train','val','test']}
    report.update(states=len(mdp['centers']),train_rows=mdp['train_rows'],
        unsupported_state_actions=int((mdp['transition_counts']<args.min_support).sum()),
        selected_val_macro_nll=best,trained=True,
        stopping=dict(reason='validation_patience' if stale>=args.patience else 'max_epochs',
                      epochs_run=len(history),best_epoch=best_epoch,patience=args.patience,min_delta=1e-5,
                      monitor='val_participant_macro_nll',mode='min'))
    # Serialize numerical state only; no joblib/pickle required for encoding/clusters.
    np.savez_compressed(out/'mdp.npz',**{k:mdp[k] for k in ['centers','transitions','raw_counts','transition_counts','guidance_index','event_cost','radius','terminal_index']})
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
    if args.trajectory_diagnostics:
        export_predictions(args.dataset,mdp,policy_np,reward.cpu().numpy(),q.cpu().numpy(),prob.cpu().numpy(),burden.cpu().numpy(),out,args.min_support)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
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
    p.add_argument('--horizon',type=int,default=25)
    p.add_argument('--gamma',type=float,default=.98)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--l2',type=float,default=.01)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--min-support',type=int,default=5)
    p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--plot-dpi',type=int,default=300,help='Automatic post-training figure resolution')
    p.add_argument('--trajectory-diagnostics',action='store_true',help='Optional diagnostic CSVs only; never used for global maps')
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
