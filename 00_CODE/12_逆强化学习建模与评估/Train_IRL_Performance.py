"""Independent IRL with known slowdown penalties and global performance fields.

python -B Train_IRL_Performance.py --dataset <existing v2 dataset> --train --device cuda
Without --train this performs data checks only. Existing Train_IRL.py is unchanged.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import trange

from irl_data import audit_dataset, assign_states, GUIDANCE
from irl_core import LinearReward, soft_policy, weighted_nll
from Spatial_IRL_Rule import reward_mask
from Train_IRL import metrics
from irl_performance import METHOD, load_prepared, build_performance_mdp, cost_to_go

HERE=Path(__file__).resolve().parent


def cache_observations(parts,mdp):
    cache={}
    for split,frame in parts.items():
        x=mdp['encoder'].transform(frame)
        states,distance=assign_states(frame,x,mdp['contexts'],mdp['centers'])
        if np.any(states<0):raise ValueError('Unseen behavior context in '+split)
        cache[split]=dict(frame=frame,x=x,states=states,distance=distance)
    return cache


def action_report(cached,policy):
    frame=cached['frame'];p=policy[cached['states'],1];y=frame.action.to_numpy(int)
    report=metrics(y,p)
    people=[]
    for _,index in frame.groupby('participant_id',sort=False).indices.items():
        people.append(metrics(y[index],p[index])['nll'])
    report['participant_macro_nll']=float(np.mean(people))
    report['by_guidance']={g:metrics(y[index],p[index]) for g,index in frame.groupby('guidance',sort=False).indices.items()}
    return report


def performance_report(cached,mdp):
    """Observed-action held-out checks; no claimed counterfactual ground truth."""
    frame=cached['frame'];source=np.flatnonzero(frame._pair_valid.to_numpy(bool))
    part=frame.iloc[source].reset_index(drop=True);x=cached['x'][source]
    prediction=np.full(len(source),np.nan);transition_probability=np.full(len(source),np.nan)
    target=part._next_cost_s.to_numpy(float);action=part.action.to_numpy(int)
    terminal=part._terminal_next.to_numpy(bool);dest=np.full(len(source),mdp['terminal_index'],int)
    dest[~terminal]=cached['states'][source[~terminal]+1]
    for rec,index,p,cost in mdp['dynamics'].blocks(x,part):
        used=action[index]==rec['action'];chosen=index[used]
        prediction[chosen]=cost[used]
        probability=(p[used]*(dest[chosen,None]==rec['classes'][None,:])).sum(1)
        transition_probability[chosen]=probability
    if not np.isfinite(prediction).all():raise ValueError('Missing factual cost prediction')
    def error(index):
        residual=prediction[index]-target[index]
        return dict(n=int(len(index)),mae_s=float(np.abs(residual).mean()),rmse_s=float(np.sqrt((residual**2).mean())),
                    observed_mean_s=float(target[index].mean()),predicted_mean_s=float(prediction[index].mean()))
    result=error(np.arange(len(source)))
    positive=np.flatnonzero(target>0)
    result.update(positive_next_cost_fraction=float((target>0).mean()),
        positive_cost_rows=error(positive) if len(positive) else None,
        next_state_nll=float(-np.log(np.clip(transition_probability,1e-12,1)).mean()),
        next_state_zero_probability_fraction=float((transition_probability<=0).mean()),
        by_action={str(a):error(np.flatnonzero(action==a)) for a in [0,1] if np.any(action==a)},
        interpretation='Factual observed-action validation only; benefits of the unchosen action are model estimates')
    return result


def trigger_plots(out,config,dpi):
    config['plot_status']='running';write_config(out,config)
    try:
        subprocess.run([sys.executable,'-B',str(HERE/'Plot_IRL_Performance.py'),'--run',str(out),'--dpi',str(dpi)],check=True)
    except Exception as exc:
        config.update(plot_status='failed',plot_error=str(exc));write_config(out,config)
        raise RuntimeError('Model saved. Replay Plot_IRL_Performance.py --run "'+str(out)+'" --force') from exc
    config['plot_status']='complete';config.pop('plot_error',None);write_config(out,config)


def write_config(out,config):
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


def validate_args(args):
    for name in ['epochs','clusters','horizon','patience','min_support','threads','reference_samples','transition_max_iter']:
        if getattr(args,name)<1:raise ValueError(name+' must be positive')
    if args.plot_dpi<50 or not 0<=args.gamma<=1:raise ValueError('Invalid DPI or gamma')
    for name in ['lr','slow_cost_weight','slow_duration_s']:
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<=0:raise ValueError(name+' must be positive and finite')
    if not np.isfinite(args.l2) or args.l2<0 or not 0<args.slow_fraction<1:raise ValueError('Invalid regularization or slowdown fraction')


def train(args):
    validate_args(args)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    audit_dataset(args.dataset,verify=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(args.threads)
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    geometry=Path(DEFAULT_GEOMETRY_PATH).resolve()
    geometry_sha=hashlib.sha256(geometry.read_bytes()).hexdigest()
    if manifest['source_hashes'][str(geometry)]!=geometry_sha:raise ValueError('Geometry changed since input construction')
    out=HERE/'Results_IRL'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(status='initializing',method=METHOD,reward_scope='spatial_residual_minus_known_slowdown_cost',
        geometry_sha256=geometry_sha,dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
        cost_definition='confirmed frozen-baseline fractional speed deficit times dt, independent of gaze',
        time_alignment='state includes speed at t; gaze action at t; cost and destination observed at t+1',
        global_comparison='force first action only; then follow fitted human policy; common floor-stage reference bank',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'irl_performance.py',
            HERE/'Plot_IRL_Performance.py',HERE/'irl_data.py',HERE/'irl_core.py',HERE/'Spatial_IRL_Rule.py',HERE/'Plot_IRL_Weakness.py']})
    write_config(out,config)
    parts,step=load_prepared(args.dataset,args.slow_fraction,args.slow_duration_s)
    config['step_s']=step;config['performance_window_s']=step*args.horizon;write_config(out,config)
    input_audit={}
    for split,frame in parts.items():
        pairs=frame.loc[frame._pair_valid]
        input_audit[split]=dict(rows=len(frame),transition_pairs=len(pairs),
            positive_cost_pairs=int((pairs._next_cost_s>0).sum()),
            observed_deficit_seconds=float(pairs._next_cost_s.sum()),terminal_pairs=int(frame._terminal_next.sum()))
    (out/'performance_input_audit.json').write_text(json.dumps(input_audit,indent=2),encoding='utf-8')
    if input_audit['train']['positive_cost_pairs']==0:
        raise ValueError('Training inputs contain no slowdown cost; cannot estimate performance differences')
    mdp=build_performance_mdp(parts['train'],step,args.clusters,args.seed,args.reference_samples,args.transition_max_iter)
    cached=cache_observations(parts,mdp)
    transitions=torch.as_tensor(mdp['transitions'],device=device)
    costs=torch.as_tensor(mdp['cost'],device=device)
    counts=torch.as_tensor(mdp['counts'],device=device)
    mask=reward_mask(mdp['encoder'].names)
    features=torch.as_tensor(mdp['centers']*mask,device=device)
    groups=torch.as_tensor(mdp['guidance_index'],dtype=torch.long,device=device)
    model=LinearReward(features.shape[1]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf');best_state=None;stale=0;history=[];best_epoch=None
    for epoch in trange(args.epochs,desc='IRL with slowdown costs'):
        optimizer.zero_grad()
        reward=model(features,groups)-args.slow_cost_weight*costs
        logp,_=soft_policy(reward,transitions,args.horizon,args.gamma,mdp['terminal_index'])
        loss=weighted_nll(logp,counts)+args.l2*(model.shared.square().mean()+model.deviation.square().mean())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite IRL loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        with torch.no_grad():
            logp,_=soft_policy(model(features,groups)-args.slow_cost_weight*costs,transitions,args.horizon,args.gamma,mdp['terminal_index'])
            policy=logp.exp().cpu().numpy()
        score=action_report(cached['val'],policy)['participant_macro_nll']
        if not np.isfinite(score):raise FloatingPointError('Nonfinite validation score')
        history.append(dict(epoch=epoch+1,loss=float(loss.detach()),val_macro_nll=score))
        if score<best-1e-5:
            best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=args.patience:break
    model.load_state_dict(best_state)
    with torch.no_grad():
        logp,q=soft_policy(model(features,groups)-args.slow_cost_weight*costs,transitions,args.horizon,args.gamma,mdp['terminal_index'])
        policy=logp.exp().cpu().numpy()
    qcost,_=cost_to_go(policy,mdp['transitions'],mdp['cost'],args.horizon)
    report={s:dict(behavior=action_report(cached[s],policy),performance=performance_report(cached[s],mdp)) for s in ['train','val','test']}
    report.update(trained=True,states=len(mdp['centers']),train_rows=mdp['train_rows'],selected_val_macro_nll=best,
        conditional_dynamics=[{k:r[k] for k in ['guidance','smoke','action','pairs','converged']} for r in mdp['dynamics'].records],
        stopping=dict(reason='validation_patience' if stale>=args.patience else 'max_epochs',epochs_run=len(history),
                      best_epoch=best_epoch,patience=args.patience,min_delta=1e-5,monitor='val_participant_macro_nll'))
    np.savez_compressed(out/'mdp.npz',**{k:mdp[k] for k in ['centers','transitions','cost','counts','raw_counts',
        'transition_counts','guidance_index','radius','terminal_index']})
    mdp['dynamics'].save(out/'conditional_dynamics.npz')
    (out/'encoder.json').write_text(json.dumps(dict(encoder=mdp['encoder'].as_dict(),contexts=mdp['contexts'],step_s=step),indent=2),encoding='utf-8')
    (out/'reference_bank.json').write_text(json.dumps(mdp['reference_bank'],indent=2,allow_nan=False),encoding='utf-8')
    torch.save(best_state,out/'reward.pt')
    deviation=best_state['deviation'].numpy();deviation=deviation-deviation.mean(0,keepdims=True)
    rows=[]
    for g,guidance in enumerate(GUIDANCE):
        for name,w,keep in zip(mdp['encoder'].names,best_state['shared'].numpy()+deviation[g],mask):
            if keep:rows.append(dict(guidance=guidance,feature=name,residual_reward_contrast_weight=float(w)))
    pd.DataFrame(rows).to_csv(out/'reward_rules.csv',index=False)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'performance_rule.json').write_text(json.dumps(dict(
        training_reward='r_theta(spatial,a) - slow_cost_weight * E[slowdown_cost_at_t_plus_1 | state_t,a]',
        field='mean_reference(Q_cost(state,0) - Q_cost(state,1))',units='equivalent full-stop seconds',
        recurrence='Q_cost_h = c + P V_cost_(h-1); V_cost_h = sum_a pi(a|s) Q_cost_h',
        positive='looking has lower predicted slowdown loss',horizon_steps=args.horizon,step_s=step,
        reference='same saved joint history sample within floor/stage for all guidance, smoke and both first actions',
        not_claimed='randomized causal effect, total evacuation time saved, or guaranteed guidance failure'),indent=2),encoding='utf-8')
    if args.trajectory_diagnostics:
        (out/'predictions').mkdir()
        for s,item in cached.items():
            frame=item['frame'][['trial_id','participant_id','guidance','smoke','floor','stage','timestamp_ns','x','z','action','_next_cost_s']].copy()
            state=item['states'];frame['p_look']=policy[state,1]
            frame['cost_not_look_s']=qcost[state,0];frame['cost_look_s']=qcost[state,1]
            frame['performance_benefit_s']=qcost[state,0]-qcost[state,1]
            frame.to_csv(out/'predictions'/f'{s}.csv.gz',index=False,compression='gzip')
    config['status']='complete';write_config(out,config)
    print('MODEL_OUTPUT='+str(out),flush=True)
    trigger_plots(out,config,args.plot_dpi)
    print('OUTPUT='+str(out))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--train',action='store_true');p.add_argument('--check-data',action='store_true')
    p.add_argument('--report',type=Path)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--clusters',type=int,default=24)
    p.add_argument('--horizon',type=int,default=25);p.add_argument('--gamma',type=float,default=.98)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--l2',type=float,default=.01)
    p.add_argument('--patience',type=int,default=15);p.add_argument('--min-support',type=int,default=5)
    p.add_argument('--seed',type=int,default=20260926);p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--plot-dpi',type=int,default=300)
    p.add_argument('--trajectory-diagnostics',action='store_true')
    p.add_argument('--slow-fraction',type=float,default=.25)
    p.add_argument('--slow-duration-s',type=float,default=.4)
    p.add_argument('--slow-cost-weight',type=float,default=1.)
    p.add_argument('--reference-samples',type=int,default=16)
    p.add_argument('--transition-max-iter',type=int,default=300)
    return p


def main():
    p=parser();args=p.parse_args()
    if args.train and args.check_data:p.error('--train and --check-data are mutually exclusive')
    if args.report and args.report.exists():p.error('Report already exists')
    if args.train:train(args)
    else:
        report=audit_dataset(args.dataset)
        print(json.dumps(report,indent=2))
        if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
