"""Internal V4 worker; invoked by Train_IRL_V4, never runs on import."""
from pathlib import Path
import argparse
import json
import math
import time
import traceback
import shutil
import numpy as np
import pandas as pd
import torch
from irl_core_v3 import LinearReward, soft_policy, weighted_nll
from Spatial_IRL_Rule import reward_mask
from irl_search_v4 import (prepare_cache, load_cache, write_json, summarize_policy,
                           selection_score, baseline_reports, deadline_remaining, METHOD)
from Spatial_IRL_Rule import SCOPE


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def configure_device(name):
    device=torch.device(name)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(0 if device.index is None else device.index)
    return device


def train_candidate(job):
    started=time.time();out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    c=job['config'];device=configure_device(job['device'])
    torch.set_num_threads(job['threads']);torch.manual_seed(c['seed'])
    mdp,observations,keys=load_cache(job['cache'],job['dataset'])
    horizon=max(1,int(math.ceil(c['horizon_seconds']/mdp['step_s'])))
    f=torch.as_tensor(mdp['centers'],dtype=torch.float32,device=device)
    f=f*torch.as_tensor(reward_mask(mdp['encoder'].names),device=device)
    transitions=torch.as_tensor(mdp['transitions'],dtype=torch.float32,device=device)
    dt=torch.as_tensor(mdp['mean_dt'],dtype=torch.float32,device=device)
    counts=torch.as_tensor(mdp['counts'],dtype=torch.float32,device=device)
    guide=torch.as_tensor(mdp['guidance_index'],dtype=torch.long,device=device)
    model=LinearReward(f.shape[1]).to(device)
    if c['optimizer']=='adam':optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'])
    elif c['optimizer']=='lbfgs':optimizer=torch.optim.LBFGS(model.parameters(),lr=c['lr'],max_iter=10,history_size=20,line_search_fn='strong_wolfe')
    else:raise ValueError('Unsupported optimizer')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,
        patience=job['lr_patience'],threshold=job['min_delta'],threshold_mode='abs',min_lr=1e-5)
    history=[];best=math.inf;best_epoch=None;best_report=None;best_state=None;stale=0;epoch_start=0
    checkpoint=out/'checkpoint.pt'
    if job.get('resume'):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['config']!=c:raise ValueError('Resume configuration differs')
        model.load_state_dict(saved['current']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];best_report=saved['best_report']
        best_state=saved['best_state'];epoch_start=saved['epoch']
        # Promotion gives the reduced learning rate another resource allocation.
        stale=0

    def current_policy():
        lp,_=soft_policy(model(f,guide),transitions,horizon,c['gamma'],mdp['terminal_index'],mean_dt=dt)
        return lp

    def objective():
        lp=current_policy()
        regularization=c['l2']*(model.shared.square().mean()+c['deviation_ratio']*model.deviation.square().mean())
        return weighted_nll(lp,counts)+regularization

    def save(reason,epoch):
        if best_state is None:return
        torch.save(best_state,out/'reward.pt')
        if not (out/'encoder.json').exists():shutil.copy2(Path(job['cache'])/'encoder.json',out/'encoder.json')
        source_manifest=json.loads((Path(job['dataset'])/'manifest.json').read_text(encoding='utf-8'))
        from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
        write_json(out/'config.json',dict(c,status='complete',method=METHOD,reward_scope=SCOPE,
            horizon=horizon,nominal_step_s=mdp['step_s'],dataset=job['dataset'],
            geometry_sha256=source_manifest['source_hashes'][str(Path(DEFAULT_GEOMETRY_PATH).resolve())],
            candidate=True,plot_status='not_requested'))
        torch.save(dict(config=c,current=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
            scheduler=cpu_tree(scheduler.state_dict()),history=history,best=best,best_epoch=best_epoch,
            best_report=best_report,best_state=best_state,epoch=epoch),checkpoint)
        report=dict(status='complete',id=job['id'],config=c,score=best,validation=best_report,
            best_epoch=best_epoch,epochs_run=epoch,reason=reason,device=str(device),
            elapsed_seconds=time.time()-started,cache=job['cache'],horizon=horizon)
        write_json(out/'result.json',report)
        return report

    reason='resource_limit';last_epoch=epoch_start
    for epoch in range(epoch_start,job['epochs']):
        if (out/'STOP').exists():
            reason='phase_budget';break
        if deadline_remaining(job['deadline'])<=2.:
            reason='time_budget';break
        model.train()
        if c['optimizer']=='lbfgs':
            def closure():
                optimizer.zero_grad();loss=objective()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite IRL loss')
                loss.backward();return loss
            optimizer.step(closure)
        else:
            optimizer.zero_grad();loss=objective()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite IRL loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        last_epoch=epoch+1
        # Validation every epoch; state counts are small and reproduce all frame contributions.
        with torch.no_grad():
            lp=current_policy();policy=lp.exp().cpu().numpy();loss_value=float(objective().cpu())
        validation=summarize_policy(observations['val'],policy,mdp['guidance_index'])
        score=selection_score(validation,job['selection_metric'])
        old_lr=optimizer.param_groups[0]['lr']
        scheduler.step(score)
        new_lr=optimizer.param_groups[0]['lr']
        improved=score<best-job['min_delta']
        if improved:
            best=score;best_state=cpu_tree(model.state_dict());best_report=validation;best_epoch=last_epoch;stale=0
        else:stale+=1
        if new_lr<old_lr:stale=0
        row=dict(epoch=last_epoch,loss=loss_value,lr=new_lr,selection_score=score,
            val_participant_macro_nll=validation['participant_macro_nll'],val_guidance_macro_nll=validation['guidance_macro_nll'],
            val_nll=validation['nll'],val_brier=validation['brier'],val_auc=validation['auc'])
        for g,v in validation['by_guidance'].items():
            for metric in ['nll','auc','brier','observed_fraction','predicted_fraction']:row[g+'_'+metric]=v[metric]
        history.append(row)
        # Small JSON progress lets the parent preserve diagnostic state on interruptions.
        write_json(out/'progress.json',dict(epoch=last_epoch,best_score=best,best_epoch=best_epoch,
            validation=best_report,elapsed_seconds=time.time()-started,current_lr=new_lr))
        if improved or last_epoch%10==0:save('running_checkpoint',last_epoch)
        if stale>=job['patience'] and last_epoch>=job['min_epochs']:
            reason='validation_patience';break
    if best_state is None:raise RuntimeError('Budget ended before the first validated checkpoint')
    result=save(reason,last_epoch)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    write_json(out/'validation_baselines.json',baseline_reports(observations['train'],observations['val'],keys))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    args=p.parse_args();job=json.loads(args.job.read_text(encoding='utf-8'))
    torch.set_num_threads(job['threads'])
    try:
        if job['kind']=='prepare':
            prepare_cache(job['dataset'],job['cache'],job['config'])
            write_json(Path(job['output'])/'result.json',dict(status='complete',kind='prepare',cache=job['cache']))
        elif job['kind']=='train':train_candidate(job)
        else:raise ValueError('Unknown worker task')
    except Exception as exc:
        Path(job['output']).mkdir(parents=True,exist_ok=True)
        write_json(Path(job['output'])/'error.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
