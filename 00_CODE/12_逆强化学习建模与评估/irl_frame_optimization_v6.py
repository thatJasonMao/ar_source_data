"""V6 shared/channel reward with condition-stratified binary composite likelihood.

The fixed empirical MDP supplies continuation values. Observed-frame reward
replaces the cluster mean immediate reward; this is not a separate policy fit.
"""
from pathlib import Path
import json,math,time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from irl_data_v6 import iter_trials,assign_states,GUIDANCE,channel_records,trial_masses
from Guidance_Channel_Contract import CONDITIONS
from irl_search_v6 import load_cache,run_duration_table,recapped_weights,write_json,digest
from Spatial_IRL_Rule_V6 import reward_mask
from spatial_reward_basis import SpatialRewardBasis
from irl_core_v3 import LinearReward,soft_policy
from IRL_V4_Worker import cpu_tree,configure_device


def training_code_hashes():
    here=Path(__file__).resolve().parent
    return {n:digest(here/n) for n in ['irl_frame_optimization_v6.py','irl_core_v3.py','irl_core.py',
        'spatial_reward_basis.py','Spatial_IRL_Rule_V6.py','irl_data_v6.py','irl_search_v6.py']}


def prepare_frames(dataset, cache, folder, splits=('train','val')):
    mdp,_,_=load_cache(cache,dataset)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((Path(dataset)/'manifest.json').read_text())
    duration=run_duration_table(dataset);mask=reward_mask(mdp['encoder'].names).astype(bool)
    meta={}
    for split in splits:
        parts=[];records=channel_records(manifest,{split})
        people=sorted({r['participant_id'] for r in records});person_index={p:i for i,p in enumerate(people)}
        masses=trial_masses(records)
        for rec,df in iter_trials(dataset,{split}):
            weights=recapped_weights(df,manifest_cap(cache),manifest['config']['run_cap_s'],duration[rec['trial_id']])
            valid=np.flatnonzero(df.valid.eq(1).to_numpy());part=df.iloc[valid]
            if not len(part):continue
            x=mdp['encoder'].transform(part);s,_=assign_states(part,x,mdp['contexts'],mdp['centers'])
            if (s<0).any():raise ValueError('Unsupported context')
            w=weights[valid]/weights[valid].sum()*masses[rec['trial_id']]
            parts.append(dict(x=x[:,mask],state=s.astype(np.int32),action=part.action.to_numpy(np.float32),
                weight=w.astype(np.float32),person=np.full(len(part),person_index[rec['participant_id']],np.int16),
                guidance=np.full(len(part),GUIDANCE.index(rec['guidance']),np.int8),
                condition=np.full(len(part),CONDITIONS.index(rec['condition']),np.int8)))
        if not parts:raise ValueError('Empty channel split: '+split)
        arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for name,arr in arrays.items():np.save(folder/f'{split}_{name}.npy',arr,allow_pickle=False)
        if split=='train':
            counts=np.zeros(mdp['counts'].shape,np.float64);np.add.at(counts,(arrays['state'],arrays['action'].astype(int)),arrays['weight'])
            if not np.allclose(counts,mdp['counts'],atol=2e-6):raise ValueError('Training exposure differs from cached MDP')
        meta[split]=dict(people=people,rows=len(arrays['action']))
    write_json(folder/'metadata.json',dict(status='complete',splits=meta,cache=str(cache),
        spatial_input_names=[n for n,b in zip(mdp['encoder'].names,mask) if b],
        manifest_sha256=digest(Path(dataset)/'manifest.json'),schema='v6_channel_frames',channels=GUIDANCE,
        worker_sha256=digest(Path(__file__)),cache_meta_sha256=digest(Path(cache)/'metadata.json'),
        files_sha256={p.name:digest(p) for p in folder.glob('*.npy')}))


def manifest_cap(cache):
    return json.loads((Path(cache)/'metadata.json').read_text())['state']['run_cap_s']


def load_frames(folder,split):
    folder=Path(folder);m=json.loads((folder/'metadata.json').read_text())
    if m.get('schema')!='v6_channel_frames' or m.get('channels')!=GUIDANCE or m.get('status')!='complete':
        raise ValueError('V6 frame cache required')
    if m['worker_sha256']!=digest(Path(__file__)) or m['cache_meta_sha256']!=digest(Path(m['cache'])/'metadata.json'):
        raise ValueError('V6 frame-cache provenance changed')
    for name,checksum in m['files_sha256'].items():
        if name.startswith(split+'_') and digest(folder/name)!=checksum:raise ValueError('Frame cache checksum differs')
    return {k:np.load(folder/f'{split}_{k}.npy',mmap_mode='r') for k in ['x','state','action','weight','person','guidance','condition']}


def frame_report(data,p,utility=None):
    y=np.asarray(data['action']);p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    nll=-(y*np.log(p)+(1-y)*np.log1p(-p));brier=(p-y)**2
    def sub(ix):
        yy=y[ix];pp=p[ix];w=np.asarray(data['weight'])[ix].astype(float)
        if not len(yy):return dict(n=0,auc=None,nll=None,participant_macro_nll=None,spatial_reward_auc=None)
        return dict(n=int(ix.sum()),people=len(np.unique(data['person'][ix])),
            auc=float(roc_auc_score(yy,pp,sample_weight=w)) if len(np.unique(yy))==2 else None,
            nll=float(nll[ix].mean()),balanced_nll=float(np.average(nll[ix],weights=w)),
            participant_macro_nll=float(np.average(nll[ix],weights=w)),
            brier=float(np.average(brier[ix],weights=w)),observed_fraction=float(yy.mean()),
            predicted_fraction=float(pp.mean()),
            spatial_reward_auc=float(roc_auc_score(yy,np.asarray(utility)[ix],sample_weight=w)) if utility is not None and len(np.unique(yy))==2 else None)
    r=sub(np.ones(len(y),bool))
    r['by_guidance']={g:sub(data['guidance']==i) for i,g in enumerate(GUIDANCE)}
    r['by_channel']=r['by_guidance']
    r['by_channel_condition']={g+'/'+c:sub((data['guidance']==i)&(data['condition']==j)) for i,g in enumerate(GUIDANCE) for j,c in enumerate(CONDITIONS) if np.any((data['guidance']==i)&(data['condition']==j))}
    r['guidance_macro_nll']=float(np.mean([d['balanced_nll'] for d in r['by_guidance'].values()]))
    def average(key):
        values=[d[key] for d in r['by_guidance'].values() if d[key] is not None]
        return float(np.mean(values)) if values else None
    r['guidance_macro_auc']=average('auc');r['spatial_macro_auc']=average('spatial_reward_auc')
    r['weighting']='equal channel / condition / participant / trial; capped duration within trial'
    return r


def state_basis_means(features,states,weights,count):
    sums=np.zeros((count,features.shape[1]),np.float64);mass=np.zeros(count,np.float64)
    np.add.at(sums,states,features*weights[:,None]);np.add.at(mass,states,weights)
    return (sums/np.maximum(mass[:,None],1e-12)).astype(np.float32)


def frame_logit(model,features,guidance,states,mean_features,mdp,transition,dt,horizon,gamma):
    reward=model(mean_features,mdp['guide_tensor'])
    _,q=soft_policy(reward,transition,horizon,gamma,mdp['terminal_index'],mean_dt=dt)
    continuation=(q[:,1]-q[:,0])-(reward[:,1]-reward[:,0])
    own=model(features,guidance)
    return own[:,1]-own[:,0]+continuation[states],own[:,1]-own[:,0]


def train(job):
    started=time.time();out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    c=job['config'];torch.set_num_threads(job.get('threads',2));torch.manual_seed(c['seed'])
    device=configure_device(job['device']);mdp,_,_=load_cache(job['cache'],job['dataset'])
    frames={s:load_frames(job['frames'],s) for s in ['train','val']}
    metadata=json.loads((Path(job['frames'])/'metadata.json').read_text())
    basis=SpatialRewardBasis.fit(frames['train']['x'],metadata['spatial_input_names'],c['basis'])
    features={s:basis.transform(frames[s]['x']) for s in frames}
    mean=state_basis_means(features['train'],frames['train']['state'],frames['train']['weight'],len(mdp['centers']))
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(np.array(a,copy=True),dtype=dtype,device=device)
    mean=tensor(mean);transition=tensor(mdp['transitions']);dt=tensor(mdp['mean_dt'])
    mdp['guide_tensor']=tensor(mdp['guidance_index'],torch.long)
    data={s:dict(features=tensor(features[s]),guide=tensor(frames[s]['guidance'],torch.long),state=tensor(frames[s]['state'],torch.long)) for s in frames}
    labels=tensor(frames['train']['action']);weight=tensor(frames['train']['weight']);weight=weight/weight.sum()
    horizon=max(1,int(math.ceil(c['horizon_seconds']/mdp['step_s'])))
    active=c.get('active_guidance',GUIDANCE)
    if active != GUIDANCE:raise ValueError('Unsupported guidance ordering')
    if active!=GUIDANCE:
        if any(key.split('/')[0] not in active for key in mdp['contexts']):raise ValueError('Excluded guidance in MDP')
        for split,d in frames.items():
            if set(np.unique(d['guidance']))!=set(range(len(active))):raise ValueError('Excluded or missing guidance in '+split)
    model=LinearReward(mean.shape[1],n_guidance=len(active)).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5,threshold=1e-5,threshold_mode='abs',min_lr=1e-5)
    def logits(split):
        d=data[split]
        return frame_logit(model,d['features'],d['guide'],d['state'],mean,mdp,transition,dt,horizon,c['gamma'])
    history=[];best=float('inf');best_state=None;stale=0;best_epoch=0;reason='epochs'
    for epoch in range(1,job['epochs']+1):
        if time.time()>job['deadline']-3 or (out/'STOP').exists():reason='budget';break
        optimizer.zero_grad();z,_=logits('train')
        loss=(torch.nn.functional.softplus(z)-labels*z).mul(weight).sum()
        regularization=c['l2']*(model.shared.square().sum()+c['deviation_ratio']*model.deviation.square().mean(0).sum())
        (loss+regularization).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if epoch%5 and epoch!=1:continue
        with torch.no_grad():
            z,u=logits('val');report=frame_report(frames['val'],torch.sigmoid(z).cpu().numpy(),u.cpu().numpy())
        score=report['guidance_macro_nll'];old=optimizer.param_groups[0]['lr'];scheduler.step(score)
        if score<best-1e-5:
            best=score;best_state=cpu_tree(model.state_dict());best_epoch=epoch;best_report=report;stale=0
            torch.save(best_state,out/'reward.pt');write_json(out/'reward_basis.json',basis.spec)
        else:stale+=1
        # LR reductions do not repeatedly erase the total early-stop counter.
        history.append(dict(epoch=epoch,train_nll=float(loss.detach()),lr=optimizer.param_groups[0]['lr'],score=score,**{g+'_auc':a['auc'] for g,a in report['by_guidance'].items()}))
        write_json(out/'progress.json',dict(epoch=epoch,best_epoch=best_epoch,score=best))
        if stale>=job.get('patience',16) and epoch>=50:reason='early_stop';break
    if best_state is None:raise RuntimeError('No validated checkpoint')
    model.load_state_dict(best_state)
    reports={}
    with torch.no_grad():
        for s in frames:
            z,u=logits(s);reports[s]=frame_report(frames[s],torch.sigmoid(z).cpu().numpy(),u.cpu().numpy())
    np.save(out/'state_reward_features.npy',mean.cpu().numpy(),allow_pickle=False)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    write_json(out/'result.json',dict(status='complete',config=c,score=best,validation=reports['val'],train=reports['train'],
        best_epoch=best_epoch,epochs_run=epoch,reason=reason,elapsed_seconds=time.time()-started,features=len(basis.names),test_used=False,
        source_sha256=training_code_hashes()))
    return out


def main():
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    job=json.loads(args.job.read_text())
    try:train(job)
    except Exception as exc:
        write_json(Path(job['output'])/'error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
