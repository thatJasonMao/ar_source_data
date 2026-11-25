"""Cache, metrics and bounded-search helpers. No fitting on import."""
from __future__ import annotations
from pathlib import Path
import hashlib
import itertools
import json
import math
import os
import time
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from irl_data_v3 import GUIDANCE, StateEncoder, assign_states, iter_trials, validate_manifest_scope

METHOD='budgeted_full_frame_spatial_IRL_v4'
HERE=Path(__file__).resolve().parent


def write_json(path, value):
    path=Path(path)
    # A reader on Windows can briefly deny replacement. Keep the previous
    # complete file, use a unique temp file, and retry only sharing/access errors.
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    tmp=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            handle.write(payload)
        for attempt in range(12):
            try:
                os.replace(tmp,path)
                break
            except PermissionError as exc:
                if attempt==11 or getattr(exc,'winerror',None) not in (None,5,32,33):raise
                time.sleep(min(.01*2**attempt,.25))
    finally:
        tmp.unlink(missing_ok=True)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def state_spec(config):
    return {k:config[k] for k in ['clusters','run_cap_s','seed','prior']}


def cache_key(config, data_hash, runs_hash):
    payload=dict(state=state_spec(config),data=data_hash,runs=runs_hash,
                 code=digest(HERE/'irl_data_v4.py'),encoder_code=digest(HERE/'irl_data_v3.py'),distance_code=digest(HERE/'irl_state_distance.py'),version=2)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]


def config_id(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]


def recapped_weights(df, cap_s, old_cap_s, durations):
    """Adjust existing duration weights exactly; preserve all labels/rows."""
    if not np.isfinite(cap_s) or cap_s<=0:raise ValueError('Invalid run cap')
    w=df.run_weight.to_numpy(float).copy()
    valid=df.valid.eq(1).to_numpy()
    for run,ix in df.loc[valid].groupby('run_id',sort=False).indices.items():
        # indices above are relative to the valid subset, not full df.
        original=np.flatnonzero(valid)[ix]
        duration=durations[int(run)]
        if not np.isfinite(duration) or duration<=0:raise ValueError('Invalid source run duration')
        factor=min(1.,cap_s/duration)/min(1.,old_cap_s/duration)
        w[original]*=factor
    return w


def run_duration_table(dataset):
    table=pd.read_csv(Path(dataset)/'action_runs.csv',usecols=['trial_id','run_id','duration_s'])
    if table.duplicated(['trial_id','run_id']).any():raise ValueError('Duplicate action-run provenance')
    return {str(t):dict(zip(p.run_id.astype(int),p.duration_s.astype(float))) for t,p in table.groupby('trial_id')}


def pack_observations(dataset, mdp, splits, cap_s, durations):
    """Sufficient counts reproduce per-frame metrics without epoch-wise sorting."""
    manifest=json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))
    old_cap=manifest['config']['run_cap_s'];states=len(mdp['centers'])
    keys=[f'{g}/{sm}/{st}' for g,sm,st in itertools.product(GUIDANCE,['HighSmoke','LowSmoke'],
        ['to_stair','to_gate','to_exit_corridor','to_branch_corridor','to_exit','arrived_exit'])]
    key_index={k:i for i,k in enumerate(keys)}
    result={}
    for split in splits:
        records=[r for r in manifest['trials'] if r['split']==split]
        people=sorted({r['participant_id'] for r in records});pindex={p:i for i,p in enumerate(people)}
        natural=np.zeros((len(people),states,2),np.float64);capped=np.zeros_like(natural);balanced=np.zeros_like(natural)
        contexts=np.zeros((len(people),len(keys),2),np.float64);context_cap=np.zeros_like(contexts);context_balanced=np.zeros_like(contexts)
        ntrials={p:sum(r['participant_id']==p and r['valid_rows']>0 for r in records) for p in people}
        for rec,df in iter_trials(dataset,{split}):
            weights=recapped_weights(df,cap_s,old_cap,durations[rec['trial_id']])
            ix=np.flatnonzero(df.valid.eq(1).to_numpy());part=df.iloc[ix]
            if not len(part):continue
            s,_=assign_states(part,mdp['encoder'].transform(part),mdp['contexts'],mdp['centers'])
            if (s<0).any():raise ValueError('Unseen state context in '+split)
            a=part.action.to_numpy(int);pid=pindex[rec['participant_id']]
            np.add.at(natural[pid],(s,a),1.)
            np.add.at(capped[pid],(s,a),weights[ix])
            bw=weights[ix]/(weights[ix].sum()*ntrials[rec['participant_id']])
            np.add.at(balanced[pid],(s,a),bw)
            ki=np.array([key_index[f'{rec["guidance"]}/{rec["smoke"]}/{st}'] for st in part.stage],int)
            np.add.at(contexts[pid],(ki,a),1.)
            np.add.at(context_cap[pid],(ki,a),weights[ix])
            np.add.at(context_balanced[pid],(ki,a),bw)
        result[split]=dict(natural=natural,capped=capped,balanced=balanced,contexts=contexts,
                           context_capped=context_cap,context_balanced=context_balanced,people=people)
    return result,keys


def save_observations(path, data, keys):
    arrays={f'{split}_{name}':v for split,p in data.items() for name,v in p.items() if name!='people'}
    arrays['metadata']=np.asarray(json.dumps(dict(keys=keys,people={s:p['people'] for s,p in data.items()})))
    np.savez_compressed(path,**arrays)


def load_observations(path):
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata']));result={}
        for split,people in meta['people'].items():
            result[split]={name:saved[f'{split}_{name}'] for name in ['natural','capped','balanced','contexts','context_capped','context_balanced']}
            result[split]['people']=people
    return result,meta['keys']


def count_metrics(counts, probabilities):
    counts=np.asarray(counts,float);p=np.clip(np.asarray(probabilities,float),1e-7,1-1e-7)
    n=float(counts.sum())
    if n<=0:return dict(n=0,nll=None,brier=None,auc=None,balanced_accuracy=None,observed_fraction=None,predicted_fraction=None)
    zero,one=counts[:,0],counts[:,1]
    nll=float(-(zero*np.log1p(-p)+one*np.log(p)).sum()/n)
    brier=float((zero*p*p+one*(1-p)**2).sum()/n)
    y=np.tile([0,1],len(p));pred=np.repeat(p,2);w=counts.ravel();present=w>0
    auc=float(roc_auc_score(y[present],pred[present],sample_weight=w[present])) if zero.sum()>0 and one.sum()>0 else None
    balanced=.5*((zero[p<.5].sum()/zero.sum())+(one[p>=.5].sum()/one.sum())) if zero.sum()>0 and one.sum()>0 else None
    return dict(n=n,nll=nll,brier=brier,auc=auc,balanced_accuracy=float(balanced) if balanced is not None else None,
                observed_fraction=float(one.sum()/n),predicted_fraction=float(((zero+one)*p).sum()/n))


def summarize_policy(obs, policy, guidance_index):
    p=np.asarray(policy)[:,1];counts=obs['natural'];total=counts.sum(0)
    report=count_metrics(total,p)
    people=[count_metrics(c,p)['nll'] for c in counts if c.sum()>0]
    report['participant_macro_nll']=float(np.mean(people))
    report['by_guidance']={};guide_person=[]
    for i,g in enumerate(GUIDANCE):
        index=guidance_index==i
        report['by_guidance'][g]=count_metrics(total[index],p[index])
        pn=[count_metrics(c[index],p[index])['nll'] for c in counts if c[index].sum()>0]
        if pn:report['by_guidance'][g]['participant_macro_nll']=float(np.mean(pn));guide_person.append(float(np.mean(pn)))
    report['guidance_macro_nll']=float(np.mean(guide_person))
    report['guidance_macro_auc']=float(np.mean([v['auc'] for v in report['by_guidance'].values() if v['auc'] is not None]))
    report['full_participant_macro_nll']=report['by_guidance']['Full'].get('participant_macro_nll')
    # Equal people/trials is applied during training. This audit view uses per-run capped exposure.
    report['capped_exposure_metrics']=count_metrics(obs['capped'].sum(0),p)
    report['balanced_metrics']=count_metrics(obs['balanced'].sum(0),p)
    return report


def baseline_reports(train, evaluation, keys):
    result={}
    for grouping in ['guidance','guidance_smoke_stage']:
        groups=[k.split('/')[0] if grouping=='guidance' else k for k in keys]
        for weighting,field in [('natural','contexts'),('balanced','context_balanced')]:
            counts=train[field].sum(0);prob=np.empty(len(keys))
            for key in set(groups):
                ix=np.flatnonzero(np.asarray(groups)==key);s=counts[ix].sum(0)
                prob[ix]=(s[1]+.5)/(s.sum()+1.)
            report=count_metrics(evaluation['contexts'].sum(0),prob)
            report['participant_macro_nll']=float(np.mean([count_metrics(c,prob)['nll'] for c in evaluation['contexts'] if c.sum()>0]))
            guide_names=np.asarray([k.split('/')[0] for k in keys])
            report['by_guidance']={g:count_metrics(evaluation['contexts'].sum(0)[guide_names==g],prob[guide_names==g]) for g in GUIDANCE}
            report['balanced_evaluation']=count_metrics(evaluation['context_balanced'].sum(0),prob)
            result[grouping+'_'+weighting]=report
    return result


def selection_score(report, metric):
    value=report.get(metric)
    if value is None or not math.isfinite(value):raise ValueError('Unavailable selection metric: '+metric)
    return float(value)


def prepare_cache(dataset, folder, config):
    from irl_data_v4 import build_mdp
    dataset=Path(dataset);folder=Path(folder)
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'));validate_manifest_scope(manifest)
    mdp=build_mdp(dataset,config['clusters'],config['seed'],config['prior'],run_cap_s=config['run_cap_s'])
    durations=run_duration_table(dataset)
    observations,keys=pack_observations(dataset,mdp,['train','val'],config['run_cap_s'],durations)
    if not np.allclose(observations['train']['balanced'].sum(0),mdp['counts'],atol=1e-6):
        raise ValueError('Cached training exposure differs from IRL likelihood counts')
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'mdp.npz',**{k:mdp[k] for k in ['centers','transitions','counts','raw_counts','transition_counts','guidance_index','radius','terminal_index','mean_dt']})
    write_json(folder/'encoder.json',dict(encoder=mdp['encoder'].as_dict(),contexts=mdp['contexts'],step_s=mdp['step_s']))
    save_observations(folder/'observations.npz',observations,keys)
    write_json(folder/'metadata.json',dict(status='complete',state=state_spec(config),train_rows=mdp['train_rows'],
        state_code_sha256={n:digest(HERE/n) for n in ['irl_data_v3.py','irl_data_v4.py','irl_state_distance.py']},
        manifest_sha256=digest(dataset/'manifest.json'),runs_sha256=digest(dataset/'action_runs.csv'),
        files_sha256={name:digest(folder/name) for name in ['mdp.npz','encoder.json','observations.npz']}))


def load_cache(folder, dataset=None):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if meta['status']!='complete':raise ValueError('Incomplete MDP cache')
    for name,expected in meta.get('state_code_sha256',{}).items():
        if digest(HERE/name)!=expected:raise ValueError('Cache state code changed: '+name)
    if dataset is not None and (meta['manifest_sha256']!=digest(Path(dataset)/'manifest.json') or meta['runs_sha256']!=digest(Path(dataset)/'action_runs.csv')):raise ValueError('Cache source changed')
    for name,expected in meta['files_sha256'].items():
        if digest(folder/name)!=expected:raise ValueError('Cache checksum failure: '+name)
    saved=json.loads((folder/'encoder.json').read_text(encoding='utf-8'))
    encoder=StateEncoder();encoder.median=np.asarray(saved['encoder']['median']);encoder.scale=np.asarray(saved['encoder']['scale'])
    if encoder.names!=saved['encoder']['names']:raise ValueError('Encoder names differ')
    with np.load(folder/'mdp.npz',allow_pickle=False) as f:mdp={k:f[k] for k in f.files}
    mdp.update(encoder=encoder,contexts={k:tuple(v) for k,v in saved['contexts'].items()},step_s=saved['step_s'],terminal_index=int(mdp['terminal_index']),train_rows=meta['train_rows'])
    observations,keys=load_observations(folder/'observations.npz')
    if set(observations)!={'train','val'}:raise ValueError('Search cache must contain train/val only')
    return mdp,observations,keys


def base_config(args):
    return dict(lr=.01,l2=.01,deviation_ratio=1.,clusters=24,run_cap_s=2.,prior=.5,
                seed=args.seed,horizon_seconds=args.horizon_seconds,gamma=args.gamma,optimizer='adam')


def coarse_configs(args):
    base=base_config(args);configs=[base]
    for lr,l2 in itertools.product([.003,.01,.03],[.0001,.001,.01,.1]):
        c=dict(base,lr=lr,l2=l2)
        if c not in configs:configs.append(c)
    rng=np.random.default_rng(args.seed);tail=configs[1:];rng.shuffle(tail)
    return configs[:1]+tail


def explore_configs(winners):
    configs=[]
    for base in winners[:2]:
        for clusters,cap in [(24,5.),(24,10.),(16,2.),(48,2.),(16,5.),(48,5.),(16,10.),(48,10.)]:
            c=dict(base,clusters=clusters,run_cap_s=cap)
            if c not in configs:configs.append(c)
    if winners:
        for ratio in [.25,4.]:configs.append(dict(winners[0],deviation_ratio=ratio))
    return configs


def rank_records(records):
    return sorted([r for r in records if r.get('status')=='complete' and r.get('score') is not None],
                  key=lambda r:(r['score'],r['validation']['brier'],r['id']))


def deadline_remaining(deadline):
    return max(0.,deadline-time.time())
