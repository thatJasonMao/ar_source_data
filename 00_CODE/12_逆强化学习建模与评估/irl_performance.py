"""Known slowdown costs, conditional dynamics, and spatial performance queries.

Fitting functions are called only by the explicitly requested --train path.
All saved objects use JSON/NPZ/state_dict; no pickle estimators are required.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, Ridge
from tqdm import tqdm

from irl_data import StateEncoder, NUMERIC, GUIDANCE, STAGES, iter_trials, context_keys, assign_states
from irl_core import LinearReward, soft_policy
from Spatial_IRL_Rule import SPATIAL, reward_mask

METHOD = 'performance_cost_irl_v1'
EXTRA = ['slow_active_now','slow_deficit_now','slow_age_s','slow_anchor_mps']
REFERENCE_COLUMNS = NUMERIC[5:]+['previous_action']+EXTRA


def slowdown_trace(speed, baseline, valid, segments, step_s, fraction=.25, duration_s=.4):
    """Causal confirmation, frozen baseline, no future backfill or gaze gating."""
    minimum=max(1,int(np.ceil(duration_s/step_s-1e-9)))
    n=len(speed);active=np.zeros(n);deficit=np.zeros(n);age=np.zeros(n);anchor_values=np.zeros(n)
    anchor=None;count=0;last_segment=None
    for i,(v,b,ok,seg) in enumerate(zip(speed,baseline,valid,segments)):
        if not ok or not np.isfinite(v):
            anchor=None;count=0;last_segment=None
            continue
        if seg!=last_segment:anchor=None;count=0
        last_segment=seg
        if anchor is None:
            if np.isfinite(b) and b>=.2 and v<=(1-fraction)*b:
                anchor=float(b);count=1
        elif v>=(1-fraction/2)*anchor:
            anchor=None;count=0
        else:count+=1
        if anchor is not None:
            anchor_values[i]=anchor;age[i]=count*step_s
            deficit[i]=np.clip(1-v/anchor,0,1)
            active[i]=float(count>=minimum)
    cost=active*deficit*step_s
    cost[~np.asarray(valid,bool)]=np.nan
    return dict(slow_active_now=active,slow_deficit_now=deficit,
                slow_age_s=age,slow_anchor_mps=anchor_values,slowdown_cost_s=cost)


def prepare_trial(df, step_s, fraction=.25, duration_s=.4):
    """S_t observes speed at t; gaze_t is followed by cost_{t+1}.

The legacy speed_prev/acceleration_prev columns are mapped to values available
at action time; encoder names explicitly document this new state timing.
"""
    result=df.copy()
    trace=slowdown_trace(df.speed.to_numpy(float),df.speed_mean_past.to_numpy(float),
        df.valid.eq(1).to_numpy(),df.segment_id.to_numpy(),step_s,fraction,duration_s)
    for key,value in trace.items():result[key]=value
    result['speed_prev']=df.speed
    result['acceleration_prev']=(df.speed-df.speed_prev)/step_s
    same_time=np.r_[np.diff(df.timestamp_ns.to_numpy(np.int64))==int(round(step_s*1e9)),False]
    valid=df.valid.eq(1).to_numpy()
    adjacent=valid & np.r_[valid[1:],False] & same_time
    adjacent &= df.segment_id.to_numpy()==np.r_[df.segment_id.to_numpy()[1:],-2]
    terminal=valid & df.after_exit.eq(0).to_numpy() & np.r_[df.stage.eq('arrived_exit').to_numpy()[1:],False] & same_time
    result['_pair_valid']=adjacent|terminal
    result['_terminal_next']=terminal
    result['_next_cost_s']=np.where(terminal,0.,np.where(adjacent,np.r_[trace['slowdown_cost_s'][1:],np.nan],np.nan))
    return result


def load_prepared(dataset, fraction=.25, duration_s=.4):
    manifest=json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))
    step=float(manifest['config']['step_s']);parts={s:[] for s in ['train','val','test']}
    for rec,df in tqdm(iter_trials(dataset),total=len(manifest['trials']),desc='Prepare performance inputs'):
        df=prepare_trial(df,step,fraction,duration_s)
        parts[rec['split']].append(df.loc[df.valid.eq(1)].copy())
    return {s:pd.concat(rows,ignore_index=True) for s,rows in parts.items()},step


class PerformanceEncoder(StateEncoder):
    @property
    def names(self):
        names=[n.replace('speed_prev','speed_at_action').replace('acceleration_prev','acceleration_at_action')
               for n in super().names]
        return names+['slow_active_now','slow_deficit_now','slow_age_over_10s','slow_anchor_over_2mps']

    def transform(self,df):
        base=super().transform(df)
        extra=df[EXTRA].to_numpy(float).copy()
        if not np.isfinite(extra).all():raise ValueError('Nonfinite slowdown state')
        extra[:,2]=np.clip(extra[:,2]/10,0,10)
        extra[:,3]=np.clip(extra[:,3]/2,0,10)
        return np.column_stack([base,extra]).astype(np.float32)


def person_weights(df, cap_runs=False):
    weights=np.asarray(df.run_weight if cap_runs else np.ones(len(df)),dtype=float).copy()
    ntrials=df.groupby('participant_id').trial_id.nunique().to_dict()
    for _,index in df.groupby('trial_id',sort=False).indices.items():
        weights[index]/=weights[index].sum()*ntrials[df.iloc[index[0]].participant_id]
    return weights


class ConditionalDynamics:
    """Linear-softmax next-state model and bounded ridge cost model per g/smoke/a."""
    def __init__(self,records,states,step_s):
        self.records=records;self.states=int(states);self.step_s=float(step_s)

    @classmethod
    def fit(cls,x,frame,dest,cost,weights,states,step_s,seed=20260926,max_iter=300):
        records=[]
        for guidance in GUIDANCE:
            for smoke in ['HighSmoke','LowSmoke']:
                for action in [0,1]:
                    index=np.flatnonzero(frame.guidance.eq(guidance)&frame.smoke.eq(smoke)&frame.action.eq(action))
                    if len(index)<2:raise ValueError(f'Too few transition pairs: {guidance}/{smoke}/{action}')
                    xx=x[index];yy=dest[index];w=weights[index];w=w/w.mean()
                    classes=np.unique(yy)
                    converged=True
                    if len(classes)==1:
                        coef=np.zeros((1,x.shape[1]));intercept=np.zeros(1)
                    else:
                        model=LogisticRegression(C=1.,solver='lbfgs',max_iter=max_iter,random_state=seed)
                        model.fit(xx,yy,sample_weight=w)
                        coef=model.coef_;intercept=model.intercept_;classes=model.classes_
                        converged=bool(np.max(model.n_iter_)<max_iter)
                        if len(classes)==2:
                            coef=np.vstack([np.zeros_like(coef),coef]);intercept=np.r_[0.,intercept]
                    cost_model=Ridge(alpha=1.,fit_intercept=False)
                    cost_model.fit(xx,cost[index],sample_weight=w)
                    records.append(dict(guidance=guidance,smoke=smoke,action=action,classes=classes.astype(int),
                        coef=coef.astype(np.float32),intercept=intercept.astype(np.float32),
                        cost_coef=cost_model.coef_.astype(np.float32),pairs=len(index),converged=converged))
        return cls(records,states,step_s)

    def blocks(self,x,frame):
        for record in self.records:
            index=np.flatnonzero(frame.guidance.eq(record['guidance'])&frame.smoke.eq(record['smoke']))
            if len(index):
                probs=softmax(x[index]@record['coef'].T+record['intercept'],axis=1)
                costs=np.clip(x[index]@record['cost_coef'],0,self.step_s)
                yield record,index,probs,costs

    def dense(self,x,frame):
        transitions=np.zeros((len(x),2,self.states),np.float32);cost=np.zeros((len(x),2),np.float32)
        for rec,index,p,c in self.blocks(x,frame):
            transitions[index[:,None],rec['action'],rec['classes'][None,:]]=p
            cost[index,rec['action']]=c
        if not np.allclose(transitions.sum(2),1,atol=1e-5):raise ValueError('Missing conditional dynamics context')
        return transitions,cost

    def cost_queries(self,x,frame,future_value):
        qcost=np.full((len(x),2),np.nan,np.float32);immediate=np.full_like(qcost,np.nan)
        for rec,index,p,c in self.blocks(x,frame):
            immediate[index,rec['action']]=c
            qcost[index,rec['action']]=c+p@future_value[rec['classes']]
        if not np.isfinite(qcost).all():raise ValueError('Missing query dynamics context')
        return qcost,immediate

    def save(self,path):
        arrays={};meta=[]
        for i,record in enumerate(self.records):
            meta.append({k:v for k,v in record.items() if k not in ['classes','coef','intercept','cost_coef']})
            for key in ['classes','coef','intercept','cost_coef']:arrays[f'{i}_{key}']=record[key]
        arrays['metadata']=np.asarray(json.dumps(dict(records=meta,states=self.states,step_s=self.step_s)))
        np.savez_compressed(path,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as arrays:
            meta=json.loads(str(arrays['metadata']));records=meta['records']
            for i,record in enumerate(records):
                for key in ['classes','coef','intercept','cost_coef']:record[key]=arrays[f'{i}_{key}'].copy()
        return cls(records,meta['states'],meta['step_s'])


def make_reference_bank(train,samples=16):
    """Shared g/smoke reference within floor+stage; equal persons then trials.

Deterministic weighted quantiles retain observed joint histories, not independent
feature means. The same bank is used for both actions and all guidance modes.
"""
    bank={}
    for (floor,stage),part in train.groupby(['floor','stage'],sort=True,observed=True):
        part=part.reset_index(drop=True);w=person_weights(part);w/=w.sum()
        index=np.minimum(np.searchsorted(np.cumsum(w),(np.arange(samples)+.5)/samples),len(part)-1)
        chosen=part.iloc[index][REFERENCE_COLUMNS].replace([np.inf,-np.inf],np.nan)
        bank[floor+'/'+stage]=json.loads(chosen.to_json(orient='records'))
    return bank


def build_performance_mdp(train,step_s,clusters=24,seed=20260926,reference_samples=16,transition_max_iter=300):
    encoder=PerformanceEncoder().fit(train);x=encoder.transform(train);keys=context_keys(train).to_numpy()
    centers=[];contexts={};meta=[];offset=0
    for key in tqdm(sorted(np.unique(keys)),desc='Performance states'):
        index=np.flatnonzero(keys==key);k=min(clusters,max(1,len(index)//50))
        clustering=MiniBatchKMeans(n_clusters=k,random_state=seed,n_init=3,batch_size=2048)
        clustering.fit(x[index],sample_weight=train.run_weight.to_numpy()[index])
        centers.append(clustering.cluster_centers_);contexts[key]=(offset,k);offset+=k
        guidance,smoke,previous=key.split('/')
        meta.extend([dict(guidance=guidance,smoke=smoke,previous_action=int(previous))]*k)
    centers=np.concatenate(centers).astype(np.float32)
    state,distance=assign_states(train,x,contexts,centers)
    terminal=len(centers);centers=np.vstack([centers,np.zeros((1,x.shape[1]),np.float32)])
    radius=np.array([max(.1,float(np.quantile(distance[state==s],.99))) if np.any(state==s) else .1
                     for s in range(terminal+1)],np.float32)
    source=np.flatnonzero(train._pair_valid.to_numpy(bool))
    is_terminal=train._terminal_next.to_numpy(bool)[source]
    dest=np.full(len(source),terminal,int)
    normal=source[~is_terminal]
    if np.any(normal+1>=len(train)):raise ValueError('Broken successor indices')
    if not ((train.trial_id.to_numpy()[normal]==train.trial_id.to_numpy()[normal+1]) &
            (train.segment_id.to_numpy()[normal]==train.segment_id.to_numpy()[normal+1]) &
            (np.diff(train.timestamp_ns.to_numpy(np.int64))[normal]==int(round(step_s*1e9)))).all():
        raise ValueError('Transition crosses gap or trial')
    dest[~is_terminal]=state[normal+1]
    weights=person_weights(train)
    dynamics=ConditionalDynamics.fit(x[source],train.iloc[source].reset_index(drop=True),dest,
        train._next_cost_s.to_numpy()[source],weights[source],len(centers),step_s,seed,transition_max_iter)
    transition,cost=dynamics.dense(centers[:-1],pd.DataFrame(meta))
    p=np.zeros((len(centers),2,len(centers)),np.float32);p[:-1]=transition;p[-1,:,-1]=1.
    c=np.vstack([cost,np.zeros((1,2),np.float32)])
    counts=np.zeros((len(centers),2),np.float32);raw=np.zeros_like(counts);pair_counts=np.zeros_like(counts)
    actions=train.action.to_numpy(int)
    np.add.at(counts,(state,actions),person_weights(train,cap_runs=True))
    np.add.at(raw,(state,actions),1)
    np.add.at(pair_counts,(state[source],actions[source]),1)
    return dict(encoder=encoder,centers=centers,contexts=contexts,radius=radius,terminal_index=terminal,
        transitions=p,cost=c,counts=counts,raw_counts=raw,transition_counts=pair_counts,
        guidance_index=np.r_[[GUIDANCE.index(row['guidance']) for row in meta],0],dynamics=dynamics,
        reference_bank=make_reference_bank(train,reference_samples),step_s=step_s,train_rows=len(train))


def cost_to_go(policy,transitions,cost,horizon):
    """Force first action; subsequently follow the fitted fixed-horizon policy.

Undiscounted deficit-seconds, not learned utility. Returns Q_C and V_{H-1}.
"""
    if horizon<1:raise ValueError('horizon must be positive')
    value=np.zeros(len(policy),np.float32)
    for _ in range(horizon-1):
        q=cost+np.einsum('sak,k->sa',transitions,value)
        value=np.sum(policy*q,axis=1)
    return cost+np.einsum('sak,k->sa',transitions,value),value


class PerformanceIRLQuery:
    """Global B(x)=E[C(no look)-C(look)] under a saved shared reference bank."""
    def __init__(self,run):
        run=Path(run);self.config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        if self.config.get('method')!=METHOD or self.config.get('status')!='complete':
            raise ValueError('Requires a completed performance_cost_irl_v1 model')
        saved=json.loads((run/'encoder.json').read_text(encoding='utf-8'))
        self.encoder=PerformanceEncoder();self.encoder.median=np.asarray(saved['encoder']['median'])
        self.encoder.scale=np.asarray(saved['encoder']['scale'])
        if self.encoder.names!=saved['encoder']['names']:raise ValueError('Encoder mismatch')
        self.contexts=saved['contexts'];self.bank=json.loads((run/'reference_bank.json').read_text(encoding='utf-8'))
        with np.load(run/'mdp.npz',allow_pickle=False) as archive:self.mdp={k:archive[k].copy() for k in archive.files}
        self.dynamics=ConditionalDynamics.load(run/'conditional_dynamics.npz')
        reward=LinearReward(len(self.encoder.names));reward.load_state_dict(torch.load(run/'reward.pt',map_location='cpu',weights_only=True))
        mask=reward_mask(self.encoder.names)
        if np.any(np.abs(reward.shared.detach().numpy()[mask==0])>1e-8) or np.any(np.abs(reward.deviation.detach().numpy()[:,mask==0])>1e-8):
            raise ValueError('Nonspatial residual reward coefficients')
        with torch.no_grad():
            r=reward(torch.from_numpy(self.mdp['centers']*mask),torch.from_numpy(self.mdp['guidance_index']).long())
            r-=self.config['slow_cost_weight']*torch.from_numpy(self.mdp['cost'])
            logp,_=soft_policy(r,torch.from_numpy(self.mdp['transitions']),self.config['horizon'],self.config['gamma'],int(self.mdp['terminal_index']))
        self.policy=logp.exp().numpy()
        self.qcost,self.future_cost=cost_to_go(self.policy,self.mdp['transitions'],self.mdp['cost'],self.config['horizon'])

    def query_features(self,frame,guidance,batch_size=512):
        if guidance not in GUIDANCE:raise ValueError('Unknown guidance')
        required=SPATIAL+['floor','stage','smoke']
        if not set(required)<=set(frame):raise ValueError('Missing spatial fields')
        if not np.isfinite(frame[SPATIAL].to_numpy(float)).all():raise ValueError('Nonfinite spatial features')
        if not frame.smoke.isin(['HighSmoke','LowSmoke']).all():raise ValueError('Unknown smoke')
        result=pd.DataFrame(index=np.arange(len(frame)),columns=['cost_not_look_s','cost_look_s','performance_benefit_s','reference_supported_fraction'],dtype=float)
        result['status']='no_reference_for_stage'
        source=frame.reset_index(drop=True)
        for (floor,stage),index in source.groupby(['floor','stage'],observed=True).indices.items():
            refs=self.bank.get(floor+'/'+stage)
            if not refs:continue
            reference=pd.DataFrame(refs);n=len(reference)
            for start in range(0,len(index),batch_size):
                subset=index[start:start+batch_size];spatial=source.iloc[subset]
                expanded=spatial.iloc[np.repeat(np.arange(len(spatial)),n)].reset_index(drop=True)
                for name in REFERENCE_COLUMNS:expanded[name]=np.tile(reference[name].to_numpy(),len(spatial))
                expanded['guidance']=guidance
                x=self.encoder.transform(expanded)
                q,_=self.dynamics.cost_queries(x,expanded,self.future_cost)
                state,distance=assign_states(expanded,x,self.contexts,self.mdp['centers'])
                safe=np.maximum(state,0)
                supported=(state>=0)&(self.mdp['transition_counts'][safe].min(1)>=self.config['min_support'])
                supported &= distance<=1.5*self.mdp['radius'][safe]
                mean=q.reshape(len(subset),n,2).mean(1)
                result.loc[subset,'cost_not_look_s']=mean[:,0];result.loc[subset,'cost_look_s']=mean[:,1]
                result.loc[subset,'performance_benefit_s']=mean[:,0]-mean[:,1]
                result.loc[subset,'reference_supported_fraction']=supported.reshape(len(subset),n).mean(1)
                result.loc[subset,'status']='model_estimate'
        return result

    def query_position(self,x,y,z,*,guidance,smoke,floor=None):
        from spatial_isovist_core import get_engine
        from Analyze_NextDecisionDistance import get_engine as get_decision
        if smoke not in ['HighSmoke','LowSmoke']:raise ValueError('Unknown smoke')
        s=get_engine().query(x,y,z,7.5 if smoke=='HighSmoke' else 12.5,floor=floor)
        d=get_decision().query(x,y,z,floor=floor)
        if s.status!='ok' or d.status!='ok':return dict(status=s.status if s.status!='ok' else d.status)
        if d.stage=='arrived_exit':return dict(status='task_complete')
        row={name:getattr(s,name) for name in SPATIAL[:4]}
        row.update(next_decision_distance=d.next_decision_distance,floor=d.floor,stage=d.stage,smoke=smoke)
        return {**row,**self.query_features(pd.DataFrame([row]),guidance).iloc[0].to_dict()}
