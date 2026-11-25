"""Streaming trial readers, train-only encoding, and empirical finite MDP."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'0_RL_SOURCE'))
from IRL_Cohort import SCHEMA as LEGACY_SCHEMA, COHORT_POLICY, GUIDANCE, validate_manifest_scope as validate_cohort
SCHEMA = 'a5_irl_full_frames_v3'

def validate_manifest_scope(manifest):
    if manifest.get('schema') != SCHEMA or manifest.get('sampling_mode') != 'all_unique_subject_timestamps':
        raise ValueError('V3 requires a new full-frame dataset from Build_IRL_Dataset_V3.py; 5 Hz inputs are rejected')
    validate_cohort(dict(manifest, schema=LEGACY_SCHEMA))
    dt=manifest['config']['step_s']
    if not np.isfinite(dt) or dt <= 0: raise ValueError('Invalid nominal full-frame interval')

STAGES = ['to_stair','to_gate','to_exit_corridor','to_branch_corridor','to_exit','arrived_exit']
NUMERIC = ['concavity','distance_variation','elongation','horizontal_size',
           'next_decision_distance','speed_prev','speed_mean_past','acceleration_prev',
           'head_rate_prev','scan_rate_prev','gaze_depth_prev','elapsed_s']
LOG_COLUMNS = ['distance_variation','elongation','horizontal_size','next_decision_distance',
               'head_rate_prev','scan_rate_prev','gaze_depth_prev','elapsed_s']


def iter_trials(dataset, splits=None, verify=False):
    dataset = Path(dataset)
    manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    validate_manifest_scope(manifest)
    for record in manifest['trials']:
        if splits is not None and record['split'] not in splits:
            continue
        path = (dataset/record['file']).resolve()
        if not path.is_relative_to(dataset.resolve()):
            raise ValueError('Trial path outside dataset')
        if verify and hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Dataset checksum mismatch: '+str(path))
        df = pd.read_csv(path)
        for column in ['trial_id','participant_id','guidance','smoke','split']:
            if column not in df or not df[column].eq(record[column]).all():
                raise ValueError('Trial identity differs from manifest: '+str(path)+' / '+column)
        yield record, df


def audit_dataset(dataset, verify=True):
    """No fit/optimizer/MDP estimation here: safe pre-training inspection."""
    totals = defaultdict(lambda: dict(rows=0, valid=0, zero=0, one=0, trials=0))
    manifest=json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))
    max_gap=manifest['config']['max_gap_s']
    people, seen, constant, mixed = {}, set(), 0, 0
    for rec, df in tqdm(iter_trials(dataset, verify=verify), desc='Audit data', total=len(manifest['trials'])):
        pid, split = rec['participant_id'], rec['split']
        if pid in people and people[pid] != split:
            raise ValueError('Participant leakage across splits')
        people[pid] = split
        if rec['trial_id'] in seen:
            raise ValueError('Duplicate trial id')
        seen.add(rec['trial_id'])
        if len(df) != rec['rows'] or not np.all(np.diff(df.timestamp_ns.to_numpy(np.int64)) > 0):
            raise ValueError('Row count or time order differs')
        if not np.array_equal(df.timestamp_ns.to_numpy(),df.subject_timestamp_ns.to_numpy()):
            raise ValueError('V3 must preserve original Subject timestamps')
        if not np.array_equal(df.frame_index.to_numpy(),np.arange(len(df))):
            raise ValueError('V3 source frame order lost')
        dt=np.diff(df.timestamp_ns.to_numpy(np.int64))/1e9
        if not np.allclose(df.dt_next_s.to_numpy()[:-1],dt): raise ValueError('Frame durations disagree')
        if not np.isfinite(df.run_weight).all() or (df.run_weight<0).any(): raise ValueError('Invalid duration weights')
        if not set(df.action.unique()) <= {-1,0,1}:
            raise ValueError('Invalid actions')
        v = df.valid.eq(1)
        if v.any() and (df.loc[v, NUMERIC[:5]].isna().any().any() or not df.loc[v,'action'].isin([0,1]).all()):
            raise ValueError('Invalid training rows marked valid')
        if (v & df.after_exit.eq(1)).any():raise ValueError('Post-exit observations marked valid')
        nxt = df.transition_valid.eq(1).to_numpy()
        eligible = v.to_numpy() & np.r_[v.to_numpy()[1:],False]
        eligible &= df.segment_id.to_numpy() == np.r_[df.segment_id.to_numpy()[1:],-2]
        eligible &= np.r_[(dt>0)&(dt<=max_gap),False]
        if not np.array_equal(nxt,eligible):
            raise ValueError('Transition bridges invalid rows or segments')
        nonstart = v & df.segment_id.eq(df.segment_id.shift(1))
        if not (df.loc[nonstart,'previous_action'].to_numpy() == df.action.shift(1)[nonstart].to_numpy()).all():
            raise ValueError('Previous action alignment failure')
        t = totals[rec['split']+'/'+rec['guidance']]
        t['rows'] += len(df); t['valid'] += int(v.sum()); t['trials'] += 1
        t['zero'] += int((v & df.action.eq(0)).sum()); t['one'] += int((v & df.action.eq(1)).sum())
        if v.any():
            constant += int(df.loc[v,'action'].nunique() == 1)
            mixed += int(df.loc[v,'action'].nunique() == 2)
    if len(seen) != manifest['cohort_trials'] or len(people) != manifest['cohort_participants']:
        raise ValueError('Cohort totals differ from manifest')
    if any(manifest['participant_splits'].get(pid) != split for pid, split in people.items()):
        raise ValueError('Participant split differs from canonical split map')
    trial_files = {p.resolve() for p in (Path(dataset)/'trials').glob('*.csv.gz')}
    declared_files = {(Path(dataset)/r['file']).resolve() for r in manifest['trials']}
    if trial_files != declared_files:
        raise ValueError('Trial directory contains missing or undeclared trial files')
    return dict(trials=len(seen), participants=len(people), cohort_policy=COHORT_POLICY,
                excluded_trials=manifest['excluded_trials'], by_split_guidance=dict(totals),
                constant_supported_trials=constant, mixed_supported_trials=mixed,
                checks='passed', trained=False)


class StateEncoder:
    """Train-only robust scaling; explicit missing flags and fixed categories."""
    def numeric(self, df):
        x = df[NUMERIC].to_numpy(float).copy()
        for name in LOG_COLUMNS:
            j = NUMERIC.index(name)
            x[:,j] = np.log1p(np.maximum(0,x[:,j]))
        x[~np.isfinite(x)] = np.nan
        return x

    def fit(self, df):
        x = self.numeric(df)
        self.median = np.array([np.nanmedian(x[:,j]) if np.isfinite(x[:,j]).any() else 0 for j in range(x.shape[1])])
        imputed = np.where(np.isfinite(x),x,self.median)
        self.scale = np.quantile(imputed,.75,axis=0)-np.quantile(imputed,.25,axis=0)
        self.scale[self.scale < .01] = 1.
        return self

    @property
    def names(self):
        return ['bias']+['scaled_'+c for c in NUMERIC]+['missing_'+c for c in NUMERIC]+[
            'previous_action_'+str(a) for a in [-1,0,1]]+['floor_'+f for f in ['Hall','Platform']]+[
            'stage_'+s for s in STAGES]+['high_smoke']

    def transform(self, df):
        x = self.numeric(df)
        missing = ~np.isfinite(x)
        z = np.clip((np.where(missing,self.median,x)-self.median)/self.scale,-8,8)
        cats = [df.previous_action.eq(a).to_numpy() for a in [-1,0,1]]
        cats += [df.floor.eq(f).to_numpy() for f in ['Hall','Platform']]
        cats += [df.stage.eq(s).to_numpy() for s in STAGES]
        cats += [df.smoke.eq('HighSmoke').to_numpy()]
        return np.column_stack([np.ones(len(df)),z,missing,*cats]).astype(np.float32)

    def as_dict(self):
        return dict(median=self.median.tolist(),scale=self.scale.tolist(),names=self.names,
                    log1p_columns=LOG_COLUMNS,clip_scaled=8)


def context_keys(df):
    return df.guidance.astype(str)+'/'+df.smoke.astype(str)+'/'+df.previous_action.astype(str)


def nearest(x, centers):
    # Float32 norm/dot cancellation previously assigned one near-tied training
    # frame to different states in full-array and trial-wise cache passes.
    from irl_state_distance import stable_nearest
    return stable_nearest(x, centers)


def assign_states(df, x, contexts, centers):
    labels = np.full(len(df),-1,int)
    distances = np.full(len(df),np.inf)
    keys = context_keys(df).to_numpy()
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        if key not in contexts:
            continue
        offset,count = contexts[key]
        k,d = nearest(x[idx],centers[offset:offset+count])
        labels[idx],distances[idx] = offset+k,d
    return labels,distances


def build_mdp(dataset, clusters=24, seed=20260926, prior=.5):
    """Called ONLY by --train. Fits clustering and transition estimates on train."""
    frames=[]
    for rec,df in iter_trials(dataset,{'train'}):
        if rec['guidance'] not in GUIDANCE:continue
        max_gap=json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))['config']['max_gap_s']
        df['_terminal_next'] = (df.valid.eq(1) & df.after_exit.eq(0) & df.after_exit.shift(-1).eq(1)
                                & df.dt_next_s.gt(0) & df.dt_next_s.le(max_gap))
        frames.append(df.loc[df.valid.eq(1)].copy())
    train = pd.concat(frames,ignore_index=True)
    if train.empty or train.action.nunique() < 2:
        raise ValueError('Training set lacks both actions')
    encoder = StateEncoder().fit(train)
    x = encoder.transform(train)
    keys = context_keys(train).to_numpy()
    centers,contexts,offset = [],{},0
    for key in sorted(np.unique(keys)):
        idx = np.flatnonzero(keys == key)
        k = min(clusters,max(1,len(idx)//50))
        model = MiniBatchKMeans(n_clusters=k,random_state=seed,n_init=3,batch_size=2048)
        model.fit(x[idx],sample_weight=train.run_weight.to_numpy()[idx])
        centers.append(model.cluster_centers_)
        contexts[key] = (offset,k); offset += k
    centers = np.concatenate(centers).astype(np.float32)
    s,_ = assign_states(train,x,contexts,centers)
    terminal = len(centers)
    centers = np.vstack([centers,np.zeros((1,centers.shape[1]),np.float32)])
    count = len(centers)
    radius = np.zeros(count, dtype=np.float32)
    for state in range(count):
        distance = np.linalg.norm(x[s == state]-centers[state], axis=1)
        radius[state] = max(.1,float(np.quantile(distance,.99))) if len(distance) else .1
    transition_counts = np.zeros((count,2,count),np.float64)
    raw_counts = np.zeros((count,2),np.float64)
    weighted_counts = np.zeros_like(raw_counts)
    trap_sum = np.zeros_like(raw_counts)
    # Equal participants, equal supported trials/person; cap each constant-action run.
    ntrials = train.groupby('participant_id').trial_id.nunique().to_dict()
    weight = train.run_weight.to_numpy(float).copy()
    for trial_id, idx in train.groupby('trial_id',sort=False).indices.items():
        weight[idx] /= weight[idx].sum()*ntrials[train.iloc[idx[0]].participant_id]
    a = train.action.to_numpy(int)
    np.add.at(raw_counts,(s,a),1)
    np.add.at(weighted_counts,(s,a),weight)
    np.add.at(trap_sum,(s,a),train.trap_observed.to_numpy(float))
    # Filtering valid rows cannot create links: original timestamp/segment checks.
    step = json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))['config']['step_s']
    max_gap = json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))['config']['max_gap_s']
    ok = (train.trial_id.to_numpy()[:-1] == train.trial_id.to_numpy()[1:])
    ok &= train.segment_id.to_numpy()[:-1] == train.segment_id.to_numpy()[1:]
    interval = np.diff(train.timestamp_ns.to_numpy(np.int64))/1e9
    ok &= (interval>0) & (interval<=max_gap)
    ok &= np.diff(train.frame_index.to_numpy(np.int64)) == 1
    ok &= train.transition_valid.to_numpy()[:-1] == 1
    i = np.flatnonzero(ok)
    np.add.at(transition_counts,(s[i],a[i],s[i+1]),1)
    ending = np.flatnonzero(train._terminal_next.to_numpy(bool))
    np.add.at(transition_counts,(s[ending],a[ending],np.full(len(ending),terminal)),1)
    transitions = np.zeros_like(transition_counts)
    transition_n = transition_counts.sum(2)
    duration_sum=np.zeros((count,2),np.float64)
    np.add.at(duration_sum,(s[i],a[i]),interval[i])
    np.add.at(duration_sum,(s[ending],a[ending]),train.dt_next_s.to_numpy()[ending])
    mean_dt=np.full((count,2),step,np.float32)
    context_by_state = np.empty(count,object)
    guidance_index = np.empty(count,int)
    for key,(lo,n) in contexts.items():
        context_by_state[lo:lo+n] = key.rsplit('/',1)[0]
        guidance_index[lo:lo+n] = GUIDANCE.index(key.split('/')[0])
    context_by_state[terminal]='terminal';guidance_index[terminal]=0
    transitions[terminal,:,terminal]=1
    for state in range(terminal):
        for action in [0,1]:
            donors = np.flatnonzero((context_by_state == context_by_state[state]) & (transition_n[:,action] > 0))
            if not len(donors):
                raise ValueError('No transitions for action/context; cannot estimate IRL MDP')
            distances = ((centers[donors]-centers[state])**2).sum(1)
            donors = donors[np.argsort(distances)[:3]]
            donor_dt=duration_sum[donors,action].sum()/transition_n[donors,action].sum()
            mean_dt[state,action]=(duration_sum[state,action]+prior*donor_dt)/(transition_n[state,action]+prior)
            pool = transition_counts[donors,action].sum(0)
            pool /= pool.sum()
            empirical = transition_counts[state,action]
            transitions[state,action] = (empirical+prior*pool)/(empirical.sum()+prior)
    # Empirical joint event cost, used for diagnosis ONLY, never as training reward/label input.
    costs = np.zeros_like(raw_counts)
    for state in range(terminal):
        for action in [0,1]:
            group = context_by_state == context_by_state[state]
            base = trap_sum[group,action].sum()/max(1,raw_counts[group,action].sum())
            costs[state,action] = (trap_sum[state,action]+prior*base)/(raw_counts[state,action]+prior)
    if not np.isfinite(transitions).all() or not np.allclose(transitions.sum(2),1):
        raise ValueError('Invalid transition matrix')
    return dict(encoder=encoder, centers=centers, contexts=contexts, radius=radius, terminal_index=terminal,
        transitions=transitions.astype(np.float32), counts=weighted_counts.astype(np.float32),
        raw_counts=raw_counts, transition_counts=transition_n,
        guidance_index=guidance_index, event_cost=costs.astype(np.float32),
        step_s=step, mean_dt=mean_dt, train_rows=len(train), seed=seed)
