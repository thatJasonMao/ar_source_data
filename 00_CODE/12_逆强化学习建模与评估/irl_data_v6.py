"""V6 channel observations and condition-specific empirical state dynamics."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '0_RL_SOURCE'))
from IRL_Channel_Cohort_V6 import validate_manifest as validate_manifest_scope, CHANNEL_COLUMNS, COHORT_POLICY
from Guidance_Channel_Contract import CHANNELS as GUIDANCE, AVAILABLE, digest
from irl_data_v3 import StateEncoder, NUMERIC, STAGES, nearest


def channel_records(manifest, splits=None):
    result = []
    for r in manifest['trials']:
        if splits is not None and r['split'] not in splits:
            continue
        for c in r['channels']:
            result.append(dict(r, source_trial_id=r['trial_id'], trial_id=r['trial_id'] + ':' + c,
                guidance=c, valid_rows=r['channel_valid_rows'][c]))
    return result


def trial_masses(records):
    """Equal channels, available conditions, people/condition and trials/person.

The same Full frame supplies several correlated channel observations, not
several independent participants. All splits retain source-person identities.
"""
    active = [r for r in records if r['valid_rows'] > 0]
    channels = {r['guidance'] for r in active}
    conditions, people, trials = defaultdict(set), defaultdict(set), defaultdict(int)
    for r in active:
        c, e, p = r['guidance'], r['condition'], r['participant_id']
        conditions[c].add(e); people[c, e].add(p); trials[c, e, p] += 1
    result = {}
    for r in active:
        c, e, p = r['guidance'], r['condition'], r['participant_id']
        result[r['trial_id']] = 1. / (len(channels) * len(conditions[c]) * len(people[c, e]) * trials[c, e, p])
    return result


def iter_trials(dataset, splits=None, verify=False):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    validate_manifest_scope(manifest)
    for record in manifest['trials']:
        if splits is not None and record['split'] not in splits:
            continue
        path = (dataset / record['file']).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError('Trial outside dataset')
        if verify and digest(path) != record['sha256']:
            raise ValueError('Trial checksum differs: ' + str(path))
        wide = pd.read_csv(path)
        for key in ['trial_id', 'participant_id', 'condition', 'smoke', 'split']:
            if not wide[key].eq(record[key]).all():
                raise ValueError('Frame identity differs: ' + key)
        if len(wide) != record['rows']:
            raise ValueError('Frame count differs')
        suffixed = [f + '_' + c for f in CHANNEL_COLUMNS for c in GUIDANCE]
        for c in GUIDANCE:
            if c not in record['channels'] and (not wide['action_' + c].eq(-1).all() or wide['valid_' + c].any()):
                raise ValueError('Unavailable channel is not masked')
        for rec in channel_records(dict(trials=[record])):
            c = rec['guidance']
            frame = wide.drop(columns=suffixed).copy()
            for f in CHANNEL_COLUMNS:
                frame[f] = wide[f + '_' + c].to_numpy()
            frame['source_trial_id'] = record['trial_id']
            frame['trial_id'] = rec['trial_id']
            frame['guidance'] = c
            yield rec, frame


def audit_dataset(dataset, verify=True):
    dataset = Path(dataset)
    m = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    validate_manifest_scope(m)
    totals, people, seen = defaultdict(lambda: dict(rows=0, valid=0, one=0, zero=0, trials=0)), {}, set()
    expected_files = {(dataset / r['file']).resolve() for r in m['trials']}
    if expected_files != {p.resolve() for p in (dataset / 'trials').glob('*.csv.gz')}:
        raise ValueError('Unexpected or missing trial files')
    for rec, df in tqdm(iter_trials(dataset, verify=verify), total=m['channel_trials'], desc='Audit V6 channel sequences'):
        pid = rec['participant_id']
        if pid in people and people[pid] != rec['split']:
            raise ValueError('Participant leakage')
        people[pid] = rec['split']; seen.add(rec['trial_id'])
        ns = df.timestamp_ns.to_numpy(np.int64); dt = np.diff(ns) / 1e9
        if np.any(dt <= 0) or not np.array_equal(ns, df.subject_timestamp_ns.to_numpy(np.int64)):
            raise ValueError('Original timestamps lost')
        if not np.array_equal(df.frame_index, np.arange(len(df))) or not np.allclose(df.dt_next_s.to_numpy()[:-1], dt):
            raise ValueError('Original frame order/durations differ')
        if not set(df.action.unique()) <= {-1, 0, 1}:
            raise ValueError('Invalid actions')
        v = df.valid.eq(1).to_numpy()
        if int(v.sum()) != rec['valid_rows'] or np.any(v & ((df.action < 0) | df.after_exit.eq(1))):
            raise ValueError('Invalid valid-row contract')
        if not np.isfinite(df.loc[v, NUMERIC[:5]].to_numpy(float)).all():
            raise ValueError('Invalid spatial features')
        linked = np.r_[False, (dt > 0) & (dt <= m['config']['max_gap_s'])]
        seg = df.segment_id.to_numpy()
        edges = v & np.r_[v[1:], False] & (seg == np.r_[seg[1:], -2]) & np.r_[linked[1:], False]
        if not np.array_equal(edges, df.transition_valid.to_numpy(bool)):
            raise ValueError('Transition bridges gap/invalid frame')
        previous = np.r_[-1, df.action.to_numpy()[:-1]]
        previous[~(v & (seg == np.r_[-2, seg[:-1]]) & linked)] = -1
        if not np.array_equal(previous, df.previous_action):
            raise ValueError('Previous channel action alignment differs')
        if not np.isfinite(df.run_weight).all() or (df.run_weight < 0).any() or (df.loc[~v, 'run_weight'] != 0).any():
            raise ValueError('Invalid exposure')
        t = totals['/'.join([rec['split'], rec['guidance'], rec['condition'], rec['smoke']])]
        t['rows'] += len(df); t['valid'] += int(v.sum()); t['trials'] += 1
        t['one'] += int((v & df.action.eq(1)).sum()); t['zero'] += int((v & df.action.eq(0)).sum())
    if len(seen) != m['channel_trials'] or len(people) != m['cohort_participants']:
        raise ValueError('Cohort/channel totals differ')
    masses = {}
    for split in ['train', 'val', 'test']:
        records = channel_records(m, {split}); weights = trial_masses(records)
        masses[split] = {c: sum(weights.get(r['trial_id'], 0) for r in records if r['guidance'] == c) for c in GUIDANCE}
        if not all(np.isclose(v, .25) for v in masses[split].values()):
            raise ValueError('Missing channel in split')
    return dict(checks='passed', trained=False, trials=m['cohort_trials'], channel_trials=len(seen),
        participants=len(people), by_split_channel_condition=dict(totals), channel_weight_mass=masses,
        loss='condition-balanced channel composite likelihood; correlated channel rows are not independent subjects')


def context_keys(df):
    return df.guidance.astype(str) + '/' + df.condition.astype(str) + '/' + df.smoke.astype(str) + '/' + df.previous_action.astype(str)


def assign_states(df, x, contexts, centers):
    labels = np.full(len(df), -1, int); distances = np.full(len(df), np.inf)
    keys = context_keys(df).to_numpy()
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key)
        if key in contexts:
            lo, n = contexts[key]; candidates = np.arange(lo, lo + n)
        else:
            # Rare previous actions can be absent in train. Never cross the
            # channel/experimental-condition/smoke boundary during fallback.
            ranges = [np.arange(lo, lo + n) for k, (lo, n) in contexts.items() if k.rsplit('/', 1)[0] == key.rsplit('/', 1)[0]]
            if not ranges:
                continue
            candidates = np.concatenate(ranges)
        k, d = nearest(x[indices], centers[candidates])
        labels[indices], distances[indices] = candidates[k], d
    return labels, distances


def build_mdp(dataset, clusters=24, seed=20260926, prior=.5, run_cap_s=2.):
    """Called ONLY by --train. Fits clustering and transition estimates on train."""
    from irl_search_v6 import run_duration_table, recapped_weights
    manifest=json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))
    durations=run_duration_table(dataset)
    old_cap=manifest['config']['run_cap_s']
    frames=[]
    for rec,df in iter_trials(dataset,{'train'}):
        if rec['guidance'] not in GUIDANCE:continue
        max_gap=json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))['config']['max_gap_s']
        df['_terminal_next'] = (df.valid.eq(1) & df.after_exit.eq(0) & df.after_exit.shift(-1).eq(1)
                                & df.dt_next_s.gt(0) & df.dt_next_s.le(max_gap))
        df['run_weight']=recapped_weights(df,run_cap_s,old_cap,durations[rec['trial_id']])
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
    masses = trial_masses(channel_records(manifest, {'train'}))
    weight = train.run_weight.to_numpy(float).copy()
    for trial_id, idx in train.groupby('trial_id',sort=False).indices.items():
        weight[idx] *= masses[trial_id] / weight[idx].sum()
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
