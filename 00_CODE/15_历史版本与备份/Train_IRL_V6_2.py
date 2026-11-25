"""V6_2: independent own-condition spatial IRL, conditional on smoke.

StaticSign uses ONLY the StaticSign condition. ArrowRibbon/MiniMap/TextBar
each use ONLY their own condition (static signs remain in the experiment).
Full is excluded. Every retained person/condition has one HighSmoke and one
LowSmoke trial. The two trials are fitted together; V6_2 exports both smoke
fields. Each channel has its own encoder, MDP, basis, optimizer and selection.

The V3_Optimize methodology is implemented here: duration-capped full-frame
binary soft-Bellman likelihood, train-only state/basis fitting, validation NLL
early stopping, then maximum spatial AUC within minimum validation person NLL
+ 0.005. No legacy trainer, monkeypatch, checkpoint or cache is used. Generic
encoding, geometry, soft-Bellman and basis primitives are shared read-only.

Train_IRL_V6_3.py calls this engine with a distinct, smoke-pooled contract:
both smoke trials are retained, four isovist features are recomputed at the
same 12.5 m radius, smoke is removed from both state and reward inputs, and
each channel exports ONE field. This is not averaging two fitted fields.

Examples (from the project root):
  python -B 0_RL_FRAMEWORK/Train_IRL_V6_2.py --dataset <V6-wide-dataset>
  python -B 0_RL_FRAMEWORK/Train_IRL_V6_2.py --dataset <V6-wide-dataset> --train
  python -B 0_RL_FRAMEWORK/Train_IRL_V6_2.py --plot-run <Best_Model>
  python -B 0_RL_FRAMEWORK/Train_IRL_V6_2.py --run <Best_Model> \
      --query X Y Z --guidance MiniMap --smoke HighSmoke

Without --train/--plot-run/--query, only a JSON plan is printed. Import does
not fit, query geometry, create directories or launch processes. The budget
is a soft search/diagnostic wall-clock limit; preparation and final export
can overrun it. Incomplete channel coverage is an error, never a partial bank.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (ROOT, ROOT / '0_RL_SOURCE'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Guidance_Channel_Contract import CHANNELS, AVAILABLE
from IRL_Channel_Cohort_V6 import CHANNEL_COLUMNS, validate_manifest
from irl_data_v3 import StateEncoder, NUMERIC, STAGES, nearest
from irl_core_v3 import soft_policy
from irl_search_v4 import write_json, digest, config_id, recapped_weights, run_duration_table
from spatial_reward_basis import SpatialRewardBasis

SMOKES = ('HighSmoke', 'LowSmoke')
VISIBILITY = {'HighSmoke': 7.5, 'LowSmoke': 12.5}
SPATIAL = NUMERIC[:5]
FAMILIES = ('linear', 'spline', 'interaction', 'rbf32', 'rbf64')
SCHEMA = 'independent_own_condition_spatial_irl_v1'
SELECTION = 'max validation spatial AUC within min validation participant-macro NLL + 0.005'
FRAME_KEYS = ('x', 'state', 'action', 'weight', 'person', 'smoke', 'trial', 'frame_index')


@dataclass(frozen=True)
class Profile:
    variant: str = 'V6_2'
    pooled_smoke: bool = False
    visibility_m: float | None = None

    def __post_init__(self):
        expected = {'V6_2': (False, None), 'V6_3': (True, 12.5)}
        if self.variant not in expected or (self.pooled_smoke, self.visibility_m) != expected[self.variant]:
            raise ValueError('Unsupported variant/smoke/geometry contract')


class Encoder(StateEncoder):
    """Pooled smoke has no smoke coordinate, including during state clustering."""
    def __init__(self, pooled_smoke=False):
        self.pooled_smoke = bool(pooled_smoke)

    @property
    def names(self):
        names = super().names
        return [n for n in names if n != 'high_smoke'] if self.pooled_smoke else names

    def transform(self, frame):
        encoded = super().transform(frame)
        if self.pooled_smoke:
            keep = [n != 'high_smoke' for n in super().names]
            encoded = encoded[:, keep]
        return encoded


class Reward(torch.nn.Module):
    """One channel's immediate action contrast; no cross-channel parameters."""
    def __init__(self, features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(features))

    def forward(self, features):
        contrast = features @ self.weight
        return torch.stack((-contrast / 2, contrast / 2), dim=1)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def code_hashes(profile):
    names = ['0_RL_FRAMEWORK/Train_IRL_V6_2.py', '0_RL_FRAMEWORK/irl_data_v3.py',
             '0_RL_FRAMEWORK/irl_state_distance.py', '0_RL_FRAMEWORK/irl_core_v3.py',
             '0_RL_FRAMEWORK/irl_search_v4.py', '0_RL_FRAMEWORK/spatial_reward_basis.py',
             '0_RL_SOURCE/IRL_Channel_Cohort_V6.py', 'Guidance_Channel_Contract.py',
             '0_RL_SOURCE/spatial_isovist_core.py', '0_RL_SOURCE/Analyze_NextDecisionDistance.py',
             'SemanticZoneQuery.py', 'UnifiedZoneQuery.py']
    if profile.variant == 'V6_3':
        names.append('0_RL_FRAMEWORK/Train_IRL_V6_3.py')
    return {name: digest(ROOT / name) for name in names}


def spatial_mask(names):
    allowed = {'bias', 'high_smoke', *('scaled_' + n for n in SPATIAL)}
    return np.asarray([n in allowed or n.startswith(('floor_', 'stage_')) for n in names], bool)


def records_for(manifest, channel, splits=None):
    if channel not in CHANNELS:
        raise ValueError('Unknown channel')
    return [r for r in manifest['trials'] if r['condition'] == channel
            and (splits is None or r['split'] in splits)]


def audit_scope(dataset):
    """Manifest-only scope audit. Test observations are not opened here."""
    dataset = Path(dataset).resolve()
    manifest = read_json(dataset / 'manifest.json')
    validate_manifest(manifest)
    selected, summaries = [], {}
    for channel in CHANNELS:
        records = records_for(manifest, channel)
        pairs = defaultdict(list)
        for r in records:
            if channel not in r['channels'] or channel not in AVAILABLE[r['condition']]:
                raise ValueError('Missing own-condition channel: ' + r['trial_id'])
            if r['channel_valid_rows'][channel] <= 0:
                raise ValueError('Empty trial would break the smoke pair: ' + r['trial_id'])
            if r['split'] != manifest['participant_splits'][r['participant_id']]:
                raise ValueError('Participant split mismatch')
            if not (dataset / r['file']).resolve().is_relative_to(dataset):
                raise ValueError('Trial outside dataset')
            pairs[r['participant_id']].append(r)
        for person, trials in pairs.items():
            if Counter(r['smoke'] for r in trials) != Counter(SMOKES):
                raise ValueError(f'Expected one high/one low trial: {channel}/{person}')
            if len({r['split'] for r in trials}) != 1:
                raise ValueError('Participant leakage')
        by_split = {}
        for split in ('train', 'val', 'test'):
            rows = [r for r in records if r['split'] == split]
            if not rows:
                raise ValueError(f'Empty split: {channel}/{split}')
            by_split[split] = dict(trials=len(rows), people=len({r['participant_id'] for r in rows}),
                                  smoke_trials=dict(Counter(r['smoke'] for r in rows)))
        summaries[channel] = dict(trials=len(records), people=len(pairs), by_split=by_split)
        selected.extend(records)
    if len({r['trial_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate selected source trial')
    return manifest, dict(checks='passed', scope='own condition only; Full excluded',
                          selected_trials=len(selected), source_trials=len(manifest['trials']),
                          by_channel=summaries, test_observations_opened=False)


def geometry_contract(manifest):
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    path = Path(DEFAULT_GEOMETRY_PATH).resolve()
    checksum = digest(path)
    if manifest['source_hashes'].get(str(path)) != checksum:
        raise ValueError('Geometry differs from the source dataset')
    return dict(path=str(path), sha256=checksum)


def validate_trial(frame, record, channel, manifest):
    """Check the chosen channel only; do not introduce OR labels or row sampling."""
    if len(frame) != record['rows']:
        raise ValueError('Source row count changed')
    for key in ('trial_id', 'participant_id', 'condition', 'smoke', 'split'):
        if not frame[key].eq(record[key]).all():
            raise ValueError('Source identity differs: ' + key)
    ns = frame.timestamp_ns.to_numpy(np.int64)
    dt = np.diff(ns) / 1e9
    if (not np.array_equal(ns, frame.subject_timestamp_ns.to_numpy(np.int64))
            or np.any(dt <= 0) or not np.array_equal(frame.frame_index, np.arange(len(frame)))
            or not np.allclose(frame.dt_next_s.to_numpy()[:-1], dt)):
        raise ValueError('Original frame timestamps/order/durations differ')
    valid = frame.valid.eq(1).to_numpy()
    action = frame.action.to_numpy()
    if (not set(np.unique(action)) <= {-1, 0, 1}
            or int(valid.sum()) != record['channel_valid_rows'][channel]
            or np.any(valid & ((action < 0) | frame.after_exit.eq(1).to_numpy()))):
        raise ValueError('Invalid label/valid-row contract')
    if not np.isfinite(frame.loc[valid, SPATIAL].to_numpy(float)).all():
        raise ValueError('Nonfinite spatial inputs on a valid frame')
    seg = frame.segment_id.to_numpy()
    linked = (dt > 0) & (dt <= manifest['config']['max_gap_s'])
    edges = valid & np.r_[valid[1:], False] & (seg == np.r_[seg[1:], -2]) & np.r_[linked, False]
    previous = np.r_[-1, action[:-1]]
    previous[~(valid & (seg == np.r_[-2, seg[:-1]]) & np.r_[False, linked])] = -1
    if (not np.array_equal(edges, frame.transition_valid.to_numpy(bool))
            or not np.array_equal(previous, frame.previous_action.to_numpy())):
        raise ValueError('Channel history/transition crosses a gap or invalid row')
    weights = frame.run_weight.to_numpy(float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights[~valid] != 0):
        raise ValueError('Invalid source exposure weights')


class TrialReader:
    """Reads one selected channel; pooled geometry caches live inside this run."""
    def __init__(self, dataset, manifest, profile, root, geometry):
        self.dataset, self.manifest, self.profile = Path(dataset), manifest, profile
        self.root, self.geometry = Path(root), geometry
        self.durations = run_duration_table(self.dataset)
        self._spatial = None

    @lru_cache(maxsize=8192)
    def fixed_metrics(self, x, z, floor):
        from spatial_isovist_core import PlanarIsovist
        if self._spatial is None:
            self._spatial = PlanarIsovist(self.geometry['path'])
        result = self._spatial.query(x, None, z, self.profile.visibility_m, floor=floor)
        if result.status != 'ok':
            raise ValueError('Fixed-radius query invalid on a source-valid frame: ' + result.status)
        values = tuple(getattr(result, n) for n in SPATIAL[:4])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite fixed-radius metrics')
        return values

    def fixed_geometry(self, frame, record):
        """Exact source positions, no trajectory rasterization or label filtering."""
        folder = self.root / 'Fixed_Geometry'
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / (record['trial_id'] + '.npz')
        meta = folder / (record['trial_id'] + '.json')
        valid = np.flatnonzero(frame.valid.eq(1).to_numpy())
        contract = dict(profile=asdict(self.profile), source_sha256=record['sha256'],
                        geometry=self.geometry, code_sha256=code_hashes(self.profile),
                        metrics=SPATIAL[:4], position_policy='exact source x/z/floor; no rounding')
        if file.exists() or meta.exists():
            saved = read_json(meta)
            if saved.get('contract') != contract or digest(file) != saved.get('sha256'):
                raise ValueError('Fixed-geometry cache contract changed')
            with np.load(file, allow_pickle=False) as arrays:
                if not np.array_equal(arrays['frame_index'], valid):
                    raise ValueError('Fixed geometry has different frame identities')
                metrics = arrays['metrics']
        else:
            coordinates = frame.loc[valid, ['x', 'z', 'floor']]
            metrics = np.asarray([self.fixed_metrics(float(x), float(z), str(floor))
                                  for x, z, floor in tqdm(coordinates.itertuples(index=False, name=None),
                                                         total=len(valid), desc='12.5m ' + record['trial_id'])])
            np.savez_compressed(file, frame_index=valid, metrics=metrics)
            write_json(meta, dict(contract=contract, sha256=digest(file)))
        if metrics.shape != (len(valid), 4) or not np.isfinite(metrics).all():
            raise ValueError('Invalid fixed geometry cache')
        frame.loc[valid, SPATIAL[:4]] = metrics
        return frame

    def trials(self, channel, splits, cap):
        for record in records_for(self.manifest, channel, splits):
            file = (self.dataset / record['file']).resolve()
            if not file.is_relative_to(self.dataset.resolve()) or digest(file) != record['sha256']:
                raise ValueError('Source trial checksum/path changed: ' + record['trial_id'])
            wide = pd.read_csv(file)
            selected = {f + '_' + channel: f for f in CHANNEL_COLUMNS}
            drops = [f + '_' + c for c in CHANNELS if c != channel for f in CHANNEL_COLUMNS]
            frame = wide.drop(columns=drops).rename(columns=selected)
            validate_trial(frame, record, channel, self.manifest)
            if self.profile.pooled_smoke:
                frame = self.fixed_geometry(frame, record)
            frame['source_trial_id'] = record['trial_id']
            frame['guidance'] = channel
            # Run IDs are channel-specific in the V6 wide-table action-run file.
            key = record['trial_id'] + ':' + channel
            frame['run_weight'] = recapped_weights(frame, cap, self.manifest['config']['run_cap_s'],
                                                   self.durations[key])
            yield record, frame


def context_keys(frame, profile):
    smoke = pd.Series('PooledSmoke', index=frame.index) if profile.pooled_smoke else frame.smoke.astype(str)
    return frame.guidance.astype(str) + '/' + smoke + '/' + frame.previous_action.astype(str)


def assign_states(frame, x, contexts, centers, profile):
    labels = np.full(len(frame), -1, np.int32)
    keys = context_keys(frame, profile).to_numpy()
    for key in np.unique(keys):
        ix = np.flatnonzero(keys == key)
        matches = [contexts[key]] if key in contexts else [v for k, v in contexts.items()
                      if k.rsplit('/', 1)[0] == key.rsplit('/', 1)[0]]
        if not matches:
            raise ValueError('No training support for context: ' + key)
        candidates = np.concatenate([np.arange(lo, lo + count) for lo, count in matches])
        local, _ = nearest(x[ix], centers[candidates])
        labels[ix] = candidates[local]
    return labels


def load_split(reader, channel, split, cap):
    records = records_for(reader.manifest, channel, {split})
    people = sorted({r['participant_id'] for r in records})
    ntrials = Counter(r['participant_id'] for r in records)
    pieces = []
    for record, frame in reader.trials(channel, {split}, cap):
        terminal = (frame.valid.eq(1) & frame.after_exit.eq(0) & frame.after_exit.shift(-1).eq(1)
                    & frame.dt_next_s.gt(0) & frame.dt_next_s.le(reader.manifest['config']['max_gap_s']))
        frame['_terminal_next'] = terminal
        part = frame.loc[frame.valid.eq(1)].copy()
        mass = part.run_weight.sum()
        if len(part) == 0 or not np.isfinite(mass) or mass <= 0:
            raise ValueError('Empty supported trial: ' + record['trial_id'])
        part['_weight'] = part.run_weight / mass / len(people) / ntrials[record['participant_id']]
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    if not np.isclose(result._weight.sum(), 1):
        raise ValueError('Participant/trial weights do not sum to one')
    return result, people


def build_mdp(train, profile, config, step_s, max_gap):
    """Independent empirical MDP; all links retain original trial boundaries."""
    if train.action.nunique() != 2:
        raise ValueError('Training channel lacks both actions')
    encoder = Encoder(profile.pooled_smoke).fit(train)
    x = encoder.transform(train)
    keys = context_keys(train, profile).to_numpy()
    centers, contexts, offset = [], {}, 0
    for key in sorted(np.unique(keys)):
        ix = np.flatnonzero(keys == key)
        count = min(config['clusters'], max(1, len(ix) // 50))
        clustering = MiniBatchKMeans(n_clusters=count, random_state=config['seed'], n_init=3, batch_size=2048)
        clustering.fit(x[ix], sample_weight=train.run_weight.to_numpy()[ix])
        centers.append(clustering.cluster_centers_)
        contexts[key] = (offset, count)
        offset += count
    centers = np.concatenate(centers).astype(np.float32)
    state = assign_states(train, x, contexts, centers, profile)
    terminal = len(centers)
    centers = np.vstack((centers, np.zeros((1, centers.shape[1]), np.float32)))
    count = len(centers)
    actions = train.action.to_numpy(int)
    exposure = np.zeros((count, 2), np.float64)
    np.add.at(exposure, (state, actions), train._weight.to_numpy(float))
    transitions_n = np.zeros((count, 2, count), np.float64)
    interval = np.diff(train.timestamp_ns.to_numpy(np.int64)) / 1e9
    linked = train.trial_id.to_numpy()[:-1] == train.trial_id.to_numpy()[1:]
    linked &= train.segment_id.to_numpy()[:-1] == train.segment_id.to_numpy()[1:]
    linked &= (interval > 0) & (interval <= max_gap)
    linked &= np.diff(train.frame_index.to_numpy()) == 1
    linked &= train.transition_valid.to_numpy()[:-1] == 1
    ix = np.flatnonzero(linked)
    ending = np.flatnonzero(train._terminal_next.to_numpy(bool))
    np.add.at(transitions_n, (state[ix], actions[ix], state[ix + 1]), 1)
    np.add.at(transitions_n, (state[ending], actions[ending], np.full(len(ending), terminal)), 1)
    durations = np.zeros((count, 2), np.float64)
    np.add.at(durations, (state[ix], actions[ix]), interval[ix])
    np.add.at(durations, (state[ending], actions[ending]), train.dt_next_s.to_numpy()[ending])
    totals = transitions_n.sum(2)
    transition = np.zeros_like(transitions_n)
    mean_dt = np.full((count, 2), step_s, np.float32)
    groups = np.empty(count, object)
    for key, (lo, n) in contexts.items():
        groups[lo:lo + n] = key.rsplit('/', 1)[0]
    groups[terminal] = 'terminal'
    transition[terminal, :, terminal] = 1
    for s in range(terminal):
        for action in (0, 1):
            donors = np.flatnonzero((groups == groups[s]) & (totals[:, action] > 0))
            if not len(donors):
                raise ValueError('No observed transitions for action/context: ' + str(groups[s]))
            distance = ((centers[donors] - centers[s]) ** 2).sum(1)
            donors = donors[np.argsort(distance)[:3]]
            pooled_dt = durations[donors, action].sum() / totals[donors, action].sum()
            prior = config['prior']
            mean_dt[s, action] = (durations[s, action] + prior * pooled_dt) / (totals[s, action] + prior)
            pool = transitions_n[donors, action].sum(0)
            pool /= pool.sum()
            empirical = transitions_n[s, action]
            transition[s, action] = (empirical + prior * pool) / (empirical.sum() + prior)
    if not np.isfinite(transition).all() or not np.allclose(transition.sum(2), 1):
        raise ValueError('Invalid transition matrix')
    return dict(encoder=encoder, centers=centers, contexts=contexts, counts=exposure,
                transitions=transition.astype(np.float32), mean_dt=mean_dt, terminal_index=terminal,
                step_s=step_s, train_rows=len(train))


def pack_frames(frame, people, mdp, profile):
    x = mdp['encoder'].transform(frame)
    states = assign_states(frame, x, mdp['contexts'], mdp['centers'], profile)
    trials = sorted(frame.trial_id.unique())
    data = dict(x=x[:, spatial_mask(mdp['encoder'].names)], state=states,
                action=frame.action.to_numpy(np.float32), weight=frame._weight.to_numpy(np.float64),
                person=frame.participant_id.map({p: i for i, p in enumerate(people)}).to_numpy(np.int16),
                smoke=frame.smoke.map({s: i for i, s in enumerate(SMOKES)}).to_numpy(np.int8),
                trial=frame.trial_id.map({t: i for i, t in enumerate(trials)}).to_numpy(np.int16),
                frame_index=frame.frame_index.to_numpy(np.int32))
    return data, dict(people=people, trials=trials, rows=len(frame))


def cache_for(reader, channel, config, root):
    spec = {k: config[k] for k in ('clusters', 'run_cap_s', 'seed', 'prior')}
    contract = dict(schema=SCHEMA, profile=asdict(reader.profile), channel=channel, state=spec,
                    manifest_sha256=digest(reader.dataset / 'manifest.json'),
                    runs_sha256=digest(reader.dataset / 'action_runs.csv'), geometry=reader.geometry,
                    code_sha256=code_hashes(reader.profile))
    folder = Path(root) / 'Caches' / channel / config_id(contract)
    if folder.exists():
        load_cache(folder, reader.dataset, reader.profile, channel)
        return folder
    train, train_people = load_split(reader, channel, 'train', config['run_cap_s'])
    val, val_people = load_split(reader, channel, 'val', config['run_cap_s'])
    mdp = build_mdp(train, reader.profile, config, reader.manifest['config']['step_s'],
                    reader.manifest['config']['max_gap_s'])
    folder.mkdir(parents=True, exist_ok=False)
    keys = ('centers', 'counts', 'transitions', 'mean_dt', 'terminal_index')
    np.savez_compressed(folder / 'mdp.npz', **{k: mdp[k] for k in keys})
    write_json(folder / 'encoder.json', dict(encoder=mdp['encoder'].as_dict(),
               contexts=mdp['contexts'], step_s=mdp['step_s'], pooled_smoke=reader.profile.pooled_smoke))
    splits = {}
    for split, frame, people in [('train', train, train_people), ('val', val, val_people)]:
        packed, splits[split] = pack_frames(frame, people, mdp, reader.profile)
        for key, array in packed.items():
            np.save(folder / f'{split}_{key}.npy', array, allow_pickle=False)
        if split == 'train':
            counts = np.zeros_like(mdp['counts'])
            np.add.at(counts, (packed['state'], packed['action'].astype(int)), packed['weight'])
            if not np.allclose(counts, mdp['counts'], atol=2e-6):
                raise ValueError('Frame and MDP exposure differ')
    files = [p for p in folder.iterdir() if p.is_file()]
    write_json(folder / 'metadata.json', dict(status='complete', contract=contract, splits=splits,
               spatial_input_names=[n for n, keep in zip(mdp['encoder'].names, spatial_mask(mdp['encoder'].names)) if keep],
               files_sha256={p.name: digest(p) for p in files}))
    return folder


def load_cache(folder, dataset, profile, channel):
    folder, dataset = Path(folder), Path(dataset)
    metadata = read_json(folder / 'metadata.json')
    contract = metadata['contract']
    if (metadata.get('status') != 'complete' or contract.get('schema') != SCHEMA
            or contract.get('profile') != asdict(profile) or contract.get('channel') != channel
            or contract.get('code_sha256') != code_hashes(profile)
            or contract.get('manifest_sha256') != digest(dataset / 'manifest.json')
            or contract.get('runs_sha256') != digest(dataset / 'action_runs.csv')):
        raise ValueError('Independent channel cache has incompatible provenance')
    for name, checksum in metadata['files_sha256'].items():
        if digest(folder / name) != checksum:
            raise ValueError('Cache checksum mismatch: ' + name)
    saved = read_json(folder / 'encoder.json')
    encoder = Encoder(profile.pooled_smoke)
    encoder.median = np.asarray(saved['encoder']['median'])
    encoder.scale = np.asarray(saved['encoder']['scale'])
    if encoder.names != saved['encoder']['names'] or saved['pooled_smoke'] != profile.pooled_smoke:
        raise ValueError('Encoder smoke contract differs')
    with np.load(folder / 'mdp.npz', allow_pickle=False) as arrays:
        mdp = {k: arrays[k] for k in arrays.files}
    mdp.update(encoder=encoder, contexts={k: tuple(v) for k, v in saved['contexts'].items()},
               step_s=saved['step_s'], terminal_index=int(mdp['terminal_index']))
    return mdp, metadata


def load_frames(folder, split):
    return {k: np.load(Path(folder) / f'{split}_{k}.npy', mmap_mode='r') for k in FRAME_KEYS}


def baseline_from_train(data, profile):
    if profile.pooled_smoke:
        return {'global': float(np.mean(data['action']))}
    return {smoke: float(np.mean(data['action'][data['smoke'] == i])) for i, smoke in enumerate(SMOKES)}


def frame_report(data, probability, utility, baseline):
    y = np.asarray(data['action'], float)
    probability = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    utility = np.asarray(utility, float)
    if not np.isfinite(probability).all() or not np.isfinite(utility).all():
        raise ValueError('Nonfinite predictions')
    base = np.full(len(y), baseline['global']) if 'global' in baseline else np.asarray([
        baseline[SMOKES[int(i)]] for i in data['smoke']])
    base = np.clip(base, 1e-7, 1 - 1e-7)
    losses = -(y * np.log(probability) + (1 - y) * np.log1p(-probability))
    base_losses = -(y * np.log(base) + (1 - y) * np.log1p(-base))

    def subset(ix):
        if not ix.any():
            return dict(n=0)
        yy, pp, uu = y[ix], probability[ix], utility[ix]
        weights = np.asarray(data['weight'])[ix]
        has_two = len(np.unique(yy)) == 2
        person = np.asarray(data['person'])[ix]
        brier = float(np.mean((yy - pp) ** 2))
        base_brier = float(np.mean((yy - base[ix]) ** 2))
        return dict(n=int(ix.sum()), people=len(np.unique(person)),
                    nll=float(losses[ix].mean()), weighted_nll=float(np.average(losses[ix], weights=weights)),
                    participant_macro_nll=float(np.mean([losses[ix][person == p].mean() for p in np.unique(person)])),
                    auc=float(roc_auc_score(yy, pp)) if has_two else None,
                    spatial_reward_auc=float(roc_auc_score(yy, uu)) if has_two else None,
                    weighted_spatial_auc=float(roc_auc_score(yy, uu, sample_weight=weights)) if has_two else None,
                    policy_ap=float(average_precision_score(yy, pp)) if has_two else None,
                    spatial_ap=float(average_precision_score(yy, uu)) if has_two else None,
                    observed_fraction=float(yy.mean()), predicted_fraction=float(pp.mean()),
                    brier=brier, baseline_nll=float(base_losses[ix].mean()), baseline_brier=base_brier,
                    nll_gain=float((base_losses[ix] - losses[ix]).mean()),
                    brier_skill=1 - brier / base_brier if base_brier > 0 else None)

    report = subset(np.ones(len(y), bool))
    # Smoke retained ONLY as audit metadata in V6_3, never as a model input.
    report['by_smoke'] = {smoke: subset(data['smoke'] == i) for i, smoke in enumerate(SMOKES)}
    report['baseline'] = baseline
    report['selection_weighting'] = 'person-macro natural-frame NLL; natural-frame spatial AUC'
    report['loss_weighting'] = 'equal person / trial; duration capped within each action run'
    return report


def state_basis_means(features, states, weights, count):
    total = np.zeros((count, features.shape[1]), np.float64)
    mass = np.zeros(count, np.float64)
    np.add.at(total, states, features * weights[:, None])
    np.add.at(mass, states, weights)
    return (total / np.maximum(mass[:, None], 1e-12)).astype(np.float32)


def frame_logits(model, features, states, means, mdp, transition, dt, horizon, gamma):
    reward = model(means)
    _, q = soft_policy(reward, transition, horizon, gamma, mdp['terminal_index'], mean_dt=dt)
    continuation = (q[:, 1] - q[:, 0]) - (reward[:, 1] - reward[:, 0])
    own = features @ model.weight
    return own + continuation[states], own


def device_for(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(0 if device.index is None else device.index)
    return device


def train_candidate(job):
    """Worker: one channel only, only train/val; no test or field selection."""
    started = time.time()
    profile = Profile(**job['profile'])
    config, channel, out = job['config'], job['channel'], Path(job['output'])
    if job['source_sha256'] != code_hashes(profile):
        raise ValueError('Worker implementation changed after launch')
    torch.set_num_threads(job['threads'])
    torch.manual_seed(config['seed'])
    device = device_for(job['device'])
    mdp, metadata = load_cache(job['cache'], job['dataset'], profile, channel)
    frames = {s: load_frames(job['cache'], s) for s in ('train', 'val')}
    basis = SpatialRewardBasis.fit(frames['train']['x'], metadata['spatial_input_names'], config['basis'])
    features = {s: basis.transform(d['x']) for s, d in frames.items()}
    means = state_basis_means(features['train'], frames['train']['state'], frames['train']['weight'], len(mdp['centers']))
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.array(a, copy=True), dtype=dtype, device=device)
    means_t, transition, dt = tensor(means), tensor(mdp['transitions']), tensor(mdp['mean_dt'])
    encoded = {s: (tensor(features[s]), tensor(frames[s]['state'], torch.long)) for s in frames}
    labels, weights = tensor(frames['train']['action']), tensor(frames['train']['weight'])
    weights = weights / weights.sum()
    baseline = baseline_from_train(frames['train'], profile)
    horizon = max(1, int(math.ceil(config['horizon_seconds'] / mdp['step_s'])))
    model = Reward(len(basis.names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5,
                    threshold=1e-5, threshold_mode='abs', min_lr=1e-5)

    def predict(split):
        return frame_logits(model, *encoded[split], means_t, mdp, transition, dt, horizon, config['gamma'])

    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    reason, history, epochs_run = 'epochs', [], 0
    for epoch in range(1, job['epochs'] + 1):
        if time.time() >= job['deadline'] - 3 or (out / 'STOP').exists():
            reason = 'budget'
            break
        optimizer.zero_grad()
        logit, _ = predict('train')
        loss = (torch.nn.functional.softplus(logit) - labels * logit).mul(weights).sum()
        objective = loss + config['l2'] * model.weight.square().sum()
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite objective')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        epochs_run = epoch
        if epoch != 1 and epoch % 5 and epoch != job['epochs']:
            continue
        with torch.no_grad():
            logit, utility = predict('val')
            report = frame_report(frames['val'], torch.sigmoid(logit).cpu().numpy(), utility.cpu().numpy(), baseline)
        score = report['participant_macro_nll']
        scheduler.step(score)
        if score < best - 1e-5:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out / 'reward.pt')
            write_json(out / 'reward_basis.json', basis.spec)
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_nll=float(loss.detach()), score=score,
                            spatial_auc=report['spatial_reward_auc'], lr=optimizer.param_groups[0]['lr']))
        write_json(out / 'progress.json', dict(epoch=epoch, best_epoch=best_epoch, best_score=best))
        if stale >= job['patience'] and epoch >= 50:
            reason = 'early_stop'
            break
    if best_state is None:
        raise RuntimeError('No validated checkpoint before the deadline')
    model.load_state_dict(best_state)
    reports = {}
    with torch.no_grad():
        for split in frames:
            logit, utility = predict(split)
            reports[split] = frame_report(frames[split], torch.sigmoid(logit).cpu().numpy(), utility.cpu().numpy(), baseline)
    np.save(out / 'state_reward_features.npy', means, allow_pickle=False)
    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    write_json(out / 'result.json', dict(status='complete', profile=asdict(profile), channel=channel,
               config=config, score=best, validation=reports['val'], train=reports['train'],
               best_epoch=best_epoch, epochs_run=epochs_run, reason=reason,
               elapsed_seconds=time.time() - started, features=len(basis.names),
               source_sha256=code_hashes(profile), test_used=False))


def rank(records):
    return sorted([r for r in records.values() if r.get('status') == 'complete'
                   and np.isfinite(r['score']) and r['validation']['spatial_reward_auc'] is not None],
                  key=lambda r: (r['score'], r['id']))


def select(records):
    ordered = rank(records)
    if not ordered:
        raise RuntimeError('No valid candidate completed for this channel')
    eligible = [r for r in ordered if r['score'] <= ordered[0]['score'] + .005]
    return sorted(eligible, key=lambda r: (-r['validation']['spatial_reward_auc'], r['score'], r['id']))[0]


def launch_batch(configs, args, profile, channel, reader, out, deadline, records, entrypoint):
    """Bounded candidate workers; cache fitting remains in the coordinator."""
    pending = [c for c in configs if config_id(c) not in records]
    active = {}

    def collect(task):
        task['log'].close()
        result_path = task['folder'] / 'result.json'
        if task['process'].returncode == 0 and result_path.is_file():
            result = read_json(result_path)
            result.update(id=task['id'], output=str(task['folder']), cache=str(task['cache']))
        else:
            result = dict(id=task['id'], channel=channel, config=task['config'], status='failed',
                          output=str(task['folder']), returncode=task['process'].returncode)
        records[task['id']] = result
        write_json(out / 'search_records.json', list(records.values()))
        print('FINISHED', channel, task['id'], result['status'], result.get('score'), flush=True)

    try:
        while pending or active:
            for pid, task in list(active.items()):
                if task['process'].poll() is not None:
                    collect(task)
                    del active[pid]
            if time.time() >= deadline:
                break
            while pending and len(active) < args.jobs and time.time() < deadline:
                config = pending.pop(0)
                cid = config_id(config)
                cache = cache_for(reader, channel, config, reader.root)
                if time.time() >= deadline - 10:
                    break
                folder = out / 'Trials' / cid
                folder.mkdir(parents=True, exist_ok=False)
                job = dict(profile=asdict(profile), channel=channel, config=config,
                           output=str(folder), cache=str(cache), dataset=str(args.dataset),
                           device=args.device, threads=args.threads, epochs=args.epochs,
                           patience=args.patience, deadline=deadline, source_sha256=code_hashes(profile))
                job_path = out / 'Tasks' / (cid + '.json')
                write_json(job_path, job)
                env = os.environ.copy()
                for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                    env[name] = str(args.threads)
                log = (folder / 'worker.log').open('w', encoding='utf-8')
                try:
                    process = subprocess.Popen([sys.executable, '-B', str(entrypoint), '--job', str(job_path)],
                                               stdout=log, stderr=subprocess.STDOUT, env=env)
                except Exception:
                    log.close()
                    raise
                active[process.pid] = dict(process=process, log=log, folder=folder, id=cid,
                                           config=config, cache=cache)
                print('TRAIN', channel, cid, config, flush=True)
            if pending or active:
                time.sleep(.5)
        if active:
            for task in active.values():
                (task['folder'] / 'STOP').touch()
            for pid, task in list(active.items()):
                try:
                    task['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    task['process'].terminate()
                    task['process'].wait(timeout=10)
                collect(task)
                del active[pid]
    finally:
        for task in active.values():
            if task['process'].poll() is None:
                task['process'].terminate()
                task['process'].wait(timeout=10)
            task['log'].close()
        write_json(out / 'search_records.json', list(records.values()))


def candidate_configs(args):
    base = dict(lr=.02, l2=.001, clusters=16, run_cap_s=10., prior=.5,
                seed=args.seed, horizon_seconds=args.horizon_seconds, gamma=.98, basis='linear')
    # Cover all families before spending time on larger state counts.
    return [dict(base, clusters=k, basis=b, l2=l)
            for k, l, b in itertools.product((16, 24), (.001, .0001, .01), args.families.split(','))]


def search_channel(args, profile, channel, reader, root, allowance, entrypoint):
    out = root / 'Search' / channel
    for name in ('Trials', 'Tasks'):
        (out / name).mkdir(parents=True, exist_ok=False)
    initial = candidate_configs(args)
    # Exact fixed-radius geometry can take longer than model fitting. Complete
    # the first train/val cache BEFORE starting this channel's search clock so
    # V6_3 cannot spend its entire allowance without validating any candidate.
    print('PREPARE CHANNEL', channel, profile.variant, flush=True)
    cache_for(reader, channel, initial[0], root)
    start = time.time()
    end = start + allowance
    # Reserve diagnostic budget explicitly, instead of silently using it all on search.
    search_end = start + allowance * (.8 if args.seed_checks else 1.)
    base_end = start + (search_end - start) * .75
    records = {}
    launch_batch(initial, args, profile, channel, reader, out, base_end, records, entrypoint)
    extra = []
    for result in rank(records)[:2]:
        extra.append(dict(result['config'], clusters=48))
        extra.extend(dict(result['config'], horizon_seconds=h) for h in (1., 5.))
        extra.extend(dict(result['config'], lr=lr) for lr in (.005, .05))
    write_json(out / 'extension_plan.json', extra)
    # A candidate appearing from both parents is trained only once.
    extra = list({config_id(c): c for c in extra}.values())
    launch_batch(extra, args, profile, channel, reader, out, search_end, records, entrypoint)
    chosen = select(records)
    write_json(out / 'selection.json', dict(candidate=chosen, rule=SELECTION,
               frozen_at=datetime.now().isoformat(), test_used_for_selection=False))
    pd.DataFrame([dict(id=r['id'], status=r['status'], **r['config'],
                      score=r.get('score'), spatial_auc=r.get('validation', {}).get('spatial_reward_auc'),
                      best_epoch=r.get('best_epoch'), epochs_run=r.get('epochs_run'))
                  for r in records.values()]).to_csv(out / 'search_summary.csv', index=False)
    checks = {}
    plans = [dict(chosen['config'], seed=args.seed + 11 * i) for i in range(1, args.seed_checks + 1)]
    diagnostics = out / 'Seed_Checks'
    for name in ('Trials', 'Tasks'):
        (diagnostics / name).mkdir(parents=True, exist_ok=False)
    write_json(diagnostics / 'plan.json', dict(configs=plans, affects_selection=False, test_used=False))
    launch_batch(plans, args, profile, channel, reader, diagnostics, end, checks, entrypoint)
    completed = sum(r['status'] == 'complete' for r in checks.values())
    write_json(out / 'seed_checks.json', dict(requested=args.seed_checks, completed=completed,
               status='complete' if completed == args.seed_checks else 'incomplete_budget_or_failure',
               records=list(checks.values()), affects_selection=False, test_used=False))
    if completed != args.seed_checks:
        print(f'SEED_CHECKS_INCOMPLETE {channel}: {completed}/{args.seed_checks}', flush=True)
    return chosen


def export_model(args, profile, channel, selected, reader, folder):
    source, cache = Path(selected['output']), Path(selected['cache'])
    folder.mkdir(parents=True, exist_ok=False)
    mdp, metadata = load_cache(cache, args.dataset, profile, channel)
    for name in ('reward.pt', 'reward_basis.json', 'state_reward_features.npy', 'training_history.csv'):
        shutil.copy2(source / name, folder / name)
    for name in ('encoder.json', 'mdp.npz'):
        shutil.copy2(cache / name, folder / name)
    basis = SpatialRewardBasis(read_json(folder / 'reward_basis.json'))
    config = selected['config']
    device = device_for(args.device)
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.array(a, copy=True), dtype=dtype, device=device)
    model = Reward(len(basis.names)).to(device)
    model.load_state_dict(torch.load(folder / 'reward.pt', map_location=device, weights_only=True))
    means = tensor(np.load(folder / 'state_reward_features.npy', allow_pickle=False))
    transition, dt = tensor(mdp['transitions']), tensor(mdp['mean_dt'])
    horizon = max(1, int(math.ceil(config['horizon_seconds'] / mdp['step_s'])))
    frames = {s: load_frames(cache, s) for s in ('train', 'val')}
    # Called only after EVERY channel's model selection was frozen.
    test, people = load_split(reader, channel, 'test', config['run_cap_s'])
    frames['test'], test_identity = pack_frames(test, people, mdp, profile)
    identity = dict(metadata['splits'], test=test_identity)
    baseline = baseline_from_train(frames['train'], profile)
    evaluations, predictions = {}, {}
    with torch.no_grad():
        for split, data in frames.items():
            z, u = frame_logits(model, tensor(basis.transform(data['x'])), tensor(data['state'], torch.long),
                                means, mdp, transition, dt, horizon, config['gamma'])
            p, utility = torch.sigmoid(z).cpu().numpy(), u.cpu().numpy()
            evaluations[split] = frame_report(data, p, utility, baseline)
            if split in ('val', 'test'):
                for key in FRAME_KEYS:
                    if key not in ('x', 'state'):
                        predictions[split + '_' + key] = data[key]
                predictions[split + '_probability'] = p
                predictions[split + '_utility'] = utility
    np.savez_compressed(folder / 'evaluation_predictions.npz', **predictions)
    write_json(folder / 'evaluation_identity.json', dict(channel=channel, smokes=SMOKES, splits=identity))
    write_json(folder / 'evaluation.json', dict(**evaluations, trained=True, selection_rule=SELECTION,
               test_used_for_selection=False, spatial_score='immediate reward contrast',
               policy_score='observed-state soft-Bellman probability'))
    pd.DataFrame(dict(feature=basis.names, contrast_weight=model.weight.detach().cpu().numpy())).to_csv(
        folder / 'reward_rules.csv', index=False)
    write_json(folder / 'config.json', dict(status='complete', schema=SCHEMA, profile=asdict(profile),
               channel=channel, direct_condition=channel, reward_scope='spatial_only',
               action_definition=reader.manifest['action_definition'], config=config,
               geometry=reader.geometry, source_manifest_sha256=digest(args.dataset / 'manifest.json'),
               source_sha256=code_hashes(profile), baseline=baseline,
               horizon=horizon, nominal_step_s=mdp['step_s'], selection_rule=SELECTION,
               selected_candidate=selected['id'], test_used_for_selection=False,
               files_sha256={p.name: digest(p) for p in folder.iterdir() if p.is_file()}))
    return evaluations


class SpatialRule:
    """Load this version's independent bank; point queries need no trajectories."""
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.config = read_json(self.folder / 'config.json')
        if self.config.get('status') != 'complete' or self.config.get('schema') != SCHEMA:
            raise ValueError('A completed V6_2/V6_3 independent model is required')
        self.profile = Profile(**self.config['profile'])
        self.models = None
        if 'model_bank' in self.config:
            if list(self.config['model_bank']) != CHANNELS:
                raise ValueError('Incomplete or reordered independent model bank')
            self.models = {}
            for channel, relative in self.config['model_bank'].items():
                child = (self.folder / relative).resolve()
                if not child.is_relative_to(self.folder):
                    raise ValueError('Model path outside bank')
                rule = SpatialRule(child)
                if (rule.profile != self.profile or rule.config['channel'] != channel
                        or rule.config['geometry'] != self.config['geometry']):
                    raise ValueError('Model bank mixes channel/profile contracts')
                self.models[channel] = rule
            return
        # Point queries depend only on these fitted artifacts, never on MDP,
        # predictions, observations or original trajectory files.
        for name in ('reward.pt', 'reward_basis.json', 'encoder.json'):
            if digest(self.folder / name) != self.config['files_sha256'][name]:
                raise ValueError('Model checksum differs: ' + name)
        saved = read_json(self.folder / 'encoder.json')['encoder']
        self.encoder = Encoder(self.profile.pooled_smoke)
        self.encoder.median, self.encoder.scale = np.asarray(saved['median']), np.asarray(saved['scale'])
        if self.encoder.names != saved['names']:
            raise ValueError('Query encoder schema differs')
        self.mask = spatial_mask(self.encoder.names)
        self.basis = SpatialRewardBasis(read_json(self.folder / 'reward_basis.json'))
        if self.basis.input_names != [n for n, keep in zip(self.encoder.names, self.mask) if keep]:
            raise ValueError('Nonspatial or incompatible reward basis')
        reward = Reward(len(self.basis.names))
        reward.load_state_dict(torch.load(self.folder / 'reward.pt', map_location='cpu', weights_only=True))
        self.weight = reward.weight.detach().numpy()

    def query_features(self, frame, channel):
        if self.models is not None:
            if channel not in self.models:
                raise ValueError('Unknown channel')
            return self.models[channel].query_features(frame, channel)
        if channel != self.config['channel']:
            raise ValueError('Wrong independent channel')
        required = SPATIAL + ['floor', 'stage'] + ([] if self.profile.pooled_smoke else ['smoke'])
        if not set(required) <= set(frame) or not np.isfinite(frame[SPATIAL].to_numpy(float)).all():
            raise ValueError('Missing/invalid spatial inputs')
        if not frame.floor.isin(['Hall', 'Platform']).all() or not frame.stage.isin(STAGES).all():
            raise ValueError('Unknown floor/stage')
        state = frame.copy()
        if self.profile.pooled_smoke:
            if ('visibility_m' not in frame
                    or not np.allclose(frame.visibility_m.to_numpy(float), self.profile.visibility_m)):
                raise ValueError('V6_3 features must explicitly declare the fixed 12.5 m geometry radius')
            state['smoke'] = 'PooledSmoke'  # discarded by Encoder, not a low-smoke substitution
        elif not frame.smoke.isin(SMOKES).all():
            raise ValueError('Unknown smoke')
        for name in NUMERIC[5:]:
            state[name] = 0.
        state['previous_action'] = -1
        x = self.encoder.transform(state)[:, self.mask]
        return self.basis.transform(x) @ self.weight

    def query_position(self, x, y, z, channel, smoke=None, floor=None):
        from spatial_isovist_core import PlanarIsovist
        from Analyze_NextDecisionDistance import NextDecisionDistance
        if self.profile.pooled_smoke and smoke is not None:
            raise ValueError('V6_3 has a single field; do not provide smoke')
        if not self.profile.pooled_smoke and smoke not in SMOKES:
            raise ValueError('V6_2 point queries require --smoke')
        geo = self.config['geometry']
        if digest(geo['path']) != geo['sha256']:
            raise ValueError('Query geometry changed')
        spatial = PlanarIsovist(geo['path'])
        decision = NextDecisionDistance(geo['path'])
        radius = self.profile.visibility_m if self.profile.pooled_smoke else VISIBILITY[smoke]
        s, d = spatial.query(x, y, z, radius, floor=floor), decision.query(x, y, z, floor=floor)
        if s.status != 'ok' or d.status != 'ok':
            return dict(status=s.status if s.status != 'ok' else d.status)
        if d.stage == 'arrived_exit':
            return dict(status='task_complete')
        row = {n: getattr(s, n) for n in SPATIAL[:4]}
        row.update(next_decision_distance=d.next_decision_distance, floor=d.floor, stage=d.stage,
                   smoke='PooledSmoke' if self.profile.pooled_smoke else smoke, visibility_m=radius)
        utility = float(self.query_features(pd.DataFrame([row]), channel)[0])
        return dict(status='ok', channel=channel, relative_information_utility=utility,
                    profile=asdict(self.profile), **row)


def build_grid(spatial, decision, profile, cell):
    import shapely
    rows = []
    conditions = {'PooledSmoke': profile.visibility_m} if profile.pooled_smoke else VISIBILITY
    for floor, domain in spatial.domains.items():
        xmin, zmin, xmax, zmax = domain.bounds
        xe = np.arange(math.floor(xmin / cell), math.ceil(xmax / cell) + 1) * cell
        ze = np.arange(math.floor(zmin / cell), math.ceil(zmax / cell) + 1) * cell
        xx, zz = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ze[:-1] + ze[1:]) / 2)
        valid = shapely.intersects_xy(domain, xx, zz) & ~shapely.intersects_xy(spatial.exclusions[floor], xx, zz)
        for iz, ix in tqdm(np.argwhere(valid), desc='Global grid ' + floor):
            x, z = float(xx[iz, ix]), float(zz[iz, ix])
            d = decision.query(x, None, z, floor=floor)
            if d.status != 'ok' or d.stage == 'arrived_exit':
                continue
            for smoke, radius in conditions.items():
                s = spatial.query(x, None, z, radius, floor=floor)
                if s.status != 'ok':
                    raise ValueError('Invalid isovist on a valid geometry cell')
                row = {n: getattr(s, n) for n in SPATIAL[:4]}
                row.update(x=x, z=z, floor=floor, stage=d.stage, smoke=smoke,
                           visibility_m=radius, next_decision_distance=d.next_decision_distance)
                rows.append(row)
    if not rows:
        raise ValueError('Empty global grid')
    return pd.DataFrame(rows)


def draw_field(frame, spatial, channel, profile, path, dpi, cell, sigma, upsample):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from spatial_isovist_core import _outline
    from gaussian_field_v3 import smooth_display, continuous_display
    conditions = ['PooledSmoke'] if profile.pooled_smoke else list(SMOKES)
    low, high = float(frame.relative_information_utility.min()), float(frame.relative_information_utility.max())
    if high <= low:
        low, high = low - .5, high + .5
    norm = Normalize(low, high)
    fig, axes = plt.subplots(len(conditions), 2, squeeze=False, figsize=(12, 8 * len(conditions)),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(left=.08, right=.82, bottom=.08, top=.90, hspace=.25, wspace=.3)
    try:
        for row, smoke in enumerate(conditions):
            for col, floor in enumerate(['Hall', 'Platform']):
                ax, domain = axes[row, col], spatial.domains[floor]
                xmin, zmin, xmax, zmax = domain.bounds
                x0, z0 = np.floor(np.array([xmin, zmin]) / cell).astype(int)
                x1, z1 = np.ceil(np.array([xmax, zmax]) / cell).astype(int)
                xe, ze = np.arange(x0, x1 + 1) * cell, np.arange(z0, z1 + 1) * cell
                raw = np.full((z1 - z0, x1 - x0), np.nan)
                part = frame.loc[frame.smoke.eq(smoke) & frame.floor.eq(floor)]
                ix = np.rint(part.x.to_numpy() / cell - .5).astype(int) - x0
                iz = np.rint(part.z.to_numpy() / cell - .5).astype(int) - z0
                raw[iz, ix] = part.relative_information_utility
                walkable = domain.difference(spatial.exclusions[floor])
                display = smooth_display(raw, (xe[:-1] + xe[1:]) / 2, (ze[:-1] + ze[1:]) / 2, walkable, sigma, cell)
                display, fx, fz = continuous_display(display, xe, ze, walkable, upsample)
                ax.pcolormesh(-fx[::-1], -fz[::-1], np.ma.masked_invalid(display[::-1, ::-1]),
                              cmap='plasma', norm=norm, shading='flat', rasterized=True)
                _outline(ax, domain, color='red', linewidth=.65)
                _outline(ax, spatial.exclusions[floor], color='gray', linewidth=.65, linestyle='--')
                ax.set_aspect('equal')
                label = 'Pooled high + low smoke' if profile.pooled_smoke else smoke
                ax.set_title(f'{label} | {floor}')
                ax.set_xlabel('Display X = -Unity X (m)')
                ax.set_ylabel('Display Z = -Unity Z (m)')
        cax = fig.add_axes([.87, .16, .025, .65])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), cax=cax,
                     label='Relative information utility (immediate reward contrast)')
        fig.suptitle(f'{channel} | {profile.variant} independent spatial IRL', fontsize=14)
        fig.text(.08, .025, f'{cell:g} m raw grid; display smoothing only; color scale fitted within this channel.', fontsize=9)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    finally:
        plt.close(fig)
    return [low, high]


def export_fields(folder, args, expected_profile):
    rule = SpatialRule(folder)
    if rule.profile != expected_profile or rule.models is None:
        raise ValueError('Plot entrypoint and independent model bank differ')
    from spatial_isovist_core import PlanarIsovist
    from Analyze_NextDecisionDistance import NextDecisionDistance
    geo = rule.config['geometry']
    if digest(geo['path']) != geo['sha256']:
        raise ValueError('Geometry changed after fitting')
    out = Path(folder) / ('Results_Global_Spatial_Field_' + rule.profile.variant)
    out.mkdir(parents=True, exist_ok=False)
    spatial, decision = PlanarIsovist(geo['path']), NextDecisionDistance(geo['path'])
    grid = build_grid(spatial, decision, rule.profile, args.cell_m)
    files, scales, fields = [], {}, []
    for channel in CHANNELS:
        part = grid.copy()
        part['guidance'] = channel
        part['relative_information_utility'] = rule.query_features(grid, channel)
        if not np.isfinite(part.relative_information_utility).all():
            raise ValueError('Nonfinite field')
        # V6_3 has one row per channel/floor/position, not duplicated smoke maps.
        unique = ['floor', 'x', 'z'] + ([] if rule.profile.pooled_smoke else ['smoke'])
        if part.duplicated(unique).any():
            raise ValueError('Duplicate global field position')
        csv_name = channel + '_global_field.csv.gz'
        part.to_csv(out / csv_name, index=False, compression='gzip')
        files.append(csv_name)
        fields.append(part)
        if not args.skip_plots:
            image_name = channel + '_spatial_demand.png'
            scales[channel] = draw_field(part, spatial, channel, rule.profile, out / image_name,
                                         args.dpi, args.cell_m, args.sigma_m, args.upsample)
            files.append(image_name)
    pd.concat(fields, ignore_index=True).to_csv(out / 'global_spatial_rule_field.csv.gz', index=False, compression='gzip')
    files.append('global_spatial_rule_field.csv.gz')
    manifest = dict(status='complete', profile=asdict(rule.profile), channels=CHANNELS,
                    fields_per_channel=1 if rule.profile.pooled_smoke else 2,
                    total_fields=4 if rule.profile.pooled_smoke else 8, cell_m=args.cell_m,
                    geometry=geo, color_scale_by_channel=scales, source_sha256=code_hashes(rule.profile),
                    training_smoothed=False, raw_csv_smoothed=False, query_smoothed=False,
                    trajectory_input=False, display_only=dict(sigma_m=args.sigma_m, upsample=args.upsample),
                    raw_field='immediate relative action utility; not a probability or causal effect',
                    model_config_sha256=digest(Path(folder) / 'config.json'),
                    files_sha256={name: digest(out / name) for name in files})
    write_json(out / 'manifest.json', manifest)
    return out


def parser_for(variant):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget-hours', type=float, default=4.,
                        help='Search/seed budget split equally across four channels; initial preparation and export are extra')
    parser.add_argument('--jobs', type=int, default=2, help='Concurrent candidate workers within one channel')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=16, help='Validation checks without NLL improvement (every 5 epochs)')
    parser.add_argument('--horizon-seconds', type=float, default=2.)
    parser.add_argument('--families', default=','.join(FAMILIES))
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--seed-checks', type=int, default=2, help='Validation-only diagnostic seeds; never select with these or test')
    parser.add_argument('--output-root', type=Path, default=HERE / 'Results_IRL')
    parser.add_argument('--skip-plots', action='store_true', help='Skip PNG rendering; still export all raw global fields')
    parser.add_argument('--cell-m', type=float, default=.4)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--sigma-m', type=float, default=.6)
    parser.add_argument('--upsample', type=int, default=4)
    parser.add_argument('--plot-run', type=Path, help='Export fields for a completed bank; refuses existing field directory')
    parser.add_argument('--run', type=Path, help='Completed model bank for point query')
    parser.add_argument('--query', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--guidance', choices=CHANNELS)
    if variant == 'V6_2':
        parser.add_argument('--smoke', choices=SMOKES)
    parser.add_argument('--floor', choices=('Hall', 'Platform'))
    parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv=None, *, variant='V6_2', entrypoint=None):
    profile = Profile(variant, variant == 'V6_3', 12.5 if variant == 'V6_3' else None)
    entrypoint = Path(entrypoint or __file__).resolve()
    parser = parser_for(variant)
    args = parser.parse_args(argv)
    if args.job:
        job = read_json(args.job)
        if job['profile'] != asdict(profile):
            parser.error('Worker entrypoint/profile mismatch')
        try:
            train_candidate(job)
        except Exception as exc:
            write_json(Path(job['output']) / 'error.json', dict(error=str(exc), traceback=traceback.format_exc()))
            raise
        return
    if sum((args.train, args.plot_run is not None, args.query is not None)) > 1:
        parser.error('Choose one of --train, --plot-run or --query')
    if (not np.isfinite([args.cell_m, args.sigma_m]).all() or args.cell_m <= 0
            or args.sigma_m <= 0 or args.dpi < 50 or not 1 <= args.upsample <= 16):
        parser.error('Invalid field display settings')
    if args.plot_run:
        print('FIELD_OUTPUT=' + str(export_fields(args.plot_run, args, profile)))
        return
    if args.query is not None:
        if not args.run or not args.guidance:
            parser.error('--query needs --run and --guidance')
        rule = SpatialRule(args.run)
        if rule.profile != profile:
            parser.error('Use the entrypoint matching the saved model version')
        print(json.dumps(rule.query_position(*args.query, channel=args.guidance,
                         smoke=getattr(args, 'smoke', None), floor=args.floor), indent=2))
        return
    if args.run or args.guidance or args.floor or getattr(args, 'smoke', None):
        parser.error('Point-query options require --query')
    if not args.dataset:
        parser.error('--dataset is required for a plan or training')
    if (not np.isfinite([args.budget_hours, args.horizon_seconds]).all()
            or args.budget_hours <= 0 or args.horizon_seconds <= 0 or args.jobs < 1
            or args.threads < 1 or args.epochs < 5 or args.patience < 1 or args.seed_checks < 0
            or not 0 <= args.seed < 2**32 - 11 * args.seed_checks):
        parser.error('Invalid training budget/stopping/seed settings')
    families = args.families.split(',')
    if not families or len(set(families)) != len(families) or any(b not in FAMILIES for b in families):
        parser.error('Invalid or repeated basis families')
    args.dataset = args.dataset.resolve()
    manifest, audit = audit_scope(args.dataset)
    plan = dict(trained=False, profile=asdict(profile), independent_channels=CHANNELS,
                direct_conditions={c: [c] for c in CHANNELS}, excluded_conditions=['Full'],
                source_scope=audit, selection_rule=SELECTION, candidates_per_channel=candidate_configs(args),
                budget_hours_total=args.budget_hours, seed_checks_per_channel=args.seed_checks,
                output_fields=4 if profile.pooled_smoke else 8,
                png_count=0 if args.skip_plots else 4, test_used_for_selection=False)
    if not args.train:
        print(json.dumps(plan, indent=2))
        return
    if args.budget_hours * 3600 / len(CHANNELS) <= 180:
        parser.error('Allow more than three minutes per channel')
    device_for(args.device)
    torch.set_num_threads(args.threads)
    geometry = geometry_contract(manifest)
    out = args.output_root.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f_' + variant)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'plan.json', plan)
    write_json(out / 'dataset_audit.json', audit)
    run_config = dict(status='running', schema=SCHEMA, profile=asdict(profile), geometry=geometry,
                      source_sha256=code_hashes(profile), dataset=str(args.dataset),
                      arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      test_used_for_selection=False)
    write_json(out / 'config.json', run_config)
    started = time.time()
    reader = TrialReader(args.dataset, manifest, profile, out, geometry)
    selected = {}
    try:
        # Fixed equal allowances avoid letting an easy channel consume another's budget.
        for channel in CHANNELS:
            selected[channel] = search_channel(args, profile, channel, reader, out,
                                                args.budget_hours * 3600 / len(CHANNELS), entrypoint)
        write_json(out / 'selection.json', dict(by_channel=selected, rule=SELECTION,
                   frozen_at=datetime.now().isoformat(), test_used_for_selection=False))
        seed_status = {c: read_json(out / 'Search' / c / 'seed_checks.json') for c in CHANNELS}
        bank = out / 'Best_Model'
        evaluations = {}
        for channel in CHANNELS:
            evaluations[channel] = export_model(args, profile, channel, selected[channel], reader, bank / 'Models' / channel)
        write_json(bank / 'config.json', dict(status='complete', schema=SCHEMA, profile=asdict(profile),
                   geometry=geometry, model_bank={c: 'Models/' + c for c in CHANNELS},
                   direct_conditions={c: [c] for c in CHANNELS}, excluded_conditions=['Full'],
                   source_sha256=code_hashes(profile), source_manifest_sha256=digest(args.dataset / 'manifest.json'),
                   independent_preprocessing=True, independent_optimization=True,
                   selection_rule=SELECTION, test_used_for_selection=False))
        write_json(bank / 'evaluation.json', dict(by_channel=evaluations, test_used_for_selection=False))
        fields = export_fields(bank, args, profile)
        run_config.update(status='complete', best_model=str(bank), fields=str(fields))
        write_json(out / 'config.json', run_config)
        write_json(out / 'completion.json', dict(status='complete', best_model=str(bank), fields=str(fields),
                   total_seconds=time.time() - started, channels=CHANNELS,
                   total_fields=4 if profile.pooled_smoke else 8, test_used_for_selection=False,
                   seed_checks_by_channel={c: {k: seed_status[c][k] for k in ('requested', 'completed', 'status')}
                                           for c in CHANNELS}))
    except Exception as exc:
        run_config.update(status='failed', error=str(exc))
        write_json(out / 'config.json', run_config)
        write_json(out / 'error.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    print('BEST_MODEL=' + str(bank))
    print('FIELD_OUTPUT=' + str(fields))
    print('OUTPUT=' + str(out))


if __name__ == '__main__':
    main()
