"""Build timestamp-aligned A5 IRL inputs; never trains a model.

common.load_trials() defines the source cohort; IRL_Cohort applies task exclusions.
Raw gaze is queried through the
public frame API. Invalid data remain masked; contiguous sequences never cross
gaps. All epoch integers explicitly use ns (including pandas >=3).
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import load_trials
from spatial_isovist_core import PlanarIsovist, DEFAULT_GEOMETRY_PATH
from Analyze_NextDecisionDistance import NextDecisionDistance
from Query_GuidanceGaze import GuidanceGazeQuery, CHANNELS
from IRL_Cohort import SCHEMA, COHORT_POLICY, select_irl_trials

REFERENCE = ROOT / 'Results_Spatial_Metric_Surfaces/_timestamp_input/extract_timestamp_metrics.py'
spec = importlib.util.spec_from_file_location('irl_timestamp_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

SPATIAL = ['concavity', 'distance_variation', 'elongation', 'horizontal_size', 'next_decision_distance']
STATE_NUMERIC = SPATIAL + ['speed_prev', 'speed_mean_past', 'acceleration_prev',
    'head_rate_prev', 'scan_rate_prev', 'gaze_depth_prev', 'elapsed_s']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ns_array(values):
    return np.asarray(values, dtype='datetime64[ns]').astype(np.int64)


def previous_indices(source_ns, query_ns, max_age_s=.05):
    indices = np.searchsorted(source_ns, query_ns, side='right')-1
    safe = np.maximum(indices, 0)
    age = (query_ns-source_ns[safe])/1e9
    valid = (indices >= 0) & (age >= 0) & (age <= max_age_s)
    return safe, valid, age


def sample_values(source_ns, values, query_ns, max_age_s=.05):
    idx, valid, _ = previous_indices(source_ns, query_ns, max_age_s)
    result = np.array(values[idx], dtype=float, copy=True)
    result[~valid] = np.nan
    return result


def eye_metrics(row, subject, pns):
    columns = ['TimeStamp', 'HeadRotX', 'HeadRotY', 'HeadRotZ', 'GazeDirX',
               'GazeDirY', 'GazeDirZ', 'Hit1_X', 'Hit1_Y', 'Hit1_Z']
    eye, audit = reference.read_raw(Path(row.EyeGazeFile), columns)
    ens = ns_array(eye.ParsedTime)
    if not len(ens):
        return ens, np.empty((0, 3)), audit
    positions = sample_values(pns, subject[['PosX', 'PosY', 'PosZ']].to_numpy(float), ens)
    head = eye[['HeadRotX', 'HeadRotY', 'HeadRotZ']].to_numpy(float)
    gaze, gaze_valid = reference.unit_vectors(eye[['GazeDirX', 'GazeDirY', 'GazeDirZ']].to_numpy(float))
    head_valid = np.isfinite(head).all(axis=1)
    rotation = Rotation.from_euler('zxy', np.nan_to_num(head)[:, [2, 0, 1]], degrees=True)
    forward = rotation.apply(np.tile([0., 0., 1.], (len(head), 1)))
    local = rotation.inv().apply(np.nan_to_num(gaze))
    dt = np.r_[np.nan, np.diff(ens)/1e9]
    pos_valid = np.isfinite(positions).all(axis=1) & (positions != 0).any(axis=1)
    step = np.r_[np.nan, np.linalg.norm(np.diff(positions, axis=0), axis=1)]
    motion = pos_valid & np.r_[False, pos_valid[:-1]] & (step <= 3)
    # Backward interval only: no centered smoothing or future observations.
    head_angle = reference.interval_angles(forward, head_valid, dt)
    scan_angle = reference.interval_angles(local, gaze_valid & head_valid, dt)
    head_rate = np.divide(head_angle, dt, out=np.full(len(dt), np.nan), where=motion & (dt > 0))
    scan_rate = np.divide(scan_angle, dt, out=np.full(len(dt), np.nan), where=motion & (dt > 0))
    hit = eye[['Hit1_X', 'Hit1_Y', 'Hit1_Z']].to_numpy(float)
    valid = gaze_valid & pos_valid & np.isfinite(hit).all(axis=1) & (hit != 0).any(axis=1)
    depth = np.linalg.norm(hit-positions, axis=1)
    depth[~valid] = np.nan
    return ens, np.column_stack([head_rate, scan_rate, depth]), audit


def segments_and_runs(valid, actions, max_run_steps):
    """Run-cap weights limit dwell dominance, without flipping/dropping labels."""
    n = len(valid)
    segment = np.full(n, -1, dtype=int)
    run = np.full(n, -1, dtype=int)
    weight = np.zeros(n)
    records = []
    segment_id = -1
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        if i == 0 or not valid[i-1]:
            segment_id += 1
        j = i+1
        while j < n and valid[j] and actions[j] == actions[i]:
            j += 1
        segment[i:j] = segment_id
        run[i:j] = len(records)
        weight[i:j] = min(1., max_run_steps/(j-i))
        records.append(dict(run_id=len(records), segment_id=segment_id,
                            start_row=i, stop_row=j, action=int(actions[i]), steps=j-i))
        i = j
    return segment, run, weight, records


def slowdown_states(speed, baseline, fraction, duration_steps):
    """Freeze pre-slowdown baseline until recovery, so a long stop stays slow.

    Confirmation is causal; onset rows are not backfilled. Missing speed resets
    the episode. These are transparent diagnostic defaults, not IRL targets.
    """
    result = np.zeros(len(speed), dtype=np.int8)
    anchor = None
    steps = 0
    for i, (v, b) in enumerate(zip(speed, baseline)):
        if not np.isfinite(v):
            anchor, steps = None, 0
            continue
        if anchor is None:
            if np.isfinite(b) and b >= .2 and v <= (1-fraction)*b:
                anchor, steps = b, 1
        elif v >= (1-fraction/2)*anchor:
            anchor, steps = None, 0
        else:
            steps += 1
        result[i] = anchor is not None and steps >= duration_steps
    return result


def split_roster(trials, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for group, rows in trials[['ParticipantID', 'Group']].drop_duplicates().groupby('Group'):
        people = sorted(rows.ParticipantID)
        rng.shuffle(people)
        # 15/group: 9 train, 3 val, 3 test. All six trials/person stay together.
        for i, person in enumerate(people):
            result[person] = 'train' if i < 9 else ('val' if i < 12 else 'test')
    return result


def build_trial(row, trial_id, split, args, spatial, decision, gaze):
    subject, sa = reference.read_raw(Path(row.SubjectFile),
        ['TimeStamp', 'PosX', 'PosY', 'PosZ', 'VeloX', 'VeloY', 'VeloZ'])
    if subject.empty:
        raise ValueError('No valid Subject timestamps: '+row.SubjectFile)
    pns = ns_array(subject.ParsedTime)
    ens, eyes, ea = eye_metrics(row, subject, pns)
    step_ns = int(round(args.step_s*1e9))
    grid = np.arange(pns[0], pns[-1]+1, step_ns, dtype=np.int64)
    idx, sync_valid, age = previous_indices(pns, grid)
    xyz = subject[['PosX', 'PosY', 'PosZ']].to_numpy(float)[idx]
    speed_raw = np.hypot(subject.VeloX.to_numpy(float), subject.VeloZ.to_numpy(float))
    speed = sample_values(pns, speed_raw, grid)
    eye_now = sample_values(ens, eyes, grid) if len(ens) else np.full((len(grid), 3), np.nan)
    df = pd.DataFrame(dict(timestamp_ns=grid, subject_timestamp_ns=pns[idx],
        sample_age_ms=age*1000, elapsed_s=(grid-grid[0])/1e9,
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], speed=speed,
        head_rate=eye_now[:, 0], scan_rate=eye_now[:, 1], gaze_depth=eye_now[:, 2]))
    df.loc[~sync_valid, ['x', 'y', 'z']] = np.nan
    # State dynamics strictly precede the current action; current speed is an outcome.
    for name in ['speed', 'head_rate', 'scan_rate', 'gaze_depth']:
        df[name+'_prev'] = df[name].shift(1)
    past_steps = max(2, round(args.history_s/args.step_s))
    df['speed_mean_past'] = df.speed.shift(1).rolling(past_steps, min_periods=2).mean()
    df['acceleration_prev'] = df.speed.diff().shift(1)/args.step_s
    ratio = (df.speed_mean_past-df.speed)/df.speed_mean_past.clip(lower=.2)
    slow_steps = max(1, round(args.slow_duration_s/args.step_s))
    df['slow_state'] = slowdown_states(df.speed.to_numpy(), df.speed_mean_past.to_numpy(), args.slow_fraction, slow_steps)
    df['slow_ratio'] = ratio
    df['stationary'] = (df.speed <= .1).astype('int8')
    df['turning'] = (df.head_rate >= 30).astype('int8')
    # Unknown values retain -1 rather than becoming actions 0.
    result = gaze.query_frame_many(row.ParticipantID, [pd.Timestamp(int(t)) for t in grid], scenario=row.Scenario)
    df['action'] = [int(r.is_gazing) if r.status == 'ok' else -1 for r in result]
    df['gaze_status'] = [r.status for r in result]
    df['gaze_sample_age_ms'] = [r.sample_age_ms for r in result]
    df['gaze_types'] = ['|'.join(r.guidance_types) for r in result]
    spatial_rows = []
    for x, y, z in df[['x', 'y', 'z']].itertuples(index=False, name=None):
        r = spatial.query(x, y, z, 7.5 if row.Smoke == 'HighSmoke' else 12.5)
        d = decision.query(x, y, z)
        spatial_rows.append(dict(concavity=r.concavity, distance_variation=r.distance_variation,
            elongation=r.elongation, horizontal_size=r.horizontal_size,
            next_decision_distance=d.next_decision_distance, floor=d.floor,
            zone_id=d.zone_id, stage=d.stage, spatial_status=r.status,
            decision_status=d.status))
    df = pd.concat([df, pd.DataFrame(spatial_rows)], axis=1)
    usable = (df.action >= 0) & (df.spatial_status == 'ok') & (df.decision_status == 'ok') & df.speed.notna() & df.speed_prev.notna()
    # Recorded post-arrival stopping is task completion, not an information trap.
    df['after_exit'] = df.stage.eq('arrived_exit').cummax().astype('int8')
    usable &= df.after_exit.eq(0)
    # Drop next-step linkage across teleports, stale timestamps or geometric holes.
    moved = np.r_[False, np.linalg.norm(np.diff(df[['x','y','z']].to_numpy(), axis=0), axis=1) > 3]
    usable &= ~moved & sync_valid
    a = df.action.to_numpy(int)
    seg, runs, weights, records = segments_and_runs(usable.to_numpy(), a, args.run_cap_s/args.step_s)
    df['segment_id'], df['run_id'], df['run_weight'] = seg, runs, weights
    df['valid'] = usable.astype('int8')
    df['previous_action'] = df.action.shift(1).fillna(-1).astype(int)
    contiguous_previous = (seg >= 0) & (seg == np.r_[-2, seg[:-1]])
    df.loc[~contiguous_previous, 'previous_action'] = -1
    df['transition_valid'] = ((seg >= 0) & (seg == np.r_[seg[1:], -2])).astype('int8')
    df['trap_observed'] = np.where(usable, ((df.slow_state == 1) & (df.action == 1)).astype(int), -1)
    for name, value in dict(trial_id=trial_id, participant_id=row.ParticipantID,
            group=row.Group, guidance=row.Guidance, smoke=row.Smoke, scenario=row.Scenario, split=split).items():
        df[name] = value
    for record in records:
        record.update(trial_id=trial_id, participant_id=row.ParticipantID,
                      scenario=row.Scenario, split=split, duration_s=record['steps']*args.step_s)
    valid_actions = df.loc[usable, 'action']
    audit = dict(trial_id=trial_id, participant_id=row.ParticipantID, scenario=row.Scenario,
        split=split, sampled_rows=len(df), valid_rows=int(usable.sum()),
        action_one=int((valid_actions == 1).sum()), action_zero=int((valid_actions == 0).sum()),
        action_fraction=float(valid_actions.mean()) if len(valid_actions) else None,
        after_exit_rows=int(df.after_exit.sum()),
        mixed_actions=bool(valid_actions.nunique() == 2), run_count=len(records),
        longest_run_s=max((r['duration_s'] for r in records), default=0),
        gaze_status_counts=dict(Counter(df.gaze_status)), spatial_status_counts=dict(Counter(df.spatial_status)),
        subject_audit=sa, eye_audit=ea)
    return df, records, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--step-s', type=float, default=.2)
    parser.add_argument('--history-s', type=float, default=1.)
    parser.add_argument('--run-cap-s', type=float, default=2.)
    parser.add_argument('--slow-fraction', type=float, default=.25)
    parser.add_argument('--slow-duration-s', type=float, default=.4)
    parser.add_argument('--seed', type=int, default=20260926)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.step_s, args.history_s, args.run_cap_s, args.slow_duration_s)) or not 0 < args.slow_fraction < 1:
        parser.error('Durations must be positive; slow-fraction must be in (0,1)')
    canonical = load_trials().sort_values(['Scenario','ParticipantID','FileName']).reset_index(drop=True)
    split = split_roster(canonical, args.seed)
    trials, excluded = select_irl_trials(canonical)
    output = args.output or HERE/'Results_IRL_Dataset'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'trials').mkdir()
    source_paths = [Path(__file__), HERE/'IRL_Cohort.py', ROOT/'common.py', HERE/'Query_GuidanceGaze.py',
        HERE/'spatial_isovist_core.py', HERE/'Analyze_NextDecisionDistance.py', Path(DEFAULT_GEOMETRY_PATH), REFERENCE]
    manifest = dict(schema=SCHEMA, status='building', created_at=datetime.now().isoformat(),
        config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        cohort_source='common.load_trials()', cohort_policy=COHORT_POLICY,
        source_cohort_trials=len(canonical), excluded_trials=len(excluded),
        exclusion_counts=excluded.exclusion_reason.value_counts().to_dict(),
        cohort_trials=len(trials), cohort_participants=trials.ParticipantID.nunique(),
        timestamp_unit='ns', state_numeric=STATE_NUMERIC,
        source_hashes={str(p):sha256(p) for p in source_paths}, participant_splits=split,
        versions={name:__import__(name).__version__ for name in ['numpy','pandas','scipy','sklearn','torch']},
        trials=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    trials.assign(split=trials.ParticipantID.map(split)).to_csv(output/'roster.csv', index=False, encoding='utf-8-sig')
    excluded.assign(split=excluded.ParticipantID.map(split))[
        ['trial_id','ParticipantID','Group','Scenario','Guidance','Smoke','split','exclusion_reason']
    ].to_csv(output/'excluded_trials.csv', index=False, encoding='utf-8-sig')
    spatial, decision = PlanarIsovist(), NextDecisionDistance()
    gaze = GuidanceGazeQuery(trials=trials)
    audits, all_runs = [], []
    started = time.perf_counter()
    for row in tqdm(trials.itertuples(index=False), total=len(trials), desc='Build IRL inputs'):
        trial_id = row.trial_id
        df, runs, audit = build_trial(row, trial_id, split[row.ParticipantID], args, spatial, decision, gaze)
        path = Path('trials')/(trial_id+'.csv.gz')
        df.to_csv(output/path, index=False, compression='gzip')
        inputs = [Path(row.SubjectFile), Path(row.EyeGazeFile)]
        if row.Guidance in CHANNELS:
            inputs.append(gaze._aoi_path(row))  # provenance only; labels use public query_frame_many
        manifest['trials'].append(dict(trial_id=trial_id, participant_id=row.ParticipantID,
            guidance=row.Guidance, smoke=row.Smoke, split=split[row.ParticipantID], file=str(path),
            rows=len(df), valid_rows=audit['valid_rows'], sha256=sha256(output/path),
            input_hashes={str(p):sha256(p) for p in inputs}))
        audits.append(audit)
        all_runs.extend(runs)
        gaze.clear_cache()
        save()
    pd.DataFrame(all_runs).to_csv(output/'action_runs.csv', index=False)
    pd.DataFrame([{k:v for k,v in a.items() if not isinstance(v,dict)} for a in audits]).to_csv(output/'trial_quality.csv', index=False)
    (output/'quality_details.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest.update(status='complete', elapsed_seconds=time.perf_counter()-started,
        sampled_rows=sum(a['sampled_rows'] for a in audits), valid_rows=sum(a['valid_rows'] for a in audits),
        mixed_action_trials=sum(a['mixed_actions'] for a in audits),
        action_one=sum(a['action_one'] for a in audits), action_zero=sum(a['action_zero'] for a in audits))
    save()
    print(json.dumps({k:manifest[k] for k in ['status','sampled_rows','valid_rows','mixed_action_trials','action_one','action_zero','elapsed_seconds']}, indent=2))
    print('DATASET='+str(output))


if __name__ == '__main__':
    main()
