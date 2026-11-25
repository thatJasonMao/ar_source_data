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
from IRL_Cohort import COHORT_POLICY, select_irl_trials

REFERENCE = ROOT / 'Results_Spatial_Metric_Surfaces/_timestamp_input/extract_timestamp_metrics.py'
spec = importlib.util.spec_from_file_location('irl_timestamp_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

SPATIAL = ['concavity', 'distance_variation', 'elongation', 'horizontal_size', 'next_decision_distance']
STATE_NUMERIC = SPATIAL + ['speed_prev', 'speed_mean_past', 'acceleration_prev',
    'head_rate_prev', 'scan_rate_prev', 'gaze_depth_prev', 'elapsed_s']


from Build_IRL_Dataset import (sha256, ns_array, sample_values, eye_metrics, split_roster)
SCHEMA = 'a5_irl_full_frames_v3'
SAMPLING = 'all_unique_subject_timestamps'

def temporal_columns(df, args):
    """All Subject frames retained; seconds, not frame counts, define history."""
    ns = df.timestamp_ns.to_numpy(np.int64)
    dt = np.r_[np.nan, np.diff(ns)/1e9]
    linked = np.isfinite(dt) & (dt > 0) & (dt <= args.max_gap_s)
    blocks = np.cumsum(~linked)
    df['frame_index'] = np.arange(len(df))
    df['dt_prev_s'] = dt
    df['dt_next_s'] = np.r_[dt[1:], np.nan]
    df['time_block'] = blocks
    for name in ['speed', 'head_rate', 'scan_rate', 'gaze_depth']:
        df[name+'_prev'] = df.groupby('time_block')[name].shift(1)
    baseline = np.full(len(df), np.nan)
    for block in np.unique(blocks):
        ix = np.flatnonzero(blocks == block)
        series = pd.Series(df.speed.to_numpy()[ix], index=pd.to_datetime(ns[ix], unit='ns'))
        baseline[ix] = series.rolling(pd.Timedelta(seconds=args.history_s), closed='left', min_periods=2).mean()
    df['speed_mean_past'] = baseline
    acceleration = df.groupby('time_block').speed.diff()/dt
    df['acceleration_prev'] = acceleration.groupby(df.time_block).shift(1)
    # Diagnostic only. Confirmation uses elapsed time and never crosses gaps.
    slow = np.zeros(len(df), np.int8)
    anchor = onset = None
    for i, (v, b) in enumerate(zip(df.speed, baseline)):
        if not linked[i] or not np.isfinite(v):
            anchor = onset = None
        if not np.isfinite(v):
            continue
        if anchor is None:
            if np.isfinite(b) and b >= .2 and v <= (1-args.slow_fraction)*b:
                anchor, onset = b, ns[i]
        elif v >= (1-args.slow_fraction/2)*anchor:
            anchor = onset = None
        slow[i] = anchor is not None and (ns[i]-onset)/1e9 >= args.slow_duration_s
    df['slow_state'] = slow
    df['slow_ratio'] = (baseline-df.speed)/np.maximum(baseline, .2)
    df['stationary'] = (df.speed <= .1).astype('int8')
    df['turning'] = (df.head_rate >= 30).astype('int8')
    return linked


def frame_runs(valid, actions, linked, dt_next, nominal_s, cap_s):
    """Time-mass weights: sum per constant-action run <= cap_s / nominal_s."""
    n = len(valid)
    segment = np.full(n, -1, int); run = np.full(n, -1, int)
    weights = np.zeros(n); records = []
    sid = -1; i = 0
    while i < n:
        if not valid[i]:
            i += 1; continue
        if i == 0 or not valid[i-1] or not linked[i]: sid += 1
        j = i+1
        while j < n and valid[j] and linked[j] and actions[j] == actions[i]: j += 1
        durations = np.asarray(dt_next[i:j], float).copy()
        # Last row before a gap/end has unknown forward duration; use one nominal frame.
        next_link = np.r_[linked[i+1:j], linked[j] if j < n else False]
        durations[~next_link | ~np.isfinite(durations) | (durations <= 0)] = nominal_s
        total = float(durations.sum())
        segment[i:j] = sid; run[i:j] = len(records)
        weights[i:j] = durations/nominal_s * min(1., cap_s/total)
        records.append(dict(run_id=len(records), segment_id=sid, start_row=i, stop_row=j,
                            action=int(actions[i]), steps=j-i, duration_s=total,
                            capped_duration_s=min(total, cap_s)))
        i = j
    return segment, run, weights, records


def build_trial(row, trial_id, split, args, spatial, decision, gaze):
    subject, sa = reference.read_raw(Path(row.SubjectFile),
        ['TimeStamp', 'PosX', 'PosY', 'PosZ', 'VeloX', 'VeloY', 'VeloZ'])
    if len(subject) < 3: raise ValueError('Too few valid Subject timestamps: '+row.SubjectFile)
    pns = ns_array(subject.ParsedTime)
    delta = np.diff(pns)/1e9
    good_delta = delta[(delta > 0) & (delta <= args.max_gap_s)]
    if not len(good_delta): raise ValueError('No usable consecutive raw frames')
    nominal = float(np.median(good_delta))
    ens, eyes, ea = eye_metrics(row, subject, pns)
    eye_now = sample_values(ens, eyes, pns) if len(ens) else np.full((len(pns), 3), np.nan)
    xyz = subject[['PosX', 'PosY', 'PosZ']].to_numpy(float)
    df = pd.DataFrame(dict(timestamp_ns=pns, subject_timestamp_ns=pns,
        sample_age_ms=np.zeros(len(pns)), elapsed_s=(pns-pns[0])/1e9,
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
        speed=np.hypot(subject.VeloX.to_numpy(float), subject.VeloZ.to_numpy(float)),
        head_rate=eye_now[:,0], scan_rate=eye_now[:,1], gaze_depth=eye_now[:,2]))
    linked = temporal_columns(df, args)
    # Exact frame by default. Missing/unknown AOI remains -1, never silently 0.
    result = gaze.query_frame_many(row.ParticipantID,
        [pd.Timestamp(int(t), unit='ns') for t in pns], scenario=row.Scenario,
        match=args.gaze_match, max_age_ms=args.gaze_max_age_ms)
    df['action'] = [int(r.is_gazing) if r.status == 'ok' else -1 for r in result]
    df['gaze_status'] = [r.status for r in result]
    df['gaze_sample_age_ms'] = [r.sample_age_ms for r in result]
    df['gaze_types'] = ['|'.join(r.guidance_types) for r in result]
    spatial_rows = []
    # Exact repeated coordinates reuse identical geometry queries; no position binning.
    cache = {}
    for x, y, z in tqdm(xyz, desc='Full frames '+trial_id, leave=False):
        key = (float(x), float(y), float(z))
        if key not in cache:
            r = spatial.query(x, y, z, 7.5 if row.Smoke == 'HighSmoke' else 12.5)
            d = decision.query(x, y, z)
            cache[key] = dict(concavity=r.concavity, distance_variation=r.distance_variation,
                elongation=r.elongation, horizontal_size=r.horizontal_size,
                next_decision_distance=d.next_decision_distance, floor=d.floor,
                zone_id=d.zone_id, stage=d.stage, spatial_status=r.status, decision_status=d.status)
        spatial_rows.append(cache[key])
    df = pd.concat([df, pd.DataFrame(spatial_rows)], axis=1)
    usable = ((df.action >= 0) & df.spatial_status.eq('ok') & df.decision_status.eq('ok')
              & np.isfinite(df.speed) & np.isfinite(df.speed_prev))
    df['after_exit'] = df.stage.eq('arrived_exit').cummax().astype('int8')
    moved = np.r_[False, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 3]
    usable &= df.after_exit.eq(0) & ~moved & np.isfinite(xyz).all(axis=1)
    a = df.action.to_numpy(int)
    seg, runs, weights, records = frame_runs(usable.to_numpy(), a, linked,
        df.dt_next_s.to_numpy(), nominal, args.run_cap_s)
    df['segment_id'], df['run_id'], df['run_weight'] = seg, runs, weights
    df['valid'] = usable.astype('int8')
    df['previous_action'] = df.action.shift(1).fillna(-1).astype(int)
    previous_ok = (seg >= 0) & (seg == np.r_[-2,seg[:-1]]) & linked
    df.loc[~previous_ok, 'previous_action'] = -1
    df['transition_valid'] = ((seg >= 0) & (seg == np.r_[seg[1:],-2]) & np.r_[linked[1:],False]).astype('int8')
    df['trap_observed'] = np.where(usable, (df.slow_state.eq(1) & df.action.eq(1)).astype(int), -1)
    for name, value in dict(trial_id=trial_id, participant_id=row.ParticipantID,
        group=row.Group, guidance=row.Guidance, smoke=row.Smoke, scenario=row.Scenario, split=split).items(): df[name] = value
    for record in records: record.update(trial_id=trial_id,participant_id=row.ParticipantID,scenario=row.Scenario,split=split)
    va = df.loc[usable, 'action']
    audit = dict(trial_id=trial_id,participant_id=row.ParticipantID,scenario=row.Scenario,split=split,
        sampled_rows=len(df),valid_rows=int(usable.sum()),action_one=int(va.eq(1).sum()),action_zero=int(va.eq(0).sum()),
        action_fraction=float(va.mean()) if len(va) else None,mixed_actions=bool(va.nunique()==2),
        nominal_step_s=nominal,frame_dt_quantiles_s=np.quantile(good_delta,[0,.5,.95,1]).tolist(),
        gap_count=int((delta>args.max_gap_s).sum()),after_exit_rows=int(df.after_exit.sum()),
        run_count=len(records),longest_run_s=max((r['duration_s'] for r in records),default=0),
        gaze_status_counts=dict(Counter(df.gaze_status)),spatial_status_counts=dict(Counter(df.spatial_status)),
        subject_audit=sa,eye_audit=ea)
    return df, records, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-gap-s', type=float, default=.1)
    parser.add_argument('--gaze-match', choices=['exact','previous'], default='exact')
    parser.add_argument('--gaze-max-age-ms', type=float, default=50.)
    parser.add_argument('--history-s', type=float, default=1.)
    parser.add_argument('--run-cap-s', type=float, default=2.)
    parser.add_argument('--slow-fraction', type=float, default=.25)
    parser.add_argument('--slow-duration-s', type=float, default=.4)
    parser.add_argument('--seed', type=int, default=20260926)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.max_gap_s, args.history_s, args.run_cap_s, args.slow_duration_s)) or not 0 < args.slow_fraction < 1:
        parser.error('Durations must be positive; slow-fraction must be in (0,1)')
    if not np.isfinite(args.gaze_max_age_ms) or args.gaze_max_age_ms < 0: parser.error('Invalid gaze age')
    canonical = load_trials().sort_values(['Scenario','ParticipantID','FileName']).reset_index(drop=True)
    split = split_roster(canonical, args.seed)
    trials, excluded = select_irl_trials(canonical)
    output = args.output or HERE/'Results_IRL_Dataset_V3'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'trials').mkdir()
    source_paths = [Path(__file__), HERE/'Build_IRL_Dataset.py', HERE/'IRL_Cohort.py', ROOT/'common.py', HERE/'Query_GuidanceGaze.py',
        HERE/'spatial_isovist_core.py', HERE/'Analyze_NextDecisionDistance.py', Path(DEFAULT_GEOMETRY_PATH), REFERENCE]
    manifest = dict(schema=SCHEMA, status='building', created_at=datetime.now().isoformat(),
        config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        cohort_source='common.load_trials()', cohort_policy=COHORT_POLICY,
        source_cohort_trials=len(canonical), excluded_trials=len(excluded),
        exclusion_counts=excluded.exclusion_reason.value_counts().to_dict(),
        cohort_trials=len(trials), cohort_participants=trials.ParticipantID.nunique(),
        timestamp_unit='ns', sampling_mode=SAMPLING, action_definition='loose GuidanceGaze query_frame_many', state_numeric=STATE_NUMERIC,
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
    train_steps=[a['nominal_step_s'] for a in audits if a['split']=='train']
    manifest['config']['step_s']=float(np.median(train_steps))
    manifest['config']['step_s_role']='train-trial median frame interval; planning approximation only, never resampling'
    manifest.update(status='complete', elapsed_seconds=time.perf_counter()-started,
        sampled_rows=sum(a['sampled_rows'] for a in audits), valid_rows=sum(a['valid_rows'] for a in audits),
        mixed_action_trials=sum(a['mixed_actions'] for a in audits),
        action_one=sum(a['action_one'] for a in audits), action_zero=sum(a['action_zero'] for a in audits))
    save()
    print(json.dumps({k:manifest[k] for k in ['status','sampled_rows','valid_rows','mixed_action_trials','action_one','action_zero','elapsed_seconds']}, indent=2))
    print('DATASET='+str(output))


if __name__ == '__main__':
    main()
