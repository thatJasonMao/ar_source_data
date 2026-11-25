"""Build wide full-frame, four-channel IRL inputs. Does not train anything.

Kinematics/geometry are computed once per original trial. An optional verified
V3 dataset can supply unchanged features; ALL actions/runs are rebuilt from V6
AOIs, including actions in the four previously supported conditions.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import Build_IRL_Dataset_V3 as base
from IRL_Channel_Cohort_V6 import SCHEMA, COHORT_POLICY, CHANNEL_COLUMNS, select_trials
from Query_GuidanceChannels_V6 import GuidanceChannelQuery, StaticFrameAdapter
from Guidance_Channel_Contract import (ROOT, CHANNELS, AVAILABLE, AOI_ROOT, AOI_SCHEMA,
    aoi_path, digest)

HERE = Path(__file__).resolve().parent


def attach_channels(df, row, query, args):
    """Rebuild validity, history, gaps and exposure independently per channel."""
    old = CHANNEL_COLUMNS + ['gaze_status', 'gaze_sample_age_ms', 'gaze_types']
    df = df.drop(columns=[c for c in old if c in df]).copy()
    ns = df.timestamp_ns.to_numpy(np.int64)
    dt = np.r_[np.nan, np.diff(ns) / 1e9]
    linked = np.isfinite(dt) & (dt > 0) & (dt <= args.max_gap_s)
    nominal = float(np.median(dt[linked]))
    xyz = df[['x', 'y', 'z']].to_numpy(float)
    moved = np.r_[False, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 3]
    df['after_exit'] = df.stage.eq('arrived_exit').cummax().astype('int8')
    eligible = (df.spatial_status.eq('ok').to_numpy() & df.decision_status.eq('ok').to_numpy()
        & np.isfinite(df.speed.to_numpy()) & np.isfinite(df.speed_prev.to_numpy())
        & df.after_exit.eq(0).to_numpy() & ~moved & np.isfinite(xyz).all(1))
    df['spatial_dynamic_valid'] = eligible.astype('int8')
    df['condition'] = row.Guidance
    df = df.drop(columns=['guidance'], errors='ignore')
    runs, audit = [], []
    for channel in CHANNELS:
        actions = query.query_frames(row, channel, ns)
        valid = eligible & (actions >= 0)
        seg, run, weights, records = base.frame_runs(valid, actions, linked,
            df.dt_next_s.to_numpy(), nominal, args.run_cap_s)
        previous = np.r_[-1, actions[:-1]].astype(np.int8)
        previous[~((seg >= 0) & (seg == np.r_[-2, seg[:-1]]) & linked)] = -1
        transition = (seg >= 0) & (seg == np.r_[seg[1:], -2]) & np.r_[linked[1:], False]
        values = dict(action=actions, valid=valid.astype('int8'), previous_action=previous,
            segment_id=seg, run_id=run, run_weight=weights, transition_valid=transition.astype('int8'),
            trap_observed=np.where(valid, df.slow_state.eq(1).to_numpy() & (actions == 1), -1))
        for field, value in values.items():
            df[field + '_' + channel] = value
        if channel not in AVAILABLE[row.Guidance]:
            if valid.any() or np.any(actions != -1):
                raise ValueError('Unavailable channel acquired labels')
            continue
        for record in records:
            record.update(trial_id=row.trial_id + ':' + channel, source_trial_id=row.trial_id,
                channel=channel, condition=row.Guidance, participant_id=row.ParticipantID,
                scenario=row.Scenario, split=df.split.iloc[0])
        runs.extend(records)
        audit.append(dict(trial_id=row.trial_id, participant_id=row.ParticipantID, condition=row.Guidance,
            smoke=row.Smoke, channel=channel, split=df.split.iloc[0], rows=len(df),
            valid_rows=int(valid.sum()), one=int((valid & (actions == 1)).sum()),
            zero=int((valid & (actions == 0)).sum()), runs=len(records), nominal_step_s=nominal))
    return df, runs, audit


def reusable_dataset(path, args):
    if path is None:
        return None, {}
    path = path.resolve()
    m = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if m['schema'] != base.SCHEMA or m['status'] != 'complete':
        raise ValueError('Feature reuse requires completed V3 full-frame data')
    for name in ['max_gap_s', 'history_s', 'slow_fraction', 'slow_duration_s', 'seed']:
        if m['config'][name] != getattr(args, name):
            raise ValueError('Feature reuse config differs: ' + name)
    for file, expected in m['source_hashes'].items():
        if digest(file) != expected:
            raise ValueError('Feature reuse source changed: ' + file)
    return m, {r['trial_id']: r for r in m['trials']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--aoi-root', type=Path, default=AOI_ROOT)
    p.add_argument('--reuse-spatial-from', type=Path)
    p.add_argument('--max-gap-s', type=float, default=.1)
    p.add_argument('--history-s', type=float, default=1.)
    p.add_argument('--run-cap-s', type=float, default=2.)
    p.add_argument('--slow-fraction', type=float, default=.25)
    p.add_argument('--slow-duration-s', type=float, default=.4)
    p.add_argument('--seed', type=int, default=20260926)
    args = p.parse_args()
    args.gaze_match, args.gaze_max_age_ms = 'exact', 50.
    if not all(np.isfinite(v) and v > 0 for v in [args.max_gap_s, args.history_s, args.run_cap_s, args.slow_duration_s]) or not 0 < args.slow_fraction < 1:
        p.error('Invalid temporal settings')
    aoi_manifest = json.loads((args.aoi_root / 'manifest.json').read_text(encoding='utf-8'))
    if aoi_manifest.get('schema') != AOI_SCHEMA or aoi_manifest.get('status') != 'complete':
        raise ValueError('Complete V6 channel AOIs first')
    aoi_records = {(r['scenario'], r['participant'], r['source_file']): r for r in aoi_manifest['trials']}
    canonical = base.load_trials()
    trials, excluded = select_trials(canonical)
    splits = base.split_roster(canonical, args.seed)
    reuse_manifest, reusable = reusable_dataset(args.reuse_spatial_from, args)
    output = (args.output or HERE / 'Results_IRL_Dataset_V6' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'trials').mkdir()
    sources = [Path(__file__), HERE / 'IRL_Channel_Cohort_V6.py', HERE / 'Query_GuidanceChannels_V6.py',
        ROOT / 'Guidance_Channel_Contract.py', HERE / 'Build_IRL_Dataset_V3.py', HERE / 'Build_IRL_Dataset.py',
        ROOT / 'common.py', HERE / 'spatial_isovist_core.py', HERE / 'Analyze_NextDecisionDistance.py',
        Path(base.DEFAULT_GEOMETRY_PATH), base.REFERENCE, args.aoi_root / 'manifest.json']
    m = dict(schema=SCHEMA, status='building', cohort_policy=COHORT_POLICY, channels=CHANNELS,
        created_at=datetime.now().isoformat(), cohort_trials=len(trials), cohort_participants=trials.ParticipantID.nunique(),
        source_cohort_trials=len(canonical), excluded_trials=len(excluded), participant_splits=splits,
        sampling_mode=base.SAMPLING, action_definition='per-channel exact raw-frame Hit1/AOI; -1 unavailable or invalid',
        config={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_hashes={str(p.resolve()): digest(p) for p in sources},
        reuse_manifest_sha256=digest(args.reuse_spatial_from / 'manifest.json') if reuse_manifest else None, trials=[])
    save = lambda: (output / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    trials.assign(split=trials.ParticipantID.map(splits)).to_csv(output / 'roster.csv', index=False)
    excluded.assign(exclusion_reason='ljn_c12_textbar_excluded_by_user').to_csv(output / 'excluded_trials.csv', index=False)
    spatial, decision = base.PlanarIsovist(), base.NextDecisionDistance()
    query = GuidanceChannelQuery(args.aoi_root)
    audits, all_runs = [], []
    started = time.perf_counter()
    for row in tqdm(trials.itertuples(), total=len(trials), desc='Build V6 channel inputs'):
        apath = aoi_path(row, args.aoi_root)
        ar = aoi_records[(row.Scenario, row.ParticipantID, row.FileName)]
        if digest(apath) != ar['sha256']:
            raise ValueError('AOI checksum changed: ' + str(apath))
        for file, checksum in ar['input_hashes'].items():
            if digest(file) != checksum:
                raise ValueError('AOI raw/legacy input changed: ' + file)
        rec = reusable.get(row.trial_id)
        if rec is not None:
            if (rec['participant_id'], rec['guidance'], rec['smoke'], rec['split']) != (row.ParticipantID, row.Guidance, row.Smoke, splits[row.ParticipantID]):
                raise ValueError('Reused trial identity differs')
            for file in [row.SubjectFile, row.EyeGazeFile]:
                if rec['input_hashes'].get(str(Path(file))) != digest(file):
                    raise ValueError('Reused raw source differs: ' + file)
            oldpath = args.reuse_spatial_from / rec['file']
            if digest(oldpath) != rec['sha256']:
                raise ValueError('Reused feature file checksum differs')
            df = pd.read_csv(oldpath)
        else:
            df, _, _ = base.build_trial(row, row.trial_id, splits[row.ParticipantID], args,
                spatial, decision, StaticFrameAdapter(query, row))
        df, runs, audit = attach_channels(df, row, query, args)
        path = Path('trials') / (row.trial_id + '.csv.gz')
        df.to_csv(output / path, index=False, compression='gzip')
        m['trials'].append(dict(trial_id=row.trial_id, participant_id=row.ParticipantID, condition=row.Guidance,
            scenario=row.Scenario, smoke=row.Smoke, source_file=row.FileName, split=splits[row.ParticipantID],
            channels=list(AVAILABLE[row.Guidance]), file=str(path), rows=len(df), sha256=digest(output / path),
            channel_valid_rows={a['channel']: a['valid_rows'] for a in audit}, reused_spatial=rec is not None,
            input_hashes={str(Path(p).resolve()): digest(p) for p in [row.SubjectFile, row.EyeGazeFile, apath]}))
        audits.extend(audit); all_runs.extend(runs); query.clear_cache()
        save()
    quality = pd.DataFrame(audits)
    quality.to_csv(output / 'trial_channel_quality.csv', index=False)
    pd.DataFrame(all_runs).to_csv(output / 'action_runs.csv', index=False)
    summary = quality.groupby(['split', 'channel', 'condition', 'smoke']).agg(trials=('rows', 'size'), rows=('rows', 'sum'), valid=('valid_rows', 'sum'), one=('one', 'sum'), zero=('zero', 'sum'))
    summary.to_csv(output / 'channel_condition_quality.csv')
    nominal = quality.loc[quality.split.eq('train')].drop_duplicates('trial_id').nominal_step_s
    m['config']['step_s'] = float(nominal.median())
    m.update(status='complete', sampled_rows=sum(r['rows'] for r in m['trials']), channel_trials=len(quality),
        channel_valid_rows=int(quality.valid_rows.sum()), elapsed_seconds=time.perf_counter() - started, trained=False)
    save()
    print('DATASET=' + str(output))


if __name__ == '__main__':
    main()
