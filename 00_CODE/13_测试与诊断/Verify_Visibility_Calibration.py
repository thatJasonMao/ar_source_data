"""Audit calibrated V6 inputs and V6_2 readers without fitting any model.

Checks every source trial and hash; recomputes sampled spatial features directly;
compares all old/new non-isovist columns; validates all selected V6_2 frames.
Only the audit JSON is written. No optimizer, state clustering or training runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / '0_RL_SOURCE', ROOT / '0_RL_FRAMEWORK'):
    sys.path.insert(0, str(path))
from Guidance_Channel_Contract import CHANNELS, digest
from IRL_Channel_Cohort_V6 import validate_manifest, CHANNEL_COLUMNS
from spatial_isovist_core import PlanarIsovist, SMOKE_VISIBILITY_M
from Analyze_NextDecisionDistance import NextDecisionDistance
import Train_IRL_V6_2 as framework

METRICS = ('concavity', 'distance_variation', 'elongation', 'horizontal_size')
EXPECTED = {'HighSmoke': 10.0, 'LowSmoke': 14.0}
DERIVED = {'spatial_status', 'spatial_dynamic_valid'} | {
    field + '_' + channel for field in CHANNEL_COLUMNS if field != 'action' for channel in CHANNELS}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def differences(a, b):
    """Preserve exact integer timestamps; tolerate only float serialization noise."""
    if len(a) != len(b):
        raise ValueError('Cannot compare different row counts')
    if is_integer_dtype(a.dtype) and is_integer_dtype(b.dtype):
        return a.to_numpy() != b.to_numpy()
    if is_numeric_dtype(a.dtype) and is_numeric_dtype(b.dtype):
        return ~np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9,
                           atol=1e-10, equal_nan=True)
    return ~(a.eq(b) | (a.isna() & b.isna())).fillna(False).to_numpy(bool)


def checked_path(dataset, record):
    path = (dataset / record['file']).resolve()
    if not path.is_relative_to(dataset) or digest(path) != record['sha256']:
        raise ValueError('Trial checksum/path differs: ' + record['trial_id'])
    return path


def _audit(dataset, baseline, legacy_model):
    started = time.perf_counter()
    if SMOKE_VISIBILITY_M != EXPECTED or framework.VISIBILITY != EXPECTED:
        raise ValueError('Source/framework calibration differs from final 10/14 m')
    current = read_json(dataset / 'manifest.json')
    previous = read_json(baseline / 'manifest.json')
    validate_manifest(current)
    validate_manifest(previous)
    if current.get('smoke_visibility_m') != EXPECTED:
        raise ValueError('Dataset does not explicitly declare 10/14 m')
    if current['participant_splits'] != previous['participant_splits']:
        raise ValueError('Participant split changed')
    old_records = {r['trial_id']: r for r in previous['trials']}
    if set(old_records) != {r['trial_id'] for r in current['trials']}:
        raise ValueError('Trial identities changed')
    expected_files = {(dataset / r['file']).resolve() for r in current['trials']}
    if expected_files != {p.resolve() for p in (dataset / 'trials').glob('*.csv.gz')}:
        raise ValueError('Unexpected or missing trial files')
    checksums = {}

    def verify_sources(entries):
        for name, expected in entries.items():
            path = Path(name).resolve()
            if path not in checksums:
                checksums[path] = digest(path)
            if checksums[path] != expected:
                raise ValueError('Source provenance hash differs: ' + str(path))

    verify_sources(current['source_hashes'])
    reuse_verified = 0
    reuse = current['config'].get('reuse_spatial_from')
    if reuse:
        reuse_path = Path(reuse).resolve()
        if digest(reuse_path / 'manifest.json') != current.get('reuse_manifest_sha256'):
            raise ValueError('Reused V3 manifest checksum differs')
        reused = read_json(reuse_path / 'manifest.json')
        if reused.get('status') != 'complete' or reused.get('smoke_visibility_m') != EXPECTED:
            raise ValueError('Reused V3 features have stale/unknown visibility')
        verify_sources(reused['source_hashes'])
        reused_records = {r['trial_id']: r for r in reused['trials']}
        for record in current['trials']:
            if record['reused_spatial']:
                checked_path(reuse_path, reused_records[record['trial_id']])
                reuse_verified += 1
    spatial, decision = PlanarIsovist(), NextDecisionDistance()
    totals, changed, nonspatial, derived = Counter(), Counter(), Counter(), Counter()
    records = []
    for record in tqdm(current['trials'], desc='Audit 10/14 m source trials'):
        prior = old_records[record['trial_id']]
        for key in ('trial_id', 'participant_id', 'condition', 'scenario', 'smoke', 'split', 'source_file', 'channels'):
            if record[key] != prior[key]:
                raise ValueError(f'Trial identity changed: {record["trial_id"]}/{key}')
        if record['input_hashes'] != prior['input_hashes']:
            raise ValueError('Raw/AOI input provenance changed: ' + record['trial_id'])
        verify_sources(record['input_hashes'])
        frame = pd.read_csv(checked_path(dataset, record))
        old = pd.read_csv(checked_path(baseline, prior))
        if len(frame) != record['rows'] or len(frame) != len(old):
            raise ValueError('Source frame count changed: ' + record['trial_id'])
        if set(frame) != set(old) | {'visibility_m'}:
            raise ValueError('Unexpected frame schema change: ' + record['trial_id'])
        for key in ('trial_id', 'participant_id', 'condition', 'smoke', 'split'):
            if not frame[key].eq(record[key]).all():
                raise ValueError('Frame identity mismatch: ' + record['trial_id'] + '/' + key)
        if not frame.visibility_m.eq(EXPECTED[record['smoke']]).all():
            raise ValueError('Incorrect frame visibility: ' + record['trial_id'])
        diff = {}
        for column in old:
            n = int(differences(frame[column], old[column]).sum())
            if n:
                diff[column] = n
                (changed if column in METRICS else derived if column in DERIVED else nonspatial)[column] += n
        xyz = frame[['x', 'y', 'z']].to_numpy(float)
        moved = np.r_[False, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 3]
        eligible = (frame.spatial_status.eq('ok').to_numpy() & frame.decision_status.eq('ok').to_numpy()
                    & np.isfinite(frame.speed) & np.isfinite(frame.speed_prev)
                    & frame.after_exit.eq(0).to_numpy() & ~moved & np.isfinite(xyz).all(1))
        if not np.array_equal(frame.spatial_dynamic_valid.to_numpy(bool), eligible):
            raise ValueError('Spatial/dynamic eligibility differs from definition')
        for channel in record['channels']:
            valid = eligible & frame['action_' + channel].ge(0)
            if not np.array_equal(frame['valid_' + channel].to_numpy(bool), valid):
                raise ValueError('Channel validity differs from geometry/action eligibility')
            drops = [f + '_' + c for c in CHANNELS if c != channel for f in CHANNEL_COLUMNS]
            selected = frame.drop(columns=drops).rename(columns={f + '_' + channel: f for f in CHANNEL_COLUMNS})
            framework.validate_trial(selected, record, channel, current)
            totals['all_channel_sequences_validated'] += 1
        # A new radius may cure a degenerate polygon: verify every changed status directly.
        for index in np.flatnonzero(differences(frame.spatial_status, old.spatial_status)):
            row = frame.iloc[index]
            status = spatial.query(row.x, row.y, row.z, EXPECTED[record['smoke']]).status
            if status != row.spatial_status:
                raise ValueError('Changed geometry status is not supported by direct query')
            totals['changed_status_queries'] += 1
        # Rows eligible under the complete new geometry/decision contract.
        indices = np.flatnonzero(frame.spatial_status.eq('ok') & frame.decision_status.eq('ok'))
        if not len(indices):
            raise ValueError('Trial has no valid geometry: ' + record['trial_id'])
        sample = np.unique(indices[np.linspace(0, len(indices) - 1, 3, dtype=int)])
        for index in sample:
            row = frame.iloc[index]
            point = spatial.query(row.x, row.y, row.z, EXPECTED[record['smoke']])
            target = decision.query(row.x, row.y, row.z)
            if point.status != 'ok' or target.status != 'ok':
                raise ValueError('Direct point query disagrees with valid source frame')
            actual = np.array([getattr(point, name) for name in METRICS])
            if not np.allclose(actual, row[list(METRICS)].to_numpy(float), rtol=1e-8, atol=1e-9):
                raise ValueError('Direct isovist recomputation differs: ' + record['trial_id'])
            if (not np.isclose(target.next_decision_distance, row.next_decision_distance,
                               rtol=1e-9, atol=1e-10)
                    or target.floor != row.floor or target.stage != row.stage):
                raise ValueError('Decision geometry differs: ' + record['trial_id'])
        totals['direct_geometry_samples'] += len(sample)
        totals['rows'] += len(frame)
        totals['trials'] += 1
        records.append(dict(trial_id=record['trial_id'], smoke=record['smoke'],
                            rows=len(frame), changed_columns=diff, geometry_samples=sample.tolist()))
    # Dataset helper tables should preserve exposure/identity with unchanged valid masks.
    table_differences = {}
    for name in ('action_runs.csv', 'roster.csv', 'excluded_trials.csv'):
        a, b = pd.read_csv(dataset / name), pd.read_csv(baseline / name)
        if list(a) != list(b) or len(a) != len(b):
            table_differences[name] = {'schema_or_row_count_changed': True}
        else:
            delta = {c: int(differences(a[c], b[c]).sum()) for c in a}
            if any(delta.values()):
                table_differences[name] = {c: n for c, n in delta.items() if n}
    manifest, scope = framework.audit_scope(dataset)
    if scope['selected_trials'] != 178 or scope['source_trials'] != 268:
        raise ValueError('Unexpected V6_2 source/selected scope')
    reader = framework.TrialReader(dataset, manifest, framework.Profile(),
                                    dataset, framework.geometry_contract(manifest))
    selected_trials = 0
    outgoing = defaultdict(lambda: [0, 0])
    for channel in CHANNELS:
        for record, frame in tqdm(reader.trials(channel, {'train', 'val', 'test'},
                                 manifest['config']['run_cap_s']),
                                 total=scope['by_channel'][channel]['trials'],
                                 desc='V6_2 read-only ' + channel):
            selected_trials += 1
            if record['split'] != 'train':
                continue
            terminal = (frame.valid.eq(1) & frame.after_exit.eq(0)
                        & frame.after_exit.shift(-1).eq(1) & frame.dt_next_s.gt(0)
                        & frame.dt_next_s.le(manifest['config']['max_gap_s']))
            support = frame.transition_valid.eq(1) | terminal
            key = channel + '/' + record['smoke']
            for action in (0, 1):
                outgoing[key][action] += int((support & frame.action.eq(action)).sum())
    if selected_trials != 178 or len(outgoing) != 8 or any(min(v) <= 0 for v in outgoing.values()):
        raise ValueError('Own-condition training action/transition support is incomplete')
    # main without --train is the documented manifest-only plan path.
    output = io.StringIO()
    with redirect_stdout(output):
        framework.main(['--dataset', str(dataset)])
    plan = json.loads(output.getvalue())
    if plan.get('trained') is not False:
        raise ValueError('Plan mode did not declare trained=False')
    stale_rejection = None
    try:
        with redirect_stdout(io.StringIO()):
            framework.main(['--dataset', str(baseline)])
    except ValueError as exc:
        stale_rejection = str(exc)
    if not stale_rejection or 'visibility' not in stale_rejection.lower():
        raise ValueError('V6_2 plan did not reject old-visibility inputs explicitly')
    legacy_checks = []
    if legacy_model is not None:
        rule = framework.SpatialRule(legacy_model)
        if rule.visibility != {'HighSmoke': 7.5, 'LowSmoke': 12.5}:
            raise ValueError('Legacy model silently acquired the new visibility')
        for channel in CHANNELS:
            path = legacy_model / 'Results_Global_Spatial_Field_V6_2' / (channel + '_global_field.csv.gz')
            frame = pd.read_csv(path, nrows=20).drop_duplicates('smoke')
            if set(frame.smoke) != {'HighSmoke', 'LowSmoke'}:
                raise ValueError('Cannot sample both legacy smoke fields')
            for row in frame.itertuples():
                result = rule.query_position(row.x, None, row.z, channel, smoke=row.smoke, floor=row.floor)
                if (result['status'] != 'ok' or result['visibility_m'] != rule.visibility[row.smoke]
                        or not np.isclose(result['relative_information_utility'], row.relative_information_utility,
                                          rtol=3e-5, atol=3e-5)):
                    raise ValueError('Legacy model point query regression: ' + channel + '/' + row.smoke)
                legacy_checks.append(dict(channel=channel, smoke=row.smoke, visibility_m=result['visibility_m']))
    invalid_table_changes = {k: v for k, v in table_differences.items()
                             if k != 'action_runs.csv' or not derived}
    checks_passed = not nonspatial and not invalid_table_changes
    return dict(status='passed' if checks_passed else 'failed', ready_for_training=checks_passed,
                smoke_visibility_m=EXPECTED, dataset=str(dataset), baseline=str(baseline),
                manifest_sha256=digest(dataset / 'manifest.json'),
                baseline_manifest_sha256=digest(baseline / 'manifest.json'),
                source_files_hash_verified=len(checksums), **dict(totals),
                reused_spatial_trials_verified=reuse_verified,
                changed_isovist_rows_by_metric=dict(changed),
                non_isovist_column_differences=dict(nonspatial),
                geometry_derived_column_differences=dict(derived),
                table_differences=table_differences, scope=scope,
                selected_trials_reader_checked=selected_trials,
                own_condition_training_transition_counts=dict(outgoing),
                stale_dataset_rejection=stale_rejection, legacy_model_checks=legacy_checks,
                optimizer_executed=False, model_fitted=False, training_executed=False,
                training_entrypoints_guarded=True,
                note='Test frames used only for input integrity; no fit, selection or outcome scoring.',
                elapsed_seconds=time.perf_counter()-started, per_trial=records)


def audit(dataset, baseline, legacy_model):
    """Fail before any model preparation/optimization can execute accidentally."""
    with ExitStack() as guards:
        for owner, name in ((framework, 'build_mdp'), (framework, 'train_candidate'),
                            (framework, 'search_channel'), (framework.Encoder, 'fit')):
            guards.enter_context(patch.object(owner, name,
                side_effect=RuntimeError('Training is forbidden in calibration audit: ' + name)))
        return _audit(dataset, baseline, legacy_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path,
                        default=ROOT/'0_RL_SOURCE/Results_IRL_Dataset_V6/20261003_010150_548829')
    parser.add_argument('--legacy-model', type=Path,
                        default=ROOT/'0_RL_FRAMEWORK/Results_IRL/20261003_230129_557741_V6_2/Best_Model')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset/'visibility_calibration_audit.json'
    try:
        result = audit(args.dataset.resolve(), args.baseline.resolve(), args.legacy_model.resolve())
    except Exception as exc:
        result = dict(status='failed', error=str(exc), optimizer_executed=False,
                      model_fitted=False, training_executed=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_trial'}, ensure_ascii=False, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
