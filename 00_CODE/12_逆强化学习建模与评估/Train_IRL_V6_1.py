"""V6_1: strategy 2B, baseline StaticSign plus channel-specific new guidance.

Direct supervision (applied equally to train, validation and final test):
    StaticSign  <- StaticSign condition only
    ArrowRibbon <- ArrowRibbon and Full conditions
    MiniMap     <- MiniMap and Full conditions
    TextBar     <- TextBar and Full conditions

Reuse the original V6 wide-table dataset, frame labels, temporal runs, features
and participant splits. Whole trial/channel sequences are selected; no gaze
frames are removed from the retained new channels and no labels are OR-ed.
The reward remains one joint shared-plus-channel-deviation model. StaticSign's
direct supervision is baseline-only; shared parameters still borrow information
from the other channels, including Full.

This is the only new source file. Explicit, process-local adapters reuse the V6
data/cache/optimizer modules without editing them. Worker processes re-enter
this file so the same profile is installed before reading any training cache.
State/frame caches and candidate reuse are isolated from the original V6.
The exported model retains version=6 as its V6 query/plot serialization schema;
training_variant=V6_1 and training_profile identify the new estimation scope.

Public arguments and search/optimization defaults match Train_IRL_V6.py.
Without --train this prints the plan only. Example from the project root:
    python -B 0_RL_FRAMEWORK/Train_IRL_V6_1.py --dataset <V6 dataset> --train
Outputs: 0_RL_FRAMEWORK/Results_IRL/<timestamp>_V6_1/Best_Model/
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

import irl_data_v6 as data_v6
import irl_search_v6 as search_v6
import irl_frame_optimization_v6 as frames_v6
import Train_IRL_V6 as trainer_v6
from IRL_V4_Worker import configure_device
from Guidance_Channel_Contract import CONDITIONS

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
GUIDANCE = data_v6.GUIDANCE
VARIANT = 'V6_1'
PROFILE_ID = 'strategy_2b_baseline_static_new_channels_with_full'
CACHE_SCHEMA = 'four_channel_full_frame_spatial_irl_v6_1_2b'
DIRECT_CONDITIONS = {
    'StaticSign': ['StaticSign'],
    'ArrowRibbon': ['ArrowRibbon', 'Full'],
    'MiniMap': ['MiniMap', 'Full'],
    'TextBar': ['TextBar', 'Full'],
}
write_json = search_v6.write_json
digest = search_v6.digest
config_id = search_v6.config_id

# Keep original functions before installing the explicit in-memory adapters.
_source_channel_records = data_v6.channel_records
_source_state_code_hashes = search_v6.state_code_hashes
_source_training_code_hashes = frames_v6.training_code_hashes
_source_prepare_frames = frames_v6.prepare_frames
_source_load_frames = frames_v6.load_frames
_profile_installed = False


def profile_metadata():
    return dict(id=PROFILE_ID, conditions_by_channel={
        channel: list(conditions) for channel, conditions in DIRECT_CONDITIONS.items()},
        label_policy='unchanged per-channel binary labels; select whole trial/channel sequences',
        applies_to=['train', 'val', 'test'],
        weighting='equal channel / selected condition / participant / trial; capped duration within trial')


def channel_records_2b(manifest, splits=None):
    """Keep baseline static supervision and all existing new-channel records."""
    return [record for record in _source_channel_records(manifest, splits)
            if record['condition'] in DIRECT_CONDITIONS[record['guidance']]]


def state_code_hashes():
    return dict(_source_state_code_hashes(), **{SCRIPT.name: digest(SCRIPT)})


def training_code_hashes():
    return dict(_source_training_code_hashes(), **{
        SCRIPT.name: digest(SCRIPT), 'Train_IRL_V6.py': digest(HERE / 'Train_IRL_V6.py')})


def require_profile(metadata):
    if (metadata.get('training_variant') != VARIANT
            or metadata.get('training_profile') != profile_metadata()
            or metadata.get('profile_source_sha256') != digest(SCRIPT)):
        raise ValueError('Requires an unchanged V6_1 strategy-2B training profile')


def prepare_frames_2b(dataset, cache, folder, splits=('train', 'val')):
    _source_prepare_frames(dataset, cache, folder, splits=splits)
    path = Path(folder) / 'metadata.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata.update(training_variant=VARIANT, training_profile=profile_metadata(),
                    profile_source_sha256=digest(SCRIPT))
    write_json(path, metadata)


def load_frames_2b(folder, split):
    metadata = json.loads((Path(folder) / 'metadata.json').read_text(encoding='utf-8'))
    require_profile(metadata)
    cache_metadata = json.loads((Path(metadata['cache']) / 'metadata.json').read_text(encoding='utf-8'))
    if (cache_metadata.get('schema') != CACHE_SCHEMA
            or cache_metadata.get('state_code_sha256') != state_code_hashes()):
        raise ValueError('V6_1 frames reference an incompatible state cache')
    arrays = _source_load_frames(folder, split)
    if set(np.unique(arrays['guidance'])) != set(range(len(GUIDANCE))):
        raise ValueError('Missing or unexpected channel in V6_1 ' + split)
    for index, channel in enumerate(GUIDANCE):
        allowed = [CONDITIONS.index(condition) for condition in DIRECT_CONDITIONS[channel]]
        selected = arrays['guidance'] == index
        if not np.isin(arrays['condition'][selected], allowed).all():
            raise ValueError('Excluded supervision condition in V6_1 ' + channel)
    return arrays


def install_profile():
    """Install only inside this process, in both coordinator and worker modes.

    V6 iter_trials and build_mdp resolve channel_records in data_v6's globals.
    The frame builder also holds a from-import binding, which must be replaced.
    Consequently encoder fitting, transitions, weights and all split frames see
    exactly the same selected sequences. The upstream manifest stays intact.
    """
    global _profile_installed
    if _profile_installed:
        return
    if list(DIRECT_CONDITIONS) != GUIDANCE:
        raise ValueError('V6 information-channel ordering changed')
    data_v6.channel_records = channel_records_2b
    frames_v6.channel_records = channel_records_2b
    search_v6.METHOD = CACHE_SCHEMA
    search_v6.state_code_hashes = state_code_hashes
    frames_v6.training_code_hashes = training_code_hashes
    frames_v6.prepare_frames = prepare_frames_2b
    frames_v6.load_frames = load_frames_2b
    # V6 export retains its own imported bindings; final test must also use 2B.
    trainer_v6.prepare_frames = prepare_frames_2b
    trainer_v6.load_frames = load_frames_2b
    _profile_installed = True


def audit_dataset_2b(dataset):
    """Audit the complete source first, then declare the selected model scope."""
    if _profile_installed:
        raise RuntimeError('Source integrity audit must precede profile installation')
    source = data_v6.audit_dataset(dataset)
    manifest = json.loads((Path(dataset) / 'manifest.json').read_text(encoding='utf-8'))
    selected = channel_records_2b(manifest)
    source_ids = {record['trial_id'] for record in manifest['trials']}
    if {record['source_trial_id'] for record in selected} != source_ids:
        raise ValueError('Strategy 2B unexpectedly dropped an original trial')
    counts = Counter(record['guidance'] for record in selected)
    mass_by_split, counts_by_split = {}, {}
    for split in ['train', 'val', 'test']:
        records = channel_records_2b(manifest, {split})
        weights = data_v6.trial_masses(records)
        masses = {channel: sum(weights.get(record['trial_id'], 0.) for record in records
                               if record['guidance'] == channel) for channel in GUIDANCE}
        if not all(np.isclose(value, .25) for value in masses.values()):
            raise ValueError('Missing channel in V6_1 ' + split)
        for channel in GUIDANCE:
            present = {record['condition'] for record in records
                       if record['guidance'] == channel and record['valid_rows'] > 0}
            if present != set(DIRECT_CONDITIONS[channel]):
                raise ValueError('Missing supervision condition in V6_1 ' + split + '/' + channel)
        mass_by_split[split] = masses
        counts_by_split[split] = dict(Counter(record['guidance'] for record in records))
    selected_totals = {}
    for key, value in source['by_split_channel_condition'].items():
        _, channel, condition, _ = key.split('/')
        if condition in DIRECT_CONDITIONS[channel]:
            selected_totals[key] = value
    if sum(value['trials'] for value in selected_totals.values()) != len(selected):
        raise ValueError('Source audit and selected channel counts disagree')
    return dict(checks='passed', trained=False, training_variant=VARIANT,
                training_profile=profile_metadata(), source_audit=source,
                source_trials=len(source_ids), participants=manifest['cohort_participants'],
                channel_trials=len(selected), channel_trial_counts=dict(counts),
                channel_trials_by_split=counts_by_split,
                excluded_channel_trials=manifest['channel_trials'] - len(selected),
                by_split_channel_condition=selected_totals, channel_weight_mass=mass_by_split,
                note='Source integrity includes all V6 channels; model supervision uses only the declared 2B scope.')


def cache_for(args, out, clusters, cap, seed):
    spec = dict(clusters=clusters, run_cap_s=cap, seed=seed, prior=.5)
    expected = state_code_hashes()
    for run in sorted((HERE / 'Results_IRL').glob('*_V6_1'), reverse=True):
        for metadata_path in run.glob('Caches/*/metadata.json'):
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (metadata.get('schema') == CACHE_SCHEMA and metadata.get('state') == spec
                    and metadata.get('status') == 'complete'
                    and metadata.get('state_code_sha256') == expected):
                try:
                    search_v6.load_cache(metadata_path.parent, args.dataset)
                except (ValueError, OSError):
                    continue
                return metadata_path.parent
    folder = out / 'Caches' / config_id(spec)
    search_v6.prepare_cache(args.dataset, folder, spec)
    return folder


def launch_batch(configs, args, out, deadline, records, caches, frames):
    pending = [config for config in configs if config_id(config) not in records]
    active = {}
    try:
        while (pending or active) and time.time() < deadline:
            for pid, task in list(active.items()):
                process = task['process']
                if process.poll() is None:
                    continue
                task['log'].close()
                folder = task['folder']
                if process.returncode == 0 and (folder / 'result.json').exists():
                    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
                    result.update(id=task['id'], output=str(folder), cache=str(task['cache']), frames=str(task['frames']))
                else:
                    result = dict(id=task['id'], config=task['config'], status='failed',
                                  output=str(folder), returncode=process.returncode)
                records[task['id']] = result
                del active[pid]
                write_json(out / 'search_records.json', list(records.values()))
                print('FINISHED', task['id'], result['status'], result.get('score'), flush=True)
            while pending and len(active) < args.jobs and time.time() < deadline:
                config = pending.pop(0)
                candidate_id = config_id(config)
                key = (config['clusters'], config['run_cap_s'], config['seed'])
                if key not in caches:
                    caches[key] = cache_for(args, out, *key)
                    frames[key] = out / 'Frame_Caches' / config_id(dict(key=list(key)))
                    print('PREPARE FRAMES', key, flush=True)
                    prepare_frames_2b(args.dataset, caches[key], frames[key])
                folder = out / 'Trials' / candidate_id
                folder.mkdir(parents=True)
                job = dict(config=config, output=str(folder), cache=str(caches[key]), frames=str(frames[key]),
                           dataset=str(args.dataset), device=args.device, threads=2, epochs=args.epochs,
                           patience=args.patience, deadline=deadline, training_variant=VARIANT,
                           training_profile=profile_metadata(), profile_source_sha256=digest(SCRIPT))
                job_path = out / 'Tasks' / (candidate_id + '.json')
                write_json(job_path, job)
                log = (folder / 'worker.log').open('w', encoding='utf-8')
                env = os.environ.copy()
                for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                    env[name] = '2'
                try:
                    process = subprocess.Popen([sys.executable, '-B', str(SCRIPT), '--job', str(job_path)],
                                               stdout=log, stderr=subprocess.STDOUT, env=env)
                except Exception:
                    log.close()
                    raise
                active[process.pid] = dict(process=process, log=log, folder=folder, id=candidate_id,
                                           config=config, cache=caches[key], frames=frames[key])
                print('TRAIN', candidate_id, config, flush=True)
            time.sleep(.5)
        if active:
            for task in active.values():
                (task['folder'] / 'STOP').touch()
            for task in active.values():
                try:
                    task['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    task['process'].terminate()
                    task['process'].wait(timeout=10)
                task['log'].close()
                if task['process'].returncode == 0 and (task['folder'] / 'result.json').exists():
                    result = json.loads((task['folder'] / 'result.json').read_text(encoding='utf-8'))
                    result.update(id=task['id'], output=str(task['folder']), cache=str(task['cache']), frames=str(task['frames']))
                    records[task['id']] = result
            active.clear()
    finally:
        for task in active.values():
            if task['process'].poll() is None:
                task['process'].terminate()
                task['process'].wait(timeout=10)
            task['log'].close()
        write_json(out / 'search_records.json', list(records.values()))


def export(args, out, selected):
    # Reuse V6's final frozen-selection evaluation/export, but stamp the profile
    # before plotting. Only the training variant changes, not the model format.
    export_args = argparse.Namespace(**vars(args))
    export_args.skip_plots = True
    folder = trainer_v6.export(export_args, out, selected)
    config_path = folder / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config.update(training_variant=VARIANT, training_profile=profile_metadata(),
                  profile_source_sha256=digest(SCRIPT), optimization='v6_1_strategy_2b_joint_shared_reward',
                  observation_weighting=profile_metadata()['weighting'],
                  interpretation='Baseline-supervised StaticSign; new channels pooled over own condition and Full; joint shared reward.',
                  plot_status='skipped' if args.skip_plots else 'pending',
                  plot_3d_status='skipped' if args.skip_plots else 'pending')
    config['source_sha256'].update(training_code_hashes())
    write_json(config_path, config)
    for name in ['evaluation.json', 'evaluation_identity.json']:
        path = folder / name
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata.update(training_variant=VARIANT, training_profile=profile_metadata())
        write_json(path, metadata)
    trainer_v6.plot_best_model(folder, skip_plots=args.skip_plots)
    return folder


def run_worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    try:
        require_profile(job)
        if (job['config'].get('training_profile_id') != PROFILE_ID
                or job['config'].get('active_guidance') != GUIDANCE):
            raise ValueError('Worker configuration differs from strategy 2B')
        install_profile()
        frames_v6.train(job)
    except Exception as exc:
        write_json(Path(job['output']) / 'error.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget-hours', type=float, default=1.)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--patience', type=int, default=16)
    parser.add_argument('--horizon-seconds', type=float, default=2.)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--families', default='linear,spline,interaction,rbf32,rbf64')
    parser.add_argument('--reuse-search', type=Path, help='Reuse compatible V6_1 strategy-2B candidates only')
    parser.add_argument('--seed-checks', type=int, default=2, help='Extra clustering seeds, diagnostic only')
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    if any(family not in ['linear', 'spline', 'interaction', 'rbf32', 'rbf64'] for family in args.families.split(',')):
        parser.error('Unknown reward family')
    if args.patience < 1 or args.seed_checks < 0:
        parser.error('Invalid stopping/seed settings')
    if args.jobs < 1 or args.epochs < 5 or args.budget_hours <= 0 or args.horizon_seconds <= 0:
        parser.error('Invalid positive budgets')
    base = dict(lr=.02, l2=.001, deviation_ratio=1., clusters=16, run_cap_s=10., prior=.5,
                seed=20260926, horizon_seconds=args.horizon_seconds, gamma=.98, basis='linear',
                active_guidance=GUIDANCE, training_profile_id=PROFILE_ID)
    configs = [dict(base, clusters=clusters, basis=basis, l2=l2)
               for clusters, basis, l2 in itertools.product([16, 24], args.families.split(','), [.0001, .001, .01])]
    if not args.train:
        print(json.dumps(dict(trained=False, version=6, training_variant=VARIANT,
                              training_profile=profile_metadata(), channels=GUIDANCE,
                              configs=configs, test_used_for_selection=False), indent=2))
        return
    configure_device(args.device)
    data_v6.validate_manifest_scope(json.loads((args.dataset / 'manifest.json').read_text(encoding='utf-8')))
    started = time.time()
    deadline = started + args.budget_hours * 3600 - 180
    if deadline <= started:
        parser.error('Budget must exceed three minutes')
    out = HERE / 'Results_IRL' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_V6_1')
    out.mkdir(parents=True)
    for name in ['Tasks', 'Trials', 'Caches', 'Frame_Caches']:
        (out / name).mkdir()
    run_config = dict(status='running', version=6, training_variant=VARIANT, active_guidance=GUIDANCE,
                      training_profile=profile_metadata(), profile_source_sha256=digest(SCRIPT),
                      **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                      test_used_for_selection=False)
    write_json(out / 'config.json', run_config)
    audit = audit_dataset_2b(args.dataset)
    write_json(out / 'dataset_audit.json', audit)
    print('V6_1 SELECTED CHANNEL TRIALS', audit['channel_trial_counts'], flush=True)
    install_profile()
    selection_rule = 'Max spatial macro AUC among models within 0.005 of best validation guidance macro NLL'
    write_json(out / 'plan.json', dict(configs=configs, training_profile=profile_metadata(),
                                      selection_rule=selection_rule, seed=20260926, test_used=False))
    records, caches, frames = {}, {}, {}
    if args.reuse_search:
        args.reuse_search = args.reuse_search.resolve()
        prior_config = json.loads((args.reuse_search / 'config.json').read_text(encoding='utf-8'))
        require_profile(prior_config)
        if prior_config.get('version') != 6 or prior_config.get('active_guidance') != GUIDANCE:
            raise ValueError('Only compatible V6_1 searches can be reused')
        for record in json.loads((args.reuse_search / 'search_records.json').read_text(encoding='utf-8')):
            if record.get('status') != 'complete':
                continue
            search_v6.load_cache(record['cache'], args.dataset)
            metadata = json.loads((Path(record['frames']) / 'metadata.json').read_text(encoding='utf-8'))
            require_profile(metadata)
            if metadata['manifest_sha256'] != digest(args.dataset / 'manifest.json'):
                raise ValueError('Reused data differs')
            if not (Path(record['output']) / 'reward.pt').exists():
                raise ValueError('Reused checkpoint missing')
            if (record['config'].get('active_guidance') != GUIDANCE
                    or record['config'].get('training_profile_id') != PROFILE_ID
                    or record['id'] != config_id(record['config'])):
                raise ValueError('Reused channel/profile scope differs')
            load_frames_2b(record['frames'], 'val')
            result = json.loads((Path(record['output']) / 'result.json').read_text(encoding='utf-8'))
            if result.get('source_sha256') != training_code_hashes():
                raise ValueError('Reused training implementation differs')
            if result.get('config') != record['config']:
                raise ValueError('Reused result configuration differs')
            record = dict(record, reused_from=str(args.reuse_search))
            records[record['id']] = record
            key = (record['config']['clusters'], record['config']['run_cap_s'], record['config']['seed'])
            caches[key], frames[key] = Path(record['cache']), Path(record['frames'])
        write_json(out / 'reused_candidates.json', list(records.values()))
    launch_batch(configs, args, out, deadline, records, caches, frames)
    best = trainer_v6.rank(records)
    extra = []
    for record in best[:2]:
        extra.append(dict(record['config'], clusters=48))
        for horizon in [1., 5.]:
            extra.append(dict(record['config'], horizon_seconds=horizon))
        for lr in [.005, .05]:
            extra.append(dict(record['config'], lr=lr))
    write_json(out / 'extension_plan.json', extra)
    launch_batch(extra, args, out, deadline, records, caches, frames)
    chosen = trainer_v6.select(records)
    write_json(out / 'selection.json', dict(candidate=chosen, frozen_at=datetime.now().isoformat(),
                                           test_used_for_selection=False, selection_rule=selection_rule,
                                           training_profile=profile_metadata()))
    rows = []
    for record in records.values():
        row = dict(id=record['id'], status=record['status'], **record['config'])
        if record['status'] == 'complete':
            row.update(score=record['score'], val_auc=record['validation']['auc'],
                       spatial_macro_auc=record['validation']['spatial_macro_auc'],
                       best_epoch=record['best_epoch'], epochs_run=record['epochs_run'])
            for guidance, report in record['validation']['by_guidance'].items():
                for metric in ['auc', 'nll', 'spatial_reward_auc']:
                    row[guidance + '_' + metric] = report[metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'search_summary.csv', index=False)
    # Diagnostic seeds do not alter selection and never use the test split.
    checks = {}
    check_configs = [dict(chosen['config'], seed=20260926 + 11 * index)
                     for index in range(1, args.seed_checks + 1)]
    write_json(out / 'seed_plan.json', check_configs)
    launch_batch(check_configs, args, out, deadline, checks, caches, frames)
    write_json(out / 'seed_checks.json', list(checks.values()))
    write_json(out / 'search_records.json', list(records.values()))
    folder = export(args, out, chosen)
    write_json(out / 'completion.json', dict(status='complete', training_variant=VARIANT,
                                            training_profile=profile_metadata(), best_model=str(folder),
                                            total_seconds=time.time() - started,
                                            completed=sum(record['status'] == 'complete' for record in records.values()),
                                            failed=sum(record['status'] != 'complete' for record in records.values()),
                                            test_used_for_selection=False))
    run_config['status'] = 'complete'
    write_json(out / 'config.json', run_config)
    print('BEST_MODEL=' + str(folder))
    print('OUTPUT=' + str(out))


if __name__ == '__main__':
    if '--job' in sys.argv[1:]:
        # Internal subprocess entry; the public coordinator CLI stays V6-compatible.
        worker_parser = argparse.ArgumentParser(description='Internal V6_1 worker')
        worker_parser.add_argument('--job', type=Path, required=True)
        run_worker(worker_parser.parse_args().job)
    else:
        main()
