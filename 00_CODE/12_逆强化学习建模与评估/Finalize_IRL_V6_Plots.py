"""Resume V6 / V6_1 post-training plots and completion records; never train.

python -B Finalize_IRL_V6_Plots.py --run "<search-run or Best_Model directory>"
Requires an already selected, exported and evaluated model. Complete 2D fields
are hash-checked and reused. Rendering errors leave the run incomplete.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json

from Train_IRL_V6 import plot_best_model, GUIDANCE
from irl_search_v6 import digest, write_json


def finalize_run(run):
    run = Path(run).resolve()
    if run.name == 'Best_Model':
        run = run.parent
    best = run / 'Best_Model'
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    selected = json.loads((run / 'selection.json').read_text(encoding='utf-8'))['candidate']
    model_config = json.loads((best / 'config.json').read_text(encoding='utf-8'))
    if (model_config.get('status') != 'complete' or model_config.get('version') != 6
            or model_config.get('method') != 'four_channel_full_frame_spatial_irl_v6'
            or model_config.get('active_guidance') != GUIDANCE):
        raise ValueError('Requires an already exported, completed V6 channel model')
    required = ['reward.pt', 'reward_basis.json', 'encoder.json', 'mdp.npz',
                'state_reward_features.npy', 'reward_rules.csv', 'training_history.csv',
                'evaluation.json', 'evaluation_identity.json', 'evaluation_predictions.npz']
    if any(not (best / name).is_file() or (best / name).stat().st_size == 0 for name in required):
        raise ValueError('Model or final evaluation export is incomplete; plotting cannot repair training data')
    evaluation = json.loads((best / 'evaluation.json').read_text(encoding='utf-8'))
    if not evaluation.get('trained') or any(split not in evaluation for split in ['train', 'val', 'test']):
        raise ValueError('Final train/validation/test evaluation must already exist')
    # Keep all numerical/model exports unchanged during post-processing.
    before = {name: digest(best / name) for name in required}
    if digest(Path(selected['output']) / 'reward.pt') != before['reward.pt']:
        raise ValueError('Best_Model checkpoint differs from the frozen selected candidate')
    plot_best_model(best)
    manifest_path = best / 'Results_Global_Spatial_Field_V6' / 'Results_3D' / 'manifest_3d.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    field = manifest_path.parent.parent / 'global_spatial_rule_field.csv.gz'
    expected = [g + '_spatial_demand_' + dimension + '.png'
                for g in GUIDANCE for dimension in ['3d', '2d']]
    if (manifest.get('status') != 'complete' or manifest.get('guidances') != GUIDANCE
            or manifest.get('png_count') != len(expected)
            or manifest.get('input_sha256') != digest(field)
            or manifest.get('script_sha256') != digest(Path(__file__).with_name('Plot_Spatial_Rule_3D.py'))
            or any(not (manifest_path.parent / name).is_file()
                   or (manifest_path.parent / name).stat().st_size == 0 for name in expected)):
        raise ValueError('3D completion manifest or images failed integrity checks')
    if {name: digest(best / name) for name in required} != before:
        raise RuntimeError('Numerical/model exports changed during plot recovery')
    now = datetime.now()
    completion_path = run / 'completion.json'
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
    else:
        records = json.loads((run / 'search_records.json').read_text(encoding='utf-8'))
        started = datetime.strptime(run.name.split('_V6')[0], '%Y%m%d_%H%M%S_%f')
        completion = dict(status='complete', best_model=str(best),
                          total_seconds=(now - started).total_seconds(), elapsed_includes_recovery=True,
                          completed=sum(record['status'] == 'complete' for record in records),
                          failed=sum(record['status'] != 'complete' for record in records),
                          test_used_for_selection=False)
        for key in ['training_variant', 'training_profile']:
            if key in config:
                completion[key] = config[key]
    completion.update(status='complete', plots_recovered_at=now.isoformat(),
                      recovery='Post-training plots only; no retraining; numerical/model exports unchanged')
    write_json(completion_path, completion)
    config.update(status='complete', plots_recovered_at=now.isoformat())
    write_json(run / 'config.json', config)
    print('BEST_MODEL=' + str(best))
    print('OUTPUT=' + str(run))
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    finalize_run(parser.parse_args().run)


if __name__ == '__main__':
    main()
