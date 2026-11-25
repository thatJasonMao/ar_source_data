"""Read-only model readiness audit; save a report without fitting/training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from irl_data_v6 import audit_dataset, iter_trials
from irl_search_v6 import write_json, digest


def validate(dataset):
    result = audit_dataset(dataset)
    manifest = json.loads((Path(dataset) / 'manifest.json').read_text(encoding='utf-8'))
    max_gap = manifest['config']['max_gap_s']
    counts = defaultdict(lambda: [0, 0])
    # State priors never borrow an action across channel/condition/smoke. Verify
    # both actions have actual outgoing training transitions before future fits.
    for rec, frame in tqdm(iter_trials(dataset, {'train'}), desc='Check training transition support'):
        key = '/'.join([rec['guidance'], rec['condition'], rec['smoke']])
        terminal_next = (frame.valid.eq(1) & frame.after_exit.eq(0) & frame.after_exit.shift(-1).eq(1)
            & frame.dt_next_s.gt(0) & frame.dt_next_s.le(max_gap))
        supported = frame.transition_valid.eq(1) | terminal_next
        for action in [0, 1]:
            counts[key][action] += int((supported & frame.action.eq(action)).sum())
    missing = {k: v for k, v in counts.items() if min(v) == 0}
    result.update(training_transition_counts=dict(counts), missing_action_contexts=missing,
        manifest_sha256=digest(Path(dataset) / 'manifest.json'),
        optimizer_executed=False, model_fitted=False, ready_for_training=not missing)
    if missing:
        raise ValueError('Training action dynamics are unsupported: ' + json.dumps(missing))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = validate(args.dataset)
    output = args.output or args.dataset / 'readiness_audit.json'
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['by_split_channel_condition', 'training_transition_counts']}, indent=2))


if __name__ == '__main__':
    main()
