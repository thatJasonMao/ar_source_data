"""Version-isolated V6 cache contract; legacy caches are never accepted."""
from pathlib import Path
import json
import numpy as np
from irl_search_v4 import write_json, digest, config_id, run_duration_table, recapped_weights, state_spec
from irl_data_v6 import StateEncoder, validate_manifest_scope, GUIDANCE

HERE = Path(__file__).resolve().parent
METHOD = 'four_channel_full_frame_spatial_irl_v6'


def state_code_hashes():
    names = ['irl_data_v6.py', 'irl_data_v3.py', 'irl_state_distance.py', 'irl_search_v6.py',
        '../0_RL_SOURCE/IRL_Channel_Cohort_V6.py', '../Guidance_Channel_Contract.py']
    return {n: digest(HERE / n) for n in names}


def prepare_cache(dataset, folder, config):
    from irl_data_v6 import build_mdp
    dataset, folder = Path(dataset), Path(folder)
    m = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    validate_manifest_scope(m)
    mdp = build_mdp(dataset, config['clusters'], config['seed'], config['prior'], run_cap_s=config['run_cap_s'])
    folder.mkdir(parents=True, exist_ok=False)
    keys = ['centers', 'transitions', 'counts', 'raw_counts', 'transition_counts', 'guidance_index', 'radius', 'terminal_index', 'mean_dt']
    np.savez_compressed(folder / 'mdp.npz', **{k: mdp[k] for k in keys})
    write_json(folder / 'encoder.json', dict(encoder=mdp['encoder'].as_dict(), contexts=mdp['contexts'], step_s=mdp['step_s']))
    write_json(folder / 'metadata.json', dict(status='complete', schema=METHOD, channels=GUIDANCE,
        state=state_spec(config), train_rows=mdp['train_rows'], state_code_sha256=state_code_hashes(),
        manifest_sha256=digest(dataset / 'manifest.json'), runs_sha256=digest(dataset / 'action_runs.csv'),
        files_sha256={n: digest(folder / n) for n in ['mdp.npz', 'encoder.json']}))


def load_cache(folder, dataset=None):
    folder = Path(folder)
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if meta.get('schema') != METHOD or meta.get('status') != 'complete' or meta.get('channels') != GUIDANCE:
        raise ValueError('Not a completed four-channel V6 cache')
    if meta['state_code_sha256'] != state_code_hashes():
        raise ValueError('V6 cache implementation changed')
    if dataset is not None and (meta['manifest_sha256'] != digest(Path(dataset) / 'manifest.json') or meta['runs_sha256'] != digest(Path(dataset) / 'action_runs.csv')):
        raise ValueError('V6 cache data changed')
    for name, checksum in meta['files_sha256'].items():
        if digest(folder / name) != checksum:
            raise ValueError('V6 cache checksum changed')
    saved = json.loads((folder / 'encoder.json').read_text(encoding='utf-8'))
    encoder = StateEncoder()
    encoder.median, encoder.scale = np.array(saved['encoder']['median']), np.array(saved['encoder']['scale'])
    if encoder.names != saved['encoder']['names']:
        raise ValueError('Encoder schema differs')
    with np.load(folder / 'mdp.npz', allow_pickle=False) as f:
        mdp = {k: f[k] for k in f.files}
    mdp.update(encoder=encoder, contexts={k: tuple(v) for k, v in saved['contexts'].items()},
        step_s=saved['step_s'], terminal_index=int(mdp['terminal_index']), train_rows=meta['train_rows'])
    return mdp, None, None
