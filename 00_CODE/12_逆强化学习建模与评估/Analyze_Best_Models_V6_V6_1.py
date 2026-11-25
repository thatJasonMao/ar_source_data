"""Compare fitted V6/V6_1 rules on saved evaluations and identical raw grids.

No training, cache reconstruction, optimization or smoothing. Comparison uses
baseline-only StaticSign for both models; other channels keep own + Full scope.
Global-grid statistics describe geometry cells, not participant-weighted rates.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

from Spatial_IRL_Rule_V6 import SpatialIRLRule, SPATIAL, GUIDANCE

HERE = Path(__file__).resolve().parent
DEFAULTS = {'V6': HERE / 'Results_IRL/20261003_014907_277732_V6/Best_Model',
            'V6_1': HERE / 'Results_IRL/20261003_020827_592825_V6_1/Best_Model'}
CELL = .4
KEYS = ['guidance', 'smoke', 'floor', 'x', 'z']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_vectors(part):
    a = part.relative_information_utility_V6.to_numpy()
    b = part.relative_information_utility_V6_1.to_numpy()
    upper_a, upper_b = a >= np.quantile(a, .9), b >= np.quantile(b, .9)
    return dict(cells=len(part), spearman=float(spearmanr(a, b).statistic),
                mean_delta=float((b-a).mean()), rmse=float(np.sqrt(np.mean((b-a)**2))),
                top10_retention=float((upper_a & upper_b).sum()/upper_a.sum()),
                top10_jaccard=float((upper_a & upper_b).sum()/(upper_a | upper_b).sum()))


def draw_raw(ax, part, norm):
    ix = np.rint(part.x.to_numpy()/CELL-.5).astype(int)
    iz = np.rint(part.z.to_numpy()/CELL-.5).astype(int)
    x0, x1, z0, z1 = ix.min(), ix.max()+1, iz.min(), iz.max()+1
    raw = np.full((z1-z0, x1-x0), np.nan)
    raw[iz-z0, ix-x0] = part.relative_information_utility
    xe = np.arange(x0, x1+1)*CELL
    ze = np.arange(z0, z1+1)*CELL
    ax.pcolormesh(-xe[::-1], -ze[::-1], np.ma.masked_invalid(raw[::-1, ::-1]),
                  norm=norm, cmap='plasma', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlabel('Display X = -Unity X (m)')
    ax.set_ylabel('Display Z = -Unity Z (m)')
    ax.ticklabel_format(style='plain', useOffset=False)


def plot_comparison(fields, channel, target, dpi):
    parts = {v: f[f.guidance.eq(channel)] for v, f in fields.items()}
    all_values = np.concatenate([f.relative_information_utility.to_numpy() for f in parts.values()])
    norm = Normalize(float(all_values.min()), float(all_values.max()))
    fig, axes = plt.subplots(2, 4, figsize=(20, 15), gridspec_kw={'width_ratios': [3, 1, 3, 1]})
    try:
        fig.subplots_adjust(left=.055, right=.90, top=.91, bottom=.08, wspace=.28, hspace=.25)
        for vi, (version, frame) in enumerate(parts.items()):
            for row, smoke in enumerate(['HighSmoke', 'LowSmoke']):
                for col, floor in enumerate(['Hall', 'Platform']):
                    ax = axes[row, vi*2+col]
                    draw_raw(ax, frame[frame.smoke.eq(smoke) & frame.floor.eq(floor)], norm)
                    ax.set_title(f'{version} | {smoke} | {floor}', fontsize=11)
        cax = fig.add_axes([.94, .16, .015, .65])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), cax=cax,
                     label='Signed immediate reward contrast (raw cells)')
        fig.suptitle(channel + ' | Common colour scale across both fitted models', fontsize=17)
        fig.text(.5, .025, 'Original 0.4 m cells; no smoothing/interpolation. Geometry-grid fields are not gaze probabilities.',
                 ha='center', fontsize=10)
        fig.savefig(target, dpi=dpi)
    finally:
        plt.close(fig)


def run_analysis(paths, output, dpi=300):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Comparison output already exists; choose a new --output directory')
    fields, configs, evaluations, rules, inputs = {}, {}, {}, {}, {}
    metric_rows, condition_rows, official_rows, context_rows, components = [], [], [], [], []
    for version, best in paths.items():
        best = Path(best).resolve()
        if best.name != 'Best_Model':
            best = best / 'Best_Model'
        configs[version] = json.loads((best/'config.json').read_text(encoding='utf-8'))
        evaluations[version] = json.loads((best/'evaluation.json').read_text(encoding='utf-8'))
        rules[version] = SpatialIRLRule(best)
        identity = json.loads((best/'evaluation_identity.json').read_text(encoding='utf-8'))
        inputs[version] = dict(best_model=str(best), people=identity['people'],
            sha256={n: sha(best/n) for n in ['config.json', 'encoder.json', 'reward.pt', 'reward_basis.json',
                                           'evaluation.json', 'Results_Global_Spatial_Field_V6/global_spatial_rule_field.csv.gz']})
        for split in ['train', 'val', 'test']:
            report = evaluations[version][split]
            official_rows.append(dict(version=version, split=split, nll=report['guidance_macro_nll'],
                policy_auc=report['guidance_macro_auc'], spatial_auc=report['spatial_macro_auc']))
            for channel in GUIDANCE:
                # Renormalized subset statistics already exist in the saved report.
                d = (report['by_channel_condition']['StaticSign/StaticSign'] if channel == 'StaticSign'
                     else report['by_guidance'][channel])
                metric_rows.append(dict(version=version, split=split, channel=channel, n=d['n'], people=d['people'],
                    nll=d['balanced_nll'], policy_auc=d['auc'], spatial_auc=d['spatial_reward_auc'],
                    brier=d['brier'], observed_fraction_unweighted=d['observed_fraction'],
                    predicted_fraction_unweighted=d['predicted_fraction']))
            for key, d in report['by_channel_condition'].items():
                condition_rows.append(dict(version=version, split=split, channel_condition=key,
                    n=d['n'], people=d['people'], nll=d['balanced_nll'], policy_auc=d['auc'],
                    spatial_auc=d['spatial_reward_auc'], observed_fraction_unweighted=d['observed_fraction']))
        frame = pd.read_csv(best/'Results_Global_Spatial_Field_V6/global_spatial_rule_field.csv.gz')
        if frame.duplicated(KEYS).any() or not np.isfinite(frame[SPATIAL+['relative_information_utility']]).all().all():
            raise ValueError('Duplicate or invalid spatial cells')
        # Forward reconstruction from saved weights/basis verifies field semantics.
        predictions = []
        for channel in GUIDANCE:
            part = frame[frame.guidance.eq(channel)]
            pred = rules[version].query_features(part, channel)
            if not np.allclose(pred, part.relative_information_utility, atol=2e-5, rtol=1e-5):
                raise ValueError('Raw field differs from saved fitted rule')
            predictions.append(pred)
            ref = part.iloc[:1].copy()
            ref['smoke'] = 'HighSmoke'; hi = rules[version].query_features(ref, channel)[0]
            ref['smoke'] = 'LowSmoke'; lo = rules[version].query_features(ref, channel)[0]
            context_rows.append(dict(version=version, channel=channel, direct_high_minus_low=float(hi-lo),
                meaning='Other five geometry features, floor and stage held fixed; not the actual visibility-radius change'))
        base = frame[frame.guidance.eq(GUIDANCE[0])][['smoke', 'floor', 'x', 'z']].copy()
        matrix = np.column_stack(predictions)
        shared = matrix.mean(axis=1)
        for i, channel in enumerate(GUIDANCE):
            components.append(dict(version=version, channel=channel, shared_mean=float(shared.mean()),
                shared_std=float(shared.std()), total_std=float(matrix[:, i].std()),
                deviation_std=float((matrix[:, i]-shared).std()),
                limitation='Shared and deviation components are correlated; variances are not additive'))
        fields[version] = frame
    if inputs['V6']['people'] != inputs['V6_1']['people']:
        raise ValueError('Participant splits differ')
    metrics = pd.DataFrame(metric_rows)
    for (split, channel), group in metrics.groupby(['split', 'channel']):
        if group.n.nunique() != 1 or group.people.nunique() != 1:
            raise ValueError('Common evaluation scopes differ')
    merged = fields['V6'].merge(fields['V6_1'], on=KEYS, suffixes=('_V6', '_V6_1'), validate='one_to_one')
    if len(merged) != len(fields['V6']) or len(merged) != len(fields['V6_1']):
        raise ValueError('Geometry grids differ')
    for name in [*SPATIAL, 'stage']:
        a, b = merged[name+'_V6'], merged[name+'_V6_1']
        if not (a.equals(b) if name == 'stage' else np.allclose(a, b, atol=1e-8, rtol=0)):
            raise ValueError('Geometry features differ')
    common_macro = metrics.groupby(['version', 'split'])[['nll', 'policy_auc', 'spatial_auc']].mean().reset_index()
    stats, stages, contrasts, correlations = [], [], [], []
    for version, frame in fields.items():
        for key, group in frame.groupby(['guidance', 'smoke', 'floor'], sort=False):
            values = group.relative_information_utility
            stats.append(dict(version=version, channel=key[0], smoke=key[1], floor=key[2], cells=len(group),
                minimum=values.min(), p10=values.quantile(.1), mean=values.mean(), median=values.median(),
                p90=values.quantile(.9), maximum=values.max(), positive_fraction=float((values>0).mean())))
        for channel, group in frame.groupby('guidance', sort=False):
            top = group.relative_information_utility >= group.relative_information_utility.quantile(.9)
            for key, stage in group.groupby(['floor', 'stage'], sort=False):
                values = stage.relative_information_utility
                stages.append(dict(version=version, channel=channel, floor=key[0], stage=key[1], cells=len(stage),
                    mean=values.mean(), positive_fraction=float((values>0).mean()),
                    share_of_global_top10=float(top.loc[stage.index].sum()/top.sum())))
            paired = group[group.smoke.eq('HighSmoke')].merge(group[group.smoke.eq('LowSmoke')],
                on=['floor', 'x', 'z'], suffixes=('_high', '_low'), validate='one_to_one')
            paired['delta'] = paired.relative_information_utility_high-paired.relative_information_utility_low
            for floor, part in paired.groupby('floor'):
                contrasts.append(dict(version=version, channel=channel, floor=floor, cells=len(part),
                    mean_high_minus_low=part.delta.mean(), median_high_minus_low=part.delta.median(),
                    fraction_higher_in_high_smoke=float((part.delta>0).mean())))
            for feature in SPATIAL:
                correlations.append(dict(version=version, channel=channel, feature=feature,
                    grid_spearman=float(spearmanr(group[feature],group.relative_information_utility).statistic),
                    meaning='Global grid association; correlated geometry/context; not causal feature importance'))
    comparison = []
    for channel, group in merged.groupby('guidance', sort=False):
        comparison.append(dict(channel=channel, smoke='ALL', floor='ALL', **compare_vectors(group)))
        for (smoke, floor), part in group.groupby(['smoke', 'floor']):
            comparison.append(dict(channel=channel, smoke=smoke, floor=floor, **compare_vectors(part)))
    output.mkdir(parents=True)
    tables = {'common_scope_metrics': metrics, 'common_scope_macro': common_macro,
              'original_scope_macro': pd.DataFrame(official_rows), 'condition_metrics': pd.DataFrame(condition_rows),
              'raw_field_statistics': pd.DataFrame(stats), 'stage_statistics': pd.DataFrame(stages),
              'smoke_contrasts': pd.DataFrame(contrasts), 'field_comparison': pd.DataFrame(comparison),
              'direct_smoke_offsets': pd.DataFrame(context_rows), 'shared_channel_decomposition': pd.DataFrame(components),
              'grid_feature_associations': pd.DataFrame(correlations)}
    for name, table in tables.items():
        table.to_csv(output/(name+'.csv'), index=False, encoding='utf-8-sig')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9})
    for channel in tqdm(GUIDANCE, desc='Render raw model comparisons'):
        plot_comparison(fields, channel, output/(channel+'_raw_model_comparison.png'), dpi)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    try:
        test = metrics[metrics.split.eq('test')]
        for ax, metric, label in zip(axes, ['policy_auc', 'spatial_auc'], ['Policy AUC', 'Spatial reward AUC']):
            for i, channel in enumerate(GUIDANCE):
                values = [test[(test.version.eq(v)) & test.channel.eq(channel)][metric].iloc[0] for v in ['V6', 'V6_1']]
                ax.plot(values, [i, i], color='gray', linewidth=1)
                ax.scatter(values[0], i, color='#31688e', label='V6' if i == 0 else None)
                ax.scatter(values[1], i, color='#b35806', marker='s', label='V6_1' if i == 0 else None)
            ax.set_yticks(range(4), GUIDANCE); ax.invert_yaxis(); ax.set_xlim(.45, .95)
            ax.axvline(.5, color='gray', linestyle=':', linewidth=.8); ax.set_xlabel(label); ax.legend()
        fig.suptitle('Test comparison | Same StaticSign baseline and new-channel evaluation scope')
        fig.tight_layout(); fig.savefig(output/'common_scope_test_metrics.png', dpi=dpi)
    finally:
        plt.close(fig)
    for version, detail in inputs.items():
        best = Path(detail['best_model'])
        if any(sha(best/name) != checksum for name, checksum in detail['sha256'].items()):
            raise RuntimeError('Source model artifacts changed during analysis')
    manifest = dict(status='complete', inputs=inputs, cell_m=CELL, grid_rows_per_model=len(merged),
        scope='Common evaluation uses StaticSign/StaticSign plus original new-channel own+Full subsets',
        computation='Saved evaluations and forward-only fitted-rule queries; no fit or optimization',
        smoothing=False, interpolation=False, participant_splits_equal=True, raw_grid_features_equal=True,
        statistics='Grid cells are descriptive geometry samples, not independent participants',
        dpi=dpi, files=[p.name for p in output.iterdir()])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMMON_TEST',common_macro[common_macro.split.eq('test')].to_json(orient='records'))
    print('FIELD_COMPARISON',tables['field_comparison'].query("smoke == 'ALL'").to_json(orient='records'))
    print('SMOKE_CONTRASTS',tables['smoke_contrasts'].to_json(orient='records'))
    print('DIRECT_SMOKE',tables['direct_smoke_offsets'].to_json(orient='records'))
    print('OUTPUT='+str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v6', type=Path, default=DEFAULTS['V6'])
    parser.add_argument('--v6-1', type=Path, default=DEFAULTS['V6_1'])
    parser.add_argument('--output', type=Path, default=HERE/'Results_Compare_Best_Models_V6_V6_1')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    run_analysis({'V6': args.v6, 'V6_1': args.v6_1}, args.output, args.dpi)


if __name__ == '__main__':
    main()
