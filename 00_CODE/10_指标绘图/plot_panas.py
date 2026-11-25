"""Plot paired pre/post PA and NA, using the canonical common.load_trials cohort."""
from pathlib import Path
import argparse
import importlib.util
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm

STYLE = {'font.family': 'Times New Roman', 'font.size': 16.5,
         'axes.linewidth': 1.2, 'axes.labelsize': 16.5, 'axes.titlesize': 19.5,
         'xtick.labelsize': 15, 'ytick.labelsize': 15,
         'pdf.fonttype': 42, 'ps.fonttype': 42}
PALETTE = {'PA': {'Pre': '#D9D9D9', 'Post': '#666666'},
           'NA': {'Pre': '#D9D9D9', 'Post': '#666666'}}
SCORES = ['PA_Pre', 'NA_Pre', 'PA_Post', 'NA_Post']


def load_paired(root):
    sys.path.insert(0, str(root))
    from common import load_trials, PROJECT_ROOT
    if Path(PROJECT_ROOT).resolve() != root.resolve():
        raise RuntimeError('Incorrect common.py imported')
    cohort = load_trials()
    folder = root / 'INDICATORS' / 'questionnaire_emotion'
    source = folder / 'PANAS_participant_summary.csv'
    summary = pd.read_csv(source)
    if summary.ParticipantID.duplicated().any():
        raise ValueError('Duplicate participant in summary')
    if set(summary.ParticipantID) != set(cohort.ParticipantID):
        raise ValueError('Summary participant IDs do not match common.load_trials()')
    expected = cohort.groupby('ParticipantID').size()
    if not summary.IncludedTrialCount.eq(summary.ParticipantID.map(expected)).all():
        raise ValueError('Summary trial counts differ from common.load_trials()')
    if not summary.MeasurementLevel.eq('participant_experiment_pre_post').all():
        raise ValueError('Unexpected measurement level')
    summary[SCORES] = summary[SCORES].apply(pd.to_numeric, errors='raise')
    paired = summary.dropna(subset=SCORES).sort_values('ParticipantID').copy()
    if paired.empty:
        raise ValueError('No complete pre/post pairs')
    if not paired[SCORES].isin(range(9, 46)).all().all():
        raise ValueError('Invalid nine-item PANAS sum')
    # Reuse the established item definitions and independently verify source sums.
    spec = importlib.util.spec_from_file_location('panas_extract', folder / 'Extract_PANAS_Summary.py')
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    books = {phase: pd.read_excel(folder / filename, sheet_name='Sheet1')
             for phase, filename in [('Pre', 'emotion_before.xlsx'), ('Post', 'emotion_after.xlsx')]}
    for _, row in tqdm(paired.iterrows(), total=len(paired), desc='Verify paired PANAS'):
        for phase in ['Pre', 'Post']:
            raw = books[phase].iloc[int(row[f'SourceExcelRow_{phase}']) - 2]
            for affect, items in [('PA', extractor.PA_ITEMS), ('NA', extractor.NA_ITEMS)]:
                values = pd.to_numeric(raw[items], errors='raise')
                if not values.isin([1, 2, 3, 4, 5]).all() or values.sum() != row[f'{affect}_{phase}']:
                    raise ValueError(f'Source score mismatch: {row.ParticipantID}/{phase}/{affect}')
    return paired, source


def draw_half_violin(ax, values, center, side, color):
    """Match scan_speed.py geometry; KDE affects only the displayed outline."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite score in paired plot')
    low, high = values.min(), values.max()
    if low == high:
        y = np.linspace(max(9, low - .1), min(45, high + .1), 50)
        density = np.ones_like(y)
    else:
        padding = .03 * (high - low)
        # Never depict PANAS sums outside the possible 9--45 range.
        y = np.linspace(max(9, low - padding), min(45, high + padding), 300)
        density = gaussian_kde(values)(y)
    width = .38 * density / density.max()
    ax.fill_betweenx(y, center, center + side * width,
                     facecolor=color, edgecolor='#555555', linewidth=1.1,
                     alpha=.92, zorder=1)


def plot(paired, output, dpi):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5.25, 5.6), facecolor='white')
    stats = []
    for center, affect in enumerate(['PA', 'NA']):
        for phase, side in [('Pre', -1), ('Post', 1)]:
            values = paired[f'{affect}_{phase}'].to_numpy(dtype=float)
            draw_half_violin(ax, values, center, side, PALETTE[affect][phase])
            ax.boxplot(values, positions=[center + side * .12], widths=.16,
                       patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.1},
                       whiskerprops={'color': 'black', 'linewidth': 1.0},
                       capprops={'color': 'black', 'linewidth': 1.0},
                       medianprops={'color': 'black', 'linewidth': 1.4}, zorder=4)
            mean = values.mean()
            ax.text(center + side * .22, mean, f'{mean:.2f}', ha='center', va='center',
                    fontsize=11, color='black', zorder=8,
                    bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white',
                          'edgecolor': '#555555', 'linewidth': .9, 'alpha': 1})
            stats.append({'Affect': affect, 'Phase': phase, 'N': len(values),
                          'Mean': mean, 'SD': values.std(ddof=1) if len(values)>1 else np.nan,
                          'Median': np.median(values), 'Min': values.min(), 'Max': values.max()})
    ax.set_title('Positive and Negative Affect', fontsize=21, fontweight='bold', pad=12)
    ax.set_xlabel('Emotion Dimension', fontsize=16.5, fontweight='bold', labelpad=9)
    ax.set_ylabel('PANAS Score (9-item sum)', fontsize=16.5, fontweight='bold', labelpad=9)
    ax.set_xticks([0, 1], ['Positive Affect (PA)', 'Negative Affect (NA)'])
    ax.set_xlim(-.6, 1.6)
    upper = max(30, np.ceil((paired[SCORES].to_numpy().max()+8)/5)*5)
    ax.set_ylim(7, upper)
    ax.tick_params(axis='x', labelsize=15, width=1, length=5, pad=6)
    ax.tick_params(axis='y', labelsize=15, width=1, length=5)
    ax.yaxis.grid(True, linestyle='--', linewidth=.6, alpha=.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(1.1)
    fig.tight_layout()
    assert len(fig.axes) == 1
    fig.savefig(output/'PA_NA_paired.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(output/'PA_NA_paired.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return pd.DataFrame(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--force', action='store_true', help='Replace this script\'s existing outputs')
    args = parser.parse_args()
    names = ['PA_NA_paired.png', 'PA_NA_paired.pdf', 'PA_NA_paired_data.csv', 'PA_NA_descriptive.csv']
    if not args.force and any((args.output/n).exists() for n in names):
        raise FileExistsError('Existing plot outputs: use --force to regenerate')
    paired, source = load_paired(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    stats = plot(paired, args.output, args.dpi)
    paired.to_csv(args.output/'PA_NA_paired_data.csv', index=False, encoding='utf-8-sig')
    stats.to_csv(args.output/'PA_NA_descriptive.csv', index=False, encoding='utf-8-sig')
    print(f'Source: {source}\nPaired participants: {len(paired)}')
    print(stats.to_string(index=False))


if __name__ == '__main__':
    main()
