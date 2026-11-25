"""Condition-only sustained AOI figures, with matched raw-hit sensitivity data.

The default CLI writes a new independent result directory; historical outputs
and the legacy functions below are preserved. See Guidance_Sustained_AOI_V6
for the validated cohort, common event/denominator contract and spatial audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

import Visualize_Guidance_Channels_V6 as maps
import Visualize_Guidance_Fixation_Proportion_Violin as violin
from Guidance_Channel_Contract import (
    AOI_ROOT, AOI_SCHEMA, FIELDS, ROOT, aoi_path, as_bool, channel_actions,
    digest, static_arrays, timestamp_ns,
)
from common import GUIDANCE_SMOKE_COLORS, load_trials

ORDER = ['StaticSign', 'Arrow', 'MiniMap', 'TextBar', 'Multimodal']
LABELS = {'StaticSign': 'StaticSign', **violin.GUIDANCE_NAME_MAP}
METRIC = 'GuideDwellProportion'
AR_CHANNELS = ('ArrowRibbon', 'MiniMap', 'TextBar')
MAP_NAME = '06_Multimodal_Observed_Acquisition.png'
VIOLIN_NAME = 'Guidance_Dwell_Proportion_Violin_With_StaticSign_V6.png'


def collect(aoi_root, zones):
    manifest = json.loads((aoi_root / 'manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema') != AOI_SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('A complete V6 AOI manifest is required')
    identities = [(t['scenario'], t['participant'], t['source_file']) for t in manifest['trials']]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate V6 trial identities')
    expected = dict(zip(identities, manifest['trials']))
    roster = load_trials()
    if set(expected) != set(map(tuple, roster[['Scenario', 'ParticipantID', 'FileName']].to_numpy())):
        raise ValueError('V6 manifest and canonical trial roster differ')
    records, points, audits = [], [], []
    for number, row in enumerate(roster.itertuples(index=False), 1):
        path = aoi_path(row, aoi_root)
        checksum = digest(path)
        entry = expected[row.Scenario, row.ParticipantID, row.FileName]
        if checksum != entry['sha256']:
            raise ValueError('V6 AOI checksum differs: ' + str(path))
        frame = pd.read_csv(path, low_memory=False)
        ns = timestamp_ns(frame.TimeStamp)
        if len(ns) < 2 or len(frame) != entry['rows']:
            raise ValueError('Invalid V6 row count: ' + str(path))
        weights = np.minimum(np.r_[np.diff(ns), np.median(np.diff(ns))] / 1e9, .1)
        channels = AR_CHANNELS if row.Guidance == 'Full' else (row.Guidance,)
        actions = [channel_actions(frame, row.Guidance, channel) for channel in channels]
        valid = np.logical_or.reduce([action >= 0 for action in actions])
        hit = np.logical_or.reduce([action == 1 for action in actions]) & valid
        if row.Guidance == 'StaticSign':
            # Independently check V6 labels against the logged name/coordinate rule.
            direct_valid, direct_hit = static_arrays(frame)
            if not np.array_equal(valid, direct_valid) or not np.array_equal(hit, direct_hit):
                raise ValueError('StaticSign extraction differs from V6 Hit1 rule')
        denom, numer = float(weights[valid].sum()), float(weights[hit].sum())
        if denom <= 0 or not 0 <= numer <= denom:
            raise ValueError('Invalid AOI dwell duration: ' + str(path))
        records.append(dict(Scenario=row.Scenario, ParticipantID=row.ParticipantID,
            FileName=row.FileName, Guidance=LABELS[row.Guidance], SourceGuidance=row.Guidance,
            Smoke=row.Smoke, ValidDuration_s=denom, GuideDwellDuration_s=numer,
            GuideDwellProportion=numer / denom, ValidFrames=int(valid.sum()),
            HitFrames=int(hit.sum()), Channels='|'.join(channels),
            AOIFile=str(path.relative_to(aoi_root)), AOISHA256=checksum))
        if row.Guidance == 'Full':
            # Apply the existing V6 event filter to each AR channel first, then
            # take the Boolean union: overlaps never double-count a frame.
            retained = np.logical_or.reduce([maps.event_flags(ns, action == 1)[0] for action in actions])
            movement = pd.read_csv(row.SubjectFile, index_col=False,
                                   usecols=['TimeStamp', 'PosX', 'PosY', 'PosZ'])
            if not np.array_equal(ns, timestamp_ns(movement.TimeStamp)):
                raise ValueError('Subject/AOI frame alignment differs')
            xyz = movement[['PosX', 'PosY', 'PosZ']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
            floor = maps.legacy.assign_floor(xyz[:, 0], xyz[:, 1], xyz[:, 2], zones)
            keep = valid & np.isin(floor, maps.legacy.FLOOR_ORDER) & np.isfinite(xyz).all(1)
            points.append(pd.DataFrame(dict(ParticipantID=row.ParticipantID, Smoke=row.Smoke,
                Floor=floor[keep], PlotX=-xyz[keep, 0], PlotY=-xyz[keep, 2],
                ValidDuration_s=weights[keep], GazeDuration_s=weights[keep] * retained[keep])))
            for level in maps.legacy.FLOOR_ORDER:
                chosen = keep & (floor == level)
                audits.append(dict(Scenario=row.Scenario, ParticipantID=row.ParticipantID,
                    FileName=row.FileName, Floor=level, ValidFrames=int(chosen.sum()),
                    EventFrames=int((chosen & retained).sum()),
                    ValidDuration_s=float(weights[chosen].sum()),
                    GazeDuration_s=float(weights[chosen & retained].sum())))
        if number % 60 == 0:
            print(f'Validated V6 AOIs: {number}/{len(roster)}', flush=True)
    return pd.DataFrame(records), pd.concat(points, ignore_index=True), pd.DataFrame(audits)


def validate_previous_ar(frame):
    """Require the original four groups to retain their established values."""
    previous = pd.read_csv(violin.DEFAULT_INPUT)
    keys = ['Scenario', 'ParticipantID', 'FileName']
    ar = frame.loc[frame.SourceGuidance.ne('StaticSign')]
    check = ar.merge(previous[keys + [METRIC]], on=keys, how='outer',
                     suffixes=('_v6', '_old'), indicator=True, validate='one_to_one')
    if not check['_merge'].eq('both').all():
        raise ValueError('Previous and current AR trial cohorts differ')
    if not np.allclose(check[METRIC + '_v6'], check[METRIC + '_old'], atol=1e-12, rtol=0):
        raise ValueError('AR dwell values changed; review before replacing the metric')
    counts = frame.groupby(['Guidance', 'Smoke']).size()
    for guidance in ORDER:
        for smoke in violin.SMOKE_ORDER:
            expected = 45 if guidance in ('StaticSign', 'Multimodal') else 15
            if counts.get((guidance, smoke), 0) != expected:
                raise ValueError('Unexpected condition sample count')
    return float((check[METRIC + '_v6'] - check[METRIC + '_old']).abs().max())


def plot_violin(frame, output, dpi, *, metric=METRIC, title='Guidance Dwell Proportion',
                y_max=None, mean_fontsize=11, legend_fontsize=11, legend_loc='upper left',
                axis_fontsize=16.5, tick_fontsize=15, static_mean_clearance_pt=0):
    violin.configure_style()
    palette = {'StaticSign': GUIDANCE_SMOKE_COLORS['StaticSign'], **violin.PALETTE}
    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor='white')
    mean_labels = []
    for index, guidance in enumerate(ORDER):
        subset = frame.loc[frame.Guidance.eq(guidance)]
        for smoke, side, offset in [('LowSmoke', -1, -.12), ('HighSmoke', 1, .12)]:
            values = subset.loc[subset.Smoke.eq(smoke), metric].dropna().to_numpy(float)
            violin.draw_half_violin(ax, values, float(index), side, palette[guidance][smoke])
            violin.draw_mean_label(ax, values, float(index), side, fontsize=mean_fontsize)
            if len(values):
                actual = ax.texts[-1].get_text()
                if actual != f'{values.mean():.2f}':
                    raise ValueError('Violin mean label differs from its source data')
                mean_labels.append(dict(Guidance=guidance, Smoke=smoke, Metric=metric,
                                        n=len(values), mean=float(values.mean()), label=actual))
            if guidance == 'StaticSign' and len(values):
                # These small means need the box above its anchor, not across zero.
                label = ax.texts[-1]
                label.set_verticalalignment('bottom')
                label.set_transform(label.get_transform() + ScaledTranslation(
                    0, static_mean_clearance_pt / 72, fig.dpi_scale_trans))
            violin.draw_boxplot(ax, values, index + offset)
    ax.set_title(title, fontsize=21, fontweight='bold', pad=12)
    ax.set_xlabel('Experimental mode', fontsize=axis_fontsize, fontweight='bold', labelpad=9)
    ax.set_ylabel(title, fontsize=axis_fontsize, fontweight='bold', labelpad=9)
    ax.set_xlim(-.52, len(ORDER) - .48)
    upper=min(1, max(.80, float(frame[metric].max()) + .08)) if y_max is None else float(y_max)
    if not np.isfinite(upper) or upper<=0 or upper>1 or upper<float(frame[metric].max()):
        raise ValueError('Violin y_max must include all proportions and remain within (0, 1]')
    ax.set_ylim(0, upper)
    ax.set_xticks(range(len(ORDER)), ORDER, fontsize=tick_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize, width=1, length=5)
    ax.tick_params(axis='x', pad=6)
    ax.yaxis.grid(True, linestyle='--', linewidth=.6, alpha=.35)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(1.1)
    ax.legend(handles=[Patch(facecolor='#dadada', edgecolor='#555555', label='Low smoke (left)'),
                       Patch(facecolor='#666666', edgecolor='#555555', label='High smoke (right)')],
              loc=legend_loc, frameon=False, fontsize=legend_fontsize)
    fig.tight_layout()
    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontproperties().get_name() != 'Times New Roman':
            raise ValueError('Unexpected figure font')
    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white',
                pil_kwargs={'compress_level': 7})
    plt.close(fig)
    return mean_labels


def plot_multimodal(points, zones, audit, output, dpi, bandwidth):
    surface_dir = output / 'multimodal_surfaces'
    surface_dir.mkdir(parents=True, exist_ok=True)
    data, scales = {}, {}
    for smoke in maps.legacy.SMOKE_ORDER:
        finite_fields = []
        for floor in maps.legacy.FLOOR_ORDER:
            selected = points.loc[points.Smoke.eq(smoke) & points.Floor.eq(floor)]
            if selected.ParticipantID.nunique() != 45:
                raise ValueError('Multimodal floor coverage must include 45 participants')
            surface = maps.legacy.participant_balanced_surface(selected, zones, floor,
                maps.legacy.geometry_extent(zones, floor), bandwidth)
            data['Multimodal', smoke, floor] = surface
            np.savez_compressed(surface_dir / f'Multimodal_{smoke}_{floor}.npz', **surface)
            field = surface['field']
            values = field[np.isfinite(field) & (field > 0)]
            if len(values):
                finite_fields.append(values)
        if not finite_fields:
            raise ValueError('No supported Multimodal gaze cells')
        scales['Multimodal', smoke] = max(float(np.quantile(np.concatenate(finite_fields), .98)), 1e-6)
    # Keep the established map typography independent of the preceding violin.
    with plt.rc_context({'font.size': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10}):
        files = maps.plot_maps(data, scales, zones, output, dpi,
            figure_specs=[(['Multimodal'], MAP_NAME)],
            figure_title='Multimodal (Full) | observed AR guidance acquisition',
            scale_label='Within-Multimodal/smoke relative acquisition (P98 = 1)')
    audit.to_csv(output / 'multimodal_trial_events.csv', index=False, encoding='utf-8-sig')
    return files, {smoke: scales['Multimodal', smoke] for smoke in maps.legacy.SMOKE_ORDER}


def legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aoi-root', type=Path, default=AOI_ROOT)
    parser.add_argument('--map-output', type=Path, default=maps.DEFAULT_OUTPUT)
    parser.add_argument('--violin-output', type=Path, default=violin.RESULT_DIR)
    parser.add_argument('--map-dpi', type=int, default=300)
    parser.add_argument('--violin-dpi', type=int, default=600)
    parser.add_argument('--bandwidth-m', type=float, default=maps.legacy.DEFAULT_BANDWIDTH_M)
    args = parser.parse_args()
    if min(args.map_dpi, args.violin_dpi) < 50 or not np.isfinite(args.bandwidth_m) or args.bandwidth_m <= 0:
        parser.error('Invalid resolution/bandwidth')
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    zones = maps.legacy.load_zones()
    frame, points, audit = collect(args.aoi_root, zones)
    difference = validate_previous_ar(frame)
    for output in (args.map_output, args.violin_output):
        output.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(['Guidance', 'Smoke'])[METRIC].agg(n='size', mean='mean', median='median', sd='std')
    summary = summary.reindex(pd.MultiIndex.from_product([ORDER, violin.SMOKE_ORDER], names=['Guidance', 'Smoke']))
    frame.to_csv(args.violin_output / 'Guidance_Dwell_Proportion_V6_Plot_Data.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(args.violin_output / 'Guidance_Dwell_Proportion_V6_Summary.csv', encoding='utf-8-sig')
    plot_violin(frame, args.violin_output / VIOLIN_NAME, args.violin_dpi)
    files, scales = plot_multimodal(points, zones, audit, args.map_output, args.map_dpi, args.bandwidth_m)
    source_hashes = {Path(p).name: digest(p) for p in [__file__, maps.__file__, maps.legacy.__file__, violin.__file__,
        ROOT / 'Guidance_Channel_Contract.py', ROOT / '0_RL_SOURCE/Query_GuidanceGaze.py']}
    common = dict(status='complete', aoi_manifest_sha256=digest(args.aoi_root / 'manifest.json'),
        source_sha256=source_hashes, full_channels=list(AR_CHANNELS), static_in_full=False)
    violin_metadata = dict(common, file=VIOLIN_NAME, cohort_participants=45, cohort_trials=len(frame),
        metric='raw AOI-hit duration / channel-valid duration; forward intervals capped at 100 ms',
        event_filter=False, floor_filter=False, previous_ar_max_abs_difference=difference,
        static_rule='V6 valid Hit1 and exact ABCD_Signal name (including 1..13 suffixes)',
        order=ORDER, smoke_order=violin.SMOKE_ORDER, dpi=args.violin_dpi)
    map_metadata = dict(common, files=files, cohort_participants=45, cohort_trials=90,
        experimental_condition='Full', metric='union of retained AR channel events / union-valid AR time',
        event_rule='per-channel >=100 ms, at least two frames; gaps >100 ms break events; Boolean union afterward',
        floor_filter=list(maps.legacy.FLOOR_ORDER), bandwidth_m=args.bandwidth_m, dpi=args.map_dpi,
        aggregation='participant-balanced local gaze/valid ratios within Full only',
        scale=scales, display='existing V6 background, zone outlines, colormap and 2x2 panel renderer')
    (args.violin_output / 'Guidance_Dwell_Proportion_V6_Metadata.json').write_text(
        json.dumps(violin_metadata, ensure_ascii=False, indent=2), encoding='utf8')
    (args.map_output / 'multimodal_manifest.json').write_text(
        json.dumps(map_metadata, ensure_ascii=False, indent=2), encoding='utf8')
    print(summary.round(6).to_string())
    print('Saved map: ' + str(args.map_output / MAP_NAME))
    print('Saved violin: ' + str(args.violin_output / VIOLIN_NAME))


if __name__ == '__main__':
    from Guidance_Sustained_AOI_V6 import main
    main()
