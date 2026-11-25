"""Within-experimental-mode sustained AOI analysis and matched raw sensitivity.

Run through Visualize_Guidance_Condition_Figures_V6.py. The historical pooled
channel figures and caches are read-only; no output directory is overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image

from Guidance_Channel_Contract import (
    ROOT, AOI_ROOT, AOI_SCHEMA, CONDITIONS, CONDITION_CHANNELS, CONDITION_LABELS,
    aoi_path, condition_gaze_masks, digest, static_arrays, timestamp_ns,
)
from common import load_trials
import Visualize_Guidance_Channels_V6 as maps
from Visualize_Guidance_Condition_Figures_V6 import plot_violin, validate_previous_ar

sys.path.insert(0, str(ROOT / '0_RL_SOURCE'))
from Query_GuidanceGaze import event_flags, sample_weights_ns

ORDER = list(CONDITION_LABELS.values())
SMOKES = maps.legacy.SMOKE_ORDER
FLOORS = maps.legacy.FLOOR_ORDER
RAW = 'RawGuidanceGazeProportion'
SUSTAINED = 'SustainedGuidanceGazeProportion'
METRIC_LABEL = 'Sustained guidance-gaze proportion'
PRIMARY_VIOLIN_OPTIONS = dict(y_max=1., mean_fontsize=18, legend_fontsize=19,
    legend_loc='upper left', axis_fontsize=20, tick_fontsize=18, static_mean_clearance_pt=5.5)
COLOR_LABEL = 'Within-mode/smoke relative sustained AOI gaze (P98 = 1)'
DEFAULT_OUTPUT = ROOT / 'Results_Guidance_Sustained_AOI_V6'
KEYS = ['Scenario', 'ParticipantID', 'FileName']


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.15g')


def duration_record(ns, masks):
    """One denominator, two numerators; valid no-event trials stay at zero."""
    weights = sample_weights_ns(ns)
    total = int(weights[masks['valid']].sum())
    raw = int(weights[masks['raw']].sum())
    sustained = int(weights[masks['sustained']].sum())
    if not 0 <= sustained <= raw <= total:
        raise ValueError('Trial numerators must be nested inside the same denominator')
    return dict(ValidDuration_ns=total, RawGazeDuration_ns=raw,
                SustainedGazeDuration_ns=sustained, ValidDuration_s=total / 1e9,
                RawGazeDuration_s=raw / 1e9, SustainedGazeDuration_s=sustained / 1e9,
                RawGuidanceGazeProportion=raw / total if total else np.nan,
                SustainedGuidanceGazeProportion=sustained / total if total else np.nan,
                ObservationStatus=('no_valid_data' if not total else
                                   'valid_no_event' if not sustained else 'valid_with_event'))


def remap_manifest_path(value):
    """Relocate the original manifest's project root without changing contents."""
    normalized = value.replace('\\', '/')
    marker = 'Experiment_Data_A5/'
    if marker not in normalized:
        raise ValueError('Unrecognized manifest input path: ' + value)
    path = (ROOT / normalized.split(marker, 1)[1]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Manifest input escapes project root')
    return path


def collect(aoi_root, zones):
    manifest = json.loads((aoi_root / 'manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema') != AOI_SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('Complete V6 AOI manifest required')
    roster = load_trials()
    identities = [(t['scenario'], t['participant'], t['source_file']) for t in manifest['trials']]
    if len(identities) != len(set(identities)) or roster.duplicated(KEYS).any():
        raise ValueError('Duplicate trial identities')
    if set(identities) != set(map(tuple, roster[KEYS].to_numpy())):
        raise ValueError('V6 manifest and canonical roster differ')
    expected = dict(zip(identities, manifest['trials']))
    records, points, audits, sources, channel_audits = [], [], [], [], []
    for number, row in enumerate(roster.itertuples(index=False), 1):
        identity = {key: getattr(row, key) for key in KEYS}
        entry = expected[tuple(identity.values())]
        path = aoi_path(row, aoi_root)
        checksum = digest(path)
        if checksum != entry['sha256']:
            raise ValueError('AOI checksum differs: ' + str(path))
        # Check all actual acquisition inputs, including raw eye and movement
        # records, rather than accepting an old output checksum alone.
        input_paths = set()
        for old_path, old_hash in entry['input_hashes'].items():
            actual = remap_manifest_path(old_path)
            actual_hash = digest(actual)
            if actual_hash != old_hash:
                raise ValueError('V6 source changed; rebuild AOIs first: ' + str(actual))
            input_paths.add(actual)
            sources.append(dict(**identity, Role='AOI_input', File=str(actual.relative_to(ROOT)),
                                SHA256=actual_hash))
        if not {Path(row.SubjectFile).resolve(), Path(row.EyeGazeFile).resolve()} <= input_paths:
            raise ValueError('AOI provenance is missing the current raw trial inputs')
        sources.append(dict(**identity, Role='V6_AOI', File=str(path.resolve().relative_to(ROOT)),
                            SHA256=checksum))
        frame = pd.read_csv(path, low_memory=False)
        ns = timestamp_ns(frame.TimeStamp)
        if len(frame) != entry['rows'] or len(ns) < 2:
            raise ValueError('Invalid frame count: ' + str(path))
        for column, value in [('Condition', row.Scenario), ('GuidanceType', row.Guidance),
                              ('SmokeLevel', row.Smoke), ('ParticipantFolder', row.ParticipantID)]:
            if not frame[column].eq(value).all():
                raise ValueError('Within-file trial identity mismatch: ' + str(path))
        masks = condition_gaze_masks(frame, row.Guidance, ns, event_flags)
        if row.Guidance == 'StaticSign':
            valid, hit = static_arrays(frame)
            if not np.array_equal(valid, masks['valid']) or not np.array_equal(hit, masks['raw']):
                raise ValueError('StaticSign V6 extraction differs from exact Hit1 rule')
        weights_ns = sample_weights_ns(ns)
        weights = weights_ns / 1e9
        duration = duration_record(ns, masks)
        raw_sum = int((masks['raw_channels'] * weights_ns).sum())
        sustained_sum = int((masks['sustained_channels'] * weights_ns).sum())
        records.append(dict(**identity, Guidance=CONDITION_LABELS[row.Guidance],
            SourceGuidance=row.Guidance, Smoke=row.Smoke, Channels='|'.join(masks['channels']),
            **duration, ValidFrames=int(masks['valid'].sum()), RawHitFrames=int(masks['raw'].sum()),
            SustainedFrames=int(masks['sustained'].sum()),
            ChannelEventCount=sum(masks['channel_event_counts'].values()),
            RawChannelSumDuration_ns=raw_sum, SustainedChannelSumDuration_ns=sustained_sum,
            RawOverlapRemoved_ns=raw_sum - duration['RawGazeDuration_ns'],
            SustainedOverlapRemoved_ns=sustained_sum - duration['SustainedGazeDuration_ns'],
            AOIFile=str(path.relative_to(aoi_root)), AOISHA256=checksum))
        for channel, action_raw, action_sustained in zip(
                masks['channels'], masks['raw_channels'], masks['sustained_channels']):
            channel_audits.append(dict(**identity, Channel=channel,
                EventCount=masks['channel_event_counts'][channel],
                RawDuration_ns=int(weights_ns[action_raw].sum()),
                SustainedDuration_ns=int(weights_ns[action_sustained].sum())))
        movement = pd.read_csv(row.SubjectFile, usecols=['TimeStamp', 'PosX', 'PosY', 'PosZ'])
        if not np.array_equal(ns, timestamp_ns(movement.TimeStamp)):
            raise ValueError('AOI and participant-position timestamps differ')
        xyz = movement[['PosX', 'PosY', 'PosZ']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        floor = maps.legacy.assign_floor(xyz[:, 0], xyz[:, 1], xyz[:, 2], zones)
        finite = np.isfinite(xyz).all(1)
        spatial = masks['valid'] & finite & np.isin(floor, FLOORS)
        points.append(pd.DataFrame(dict(Guidance=CONDITION_LABELS[row.Guidance],
            SourceGuidance=row.Guidance, Smoke=row.Smoke, ParticipantID=row.ParticipantID,
            Floor=floor[spatial], PlotX=-xyz[spatial, 0], PlotY=-xyz[spatial, 2],
            ValidDuration_s=weights[spatial], RawGazeDuration_s=weights[spatial]*masks['raw'][spatial],
            GazeDuration_s=weights[spatial]*masks['sustained'][spatial])))
        # Spatial exclusions do not shorten the whole-trial denominator.
        for level in [*FLOORS, 'Unassigned_or_invalid_position']:
            keep = spatial & (floor == level) if level in FLOORS else masks['valid'] & ~spatial
            audits.append(dict(**identity, Guidance=CONDITION_LABELS[row.Guidance], Smoke=row.Smoke,
                Floor=level, ValidDuration_ns=int(weights_ns[keep].sum()),
                RawGazeDuration_ns=int(weights_ns[keep & masks['raw']].sum()),
                SustainedGazeDuration_ns=int(weights_ns[keep & masks['sustained']].sum()),
                ValidFrames=int(keep.sum())))
        if number % 30 == 0:
            print(f'Validated sources, events and participant positions: {number}/{len(roster)}', flush=True)
    return (pd.DataFrame(records), pd.concat(points, ignore_index=True), pd.DataFrame(audits),
            pd.DataFrame(sources), pd.DataFrame(channel_audits))


def validate_trials(frame, audit):
    if len(frame) != 270 or frame.ParticipantID.nunique() != 45 or frame.duplicated(KEYS).any():
        raise ValueError('Expected 45 participants and 270 unique trials')
    counts = frame.groupby(['Guidance', 'Smoke']).size()
    for mode in ORDER:
        for smoke in SMOKES:
            expected = 45 if mode in ('StaticSign', 'Multimodal') else 15
            subset = frame.loc[frame.Guidance.eq(mode) & frame.Smoke.eq(smoke)]
            if counts.get((mode, smoke), 0) != expected or subset.ParticipantID.nunique() != expected:
                raise ValueError(f'Unexpected participant/trial count: {mode}/{smoke}')
    for row in frame.itertuples(index=False):
        if row.Guidance != CONDITION_LABELS[row.SourceGuidance] or row.Channels != '|'.join(CONDITION_CHANNELS[row.SourceGuidance]):
            raise ValueError('Cross-mode channel contamination')
    good = frame.ValidDuration_ns.gt(0)
    if (frame.loc[good, SUSTAINED] > frame.loc[good, RAW] + 1e-15).any():
        raise ValueError('Filtered proportion exceeds matched raw proportion')
    if not frame.loc[~good, [RAW, SUSTAINED]].isna().all().all():
        raise ValueError('No valid observations must be missing, not zero')
    if not frame.loc[good & frame.SustainedGazeDuration_ns.eq(0), SUSTAINED].eq(0).all():
        raise ValueError('Valid trials without events must be zero')
    durations = ['ValidDuration_ns', 'RawGazeDuration_ns', 'SustainedGazeDuration_ns']
    totals = audit.groupby(KEYS)[durations].sum().reindex(pd.MultiIndex.from_frame(frame[KEYS]))
    np.testing.assert_array_equal(totals.to_numpy(), frame[durations].to_numpy())
    # A matched comparison to the established 180 AR raw trial values.
    previous_difference = validate_previous_ar(frame.assign(GuideDwellProportion=frame[RAW]))
    return dict(participants=45, trials=270, duplicate_trials=0,
        cross_mode_contamination=False, filtered_le_raw=True,
        no_valid_trials=int((~good).sum()),
        valid_no_sustained_event_trials=int((good & frame.SustainedGazeDuration_ns.eq(0)).sum()),
        previous_ar_raw_max_abs_difference=previous_difference,
        raw_overlap_removed_s=float(frame.RawOverlapRemoved_ns.sum()/1e9),
        sustained_overlap_removed_s=float(frame.SustainedOverlapRemoved_ns.sum()/1e9),
        spatial_time_reconciles_to_trial=True)


def comparison_tables(frame):
    records = []
    for smoke in [*SMOKES, 'AllSmoke']:
        for mode in ORDER:
            part = frame.loc[frame.Guidance.eq(mode)]
            if smoke != 'AllSmoke':
                part = part.loc[part.Smoke.eq(smoke)]
            raw, sustained = part[RAW], part[SUSTAINED]
            change = float(sustained.mean() - raw.mean())
            records.append(dict(Guidance=mode, Smoke=smoke, n_trials=len(part),
                n_participants=part.ParticipantID.nunique(), n_valid=int(raw.count()),
                n_no_valid=int(raw.isna().sum()),
                n_valid_no_event=int((part.ValidDuration_ns.gt(0) & part.SustainedGazeDuration_ns.eq(0)).sum()),
                raw_mean=raw.mean(), raw_median=raw.median(), raw_sd=raw.std(ddof=1),
                sustained_mean=sustained.mean(), sustained_median=sustained.median(),
                sustained_sd=sustained.std(ddof=1), mean_change=change,
                mean_change_percentage_points=change*100,
                relative_mean_reduction_pct=(-change/raw.mean()*100 if raw.mean() else np.nan),
                changed_trials=int((raw-sustained > 1e-12).sum())))
    summary = pd.DataFrame(records)
    for metric in ('raw', 'sustained'):
        summary[metric+'_rank'] = summary.groupby('Smoke')[metric+'_mean'].rank(ascending=False, method='min').astype('Int64')
    return summary


def compute_surfaces(points, zones, output):
    """Recompute every field; old condition caches are comparison inputs only."""
    primary, raw, primary_scales, raw_scales, records, cache_audit = {}, {}, {}, {}, [], []
    for mode in ORDER:
        for smoke in SMOKES:
            for floor in FLOORS:
                part = points.loc[points.Guidance.eq(mode) & points.Smoke.eq(smoke) & points.Floor.eq(floor)]
                if not part.SourceGuidance.eq(next(k for k,v in CONDITION_LABELS.items() if v == mode)).all():
                    raise ValueError('Spatial points include another experimental mode')
                expected_n = 45 if mode in ('StaticSign', 'Multimodal') else 15
                # Cohort completeness was checked at trial level. A participant
                # without valid AOI observations on this floor contributes no
                # supported local ratio, rather than an artificial zero.
                if part.ParticipantID.nunique() > expected_n:
                    raise ValueError(f'Unexpected spatial participants: {mode}/{smoke}/{floor}')
                extent = maps.legacy.geometry_extent(zones, floor)
                sustained = maps.legacy.participant_balanced_surface(part, zones, floor, extent, 2.)
                unfiltered = maps.legacy.participant_balanced_surface(
                    part.assign(GazeDuration_s=part.RawGazeDuration_s), zones, floor, extent, 2.)
                np.testing.assert_array_equal(sustained['support_n'], unfiltered['support_n'])
                if not np.array_equal(np.isfinite(sustained['field']), np.isfinite(unfiltered['field'])):
                    raise ValueError('Filtering changed spatial support/denominator')
                if np.any(sustained['field'] > unfiltered['field'] + 1e-12):
                    raise ValueError('Sustained local rate exceeds raw rate')
                primary[mode,smoke,floor], raw[mode,smoke,floor] = sustained, unfiltered
                name = f'{mode}_{smoke}_{floor}.npz'
                np.savez_compressed(output/'surfaces_sustained'/name, **sustained)
                np.savez_compressed(output/'sensitivity_raw'/'surfaces'/name, **unfiltered)
                field = sustained['field']
                records.append(dict(Guidance=mode, Smoke=smoke, Floor=floor,
                    participants=sustained['participant_n'], cohort_participants=expected_n,
                    participants_without_valid_floor_data=expected_n-sustained['participant_n'],
                    min_support=sustained['min_support'],
                    supported_cells=int(np.isfinite(field).sum()), supported_zero_cells=int((field == 0).sum()),
                    missing_cells=int(np.isnan(field).sum()),
                    ValidDuration_s=float(part.ValidDuration_s.sum()),
                    RawGazeDuration_s=float(part.RawGazeDuration_s.sum()),
                    SustainedGazeDuration_s=float(part.GazeDuration_s.sum())))
                source = next(k for k,v in CONDITION_LABELS.items() if v == mode)
                old_path = (maps.DEFAULT_OUTPUT/'multimodal_surfaces'/name if mode == 'Multimodal' else
                            maps.DEFAULT_OUTPUT/'condition_surfaces'/f'{source}_{source}_{smoke}_{floor}.npz')
                check = dict(Guidance=mode, Smoke=smoke, Floor=floor, reused=False,
                    decision='Recomputed from verified 270-trial inputs; old cache has no per-trial position provenance',
                    old_cache=str(old_path.relative_to(ROOT)), exists=old_path.exists())
                if old_path.exists():
                    with np.load(old_path) as old:
                        old_field = old['field']
                        same_shape = old_field.shape == field.shape
                        same_mask = same_shape and np.array_equal(np.isfinite(old_field), np.isfinite(field))
                        same_extent = np.allclose(old['extent'], extent, atol=1e-12, rtol=0)
                        delta = float(np.max(np.abs(old_field[np.isfinite(field)]-field[np.isfinite(field)]))) if same_mask else None
                        check.update(SHA256=digest(old_path), same_shape=same_shape,
                                     same_support_mask=same_mask, same_extent=same_extent,
                                     max_abs_difference=delta,
                                     numerically_equal=bool(same_mask and same_extent and delta <= 1e-12))
                cache_audit.append(check)
            for data, scales in [(primary, primary_scales), (raw, raw_scales)]:
                positive = [data[mode,smoke,floor]['field'] for floor in FLOORS]
                positive = np.concatenate([f[np.isfinite(f) & (f > 0)] for f in positive])
                scales[mode,smoke] = max(float(np.quantile(positive,.98)),1e-6) if len(positive) else 1.
            print(f'Recomputed sustained + raw surfaces: {mode}/{smoke}', flush=True)
    surface_summary = pd.DataFrame(records)
    surface_summary['sustained_positive_P98'] = [primary_scales[r.Guidance,r.Smoke] for r in surface_summary.itertuples()]
    surface_summary['raw_positive_P98'] = [raw_scales[r.Guidance,r.Smoke] for r in surface_summary.itertuples()]
    return primary, raw, primary_scales, raw_scales, surface_summary, cache_audit


def plot_overview(data, scales, zones, output, dpi, *, raw=False):
    """Ten paired maps in a 2-row x 5-column grid, preserving floor aspect."""
    maps.legacy.configure_style()
    bounds = {floor: maps.cropped_panel_bounds(zones,floor) for floor in FLOORS}
    ratios = {floor: (b[2]-b[0])/(b[3]-b[1]) for floor,b in bounds.items()}
    panel_height, floor_gap, group_gap, row_gap = 5.0, .075, .22, .50
    pair_width = panel_height*sum(ratios.values())+floor_gap
    left, right, bottom, top = .36,.14,.95,.68
    width = left+5*pair_width+4*group_gap+right
    height = bottom+2*panel_height+row_gap+top
    fig=plt.figure(figsize=(width,height))
    for r,smoke in enumerate(SMOKES):
        y = bottom+(1-r)*(panel_height+row_gap)
        for c,mode in enumerate(ORDER):
            x = left+c*(pair_width+group_gap)
            fig.text((x+pair_width/2)/width,(y+panel_height+.12)/height,
                     f'{mode} | {"Low smoke" if r == 0 else "High smoke"}',
                     ha='center',va='bottom',fontsize=12,fontweight='bold')
            for floor in FLOORS:
                panel_width = panel_height*ratios[floor]
                ax=fig.add_axes([x/width,y/height,panel_width/width,panel_height/height])
                maps.legacy.draw_panel(ax,zones,data[mode,smoke,floor],scales[mode,smoke],floor,None,zero_colored=True)
                b=bounds[floor]
                ax.set_xlim(b[0],b[2]); ax.set_ylim(b[1],b[3]); ax.set_axis_off()
                x += panel_width+floor_gap
    title='Raw guidance-AOI gaze: sensitivity check' if raw else 'Sustained guidance-AOI gaze: within experimental modes'
    fig.text(.5,1-.12/height,title,ha='center',va='top',fontsize=18)
    cax=fig.add_axes([.30,.64/height,.40,.12/height])
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=maps.legacy.HEAT_CMAP),cax=cax,orientation='horizontal')
    cbar.set_label('Within-mode/smoke relative value (positive P98 = 1)',fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.text(.5,.04/height,'Hall left; Platform right. Relative colours cannot compare absolute proportions across conditions.',
             ha='center',va='bottom',fontsize=10)
    fig.savefig(output,dpi=dpi,facecolor='white')
    plt.close(fig)


def protected_files():
    """Hash old plot/data artifacts so the new workflow proves preservation."""
    roots=[maps.DEFAULT_OUTPUT, ROOT/'Results_Guidance_Fixation_Proportion_Violin']
    return {str(p.relative_to(ROOT)):digest(p) for root in roots for p in root.rglob('*') if p.is_file()}


def run(args):
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing results are never overwritten: '+str(output))
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('Output must be a new subdirectory of the project')
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    preserved=protected_files()
    zones=maps.legacy.load_zones()
    frame,points,audit,sources,channel_audit=collect(args.aoi_root,zones)
    validation=validate_trials(frame,audit)
    summary=comparison_tables(frame)
    output.mkdir()
    for folder in ['surfaces_sustained','sensitivity_raw/surfaces']:
        (output/folder).mkdir(parents=True)
    save_csv(frame,output/'Trial_Proportions_Raw_vs_Sustained.csv')
    save_csv(summary,output/'Summary_Raw_vs_Sustained.csv')
    save_csv(audit,output/'Trial_Floor_Time_Audit.csv')
    save_csv(sources,output/'Input_Source_Checksums.csv')
    save_csv(channel_audit,output/'Trial_Channel_Event_Audit.csv')
    primary,raw,scales,raw_scales,spatial_summary,cache_audit=compute_surfaces(points,zones,output)
    del points
    save_csv(spatial_summary,output/'Spatial_Surface_Summary.csv')
    (output/'Cache_Audit.json').write_text(json.dumps(cache_audit,ensure_ascii=False,indent=2),encoding='utf8')
    labels=plot_violin(frame,output/'Sustained_Guidance_Gaze_Proportion_Violin.png',args.violin_dpi,
                       metric=SUSTAINED,title=METRIC_LABEL,**PRIMARY_VIOLIN_OPTIONS)
    labels+=plot_violin(frame,output/'sensitivity_raw'/'Raw_Guidance_Gaze_Proportion_Violin.png',args.violin_dpi,
                        metric=RAW,title='Raw guidance-gaze proportion')
    label_frame=pd.DataFrame(labels)
    for row in label_frame.itertuples():
        expected=summary.loc[summary.Guidance.eq(row.Guidance)&summary.Smoke.eq(row.Smoke)].iloc[0]
        mean=expected.sustained_mean if row.Metric==SUSTAINED else expected.raw_mean
        if row.label != f'{mean:.2f}' or row.n != expected.n_valid:
            raise ValueError('Figure mean labels disagree with summary table')
    save_csv(label_frame,output/'Violin_Mean_Labels.csv')
    files,layout=[],{}
    with plt.rc_context({'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10}):
        for mode in ORDER:
            for smoke in SMOKES:
                name,positions=maps.plot_smoke_pair(primary,scales,zones,output,mode,smoke,args.map_dpi,True)
                files.append(name); layout[name]=positions
                # Matched raw maps are separated from the primary figures.
                maps.plot_smoke_pair(raw,raw_scales,zones,output/'sensitivity_raw',mode,smoke,args.map_dpi,True)
            print('Rendered primary and sensitivity maps: '+mode,flush=True)
        plot_overview(primary,scales,zones,output/'All_Modes_Sustained_Hall_Platform.png',args.map_dpi)
        plot_overview(raw,raw_scales,zones,output/'sensitivity_raw'/'All_Modes_Raw_Hall_Platform.png',args.map_dpi,raw=True)
        for orientation in ['horizontal','vertical']:
            maps.plot_shared_colorbar(output,args.map_dpi,orientation,label='Within-mode/smoke relative AOI gaze (P98 = 1)')
    image_info=[]
    for path in output.rglob('*.png'):
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            image_info.append(dict(file=str(path.relative_to(output)),width_px=im.width,height_px=im.height))
    sizes={(v['width_px'],v['height_px']) for v in image_info if v['file'] in files}
    if len(files)!=10 or len(sizes)!=1:
        raise ValueError('Expected ten identically sized primary paired maps')
    if protected_files()!=preserved:
        raise ValueError('Historical outputs changed during independent analysis')
    validation.update(historical_files_unchanged=len(preserved),primary_paired_maps=10,
        spatial_filtered_le_raw=True,spatial_raw_sustained_support_identical=True,
        mean_labels_match_summary=True,all_png_files_readable=True,
        old_condition_caches_numerically_equal=sum(c.get('numerically_equal',False) for c in cache_audit),
        no_valid_floor_observation_trials=audit.loc[audit.Floor.isin(FLOORS)&audit.ValidDuration_ns.eq(0),
            KEYS+['Guidance','Smoke','Floor']].to_dict('records'))
    metadata=dict(schema='a5_condition_sustained_aoi_v6_1',status='complete',
        created_at=datetime.now(timezone.utc).isoformat(),
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        metric_name=METRIC_LABEL,interpretation='持续AOI注视; not standard fixation identification or proof of comprehension',
        condition_channels=CONDITION_CHANNELS,condition_labels=CONDITION_LABELS,
        cohort_source='common.load_trials(); established exclusions, unchanged',validation=validation,
        sample_counts=summary.loc[summary.Smoke.ne('AllSmoke'),
            ['Guidance','Smoke','n_trials','n_participants','n_valid','n_no_valid','n_valid_no_event']].to_dict('records'),
        source_aoi_root=str(args.aoi_root.resolve()),aoi_schema=AOI_SCHEMA,
        source_manifest_sha256=digest(args.aoi_root/'manifest.json'),
        source_inputs='Input_Source_Checksums.csv; every AOI output and its acquisition inputs verified against V6 manifest',
        event=dict(function='0_RL_SOURCE/Query_GuidanceGaze.py:event_flags',threshold_ms=100,min_samples=2,
            breaks='miss, channel-invalid sample, or inter-sample interval >100 ms',
            weights='forward interval capped at 100 ms; final sample rounded median interval (integer nanoseconds)',
            qualified_duration='entire run, including its first 100 ms',
            multimodal='per-channel event filter, then Boolean OR of AR3; StaticSign excluded'),
        denominator=dict(single_mode='all valid observation time for the matching channel',
            multimodal='Boolean OR of ArrowRibbon, MiniMap, TextBar valid observations',
            filtering='same denominator for raw and sustained; only numerator filtered',
            no_events='zero if denominator >0; missing only if no valid duration',
            whole_trial='includes valid time outside spatial floor/position support'),
        spatial=dict(position='participant Subject PosX/PosY/PosZ at matching AOI timestamps',
            transform='PlotX=-PosX; PlotY=-PosZ; floor from existing height classifier',
            floors=FLOORS,bandwidth_m=2.,grid_width=maps.legacy.GRID_WIDTH,
            kernel='Gaussian scipy.ndimage; constant boundary; sigma=max(2 m/cell size,1 cell)',
            per_participant='smooth gaze and valid histograms separately, then divide',
            aggregation='equal mean of supported participants within the experimental mode only',
            support='valid_smooth > max(1% participant maximum,1e-7); >=max(2,ceil(0.08*N)) participants',
            footprint='existing floor footprint buffered by 0.15 m',
            normalization='positive-rate P98 pooled across Hall+Platform for each mode/smoke independently',
            scale={m+'/'+s:value for (m,s),value in scales.items()},
            raw_sensitivity_scale={m+'/'+s:value for (m,s),value in raw_scales.items()},
            colour='relative within mode/smoke; clip at P98; not absolute cross-condition comparability',
            supported_zero='blue; constant supported alpha=0.92',unsupported='NaN and transparent; never filled zero',
            background_calibration=maps.legacy.BACKGROUND_CALIBRATION,
            cache_policy='all fields recomputed; previous condition caches compared in Cache_Audit.json'),
        violin=dict(unit='one ratio per trial',smoke_sides={'LowSmoke':'left','HighSmoke':'right'},
            primary_style=dict(y_limits=[0,1], mean_font_pt=PRIMARY_VIOLIN_OPTIONS['mean_fontsize'],
                smoke_legend_font_pt=PRIMARY_VIOLIN_OPTIONS['legend_fontsize'],
                axis_label_font_pt=PRIMARY_VIOLIN_OPTIONS['axis_fontsize'],
                tick_font_pt=PRIMARY_VIOLIN_OPTIONS['tick_fontsize'], title_font_pt=21,
                smoke_legend_location=PRIMARY_VIOLIN_OPTIONS['legend_loc'], inset_space='upper area'),
            mean='arithmetic mean of trial proportions; labels rounded to 2 decimals',
            box='median and quartiles, whiskers 1.5 IQR, outliers not shown',
            significance='not tested; rankings and changes are descriptive'),
        rendering=dict(font='Times New Roman',colormap=maps.legacy.HEAT_CMAP_NUMBER,
            map_dpi=args.map_dpi,violin_dpi=args.violin_dpi,layout=layout,images=image_info),
        source_sha256={str(Path(p).relative_to(ROOT)):digest(p) for p in [
            __file__,ROOT/'Visualize_Guidance_Condition_Figures_V6.py',maps.__file__,maps.legacy.__file__,
            ROOT/'Guidance_Channel_Contract.py',ROOT/'0_RL_SOURCE/Query_GuidanceGaze.py',ROOT/'common.py',
            ROOT/'Visualize_Guidance_Fixation_Proportion_Violin.py',
            ROOT/'Results_Guidance_Spatial_Causal_Analysis/data/ar_information_acquisition_features.csv',
            ROOT/'UnifiedZoneQuery.py',ROOT/'a5_colormaps.py',maps.legacy.ZONES_FILE,
            maps.legacy.ZONES_FILE.with_suffix('.json'),
            *[maps.legacy.BACKGROUND_DIR/(f+'.png') for f in FLOORS]]})
    (output/'Metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 持续引导 AOI 注视：实验模式内部分析','',
        '主要指标：Sustained guidance-gaze proportion。≥100 ms 且至少两个样本的完整事件时长 / 全部通道有效观测时长。',
        '该指标描述持续 AOI 注视，不等同于标准 fixation 识别或信息理解。',
        'StaticSign、Arrow、MiniMap、TextBar 仅取相同实验模式；Multimodal 仅取 Full 的三个 AR 通道，分别筛选后布尔并集。',
        '低烟在左、高烟在右；小提琴与均值按 trial 比例计算。有效但无事件记 0，无有效观测记缺失。','',
        '热图在参与者位置累计时长，按站厅与站台分别做 2 m 高斯平滑分子/分母，得到个人局部比例后在模式内部等权平均。',
        '热图与整个 trial 比例统计尺度不同，不应要求数值相等。蓝色包括有支持的零值，无支持透明。',
        '每模式—烟雾条件单独采用站厅/站台共同的正值 P98；颜色表示图内相对值，不能直接跨条件比较绝对比例。','',
        '主目录：10 张配对热图、总览、共享横/竖色条、持续注视小提琴图、逐 trial 数据与汇总比较表。',
        'sensitivity_raw/：同一批 trial、同一分母下未筛选的热图、小提琴图与数值表面。',
        'Metadata.json 记录方法、输入和程序校验值、样本与验证结果；Cache_Audit.json 记录旧缓存对照。','',
        f'验证：{validation["participants"]} 人 / {validation["trials"]} trial；旧结果 {len(preserved)} 个文件未修改。',
        '每烟雾条件 StaticSign/Multimodal 各 45 trial；Arrow/MiniMap/TextBar 各 15 trial。',
        *[f'{r["Guidance"]} / {r["Smoke"]} / {r["Floor"]} 有一个 trial 无有效空间观测，'
          '该 trial 保留在全程比例数据中，空间表面不将缺测补零。'
          for r in validation['no_valid_floor_observation_trials']],
        '汇总表 mean_change_percentage_points 为筛选后减筛选前的百分点差；relative_mean_reduction_pct 为相对均值降幅。',
        'raw_rank / sustained_rank 在相同烟雾条件内按均值从高到低排序，AllSmoke 为两种烟雾的 trial 合并分布。',
        '未进行显著性检验，差值与排序仅作描述。','',
        '重新运行：','```powershell',
        '& "C:\\Users\\11274\\.conda\\envs\\yoyo\\python.exe" Visualize_Guidance_Condition_Figures_V6.py --output 新的独立目录',
        '```']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(summary.round(6).to_string(index=False),flush=True)
    print('Saved independent results: '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aoi-root',type=Path,default=AOI_ROOT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--map-dpi',type=int,default=300)
    parser.add_argument('--violin-dpi',type=int,default=600)
    args=parser.parse_args()
    if min(args.map_dpi,args.violin_dpi)<50:
        parser.error('Resolution must be at least 50 dpi')
    run(args)


if __name__=='__main__':
    main()
