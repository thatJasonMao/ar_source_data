"""Four-channel descriptive acquisition maps, retaining condition-specific data.

Uses the existing 2 m Gaussian AOI/valid-time estimator and backgrounds. First
average participants within each condition, then supported conditions equally.
These observed >=100 ms AOI maps are not trained IRL reward fields.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
import Visualize_Guidance_Information_Acquisition_2D_Heatmaps as legacy
from Guidance_Channel_Contract import (ROOT, AOI_ROOT, CHANNELS, CONDITIONS, AVAILABLE,
    FIELDS, AOI_SCHEMA, aoi_path, timestamp_ns, channel_actions, digest)
from common import load_trials
sys.path.insert(0, str(ROOT / '0_RL_SOURCE'))
from Query_GuidanceGaze import event_flags

DEFAULT_OUTPUT = legacy.RESULT_DIR / 'V6_Channels'


def load_points(aoi_root, zones):
    m = json.loads((aoi_root / 'manifest.json').read_text(encoding='utf-8'))
    if m.get('schema') != AOI_SCHEMA or m.get('status') != 'complete':
        raise ValueError('Complete V6 AOIs first')
    checksums = {str((aoi_root / r['file']).resolve()): r['sha256'] for r in m['trials']}
    parts, audits = [], []
    trials = load_trials()
    for row in tqdm(trials.itertuples(), total=len(trials), desc='Map channel events to positions'):
        path = aoi_path(row, aoi_root)
        if digest(path) != checksums[str(path.resolve())]:
            raise ValueError('AOI checksum differs')
        aoi = pd.read_csv(path, low_memory=False)
        ns = timestamp_ns(aoi.TimeStamp)
        movement = pd.read_csv(row.SubjectFile, index_col=False, usecols=['TimeStamp', 'PosX', 'PosY', 'PosZ'])
        if not np.array_equal(ns, timestamp_ns(movement.TimeStamp)):
            raise ValueError('Frame alignment differs')
        xyz = movement[['PosX', 'PosY', 'PosZ']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        floor = legacy.assign_floor(xyz[:, 0], xyz[:, 1], xyz[:, 2], zones)
        weights = np.minimum(np.r_[np.diff(ns), np.median(np.diff(ns))] / 1e9, .1)
        for channel in AVAILABLE[row.Guidance]:
            action = channel_actions(aoi, row.Guidance, channel)
            sustained = event_flags(ns, action == 1)[0]
            keep = (action >= 0) & np.isin(floor, legacy.FLOOR_ORDER) & np.isfinite(xyz).all(1)
            parts.append(pd.DataFrame(dict(Condition=row.Guidance, Channel=channel, Smoke=row.Smoke,
                ParticipantID=row.ParticipantID, Floor=floor[keep], PlotX=-xyz[keep, 0], PlotY=-xyz[keep, 2],
                ValidDuration_s=weights[keep], GazeDuration_s=weights[keep] * sustained[keep])))
            audits.append(dict(Scenario=row.Scenario, ParticipantID=row.ParticipantID, FileName=row.FileName,
                Channel=channel, ValidDuration_s=float(weights[keep].sum()),
                RawHitFrames=int((action == 1).sum()), EventFrames=int(sustained.sum()),
                GazeDuration_s=float((weights[keep] * sustained[keep]).sum())))
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audits)


def surfaces(points, zones, bandwidth, output):
    result, scales, grids = {}, {}, []
    for channel in tqdm(CHANNELS, desc='Compute condition-balanced channel surfaces'):
        conditions = [e for e in CONDITIONS if channel in AVAILABLE[e]]
        for smoke in legacy.SMOKE_ORDER:
            fields = []
            for floor in legacy.FLOOR_ORDER:
                extent = legacy.geometry_extent(zones, floor)
                conditional = []
                subset = points.loc[points.Channel.eq(channel) & points.Smoke.eq(smoke) & points.Floor.eq(floor)]
                for condition in conditions:
                    part = subset.loc[subset.Condition.eq(condition)]
                    surface = legacy.participant_balanced_surface(part, zones, floor, extent, bandwidth)
                    conditional.append(surface)
                    name = f'{channel}_{condition}_{smoke}_{floor}.npz'
                    np.savez_compressed(output / 'condition_surfaces' / name, **surface)
                stack = np.stack([s['field'] for s in conditional])
                supported = np.isfinite(stack).sum(0)
                field = np.divide(np.nansum(stack, axis=0), supported,
                    out=np.full_like(stack[0], np.nan), where=supported > 0)
                # Alpha is average support fraction across the available conditions.
                fraction = np.mean([s['support_n'] / max(s['participant_n'], 1) for s in conditional], axis=0)
                surface = dict(field=field, extent=extent, support_n=fraction,
                    participant_n=1, condition_support_n=supported, condition_n=len(conditions))
                result[channel, smoke, floor] = surface
                np.savez_compressed(output / 'channel_surfaces' / f'{channel}_{smoke}_{floor}.npz', **surface)
                finite = field[np.isfinite(field) & (field > 0)]
                if len(finite):
                    fields.append(finite)
                grids.append(dict(channel=channel, smoke=smoke, floor=floor, conditions='|'.join(conditions),
                    participants=subset.ParticipantID.nunique(), supported_cells=int((supported > 0).sum()),
                    all_conditions_supported_cells=int((supported == len(conditions)).sum())))
            scales[channel, smoke] = max(float(np.quantile(np.concatenate(fields), .98)), 1e-6) if fields else 1.
    return result, scales, pd.DataFrame(grids)


def plot_maps(data, scales, zones, out, dpi, *, figure_specs=None,
              figure_title='Observed guidance acquisition | equal condition contribution',
              scale_label='Within-channel/smoke relative acquisition (P98 = 1)',
              zero_colored=False):
    legacy.configure_style()
    row_keys = [(smoke, floor) for smoke in legacy.SMOKE_ORDER for floor in legacy.FLOOR_ORDER]
    files = []
    if figure_specs is None:
        figure_specs = [(CHANNELS, '01_All_Channels_Hall_Platform.png')] + [([c], f'{i+2:02}_{c}_Observed_Acquisition.png') for i, c in enumerate(CHANNELS)]
    for channels, filename in figure_specs:
        if len(channels) == 1:
            fig, axes = plt.subplots(2, 2, figsize=(9, 12), squeeze=False)
            panels = [(axes[r, col], channels[0], smoke, floor) for r, smoke in enumerate(legacy.SMOKE_ORDER) for col, floor in enumerate(legacy.FLOOR_ORDER)]
        else:
            fig, axes = plt.subplots(4, 4, figsize=(14, 20), squeeze=False)
            panels = [(axes[r, col], c, smoke, floor) for r, (smoke, floor) in enumerate(row_keys) for col, c in enumerate(channels)]
        for ax, c, smoke, floor in panels:
            legacy.draw_panel(ax, zones, data[c, smoke, floor], scales[c, smoke], floor, None,
                              zero_colored=zero_colored)
            label='Arrow' if c=='ArrowRibbon' else c
            ax.set_title(f'{label}\n{smoke} | {floor}', fontsize=11)
        fig.suptitle(figure_title, fontsize=15)
        fig.subplots_adjust(top=.93, bottom=.10, hspace=.18, wspace=.08)
        cax = fig.add_axes([.28, .06, .44, .012])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=legacy.HEAT_CMAP), cax=cax, orientation='horizontal', label=scale_label)
        fig.text(.5, .015, 'AOI events >=100 ms; 2 m Gaussian estimator; raw numerical rates saved separately.', ha='center', fontsize=9)
        fig.savefig(out / filename, dpi=dpi, facecolor='white', bbox_inches='tight')
        plt.close(fig); files.append(filename)
    return files


def cropped_panel_bounds(zones, floor):
    """Crop empty map margins without moving the calibrated background or data."""
    extent=legacy.geometry_extent(zones,floor)
    background=legacy.calibrated_background_extent(extent,floor)
    image=legacy.background_image(floor)
    visible=(image[:,:,:3].min(axis=2)<235)&(image[:,:,3]>0)
    rows,cols=np.nonzero(visible)
    if not len(rows):
        raise ValueError('Empty floor-plan background')
    height,width=image.shape[:2]
    bounds=np.array([background[0]+cols.min()/width*(background[1]-background[0]),
        background[3]-(rows.max()+1)/height*(background[3]-background[2]),
        background[0]+(cols.max()+1)/width*(background[1]-background[0]),
        background[3]-rows.min()/height*(background[3]-background[2])])
    geometry=legacy.floor_footprint(zones,floor).bounds
    bounds[:2]=np.minimum(bounds[:2],geometry[:2])
    bounds[2:]=np.maximum(bounds[2:],geometry[2:])
    pad_x=(bounds[2]-bounds[0])*.01
    pad_y=(bounds[3]-bounds[1])*.005
    return bounds+np.array([-pad_x,-pad_y,pad_x,pad_y])


def plot_smoke_pair(data, scales, zones, output, mode, smoke, dpi, zero_colored):
    """Render two undistorted floors with identical top/bottom canvas positions."""
    legacy.configure_style()
    bounds={floor:cropped_panel_bounds(zones,floor) for floor in legacy.FLOOR_ORDER}
    panel_height,margin,gap=8.0,.04,.12
    widths={floor:panel_height*(b[2]-b[0])/(b[3]-b[1]) for floor,b in bounds.items()}
    width=2*margin+sum(widths.values())+gap
    height=panel_height+2*margin
    fig=plt.figure(figsize=(width,height),dpi=dpi)
    axes=[]
    x=margin
    for floor in legacy.FLOOR_ORDER:
        ax=fig.add_axes([x/width,margin/height,widths[floor]/width,panel_height/height])
        legacy.draw_panel(ax,zones,data[mode,smoke,floor],scales[mode,smoke],floor,None,
                          zero_colored=zero_colored)
        b=bounds[floor]
        ax.set_xlim(b[0],b[2]); ax.set_ylim(b[1],b[3])
        ax.set_axis_off()
        axes.append(ax)
        x+=widths[floor]+gap
    fig.canvas.draw()
    positions=[ax.get_window_extent().bounds for ax in axes]
    if not np.allclose([p[1] for p in positions],positions[0][1],atol=1e-6,rtol=0) or not np.allclose(
            [p[3] for p in positions],positions[0][3],atol=1e-6,rtol=0):
        raise ValueError('Floor panels do not align')
    if any(text.get_text() for ax in axes for text in ax.texts):
        raise ValueError('Unexpected subplot annotation')
    name=f'{mode}_{smoke}_Hall_Platform.png'
    temporary=output/('.'+name)
    try:
        fig.savefig(temporary,dpi=dpi,facecolor='white')
        temporary.replace(output/name)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    return name,dict(panel_height_px=positions[0][3],panel_bottom_px=positions[0][1],
                     panel_bounds_m={floor:value.tolist() for floor,value in bounds.items()})


def plot_shared_colorbar(output,dpi,orientation='horizontal', *,
                        label='Within-mode/smoke relative acquisition (P98 = 1)'):
    legacy.configure_style()
    if orientation not in ('horizontal','vertical'):
        raise ValueError('Invalid colorbar orientation')
    vertical=orientation=='vertical'
    fig=plt.figure(figsize=(1.1,7.2) if vertical else (7.2,.75),dpi=dpi)
    ax=fig.add_axes([.12,.04,.17,.92] if vertical else [.04,.56,.92,.22])
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=legacy.HEAT_CMAP),
                      cax=ax,orientation=orientation,ticks=np.linspace(0,1,6))
    cbar.ax.tick_params(labelsize=11,pad=3)
    cbar.set_label(label,fontsize=11,labelpad=5)
    name='Shared_Colorbar_Vertical.png' if vertical else 'Shared_Colorbar.png'
    fig.savefig(output/name,dpi=dpi,facecolor='white')
    plt.close(fig)
    return name


def render_existing(source, output, dpi, zero_colored, split_smoke=False):
    """Re-render stored V6 numerical fields without changing their estimates."""
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    multimodal=json.loads((source/'multimodal_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete' or multimodal.get('status')!='complete':
        raise ValueError('Complete V6 channel and Multimodal surfaces first')
    if manifest['aoi_manifest_sha256']!=multimodal['aoi_manifest_sha256']:
        raise ValueError('V6 channel and Multimodal AOI sources differ')
    data, scales, inputs={}, {}, {}
    for mode in CHANNELS+['Multimodal']:
        for smoke in legacy.SMOKE_ORDER:
            scales[mode,smoke]=(multimodal['scale'][smoke] if mode=='Multimodal'
                                else manifest['scale'][mode+'/'+smoke])
            for floor in legacy.FLOOR_ORDER:
                folder='multimodal_surfaces' if mode=='Multimodal' else 'channel_surfaces'
                path=source/folder/f'{mode}_{smoke}_{floor}.npz'
                with np.load(path) as stored:
                    surface={key:stored[key].copy() for key in stored.files}
                values=surface['field'][np.isfinite(surface['field'])]
                if not len(values) or (values<0).any() or (values>1).any():
                    raise ValueError('Invalid stored heatmap rates: '+str(path))
                data[mode,smoke,floor]=surface
                inputs[str(path.relative_to(source))]=digest(path)
    output.mkdir(parents=True,exist_ok=True)
    zones=legacy.load_zones()
    files=[]
    layout={}
    with plt.rc_context({'font.size':10,'axes.labelsize':10,
                         'xtick.labelsize':10,'ytick.labelsize':10}):
        for index,mode in enumerate(CHANNELS+['Multimodal'],2):
            if split_smoke:
                for smoke in legacy.SMOKE_ORDER:
                    name,positions=plot_smoke_pair(data,scales,zones,output,mode,smoke,dpi,zero_colored)
                    files.append(name)
                    layout[name]=positions
                continue
            files.extend(plot_maps(data,scales,zones,output,dpi,
                figure_specs=[([mode],f'{index:02}_{mode}_Observed_Acquisition.png')],
                figure_title=('Multimodal (Full) | observed AR guidance acquisition' if mode=='Multimodal'
                              else 'Observed guidance acquisition | equal condition contribution'),
                scale_label=('Within-Multimodal/smoke relative acquisition (P98 = 1)' if mode=='Multimodal'
                             else 'Within-channel/smoke relative acquisition (P98 = 1)'),
                zero_colored=zero_colored))
        colorbar=plot_shared_colorbar(output,dpi) if split_smoke else None
    result=dict(status='complete',files=files,dpi=dpi,colormap=legacy.HEAT_CMAP_NUMBER,
        zero_colored=zero_colored,supported_cell_alpha=.92 if zero_colored else 'value/support dependent',
        missing_support='transparent; never replaced with zero',
        input_surfaces_sha256=inputs,source_manifest_sha256=digest(source/'manifest.json'),
        multimodal_manifest_sha256=digest(source/'multimodal_manifest.json'),
        scale={mode+'/'+smoke:value for (mode,smoke),value in scales.items()},
        source_sha256={Path(p).name:digest(p) for p in [__file__,legacy.__file__]},
        aggregation={'channels':manifest['aggregation'],'Multimodal':multimodal['aggregation']},
        event_rule={'channels':manifest['event_rule'],'Multimodal':multimodal['event_rule']})
    result.update(split_smoke=split_smoke,shared_colorbar=colorbar,layout=layout)
    (output/'render_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Saved '+str(len(files))+' mode figures: '+str(output.resolve()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aoi-root', type=Path, default=AOI_ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('--render-existing',action='store_true',help='Render cached channel and Multimodal surfaces only')
    p.add_argument('--source',type=Path,default=DEFAULT_OUTPUT,help='Existing V6 surface directory')
    p.add_argument('--zero-colored',action='store_true',help='Show supported zero rates in blue, leaving NaN transparent')
    p.add_argument('--split-smoke',action='store_true',help='One title-free floor pair per mode/smoke, plus a shared colorbar')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--bandwidth-m', type=float, default=legacy.DEFAULT_BANDWIDTH_M)
    args = p.parse_args()
    if args.dpi < 50 or not np.isfinite(args.bandwidth_m) or args.bandwidth_m <= 0:
        p.error('Invalid resolution/bandwidth')
    if args.split_smoke and not args.render_existing:
        p.error('--split-smoke requires --render-existing')
    if args.render_existing:
        output=args.output or args.source/('Zero_Colored' if args.zero_colored else 'Rerendered')
        render_existing(args.source,output,args.dpi,args.zero_colored,args.split_smoke)
        return
    args.output=args.output or DEFAULT_OUTPUT
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['condition_surfaces', 'channel_surfaces']:
        (args.output / name).mkdir()
    zones = legacy.load_zones()
    points, audit = load_points(args.aoi_root, zones)
    audit.to_csv(args.output / 'trial_channel_events.csv', index=False)
    audit.groupby(['Scenario', 'Channel']).agg(trials=('ParticipantID', 'size'), valid_s=('ValidDuration_s', 'sum'), gaze_s=('GazeDuration_s', 'sum')).to_csv(args.output / 'condition_channel_events.csv')
    data, scales, support = surfaces(points, zones, args.bandwidth_m, args.output)
    support.to_csv(args.output / 'channel_surface_support.csv', index=False)
    files = plot_maps(data, scales, zones, args.output, args.dpi,zero_colored=args.zero_colored)
    manifest = dict(status='complete', trained=False, channels=CHANNELS, trial_channel_records=len(audit),
        cohort_trials=270, cohort_participants=45, files=files, bandwidth_m=args.bandwidth_m,
        aggregation='participant-balanced rates per condition, equal mean of supported conditions per cell',
        event_rule='>=100 ms, at least two frames, gaps >100 ms break events',
        scale={c + '/' + s: v for (c, s), v in scales.items()},
        aoi_manifest_sha256=digest(args.aoi_root / 'manifest.json'),
        source_sha256={str(Path(f).resolve()): digest(f) for f in [__file__, legacy.__file__]},
        note='Observed maps, not IRL rewards. Cross-condition maps do not have a single target exit.')
    manifest['zero_colored']=args.zero_colored
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('OUTPUT=' + str(args.output.resolve()))


if __name__ == '__main__':
    main()
