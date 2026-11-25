"""Export separate, assembly-ready components for Full-condition guidance gaze.

Composition uses mutually exclusive sustained-event states / sustained AR
union time, per trial then equal mean. Spatial channel rates use the common
AR-union valid denominator and an absolute 0--1 colour scale for all panels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

from Guidance_Channel_Contract import (
    ROOT, AOI_ROOT, AOI_SCHEMA, aoi_path, condition_gaze_masks, digest, timestamp_ns,
)
from Guidance_Sustained_AOI_V6 import remap_manifest_path, save_csv
from common import GUIDANCE_SMOKE_COLORS, load_trials
from a5_colormaps import get_cmap
import Visualize_Guidance_Channels_V6 as maps
from Query_GuidanceGaze import event_flags, sample_weights_ns

PARENT=ROOT/'Results_Guidance_Sustained_AOI_V6'
DEFAULT_OUTPUT=PARENT/'Multimodal_Components'
CHANNELS=('ArrowRibbon','MiniMap','TextBar')
LABELS={'ArrowRibbon':'Arrow','MiniMap':'MiniMap','TextBar':'TextBar'}
STATES=('ArrowOnly','MiniMapOnly','TextBarOnly','Overlap')
STATE_LABELS=('Arrow only','MiniMap only','TextBar only','Overlap')
COLORS=[GUIDANCE_SMOKE_COLORS[c]['HighSmoke'] for c in CHANNELS]+['#B0B0B0']
HEAT_CMAP_NUMBER=29
HEAT_CMAP=get_cmap(HEAT_CMAP_NUMBER)
KEYS=['Scenario','ParticipantID','FileName']
SMOKES=maps.legacy.SMOKE_ORDER
FLOORS=maps.legacy.FLOOR_ORDER


def exclusive_states(channel_flags):
    flags=np.asarray(channel_flags,dtype=bool)
    if flags.ndim!=2 or flags.shape[0]!=3:
        raise ValueError('Three separately filtered AR channel arrays required')
    count=flags.sum(axis=0)
    result=np.stack([*(hit & (count==1) for hit in flags),count>=2])
    if not np.array_equal(result.sum(axis=0),flags.any(axis=0).astype(int)):
        raise ValueError('Composition states must partition the retained union exactly')
    return result


def collect(zones):
    manifest=json.loads((AOI_ROOT/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema')!=AOI_SCHEMA or manifest.get('status')!='complete':
        raise ValueError('Complete V6 AOI input required')
    old=pd.read_csv(PARENT/'Trial_Proportions_Raw_vs_Sustained.csv')
    old=old.loc[old.SourceGuidance.eq('Full')].set_index(KEYS)
    channel_old=pd.read_csv(PARENT/'Trial_Channel_Event_Audit.csv').set_index(KEYS+['Channel'])
    roster=load_trials().loc[lambda f:f.Guidance.eq('Full')]
    if len(roster)!=90 or roster.ParticipantID.nunique()!=45 or roster.duplicated(KEYS).any():
        raise ValueError('Full condition requires 90 unique trials / 45 participants')
    if set(map(tuple,roster[KEYS].to_numpy()))!=set(old.index):
        raise ValueError('Full trial cohort changed since the primary figures')
    entries={(t['scenario'],t['participant'],t['source_file']):t for t in manifest['trials']}
    composition,channel_records,parts,inputs=[],[],[],[]
    for number,row in enumerate(roster.itertuples(index=False),1):
        identity={k:getattr(row,k) for k in KEYS}
        key=tuple(identity.values()); entry=entries[key]; path=aoi_path(row)
        if digest(path)!=entry['sha256']:
            raise ValueError('AOI source checksum differs')
        inputs.append(dict(**identity,File=str(path.relative_to(ROOT)),SHA256=entry['sha256']))
        for source,checksum in entry['input_hashes'].items():
            actual=remap_manifest_path(source)
            if digest(actual)!=checksum:
                raise ValueError('AOI acquisition source changed: '+str(actual))
            inputs.append(dict(**identity,File=str(actual.relative_to(ROOT)),SHA256=checksum))
        frame=pd.read_csv(path,low_memory=False)
        if len(frame)!=entry['rows']:
            raise ValueError('AOI row count differs')
        for column,value in [('Condition',row.Scenario),('GuidanceType','Full'),
                             ('SmokeLevel',row.Smoke),('ParticipantFolder',row.ParticipantID)]:
            if not frame[column].eq(value).all():
                raise ValueError('AOI trial identity differs')
        ns=timestamp_ns(frame.TimeStamp)
        masks=condition_gaze_masks(frame,'Full',ns,event_flags)
        assert masks['channels']==CHANNELS
        weights_ns=sample_weights_ns(ns)
        union_ns=int(weights_ns[masks['sustained']].sum())
        valid_ns=int(weights_ns[masks['valid']].sum())
        if union_ns!=old.loc[key,'SustainedGazeDuration_ns'] or valid_ns!=old.loc[key,'ValidDuration_ns']:
            raise ValueError('Events or denominator differ from the existing Multimodal main figure')
        state_flags=exclusive_states(masks['sustained_channels'])
        state_ns=[int(weights_ns[flag].sum()) for flag in state_flags]
        if sum(state_ns)!=union_ns:
            raise ValueError('Exclusive durations do not sum to the sustained union')
        record=dict(**identity,Smoke=row.Smoke,ValidDuration_ns=valid_ns,SustainedUnionDuration_ns=union_ns,
                    CompositionStatus='defined' if union_ns else 'no_sustained_union')
        for state,duration in zip(STATES,state_ns):
            record[state+'Duration_ns']=duration
            record[state+'Share']=duration/union_ns if union_ns else np.nan
        composition.append(record)
        for channel,flag in zip(CHANNELS,masks['sustained_channels']):
            duration=int(weights_ns[flag].sum())
            if duration!=channel_old.loc[(*key,channel),'SustainedDuration_ns']:
                raise ValueError('Channel event duration differs from the previous audit')
            channel_records.append(dict(**identity,Smoke=row.Smoke,Channel=LABELS[channel],
                SustainedDuration_ns=duration,UnionValidDuration_ns=valid_ns,
                SustainedUnionDuration_ns=union_ns,
                ProportionOfUnionValidTime=duration/valid_ns if valid_ns else np.nan,
                NonexclusiveShareOfUnionGaze=duration/union_ns if union_ns else np.nan))
        movement=pd.read_csv(row.SubjectFile,usecols=['TimeStamp','PosX','PosY','PosZ'])
        if not np.array_equal(ns,timestamp_ns(movement.TimeStamp)):
            raise ValueError('AOI/participant-position sample alignment differs')
        xyz=movement[['PosX','PosY','PosZ']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        floor=maps.legacy.assign_floor(xyz[:,0],xyz[:,1],xyz[:,2],zones)
        keep=masks['valid'] & np.isfinite(xyz).all(axis=1) & np.isin(floor,FLOORS)
        weights=weights_ns[keep]/1e9
        data=dict(ParticipantID=row.ParticipantID,Smoke=row.Smoke,Floor=floor[keep],
                  PlotX=-xyz[keep,0],PlotY=-xyz[keep,2],ValidDuration_s=weights,
                  UnionGaze_s=weights*masks['sustained'][keep])
        for channel,flag in zip(CHANNELS,masks['sustained_channels']):
            data[channel+'Gaze_s']=weights*flag[keep]
        parts.append(pd.DataFrame(data))
        if number%30==0:
            print(f'Checked Full sources and events: {number}/90',flush=True)
    trial=pd.DataFrame(composition)
    for smoke in SMOKES:
        part=trial.loc[trial.Smoke.eq(smoke)]
        if len(part)!=45 or part.ParticipantID.nunique()!=45:
            raise ValueError('Each smoke group needs 45 Full trials')
    shares=trial[[s+'Share' for s in STATES]]
    np.testing.assert_allclose(shares.sum(axis=1,min_count=4).dropna(),1,atol=1e-14,rtol=0)
    return trial,pd.DataFrame(channel_records),pd.concat(parts,ignore_index=True),pd.DataFrame(inputs)


def summaries(trial,channels):
    rows=[]
    for smoke in SMOKES:
        part=trial.loc[trial.Smoke.eq(smoke)]
        for state,label,color in zip(STATES,STATE_LABELS,COLORS):
            values=part[state+'Share'].dropna()
            rows.append(dict(Smoke=smoke,State=state,Label=label,Color=color,
                n_trials=len(part),n_defined=len(values),mean=values.mean(),median=values.median(),sd=values.std(ddof=1)))
    summary=pd.DataFrame(rows)
    np.testing.assert_allclose(summary.groupby('Smoke')['mean'].sum(),1,atol=1e-14,rtol=0)
    rates=channels.groupby(['Smoke','Channel'],sort=False).ProportionOfUnionValidTime.agg(
        n='count',mean='mean',median='median',sd='std').reset_index()
    return summary,rates


def save_vector_and_png(fig,stem,dpi):
    """Transparent components; SVG preserves editable Times New Roman text."""
    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontproperties().get_name()!='Times New Roman':
            raise ValueError('Unexpected component font')
    fig.savefig(stem.with_suffix('.png'),dpi=dpi,transparent=True,bbox_inches='tight',pad_inches=.04)
    with plt.rc_context({'svg.fonttype':'none'}):
        fig.savefig(stem.with_suffix('.svg'),transparent=True,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def plot_composition(summary,out,dpi):
    maps.legacy.configure_style()
    fig=plt.figure(figsize=(4.7,1.37))
    ax=fig.add_axes([.21,.36,.67,.60])
    labels=[]
    for y,smoke in zip([.70,0],SMOKES):
        left=0.
        for state,color in zip(STATES,COLORS):
            value=float(summary.loc[summary.Smoke.eq(smoke)&summary.State.eq(state),'mean'].iloc[0])
            ax.barh(y,value,left=left,height=.55,color=color,edgecolor='white',linewidth=.45)
            if state!='Overlap':
                luminance=np.dot(to_rgb(color),[.2126,.7152,.0722])
                label=f'{value:.1%}'
                ax.text(left+value/2,y,label,ha='center',va='center',fontsize=12,fontweight='bold',
                        color='white' if luminance<.52 else '#111111')
            else:
                label=f'{value:.1%}'
                ax.annotate(label,xy=(left+value/2,y),xytext=(1.035,y),
                    ha='left',va='center',fontsize=11.5,fontweight='bold',annotation_clip=False,
                    arrowprops=dict(arrowstyle='-',color='#555555',lw=.6,shrinkA=2,shrinkB=0))
            labels.append(dict(Smoke=smoke,State=state,mean=value,label=label))
            left+=value
    ax.set_yticks([.70,0],['Low smoke','High smoke'],fontsize=14,fontweight='bold')
    ax.set_xticks([0,.5,1],['0','50','100'],fontsize=12,fontweight='bold')
    ax.set_xlabel('Share of sustained guidance-gaze time (%)',fontsize=12.5,fontweight='bold',labelpad=3)
    ax.set_xlim(0,1); ax.set_ylim(-.40,1.10)
    ax.tick_params(axis='y',length=0,pad=6)
    ax.tick_params(axis='x',length=3,width=.65,pad=2)
    for name,spine in ax.spines.items():
        spine.set_visible(name=='bottom'); spine.set_linewidth(.65)
    save_vector_and_png(fig,out/'Composition_Inset',dpi)
    return pd.DataFrame(labels)


def plot_legend(out,dpi):
    maps.legacy.configure_style()
    fig=plt.figure(figsize=(1.95,1.02))
    handles=[Patch(facecolor=COLORS[i],edgecolor='#555555',linewidth=.45,label=STATE_LABELS[i])
             for i in range(4)]
    fig.legend(handles=handles,loc='center',ncol=1,frameon=False,
               prop={'family':'Times New Roman','weight':'bold','size':12},handlelength=1.1,
               handletextpad=.4,labelspacing=.20,borderaxespad=0)
    save_vector_and_png(fig,out/'Composition_Legend',dpi)


def surfaces(points,zones,out):
    data,records={},[]
    for smoke in SMOKES:
        for floor in FLOORS:
            part=points.loc[points.Smoke.eq(smoke)&points.Floor.eq(floor)]
            if part.ParticipantID.nunique()!=45:
                raise ValueError('Unexpected Full spatial participant coverage')
            extent=maps.legacy.geometry_extent(zones,floor)
            union=maps.legacy.participant_balanced_surface(part.assign(GazeDuration_s=part.UnionGaze_s),zones,floor,extent,2.)
            old_path=PARENT/'surfaces_sustained'/f'Multimodal_{smoke}_{floor}.npz'
            with np.load(old_path) as old:
                np.testing.assert_allclose(union['field'],old['field'],atol=1e-12,rtol=0,equal_nan=True)
                np.testing.assert_array_equal(union['support_n'],old['support_n'])
            np.savez_compressed(out/'surfaces'/f'Union_Reference_{smoke}_{floor}.npz',**union)
            for channel in CHANNELS:
                surface=maps.legacy.participant_balanced_surface(
                    part.assign(GazeDuration_s=part[channel+'Gaze_s']),zones,floor,extent,2.)
                field=surface['field']
                np.testing.assert_array_equal(np.isfinite(field),np.isfinite(union['field']))
                np.testing.assert_array_equal(surface['support_n'],union['support_n'])
                if np.any(field>union['field']+1e-12) or np.any(field[np.isfinite(field)]<0):
                    raise ValueError('Channel surface must be bounded by the Full union')
                name=LABELS[channel]; data[name,smoke,floor]=surface
                np.savez_compressed(out/'surfaces'/f'{name}_{smoke}_{floor}.npz',**surface)
                records.append(dict(Channel=name,Smoke=smoke,Floor=floor,
                    participants=45,min_support=surface['min_support'],
                    supported_cells=int(np.isfinite(field).sum()),zero_cells=int((field==0).sum()),
                    absolute_max=float(np.nanmax(field)),colour_min=0.,colour_max=1.))
        print('Recomputed common-denominator channel surfaces: '+smoke,flush=True)
    return data,pd.DataFrame(records)


def plot_pair(data,zones,out,channel,smoke,dpi):
    bounds={floor:maps.cropped_panel_bounds(zones,floor) for floor in FLOORS}
    height,margin,gap=8.,.04,.12
    widths={floor:height*(b[2]-b[0])/(b[3]-b[1]) for floor,b in bounds.items()}
    fw=2*margin+sum(widths.values())+gap; fh=height+2*margin
    fig=plt.figure(figsize=(fw,fh))
    axes=[]; x=margin
    for floor in FLOORS:
        ax=fig.add_axes([x/fw,margin/fh,widths[floor]/fw,height/fh])
        # One absolute scale for all three channels and both smoke levels.
        maps.legacy.draw_panel(ax,zones,data[channel,smoke,floor],1.,floor,None,zero_colored=True)
        # Recolour this artist only: the five-mode main figure keeps its palette.
        ax.images[1].set_cmap(HEAT_CMAP)
        b=bounds[floor]; ax.set_xlim(b[0],b[2]); ax.set_ylim(b[1],b[3]); ax.set_axis_off()
        axes.append(ax); x+=widths[floor]+gap
    fig.canvas.draw()
    positions=np.array([ax.get_position().bounds for ax in axes])
    np.testing.assert_allclose(positions[:,1],positions[0,1],atol=1e-12,rtol=0)
    np.testing.assert_allclose(positions[:,3],positions[0,3],atol=1e-12,rtol=0)
    assert not any(ax.texts for ax in axes)
    filename=f'{channel}_{smoke}_Hall_Platform.png'
    fig.savefig(out/filename,dpi=dpi,transparent=True)
    plt.close(fig)
    return filename,dict(height_in=height,width_in=fw,transparent_outer_canvas=True,
                         floor_bounds_m={f:b.tolist() for f,b in bounds.items()})


def plot_colorbar(out,dpi,orientation):
    maps.legacy.configure_style()
    vertical=orientation=='vertical'
    fig=plt.figure(figsize=(1.15,6.) if vertical else (5.7,.80))
    ax=fig.add_axes([.10,.04,.18,.92] if vertical else [.04,.59,.92,.22])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=HEAT_CMAP),
                    cax=ax,orientation=orientation,ticks=np.linspace(0,1,6))
    bar.ax.tick_params(labelsize=12,pad=3,width=.6)
    bar.outline.set_linewidth(.6)
    bar.set_label('Local sustained guidance-gaze proportion',fontsize=12,labelpad=6)
    stem='Heatmap_Colorbar_Vertical' if vertical else 'Heatmap_Colorbar_Horizontal'
    save_vector_and_png(fig,out/stem,dpi)


def run(args):
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new directory: '+str(out))
    if not out.is_relative_to(PARENT) or out==PARENT:
        raise ValueError('Components must be saved in a new subdirectory of the requested parent')
    findfont(FontProperties(family='Times New Roman'),fallback_to_default=False)
    protected={str(p.relative_to(PARENT)):digest(p) for p in PARENT.rglob('*') if p.is_file()}
    # Independent hand-coded states include singleton, pair and triple overlaps.
    test=np.array([[1,0,0,1,1,0],[0,1,0,1,1,0],[0,0,1,0,1,0]],bool)
    np.testing.assert_array_equal(exclusive_states(test),[
        [1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,1,0]])
    zones=maps.legacy.load_zones()
    trial,channels,points,inputs=collect(zones)
    summary,rates=summaries(trial,channels)
    out.mkdir(); (out/'surfaces').mkdir()
    save_csv(trial,out/'Trial_Exclusive_Composition.csv')
    save_csv(summary,out/'Composition_Summary.csv')
    save_csv(channels,out/'Trial_Channel_Proportions.csv')
    save_csv(rates,out/'Channel_Proportion_Summary.csv')
    save_csv(inputs,out/'Input_Checksums.csv')
    labels=plot_composition(summary,out,args.dpi)
    save_csv(labels,out/'Composition_Plot_Labels.csv')
    plot_legend(out,args.dpi)
    fields,surface_summary=surfaces(points,zones,out)
    save_csv(surface_summary,out/'Spatial_Surface_Summary.csv')
    layout={}
    for channel in LABELS.values():
        for smoke in SMOKES:
            name,positions=plot_pair(fields,zones,out,channel,smoke,args.dpi)
            layout[name]=positions
        print('Exported standalone map pairs: '+channel,flush=True)
    for orientation in ['horizontal','vertical']:
        plot_colorbar(out,args.dpi,orientation)
    for name,checksum in protected.items():
        if digest(PARENT/name)!=checksum:
            raise ValueError('Existing primary figure/data changed: '+name)
    images=[]
    for file in out.glob('*.png'):
        with Image.open(file) as im:
            im.verify()
        with Image.open(file) as im:
            assert im.mode=='RGBA' and im.getextrema()[3][0]==0
            images.append(dict(file=file.name,width_px=im.width,height_px=im.height,dpi=args.dpi))
    methods=['# Multimodal 可拼接组件','',
        '所有组件只来自 Full 试次：45 人、90 trial，每烟雾条件 45 trial，不包含 StaticSign。',
        '事件复用 event_flags：逐通道 >=100 ms 且至少双样本，未命中/无效/>100 ms 间隔中断；保留完整时长。','',
        '- Composition_Inset.png / .svg：低、高烟两条100%堆叠条，无标题和内嵌图例。',
        '- Composition_Legend.png / .svg：独立一列四行竖版图例，从上到下为 Arrow only、MiniMap only、TextBar only、Overlap；紧凑行距。',
        '组成小窗及图例文字加粗，两条柱之间的物理空隙约0.082英寸（约2.1 mm）。',
        '- Arrow、MiniMap、TextBar × LowSmoke、HighSmoke 共6张配对热图：左站厅、右站台，等高，无面板标题/坐标/色条。',
        '- Heatmap_Colorbar_Horizontal / Vertical：独立绝对比例色条，PNG 与 SVG。','',
        '组成条：对分别筛选后的三个通道标记做互斥拆分，三种单独通道与任意至少两通道同时命中的 Overlap。',
        '每 trial 的状态时长 / 持续 AR 联合时长，再对45个 trial 等权平均；不是汇总所有时长后计算。',
        '图中百分比独立保留1位小数，显示值之和可能略偏离100%；精确份额之和为100%。',
        '灰色小段的百分比标在条右端。无联合持续事件时组成未定义，不能将其记为四类均为零；本批数据90个 trial 均有持续事件。','',
        '热图：各通道的合格事件时长 / 三个 AR 通道联合有效观测时长。受试者位置与AOI时间戳严格对齐。',
        '个人分子和共同分母分别2 m高斯平滑，计算局部比例，再对有支持参与者等权平均；沿用底图、分区和支持筛选。',
        '三通道、两烟雾共用绝对 0–1 比例色标，不作单图 P98 归一化。浅紫到深紫采用 a5_29 色带，有支持的零为浅紫，无支持透明。',
        '此独立色条与原五模式主图的 P98 相对色条分开使用，不可互换。',
        '组成条的分母是持续联合注视时间，热图的分母是联合有效观测时间，两个数值不要求相等。','',
        'PNG 为600 dpi，外部画布透明；组成小窗与图例的 SVG 保留可编辑文字，字体 Times New Roman。',
        '仅提供新增组件，不改动或重排已有组图。CSV、数值表面与 Metadata.json 供核对。','',
        '重跑：python Visualize_Multimodal_Guidance_Components_V6.py --output Results_Guidance_Sustained_AOI_V6/Multimodal_Components_Rerun']
    (out/'README.md').write_text('\n'.join(methods)+'\n',encoding='utf8')
    files=[__file__,ROOT/'Guidance_Channel_Contract.py',ROOT/'0_RL_SOURCE/Query_GuidanceGaze.py',
           ROOT/'common.py',maps.__file__,maps.legacy.__file__,ROOT/'Guidance_Sustained_AOI_V6.py',
           ROOT/'UnifiedZoneQuery.py',ROOT/'a5_colormaps.py',maps.legacy.ZONES_FILE,
           maps.legacy.ZONES_FILE.with_suffix('.json'),*[maps.legacy.BACKGROUND_DIR/(f+'.png') for f in FLOORS]]
    metadata=dict(schema='a5_multimodal_components_v6_1',status='complete',created_at=datetime.now(timezone.utc).isoformat(),
        source_condition='Full',display_condition='Multimodal',channels=list(CHANNELS),static_sign_included=False,
        participants=45,trials=90,trials_per_smoke=45,event_function='Query_GuidanceGaze.event_flags',
        min_duration_ms=100,min_samples=2,weights='sample_weights_ns; forward capped100ms; median terminal interval',
        composition=dict(states=dict(zip(STATES,STATE_LABELS)),colors=dict(zip(STATES,COLORS)),
            denominator='per-trial sustained AR union duration',aggregation='equal mean of per-trial shares by smoke',
            label_rounding='one decimal independently; exact shares sum to one',
            overlap='any two or three retained channels; one exclusive state, not apportioned',
            missing_union_trials=int(trial.SustainedUnionDuration_ns.eq(0).sum()),labels='Composition_Plot_Labels.csv'),
        spatial=dict(denominator='same Boolean union of AR channel-valid observation times for every channel',
            position='Subject PosX/PosY/PosZ, at matching AOI timestamp; display -X,-Z',
            bandwidth_m=2.,grid_width=maps.legacy.GRID_WIDTH,aggregation='equal mean of supported individual local rates',
            support='inherited 1%/1e-7 personal valid smooth threshold; >=max(2,ceil(.08*N)); footprint buffer .15m',
            normalization='none; absolute 0--1 across all 12 channel/smoke/floor surfaces',
            supported_zero='light lavender (#E8D8FC)',missing_support='NaN/transparent',layout=layout),
        validation=dict(exclusive_partition_test=True,trial_shares_sum_to_one=True,
            event_durations_match_primary_trial_and_channel_audits=True,
            union_surfaces_match_primary=True,shared_spatial_support=True,channel_fields_le_union=True,
            original_files_unchanged=len(protected),composition_label_count=len(labels),map_pairs=6),
        rendering=dict(font='Times New Roman',dpi=args.dpi,images=images,
            svg_text='editable text; fonts not converted to outlines',colormap=HEAT_CMAP_NUMBER,
            colormap_name=HEAT_CMAP.name),
        parent_metadata_sha256=digest(PARENT/'Metadata.json'),
        aoi_manifest_sha256=digest(AOI_ROOT/'manifest.json'),
        source_sha256={str(Path(f).relative_to(ROOT)):digest(f) for f in files},
        output_sha256={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file()})
    (out/'Metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    print(summary[['Smoke','Label','n_defined','mean']].to_string(index=False),flush=True)
    print('Saved components: '+str(out),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args()
    if args.dpi<100:
        parser.error('DPI must be >=100')
    run(args)


if __name__=='__main__':
    main()
