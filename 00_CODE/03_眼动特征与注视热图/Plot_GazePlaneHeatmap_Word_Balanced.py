"""Additional three-row background heatmap layout for Word.

Run: python Plot_GazePlaneHeatmap_Word_Balanced.py
Rows contain StaticSign / Arrow, MiniMap / TextBar, then centered Multimodal.
Each modality is a Low smoke / High smoke pair. Existing scripts and figures
are preserved; densities are read from the verified existing preview caches.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import Extract_GazePlaneHeatmap_Background as background

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.text import Annotation
from matplotlib.transforms import ScaledTranslation
import numpy as np


DEFAULT_LAYOUT = {
    'figure_width_in':8.267717,
    'margin_left_in':.511417, 'margin_right_in':0,
    'margin_top_in':.25, 'margin_bottom_in':.775984,
    'pair_gap_in':0, 'group_gap_in':.173228, 'row_gap_in':.438976,
    'group_title_offset_in':.190551,
    'group_font_pt':10, 'smoke_font_pt':9, 'annotation_font_pt':8,
    'tick_font_pt':7.5, 'axis_font_pt':9.5, 'colorbar_font_pt':9,
    'group_bold':True, 'smoke_bold':False, 'annotation_bold':False,
    'tick_bold':False, 'axis_bold':False, 'colorbar_bold':False,
    'spine_width_pt':.45, 'grid_width_pt':.25, 'leader_width_pt':.6,
    'marker_size_pt':4.5,
}

# Single-line panel titles leave only the necessary title clearance between rows.
COMPACT_LAYOUT = {
    **DEFAULT_LAYOUT,
    'margin_left_in':.40, 'margin_top_in':.18, 'margin_bottom_in':.65,
    'pair_gap_in':0, 'group_gap_in':.035, 'row_gap_in':.18,
}


def validate_layout(values, compact=False):
    unknown=set(values)-set(DEFAULT_LAYOUT)
    if unknown:
        raise ValueError('Unknown layout parameters: '+', '.join(sorted(unknown)))
    layout={**(COMPACT_LAYOUT if compact else DEFAULT_LAYOUT),**values}
    for key,default in DEFAULT_LAYOUT.items():
        value=layout[key]
        if isinstance(default,bool):
            if not isinstance(value,bool):
                raise ValueError(f'{key} must be boolean')
        elif isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        elif value<0:
            raise ValueError(f'{key} must be nonnegative')
    if not 5 <= layout['figure_width_in'] <= 14:
        raise ValueError('Figure width must be between 5 and 14 inches')
    if layout['margin_bottom_in']<.65 or layout['margin_top_in']<(.16 if compact else .25):
        raise ValueError('Top/bottom margins are too small for the shared labels')
    if layout['margin_left_in']<.35:
        raise ValueError('Left margin must be at least .35 inches')
    minimum_gap=.14 if compact else .25
    if layout['row_gap_in']<minimum_gap:
        raise ValueError(f'Row gap must be at least {minimum_gap} inches')
    for key in ('group_font_pt','smoke_font_pt','annotation_font_pt','tick_font_pt',
                'axis_font_pt','colorbar_font_pt'):
        if not 4 <= layout[key] <= 24:
            raise ValueError(f'{key} must be between 4 and 24 points')
    available=(layout['figure_width_in']-layout['margin_left_in']-layout['margin_right_in']
               -2*layout['pair_gap_in']-layout['group_gap_in'])
    if available/4<.8:
        raise ValueError('Margins and gaps leave less than .8 inches for each panel')
    return layout


def font_weight(layout, key):
    return 'bold' if layout[key+'_bold'] else 'normal'


def draw_balanced_overview(densities, config, outlines, layout=None, output=None, *, compact=False):
    layout=validate_layout(layout or {},compact)
    if compact:
        config=deepcopy(config)
        top=max(background.extent(background.group_for(scenario,config),config)[3]
                for scenario in densities)
        config['coordinate_system']['vertical_range_m']=[1.20,top+.03]
        config['rendering']['arrow_outline']['width_screen_px']*=2.25
    xr=config['coordinate_system']['x_range_m']
    yr=config['coordinate_system']['vertical_range_m']
    width=layout['figure_width_in']
    left,right,top,bottom=(layout['margin_'+key+'_in'] for key in ('left','right','top','bottom'))
    pair_gap,group_gap,row_gap=(layout[key+'_gap_in'] for key in ('pair','group','row'))
    panel_w=(width-left-right-2*pair_gap-group_gap)/4
    panel_h=panel_w*(yr[1]-yr[0])/(xr[1]-xr[0])
    span=4*panel_w+2*pair_gap+group_gap
    pair_w=2*panel_w+pair_gap
    height=top+bottom+3*panel_h+2*row_gap
    fig=plt.figure(figsize=(width,height),dpi=600)
    rows=(('StaticSign','ArrowRibbon'),('MiniMap','TextBar'),('Full',))
    panels=[]
    for row,keys in enumerate(rows):
        y=height-top-(row+1)*panel_h-row*row_gap
        for group,key in enumerate(keys):
            pair_left=(left+group*(pair_w+group_gap) if len(keys)==2
                       else left+(span-pair_w)/2)
            if not compact:
                fig.text((pair_left+pair_w/2)/width,(y+panel_h+layout['group_title_offset_in'])/height,
                         background.NAMES[key],ha='center',va='center',
                         fontsize=layout['group_font_pt'],fontweight=font_weight(layout,'group'))
            for col,smoke in enumerate(background.COMBINED_COLUMNS):
                scenario=f'{key}-{smoke}'
                x=pair_left+col*(panel_w+pair_gap)
                ax=fig.add_axes([x/width,y/height,panel_w/width,panel_h/height])
                background.draw_panel(ax,scenario,densities[scenario][0],config)
                ax.set_xticks([xr[0],np.mean(xr),xr[1]])
                ax.set_yticks(np.arange(yr[0],yr[1]+1e-9,.20) if compact
                             else [yr[0],np.mean(yr),yr[1]])
                ax.set_xticks(np.arange(xr[0],xr[1]+1e-9,.25),minor=True)
                ax.set_yticks([] if compact else np.arange(yr[0],yr[1]+1e-9,.25),minor=True)
                ax.set_xlim(xr); ax.set_ylim(yr)
                ax.grid(which='both',color='lightgray',linewidth=layout['grid_width_pt'],alpha=.7)
                show_y=group==0 and col==0
                ax.tick_params(axis='both',which='major',labelsize=layout['tick_font_pt'],
                               length=2,width=layout['spine_width_pt'],pad=2,
                               labelleft=show_y,labelbottom=row==2)
                ax.tick_params(which='minor',length=0)
                for spine in ax.spines.values():
                    spine.set_linewidth(layout['spine_width_pt'])
                for label in ax.get_xticklabels()+ax.get_yticklabels():
                    label.set_fontweight(font_weight(layout,'tick'))
                if row==2:
                    first,last=ax.get_xticklabels()[0],ax.get_xticklabels()[-1]
                    first.set_horizontalalignment('left')
                    last.set_horizontalalignment('right')
                    # Separate edge labels even when the two panel borders touch.
                    first.set_transform(first.get_transform()+ScaledTranslation(2/72,0,fig.dpi_scale_trans))
                    last.set_transform(last.get_transform()+ScaledTranslation(-2/72,0,fig.dpi_scale_trans))
                if show_y:
                    ax.get_yticklabels()[0].set_verticalalignment('bottom')
                    ax.get_yticklabels()[-1].set_verticalalignment('top')
                for label in ax.texts:
                    label.set_fontsize(layout['annotation_font_pt'])
                    label.set_fontweight(font_weight(layout,'annotation'))
                    if compact and label.get_text().startswith('H = '):
                        label.set_fontweight('bold')
                        bounds=background.extent(background.group_for(scenario,config),config)
                        label.set_position((bounds[0]+.025,bounds[3]-.020))
                        label.set_horizontalalignment('left')
                        label.set_verticalalignment('top')
                    if isinstance(label,Annotation):
                        _,py=background.peak(densities[scenario][0])
                        below=key in ('TextBar','Full')
                        label.set_position((xr[0]+.025,py-.16 if below else py+.13))
                        label.set_horizontalalignment('left')
                        label.set_verticalalignment('top' if below else 'bottom')
                        label.arrow_patch.set_linewidth(layout['leader_width_pt'])
                for line in ax.lines:
                    if line.get_marker()=='^':
                        line.set_markersize(layout['marker_size_pt'])
                smoke_label='Low smoke' if col==0 else 'High smoke'
                ax.set_title((background.NAMES[key]+' | '+smoke_label) if compact else smoke_label,
                             fontsize=layout['smoke_font_pt'],fontweight=font_weight(layout,'smoke'),
                             pad=1 if compact else 3)
                panels.append((ax,scenario))
    fig.text(.075/width,(bottom+(3*panel_h+2*row_gap)/2)/height,
             'Vertical (m)',rotation=90,ha='center',va='center',
             fontsize=layout['axis_font_pt'],fontweight=font_weight(layout,'axis'))
    center=(left+span/2)/width
    fig.text(center,.42/height,'Horizontal (m)',ha='center',va='center',
             fontsize=layout['axis_font_pt'],fontweight=font_weight(layout,'axis'))
    # Shared relative-density colorbar; the normalization is unchanged.
    colorbar_bottom=.16 if compact else .24
    cax=fig.add_axes([left/width,colorbar_bottom/height,span/width,.065/height])
    cbar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),
                      cmap=background.original.get_custom_cmap()),
                      cax=cax,orientation='horizontal',ticks=np.linspace(0,1,6))
    cbar.set_label('Relative density',fontsize=layout['colorbar_font_pt'],
                   fontweight=font_weight(layout,'colorbar'),labelpad=0 if compact else 2)
    cbar.ax.tick_params(labelsize=layout['tick_font_pt'],length=2,width=layout['spine_width_pt'],pad=1)
    cbar.outline.set_linewidth(layout['spine_width_pt'])
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight(font_weight(layout,'tick'))
    fig.canvas.draw()
    if compact:
        # Ensure the tighter headers still clear the row above them in display pixels.
        renderer=fig.canvas.get_renderer()
        for index,(ax,scenario) in enumerate(panels):
            if index>=4:
                row_above=[upper.get_window_extent().y0 for upper,_ in panels[:4] if index<8]
                if index>=8:
                    row_above=[upper.get_window_extent().y0 for upper,_ in panels[4:8]]
                if ax.title.get_window_extent(renderer).y1>min(row_above)-1:
                    raise ValueError('Compact panel title overlaps the previous row')
    for ax,scenario in panels:
        background.draw_outlines(ax,scenario,config,outlines)
    background.validate_figure_font(fig)
    output=Path(output) if output else background.OUTPUT/'GazePlaneHeatmap_All_Conditions_Background_Word_Balanced_3x4.png'
    pending=output.with_name('.'+output.stem+'_render.png')
    try:
        fig.savefig(pending,dpi=600,bbox_inches='tight',pad_inches=.03,facecolor='white')
        pending.replace(output)
    finally:
        plt.close(fig)
        if pending.exists():
            pending.unlink()
    print('Saved '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=background.ASSETS/'plot_config.json')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--layout-config',type=Path,help='Layout JSON exported by the HTML editor')
    parser.add_argument('--output',type=Path,help='Optional alternative PNG destination')
    parser.add_argument('--compact',action='store_true',
                        help='Three compact rows; single-line titles, H inside background, tight crop and gaps')
    args=parser.parse_args()
    config=background.read_json(args.config)
    outlines=background.read_json(background.asset_path(
        config['rendering']['arrow_outline']['contours_file']))['groups']
    background.validate(config,outlines)
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),
                          fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.serif':['Times New Roman'],
                         'mathtext.fontset':'stix','axes.unicode_minus':False})
    densities=background.densities_from_preview()
    layout={}
    if args.layout_config:
        exported=background.read_json(args.layout_config)
        if exported.get('schema')!='gaze-plane-balanced-layout' or exported.get('schema_version')!=1:
            raise ValueError('Unrecognized balanced-layout configuration')
        expected_background=exported.get('background_config_sha256')
        actual_background=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        if expected_background and expected_background!=actual_background:
            raise ValueError('Layout JSON was exported for a different background configuration')
        expected_densities=exported.get('density_fingerprints',{})
        for scenario,fingerprint in expected_densities.items():
            if scenario not in densities or densities[scenario][1]!=fingerprint:
                raise ValueError(f'Layout JSON density source changed: {scenario}')
        layout=exported['parameters']
    layout=validate_layout(layout,args.compact)
    if args.dry_run:
        print('Validated layout, backgrounds, contours, font and ten unchanged density/entropy values.')
        return
    draw_balanced_overview(densities,config,outlines,layout,args.output,compact=args.compact)


if __name__=='__main__':
    main()
