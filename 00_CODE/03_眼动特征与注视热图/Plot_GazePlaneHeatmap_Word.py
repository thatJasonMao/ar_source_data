"""Additional compact 2-row x 5-column background heatmap for Word.

Run with the same scientific Python environment as the background plot script:
    python Plot_GazePlaneHeatmap_Word.py

This layout-only entry point reuses the ten verified preview density caches.
It preserves the existing plotting script, single plots, 5-by-2 overview and HTML.
If the underlying data change, first refresh them with the background plot script.
"""
import argparse
from pathlib import Path

import Extract_GazePlaneHeatmap_Background as background

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.text import Annotation
import numpy as np


def draw_word_overview(densities, config, outlines):
    """Use typography sized for an approximately full-width Word page figure."""
    xr=config['coordinate_system']['x_range_m']
    yr=config['coordinate_system']['vertical_range_m']
    width=7.2
    left,right,top,bottom=.80,.08,.34,.70
    gap_x,gap_y=.06,.10
    panel_w=(width-left-right-4*gap_x)/5
    panel_h=panel_w*(yr[1]-yr[0])/(xr[1]-xr[0])
    height=top+bottom+2*panel_h+gap_y
    fig=plt.figure(figsize=(width,height),dpi=600)
    panels=[]
    for row,smoke in enumerate(background.COMBINED_COLUMNS):
        y=height-top-(row+1)*panel_h-row*gap_y
        for col,key in enumerate(background.COMBINED_ROWS):
            scenario=f'{key}-{smoke}'
            x=left+col*(panel_w+gap_x)
            ax=fig.add_axes([x/width,y/height,panel_w/width,panel_h/height])
            background.draw_panel(ax,scenario,densities[scenario][0],config)
            # Sparse labeled ticks; the existing quarter-metre grid remains.
            ax.set_xticks([-.75,0,.75])
            ax.set_yticks([1.125,1.75,2.375])
            ax.set_xticks(np.arange(xr[0],xr[1]+1e-9,.25),minor=True)
            ax.set_yticks(np.arange(yr[0],yr[1]+1e-9,.25),minor=True)
            ax.grid(which='minor',color='lightgray',linewidth=.35,alpha=.7)
            ax.grid(which='major',color='lightgray',linewidth=.35,alpha=.7)
            ax.tick_params(axis='both',which='major',labelsize=7.5,length=2,pad=2,
                           labelleft=col==0,labelbottom=row==1)
            ax.tick_params(which='minor',length=0)
            for label in ax.get_xticklabels()+ax.get_yticklabels():
                label.set_fontweight('bold')
            if row==1:
                ax.get_xticklabels()[0].set_horizontalalignment('left')
                ax.get_xticklabels()[-1].set_horizontalalignment('right')
            if col==0:
                ax.get_yticklabels()[0].set_verticalalignment('bottom')
                ax.get_yticklabels()[-1].set_verticalalignment('top')
            for label in ax.texts:
                label.set_fontsize(8)
                label.set_fontweight('bold')
                if isinstance(label,Annotation):
                    # Place the coordinate text inside the small panel instead
                    # of retaining the single-plot leader's long left offset.
                    _,py=background.peak(densities[scenario][0])
                    below=key in ('TextBar','Full')
                    label.set_position((xr[0]+.025,py-.16 if below else py+.13))
                    label.set_horizontalalignment('left')
                    label.set_verticalalignment('top' if below else 'bottom')
                    label.arrow_patch.set_linewidth(.5)
            # The original marker was designed for an eight-inch single plot.
            for line in ax.lines:
                if line.get_marker()=='^':
                    line.set_markersize(2.5)
            if row==0:
                ax.set_title(background.NAMES[key],fontsize=11,fontweight='bold',pad=6)
            panels.append((ax,scenario))
        fig.text(.35/width,(y+panel_h/2)/height,
                 'Low smoke' if row==0 else 'High smoke',rotation=90,
                 ha='center',va='center',fontsize=10,fontweight='bold')
    fig.text(.08/width,(bottom+panel_h+gap_y/2)/height,'Vertical (m)',
             rotation=90,ha='center',va='center',fontsize=10,fontweight='bold')
    center=(left+(5*panel_w+4*gap_x)/2)/width
    fig.text(center,.42/height,'Horizontal (m)',ha='center',va='center',
             fontsize=10,fontweight='bold')
    # Match the original relative-density scale, with one horizontal legend.
    cax=fig.add_axes([left/width,.24/height,(5*panel_w+4*gap_x)/width,.065/height])
    cbar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),
                      cmap=background.original.get_custom_cmap()),
                      cax=cax,orientation='horizontal',ticks=np.linspace(0,1,6))
    cbar.set_label('Relative density',fontsize=9,fontweight='bold',labelpad=2)
    cbar.ax.tick_params(labelsize=7.5,length=2,pad=1)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
    fig.canvas.draw()
    for ax,scenario in panels:
        background.draw_outlines(ax,scenario,config,outlines)
    background.validate_figure_font(fig)
    output=background.OUTPUT/'GazePlaneHeatmap_All_Conditions_Background_Word_2x5.png'
    fig.savefig(output,dpi=600,bbox_inches='tight',pad_inches=.03,facecolor='white')
    plt.close(fig)
    print('Saved '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=background.ASSETS/'plot_config.json')
    parser.add_argument('--dry-run',action='store_true')
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
    if args.dry_run:
        print('Validated backgrounds, contours, font and ten unchanged density/entropy values.')
        return
    draw_word_overview(densities,config,outlines)


if __name__=='__main__':
    main()
