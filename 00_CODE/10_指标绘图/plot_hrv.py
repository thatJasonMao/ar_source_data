"""Three panels, three complete violins per panel, using shared audited HRV input."""
from pathlib import Path
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'INDICATORS_SIGNIFICANCE'))
from HRVInput import prepare_hrv, group_wide, MODES, LABELS


def run(root, output, mapping=None, force=False):
    output=Path(output)
    if not force and any((output/name).exists() for name in
            ['HRV_three_stages.png','HRV_three_stages.pdf','HRV_plot_data.csv',
             'stage_value_audit.csv','cohort_coverage.csv','descriptive.csv']):
        raise FileExistsError('Output exists; use --force or another output')
    data,audit,coverage,_,_=prepare_hrv(root,mapping)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':16.5,
        'axes.linewidth':1.2,'axes.labelsize':16.5,'axes.titlesize':19.5,
        'xtick.labelsize':15,'ytick.labelsize':15,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes=plt.subplots(1,3,figsize=(14,5.6),sharey=True,facecolor='white')
    colors=['#D9D9D9','#999999','#666666']
    for ax,group in tqdm(list(zip(axes,['A','B','C'])),desc='HRV panels'):
        wide=group_wide(data,group)
        for x,mode in enumerate(MODES[group]):
            values=wide[mode].to_numpy()
            low,high=values.min(),values.max()
            if low==high:
                y=np.linspace(low-.01,high+.01,50);density=np.ones(50)
            else:
                pad=(high-low)*.03
                y=np.linspace(low-pad,high+pad,300)
                density=gaussian_kde(values)(y)
            width=.35*density/density.max()
            ax.fill_betweenx(y,x-width,x+width,color=colors[x],edgecolor='#555555',linewidth=1.1,alpha=.92,zorder=1)
            ax.boxplot(values,positions=[x],widths=.16,patch_artist=True,showfliers=False,manage_ticks=False,
                boxprops={'facecolor':'white','edgecolor':'black','linewidth':1.1},
                medianprops={'color':'black','linewidth':1.4},
                whiskerprops={'color':'black','linewidth':1},capprops={'color':'black','linewidth':1},zorder=4)
            ax.text(x+.23,values.mean(),f'{values.mean():.2f}',ha='center',va='center',fontsize=11,zorder=8,
                bbox={'boxstyle':'round,pad=.25','facecolor':'white','edgecolor':'#555555','linewidth':.9})
        ax.set_title(f'({group.lower()}) {LABELS[MODES[group][1]]} Group',fontsize=19.5,fontweight='bold',pad=12)
        ax.set_xticks(range(3),[LABELS[m] for m in MODES[group]])
        ax.set_xlabel('Guidance',fontweight='bold',labelpad=9)
        ax.set_xlim(-.6,2.65)
        ax.yaxis.grid(True,linestyle='--',linewidth=.6,alpha=.35);ax.set_axisbelow(True)
        for side in ['top','right']:ax.spines[side].set_visible(False)
        ax.tick_params(width=1,length=5,pad=6)
    axes[0].set_ylabel('Stress Index Change',fontweight='bold',labelpad=9)
    low,high=data.HRVChange.min(),data.HRVChange.max()
    margin=max(.3,(high-low)*.08)
    axes[0].set_ylim(low-margin,high+margin)
    fig.tight_layout(w_pad=1.3)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'HRV_three_stages.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(output/'HRV_three_stages.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
    data.to_csv(output/'HRV_plot_data.csv',index=False,encoding='utf-8-sig')
    audit.to_csv(output/'stage_value_audit.csv',index=False,encoding='utf-8-sig')
    coverage.to_csv(output/'cohort_coverage.csv',index=False,encoding='utf-8-sig')
    data.groupby(['Group','StageOrder','Guidance']).HRVChange.agg(['count','mean','std','median','min','max']).to_csv(output/'descriptive.csv',encoding='utf-8-sig')
    print(data.groupby('Group').ParticipantID.nunique())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--mapping',type=Path)
    p.add_argument('--force',action='store_true')
    a=p.parse_args();run(a.root,a.output,a.mapping,a.force)
