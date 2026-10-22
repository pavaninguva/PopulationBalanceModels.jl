"""Compare actual Qamar moment-family runs, retaining incomplete trajectories."""
from pathlib import Path
import argparse
from difflib import SequenceMatcher
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'docs/src/assets/qamar_moments'
METHODS=('QMOM','RankAdaptive','GammaEQMOM','DQMOM','LogNormal')
LABELS=('QMOM','Rank-adaptive QMOM','Gamma EQMOM','DQMOM','Lognormal')
STYLES=('-', '--', '-.', ':', '-')

def save(fig,folder,name):
    fig.tight_layout()
    for ext in ('png','svg'):
        path=folder/f'{name}.{ext}'
        before=path.read_text().splitlines() if ext=='svg' and path.exists() else []
        fig.savefig(path,dpi=210,bbox_inches='tight')
        if ext=='svg':
            lines=[line.rstrip() for line in path.read_text().splitlines()]
            # Keep unchanged path formatting; new paths have no trailing spaces.
            matcher=SequenceMatcher(None,[line.rstrip() for line in before],lines,autojunk=False)
            for block in matcher.get_matching_blocks():
                lines[block.b:block.b+block.size]=before[block.a:block.a+block.size]
            path.write_text('\n'.join(lines)+'\n')
    plt.close(fig)

def main(folder=FOLDER,usetex=False):
    configure(usetex)
    data=np.genfromtxt(folder/'method_comparison.csv',names=True,delimiter=',',dtype=None,encoding='utf8')
    summary=np.genfromtxt(folder/'method_summary.csv',names=True,delimiter=',',dtype=None,encoding='utf8')
    ref=np.genfromtxt(folder/'sectional_reference.csv',names=True,delimiter=',')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for i,(method,label,ls) in enumerate(zip(METHODS,LABELS,STYLES)):
        scenario=data[(data['scenario']=='aggregation_breakage')&(data['method']==method)]
        for ax,k in zip(axes[0],(1,2)):
            q=scenario[scenario['order']==k]
            ax.semilogy(q['time'][1:],100*np.maximum(q['relative_error'][1:],1e-14),color=f'C{i}',ls=ls,label=label)
            ax.set(xlabel=r'$t\ (\mathrm{s})$',ylabel=rf'Relative $\mu_{k}$ error $(\%)$')
        q=scenario[scenario['order']==3]
        axes[1,0].plot(q['time'],q['moment']/q['reference'],color=f'C{i}',ls='--' if method=='LogNormal' else ls,
            label=label+(' (reconstructed)' if method=='LogNormal' else ''))
        r=summary[(summary['scenario']=='aggregation_breakage')&(summary['method']==method)][0]
        axes[1,1].bar(i,r['seconds'],color=f'C{i}',width=.6)
        axes[1,1].annotate(f"{r['states']} states",(i,r['seconds']),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axes[0,0].legend(fontsize=8)
    axes[1,0].axhline(1,color='black',lw=.8)
    axes[1,0].set(xlabel=r'$t\ (\mathrm{s})$',ylabel=r'$\mu_3(t)/V_0$')
    axes[1,0].legend(fontsize=7)
    axes[1,1].set(xticks=range(5),xticklabels=['QMOM','Adaptive','EQMOM','DQMOM','Lognormal'],yscale='log',ylabel=r'One measured run $(\mathrm{s})$')
    save(fig,folder,'method_aggregation')

    fig,axes=plt.subplots(2,3,figsize=(12,6.8))
    for row,scenario in enumerate(('growth_nucleation','growth_nucleation_collisions')):
        for col,k in enumerate((0,1,3)):
            ax=axes[row,col]
            for i,(method,label,ls) in enumerate(zip(METHODS,LABELS,STYLES)):
                q=data[(data['scenario']==scenario)&(data['method']==method)&(data['order']==k)]
                if scenario=='growth_nucleation':truth=q['reference']
                else:
                    r=ref[(ref['bins']==max(ref['bins']))&(ref['order']==k)]
                    truth=np.interp(q['time'],r['time'],r['moment'])
                # Plot-only floor; evolved moments and exported errors are untouched.
                err=np.maximum(abs(q['moment']/truth-1),1e-16)
                ax.semilogy(q['time'][1:],err[1:],ls=ls,color=f'C{i}',label=label+(' (inferred)' if method=='LogNormal' and k==3 else ''))
                status=summary[(summary['scenario']==scenario)&(summary['method']==method)][0]
                if status['status']!='completed' and len(q)>1:
                    ax.plot(q['time'][-1],err[-1],'x',ms=7,mew=1.5,color=f'C{i}')
            ax.set(xlabel=r'$t\ (\mathrm{s})$',ylabel=rf'Relative $\mu_{k}$ difference',ylim=(1e-12,1e3))
            if row==0:ax.set_title(rf'$\mu_{k}$')
    axes[0,0].text(.03,.95,'Growth and nucleation',transform=axes[0,0].transAxes,va='top',fontsize=9)
    axes[1,0].text(.03,.95,'With collisions',transform=axes[1,0].transAxes,va='top',fontsize=9)
    handles,labels=axes[0,2].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,ncol=5,loc='upper center',bbox_to_anchor=(.5,0))
    fig.suptitle('Analytical reference above; refined sectional comparison below; crosses mark incomplete runs',fontsize=10)
    save(fig,folder,'method_pulse')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=FOLDER);parser.add_argument('--usetex',action='store_true')
    args=parser.parse_args();main(args.data_dir,args.usetex)
