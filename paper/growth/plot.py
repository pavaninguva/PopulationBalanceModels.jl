"""Plot the two manuscript growth figures from retained numerical exports."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER))
from _support import figures
from _support.figures import export, analysis_table
from _support.growth_results import load_results, refinement, profile
from plotting_style import panel,method_style,DOUBLE,METHODS
from pbm_plotting import plot_distribution


def panels():
    fig,axes=plt.subplots(1,2,figsize=(DOUBLE,3.25))
    fig.subplots_adjust(left=.105,right=.98,bottom=.18,top=.92,wspace=.34)
    for letter,ax in zip('ab',axes):
        panel(ax,letter)
    return fig,axes


def mesh_axes(axes,bins):
    for ax in axes:
        ax.set_xticks(bins,labels=[str(n) for n in bins])
        ax.xaxis.set_minor_locator(NullLocator())


def growth(input_path=None):
    if input_path is None and figures.MODE=='smoke':
        input_path=figures.ROOT/'.cache/paper/checks/growth'
    data=load_results(input_path)
    cfg=data['configuration']
    producer=data['provenance']
    info=dict(configuration=cfg,source_digest=producer['source_digest'],
              workflow_digest=producer['workflow_digest'],checkout_verified=False,retained_redraw=True)
    figures.INPUTS[1]=data.get('selection_files',{'original_index':producer.get('index_sha256','new calculation')})
    n,tf=cfg['profile_bins'],cfg['final_time']
    if figures.MODE=='production' and (n!=200 or tf!=4.):
        raise ValueError('The Gaussian manuscript profiles require 200 bins at t=4 s')
    if figures.TABLES:
        base=Path(input_path) if input_path is not None else figures.ROOT/'docs/src/assets/constant_growth'
        analysis_table(1,base,'growth_comparison.csv',
            [{name:row[name].item() for name in data['metrics'].dtype.names} for row in data['metrics']])
    info['display_conventions']=dict(
        profile='Gaussian on [0,1]; two Courant panels',
        density='bin integrals divided by actual widths; straight lines connect bin centres',
        errors='integrated L1 and width-weighted RMSE; no clipping',
        orders='successive Gaussian refinements; undefined at a density-scaled floating-point screen',
        benchmark='the same Gaussian, zero inflow/open outflow, G=0.1 and t=4 in both figures',
        timing_producer=producer['environment'],timing_protocol=producer['timing_protocol'])

    fig,axes=panels()
    for ax,courant in zip(axes,cfg['profile_courants']):
        exact=profile(data,'upwind',courant)
        settings=data['benchmark']
        a=np.linspace(0,1,4001)
        initial_a=a-settings['growth']*tf
        truth=np.where((initial_a>=0)&(initial_a<=1),
            settings['height']*np.exp(-.5*((initial_a-settings['center'])/settings['standard_deviation'])**2),0.)
        ax.plot(a,truth,color='.5',lw=1.2,label='Analytical density')
        centers=(exact['left']+exact['right'])/2
        ax.plot(centers,exact['exact_density'],'k.',ms=2,label='Analytical bin averages')
        for i,method in enumerate(METHODS):
            if method=='inguva2022' and courant!=1.:
                continue
            values=profile(data,method,courant)
            x=(values['left']+values['right'])/2
            style=method_style(method)
            style.update(markevery=(i*2,17),ms=2.8)
            edges=np.r_[values['left'],values['right'][-1]]
            plot_distribution(values['density'],edges,ax=ax,curve='centers',styles=[style],legend=False)
        ax.axhline(0,color='.75',lw=.5)
        ax.set(xlim=(0,1),xlabel=r'Size coordinate $a$',ylabel=r'$n(a,4\,\mathrm{s})$')
        ax.text(.97,1.035,rf'$\mathrm{{CFL}}={courant:g},\ N={n}$',transform=ax.transAxes,
                ha='right',va='bottom',fontsize=8)
    limits=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_ylim(limits)
    axes[1].set_ylabel('')
    axes[1].legend(loc='upper left',fontsize=6.5,handlelength=2.2,
                   borderaxespad=.4,labelspacing=.35)
    export(fig,1,'growth_profiles',info)

    fig,axes=panels()
    for method in METHODS:
        points=refinement(data,method)
        seconds=np.asarray([float(row['seconds']) for row in points])
        rms=np.asarray([float(row['rmse']) for row in points])
        axes[0].loglog(cfg['bins'],seconds,**method_style(method))
        axes[1].loglog(seconds,rms,**method_style(method))
    axes[0].set(xlabel='Number of bins',ylabel=r'Integration wall time (s)')
    axes[1].set(xlabel=r'Integration wall time (s)',ylabel='RMSE')
    mesh_axes([axes[0]],cfg['bins'])
    # Leave space above the measured curves for an in-panel legend.
    axes[0].set_ylim(top=1.)
    axes[0].legend(loc='upper left',fontsize=6.5,handlelength=2.2,
                   borderaxespad=.4,labelspacing=.35)
    export(fig,1,'growth_work_precision',info)


if __name__=='__main__':
    figures.run_plot(1,growth)
