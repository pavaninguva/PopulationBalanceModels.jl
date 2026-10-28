"""Plot the Gaussian growth comparison in the same sequence as the manual."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER))
from _support import figures
from _support.figures import load_case, rows, table, export, analysis_table, gaussian_key
from plotting_style import panel,method_style,DOUBLE,METHODS


def panels():
    fig,axes=plt.subplots(1,2,figsize=(DOUBLE,3.65))
    fig.subplots_adjust(left=.105,right=.98,bottom=.30,top=.92,wspace=.34)
    for letter,ax in zip('ab',axes):
        panel(ax,letter)
    return fig,axes


def legend(fig,axis):
    fig.legend(*axis.get_legend_handles_labels(),loc='lower center',ncol=3,
               bbox_to_anchor=(.52,.005),fontsize=7.5)


def mesh_axes(axes,bins):
    for ax in axes:
        ax.set_xticks(bins,labels=[str(n) for n in bins])
        ax.xaxis.set_minor_locator(NullLocator())


def growth():
    base,info,runs=load_case(1)
    cfg=info['configuration']
    n,tf=cfg['profile_bins'],cfg['final_time']
    if figures.MODE=='production' and (n!=200 or tf!=4.):
        raise ValueError('The Gaussian manuscript profiles require 200 bins at t=4 s')
    summaries=[]
    for name,(folder,record) in runs.items():
        row=rows(folder/'metrics.csv')[0]
        row['dataset']=name
        row['api_time_method']=record['settings']['api_time_method']
        summaries.append(row)
    analysis_table(1,base,'growth_comparison.csv',summaries)
    info['display_conventions']=dict(
        profile='Gaussian on [0,1]; two Courant panels',
        density='bin integrals divided by actual widths; straight lines connect bin centres',
        errors='integrated L1 and width-weighted RMSE; no clipping',
        orders='successive smooth-periodic refinements; undefined at the recorded roundoff limit')

    fig,axes=panels()
    for ax,courant in zip(axes,cfg['profile_courants']):
        folder,record=runs[gaussian_key('upwind',n,courant)]
        exact=table(folder/'profile.csv')
        settings=record['settings']
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
            data=table(runs[gaussian_key(method,n,courant)][0]/'profile.csv')
            x=(data['left']+data['right'])/2
            style=method_style(method)
            style.update(markevery=(i*2,17),ms=2.8)
            ax.plot(x,data['density'],**style)
        ax.axhline(0,color='.75',lw=.5)
        ax.set(xlim=(0,1),xlabel=r'Size coordinate $a$',ylabel=r'$\bar f(a,4\,\mathrm{s})$')
        ax.text(.03,.95,rf'$C={courant:g},\ N={n}$',transform=ax.transAxes,
                ha='left',va='top',fontsize=8)
    limits=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_ylim(limits)
    axes[1].set_ylabel('')
    legend(fig,axes[1])
    export(fig,1,'growth_profiles',info)

    for profile,filename in (('gaussian','growth_gaussian_errors'),('smooth','growth_smooth_refinement')):
        fig,axes=panels()
        for method in METHODS:
            points=[]
            for bins in cfg['bins']:
                c=1. if method=='inguva2022' else cfg['courant']
                key=gaussian_key(method,bins,c) if profile=='gaussian' else f'smooth_{method}_N{bins}'
                points.append(rows(runs[key][0]/'metrics.csv')[0])
            for ax,norm in zip(axes,('l1','rmse')):
                error=np.asarray([float(row[norm]) for row in points])
                ax.loglog(cfg['bins'],np.where(error>0,error,np.nan),**method_style(method))
        axes[0].set(xlabel='Size bins',ylabel=r'$L^1$ density error')
        axes[1].set(xlabel='Size bins',ylabel='RMSE')
        mesh_axes(axes,cfg['bins']);legend(fig,axes[0])
        export(fig,1,filename,info)

    order=table(base/'observed_orders.csv')
    fig,axes=panels()
    for method in METHODS:
        data=order[order['method']==method]
        if not any(np.any(np.isfinite(data[key])) for key in ('observed_l1_order','observed_rmse_order')):
            continue
        for ax,column in zip(axes,('observed_l1_order','observed_rmse_order')):
            ax.semilogx(data['bins'],data[column],**method_style(method))
    axes[0].set(xlabel='Size bins',ylabel=r'$p_{\mathrm{obs}}$ ($L^1$)')
    axes[1].set(xlabel='Size bins',ylabel=r'$p_{\mathrm{obs}}$ (RMSE)')
    mesh_axes(axes,cfg['bins']);legend(fig,axes[0])
    export(fig,1,'growth_observed_orders',info)

    fig,axes=panels()
    for method in METHODS:
        points=[rows(runs[f'smooth_{method}_N{bins}'][0]/'metrics.csv')[0] for bins in cfg['bins']]
        seconds=np.asarray([float(row['seconds']) for row in points])
        rms=np.asarray([float(row['rmse']) for row in points])
        axes[0].loglog(cfg['bins'],seconds,**method_style(method))
        axes[1].loglog(seconds,rms,**method_style(method))
    axes[0].set(xlabel='Size bins',ylabel=r'Integration wall time (s)')
    axes[1].set(xlabel=r'Integration wall time (s)',ylabel='RMSE')
    mesh_axes([axes[0]],cfg['bins']);legend(fig,axes[0])
    export(fig,1,'growth_work_precision',info)


if __name__=='__main__':
    figures.run_plot(1,growth)
