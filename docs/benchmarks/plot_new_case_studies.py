"""Export paper-data comparisons and analytic verification; no grid lines."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure
ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'docs/src/assets'

def save(fig,path):
    fig.tight_layout()
    for ext in ('png','svg'): fig.savefig(path.with_suffix('.'+ext),dpi=210,bbox_inches='tight')
    plt.close(fig)

def moments():
    folder=ASSETS/'quadrature_moments';d=np.genfromtxt(folder/'density.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(5.4,3.3))
    ax.plot(d['size'],d['exact'],color='black',label='Analytical')
    ax.plot(d['size'][::8],d['density'][::8],'o',ms=3,mfc='none',label='Gamma EQMOM')
    ax.set(xlabel=r'$x$',ylabel=r'$n(x,2)$');ax.legend();save(fig,folder/'density')

def precipitation():
    folder=ASSETS/'continuous_precipitation'
    d=np.genfromtxt(folder/'field.csv',delimiter=',',names=True)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True,layout='constrained')
    vmax=np.max(d['number_density'])
    for ax,t in zip(axes,[80,200]):
        q=d[d['time']==t];z=np.unique(q['z_m']);x=np.unique(q['size_m'])*1e6
        # Input is axial-major; transpose to size × axial position.
        values=q['number_density'].reshape(len(z),len(x)).T
        ze=np.r_[np.unique(q['z_left']),np.max(q['z_right'])]
        xe=np.r_[np.unique(q['x_left']),np.max(q['x_right'])]*1e6
        im=ax.pcolormesh(ze,xe,values,shading='flat',cmap='magma',vmin=0,vmax=vmax,rasterized=True)
        ax.set(xlabel=r'$z\ (\mathrm{m})$',title=rf'$t={t}\ \mathrm{{s}}$',yscale='log',ylim=(1,900))
    axes[0].set_ylabel(r'$x\ (\mu\mathrm{m})$')
    fig.colorbar(im,ax=axes.ravel().tolist(),label=r'$n\ (\mathrm{m}^{-4})$',pad=.03)
    for ext in ('png','svg'):fig.savefig((folder/'fields').with_suffix('.'+ext),dpi=210,bbox_inches='tight')
    plt.close(fig)
    d=np.genfromtxt(folder/'outlets.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for method in np.unique(d['method']):
        for nx in np.unique(d['nx']):
            q=d[(d['method']==method)&(d['nx']==nx)]
            axes[0].plot(q['size_m']*1e6,q['number_density'],label=f'{method}, {nx} bins')
            # Convert the normalized volume density from per metre to per micrometre.
            axes[1].plot(q['size_m']*1e6,q['volume_density']*1e-6)
    for ax in axes:ax.set(xlabel=r'$x\ (\mu\mathrm{m})$',xscale='log')
    axes[0].set_ylabel(r'$n(x,L,200)\ (\mathrm{m}^{-4})$');axes[0].legend(fontsize=7)
    axes[1].set_ylabel(r'Volume fraction density $(\mu\mathrm{m}^{-1})$');save(fig,folder/'outlets')
    d=np.genfromtxt(folder/'performance.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    fig,ax=plt.subplots(figsize=(5.4,3.5))
    for method in np.unique(d['method']):
        q=d[d['method']==method];ax.loglog(q['seconds'],q['normalized_L1'],'o-',label=method)
        for v in q:ax.annotate(f"{v['nx']}×{v['nz']}",(v['seconds'],v['normalized_L1']),xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set(xlabel=r'Elapsed time $(\mathrm{s})$',ylabel='Normalized outlet $L^1$ error');ax.legend();save(fig,folder/'performance')

def protein():
    folder=ASSETS/'protein_crystallization';d=np.genfromtxt(folder/'predictions.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    obs=np.genfromtxt(ROOT/'examples/data/protein_crystallization/figure4_observations.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for ax,observable,ycolumn,label in zip(axes,['concentration','D50'],['concentration_g_L','D50_um'],[r'$c\ (\mathrm{g\,L}^{-1})$',r'$D_{50}\ (\mu\mathrm{m})$']):
        q=obs[obs['observable']==observable]
        ax.errorbar(q['time_min'],q['value'],yerr=q['assumed_sigma'],fmt='o',ms=3,color='black',capsize=2,label='Graph-read experiment')
        for method in ['reported_parameters','MAP','MAP_refined']:
            q=d[d['method']==method];ax.plot(q['time_min'],q[ycolumn],label=method.replace('_',' '))
        ax.set(xlabel=r'$t\ (\mathrm{min})$',ylabel=label)
    axes[1].set_ylim(0,14)
    axes[0].legend(fontsize=7);save(fig,folder/'fit')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['moments','precipitation','protein','all']);p.add_argument('--usetex',action='store_true');a=p.parse_args();configure(a.usetex)
    for name,fun in [('moments',moments),('precipitation',precipitation),('protein',protein)]:
        if a.case in (name,'all'):fun()
