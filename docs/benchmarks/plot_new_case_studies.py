"""Export paper-data comparisons and analytic verification; no grid lines."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from plot_style import configure
ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'docs/src/assets'

def save(fig,path):
    fig.tight_layout()
    for ext in ('png','svg'):
        target = path.with_suffix('.'+ext)
        fig.savefig(target,dpi=210,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        if ext=='svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
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
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout='constrained')
    vmax=np.max(d['number_density'])
    for ax,t in zip(axes.ravel(),[0,40,80,200]):
        q=d[d['time']==t];z=np.unique(q['z_m']);x=np.unique(q['size_m'])*1e6
        # Input is axial-major; transpose to size × axial position.
        values=q['number_density'].reshape(len(z),len(x)).T
        ze=np.r_[np.unique(q['z_left']),np.max(q['z_right'])]
        xe=np.r_[np.unique(q['x_left']),np.max(q['x_right'])]*1e6
        im=ax.pcolormesh(ze,xe,values,shading='flat',cmap='magma',vmin=0,vmax=vmax,rasterized=True)
        ax.set(xlabel=r'$z\ (\mathrm{m})$',title=rf'$t={t}\ \mathrm{{s}}$',yscale='log',ylim=(1,900))
    for ax in axes[:,0]:ax.set_ylabel(r'$x\ (\mu\mathrm{m})$')
    fig.colorbar(im,ax=axes.ravel().tolist(),label=r'$n\ (\mathrm{m}^{-4})$',pad=.03)
    for ext in ('png','svg'):
        target = (folder/'fields').with_suffix('.'+ext)
        fig.savefig(target,dpi=210,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        if ext=='svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    d=np.genfromtxt(folder/'outlets.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    # Select fixed-size / varying-axial slices, rather than concatenating meshes.
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for nz in np.unique(d['nz']):
        q=d[(d['mesh']=='uniform')&(d['size_method']=='muscl')&(d['axial_method']=='muscl')
            &(d['nx']==max(d['nx']))&(d['nz']==nz)]
        for ax,key,scale in zip(axes,['number_density','volume_density'],[1.,1e-6]):
            ax.plot(q['size_m']*1e6,q[key]*scale,label=rf'$N_x={int(max(d["nx"]))},\ N_z={int(nz)}$')
    reference=np.genfromtxt(folder/'field.csv',delimiter=',',names=True)
    q=reference[(reference['time']==200)&(reference['z_m']==max(reference['z_m']))]
    volume=np.sum(q['number_density']*(q['x_right']-q['x_left'])*q['size_m']**3)
    axes[0].plot(q['size_m']*1e6,q['number_density'],'k--',label='Numerical reference')
    axes[1].plot(q['size_m']*1e6,q['number_density']*q['size_m']**3/volume*1e-6,'k--')
    for ax in axes:ax.set(xlabel=r'$x\ (\mu\mathrm{m})$',xscale='log')
    axes[0].set_ylabel(r'$n(x,L,200)\ (\mathrm{m}^{-4})$');axes[0].legend(fontsize=7)
    axes[1].set_ylabel(r'Volume fraction density $(\mu\mathrm{m}^{-1})$');save(fig,folder/'outlets')

    d=np.genfromtxt(folder/'performance.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    q=d[d['mesh']=='uniform']
    max_nx,max_nz=max(q['nx']),max(q['nz'])
    # Connect only one-coordinate refinement paths, rather than a mixed grid
    # Cartesian product or a visually misleading all-point performance line.
    for filename,work in [('coordinate_refinement',False),('performance',True)]:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for ax,variable,fixed,fixed_value in (
            (axes[0],'nx','nz',max_nz),(axes[1],'nz','nx',max_nx)):
            for color,name in zip(('C0','C1','C2'),('upwind','muscl','weno5')):
                r=q[(q['size_method']==name)&(q[fixed]==fixed_value)]
                r=r[np.argsort(r[variable])]
                spacing=(899/r['nx'] if variable=='nx' else .47/r['nz'])
                x=r['seconds'] if work else spacing
                ax.loglog(x,r['normalized_L1'],'o-',ms=4,color=color,label=name.upper())
                if work:
                    for xx,yy,n in zip(x,r['normalized_L1'],r[variable]):
                        ax.annotate(str(int(n)),(xx,yy),xytext=(3,4),textcoords='offset points',fontsize=6,color=color)
            ax.set(ylabel='Normalized outlet $L^1$ error',
                   title=(rf'Size refinement, $N_z={int(max_nz)}$' if variable=='nx'
                          else rf'Axial refinement, $N_x={int(max_nx)}$'))
            if work:ax.set_xlabel(r'Elapsed time $(\mathrm{s})$')
            else:
                ax.set_xlabel(r'$\Delta x\ (\mu\mathrm{m})$' if variable=='nx' else r'$\Delta z\ (\mathrm{m})$')
                ax.set_xticks(spacing,[f'{value:.3g}' for value in spacing])
                ax.xaxis.set_minor_locator(NullLocator())
                ax.invert_xaxis()
            ax.legend(fontsize=8,title='Size scheme; axial MUSCL',title_fontsize=8)
        save(fig,folder/filename)


def pareto_mask(work,error):
    """Keep points for which no measured point is cheaper AND at least as accurate.

    Equal duplicates may both be retained; a strict improvement in either
    objective is needed for domination. Fronts are computed within each pairing.
    """
    work,error=np.asarray(work),np.asarray(error)
    return np.array([not np.any((work<=w)&(error<=e)&((work<w)|(error<e)))
                     for w,e in zip(work,error)])

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
