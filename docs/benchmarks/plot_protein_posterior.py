"""Protein posterior plots from exported samples, with convergence status visible."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'docs/src/assets/protein_crystallization'
NAMES=('logAJ','j','AG','g')
LABELS=(r'$\log A_J$',r'$j$',r'$A_G\ (\mathrm{nm\,min}^{-1})$',r'$g$')

def read(folder,name):return np.genfromtxt(folder/name,delimiter=',',names=True,dtype=None,encoding='utf8')
def passed(d):return all(str(v).lower()=='true' for v in np.atleast_1d(d['passed']))
def save(fig,folder,name):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(folder/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)

def main(folder=DEFAULT,usetex=False):
    configure(usetex)
    draws=read(folder,'posterior_draws.csv');priors=read(folder,'prior_density.csv')
    diag=read(folder,'sampler_diagnostics.csv');ok=passed(diag)
    status='Sampler diagnostic screen passed' if ok else 'Provisional: sampler diagnostics failed'
    fig,axes=plt.subplots(4,4,figsize=(10,9))
    for i,name in enumerate(NAMES):
        for j,other in enumerate(NAMES):
            ax=axes[i,j]
            if j>i:ax.axis('off');continue
            if i==j:
                ax.hist(draws[name],bins=35,density=True,color='C0',alpha=.7)
                ax.set_yticks([])
            else:
                ax.scatter(draws[other],draws[name],s=2,alpha=.15,rasterized=True,color='C0',edgecolors='none')
            if i==3:ax.set_xlabel(LABELS[j])
            else:ax.tick_params(labelbottom=False)
            if j==0 and i>0:ax.set_ylabel(LABELS[i])
            elif i!=j:ax.tick_params(labelleft=False)
    fig.suptitle('Protein kinetic posterior: '+status,fontsize=11)
    save(fig,folder,'posterior_corner')

    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for i,(name,label) in enumerate(zip(NAMES,LABELS)):
        q=priors[priors['parameter']==name]
        x=draws[name];lo,hi=np.quantile(x,[.005,.995]);width=max(hi-lo,1e-10)
        axes[0,i].plot(q['value'],q['density'],color='C1',label='Declared prior')
        axes[0,i].axvspan(lo,hi,color='C0',alpha=.25,label=r'Posterior 99\% span')
        if name=='AG':axes[0,i].set_xscale('log')
        axes[0,i].set(xlabel=label,ylabel='Prior density',ylim=(0,1.18*max(q['density'])))
        ax=axes[1,i];ax.hist(x,bins=35,density=True,alpha=.65,color='C0',label='Posterior draws')
        xx=np.linspace(max(q['value'][0],lo-.12*width),min(q['value'][-1],hi+.12*width),500)
        ax.plot(xx,np.interp(xx,q['value'],q['density']),color='C1',label='Unscaled prior density')
        ax.set(xlabel=label,ylabel='Density (posterior range)')
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('Prior scale above; posterior detail below. '+status,fontsize=10)
    save(fig,folder,'prior_posterior')

    fig,axes=plt.subplots(4,2,figsize=(11,8.4))
    chain_ids=np.unique(draws['chain'])
    for i,(name,label) in enumerate(zip(NAMES,LABELS)):
        pooled=draws[name];ranks=np.empty(len(pooled),dtype=int);ranks[np.argsort(pooled)]=np.arange(len(pooled))
        for c in chain_ids:
            mask=draws['chain']==c;q=draws[mask];order=np.argsort(q['draw'])
            axes[i,0].plot(q['draw'][order],q[name][order],lw=.45,alpha=.75,label=f'Chain {c}')
            axes[i,1].hist(ranks[mask],bins=np.linspace(0,len(pooled),13),histtype='step',lw=1.,label=f'Chain {c}')
        d=diag[diag['parameter']==name][0]
        axes[i,0].set(ylabel=label)
        axes[i,1].set_title(rf'$\widehat R={d["rhat"]:.3f}$; bulk ESS $={d["ess_bulk"]:.0f}$; tail ESS $={d["ess_tail"]:.0f}$',fontsize=9)
    axes[-1,0].set_xlabel('Retained draw');axes[-1,1].set_xlabel('Pooled rank')
    axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle(status,fontsize=11)
    save(fig,folder,'posterior_chains')

    path=folder/'posterior_predictive.csv'
    if path.exists():
        d=read(folder,'posterior_predictive.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for ax,obs,label in zip(axes,('concentration','D50'),(r'$c\ (\mathrm{g\,L}^{-1})$',r'$D_{50}\ (\mu\mathrm{m})$')):
            q=d[d['observable']==obs];times=np.unique(q['time_min'])
            latent=np.array([np.quantile(q['latent'][q['time_min']==t],[.025,.5,.975]) for t in times])
            noisy=np.array([np.quantile(q['observed'][q['time_min']==t],[.025,.975]) for t in times])
            y=np.array([q['data'][q['time_min']==t][0] for t in times])
            ax.fill_between(times,noisy[:,0],noisy[:,1],color='C0',alpha=.15,label=r'95\% predictive (with noise)')
            ax.fill_between(times,latent[:,0],latent[:,2],color='C0',alpha=.4,label=r'95\% latent interval')
            ax.plot(times,latent[:,1],color='C0',lw=1);ax.plot(times,y,'o',ms=3,color='black',label='Digitized data')
            ax.set(xlabel=r'$t\ (\mathrm{min})$',ylabel=label)
        axes[0].legend(fontsize=7);fig.suptitle('Conditional predictive check: '+status,fontsize=10)
        save(fig,folder,'posterior_predictive')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=DEFAULT);parser.add_argument('--usetex',action='store_true')
    args=parser.parse_args();main(args.data_dir,args.usetex)
