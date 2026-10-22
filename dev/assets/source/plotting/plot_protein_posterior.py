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
    for ext in ('png','svg'):
        path = folder/f'{name}.{ext}'
        fig.savefig(path,dpi=210,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        if ext=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)

SUPPORT = ((0.,100.), (1.,10.), (0.,np.inf), (1.,3.))

def marginal_density(values, support, points=400):
    """Gaussian KDE with reflection at finite support endpoints (display only)."""
    values = np.asarray(values)
    h = 1.06*np.std(values,ddof=1)*len(values)**(-.2)
    lower, upper = support
    x = np.linspace(max(lower,values.min()-4*h),min(upper,values.max()+4*h),points)
    density = np.zeros_like(x)
    for centers in (values, 2*lower-values if np.isfinite(lower) else [],
                    2*upper-values if np.isfinite(upper) else []):
        if len(centers):
            density += np.exp(-.5*((x[:,None]-centers)/h)**2).sum(axis=1)
    density /= len(values)*h*np.sqrt(2*np.pi)
    return x,density

def probability_levels(density, probabilities=(.95,.68)):
    """Highest-density thresholds by integrated grid mass, not peak fractions.

    The affine grid used below has constant cell area, which cancels here.
    """
    ranked = np.sort(density[density>0])[::-1]
    mass = np.cumsum(ranked)/ranked.sum()
    return np.sort([ranked[min(np.searchsorted(mass,p),len(ranked)-1)] for p in probabilities])

def joint_density(x,y,xsupport,ysupport,points=181):
    """KDE on a sheared/standardized grid resolving a thin correlation ridge.

    The shear has determinant one; scaling has a constant Jacobian. Masking
    outside declared support leaves contours inside the parameter domain.
    This is a summary of the saved draws, not an inference approximation.
    """
    slope = np.cov(x,y)[0,1]/np.var(x,ddof=1)
    residual = y-y.mean()-slope*(x-x.mean())
    scales = np.array([x.std(ddof=1),residual.std(ddof=1)])
    centers = np.column_stack(((x-x.mean())/scales[0],residual/scales[1]))
    h = len(x)**(-1/6)
    u = np.linspace(centers[:,0].min()-4*h,centers[:,0].max()+4*h,points)
    v = np.linspace(centers[:,1].min()-4*h,centers[:,1].max()+4*h,points)
    U,V = np.meshgrid(u,v)
    X = x.mean()+scales[0]*U
    Y = y.mean()+slope*(X-x.mean())+scales[1]*V
    density = np.zeros(U.size)
    query = np.column_stack((U.ravel(),V.ravel()))
    for start in range(0,len(query),256):
        delta = (query[start:start+256,None,:]-centers[None,:,:])/h
        density[start:start+256] = np.exp(-.5*np.sum(delta*delta,axis=2)).sum(axis=1)
    density = density.reshape(U.shape)/(len(x)*2*np.pi*h*h*np.prod(scales))
    inside = (X>=xsupport[0])&(X<=xsupport[1])&(Y>=ysupport[0])&(Y<=ysupport[1])
    density[~inside] = 0.
    return X,Y,density

def kinetic_tradeoffs(folder,draws):
    logref = np.log(8.)
    etaJ = draws['logAJ']+draws['j']*logref
    etaG = np.log(draws['AG'])+draws['g']*logref
    prediction = read(folder,'predictions.csv')
    q = prediction[(prediction['method']=='MAP')&(prediction['time_min']>0)]
    supersaturation = q['concentration_g_L']/2.16-1
    smin,smax = max(1e-4,supersaturation.min()),supersaturation.max()
    fig,axes = plt.subplots(2,2,figsize=(10,7))
    for ax,x,y,xlabel,ylabel in (
        (axes[0,0],draws['logAJ'],draws['j'],LABELS[0],LABELS[1]),
        (axes[0,1],etaJ,draws['j'],r'$\eta_J=\log J(8)$',LABELS[1])):
        X,Y,density = joint_density(x,y,(-np.inf,np.inf),(1.,10.))
        ax.contourf(X,Y,density,levels=[*probability_levels(density),density.max()*1.001],
                    colors=['#a6cbe3','#377eb8'],alpha=.85)
        ax.set(xlabel=xlabel,ylabel=ylabel)
    ss = np.geomspace(max(.01,smin/2),max(16,2*smax),350)
    for ax,logcoef,exponent,label in (
        (axes[1,0],draws['logAJ'],draws['j'],r'$J(s)\ (\mathrm{m}^{-3}\,\mathrm{s}^{-1})$'),
        (axes[1,1],np.log(draws['AG']),draws['g'],r'$G(s)\ (\mathrm{nm}\,\mathrm{min}^{-1})$')):
        rates = np.exp(logcoef[:,None]+exponent[:,None]*np.log(ss))
        lo,mid,hi = np.quantile(rates,[.025,.5,.975],axis=0)
        ax.axvspan(smin,smax,color='0.9',label='MAP trajectory range (saved times)')
        ax.fill_between(ss,lo,hi,color='C0',alpha=.3,label=r'95\% pointwise NUTS interval')
        ax.plot(ss,mid,color='C0');ax.axvline(8.,color='black',ls='--',lw=.9,label=r'$s_*=8$')
        ax.set(xlabel=r'Supersaturation $s=c/c_{\mathrm{eq}}-1$',ylabel=label,xscale='log',yscale='log')
    axes[1,0].legend(fontsize=7)
    save(fig,folder,'rate_identifiability')
    rows = [('corr_logAJ_j',np.corrcoef(draws['logAJ'],draws['j'])[0,1]),
            ('corr_etaJ_j',np.corrcoef(etaJ,draws['j'])[0,1]),
            ('corr_logAG_g',np.corrcoef(np.log(draws['AG']),draws['g'])[0,1]),
            ('corr_etaG_g',np.corrcoef(etaG,draws['g'])[0,1]),
            ('MAP_saved_supersaturation_min',smin),('MAP_saved_supersaturation_max',smax)]
    for name in NAMES:
        rows += [(name+'_q'+str(p),value) for p,value in zip((.025,.5,.975),np.quantile(draws[name],[.025,.5,.975]))]
    with (folder/'posterior_summaries.csv').open('w') as io:
        io.write('quantity,value\n')
        for name,value in rows:io.write(f'{name},{value:.17g}\n')

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
                x,density = marginal_density(draws[name],SUPPORT[i])
                ax.fill_between(x,density,color='C0',alpha=.3)
                ax.plot(x,density,color='C0')
                ax.set_yticks([])
            else:
                X,Y,density = joint_density(draws[other],draws[name],SUPPORT[j],SUPPORT[i])
                levels = probability_levels(density)
                ax.contourf(X,Y,density,levels=[*levels,density.max()*1.001],colors=['#a6cbe3','#377eb8'],alpha=.85)
                ax.contour(X,Y,density,levels=levels,colors='C0',linewidths=.7)
            if i==3:ax.set_xlabel(LABELS[j])
            else:ax.tick_params(labelbottom=False)
            if j==0 and i>0:ax.set_ylabel(LABELS[i])
            elif i!=j:ax.tick_params(labelleft=False)
    fig.suptitle('NUTS draws: smoothed marginals; 68\\% and 95\\% joint density regions. '+status,fontsize=11)
    save(fig,folder,'posterior_corner')

    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for i,(name,label) in enumerate(zip(NAMES,LABELS)):
        q=priors[priors['parameter']==name]
        x=draws[name];lo,hi=np.quantile(x,[.005,.995]);width=max(hi-lo,1e-10)
        axes[0,i].plot(q['value'],q['density'],color='C1',label='Declared prior')
        axes[0,i].axvspan(lo,hi,color='C0',alpha=.25,label=r'Posterior 99\% span')
        if name=='AG':axes[0,i].set_xscale('log')
        axes[0,i].set(xlabel=label,ylabel='Prior density',ylim=(0,1.18*max(q['density'])))
        ax=axes[1,i]
        xxpost,density = marginal_density(x,SUPPORT[i])
        ax.fill_between(xxpost,density,alpha=.3,color='C0')
        ax.plot(xxpost,density,color='C0',label='Smoothed NUTS draws')
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

    kinetic_tradeoffs(folder,draws)

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
