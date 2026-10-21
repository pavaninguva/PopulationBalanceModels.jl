"""Plot literal Li S11 audit data; failed runs remain in the accompanying CSV."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure
from plot_case_studies import save_figure
p=argparse.ArgumentParser(__doc__)
p.add_argument('--usetex',action='store_true')
args=p.parse_args();configure(usetex=args.usetex)
folder=Path(__file__).resolve().parents[1]/'src/assets/li_supplement'
d=np.genfromtxt(folder/'errors.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for kind,label in [('constant','Constant'),('affine','Affine'),('exponential','Exponential')]:
    for dimension,ax in [(1,axes[0]),(2,axes[1])]:
        r=d[(d['dimension']==dimension)&(d['growth']==kind)&(d['courant']==.4)]
        ax.loglog(r['intervals'],r['rmse'],'o-',ms=3,label=label+r', $C_{\max}=0.4$')
    r=d[(d['dimension']==1)&(d['growth']==kind)&(d['courant']==.8)]
    if kind!='constant':axes[0].loglog(r['intervals'],r['rmse'],'s--',ms=3,label=label+r', $C_{\max}=0.8$')
c=np.genfromtxt(folder/'consistent_variant.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
for kind,label in [('affine','Affine'),('exponential','Exponential')]:
    r=c[(c['dimension']==1)&(c['growth']==kind)]
    axes[0].loglog(r['intervals'],r['rmse'],'x-.',ms=4,label=label+', corrected variant')
x=np.array([400.,3200.]);axes[0].loglog(x,1e-6*(x/400)**-2,'k:',label=r'$h^2$ guide')
x=np.array([100.,400.]);axes[1].loglog(x,5e-5*(x/100)**-4,'k:',label=r'$h^4$ guide')
for ax,title in zip(axes,['1D: fine meshes expose the order change','2D: product of directional updates']):
    ax.set(xlabel='Intervals per coordinate',ylabel='Nodal density RMSE',title=title)
    ax.legend(fontsize=7)
save_figure(fig,folder,'refinement')
