"""Plot the native dissolution verification. Run after realizable_dissolution.jl."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'docs/src/assets/dissolution'

def reference(k, t):
    return (np.maximum(2-t, 0)**(k+1)-np.maximum(1-t, 0)**(k+1))/(k+1)

def main(usetex=False):
    configure(usetex)
    data = np.genfromtxt(FOLDER/'dissolution.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.1))
    time = np.linspace(0, 2.2, 700)
    selections = [('sectional',160,'Upwind, 160 bins','C0'), ('moments',2,'Adaptive QMOM, 2 nodes','C1'),
                  ('moments',6,'Adaptive QMOM, 6 nodes','C3')]
    for ax, key, label, k in zip(axes.flat[:4], ['M0','M1','M2','M3'],[r'$M_0$',r'$M_1$',r'$M_2$',r'$M_3$'],range(4)):
        ax.plot(time, reference(k,time), color='black', lw=1.4, label='Analytical')
        for method,n,legend,color in selections:
            d=data[(data['method']==method)&(data['resolution']==n)]
            ax.plot(d['time'],d[key],color=color,lw=1.1,label=legend)
        ax.set(xlabel=r'$t$',ylabel=label)
    axes[0,0].legend(fontsize=8)
    axes[1,1].plot(time,1+.7*(reference(3,0)-reference(3,time)),color='black',lw=1.4)
    for method,n,legend,color in selections:
        d=data[(data['method']==method)&(data['resolution']==n)]
        axes[1,1].plot(d['time'],d['solute'],color=color,lw=1.1)
        axes[1,2].plot(d['time'],d['material_error'],color=color,lw=1.1)
    axes[1,1].set(xlabel=r'$t$',ylabel=r'$c$')
    axes[1,2].set(xlabel=r'$t$',ylabel=r'$c+\rho k_v M_3-C_{\mathrm{initial}}$')
    axes[1,2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.tight_layout()
    for extension in ('png','svg'):
        fig.savefig(FOLDER/f'dissolution.{extension}',dpi=210,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--usetex',action='store_true')
    main(parser.parse_args().usetex)
