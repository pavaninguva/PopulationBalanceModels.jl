"""Long-horizon reconstructed LNP model; no experimental curves or fitted replacements."""
from pathlib import Path
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure
from plot_case_studies import save_figure

def read(p):return np.genfromtxt(p,delimiter=',',names=True)
def run_label(folder):
    text=(folder/'frr3/inputs.txt').read_text()
    match=re.search(r'bins=(\d+); reltol=([\deE.+-]+)',text)
    return f'{match[1]} bins, relative tolerance {match[2]}' if match else 'Reconstruction'

def plots(folder,comparison=None):
    histories={f:read(folder/f'frr{f}/history.csv') for f in (1,3,5)}
    d=histories[3]
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    valid=(d['time']>0)&np.isfinite(d['intensity_mean_m'])&(d['intensity_mean_m']>0)
    ax.semilogx(d['time'][valid]/60,1e9*d['intensity_mean_m'][valid],label=run_label(folder))
    if comparison:
        c=read(comparison/'frr3/history.csv');valid=(c['time']>0)&np.isfinite(c['intensity_mean_m'])&(c['intensity_mean_m']>0)
        ax.semilogx(c['time'][valid]/60,1e9*c['intensity_mean_m'][valid],'--',label=run_label(comparison))
    initial=d['free_lipid'][0]
    flagged=d['time'][d['escaped_lipid']>1e-3*initial]
    if len(flagged):ax.axvspan(flagged[0]/60,4320,color='.9',label=r'More than 0.1\% mass escaped')
    ax.set(xlabel=r'$t\,/\,\mathrm{min}$',ylabel=r'$M_7/M_6\,/\,\mathrm{nm}$',
           title='Reconstructed inputs: agreement and convergence not established',xlim=(1e-6,4320))
    ax.legend(fontsize=8);save_figure(fig,folder,'figure4b_model')

    psd=read(folder/'frr3/psd.csv')
    fig,axes=plt.subplots(1,5,figsize=(13,3.1),layout='constrained')
    for ax,minutes in zip(axes,(1,40,100,170,4320)):
        r=psd[np.isclose(psd['time'],60*minutes,rtol=1e-10,atol=1e-10)]
        ax.semilogx(r['diameter_m']*1e9,r['intensity_peak_scaled'])
        ax.set(xlabel=r'$L\,/\,\mathrm{nm}$',title=fr'$t={minutes}\,\mathrm{{min}}$',ylim=(-.02,1.05))
    axes[0].set_ylabel('Intensity density / peak')
    fig.suptitle('Reconstructed inputs; inspect the mass-loss diagnostic before interpretation')
    save_figure(fig,folder,'figure4a_model')

    fig,axes=plt.subplots(2,4,figsize=(13,6.3),layout='constrained')
    for f,d in histories.items():
        r=d[(d['time']<=.15)&(d['time']>=0)]
        label=fr'$\mathrm{{FRR}}={f}$'
        for ax,key in zip(axes.flat[:4],['free_lipid','supersaturation','nucleation_rate','number_mean_growth_m_s']):
            ax.plot(r['time'],r[key]*(1e9 if key=='number_mean_growth_m_s' else 1),label=label)
        axes[1,0].plot(r['time'],r['nucleation_uptake_kg_m3_s'])
        axes[1,1].plot(r['time'],r['growth_uptake_kg_m3_s'])
        valid=(r['nucleated_lipid']>0)&(r['grown_lipid']>0)
        axes[1,2].plot(r['time'][valid],np.log10(r['grown_lipid'][valid]/r['nucleated_lipid'][valid]))
        psd=read(folder/f'frr{f}/psd.csv');p=psd[np.isclose(psd['time'],.15,rtol=1e-10,atol=1e-10)]
        axes[1,3].semilogx(p['diameter_m']*1e9,p['intensity_peak_scaled'])
    labels=[r'$c\,/\,(\mathrm{kg\,m^{-3}})$',r'$S$',r'$J\,/\,(\mathrm{m^{-3}\,s^{-1}})$',
            r'$\langle G\rangle_N\,/\,(\mathrm{nm\,s^{-1}})$',
            r'$q_{\mathrm{nuc}}\,/\,(\mathrm{kg\,m^{-3}\,s^{-1}})$',
            r'$q_{\mathrm{growth}}\,/\,(\mathrm{kg\,m^{-3}\,s^{-1}})$',
            r'$\log_{10}(Q_{\mathrm{growth}}/Q_{\mathrm{nuc}})$','Intensity density / peak']
    for i,(ax,label) in enumerate(zip(axes.flat,labels)):
        ax.set_ylabel(label);ax.set_xlabel(r'$L\,/\,\mathrm{nm}$' if i==7 else r'$t\,/\,\mathrm{s}$')
        ax.set_title(['Free lipid','Supersaturation','Nucleation','Number-weighted growth',
                      'Nucleation uptake','Growth uptake','Cumulative uptake ratio',r'PSD at $0.15\,\mathrm{s}$'][i],fontsize=10)
    axes[0,0].legend(fontsize=8);fig.suptitle('Formation dynamics for three flow-rate ratios: reconstructed material inputs')
    save_figure(fig,folder,'figure8_dynamics')

    fig,panels=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    axes=panels.ravel()
    report=[]
    for index,(f,d) in enumerate(histories.items()):
        t=d['time'];initial=d['free_lipid'][0];w=f/(1+f)
        escaped=d['escaped_lipid']/initial
        corrected=(d['material_element_lipid']+d['escaped_lipid']/(1-w))/(initial/(1-w))-1
        label=fr'$\mathrm{{FRR}}={f}$';valid=t>0
        axes[0].semilogx(t[valid]/60,escaped[valid],label=label)
        axes[1].loglog(t[valid]/60,np.maximum(abs(corrected[valid]),1e-16))
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][index]
        axes[2].semilogx(t[valid]/60,d['lower_volume_fraction'][valid],color=color,label=label+', lower')
        axes[2].semilogx(t[valid]/60,d['upper_volume_fraction'][valid],'--',color=color,label=label+', upper')
        axes[3].loglog(t[valid]/60,np.maximum(d['negative_intensity_fraction'][valid],1e-18),color=color,label=label)
        flagged=t[escaped>1e-3]
        report.append((f,t[-1],np.max(abs(corrected)),escaped[-1],flagged[0] if len(flagged) else np.nan,
                       d['lower_volume_fraction'][-1],d['upper_volume_fraction'][-1],np.max(d['negative_intensity_fraction'])))
    axes[0].set(ylabel='Cumulative escaped mass / initial lipid',title='Finite diameter domain')
    axes[1].set(ylabel='Relative corrected mass residual',title='Includes recorded escaped mass')
    axes[2].set(ylabel='Fraction of resolved particle volume',title='First / last two diameter bins')
    axes[3].set(ylabel='Negative / positive sixth-moment contribution',title='Sensitivity to small signed tail errors')
    for ax in axes:ax.set_xlabel(r'$t\,/\,\mathrm{min}$')
    axes[0].legend(fontsize=8);axes[2].legend(fontsize=7,ncol=2)
    save_figure(fig,folder,'numerical_checks')
    np.savetxt(folder/'verification.csv',report,delimiter=',',comments='',header='FRR,final_time_s,maximum_corrected_mass_residual,final_escaped_mass_fraction,first_time_escaped_fraction_above_0.001_s,final_lower_volume_fraction,final_upper_volume_fraction,maximum_negative_intensity_fraction')
    expanded=folder/'domain_check/frr3/history.csv'
    if expanded.is_file():
        wide=read(expanded);base=histories[3]
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for data,label in [(base,r'$L_{\max}=5\,\mu\mathrm{m}$, 80 bins'),
                           (wide,r'$L_{\max}=50\,\mu\mathrm{m}$, 102 bins')]:
            valid=data['time']>=1.0
            axes[0].semilogx(data['time'][valid]/60,data['intensity_mean_m'][valid]*1e9,label=label)
            axes[1].semilogx(data['time'][valid]/60,data['escaped_lipid'][valid]/data['free_lipid'][0],label=label)
        axes[0].set(ylabel=r'$M_7/M_6\,/\,\mathrm{nm}$',title='Diameter remains domain dependent')
        axes[1].set(ylabel='Escaped mass / initial lipid',title='Similar logarithmic bin spacing')
        for ax in axes:ax.set_xlabel(r'$t\,/\,\mathrm{min}$');ax.legend(fontsize=8)
        save_figure(fig,folder,'domain_refinement')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'src/assets/shin_case')
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--usetex',action='store_true')
    args=parser.parse_args();configure(usetex=args.usetex);plots(args.data_dir,args.comparison)
