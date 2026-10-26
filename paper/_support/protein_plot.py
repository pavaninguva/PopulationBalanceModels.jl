"""Two-model isothermal comparison, generated only from the active exports."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pymupdf
from protein_provenance import verify_compatibility
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plotting_style import configure, DOUBLE
from _support.posterior_density import joint_density, probability_levels

PAPER=Path(__file__).resolve().parents[1]
ROOT=PAPER.parent
BASE=PAPER/'protein/data'
OUT=PAPER/'protein/figures'
DOCS=ROOT/'docs/src/assets/protein_isothermal'
TABLES=PAPER/'protein/data/analysis/tables'
TIMES=(74.,180.,420.,660.)
DENSITY_YMAX=.16 # common declared display bound, checked against every curve/band
COLORS={74.:'#0072B2',180.:'#D55E00',420.:'#009E73',660.:'#6A3D9A'}
QUANTILES={.1:('D10_um',r'$d_{10}$','#0072B2','-'),.5:('D50_um',r'$d_{50}$','#D55E00','--'),.9:('D90_um',r'$d_{90}$','#009E73','-.')}
LABELS={'logAJ':r'$\ln(A_J/A_J^\circ)$','gamma':r'$\gamma$ (mJ m$^{-2}$)',
        'AG':r'$\ln(A_G/A_G^\circ)$','g':r'$g$',
        'grd_strength':r'$\ln(\theta_{\mathrm{GRD}}/t^\circ)$'}

def read(folder,name):
    return np.atleast_1d(np.genfromtxt(folder/name,delimiter=',',names=True,dtype=None,encoding='utf8'))

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def verify_shared_data():
    """Plot-only regeneration must use the supplied readings, not edited exports."""
    source=ROOT/'examples/data/protein_crystallization/tang2018_isothermal'
    shared=BASE/'shared';settings=json.loads((shared/'observation_model.json').read_text())
    for name,digest in settings['data_hashes'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Changed source observations: {name}')
    a=read(source,'concentration.csv');b=read(shared,'concentration.csv')
    for left,right in (('observation_id','id'),('model_time_min','time_min'),('concentration_g_L','observed_g_L'),('split','split')):
        if not np.array_equal(a[left],b[right]): raise ValueError('Shared concentrations do not match supplied readings')
    a=read(source,'psd_cdf.csv');a=a[a['window']=='isothermal_core'];b=read(shared,'psd.csv')
    a=np.sort(a,order=['time_min','size_um']);b=np.sort(b,order=['time_min','size_um'])
    for name in ('time_min','size_um','conditional_density_per_um','conditional_cdf','split'):
        if not np.array_equal(a[name],b[name]): raise ValueError('Shared PSDs do not match supplied isothermal_core readings')

def export(fig,name):
    OUT.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    creator='PopulationBalanceModels isothermal CNT/GRD comparison'
    pdf=OUT/f'{name}.pdf'
    fig.savefig(pdf,dpi=300,metadata={'Creator':creator,'CreationDate':None,'ModDate':None})
    # Rasterize the final vector artwork so PNG and PDF have identical layout.
    # This route uses actual LaTeX without requiring Agg's separate dvipng step.
    with pymupdf.open(pdf) as document:
        document[0].get_pixmap(dpi=300,alpha=False).save(OUT/f'{name}.png')
    svg=OUT/f'{name}.svg'
    fig.savefig(svg,dpi=300,metadata={'Creator':creator,'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    for suffix in ('png','svg','pdf'):
        path=OUT/f'{name}.{suffix}'
        shutil.copy2(path,DOCS/path.name)
    plt.close(fig)
    print(f'Rendered {name}',flush=True)

def prior_density(meta,display):
    """Normalized analytical prior in the DISPLAY coordinate; never peak-scaled."""
    x=np.asarray(display,float)
    theta=np.exp(x) if meta['plotting_coordinate']=='log' else x
    jac=np.exp(x) if meta['plotting_coordinate']=='log' else np.ones_like(x)
    value=theta.copy()
    if meta['density_coordinates']=='inference':
        if meta['transform']=='log':
            with np.errstate(divide='ignore',invalid='ignore'): value=np.log(theta);jac=jac/theta
        elif meta['transform']=='logit':
            a,b=meta['lower'],meta['upper']
            with np.errstate(divide='ignore',invalid='ignore'):
                value=np.log((theta-a)/(b-theta));jac=jac*(b-a)/((theta-a)*(b-theta))
    args=meta['arguments'];family=meta['family'];density=np.zeros_like(value)
    if family=='uniform':
        a,b=args;density=np.where((value>=a)&(value<=b),1/(b-a),0.)
    elif family=='lognormal':
        mu,s=args;valid=value>0
        density[valid]=np.exp(-.5*((np.log(value[valid])-mu)/s)**2)/(value[valid]*s*np.sqrt(2*np.pi))
    elif family=='exponential':
        scale=args[0];valid=value>=0;density[valid]=np.exp(-value[valid]/scale)/scale
    else: raise ValueError(f'Unsupported prior family {family}')
    return np.where(np.isfinite(density*jac),density*jac,0.)

def support(meta):
    if meta['plotting_coordinate']=='log':
        return tuple(meta['arguments']) if meta['density_coordinates']=='inference' and meta['family']=='uniform' else (-np.inf,np.inf)
    if meta['family']=='uniform' and meta['density_coordinates']=='physical': return tuple(meta['arguments'])
    return (0.,np.inf) if meta['family'] in ('lognormal','exponential') else (-np.inf,np.inf)

def prior_description(m):
    a=m['arguments']
    if m['family']=='uniform':
        value=f'Uniform({a[0]:.5g}, {a[1]:.5g})'
        return f"log-uniform AG on [{np.exp(a[0]):g}, {np.exp(a[1]):g}]" if m['name']=='AG' else value
    if m['family']=='lognormal': return f"LogNormal(log({np.exp(a[0]):g}), {a[1]:g})"
    return f'Exponential(scale/mean {a[0]:g} min)'

def prior_tables(metadata):
    rows=[dict(parameter=m['name'],meaning=m['meaning'],units=m['units'],prior=prior_description(m),
        prior_coordinates=m['density_coordinates'],transform=m['transform'],model='CNT + GRD only' if m['name']=='grd_strength' else 'Both') for m in metadata]
    write_csv(TABLES/'prior_table.csv',rows)
    md='| Parameter | Meaning and units | Prior | Model |\n|:--|:--|:--|:--|\n'
    for r in rows: md+=f"| `{r['parameter']}` | {r['meaning']} ({r['units']}) | {r['prior']} | {r['model']} |\n"
    (TABLES/'prior_table.md').write_text(md);DOCS.mkdir(parents=True,exist_ok=True)
    (DOCS/'prior_table.txt').write_text(md)
    tex='\\begin{tabular}{llll}\n\\hline\nParameter & Units & Prior & Model \\\\\n\\hline\n'
    for r in rows:
        parameter=r["parameter"].replace("_",r"\_")
        units=r["units"].replace("²",r"$^2$")
        tex+=f"{parameter} & {units} & {r['prior']} & {r['model']} \\\\\n"
    (TABLES/'prior_table.tex').write_text(tex+'\\hline\n\\end{tabular}\n')

def verify_model(folder):
    m=json.loads((folder/'manifest.json').read_text())
    if not m['complete'] or m['mode']!='production': raise ValueError(f'{folder.name}: full verified posterior run required')
    for name,digest in m['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest: raise ValueError(f'Changed numerical result {folder/name}')
    changed = any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest
                  for name,digest in m['source_files'].items())
    if changed:
        verify_compatibility(folder,m)
    d=json.loads((folder/'sampling_diagnostics.json').read_text())
    if not d['passed']: raise ValueError(f'{folder.name}: posterior sampling screen failed')
    verification=folder/'target_verification.json'
    if not verification.exists():
        raise ValueError(f'{folder.name}: verify this retained posterior with run.jl --verify before plotting')
    checked=json.loads(verification.read_text())
    if not checked['passed'] or checked.get('target_hash')!=m['target_hash']:
        raise ValueError(f'{folder.name}: verify this retained posterior with run.jl --verify')
    for filename,key in [('posterior_draws.csv','posterior_draws_sha256'),('sampler_statistics.csv','sampler_statistics_sha256')]:
        if checked.get(key)!=hashlib.sha256((folder/filename).read_bytes()).hexdigest():
            raise ValueError(f'{folder.name}: target verification belongs to different retained draws')
    if d.get('sampling_coordinates') in ('CNT_soft_barrier_rates','CNT_transport_rates'):
        provenance=json.loads((folder/'sampling_provenance.json').read_text())
        helper=ROOT/'examples/support/isothermal_protein_sampling.jl'
        if provenance['coordinate_source_sha256']!=hashlib.sha256(helper.read_bytes()).hexdigest():
            raise ValueError('Changed sampling-coordinate implementation; verify/reproduce the posterior first')
        if d.get('sampling_coordinates')=='CNT_transport_rates':
            path=ROOT/'paper/protein/sampling_transport.json'
            if provenance['transport_configuration_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Changed sampler transport configuration')
    return m

def layout():
    fig,axes=plt.subplots(2,2,figsize=(DOUBLE,5.45))
    fig.subplots_adjust(left=.095,right=.985,bottom=.18,top=.95,wspace=.30,hspace=.36)
    for letter,ax in zip('abcd',axes.flat):
        ax.text(0.,1.025,f'({letter})',transform=ax.transAxes,ha='left',va='bottom',fontsize=9)
        ax.tick_params(direction='out',length=3,width=.7,pad=3,labelsize=8)
        for spine in ax.spines.values(): spine.set_linewidth(.7)
    axes[0,0].set(xlim=(-8,690),ylim=(0,22),xlabel='Time (min)',ylabel=r'$c$ (g L$^{-1}$)')
    axes[0,0].set_xticks([0,200,400,600]);axes[0,0].set_yticks([0,5,10,15,20])
    for ax in (axes[0,1],axes[1,0]):
        ax.set(xlim=(2.5,25),xlabel=r'Equivalent diameter ($\mu$m)',ylabel=r'Conditional density ($\mu$m$^{-1}$)')
        ax.set_xticks([5,10,15,20,25]);ax.set_yticks([0,.04,.08,.12,.16])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '0' if x==0 else f'{x:.2f}'))
    axes[1,1].set(xlim=(-8,690),ylim=(2.5,25),xlabel='Time (min)',ylabel=r'Diameter quantile ($\mu$m)')
    axes[1,1].set_xticks([0,200,400,600]);axes[1,1].set_yticks([5,10,15,20,25])
    return fig,axes

def figure_legend(fig,*,fitted):
    handles=[]
    if fitted:
        handles.extend([
            Line2D([],[],color='.15',lw=1.3,label='MAP prediction'),
            Line2D([],[],color='.35',ls='--',lw=1.,marker='o',ms=3,label='Digitized PSD'),
            Patch(facecolor='.55',alpha=.2,edgecolor='none',label=r'95\% latent interval'),
        ])
    handles.extend([
        Line2D([],[],color='.2',ls='none',marker='o',ms=3.5,label='Training observation'),
        Line2D([],[],color='.2',ls='none',marker='s',mfc='white',ms=3.5,label='Test observation'),
        Line2D([],[],color='.2',ls='none',marker='D',ms=3.5,label='Fixed initial condition'),
    ])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.025),ncol=3,
               fontsize=7.2,columnspacing=1.7,handlelength=2.,handletextpad=.6)

def observation_panels(axes,data,psds,quantiles,*,guide=False,fitted=False):
    ax=axes[0,0]
    for split,marker,label,fill in (('calibration','o','Training',True),('validation','s','Test',False),('initial_condition','D','Fixed initial',True)):
        q=data[data['split']==split]
        if len(q):
            ax.errorbar(q['time_min'],q['observed_g_L'],yerr=q['sigma_g_L'],fmt=marker,ms=3.2,
                mfc='.2' if fill else 'white',mec='.2',color='.2',elinewidth=.65,capsize=2,zorder=4)
    for t in TIMES:
        q=psds[psds['time_min']==t];ax=axes[0,1] if t<660 else axes[1,0]
        marker='o' if t<660 else 's'
        ax.plot(q['size_um'],q['conditional_density_per_um'],ls='--',marker=marker,
            markevery=6,ms=2.7,mfc=COLORS[t] if t<660 else 'white',mew=.7,
            lw=.9,color=COLORS[t],zorder=3)
    ax=axes[1,1]
    for p,(key,label,color,ls) in QUANTILES.items():
        q=quantiles[quantiles['probability']==p]
        if guide: ax.plot(q['time_min'],q['observed_um'],ls=':',lw=.7,color=color)
        for split,marker in (('calibration','o'),('validation','s')):
            s=q[q['split']==split]
            ax.plot(s['time_min'],s['observed_um'],ls='none',marker=marker,ms=3.5,
                mfc=color if split=='calibration' else 'white',mec=color,color=color,zorder=4)
    for ax,times in ((axes[0,1],TIMES[:-1]),(axes[1,0],TIMES[-1:])):
        handles=[Line2D([],[],color=COLORS[t],lw=1.2,label=f'{t:g} min') for t in times]
        ax.legend(handles=handles,fontsize=7.5,loc='upper right',handlelength=1.7,labelspacing=.35)
    handles=[Line2D([],[],color=color,ls=ls if fitted else ':',lw=1.2,label=label)
             for key,label,color,ls in QUANTILES.values()]
    axes[1,1].legend(handles=handles,fontsize=7.5,loc='upper right',ncol=3,
                     handlelength=1.5,columnspacing=.8)
    figure_legend(axes[0,0].figure,fitted=fitted)

def interval_at(table,time,quantity):
    values=table[table['time_min']==time][quantity]
    return np.quantile(values,[.025,.975]) if len(values) and np.isfinite(values).all() else [np.nan,np.nan]

def four_panel(name,data,psds,quantiles,ymax):
    folder=BASE/name;trajectory=read(folder,'trajectory.csv');latent=read(folder,'latent_predictions.csv')
    curves=read(folder,'distribution_curves.csv');drawcurves=read(folder,'latent_distribution_curves.csv')
    fig,axes=layout()
    # Curves first, so observed markers and error bars remain readable.
    a=axes[0,0];times=np.unique(latent['time_min'])
    ci=np.array([interval_at(latent,t,'concentration_g_L') for t in times])
    a.fill_between(times,ci[:,0],ci[:,1],color='.4',alpha=.2,label=r'95\% latent interval')
    a.plot(trajectory['time_min'],trajectory['concentration_g_L'],'k-',lw=1.2,label='MAP')
    a.axvline(420,color='.5',ls=':',lw=.7,ymax=.65)
    for t in TIMES:
        a=axes[0,1] if t<660 else axes[1,0];q=curves[curves['time_min']==t]
        draws=drawcurves[drawcurves['time_min']==t]
        values=np.array([draws[draws['left_um']==x]['conditional_density_per_um'] for x in q['left_um']])
        lo,hi=np.quantile(values,[.025,.975],axis=1)
        centers=.5*(q['left_um']+q['right_um'])
        # Display the stored bin averages at their centers; no smoothing,
        # remeshing or forward-model evaluation is performed.
        a.fill_between(centers,lo,hi,alpha=.15,color=COLORS[t],linewidth=0)
        a.plot(centers,q['conditional_density_per_um'],color=COLORS[t],lw=1.35)
    a=axes[1,1]
    for p,(key,label,color,ls) in QUANTILES.items():
        ci=np.array([interval_at(latent,t,key) for t in times])
        a.fill_between(times,ci[:,0],ci[:,1],color=color,alpha=.12)
        a.plot(trajectory['time_min'],trajectory[key],ls=ls,color=color,lw=1.1)
    observation_panels(axes,data,psds,quantiles,fitted=True)
    for a in (axes[0,1],axes[1,0]): a.set_ylim(0,ymax)
    export(fig,'protein_fit_'+name)

def marginal(values,x,bounds):
    h=1.06*np.std(values,ddof=1)*len(values)**(-.2)
    if not h>0: raise ValueError('Degenerate posterior')
    centers=[values]+[2*b-values for b in bounds if np.isfinite(b)]
    return sum(np.exp(-.5*((x[:,None]-v)/h)**2).sum(axis=1) for v in centers)/(len(values)*h*np.sqrt(2*np.pi))

def corner(name,metadata,*,write_tables=True):
    folder=BASE/name;draws=read(folder,'posterior_draws.csv');mode=json.loads((folder/'map.json').read_text())['parameters']
    metas=json.loads((folder/'priors.json').read_text());n=len(metas)
    samples=np.column_stack([np.log(draws[m['name']]) if m['plotting_coordinate']=='log' else draws[m['name']] for m in metas])
    point=[np.log(mode[m['name']]) if m['plotting_coordinate']=='log' else mode[m['name']] for m in metas]
    bounds=[support(m) for m in metas]
    fig,axes=plt.subplots(n,n,figsize=(DOUBLE,5.9 if n==4 else 6.8))
    fig.subplots_adjust(left=.12,right=.98,bottom=.12,top=.96,wspace=.13,hspace=.13)
    limits=[]
    for i in range(n):
        q=np.quantile(samples[:,i],[.0005,.9995]);margin=.15*(q[1]-q[0])
        limits.append((max(q[0]-margin,bounds[i][0]),min(q[1]+margin,bounds[i][1])))
    for i in range(n):
        lo,hi=limits[i]
        for j in range(n):
            a=axes[i,j]
            if j>i: a.axis('off');continue
            if i==j:
                x=np.linspace(lo,hi,601)
                density=marginal(samples[:,i],x,bounds[i])
                a.fill_between(x,0,density,color='#0072B2',alpha=.13,linewidth=0)
                a.plot(x,density,color='#0072B2',lw=1.2)
                a.plot(x,prior_density(metas[i],x),color='.35',ls='--',lw=1.)
                a.axvline(point[i],color='#D55E00',lw=.8,ls=':')
                a.set_xlim(lo,hi);a.set_ylim(bottom=0);a.tick_params(labelleft=False)
                a.set_yticks([])
            else:
                curved={metas[j]['name'],metas[i]['name']}=={'logAJ','gamma'}
                if not curved:
                    X,Y,density=joint_density(samples[:,j],samples[:,i],bounds[j],bounds[i],points=151)
                    levels=probability_levels(density)
                    a.contourf(X,Y,density,levels=[*levels,density.max()*1.001],
                               colors=['#e7f0f7','#bad7e8'])
                    a.contour(X,Y,density,levels=levels,colors=['#86b2cc','#0072B2'],linewidths=[.65,.9])
                # This CNT relation is curved and thinner than a Cartesian KDE
                # bandwidth. Actual draws preserve its width without a blur.
                if curved:
                    index=np.linspace(0,len(samples)-1,min(2000,len(samples)),dtype=int)
                    a.plot(samples[index,j],samples[index,i],'.',color='#0072B2',ms=.8,alpha=.35)
                a.plot(point[j],point[i],'x',color='#D55E00',ms=4,mew=.8)
                a.set_ylim(lo,hi);a.set_xlim(*limits[j])
            if i==n-1: a.set_xlabel(LABELS[metas[j]['name']],fontsize=8.5,labelpad=5)
            else: a.tick_params(labelbottom=False)
            if j==0 and i!=j: a.set_ylabel(LABELS[metas[i]['name']],fontsize=8.5,labelpad=5)
            elif i!=j: a.tick_params(labelleft=False)
            a.tick_params(labelsize=7.2,direction='out',length=2.5,width=.65,pad=2)
            for spine in a.spines.values(): spine.set_linewidth(.65)
            a.locator_params(axis='x',nbins=3)
            if i!=j: a.locator_params(axis='y',nbins=3)
    handles=[Line2D([],[],color='#0072B2',lw=1.2,label='NUTS posterior'),
             Line2D([],[],color='.35',ls='--',lw=1.,label='Analytical prior'),
             Line2D([],[],color='#D55E00',ls='none',marker='x',ms=4,label='MAP'),
             Patch(facecolor='#bad7e8',edgecolor='#0072B2',lw=.6,label=r'68\% joint region'),
             Patch(facecolor='#e7f0f7',edgecolor='#86b2cc',lw=.6,label=r'95\% joint region')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.965,.96),fontsize=7.5,
               handlelength=2.,labelspacing=.5)
    reference=(r'$A_J^\circ=1\ \mathrm{m}^{-3}\mathrm{s}^{-1}$'+'\n'+r'$A_G^\circ=1\ \mathrm{nm\,min}^{-1}$')
    if n==5: reference+='\n'+r'$t^\circ=1\ \mathrm{min}$'
    fig.text(.68,.72 if n==5 else .68,reference,fontsize=8,va='top',linespacing=1.6)
    export(fig,'protein_corner_'+name)
    coordinates=[('log_'+m['name']) if m['plotting_coordinate']=='log' else m['name'] for m in metas]
    if write_tables:
        write_csv(TABLES/f'posterior_correlations_{name}.csv',[
            dict(coordinate=coordinates[i],**dict(zip(coordinates,row)))
            for i,row in enumerate(np.corrcoef(samples.T))])

def result_tables(metadata):
    rows=[];comp=[];numerical=[]
    for name in ('cnt','cnt_grd'):
        folder=BASE/name;draws=read(folder,'posterior_draws.csv');mode=json.loads((folder/'map.json').read_text())['parameters']
        for m in metadata:
            key=m['name'];fixed=name=='cnt' and key=='grd_strength'
            q=[np.nan]*3 if fixed else np.quantile(draws[key],[.025,.5,.975])
            rows.append(dict(model=name,parameter=key,units=m['units'],MAP=0. if fixed else mode[key],
                median=q[1],q025=q[0],q975=q[2],fixed=fixed))
        v=json.loads((folder/'validation_metrics.json').read_text());diag=json.loads((folder/'sampling_diagnostics.json').read_text())
        comp.append(dict(model=name,training_concentration_RMSE=v['calibration_concentration_RMSE_g_L'],
            test_concentration_RMSE=v['validation_concentration_RMSE_g_L'],maximum_Rhat=max(diag['rhat']),
            minimum_bulk_ESS=min(diag['ess_bulk']),minimum_tail_ESS=min(diag['ess_tail']),divergences=diag['divergences']))
        settings=json.loads((folder/'run_settings.json').read_text());refinement=read(folder,'refinement.csv')
        finer=json.loads((folder/'refined_map.json').read_text());drawchecks=read(folder,'latent_draw_diagnostics.csv')
        numerical.append(dict(model=name,bins=settings['bins'],steps=v['steps'],solve_seconds=v['solve_seconds'],
            maximum_material_error_g_L=max(v['maximum_material_error_g_L'],float(np.max(drawchecks['maximum_material_error_g_L']))),
            maximum_upper_bin_fraction=max(v['maximum_upper_bin_fraction'],float(np.max(drawchecks['maximum_upper_bin_fraction']))),
            **{f'{kind}_CDF_change':float(refinement[refinement['check']==kind]['maximum_CDF_change'][0]) for kind in ('time','size','domain')},
            finer_refit_CDF_change=finer['calibration_CDF_prediction_change'],
            finer_refit_concentration_change_g_L=finer['calibration_concentration_prediction_change_g_L']))
    write_csv(TABLES/'parameter_estimates.csv',rows);write_csv(TABLES/'model_comparison.csv',comp)
    write_csv(TABLES/'numerical_verification.csv',numerical)
    text='| Model | Size bins | Steps (displayed MAP trajectory) | Max CDF change: time / size / domain | Max material error (g/L) |\n|:--|--:|--:|:--|--:|\n'
    for r in numerical:
        changes=' / '.join(f"{r[k+'_CDF_change']:.2g}" for k in ('time','size','domain'))
        text+=f"| {r['model']} | {r['bins']} | {r['steps']} | {changes} | {r['maximum_material_error_g_L']:.2g} |\n"
    (TABLES/'numerical_verification.md').write_text(text);(DOCS/'numerical_verification.txt').write_text(text)
    distributions=[];contacts=[]
    for name in ('cnt','cnt_grd'):
        folder=BASE/name;quant=read(folder,'quantile_predictions.csv');raw=read(folder,'posterior_draws.csv')
        for r in read(folder,'distribution_metrics.csv'):
            q=quant[quant['time_min']==r['time_min']]
            distributions.append(dict(model=name,time_min=float(r['time_min']),split=r['split'],CDF_RMSE=float(r['CDF_RMSE']),W2_um=float(r['W2_um']),
                **{f'd{int(100*p)}_error_um':float(q[q['probability']==p]['residual_um'][0]) for p in (.1,.5,.9)}))
        for m in json.loads((folder/'priors.json').read_text()):
            if m['family']=='uniform':
                values=np.log(raw[m['name']]) if m['density_coordinates']=='inference' else raw[m['name']]
                a,b=m['arguments'];band=.01*(b-a)
                contacts.append(dict(model=name,parameter=m['name'],coordinates=m['density_coordinates'],
                    fraction_lower_one_percent=float(np.mean(values<a+band)),fraction_upper_one_percent=float(np.mean(values>b-band))))
    write_csv(TABLES/'distribution_comparison.csv',distributions);write_csv(TABLES/'bound_contact.csv',contacts)
    md='| Parameter (units) | CNT MAP | CNT median [95% interval] | CNT + GRD MAP | CNT + GRD median [95% interval] |\n|:--|--:|:--|--:|:--|\n'
    for m in metadata:
        pair=[next(r for r in rows if r['model']==name and r['parameter']==m['name']) for name in ('cnt','cnt_grd')]
        fields=[]
        for r in pair:
            fields+=['0 (fixed)','—'] if r['fixed'] else [f"{r['MAP']:.4g}",f"{r['median']:.4g} [{r['q025']:.4g}, {r['q975']:.4g}]"]
        md+=f"| `{m['name']}` ({m['units']}) | "+' | '.join(fields)+' |\n'
    (TABLES/'parameter_estimates.md').write_text(md);(DOCS/'parameter_estimates.txt').write_text(md)
    tex='\\begin{tabular}{lllll}\n\\hline\nParameter & CNT MAP & CNT median [95\\%] & GRD MAP & GRD median [95\\%] \\\\\n\\hline\n'
    for line in md.splitlines()[2:]:
        fields=[a.strip().replace('`','').replace('_',r'\_').replace('—','--').replace('²',r'$^2$') for a in line.split('|')[1:-1]]
        tex+=' & '.join(fields)+' \\\\\n'
    (TABLES/'parameter_estimates.tex').write_text(tex+'\\hline\n\\end{tabular}\n')
    text='| Model | Training concentration RMSE (g/L) | Test concentration RMSE (g/L) | Max Rhat | Min bulk/tail ESS |\n|:--|--:|--:|--:|:--|\n'
    for r in comp: text+=f"| {r['model']} | {r['training_concentration_RMSE']:.3g} | {r['test_concentration_RMSE']:.3g} | {r['maximum_Rhat']:.4f} | {r['minimum_bulk_ESS']:.0f}/{r['minimum_tail_ESS']:.0f} |\n"
    text+='\n| Model | PSD time (min) | Use | CDF RMSE (fraction) | W2 (µm) | d10 error (µm) | d50 error (µm) | d90 error (µm) |\n|:--|--:|:--|--:|--:|--:|--:|--:|\n'
    for r in distributions:
        text+=f"| {r['model']} | {r['time_min']:g} | {r['split']} | {r['CDF_RMSE']:.4f} | {r['W2_um']:.3f} | {r['d10_error_um']:.3f} | {r['d50_error_um']:.3f} | {r['d90_error_um']:.3f} |\n"
    (TABLES/'fit_comparison.md').write_text(text);(DOCS/'fit_comparison.txt').write_text(text)

def captions(data_only=False):
    text=r'''\newcommand{\ProteinDataCaption}{Original graph-digitized Tang et al. (2018) 100 mL isothermal lysozyme observations at 20 degrees C. (a) Dissolved concentration with assumed 5\% standard deviations; the diamond fixes the initial condition, filled circles train and open squares test. (b) Training conditional number densities at 74, 180 and 420 min. (c) Held-out 660-min density. (d) Conditional number $d_{10},d_{50},d_{90}$; connecting dotted lines are guides only. All sizes and quantiles condition on 2.5--25 $\mu$m. No model prediction or invented PSD uncertainty is shown.}
\newcommand{\ProteinFitCaption}{CNT or CNT with native random-fluctuation GRD, identified by the filename. Panels have the same meanings and axes as the data overview. Solid curves/stairs show the inference-coordinate MAP; observations are dashed density reconstructions and markers. Shading is the pointwise 95\% latent interval from screened NUTS draws, conditional on fixed observation errors and physical constants; it excludes additional measurement noise. The vertical dotted line marks the end of training at 420 min. The 660-min PSD and later concentrations are predicted without refitting. Quantile segments are omitted when the conditioning window is empty; quantiles derived from PSDs are not independent likelihood data. Numerical densities use actual bin edges without smoothing.}
\newcommand{\ProteinCornerCaption}{Four-parameter CNT or five-parameter CNT+GRD NUTS posterior, identified by filename. Diagonals show normalized posterior KDEs and dashed analytical priors generated from the active specifications; positive wide-scale parameters use labelled natural-log plotting coordinates, including the correct prior Jacobian. Broad priors are not renormalized to the displayed posterior window; full support is in the prior table. Where shown, joint KDE contours enclose approximately 68\% and 95\% probability. The thin, curved nucleation-prefactor/interfacial-energy relation is shown directly with retained draws. Orange markers denote the transformed inference-coordinate MAP. Sampling convergence does not establish numerical convergence, identifiability or physical model adequacy.}
'''
    text=text.replace('Solid curves/stairs show','Solid curves show')
    text=text.replace('Numerical densities use actual bin edges without smoothing.',
        'Numerical density curves and bands connect the stored bin averages at their bin centers; no smoothing or extra forward solves are used.')
    text=text.replace('Orange markers denote the transformed inference-coordinate MAP.',
        r'Orange markers denote the transformed inference-coordinate MAP. The reference units are $A_G^\circ=1\,\mathrm{nm\,min}^{-1}$ and $t^\circ=1\,\mathrm{min}$.')
    text=text.replace(r'\ProteinFitCaption',r'\ProteinFitDetails').replace(r'\ProteinCornerCaption',r'\ProteinCornerDetails')
    text=text.replace('CNT or CNT with native random-fluctuation GRD, identified by the filename. Panels have the same meanings and axes as the data overview.',
        '(a) Concentration; (b) training conditional number densities at 74, 180 and 420 min; (c) held-out 660-min density; (d) conditional number quantiles. All densities and quantiles condition on 2.5--25 $\\mu$m.')
    text=text.replace('Four-parameter CNT or five-parameter CNT+GRD NUTS posterior, identified by filename. ','')
    for name,prefix,identity in (('cnt','CNT','CNT nucleation and size-independent mean growth'),
        ('cnt_grd','GRD',r'CNT nucleation and mean growth with random-fluctuation dispersion $D_G=\theta_{\mathrm{GRD}}G^2$')):
        extra='';folder=BASE/name
        if not data_only and (folder/'sampling_diagnostics.json').exists() and json.loads((folder/'sampling_diagnostics.json').read_text())['passed']:
            settings=json.loads((folder/'run_settings.json').read_text())
            diag=json.loads((folder/'sampling_diagnostics.json').read_text())
            latent=json.loads((folder/'latent_settings.json').read_text())
            extra=(f" MUSCL growth and native SSPRK33 use {settings['bins']} bins on 0--50 $\\mu$m"
                f" with combined CFL {settings['cfl']:g} and maximum step {settings['maximum_dt_min']:g} min;"
                f" {latent['draws']} retained parameter draws supply the bands."
                f" The posterior retains {diag['chains']} chains of {diag['samples']} draws each.")
        text+='\\newcommand{\\Protein'+prefix+'FitCaption}{'+identity+'. \\ProteinFitDetails'+extra+'}\n'
        text+='\\newcommand{\\Protein'+prefix+'CornerCaption}{'+identity+': NUTS posterior. \\ProteinCornerDetails}\n'
    text+=r'\newcommand{\ProteinPriorTableCaption}{Implemented priors, physical units and applicable models. Inference transformations are separate from prior-density coordinates; the dispersion Exponential prior has mean/scale 10 min.}'+'\n'
    text+=r'\newcommand{\ProteinParameterTableCaption}{Inference-coordinate MAP estimates transformed to physical units, posterior medians and equal-tailed 95\% credible intervals. Model 1 fixes dispersion to zero. Intervals condition on the model, fixed observation errors and physical constants.}'+'\n'
    (OUT/'captions.tex').write_text(text)

def plot(data_only=False,figures_only=True):
    configure(require_dvipng=False)
    OUT.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    verify_shared_data()
    shared=BASE/'shared';data=read(shared,'concentration.csv');psds=read(shared,'psd.csv');qs=read(shared,'quantiles.csv')
    metadata=json.loads((shared/'priors.json').read_text())
    if not figures_only:
        TABLES.mkdir(parents=True,exist_ok=True)
        prior_tables(metadata)
    ymax=DENSITY_YMAX
    required=1.05*np.max(psds['conditional_density_per_um'])
    if not data_only:
        for name in ('cnt','cnt_grd'):
            verify_model(BASE/name)
            curves=read(BASE/name,'distribution_curves.csv');latent=read(BASE/name,'latent_distribution_curves.csv')
            upper=max(np.quantile(latent[(latent['time_min']==t)&(latent['left_um']==x)]['conditional_density_per_um'],.975)
                for t in TIMES for x in np.unique(curves[curves['time_min']==t]['left_um']))
            required=max(required,1.05*np.max(curves['conditional_density_per_um']),1.05*upper)
    if required>ymax:
        raise ValueError('The declared common density limit would clip a curve/band; increase DENSITY_YMAX for all three figures')
    fig,axes=layout();observation_panels(axes,data,psds,qs,guide=True)
    for a in (axes[0,1],axes[1,0]): a.set_ylim(0,ymax)
    export(fig,'protein_data')
    if not data_only:
        for name in ('cnt','cnt_grd'):
            four_panel(name,data,psds,qs,ymax);corner(name,metadata,write_tables=not figures_only)
        if not figures_only: result_tables(metadata)
    captions(data_only=data_only)
    names=('protein_data',) if data_only else ('protein_data','protein_fit_cnt','protein_fit_cnt_grd','protein_corner_cnt','protein_corner_cnt_grd')
    files={f'{name}.{ext}':hashlib.sha256((OUT/f'{name}.{ext}').read_bytes()).hexdigest()
        for name in names for ext in ('png','pdf','svg')}
    inputs={f'{name}/{file}':hashlib.sha256((BASE/name/file).read_bytes()).hexdigest()
        for name in (('shared',) if data_only else ('shared','cnt','cnt_grd'))
        for file in (('concentration.csv','psd.csv','quantiles.csv') if name=='shared' else
                     ('manifest.json','posterior_draws.csv','map.json','latent_predictions.csv','latent_distribution_curves.csv'))}
    (OUT/'plot_provenance.json').write_text(json.dumps(dict(renderer='actual LaTeX',dpi=300,files=files,
        retained_inputs=inputs,figures_only=figures_only,
        python=platform.python_version(),numpy=np.__version__,matplotlib=plt.matplotlib.__version__,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        density_helpers_sha256=hashlib.sha256((PAPER/'_support/posterior_density.py').read_bytes()).hexdigest()),indent=2)+'\n')
    print(OUT)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only',action='store_true')
    parser.add_argument('--tables',action='store_true',help='Also export derived tables to analysis/, leaving producer data immutable')
    parser.add_argument('--figures-only',action='store_true',help='Regenerate figures without rewriting result tables')
    args=parser.parse_args()
    plot(data_only=args.data_only,figures_only=not args.tables)
