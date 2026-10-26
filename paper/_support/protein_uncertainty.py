"""Read-only local-Gaussian/MCMC comparisons, including explicit rejections."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from protein_plot import (
    BASE, OUT, DOCS, LABELS, export, joint_density, marginal,
    prior_density, probability_levels, read, support, verify_model,
)
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plotting_style import configure
from protein_provenance import ANALYSIS, sha, verify_analysis_compatibility

APPROXIMATIONS = {'laplace': 'Laplace', 'gauss_newton': 'Gauss--Newton'}


# These maps describe the existing unit-scale inference transformations.
# The priors remain distinct; Gaussian draws are never truncated to prior support.
def physical(z, meta):
    z = np.asarray(z, dtype=float)
    if meta['transform'] == 'log':
        with np.errstate(over='ignore'):
            return np.exp(z)
    if meta['transform'] == 'logit':
        # Stable logistic without evaluating the overflowing branch of np.where.
        value = np.empty_like(z)
        positive = z >= 0
        value[positive] = 1 / (1 + np.exp(-z[positive]))
        e = np.exp(z[~positive])
        value[~positive] = e / (1 + e)
        return meta['lower'] + (meta['upper']-meta['lower']) * value
    if meta['transform'] == 'identity':
        return z.copy()
    raise ValueError(f"Unimplemented inference transformation: {meta['transform']}")


def display(z, meta):
    if meta['plotting_coordinate'] == 'log' and meta['transform'] == 'log':
        return np.asarray(z, dtype=float)
    theta = physical(z, meta)
    return np.log(theta) if meta['plotting_coordinate'] == 'log' else theta


def gaussian_marginal(x, mean, variance, meta):
    """Exact pushforward marginal density, including dz/d(display)."""
    x = np.asarray(x, dtype=float)
    theta = np.exp(x) if meta['plotting_coordinate'] == 'log' else x
    dtheta = theta if meta['plotting_coordinate'] == 'log' else np.ones_like(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if meta['transform'] == 'log':
            z, jac = np.log(theta), dtheta/theta
            valid = theta > 0
        elif meta['transform'] == 'logit':
            a, b = meta['lower'], meta['upper']
            z = np.log((theta-a)/(b-theta))
            jac = dtheta*(b-a)/((theta-a)*(b-theta))
            valid = (theta > a) & (theta < b)
        elif meta['transform'] == 'identity':
            z, jac, valid = theta, dtheta, np.ones_like(x, dtype=bool)
        else:
            raise ValueError(f"Unimplemented transformation: {meta['transform']}")
        density = np.exp(-.5*(z-mean)**2/variance)/np.sqrt(2*np.pi*variance)*jac
    return np.where(valid, density, 0.)


def gaussian_draws(report, count=100000, seed=20261026):
    """Cheap parameter draws only. Report all support violations, never filter."""
    mean = np.asarray(report['center'])
    covariance = np.asarray(report['covariance'])
    factor = np.linalg.cholesky(covariance)
    z = mean + np.random.default_rng(seed).normal(size=(count, len(mean))) @ factor.T
    theta = np.column_stack([physical(z[:, i], m) for i, m in enumerate(report['priors'])])
    shown = np.column_stack([display(z[:, i], m) for i, m in enumerate(report['priors'])])
    violations = {}
    for i, meta in enumerate(report['priors']):
        values = z[:, i] if meta['density_coordinates'] == 'inference' else theta[:, i]
        if meta['family'] == 'uniform':
            a, b = meta['arguments']
            invalid = (values < a) | (values > b)
        else:
            invalid = values <= 0
        violations[meta['name']] = int(invalid.sum())
    diagnostics = dict(seed=seed, draws=count, prior_support_violations=violations,
                       nonfinite_physical=int((~np.isfinite(theta)).sum()),
                       nonfinite_display=int((~np.isfinite(shown)).sum()),
                       filtering='none; no rejection, truncation or renormalization')
    if diagnostics['nonfinite_physical'] or diagnostics['nonfinite_display']:
        raise ValueError(f'Nonfinite Gaussian transformation: {diagnostics}')
    return z, theta, shown, diagnostics


def draw_corner(canvas, samples, metadata, point, limits, *, approximation=None):
    n = len(metadata)
    axes = canvas.subplots(n, n, squeeze=False)
    canvas.subplots_adjust(left=.14, right=.97, bottom=.13, top=.91, wspace=.13, hspace=.13)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                x = np.linspace(*limits[i], 601)
                if approximation is None:
                    density = marginal(samples[:, i], x, support(metadata[i]))
                else:
                    # A Gaussian in logit coordinates can have a narrow
                    # physical-density peak near a bound. Resolve it using
                    # mapped coordinates instead of a coarse uniform x grid.
                    mean=approximation['center'][i]
                    sd=np.sqrt(approximation['covariance'][i][i])
                    mapped=display(np.linspace(mean-8*sd,mean+8*sd,1601),metadata[i])
                    x=np.unique(np.r_[x,mapped[(mapped>=limits[i][0])&(mapped<=limits[i][1])]])
                    density = gaussian_marginal(x, approximation['center'][i],
                                                approximation['covariance'][i][i], metadata[i])
                ax.fill_between(x, 0, density, color='#0072B2', alpha=.14)
                ax.plot(x, density, color='#0072B2', lw=1)
                ax.plot(x, prior_density(metadata[i], x), '--', color='.35', lw=1)
                ax.axvline(point[i], color='#D55E00', lw=.8, ls=':')
                ax.set(xlim=limits[i], ylim=(0, None))
                ax.tick_params(labelleft=False)
            else:
                if approximation is not None:
                    # These are transformed probability ellipses, NOT HDR
                    # contours of the generally non-Gaussian displayed density.
                    indices = [j, i]
                    cov = np.asarray(approximation['covariance'])[np.ix_(indices, indices)]
                    factor = np.linalg.cholesky(cov)
                    angle = np.linspace(0, 2*np.pi, 500)
                    circle = np.column_stack([np.cos(angle), np.sin(angle)])
                    for probability, color in ((.95, '#86b2cc'), (.68, '#0072B2')):
                        ellipse = (np.asarray(approximation['center'])[indices] +
                                   np.sqrt(-2*np.log1p(-probability))*circle @ factor.T)
                        ax.plot(display(ellipse[:, 0], metadata[j]),
                                display(ellipse[:, 1], metadata[i]), color=color, lw=.8)
                elif j == 0 and i == 1:
                    selected = np.linspace(0, len(samples)-1, min(2000, len(samples)), dtype=int)
                    ax.scatter(samples[selected, j], samples[selected, i], s=1,
                               color='#0072B2', alpha=.2, linewidths=0, rasterized=True)
                else:
                    x, y, density = joint_density(samples[:, j], samples[:, i],
                                                  limits[j], limits[i], points=75)
                    levels = probability_levels(density, (.95, .68))
                    if len(levels) == 2:
                        ax.contourf(x, y, density, levels=[*levels, density.max()+1],
                                    colors=['#e7f0f7', '#bad7e8'])
                        ax.contour(x, y, density, levels=levels, colors=['#86b2cc', '#0072B2'], linewidths=.7)
                ax.plot(point[j], point[i], 'x', color='#D55E00', ms=4, mew=.8)
                ax.set(xlim=limits[j], ylim=limits[i])
            if i == n-1:
                ax.set_xlabel(LABELS[metadata[j]['name']], fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != j:
                ax.set_ylabel(LABELS[metadata[i]['name']], fontsize=8)
            elif i != j:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=7, length=2.5, pad=2)
            ax.locator_params(axis='x', nbins=3)
            if i != j:
                ax.locator_params(axis='y', nbins=3)
    label=APPROXIMATIONS[approximation.get('approximation','laplace')] if approximation else 'Retained NUTS'
    handles = [Line2D([], [], color='#0072B2', label=label),
               Line2D([], [], color='.35', ls='--', label='Analytical prior'),
               Line2D([], [], color='#D55E00', marker='x', ls='none', label='MAP')]
    canvas.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .90), fontsize=8)
    canvas.text(.65, .72, r'$A_J^\circ=1\,\mathrm{m}^{-3}\mathrm{s}^{-1}$'+'\n'+r'$A_G^\circ=1\,\mathrm{nm\,min}^{-1}$' +
                ('\n'+r'$t^\circ=1\,\mathrm{min}$' if n == 5 else ''), fontsize=8)


def rejected_panel(canvas, report):
    ax = canvas.add_subplot(111)
    ax.axis('off')
    diagnostic = report['diagnostics']
    eigen = diagnostic.get('eigenvalues', [])
    kind=report.get('approximation','laplace')
    label=APPROXIMATIONS[kind]
    number=lambda value: 'unavailable' if value is None else f'{value:.3g}'
    resolution=('Derivative-verification resolution: '+number(diagnostic.get('curvature_resolution'))
                if kind=='laplace' else 'Covariance-scaled information error: '+number(diagnostic.get('information_error')))
    lines = [r'\textbf{'+label+' approximation rejected}', '', diagnostic['reason']+'.', '',
             'Original inference coordinates; declared priors included.',
             'No Gaussian posterior or covariance is reported.', '',
             *([f"Smallest curvature eigenvalue: {min(eigen):.3g}",
                resolution,
                'Condition number: '+number(diagnostic.get('condition_number')),
                f"Gradient infinity norm: {max(abs(x) for x in diagnostic['gradient']):.3g}"] if eigen else
               ['Curvature was not available; no covariance was constructed.']), '',
             f"{len(report['attempts'])} verification step sizes were attempted.",
             'Checks and target were retained; no jitter was added.', '',
             'Right: retained, screened NUTS draws.',
             'Chain convergence does not validate local Gaussian curvature.']
    # Wrap the long scientific reason at a word boundary for manuscript width.
    import textwrap
    lines[2] = '\n'.join(textwrap.wrap(lines[2], 60))
    ax.text(.08, .85, '\n'.join(lines), transform=ax.transAxes, va='top',
            fontsize=9, linespacing=1.6)


def comparison(model,kind='laplace'):
    folder = BASE/model
    verify_model(folder)
    analysis = ANALYSIS/model
    report_path=analysis/(kind+'.json')
    report = json.loads(report_path.read_text())
    compatibility = verify_analysis_compatibility(
        folder, json.loads((folder/'manifest.json').read_text()), report)
    source = Path(__file__).parent/'protein_analysis.jl'
    producer = ANALYSIS/'producer/laplace.jl'
    if report['analysis_source_sha256'] not in (sha(source), sha(producer)):
        raise ValueError('Uncertainty analysis implementation changed: regenerate its report')
    metadata = json.loads((folder/'priors.json').read_text())
    if report['priors'] != metadata:
        raise ValueError('Approximation/MCMC parameter specifications differ')
    raw = read(folder, 'posterior_draws.csv')
    samples = np.column_stack([np.log(raw[m['name']]) if m['plotting_coordinate']=='log'
                               else raw[m['name']] for m in metadata])
    point = [display(np.array([report['center'][i]]), m)[0] for i, m in enumerate(metadata)]
    cheap = diagnostics = None
    if report['status'] == 'accepted':
        _, theta, cheap, diagnostics = gaussian_draws(report)
    elif report['status'] != 'rejected':
        raise ValueError('Unknown approximation status')
    limits = []
    for i, meta in enumerate(metadata):
        values = samples[:, i] if cheap is None else np.r_[samples[:, i], cheap[:, i]]
        lo, hi = np.quantile(values, [.0005,.9995])
        margin = .15*(hi-lo)
        a,b = support(meta)
        limits.append((max(lo-margin,a),min(hi+margin,b)))
    height = 6.8 if len(metadata)==5 else 5.9
    fig = plt.figure(figsize=(14, height))
    left, right = fig.subfigures(1, 2)
    if cheap is None:
        rejected_panel(left, report)
    else:
        draw_corner(left, cheap, metadata, point, limits, approximation=report)
    draw_corner(right, samples, metadata, point, limits)
    fig.text(.01,.98,'(a)',va='top',fontsize=10)
    fig.text(.51,.98,'(b)',va='top',fontsize=10)
    figure_names=['protein_'+kind+'_mcmc_'+model]
    export(fig, figure_names[-1])
    for label, values in ((kind,cheap),('mcmc',samples)):
        fig=plt.figure(figsize=(7,height))
        if values is None:
            rejected_panel(fig, report)
        else:
            draw_corner(fig, values, metadata, point, limits,
                        approximation=report if label==kind else None)
        prefix=kind+'_' if kind!='laplace' and label=='mcmc' else ''
        figure_names.append('protein_'+prefix+label+'_panel_'+model)
        export(fig, figure_names[-1])
    rows=[]
    for i, meta in enumerate(metadata):
        q=np.quantile(raw[meta['name']],[.025,.5,.975])
        lq=physical(np.asarray(report['center'][i]) + np.sqrt(report['covariance'][i][i]) *
                    np.array([-1.959963984540054,0,1.959963984540054]),meta) if cheap is not None else [None]*3
        rows.append(dict(parameter=meta['name'],units=meta['units'],
                         MCMC_median=q[1],MCMC_q025=q[0],MCMC_q975=q[2],
                         Laplace_status=report['status'],Laplace_median=lq[1],Laplace_q025=lq[0],Laplace_q975=lq[2]))
    write_tables(analysis, rows, samples, cheap, metadata,kind)
    receipt=dict(model=model,approximation=kind,status=report['status'],approximation_sha256=sha(report_path),
                 compatibility=compatibility,
                 posterior_sha256=sha(folder/'posterior_draws.csv'),gaussian_draws=diagnostics,
                 renderer='actual LaTeX; vector PDF rendered at 300 dpi',
                 plotting_source_sha256=sha(Path(__file__)),
                 density_helpers_sha256=sha(Path(__file__).resolve().parents[1]/'_support/posterior_density.py'),
                 figures={f'{name}.{ext}':sha(OUT/f'{name}.{ext}')
                          for name in figure_names for ext in ('pdf','svg','png')},
                 contour_convention='MCMC whitened KDE 68/95%; local Gaussian transformed probability ellipses, not displayed-coordinate HDRs',
                 numerical_production_files='read only; no fit, sampling or latent propagation')
    prefix=kind+'_' if kind!='laplace' else ''
    if kind=='laplace':
        receipt['laplace_sha256']=receipt['approximation_sha256'] # existing receipt readers
    (analysis/(prefix+'comparison_plot.json')).write_text(json.dumps(receipt,indent=2)+'\n')


def write_tables(folder, rows, samples, cheap, metadata,kind='laplace'):
    prefix=kind+'_' if kind!='laplace' else ''
    label=APPROXIMATIONS[kind]
    column='GaussNewton' if kind=='gauss_newton' else 'Laplace'
    csv_rows=[{k.replace('Laplace',column):v for k,v in row.items()} for row in rows]
    with (folder/(prefix+'marginal_comparison.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)
    text=f"| Parameter (units) | Retained NUTS median [95% interval] | {label.replace('--','–')} |\n|:--|:--|:--|\n"
    tex=r'\begin{tabular}{lll}'+'\n'+r'Parameter & NUTS median [95\% interval] & '+label+r' \\'+'\n'+r'\hline'+'\n'
    for row in rows:
        mcmc=f"{row['MCMC_median']:.4g} [{row['MCMC_q025']:.4g}, {row['MCMC_q975']:.4g}]"
        laplace='rejected' if row['Laplace_status']=='rejected' else f"{row['Laplace_median']:.4g} [{row['Laplace_q025']:.4g}, {row['Laplace_q975']:.4g}]"
        text+=f"| `{row['parameter']}` ({row['units']}) | {mcmc} | {laplace} |\n"
        unit=row['units'].replace('²',r'$^2$')
        tex+=row['parameter'].replace('_',r'\_')+' ('+unit+') & '+mcmc+' & '+laplace+r' \\'+'\n'
    (folder/(prefix+'marginal_comparison.md')).write_text(text)
    (folder/(prefix+'marginal_comparison.tex')).write_text(tex+r'\end{tabular}'+'\n')
    correlations=[]
    for i,j in ((0,1),(2,3)):
        correlations.append(dict(first=metadata[i]['name'],second=metadata[j]['name'],
                                 coordinates='display coordinates, including declared natural logs',
                                 MCMC=float(np.corrcoef(samples[:,i],samples[:,j])[0,1]),
                                 **{column:None if cheap is None else float(np.corrcoef(cheap[:,i],cheap[:,j])[0,1])}))
    with (folder/(prefix+'correlation_comparison.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(correlations[0]),lineterminator='\n');writer.writeheader();writer.writerows(correlations)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['cnt','cnt_grd','both'],default='both')
    parser.add_argument('--approximation',choices=list(APPROXIMATIONS),default='laplace')
    args=parser.parse_args(argv)
    configure(require_dvipng=False)
    for model in ('cnt','cnt_grd') if args.model=='both' else (args.model,):
        comparison(model,args.approximation)
    caption=r'''\newcommand{\ProteinLaplaceComparisonCaption}{Full-posterior Laplace verification versus retained NUTS draws for the model identified by the filename. (a) The local approximation is rejected when derivative agreement or weak-curvature resolution fails; no Gaussian draws, covariance or intervals are substituted. (b) Retained NUTS marginal densities and joint summaries, with normalized analytical priors and the inference-coordinate MAP. A separate single-corner panel is provided for legible manuscript layouts. Logarithmic coordinates use the reference units defined in the main corner caption. Curved compensation, prior-bound contact and multiple calibration basins need not be captured by a local approximation, even when its Hessian passes. The MCMC latent bands in the fit figures are unchanged.}'''
    if args.approximation=='gauss_newton':
        caption=r'''\newcommand{\ProteinGaussNewtonComparisonCaption}{Sensitivity-based Gauss--Newton uncertainty versus retained NUTS draws for the model identified by the filename. (a) A joint Gaussian in the original inference coordinates, centred at the retained MAP and using the whitened prediction Jacobian plus the actual transformed-prior curvature. Residual-weighted second prediction derivatives are omitted. Diagonals are exact transformed Gaussian marginal densities; joint curves are transformed 68\% and 95\% Gaussian probability ellipses, not highest-density contours in the displayed physical coordinates. (b) Retained NUTS draws summarized as in the main corner figures. Axis limits, prior overlays and MAP markers match between panels. Gaussian parameter draws are untruncated; prior-support violations are reported. All intervals are local, conditional on the fixed likelihood and discretization. The approximation need not capture prior-bound effects, curved compensation or multiple basins. Existing MCMC latent bands and numerical production results are unchanged.}'''
        publish_gauss_newton_tables()
    (OUT/(args.approximation+'_captions.tex')).write_text(caption+'\n')


def publish_gauss_newton_tables():
    """Small manuscript/manual summaries; immutable production inputs untouched."""
    import shutil
    rows=[]
    for model in ('cnt','cnt_grd'):
        folder=ANALYSIS/model
        if not (folder/'gauss_newton.json').exists():
            continue
        report=json.loads((folder/'gauss_newton.json').read_text())
        d=report['diagnostics']
        rows.append(dict(model=model,status=report['status'],data_rank=d.get('data_rank'),
                         condition_number=d.get('condition_number'),
                         information_error=d.get('information_error'),
                         gradient_verification=d.get('gradient_verification'),
                         forward_solves=d.get('forward_solves')))
        for name in ('marginal_comparison','correlation_comparison'):
            source=folder/('gauss_newton_'+name+'.csv')
            if source.exists():
                shutil.copy2(source,DOCS/('gauss_newton_'+name+'_'+model+'.csv'))
    path=ANALYSIS/'gauss_newton_verification.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    text='| Model | Status | Data Jacobian rank | Precision condition number | Relative information discrepancy |\n|:--|:--|--:|--:|--:|\n'
    for r in rows:
        error='unresolved' if r['information_error'] is None else f"{100*r['information_error']:.3g}%"
        condition='unresolved' if r['condition_number'] is None else f"{r['condition_number']:.3g}"
        text+=f"| {r['model']} | {r['status']} | {r['data_rank']} | {condition} | {error} |\n"
    (ANALYSIS/'gauss_newton_verification.md').write_text(text)
    (DOCS/'gauss_newton_verification.txt').write_text(text)


if __name__=='__main__':
    main()
