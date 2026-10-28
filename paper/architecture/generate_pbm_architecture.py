#!/usr/bin/env python3
"""Source-checked Matplotlib adaptation of the supplied two-panel schematic.

No simulations run. The density glyph is conceptual, not measured data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as GeometryPath
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(PAPER))
from plotting_style import configure, save_png

COLORS = dict(ink='#243548', muted='#5D6B78', line='#C7D0D8',
              dependency_heading='#080808', dependency_category='#303030',
              dependency_package='#555555',
              model='#366A9F', model_fill='#EFF5FB',
              numerics='#287D7E', numerics_fill='#EDF7F5',
              inference='#A4672E', inference_fill='#FCF5EC',
              neutral='#F5F7F9', white='#FFFFFF')


def text(ax, x, y, value, size=7.5, color='ink', ha='center', weight='normal'):
    return ax.text(x, y, value, fontsize=size, color=COLORS.get(color,color),
                   ha=ha, va='center', fontweight=weight, zorder=6, linespacing=1.35)


def rect(ax, x, y, w, h, fill='white', edge='line'):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0,rounding_size=.8',
                 facecolor=COLORS[fill], edgecolor=COLORS[edge], linewidth=.65, zorder=1))


def arrow(ax, points, color='ink', dashed=False):
    path = GeometryPath(points,[GeometryPath.MOVETO]+[GeometryPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=7,
                 linewidth=.8, color=COLORS[color], linestyle='--' if dashed else '-',
                 capstyle='round', joinstyle='round', zorder=4))


def card(ax, x,y,w,h,title,role):
    rect(ax,x,y,w,h,fill=role+'_fill')
    ax.add_patch(Rectangle((x+.7,y+h-1.3),w-1.4,.7,facecolor=COLORS[role],linewidth=0))
    text(ax,x+w/2,y+h-5,title,color=role,weight='bold',size=8)


def density_glyph(ax,x,y,w,h):
    u=np.linspace(0,1,180)
    edges=np.linspace(0,1,10)
    for a,b in zip(edges[:-1],edges[1:]):
        height=np.exp(-(((a+b)/2-.43)/.19)**2)*h
        ax.add_patch(Rectangle((x+a*w,y),(b-a)*w,height,facecolor=COLORS['numerics'],
                              alpha=.18,edgecolor=COLORS['numerics'],linewidth=.4))
    ax.plot(x+u*w,y+h*np.exp(-((u-.43)/.19)**2),color=COLORS['numerics'],lw=1)


def workflow(ax):
    text(ax,3,207,'(a)',size=11,ha='left')
    card(ax,3,104,45,96,'Physical model','model')
    card(ax,54,128,73,72,'Numerical preparation','numerics')
    card(ax,133,112,44,88,'Parameter estimation','inference')
    text(ax,25.5,187,r'\texttt{PBMModel}',color='model',size=8.5)
    text(ax,25.5,179,'Particles, cells,\nother populations')
    text(ax,25.5,165,r'$\displaystyle\frac{\partial n}{\partial t}=\sum_i f_i[n]$',size=13)
    text(ax,25.5,148,'Growth, size diffusion\nNucleation, birth, death\nAggregation, breakage')
    text(ax,25.5,129,'Differential/integral terms,\nsources and sinks',color='muted')
    text(ax,25.5,115,r'$G(x,\ldots),\;D(x,\ldots)$'+'\n'+r'$K(x,y,\ldots)$',size=8)
    rect(ax,57,157,32,32);rect(ax,93,157,31,32)
    text(ax,73,183,'Direct PDE solution',weight='bold')
    text(ax,73,176,r'\texttt{Grid1D / Grid2D}')
    text(ax,73,170,'Bin-integrated populations',size=7)
    density_glyph(ax,61,159,24,7)
    text(ax,108.5,183,'Moment methods',weight='bold')
    text(ax,108.5,176,r'\texttt{MomentSet} + method')
    text(ax,108.5,165,'Raw moments, or\nweights and weighted\nabscissas (DQMOM)',size=7)
    rect(ax,57,131,67,20,edge='numerics')
    text(ax,90.5,146,r'\texttt{TermMethods + compile}',size=8,color='numerics')
    text(ax,90.5,137,'Compatibility, implementation selection,\nreusable numerical preparation',size=7.2)
    arrow(ax,[(73,157),(73,151)],'numerics');arrow(ax,[(108.5,157),(108.5,151)],'numerics')
    arrow(ax,[(48,141),(57,141)],'model')
    rect(ax,54,105,73,19,fill='neutral')
    text(ax,90.5,119,r'Optional \texttt{PBMSystem}: populations + bulk')
    text(ax,90.5,110,r'\texttt{axial\_problem + AxialDispersion}')
    arrow(ax,[(90.5,131),(90.5,124)],'numerics')
    rect(ax,54,78,35,23,edge='numerics');rect(ax,93,78,34,23,edge='numerics')
    text(ax,71.5,96,r'\texttt{PBMProblem}',color='numerics',size=8)
    text(ax,71.5,90,'Initial state, time span')
    text(ax,71.5,83,r'$\downarrow$ \texttt{simulate}',size=8)
    text(ax,110,96,r'\texttt{StationaryPBMProblem}',color='numerics',size=7)
    text(ax,110,90,'Numerical guess')
    text(ax,110,83,r'$\downarrow$ \texttt{solve\_stationary}',size=7.5)
    arrow(ax,[(71.5,105),(71.5,101)],'numerics');arrow(ax,[(110,105),(110,101)],'numerics')
    rect(ax,3,78,45,23,fill='neutral')
    text(ax,25.5,95,'Forward inputs',weight='bold')
    text(ax,25.5,85,'Parameters, prescribed fields,\nphysical boundaries')
    arrow(ax,[(48,89),(54,89)])
    rect(ax,136,171,38,17)
    text(ax,155,183,'Observation operators',weight='bold')
    text(ax,155,175,'Moments, bins\nCDFs, quantiles',size=7)
    arrow(ax,[(71.5,78),(71.5,76),(130,76),(130,179),(136,179)],'numerics')
    arrow(ax,[(110,78),(110,76),(130,76)],'numerics')
    text(ax,168,163,'Data',color='inference')
    arrow(ax,[(155,171),(155,153)],'inference')
    arrow(ax,[(173,160),(155,160)],'inference')
    rect(ax,136,141,38,12,edge='inference')
    text(ax,155,147,'Bayesian MAP fitting',weight='bold')
    arrow(ax,[(155,141),(155,136)],'inference')
    text(ax,155,129,'Gauss--Newton / Laplace\nOptional MCMC / NUTS')
    text(ax,155,118,'Transient and stationary\nforward problems',size=7,color='muted')
    arrow(ax,[(155,112),(155,72),(25.5,72),(25.5,78)],'inference',dashed=True)
    text(ax,78,73.5,'Update fitted parameters',size=7,color='inference')
    text(ax,90,67,'Observation/solver restrictions apply; stationary sensitivities require a smooth isolated root.',
         size=7,color='muted')


def dependencies(ax):
    # Panel (b): dark headings, medium category labels, lighter package names.
    text(ax,3,60,'(b)',size=11,ha='left')
    for x,w in ((3,49),(56,76),(136,41)):
        rect(ax,x,3,w,50,fill='neutral')
    text(ax,27.5,48,'Required dependencies',weight='bold',size=8,color='dependency_heading')
    text(ax,27.5,38,'Julia standard libraries',color='dependency_category')
    text(ax,27.5,32,'LinearAlgebra, Random',color='dependency_package')
    text(ax,27.5,23,'External packages',color='dependency_category')
    text(ax,27.5,15,'Distributions, ForwardDiff\nSpecialFunctions',color='dependency_package')
    text(ax,94,48,'Optional extensions',weight='bold',size=8,color='dependency_heading')
    for x,y,title,body in (
        (59,39,'Optimization','Optim'),
        (97,39,'Stationary solving','NonlinearSolve'),
        (59,26,'External ODE integration','SciMLBase + solver'),
        (97,26,'Continuous sensitivities','SciMLBase,\nSciMLSensitivity,\nReverseDiff'),
        (59,12,'Posterior sampling','AdvancedHMC, MCMCDiagnosticTools')):
        text(ax,x,y,title,size=7,weight='bold',ha='left',color='dependency_category')
        text(ax,x,y-(8 if title=='Continuous sensitivities' else 5),body,size=7,ha='left',color='dependency_package')
    text(ax,156.5,48,'Development tools',weight='bold',size=8,color='dependency_heading')
    text(ax,156.5,38,'Docs and verification',color='dependency_category')
    text(ax,156.5,30,'Documenter, Aqua, JET',color='dependency_package')
    text(ax,156.5,21,'Figure generation',color='dependency_category')
    text(ax,156.5,12,'Python, NumPy, Matplotlib\nPillow, PyMuPDF',color='dependency_package')


def check_architecture():
    project=tomllib.loads((ROOT/'Project.toml').read_text())
    required={'LinearAlgebra','Random','Distributions','ForwardDiff','SpecialFunctions'}
    optional={'Optim','NonlinearSolve','SciMLBase','SciMLSensitivity','ReverseDiff',
              'AdvancedHMC','MCMCDiagnosticTools'}
    if set(project['deps'])!=required or set(project['weakdeps'])!=optional:
        raise ValueError('Dependency roles changed: review the diagram against Project.toml')
    expected={'Optim':['Optim'],'NonlinearSolve':['NonlinearSolve'],'SciMLBase':['SciMLBase'],
              'SciMLSensitivity':['SciMLBase','SciMLSensitivity','ReverseDiff'],
              'AdvancedHMC':['AdvancedHMC','MCMCDiagnosticTools']}
    for name,deps in expected.items():
        actual=project['extensions']['PopulationBalanceModels'+name+'Ext']
        if set([actual] if isinstance(actual,str) else actual)!=set(deps):
            raise ValueError(f'Changed extension activation: {name}')
    return project


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PAPER/'figures/00_architecture')
    parser.add_argument('--usetex',action='store_true',help='Actual LaTeX (also the default)')
    parser.add_argument('--preview',action='store_true',help='Explicit non-publication preview')
    args=parser.parse_args()
    if args.preview and args.usetex:
        parser.error('Choose --preview or --usetex')
    project=check_architecture()
    configure(args.preview,require_dvipng=False)
    fig=plt.figure(figsize=(7.2,8.4))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,180),ylim=(0,210));ax.axis('off')
    workflow(ax);dependencies(ax)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    name='section2_architecture_revised'+('_preview' if args.preview else '')
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    save_png(fig,out/(name+'.png'),creator='generate_pbm_architecture.py')
    plt.close(fig)
    receipt=dict(revision=revision,package_version=project['version'],
                 renderer='preview mathtext' if args.preview else 'actual LaTeX',dpi=300,
                 width_inches=7.2,height_inches=8.4,
                 project_sha256=hashlib.sha256((ROOT/'Project.toml').read_bytes()).hexdigest(),
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 dirty_state=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain=v1'],text=True),
                 scope='Conceptual glyph; shared observation/statistical API for eligible transient and stationary problems; representation, root-branch and derivative restrictions apply')
    (out/(name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    if not args.preview:
        docs=ROOT/'docs/src/assets/architecture';docs.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/(name+'.png'),docs/(name+'.png'))
    print(out)


if __name__=='__main__':
    main()
