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

ARCHITECTURE_SOURCES = tuple(Path(path) for path in (
    'src/PopulationBalanceModels.jl', 'src/compilation/moments.jl',
    'src/numerics/moments/automatic.jl', 'src/time_integration/initialization.jl',
    'src/time_integration/preparation.jl', 'src/time_integration/forward_execution.jl',
    'src/time_integration/operating_history.jl', 'src/modeling/laws.jl',
    'src/coupling/systems.jl', 'src/compilation/axial.jl',
    'src/stationary/problems.jl', 'src/inference/stationary.jl',
    'src/inference/parameters.jl', 'src/uq/propagation.jl',
))


COLORS = dict(ink='#243548', muted='#5D6B78', line='#C7D0D8',
              dependency_heading='#080808', dependency_category='#303030',
              dependency_package='#555555',
              model='#366A9F', model_fill='#EFF5FB',
              numerics='#287D7E', numerics_fill='#EDF7F5',
              inference='#A4672E', inference_fill='#FCF5EC',
              uq='#70548C', uq_fill='#F4F0F8',
              neutral='#F5F7F9', white='#FFFFFF')


def text(ax, x, y, value, size=7.5, color='ink', ha='center', weight='normal',
         family=None, usetex=None):
    use_tex = plt.rcParams['text.usetex'] if usetex is None else usetex
    if use_tex and '$' not in value:
        value = value.replace('_', r'\_')
    return ax.text(x, y, value, fontsize=size, color=COLORS.get(color,color),
                   ha=ha, va='center', fontweight=weight, fontfamily=family,
                   usetex=use_tex, zorder=6, linespacing=1.35)


def rect(ax, x, y, w, h, fill='white', edge='line', dashed=False):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0,rounding_size=.8',
                 facecolor=COLORS[fill], edgecolor=COLORS[edge], linewidth=.65,
                 linestyle='--' if dashed else '-', zorder=1))


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


def workflow(ax, *, equation_usetex=True):
    text(ax,3,207,'(a)',size=11,ha='left')
    card(ax,3,104,45,96,'Physical model','model')
    card(ax,54,141,73,59,'Numerical preparation','numerics')
    card(ax,133,112,44,88,'Parameter estimation','inference')
    text(ax,25.5,187,'PBMModel',color='model',size=8.5,family='monospace')
    text(ax,25.5,179,'Particles, cells,\nother populations')
    equation = (r'$\displaystyle\frac{\partial n}{\partial t}=\sum_i f_i[n]$'
                if equation_usetex else r'$\frac{\partial n}{\partial t}=\sum_i f_i[n]$')
    text(ax,25.5,165,equation,size=13,
         family='serif' if equation_usetex else 'DejaVu Serif',usetex=equation_usetex)
    text(ax,25.5,144,'Growth\nGrowth Rate Dispersion\nNucleation and Birth\n'
         'Death and Dilution\nAggregation and Breakage')
    text(ax,25.5,125,'Rate, kernel and daughter laws',color='muted',size=7)
    text(ax,25.5,115,r'$G(x,\ldots),\;D(x,\ldots)$'+'\n'+r'$K(x,y,\ldots)$',size=8)

    # Selection belongs to each route. MomentSet is a basis, not a universal
    # state layout; automatic mixed moment evaluation is prepared by compile.
    rect(ax,57,165,32,26.5);rect(ax,93,165,31,26.5)
    text(ax,73,188,'Direct discretization',weight='bold',size=7)
    text(ax,73,182.5,'Grid1D / Grid2D',family='monospace',size=7)
    text(ax,73,178,'TermMethods',family='monospace',size=7,color='numerics')
    density_glyph(ax,61,166,24,7.5)
    text(ax,108.5,188,'Moment formulations',weight='bold',size=7)
    text(ax,108.5,180,'MomentSet',family='monospace',size=7)
    text(ax,108.5,171,'moment_method',family='monospace',size=7,color='numerics')
    rect(ax,57,144,67,16,edge='numerics')
    text(ax,90.5,156,'compile',size=8,color='numerics',family='monospace')
    text(ax,90.5,149,'Compatibility checks and numerical preparation',size=7)
    arrow(ax,[(73,165),(73,160)],'numerics')
    arrow(ax,[(108.5,165),(108.5,160)],'numerics')
    arrow(ax,[(48,152),(57,152)],'model')

    # Solid route: standalone population. Dashed side inputs are optional and
    # equally prominent; neither is required before constructing a problem.
    rect(ax,54,120,31,18,fill='neutral',dashed=True)
    rect(ax,96,120,31,18,fill='neutral',dashed=True)
    text(ax,69.5,134,'Optional coupling',weight='bold',size=7)
    text(ax,69.5,128.5,'PBMSystem',size=7.5,family='monospace')
    text(ax,69.5,123,'Populations + bulk',size=7)
    text(ax,111.5,134,'Optional transport',weight='bold',size=7)
    text(ax,111.5,128.5,'AxialDispersion',size=7.5,family='monospace')
    text(ax,111.5,123,'compile / axial_problem',size=6.8)
    arrow(ax,[(90.5,144),(90.5,119)],'numerics')
    arrow(ax,[(85,128),(90.5,128)],'numerics',dashed=True)
    arrow(ax,[(96,128),(90.5,128)],'numerics',dashed=True)

    # Keep only problem inputs and solve calls in the overview. Preparation,
    # execution controls and eligibility details belong to the caption/guide.
    # No arrows between the labels inside these branches.
    rect(ax,54,91,35,24,edge='numerics');rect(ax,93,91,34,24,edge='numerics')
    text(ax,71.5,110,'PBMProblem',color='numerics',size=8,family='monospace')
    text(ax,71.5,103,'Initial state\nand time span',size=7)
    text(ax,71.5,96.5,'simulate',size=8,family='monospace')
    text(ax,110,110,'StationaryPBMProblem',color='numerics',size=7,family='monospace')
    text(ax,110,103,'Numerical guess',size=7)
    text(ax,110,96.5,'solve_stationary',size=7.5,family='monospace')
    arrow(ax,[(90.5,119),(71.5,119),(71.5,115)],'numerics')
    arrow(ax,[(90.5,119),(110,119),(110,115)],'numerics')
    rect(ax,3,86,45,15,fill='neutral')
    text(ax,25.5,96,'Forward inputs',weight='bold')
    text(ax,25.5,90.5,'Parameters, prescribed fields,\nphysical boundaries',size=7)
    arrow(ax,[(48,94),(50.5,94),(50.5,119),(90.5,119)])

    rect(ax,136,171,38,17)
    text(ax,155,183,'Observation operators',weight='bold')
    text(ax,155,176,'Moments, bins\nCDFs, quantiles',size=7)
    arrow(ax,[(71.5,91),(71.5,89),(130,89),(130,179),(136,179)],'numerics')
    # Shared output connector has no extra arrowhead under solve_stationary.
    ax.plot([110,110,130],[91,89,89],color=COLORS['numerics'],lw=.8,zorder=3)
    text(ax,168,163,'Data',color='inference')
    arrow(ax,[(155,171),(155,153)],'inference')
    arrow(ax,[(173,160),(155,160)],'inference')
    rect(ax,136,141,38,12,edge='inference')
    text(ax,155,147,'Bayesian MAP fitting',weight='bold')
    arrow(ax,[(155,141),(155,136)],'inference')
    text(ax,155,134,'Optional uncertainty',weight='bold',size=7)
    text(ax,155,127,'Gauss–Newton / Laplace\nMCMC / NUTS')
    text(ax,155,118,'Transient and stationary\nforward problems',size=7,color='muted')
    arrow(ax,[(155,112),(155,81),(25.5,81),(25.5,86)],'inference',dashed=True)
    text(ax,78,83,'Update fitted parameters',size=7,color='inference')


def forward_uq(ax):
    """A wrapper around the forward problems, separate from fitting data."""
    card(ax,3,44,174,32,'Forward uncertainty quantification','uq')
    for x,w in ((6,45),(59,57),(124,50)):
        rect(ax,x,48,w,16,edge='uq')
    text(ax,28.5,60,'UncertainInputs',color='uq',size=8,family='monospace')
    text(ax,28.5,52.5,'Prescribed joint law\nor posterior draws',size=7)
    text(ax,87.5,60,'UQProblem: scenario setup',color='uq',size=8)
    text(ax,87.5,52.5,'Reuse forward solves + observables\nMonte Carlo / non-intrusive PCE',size=7)
    text(ax,149,60,'Outputs and validation',color='uq',size=8)
    text(ax,149,52.5,'Means, covariance, quantiles, events\nIndependent PCE validation',size=7)
    arrow(ax,[(51,56),(59,56)],'uq')
    arrow(ax,[(116,56),(124,56)],'uq')
    # Uncertain scenario inputs enter the same forward construction above.
    # This stays clear of the fitted-parameter return path at y=81.
    arrow(ax,[(10,76),(10,86)],'uq')


def dependencies(ax):
    # Panel (b): dark headings, medium category labels, lighter package names.
    origin=-22
    text(ax,3,origin+60,'(b)',size=11,ha='left')
    for x,w in ((3,49),(56,76),(136,41)):
        rect(ax,x,origin+3,w,50,fill='neutral')
    text(ax,27.5,origin+48,'Required dependencies',weight='bold',size=8,color='dependency_heading')
    text(ax,27.5,origin+38,'Julia standard libraries',color='dependency_category')
    text(ax,27.5,origin+32,'LinearAlgebra, Random',color='dependency_package')
    text(ax,27.5,origin+23,'External packages',color='dependency_category')
    text(ax,27.5,origin+15,'Distributions, ForwardDiff\nSpecialFunctions',color='dependency_package')
    text(ax,27.5,origin+6,'MC/PCE: no additional backend',size=7,color='dependency_package')
    text(ax,94,origin+48,'Optional extensions',weight='bold',size=8,color='dependency_heading')
    for x,y,title,body in (
        (59,39,'Optimization','Optim'),
        (97,39,'Stationary solving','NonlinearSolve'),
        (59,26,'External ODE integration','SciMLBase + solver'),
        (97,26,'Continuous sensitivities','SciMLBase, SciMLSensitivity\nReverseDiff'),
        (59,12,'Posterior sampling','AdvancedHMC\nMCMCDiagnosticTools'),
        (97,12,'Plotting','PythonPlot (PythonCall)')):
        text(ax,x,origin+y,title,size=7,weight='bold',ha='left',color='dependency_category')
        text(ax,x,origin+y-(6.5 if title=='Continuous sensitivities' else 5),body,size=7,ha='left',color='dependency_package')
    text(ax,156.5,origin+48,'Development tools',weight='bold',size=8,color='dependency_heading')
    text(ax,156.5,origin+38,'Docs and verification',color='dependency_category')
    text(ax,156.5,origin+30,'Documenter, Aqua, JET',color='dependency_package')
    text(ax,156.5,origin+21,'Figure generation',color='dependency_category')
    text(ax,156.5,origin+12,'Python, NumPy, Matplotlib\nPillow, PyMuPDF',color='dependency_package')


def check_architecture():
    project=tomllib.loads((ROOT/'Project.toml').read_text())
    required={'LinearAlgebra','Random','Distributions','ForwardDiff','SpecialFunctions'}
    optional={'Optim','NonlinearSolve','SciMLBase','SciMLSensitivity','ReverseDiff',
              'AdvancedHMC','MCMCDiagnosticTools','PythonPlot','PythonCall'}
    if set(project['deps'])!=required or set(project['weakdeps'])!=optional:
        raise ValueError('Dependency roles changed: review the diagram against Project.toml')
    expected={'Optim':['Optim'],'NonlinearSolve':['NonlinearSolve'],'SciMLBase':['SciMLBase'],
              'SciMLSensitivity':['SciMLBase','SciMLSensitivity','ReverseDiff'],
              'AdvancedHMC':['AdvancedHMC','MCMCDiagnosticTools'],
              'PythonPlot':['PythonPlot','PythonCall']}
    for name,deps in expected.items():
        actual=project['extensions']['PopulationBalanceModels'+name+'Ext']
        if set([actual] if isinstance(actual,str) else actual)!=set(deps):
            raise ValueError(f'Changed extension activation: {name}')
    missing=[str(path) for path in ARCHITECTURE_SOURCES if not (ROOT/path).is_file()]
    if missing:
        raise ValueError('Architecture review sources moved: '+', '.join(missing))
    return project


def configure_schematic(usetex=False):
    """Use the original softer sans-serif lettering, with optional external TeX."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.serif': ['serif'],  # Computer Modern for the per-artist LaTeX equation.
        'font.sans-serif': ['sans-serif'] if usetex else ['DejaVu Sans'],
        'font.monospace': ['Courier'] if usetex else ['DejaVu Sans Mono'],
        'mathtext.fontset': 'stix',
    })


def build_figure(*, equation_usetex=True):
    """Compose the reviewed diagram; caller owns the scoped font/render style."""
    fig=plt.figure(figsize=(7.2,8.2))
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,180),ylim=(-22,210));ax.axis('off')
    workflow(ax,equation_usetex=equation_usetex)
    forward_uq(ax);dependencies(ax)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PAPER/'figures/00_architecture')
    parser.add_argument('--usetex',action='store_true',help='Opt in to external LaTeX instead of Matplotlib lettering')
    parser.add_argument('--preview',action='store_true',help='Use mathtext for the equation too; write a separate preview without updating docs')
    args=parser.parse_args()
    if args.preview and args.usetex:
        parser.error('Choose --preview or --usetex')
    if not args.preview and not shutil.which('latex'):
        parser.error('The equation requires LaTeX; install latex or use --preview for mathtext')
    project=check_architecture()
    with configure(preview=not args.usetex):
        configure_schematic(usetex=args.usetex)
        fig=build_figure(equation_usetex=not args.preview)
        out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
        name='architecture'+('_preview' if args.preview else '')
        receipt_name='section2_architecture_revised'+('_preview' if args.preview else '')
        revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        export=save_png(fig,out/(name+'.png'),creator='generate_pbm_architecture.py')
        receipt=dict(revision=revision,package_version=project['version'],
                     renderer=('actual LaTeX' if args.usetex else 'Matplotlib mathtext'
                               if args.preview else 'Matplotlib text with a LaTeX equation'),
                     equation_renderer='Matplotlib mathtext' if args.preview else 'actual LaTeX',
                     font_family='sans-serif' if args.usetex else 'DejaVu Sans',
                     math_font=('external LaTeX' if args.usetex else 'STIX' if args.preview
                                else 'Computer Modern equation; STIX rate symbols'),
                     preview=args.preview,dpi=300,png_filename=name+'.png',**export,
                     width_inches=7.2,height_inches=8.2,
                     project_sha256=hashlib.sha256((ROOT/'Project.toml').read_bytes()).hexdigest(),
                     generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     png_sha256=hashlib.sha256((out/(name+'.png')).read_bytes()).hexdigest(),
                     architecture_source_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                         for path in ARCHITECTURE_SOURCES},
                     uq_source_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                         for path in (Path('src/uq/inputs.jl'),Path('src/uq/propagation.jl'),
                                      Path('src/uq/polynomial_chaos.jl'),Path('src/uq/bases.jl'),
                                      Path('src/uq/refinement.jl'))},
                     dirty_state=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain=v1'],text=True),
                     scope='Conceptual glyph; direct/moment method selection and automatic mixed moment evaluation; physical initialization, parameter-bound run preparation, configured transient execution and optional operating history; standalone forward path with optional population/bulk coupling and optional axial transport; shared transient/stationary forward problems and observations for estimation and forward MC/PCE; built-in PCE bases cover independent normal/uniform/beta reference marginals with an explicit physical-input map; representation, root-branch and derivative restrictions apply')
        (out/(receipt_name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        if not args.preview:
            docs=ROOT/'docs/src/assets/architecture';docs.mkdir(parents=True,exist_ok=True)
            docs_name='section2_architecture_revised'
            shutil.copy2(out/(name+'.png'),docs/(docs_name+'.png'))
            # Preserve the existing downloadable documentation formats, while
            # keeping the paper output PNG-only. They use the identical canvas.
            for ext in ('pdf','svg'):
                destination=docs/(docs_name+'.'+ext)
                fig.savefig(destination,dpi=300)
                if ext=='svg':
                    destination.write_text('\n'.join(
                        line.rstrip() for line in destination.read_text().splitlines())+'\n')

        plt.close(fig)
        print(out)


if __name__=='__main__':
    main()
