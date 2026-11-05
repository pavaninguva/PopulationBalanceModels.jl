"""Plot exported solver comparisons. Requires NumPy, Matplotlib and optional LaTeX."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from plot_style import configure
from pbm_plotting import plot_observable
from plot_case_studies import save_figure

ASSETS = Path(__file__).resolve().parents[1] / "src" / "assets"


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def constant_growth(input_path=None,output_dir=None):
    # Both manuals and manuscript plots select this same retained Gaussian.
    import sys
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'paper'))
    from _support.growth_results import load_results,refinement,profile,observed_orders
    from pbm_plotting.style import method_style,panel_letters
    data=load_results(input_path)
    folder=Path(output_dir) if output_dir is not None else ASSETS/'constant_growth'
    cfg,benchmark=data['configuration'],data['benchmark']
    names=[('upwind','Upwind + Euler'),('muscl','MUSCL + SSPRK3'),
           ('weno3','WENO3 + SSPRK3'),('weno5_ssprk33','WENO5 + SSPRK3'),
           ('weno5','WENO5 + RK4'),('hoc','HOC FV + midpoint'),
           ('inguva2022',r'Inguva2022 ($\mathrm{CFL}=1$)')]

    def options(method,label,*,markers=True):
        result=method_style(method,markers=markers)
        result['label']=label
        if method=='hoc':result.update(ls='--',lw=1.8,zorder=5)
        return result

    def panels(*,legend=False):
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.7 if legend else 3.25))
        fig.subplots_adjust(left=.11,right=.985,bottom=.29 if legend else .18,
                            top=.92,wspace=.40)
        panel_letters(axes)
        return fig,axes

    def shared_legend(fig,axes):
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',
                   ncol=3,fontsize=6.6,handlelength=2.2,columnspacing=1.2)

    fig,axes=panels()
    a=np.linspace(0,1,4001)
    original=a-benchmark['growth']*benchmark['final_time']
    truth=np.where((original>=0)&(original<=1),benchmark['height']*
                   np.exp(-.5*((original-benchmark['center'])/benchmark['standard_deviation'])**2),0.)
    for ax,courant in zip(axes,cfg['profile_courants']):
        ax.plot(a,truth,color='.5',lw=1.2,label='Analytical density')
        reference=profile(data,'upwind',courant)
        x=(reference['left']+reference['right'])/2
        ax.plot(x,reference['exact_density'],'k.',ms=2,label='Analytical bin averages')
        for method,label in names:
            if method=='inguva2022' and courant!=1:continue
            curve=profile(data,method,courant)
            plot_observable((curve['left']+curve['right'])/2,curve['density'],ax=ax,
                            styles=[options(method,label,markers=False)],legend=False)
        ax.axhline(0,color='.75',lw=.5)
        ax.set(xlim=(0,1),xlabel=r'Size coordinate $a$',ylabel=r'$n(a,4\,\mathrm{s})$')
        ax.text(.97,1.035,rf'$\mathrm{{CFL}}={courant:g},\ N={cfg["profile_bins"]}$',
                transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
    limits=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:ax.set_ylim(limits)
    axes[1].set_ylabel('')
    axes[1].legend(loc='upper left',fontsize=6.2,handlelength=2.0,labelspacing=.3)
    save_figure(fig,folder,'solutions')

    fig,axes=panels(legend=True)
    for method,label in names:
        values=refinement(data,method)
        for ax,key in zip(axes,('l1','rmse')):
            style=options(method,label)
            if method=='inguva2022':style['ls']='none' # no fitted trend through roundoff
            ax.loglog(values['bins'],np.where(values[key]>0,values[key],np.nan),**style)
    axes[0].set(xlabel='Number of bins',ylabel=r'$L^1$ density error')
    axes[1].set(xlabel='Number of bins',ylabel='RMSE')
    for ax in axes:
        ax.set_xticks(cfg['bins'],labels=[str(n) for n in cfg['bins']])
        ax.xaxis.set_minor_locator(NullLocator())
    shared_legend(fig,axes)
    save_figure(fig,folder,'performance')

    fig,axes=panels(legend=True)
    orders=observed_orders(data)
    for method,label in names:
        if method=='inguva2022':continue
        values=[row for row in orders if row['method']==method]
        for ax,key in zip(axes,('observed_l1_order','observed_rmse_order')):
            ax.semilogx([row['bins'] for row in values],[row[key] for row in values],
                        **options(method,label))
    axes[0].set(xlabel='Number of bins',ylabel=r'Measured order in $L^1$')
    axes[1].set(xlabel='Number of bins',ylabel='Measured order in RMSE')
    for ax in axes:
        ax.set_xticks(cfg['bins'][1:],labels=[str(n) for n in cfg['bins'][1:]])
        ax.xaxis.set_minor_locator(NullLocator())
    shared_legend(fig,axes)
    save_figure(fig,folder,'observed_orders')

    fig,axes=panels()
    for method,label in names:
        values=refinement(data,method)
        axes[0].loglog(values['bins'],values['seconds'],**options(method,label))
        axes[1].loglog(values['seconds'],values['rmse'],**options(method,label))
    axes[0].set(xlabel='Number of bins',ylabel='Integration wall time (s)')
    axes[1].set(xlabel='Integration wall time (s)',ylabel='RMSE')
    axes[0].set_xticks(cfg['bins'],labels=[str(n) for n in cfg['bins']])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylim(top=max(1.,axes[0].get_ylim()[1]))
    axes[0].legend(loc='upper left',fontsize=6.2,handlelength=2.0,labelspacing=.3)
    save_figure(fig,folder,'timings')

def cooling():
    folder = ASSETS / "gunawan_example2"
    runs = read(folder / "moments.csv")
    moments = read(folder / "reference.csv")
    ref = read(folder / "reference_psd.csv")
    errors = read(folder / "errors.csv")
    finest = int(np.max(errors[errors["method"] == "muscl"]["bins"]))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    axes[0, 0].plot(moments["time"], moments["concentration"], "k-", label="Closed moment ODEs")
    for name, label in (("upwind", "Upwind"), ("muscl", "MUSCL")):
        data = runs[(runs["method"] == name) & (runs["bins"] == finest)]
        axes[0, 0].plot(data["time"], data["concentration"], "--", label=label)
        for ax in axes[1, :]:
            psd = read(folder / f"{name}_{finest}_psd.csv")
            ax.plot(psd["length"], psd["density"], label=f"{label}, {finest} bins")
    for ax in axes[1, :]:
        ax.plot(ref["length"], ref["density"], "k--", lw=1.2, label="Fine-grid reference")
        ax.set(xlabel=r"$L\,/\,\mu\mathrm{m}$", ylabel=r"$f(L,1000)$")
    axes[1, 0].set(xlim=(0, 35), title="Newly nucleated crystals")
    grown = (ref["length"] > 150) & (ref["density"] > 1e-4 * np.max(ref["density"][ref["length"] > 150]))
    low, high = np.min(ref["length"][grown]), np.max(ref["length"][grown])
    margin = 0.15 * (high - low)
    axes[1, 1].set(xlim=(low - margin, high + margin), title="Growth of the initial population")
    axes[1, 1].legend(fontsize=8)
    fine = runs[runs["method"] == "reference"]
    for k in range(4):
        relative = np.abs(fine[f"mu{k}"] / moments[f"mu{k}"] - 1)
        axes[0, 1].semilogy(fine["time"], np.maximum(relative, 1e-16), label=fr"$\mu_{k}$")
    axes[0, 0].set(xlabel=r"$t\,/\,\mathrm{s}$", ylabel=r"$c\,/\,(\mathrm{g/g\ water})$",
                   title="Coupled concentration")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel=r"$t\,/\,\mathrm{s}$", ylabel="Relative moment error",
                   title="Fine-grid reference versus moment ODEs")
    axes[0, 1].legend(fontsize=8)
    save_figure(fig, folder, "coupled_solution")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for method in ("upwind", "muscl"):
        data = errors[errors["method"] == method]
        axes[0].loglog(1600 / data["bins"], data["mean_absolute_error"], "o-", label=method.upper())
        axes[1].loglog(1600 / data["bins"], data["relative_mu3_error"], "o-", label=method.upper())
    axes[0].set(xlabel=r"Nominal $\Delta L\,/\,\mu\mathrm{m}$", ylabel="Mean absolute density error",
                title="Against the finer sectional distribution")
    axes[1].set(xlabel=r"Nominal $\Delta L\,/\,\mu\mathrm{m}$", ylabel=r"Relative error in $\mu_3$",
                title="Against independent moment ODEs")
    for ax in axes:
        ax.invert_xaxis()
        ax.legend()
    save_figure(fig, folder, "refinement")


def axial():
    folder = ASSETS / "axial_growth"
    data = read(folder / "solution.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, t in zip(axes, (1, 3, 5)):
        d = data[data["time"] == t]
        z, a = np.unique(d["position"]), np.unique(d["size"])
        image = ax.pcolormesh(z, a, d["density"].reshape(len(z), len(a)).T,
                              shading="nearest", cmap="viridis", vmin=0, vmax=1)
        ax.set(xlabel=r"$z\,/\,\mathrm{m}$", ylabel=r"$a\,/\,\mathrm{mm}$", title=fr"$t={t}\,\mathrm{{s}}$")
        fig.colorbar(image, ax=ax, label="$f(a,z,t)$")
    save_figure(fig, folder, "solution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("case", choices=("constant", "cooling", "axial", "all"))
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--input",type=Path,help="Explicit Gaussian calculation or retained selection")
    parser.add_argument("--output",type=Path,help="Separate output directory for constant-growth figures")
    args = parser.parse_args()
    with configure(usetex=args.usetex):
        for name, plot in (("constant", constant_growth), ("cooling", cooling), ("axial", axial)):
            if args.case in (name, "all"):
                plot(args.input,args.output) if name=="constant" else plot()
