"""Plot exported solver comparisons. Requires NumPy, Matplotlib and optional LaTeX."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from plot_style import configure
from plot_case_studies import save_figure

ASSETS = Path(__file__).resolve().parents[1] / "src" / "assets"


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def constant_growth():
    folder = ASSETS / "constant_growth"
    front = read(folder / "errors.csv")
    smooth = read(folder / "smooth_errors.csv")
    choices = [("upwind","euler","Upwind + Euler", "C0"),
               ("muscl","ssprk33","MUSCL + SSPRK3", "C1"),
               ("weno5","ssprk33","WENO5 + SSPRK3", "C2"),
               ("weno5","rk4","WENO5 + RK4", "C3"),
               ("lax_wendroff","coupled","Lax--Wendroff", "C4"),
               ("inguva2022","coupled","Inguva2022 ($C=1$)", "C5")]
    fig, axes = plt.subplots(1,2,figsize=(10,3.8),layout="constrained",sharey=True)
    a=np.linspace(0,1,4001)
    truth=np.where(a>=.4,100*np.exp(-(a-.4)/.01),0.)
    for ax,C in zip(axes,(.5,1.)):
        ax.plot(a,truth,"k-",lw=1.2,label="Analytical density")
        reference=read(folder/f"upwind_euler_{C}_200.csv")
        ax.plot(reference["size"],reference["exact"],"k.",ms=3,label="Analytical bin averages")
        for name,time,label,color in choices:
            if name=="inguva2022" and C!=1:continue
            d=read(folder/f"{name}_{time}_{C}_200.csv")
            ax.plot(d['size'],d['density'],color=color,lw=1.2,label=label)
        ax.set(xlim=(.32,.62),xlabel="$a$",title=fr"$C={C:g}$, 200 bins")
    axes[0].set_ylabel("$f(a,4)$");axes[1].legend(fontsize=7)
    save_figure(fig,folder,"solutions")

    # Two physical inputs, with the SAME two error norms for each.
    # Exact shifts have roundoff-level cell-integral errors, not a power-law order.
    for source,filename,title in [(front,'performance','Exponential with zero inflow'),
                                  (smooth,'smooth_refinement','Smooth periodic translation')]:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
        for name,time,label,color in choices:
            data=source[(source['scheme']==name)&(source['time_method']==time)]
            if source is front:data=data[data['courant']==(1. if name=='inguva2022' else .5)]
            for ax,key in zip(axes,['l1','rmse']):
                # Display roundoff as roundoff, without a fitted convergence line.
                ax.loglog(data['bins'],np.maximum(data[key],1e-16),'o-',ms=3,color=color,label=label)
        axes[0].set(xlabel='Number of bins',ylabel=r'$L^1$ density error')
        axes[1].set(xlabel='Number of bins',ylabel='RMSE')
        axes[0].legend(fontsize=7);fig.suptitle(title)
        save_figure(fig,folder,filename)

    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for name,time,label,color in choices:
        data=smooth[(smooth['scheme']==name)&(smooth['time_method']==time)]
        axes[0].loglog(data['bins'],data['seconds'],'o-',ms=3,color=color,label=label)
        axes[1].loglog(data['seconds'],np.maximum(data['rmse'],1e-16),'o-',ms=3,color=color,label=label)
    axes[0].set(xlabel='Number of bins',ylabel=r'Elapsed time $(\mathrm{s})$',title='One measured solve per configuration')
    axes[1].set(xlabel=r'Elapsed time $(\mathrm{s})$',ylabel='RMSE',title='Error versus elapsed work')
    axes[0].legend(fontsize=7);save_figure(fig,folder,'timings')

    # Measured refinement orders explain the curves without arbitrary h^p guides.
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained",sharey=True)
    for name,time,label,color in choices:
        if name=='inguva2022':continue # order is undefined at roundoff
        data=smooth[(smooth['scheme']==name)&(smooth['time_method']==time)]
        for ax,key in zip(axes,['observed_l1_order','observed_rmse_order']):
            ax.semilogx(data['bins'],data[key],'o-',ms=3,color=color,label=label)
    axes[0].set(xlabel='Number of bins',ylabel='Measured refinement order',title=r'$L^1$')
    axes[1].set(xlabel='Number of bins',title='RMSE');axes[0].legend(fontsize=7)
    save_figure(fig,folder,'observed_orders')

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
    args = parser.parse_args()
    configure(usetex=args.usetex)
    for name, plot in (("constant", constant_growth), ("cooling", cooling), ("axial", axial)):
        if args.case in (name, "all"):
            plot()
