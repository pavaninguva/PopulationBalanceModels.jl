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
    errors = read(folder / "errors.csv")
    colors = {name: f"C{i}" for i, name in enumerate(
        ("upwind", "muscl", "vanleer", "weno3", "weno5", "centered",
         "linear_upwind", "lax_wendroff", "beam_warming"))}
    def appearance(name, time):
        return dict(color=colors[name], linestyle={"euler": ":", "ssprk33": "--",
                    "ssprk54": "-.", "rk4": "-", "coupled": "-"}[time])
    choices = [("upwind", "euler", "Upwind + Euler"),
               ("upwind", "rk4", "Upwind + RK4"),
               ("muscl", "ssprk33", "MUSCL + SSPRK3"),
               ("muscl", "rk4", "MUSCL + RK4"),
               ("vanleer", "rk4", "van Leer + RK4"),
               ("weno3", "rk4", "WENO3 + RK4"),
               ("weno5", "rk4", "WENO5 + RK4"),
               ("centered", "rk4", "Centered + RK4"),
               ("linear_upwind", "rk4", "Linear upwind + RK4"),
               ("lax_wendroff", "coupled", "Lax--Wendroff"),
               ("beam_warming", "coupled", "Beam--Warming")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for index, ax in enumerate(axes.flat):
        courant = (0.5, 1.0)[index % 2]
        profile = read(folder / f"upwind_euler_{courant}_200.csv")
        # Draw the continuous analytical density, including the zero-inflow front.
        a = np.linspace(0, 1, 4001)
        exact = np.where(a >= 0.4, 100 * np.exp(-(a - 0.4) / 0.01), 0)
        ax.plot(a, exact, "k-", lw=1.3, label="Analytical density")
        ax.plot(profile["size"], profile["exact"], "k.", ms=2.5, label="Analytical bin averages")
        for name, time, label in choices:
            if (index < 2) != (name in ("upwind", "muscl", "vanleer", "weno3", "weno5")):
                continue
            if name == "linear_upwind" and courant == 1.0:
                continue
            data = read(folder / f"{name}_{time}_{courant}_200.csv")
            ax.plot(data["size"], data["density"], lw=1.1, label=label, **appearance(name,time))
        if index == 3:
            data = read(folder / "inguva2022_coupled_1.0_200.csv")
            ax.plot(data["size"], data["density"], "--", label="Inguva2022, matched shift")
        ax.set(xlim=(0.32, 0.62), xlabel="$a$", ylabel="$f(a,4)$",
               title=fr"$C={courant:g}$, 200 bins")
        ax.legend(fontsize=7)
    save_figure(fig, folder, "solutions")

    smooth = read(folder / "smooth_errors.csv")
    choices += [("weno5", "ssprk33", "WENO5 + SSPRK3"),
                ("weno5", "ssprk54", "WENO5 + SSPRK(5,4)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name, time, label in choices:
        front = errors[(errors["scheme"] == name) & (errors["time_method"] == time)
                       & (errors["courant"] == 0.5)]
        data = smooth[(smooth["scheme"] == name) & (smooth["time_method"] == time)
                      & (smooth["step_policy"] == "fixed_cfl")]
        axes[0].loglog(front["bins"], front["rmse"], marker="o", ms=3, label=label, **appearance(name,time))
        axes[1].loglog(data["bins"], data["rmse"], marker="o", ms=3, label=label, **appearance(name,time))
    axes[0].set(title="Exponential with a moving front", xlabel="Number of bins", ylabel="RMSE")
    axes[1].set(title="Smooth periodic translation", xlabel="Number of bins", ylabel="RMSE")
    axes[0].legend(fontsize=7)
    save_figure(fig, folder, "performance")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for name, time, label in choices:
        data = smooth[(smooth["scheme"] == name) & (smooth["time_method"] == time)
                      & (smooth["step_policy"] == "fixed_cfl")]
        axes[0].loglog(data["bins"], data["seconds"], marker="o", ms=3, label=label, **appearance(name,time))
        axes[0].fill_between(data["bins"],data["timing_minimum"],data["timing_maximum"],alpha=.1,color=colors[name])
        axes[1].loglog(data["seconds"], data["rmse"], marker="o", ms=3, label=label, **appearance(name,time))
    axes[0].set(xlabel="Number of bins", ylabel="Elapsed time / s", title="Median of three warmed runs")
    axes[1].set(xlabel="Elapsed time / s", ylabel="RMSE", title="Error versus work")
    axes[0].legend(fontsize=7)
    save_figure(fig, folder, "timings")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for name, time, label in (("muscl","rk4","MUSCL + RK4"),
                             ("vanleer","rk4","van Leer + RK4"),
                             ("weno5","ssprk33","WENO5 + SSPRK3"),
                             ("weno5","rk4","WENO5 + RK4")):
        data = smooth[(smooth["scheme"] == name) & (smooth["time_method"] == time)
                      & (smooth["step_policy"] == "fixed_cfl")]
        axes[0].loglog(data["bins"], data["l1"], "o-", ms=3, label=label)
    for policy,label in (("fixed_cfl",r"$\Delta t\propto h$"),("spatial",r"$\Delta t\propto h^{5/4}$")):
        data = smooth[(smooth["scheme"] == "weno5") & (smooth["time_method"] == "rk4")
                      & (smooth["step_policy"] == policy)]
        axes[1].loglog(data["bins"], data["rmse"], "o-", ms=3,label="WENO5 + RK4, "+label)
    x=np.array([80.,640.])
    axes[0].loglog(x,1e-3*(x/80)**-2,"k:",label=r"$h^2$ guide")
    axes[1].loglog(x,3e-6*(x/80)**-5,"k:",label=r"$h^5$ guide")
    axes[0].set(xlabel="Number of bins",ylabel=r"$L^1$ density error",title="Limited reconstructions near smooth extrema")
    axes[1].set(xlabel="Number of bins",ylabel="RMSE",title="Spatial and temporal refinement")
    for ax in axes: ax.legend(fontsize=8)
    save_figure(fig,folder,"smooth_refinement")


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
