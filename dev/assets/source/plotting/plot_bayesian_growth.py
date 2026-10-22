"""Plot the recorded nonnegative truncated-Gaussian growth study."""
from pathlib import Path
import argparse
import csv
from statistics import NormalDist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_style import configure

ROOT = Path(__file__).resolve().parents[1] / "src" / "assets" / "bayesian_growth"
SCENARIOS = ("lognormal", "uniform")
LABELS = {"lognormal": "Lognormal prior", "uniform": "Uniform prior"}


def read(folder, name):
    return np.genfromtxt(folder / name, delimiter=",", names=True, dtype=None, encoding="utf-8")


def save(fig, folder, name):
    for axis in fig.axes:
        axis.grid(False, which="both")
    fig.savefig(folder / f"{name}.png", dpi=180)
    svg = folder / f"{name}.svg"
    fig.savefig(svg, dpi=180, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def kde(values, x):
    # A visual smoothing of recorded parameter draws, not a fitted likelihood.
    bandwidth = 1.06 * np.std(values, ddof=1) * len(values) ** (-0.2)
    z = (x[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def plot(folder):
    data = read(folder, "measurements.csv")
    if np.any(data["data"] < 0):
        raise ValueError("Regenerate observations with the nonnegative measurement model before plotting")
    curves = read(folder, "curves.csv")
    parameters = read(folder, "parameters.csv")
    draws = read(folder, "draws.csv")
    chains = read(folder, "chains.csv")
    priors = read(folder, "priors.csv")
    mcmc = {s: draws[(draws["prior"] == s) & (draws["method"] == "mcmc")] for s in SCENARIOS}
    laplace = {s: draws[(draws["prior"] == s) & (draws["method"] == "laplace")] for s in SCENARIOS}
    estimates = {s: parameters[parameters["prior"] == s] for s in SCENARIOS}
    with (folder / "credible_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["prior", "parameter", "map", "mcmc_mean", "mcmc_low95", "mcmc_high95", "laplace_low95", "laplace_high95"])
        for s in SCENARIOS:
            for p in estimates[s]:
                name = p["name"]
                writer.writerow([s, name, p["map"], mcmc[s][name].mean(),
                                 *np.quantile(mcmc[s][name], [.025, .975]),
                                 *np.quantile(laplace[s][name], [.025, .975])])

    # One continuous-size plot, showing every noisy observation at all six times.
    fig, axis = plt.subplots(figsize=(10, 4.7), layout="constrained")
    for i, time in enumerate(np.unique(data["time"])):
        block = data[data["time"] == time]
        curve = curves[(curves["prior"] == "lognormal") & (curves["time"] == time)]
        axis.scatter(block["size"], block["data"], s=5, alpha=.25, color=f"C{i}", rasterized=True)
        axis.plot(curve["size"], curve["map"], color=f"C{i}", label=rf"$t={time:g}$")
    axis.set(xlabel=r"Size $a$", ylabel=r"Number density $f(a,t)$", xlim=(0, 1.4),
             title="Nonnegative noisy densities; lines show the MAP fit")
    axis.legend(ncol=2, fontsize=9, title="Observation time")
    save(fig, folder, "distributions")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), layout="constrained")
    a = np.linspace(0, 1.4, 180)
    for k, s in enumerate(SCENARIOS):
        sample = mcmc[s]
        growth = sample["g0"][:, None] + sample["g1"][:, None] * a
        axes[0, 0].fill_between(a, *np.quantile(growth, [.025, .975], axis=0), color=f"C{k}", alpha=.25)
        axes[0, 0].plot(a, np.median(growth, axis=0), color=f"C{k}", label=LABELS[s])
        axes[0, 1].scatter(sample["g0"][::8], sample["g1"][::8], s=4, alpha=.2, color=f"C{k}", label=LABELS[s])
    axes[0, 0].plot(a, .434 + .2604 * a, "k--", lw=1, label="Generating law")
    axes[0, 0].set(xlabel=r"$a$", ylabel=r"$G(a)=g_0+g_1a$", title=r"Pointwise $95\%$ posterior bands")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(.434, .2604, "kx", label="Generating parameters")
    axes[0, 1].set(xlabel=r"$g_0$", ylabel=r"$g_1$", title="Joint posterior draws")
    axes[0, 1].legend(fontsize=8)
    for axis, name, truth in ((axes[1, 0], "g0", .434), (axes[1, 1], "g1", .2604)):
        values = np.r_[mcmc["lognormal"][name], mcmc["uniform"][name]]
        x = np.linspace(*np.quantile(values, [.001, .999]), 250)
        for k, s in enumerate(SCENARIOS):
            axis.plot(x, kde(mcmc[s][name], x), color=f"C{k}", label=LABELS[s] + ": NUTS")
            axis.plot(x, kde(laplace[s][name], x), color=f"C{k}", linestyle="--", lw=1, label=LABELS[s] + ": Laplace")
        axis.axvline(truth, color="k", linestyle=":", label="Generating value")
        axis.set(xlabel=r"$g_0$" if name == "g0" else r"$g_1$", ylabel="Posterior density")
        axis.legend(fontsize=6)
    save(fig, folder, "posterior")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), layout="constrained")
    for axis, name, truth in zip(axes, ("g0", "g1"), (.434, .2604)):
        for k, s in enumerate(SCENARIOS):
            p = priors[(priors["prior"] == s) & (priors["parameter"] == name)]
            axis.plot(p["value"], p["density"], color=f"C{k}", label=LABELS[s])
        axis.axvline(truth, color="k", linestyle=":", label="Generating value")
        axis.set(xlabel=r"$g_0$" if name == "g0" else r"$g_1$", ylabel="Prior density", xlim=(0, 1.2))
        axis.legend(fontsize=8)
    save(fig, folder, "priors")

    ref = estimates["lognormal"]
    residual_data = read(folder, "residuals.csv")
    residuals = residual_data["quantile_residual"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for k, time in enumerate(np.unique(data["time"])):
        d = residual_data[residual_data["time"] == time]
        axes[0].scatter(d["size"], d["quantile_residual"], s=4, alpha=.3, color=f"C{k}")
    axes[0].axhline(0, color="k", lw=.7)
    axes[0].set(xlabel=r"$a$", ylabel="Normal quantile residual", title="Truncated-Gaussian measurement check")
    observed = np.sort(residuals)
    expected = np.array([NormalDist().inv_cdf((i + .5) / len(observed)) for i in range(len(observed))])
    axes[1].plot(expected, observed, color="C0", lw=1)
    axes[1].plot(expected, expected, "k--", lw=1)
    axes[1].set(xlabel="Standard normal quantile", ylabel="Residual quantile")
    changes = np.log(estimates["uniform"]["map"] / ref["map"]) / ref["laplace_sd_z"]
    axes[2].bar([0, 1], changes, color=["C0", "C1"])
    axes[2].axhline(0, color="k", lw=.7)
    axes[2].set(xticks=[0, 1], xticklabels=[r"$g_0$", r"$g_1$"], ylabel="MAP change / reference posterior SD",
                title="Uniform versus lognormal prior")
    save(fig, folder, "checks")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for row, s in enumerate(SCENARIOS):
        for col, name in enumerate(("g0", "g1")):
            for c in np.unique(chains["chain"]):
                block = chains[(chains["prior"] == s) & (chains["chain"] == c)]
                axes[row, col].plot(block["draw"], block[name], lw=.45, alpha=.65, label=f"Chain {c}")
            axes[row, col].set(xlabel="Retained draw", ylabel=r"$g_0$" if name == "g0" else r"$g_1$", title=LABELS[s])
    axes[0, 0].legend(ncol=2, fontsize=7)
    save(fig, folder, "chains")

    file = folder / "refinement.csv"
    if file.exists():
        rows = read(folder, "refinement.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        x = np.arange(len(rows));valid = np.char.lower(rows["converged"].astype(str)) == "true"
        for offset, name, label in ((-.17, "g0_shift_sd", r"$g_0$"), (.17, "g1_shift_sd", r"$g_1$")):
            bars = axes[0].bar(x + offset, rows[name], width=.32, label=label)
            for bar, ok in zip(bars, valid):
                if not ok:bar.set_hatch("///")
        axes[0].axhline(0, color="k", lw=.7)
        axes[0].set(ylabel="Parameter change / analytical posterior SD")
        axes[0].legend(fontsize=8)
        if not np.all(valid):
            axes[0].legend([Patch(facecolor="C0"), Patch(facecolor="C1"),
                            Patch(facecolor="white", edgecolor="black", hatch="///")],
                           [r"$g_0$", r"$g_1$", "Unconverged candidate"], fontsize=7)
        axes[1].bar(x, rows["max_density_error_sigma"], color="C2")
        axes[1].set(ylabel=r"Maximum density error / $\sigma$")
        labels = {"inguva_010": r"Inguva, $\Delta t=0.01$",
                  "inguva_005": r"Inguva, $\Delta t=0.005$",
                  "space_280": "MUSCL, 280 bins", "space_560": "MUSCL, 560 bins",
                  "time_half": r"MUSCL, half $\Delta t$", "domain_larger": r"Domain $a_{\max}=2$"}
        for axis in axes:
            axis.set(xticks=x, xticklabels=[labels.get(v, v) for v in rows["variant"]])
            axis.tick_params(axis="x", rotation=35, labelsize=7)
        save(fig, folder, "refinement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    configure(usetex=args.usetex)
    plot(args.data_dir)
