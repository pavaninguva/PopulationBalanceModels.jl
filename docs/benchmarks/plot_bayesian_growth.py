"""Render only recorded Bayesian-growth results; requires NumPy/Matplotlib and optional LaTeX."""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from plot_style import configure

ROOT = Path(__file__).resolve().parents[1] / "src" / "assets" / "bayesian_growth"


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


def plot(folder):
    data = read(folder, "measurements.csv")
    bulk = read(folder, "bulk.csv")
    parameters = read(folder, "parameters.csv")
    draws = read(folder, "draws.csv")
    mcmc = draws[draws["method"] == "mcmc"]
    laplace = draws[draws["method"] == "laplace"]
    with (folder / "credible_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["parameter", "map", "mcmc_mean", "mcmc_low95", "mcmc_high95", "laplace_low95", "laplace_high95"])
        for i, name in enumerate(parameters["name"]):
            writer.writerow([name, parameters["map"][i], mcmc[name].mean(),
                             *np.quantile(mcmc[name], [.025, .975]),
                             *np.quantile(laplace[name], [.025, .975])])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    for axis, time in zip(axes.flat, np.unique(data["time"])):
        block = data[data["time"] == time]
        width = block["upper"] - block["lower"]
        centers = (block["upper"] + block["lower"]) / 2
        axis.bar(centers, block["count"] / block["count"].sum() / width,
                 width=width, alpha=.35, label="Synthetic sampled particles")
        axis.stairs(block["model_probability"] / width,
                    np.r_[block["lower"], block["upper"][-1]], color="C1", label="MAP bin integrals")
        axis.set(xlabel=r"$a$", ylabel=r"Conditional size density", title=rf"$t={time:g}$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Size-dependent growth: synthetic dynamic measurements")
    save(fig, folder, "distributions")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    x = np.linspace(0, 1.4, 120)
    growth = mcmc["g0"][:, None] + mcmc["g1"][:, None] * x
    quantiles = np.quantile(growth, [.025, .975], axis=0)
    axes[0, 0].fill_between(x, *quantiles, alpha=.3, label=r"MCMC pointwise $95\%$")
    axes[0, 0].plot(x, parameters["map"][0] + parameters["map"][1] * x, label="MAP")
    axes[0, 0].plot(x, .434 + .2604 * x, "k--", label="Generating law")
    axes[0, 0].set(xlabel=r"$a$", ylabel=r"$G(a)=g_0+g_1a$")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].scatter(mcmc["g0"][::4], mcmc["g1"][::4], s=3, alpha=.25, rasterized=True, label="NUTS")
    axes[0, 1].scatter(laplace["g0"][::2], laplace["g1"][::2], s=3, alpha=.15, rasterized=True, label="Laplace")
    axes[0, 1].plot(.434, .2604, "kx", label="Generating parameters")
    axes[0, 1].set(xlabel=r"$g_0$", ylabel=r"$g_1$")
    axes[0, 1].legend(fontsize=8)
    for axis, name, i in ((axes[1, 0], "g0", 0), (axes[1, 1], "g1", 1)):
        axis.hist(mcmc[name], bins=35, density=True, alpha=.4, label="NUTS")
        axis.hist(laplace[name], bins=35, density=True, histtype="step", label="Laplace")
        axis.axvline(parameters["true"][i], color="k", linestyle="--", label="Generating value")
        axis.set(xlabel=rf"${'g_0' if i == 0 else 'g_1'}$", ylabel="Marginal density")
        axis.legend(fontsize=8)
    fig.suptitle("Joint inference includes initial-amplitude and noise nuisances")
    save(fig, folder, "posterior")

    predictive = read(folder, "predictive.csv")
    profile = read(folder, "profiles.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    times = bulk["time"]
    latent = np.array([predictive[predictive["time"] == t]["latent_concentration"] for t in times])
    noisy = np.array([predictive[predictive["time"] == t]["noisy_concentration"] for t in times])
    axes[0].fill_between(times, *np.quantile(noisy, [.05, .95], axis=1), alpha=.18, color="C1", label=r"Noisy predictions: empirical $90\%$")
    axes[0].fill_between(times, *np.quantile(latent, [.05, .95], axis=1), alpha=.4, label=r"Latent predictions: empirical $90\%$")
    axes[0].plot(times, bulk["map"], label="MAP")
    axes[0].plot(times, bulk["truth"], "k--", label="Generating trajectory")
    axes[0].plot(times, bulk["data"], "ko", ms=3, label="Synthetic observations")
    axes[0].set(xlabel=r"$t$", ylabel=r"Scaled concentration $c$")
    axes[0].legend(fontsize=6)
    for name, label in (("likelihood", "Likelihood profile"), ("posterior", "Posterior profile")):
        p = profile[profile["target"] == name]
        valid = np.char.lower(p["converged"].astype(str)) == "true"
        axes[1].plot(np.exp(p["log_g1"][valid]), p["delta"][valid], label=label)
    axes[1].set(xlabel=r"$g_1$", ylabel=r"Profile negative log density change")
    axes[1].legend(fontsize=7)
    shift = np.log(parameters["alternative_map"] / parameters["map"]) / parameters["laplace_sd_z"]
    axes[2].bar(np.arange(4), shift)
    axes[2].axhline(0, color=".5", lw=.6)
    axes[2].set(xticks=np.arange(4), xticklabels=[r"$g_0$", r"$g_1$", r"$A$", r"$\sigma_c$"],
                ylabel="MAP change / reference posterior SD", title="Broader kinetic priors")
    fig.suptitle("Independent concentration measurements, profiles and prior sensitivity")
    save(fig, folder, "checks")

    refinement = read(folder, "refinement.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    labels = ["280 bins", "560 bins", "Half time step", "Larger domain"]
    for axis, field, symbol in ((axes[0], "g0_shift_sd", r"$\log g_0$"), (axes[1], "g1_shift_sd", r"$\log g_1$")):
        bars = axis.bar(np.arange(len(refinement)), refinement[field])
        valid = np.char.lower(refinement["converged"].astype(str)) == "true"
        for bar, converged in zip(bars, valid):
            if not converged:
                bar.set(facecolor=".75", hatch="///", label="Refit did not converge")
        if not np.all(valid):
            axis.legend(handles=[next(bar for bar, flag in zip(bars, valid) if not flag)], fontsize=7)
        axis.axhline(0, color=".5", lw=.6)
        axis.set(xticks=np.arange(len(refinement)), xticklabels=labels, ylabel=f"{symbol} mode change / reference posterior SD")
        axis.tick_params(axis="x", labelrotation=18)
    fig.suptitle("Sectional refits compared with characteristic-solution inference")
    save(fig, folder, "refinement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    configure(usetex=args.usetex)
    plot(args.data_dir)
