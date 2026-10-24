"""Plot the CNT batch fit and (when available) verified NUTS marginals."""
from pathlib import Path
import hashlib
import json
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
from plotting_style import configure, panel, DOUBLE

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
BASE = PAPER / "results/04_protein_crystallization/production"
OUT = PAPER / "figures/04_protein_crystallization/production"
DOCS = ROOT / "docs/src/assets/protein_isothermal"


def read(name):
    return np.atleast_1d(np.genfromtxt(BASE / name, delimiter=",", names=True, dtype=None, encoding="utf8"))


def export(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        path=OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, metadata={"Creator": "PopulationBalanceModels CNT case study"})
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    for suffix in ("png", "svg"):
        shutil.copy2(OUT / f"{name}.{suffix}", DOCS / f"{name}.{suffix}")
    plt.close(fig)


def marginal_density(values, x, support):
    """Display-only Gaussian KDE, reflected at finite support boundaries."""
    h = 1.06*np.std(values, ddof=1)*len(values)**(-.2)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Degenerate posterior draws; inspect sampling")
    centers = [values]
    for endpoint in support:
        if np.isfinite(endpoint):
            centers.append(2*endpoint-values)
    density = sum(np.exp(-.5*((x[:, None]-v)/h)**2).sum(axis=1) for v in centers)
    return density/(len(values)*h*np.sqrt(2*np.pi))


def plot():
    manifest = json.loads((BASE / "manifest.json").read_text())
    if manifest["mode"] != "production" or not manifest["complete"]:
        raise ValueError("Completed production data required")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed result: {name}")
    for name, digest in manifest["source_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed numerical source: {name}; rerun case 4")
    frozen = json.loads((BASE / "frozen_model.json").read_text())
    if frozen["validation_used"]:
        raise ValueError("Validation observations must not select the fit")
    # A later MAP-only run must not silently inherit earlier posterior draws.
    for name in ("posterior_draws.csv", "latent_predictions.csv"):
        if (BASE / name).exists() and (BASE / name).stat().st_mtime < (BASE / "map.json").stat().st_mtime:
            raise ValueError(f"{name} predates the current MAP; remove obsolete posterior exports or rerun --sample")
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.3))
    fig.subplots_adjust(left=.09, right=.98, bottom=.10, top=.95, hspace=.43, wspace=.34)
    a, b, c, d = axes.flat
    traj, conc = read("trajectory.csv"), read("concentration_predictions.csv")
    curves, observed = read("distribution_curves.csv"), read("psd_predictions.csv")
    blue, orange = "#0072B2", "#D55E00"
    if (BASE / "latent_predictions.csv").exists():
        draws = read("latent_predictions.csv")
        ts = np.unique(draws["time_min"])
        lo, hi = np.array([np.quantile(draws["concentration_g_L"][draws["time_min"] == t], [.05, .95]) for t in ts]).T
        a.fill_between(ts, lo, hi, color=blue, alpha=.18, lw=0, label=r"90\% latent interval")
    a.plot(traj["time_min"], traj["concentration_g_L"], color=blue, lw=1.3, label="CNT MAP")
    for split, marker, color, label in [("initial_condition", "D", ".4", "Initial condition"),
                                       ("calibration", "o", "k", "Calibration"),
                                       ("validation", "s", orange, "Held out")]:
        rows = conc[conc["split"] == split]
        a.scatter(rows["time_min"], rows["observed_g_L"], s=14, marker=marker,
                  facecolors=color if split == "calibration" else "white", edgecolors=color, label=label, zorder=3, clip_on=False)
    a.axvline(420, color=".6", ls=":", lw=.8)
    a.set(xlabel="Time (min)", ylabel=r"Dissolved concentration (g L$^{-1}$)", xlim=(0, 680))
    a.legend(fontsize=7, loc="upper right")
    colors = {74: blue, 180: "#009E73", 420: "#CC79A7", 660: orange}
    for t in (74, 180, 420, 660):
        ax = c if t == 660 else b
        model = curves[curves["time_min"] == t]
        obs = observed[observed["time_min"] == t]
        edges = np.r_[model["left_um"], model["right_um"][-1]]
        ax.stairs(model["conditional_density_per_um"], edges, color=colors[t], lw=1.2, label=rf"{t} min")
        ax.plot(obs["size_um"], obs["observed_density_per_um"], color=colors[t], ls="--", lw=.9)
        ax.plot(obs["size_um"][::4], obs["observed_density_per_um"][::4], color=colors[t],
                ls="none", marker="s" if t == 660 else "o", ms=2.5, mfc="white")
    for ax in (b, c):
        ax.set(xlabel=r"Equivalent diameter ($\mu$m)", ylabel=r"Conditional number density ($\mu$m$^{-1}$)", xlim=(2.5, 25), ylim=(0, None))
    b.legend(fontsize=7, loc="upper right")
    c.text(.98, 1.02, r"$t=660$ min (held out)", transform=c.transAxes, ha="right", va="bottom", fontsize=8)
    # CDF residuals make distribution-shape disagreement visible, without
    # repeating the quantiles derived from these same observations as data.
    for t in (74, 180, 420, 660):
        rows = observed[observed["time_min"] == t]
        d.plot(rows["size_um"], rows["predicted_cdf"] - rows["observed_cdf"], color=colors[t],
               ls="--" if t == 660 else "-", lw=1.2, label=rf"{t} min")
    d.axhline(0, color=".5", lw=.6)
    d.set(xlabel=r"Equivalent diameter ($\mu$m)", ylabel=r"$F_{\mathrm{MAP}}-F_{\mathrm{obs}}$", xlim=(2.5, 25))
    d.legend(fontsize=7, ncol=2)
    for ax, letter in zip(axes.flat, "abcd"):
        panel(ax, letter)
    export(fig, "protein_crystallization")

    diagnostic_path = BASE / "sampling_diagnostics.json"
    if diagnostic_path.exists():
        diag = json.loads(diagnostic_path.read_text())
        if not diag["passed"]:
            print("NUTS screen failed; no posterior summary figure or credible bands")
        else:
            draws = read("posterior_draws.csv")
            fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.7))
            fig.subplots_adjust(left=.08, right=.98, bottom=.14, top=.95, hspace=.5, wspace=.4)
            specs = [("logAJ", r"$\ln A_J$", lambda x: np.where((x > 0) & (x < 100), .01, 0)),
                     ("gamma", r"$\gamma$ (mJ m$^{-2}$)", lambda x: np.exp(-.5*((np.log(x)-np.log(.64))/.7)**2)/(x*.7*np.sqrt(2*np.pi))),
                     ("AG", r"$\ln[A_G/(\mathrm{nm\,min}^{-1})]$", lambda x: np.where((x > np.log(.01)) & (x < np.log(1500)), 1/np.log(150000), 0)),
                     ("g", r"$g$", lambda x: np.where((x > 1) & (x < 3), .5, 0)),
                     ("sigma_c", r"$\sigma_c$ (g L$^{-1}$)", lambda x: np.exp(-x)),
                     ("sigma_F", r"$\sigma_F$", lambda x: 10*np.exp(-10*x))]
            for i, (ax, (name, label, prior)) in enumerate(zip(axes.flat, specs)):
                values = np.log(draws[name]) if name == "AG" else draws[name]
                low, high = np.quantile(values, [.001, .999]); span = high-low
                lo, hi = low-.2*span, high+.2*span
                if name != "AG":
                    lo = max(lo, 1.0001 if name == "g" else 1e-6)
                if name == "g":
                    hi = min(hi, 2.9999)
                x = np.linspace(lo, hi, 500)
                ax.plot(x, prior(x), color=".4", ls="--", lw=1, label="Prior")
                support = {"logAJ": (0,100), "gamma": (0,np.inf),
                           "AG": (np.log(.01),np.log(1500)), "g": (1,3),
                           "sigma_c": (0,np.inf), "sigma_F": (0,np.inf)}[name]
                ax.plot(x, marginal_density(values, x, support), color=blue, lw=1.3, label="NUTS draws")
                ax.set(xlabel=label, ylabel="Density")
                panel(ax, "abcdef"[i])
            axes.flat[0].legend(fontsize=7)
            export(fig, "protein_posterior")
            names = ["logAJ", "gamma", "AG", "g"]
            labels = [r"$\ln A_J$", r"$\gamma$ (mJ m$^{-2}$)", r"$\ln A_G$", r"$g$"]
            samples = np.column_stack([np.log(draws[n]) if n == "AG" else draws[n] for n in names])
            supports = [(0,100), (0,np.inf), (np.log(.01),np.log(1500)), (1,3)]
            fig, axes = plt.subplots(4, 4, figsize=(DOUBLE, 6.0))
            fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.98, wspace=.12, hspace=.12)
            for i in range(4):
                for j in range(4):
                    ax = axes[i,j]
                    if j > i:
                        ax.set_visible(False)
                        continue
                    if i == j:
                        v = samples[:,i]
                        lo, hi = np.quantile(v, [.001,.999])
                        span = hi-lo
                        x = np.linspace(max(supports[i][0],lo-.2*span), min(supports[i][1],hi+.2*span), 400)
                        ax.plot(x, marginal_density(v,x,supports[i]),color=blue,lw=1.1)
                        ax.set_yticks([])
                    else:
                        # Every point is an actual saved draw. Rasterize only
                        # this cloud; labels and curves remain vector objects.
                        ax.scatter(samples[:,j],samples[:,i],s=1.5,alpha=.15,
                                   color=blue,edgecolors="none",rasterized=True)
                    if i == 3:
                        ax.set_xlabel(labels[j],fontsize=8)
                    else:
                        ax.tick_params(labelbottom=False)
                    if j == 0 and i > 0:
                        ax.set_ylabel(labels[i],fontsize=8)
                    elif j != 0:
                        ax.tick_params(labelleft=False)
                    if j == 3:
                        ax.set_xlim(right=3)
                    if i == 3 and i != j:
                        ax.set_ylim(top=3)
                    ax.tick_params(labelsize=8)
                    ax.locator_params(axis="x",nbins=3)
                    if i != j:
                        ax.locator_params(axis="y",nbins=3)
            export(fig,"protein_corner")
            with (OUT / "posterior_summaries.csv").open("w",newline="") as stream:
                writer=csv.writer(stream,lineterminator="\n")
                writer.writerow(["parameter","q025","median","q975"])
                for name in ("logAJ","gamma","AG","g","sigma_c","sigma_F"):
                    writer.writerow([name,*np.quantile(draws[name],[.025,.5,.975])])
            with (OUT / "posterior_correlations.csv").open("w",newline="") as stream:
                writer=csv.writer(stream,lineterminator="\n")
                writer.writerow(["coordinate","logAJ","gamma","logAG","g"])
                for name,row in zip(("logAJ","gamma","logAG","g"),np.corrcoef(samples.T)):
                    writer.writerow([name,*row])
    settings = json.loads((BASE / "run_settings.json").read_text())
    m = settings["metrics"]
    (OUT / "caption.tex").write_text(rf"""\newcommand{{\ProteinCrystallizationCaption}}{{%
Bayesian calibration and held-out prediction of the original Tang et al. (2018)
100 mL isothermal lysozyme batch at $20\,^\circ\mathrm{{C}}$.
CNT nucleation and empirical growth follow Mitchell et al. (2023), Case 3,
with $c_*=2.16\,\mathrm{{g\,L}}^{{-1}}$, no initial seeds, and conservative solute exchange.
MUSCL/SSPRK3 advances {settings['bins']} length bins on $0$--$50\,\mu\mathrm{{m}}$,
with growth CFL at most 0.35. (a) Dissolved concentration; filled markers calibrate,
open squares validate, the diamond fixes the initial state. Any band is a 90\%
latent interval from diagnostically screened NUTS draws, excluding observation noise.
(b) Conditional number distributions at calibration times; solid stairs are MAP
bin densities, dashed curves and sparse markers are supplied graph reconstructions.
(c) The held-out 660-min distribution. (d) Signed conditional-CDF residuals.
Distributions are conditioned on $2.5$--$25\,\mu\mathrm{{m}}$.
The likelihood uses correlated CDF blocks rather than particle counts or independent
reconstruction points. Concentration calibration RMSE is {m['calibration_concentration_RMSE_g_L']:.3f}
and held-out RMSE is {m['validation_concentration_RMSE_g_L']:.3f}$\,\mathrm{{g\,L}}^{{-1}}$;
held-out $W_2={m['validation_W2_um']:.3f}\,\mu\mathrm{{m}}$ is diagnostic only.
Residuals and numerical refinement limit physical interpretation of the parameters.}}

\newcommand{{\ProteinPosteriorCaption}}{{%
Marginal summaries of retained NUTS draws and declared priors for the CNT batch model.
Densities are shown over the posterior region; full prior support is listed in the tutorial.
The growth-coefficient panel uses its natural logarithm, including the density-coordinate change.
These are conditional posterior distributions under the stated observation discrepancy model,
fixed solubility and discretization, not independent experimental kinetic measurements.}}

\newcommand{{\ProteinCornerCaption}}{{%
Kinetic-parameter marginals and pairwise posterior draws for the same CNT batch fit.
The diagonal uses support-reflected kernel density summaries. Every scatter point is
an actual retained NUTS draw; nuisance discrepancy scales are marginalized rather
than fixed. Correlated ridges show compensating coefficient/barrier and
coefficient/exponent combinations. Sampling diagnostics and the raw draws accompany
the numerical exports; a mixed chain does not establish physical model adequacy.}}
""")
    (OUT / "plot_provenance.json").write_text(json.dumps({"renderer": "actual LaTeX", "png_dpi": 300,
        "numerical_manifest_sha256": hashlib.sha256((BASE / "manifest.json").read_bytes()).hexdigest(),
        "plot_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+"\n")
    print(OUT)


if __name__ == "__main__":
    plot()
