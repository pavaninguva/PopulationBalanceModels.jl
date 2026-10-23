"""Four-panel Tang batch figure; consumes only the frozen production export."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pymupdf
from plotting_style import configure, panel, DOUBLE
from PIL import Image

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent

def read(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8"))

def plot():
    base = PAPER/"results/04_protein_crystallization/production"
    manifest = json.loads((base/"manifest.json").read_text())
    if manifest["mode"] != "production" or not manifest["complete"]:
        raise ValueError("A completed production calculation is required")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed numerical export: {name}")
    for name, expected in manifest["source_files"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed numerical source: {name}; rerun case 4")
    frozen = json.loads((base/"frozen_model.json").read_text())
    if frozen["validation_used"]:
        raise ValueError("Held-out observations cannot select the fit")
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.4))
    fig.subplots_adjust(left=.085, right=.98, bottom=.09, top=.95, hspace=.42, wspace=.32)
    a,b,c,d = axes.flat
    traj = read(base/"trajectory.csv")
    conc = read(base/"concentration_predictions.csv")
    obs = read(base/"psd_predictions.csv")
    curves = read(base/"distribution_curves.csv")
    quant = read(base/"quantile_predictions.csv")
    blue, orange = "#0072B2", "#D55E00"
    a.plot(traj["time_min"], traj["concentration_g_L"], color=blue, lw=1.4, label="Frozen model")
    initial=conc[conc["split"]=="initial_condition"]
    a.scatter(initial["time_min"],initial["observed_g_L"],s=16,marker="D",facecolors="white",
              edgecolors=".4",label="Initial condition",zorder=4,clip_on=False)
    for split, marker, color, label in [("calibration","o","k","Calibration"),
                                        ("validation","s",orange,"Held out")]:
        rows = conc[conc["split"] == split]
        a.scatter(rows["time_min"], rows["observed_g_L"], s=14, marker=marker,
                  facecolors="white" if split=="validation" else color, edgecolors=color, label=label, zorder=3)
    a.axvline(420,color=".6",ls=":",lw=.8)
    a.set(xlabel=r"Time (min)", ylabel=r"Dissolved concentration (g L$^{-1}$)", xlim=(0,680))
    a.legend(loc="upper right",fontsize=8)
    colors = {74:blue, 180:"#009E73", 420:"#CC79A7", 660:orange}
    for t in (74,180,420,660):
        ax = c if t==660 else b
        measured=obs[obs["time_min"]==t]
        model=curves[curves["time_min"]==t]
        edges=np.r_[model["left_um"],model["right_um"][-1]]
        ax.stairs(model["conditional_density_per_um"], edges,color=colors[t],lw=1.3,label=rf"{t} min")
        # Original reconstructed ordinates, not fabricated instrument bin counts.
        ax.plot(measured["size_um"],measured["observed_density_per_um"],color=colors[t],ls="--",lw=.9)
        ax.plot(measured["size_um"][::4],measured["observed_density_per_um"][::4],
                color=colors[t],ls="none",marker="s" if t==660 else "o",ms=2.5,mfc="white")
    for ax in (b,c):
        ax.set(xlabel=r"Equivalent diameter ($\mu$m)",ylabel=r"Conditional number density ($\mu$m$^{-1}$)",xlim=(2.5,25),ylim=(0,None))
    b.legend(loc="upper right",fontsize=8)
    c.text(.95,.92,r"$t=660$ min",transform=c.transAxes,ha="right",va="top")
    for column, probability, color, style in [("D10_um",.1,blue,"--"),
        ("D50_um",.5,"#222222","-"),("D90_um",.9,orange,"-.")]:
        d.plot(traj["time_min"],traj[column],ls=style,color=color,lw=1.3,label=rf"$D_{{{int(probability*100)}}}$")
        q=quant[np.isclose(quant["probability"],probability)]
        for split,marker in [("calibration","o"),("validation","s")]:
            r=q[q["split"]==split]
            d.scatter(r["time_min"],r["observed_um"],s=15,marker=marker,
                facecolors=color if split=="calibration" else "white",edgecolors=color,zorder=3)
    d.axvline(420,color=".6",ls=":",lw=.8)
    d.set(xlabel="Time (min)",ylabel=r"Conditional number quantile ($\mu$m)",xlim=(0,680))
    d.legend(loc="lower right",fontsize=8,ncol=3)
    for ax, letter in zip(axes.flat,"abcd"):
        panel(ax,letter)
    out=PAPER/"figures/04_protein_crystallization/production"
    out.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","png","svg"):
        fig.savefig(out/f"protein_crystallization.{ext}",dpi=300)
    plt.close(fig)
    with Image.open(out/"protein_crystallization.png") as im:
        assert all(abs(x-300)<1 for x in im.info["dpi"])
        assert im.size==(2100,1620)
    with pymupdf.open(out/"protein_crystallization.pdf") as document:
        for page in document:
            for font in page.get_fonts():
                if not font[0] or not document.extract_font(font[0])[3]:
                    raise ValueError(f"PDF font is not embedded: {font}")
    docs=ROOT/"docs/src/assets/protein_isothermal"
    docs.mkdir(parents=True,exist_ok=True)
    for ext in ("png","svg"):
        shutil.copy2(out/f"protein_crystallization.{ext}",docs/f"protein_crystallization.{ext}")
    (out/"plot_provenance.json").write_text(json.dumps({"renderer":"actual LaTeX",
        "numerical_manifest_sha256":hashlib.sha256((base/"manifest.json").read_bytes()).hexdigest(),
        "plot_driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "width_inches":7,"height_inches":5.4,"png_dpi":300},indent=2)+"\n")
    settings=json.loads((base/"run_settings.json").read_text())
    metrics=settings["metrics"]
    family="power-law nucleation with logistic onset" if settings["model"]=="logistic_activation" else "power-law nucleation"
    caption=rf"""\newcommand{{\ProteinCrystallizationCaption}}{{%
Calibration and held-out prediction of an initially unseeded 100 mL lysozyme batch
at $20\,^\circ\mathrm{{C}}$, using original Tang et al. (2018), Figures 11(d) and
12(a), doi:10.1016/j.jcrysgro.2018.06.023. The selected model uses {family},
size-independent growth and conservative dissolved-material exchange, with fixed
$c_*=2.16\,\mathrm{{g\,L}}^{{-1}}$, $\rho=1240\,\mathrm{{kg\,m}}^{{-3}}$ and $k_v=0.81$.
MUSCL/SSPRK3 advances {settings['selected']['bins']} bins on $0$--$50\,\mu\mathrm{{m}}$
with growth CFL at most 0.35. (a) Concentration; the diamond fixes the initial
condition, filled circles calibrate and open
squares validate. The vertical line marks the 420-min cutoff. (b) Conditional
number distributions used for calibration, at 74, 180 and 420 min. (c) The
entirely held-out distribution at 660 min. Solid stairs are computed bin
densities; dashed curves and sparse markers are the supplied reconstructed
observations. All size curves are conditioned on $2.5$--$25\,\mu\mathrm{{m}}$.
(d) Corresponding number quantiles; these are diagnostics, not additional fit
residuals. Empty-window quantiles remain undefined; before 74 min, small
numerical transport tails can control the unobserved conditional trajectories.
Calibration jointly minimizes
the mean squared concentration discrepancy (scale $0.1\,\mathrm{{g\,L}}^{{-1}}$)
and mean squared Wasserstein distance (scale $0.5\,\mu\mathrm{{m}}$), without
claiming a calibrated noise likelihood. Frozen held-out concentration RMSE is
{metrics['validation_concentration_RMSE_g_L']:.3f}$\,\mathrm{{g\,L}}^{{-1}}$ and
660-min $W_2$ is {metrics['validation_W2_um']:.3f}$\,\mu\mathrm{{m}}$.
These are conditional model predictions, not universal kinetic estimates.}}
"""
    (out/"caption.tex").write_text(caption)
    print(out)

if __name__ == "__main__":
    plot()
