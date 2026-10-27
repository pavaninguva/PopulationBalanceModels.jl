"""Optional figure generation; Julia benchmark itself has no Python dependency."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from plot_style import configure, export_figure, panel_letters
configure(usetex=True)

root = Path(__file__).resolve().parents[1] / "src" / "assets"
with (root / "results_2d.csv").open() as handle:
    rows = list(csv.DictReader(handle))
methods = [
    ("WENO5_SSPRK33", "WENO5 + SSPRK3", "#1769aa", "o"),
    ("HOCADI4_variant", "Compact–ADI variant", "#d26b15", "s"),
    ("InguvaBraatz_constant_shift", "Inguva–Braatz: constant growth", "#21845b", "^"),
]
fig, axes = plt.subplots(2, 2, figsize=(7, 5.4))
fig.subplots_adjust(left=.11,right=.985,bottom=.20,top=.94,hspace=.5,wspace=.38)
panel_letters(axes)
for key, label, color, marker in methods:
    smooth = [r for r in rows if r["method"] == key and r["profile"] == "smooth"]
    sharp = [r for r in rows if r["method"] == key and r["profile"] == "rectangle"]
    ns = [int(r["n"]) for r in smooth]
    axes[0, 0].loglog(ns, [max(float(r["L1"]), 1e-18) for r in smooth],
                      marker=marker, color=color, label=label)
    axes[0, 1].loglog([1e3*float(r["seconds"]) for r in smooth], [max(float(r["L1"]), 1e-18) for r in smooth],
                      marker=marker, color=color, label=label)
    axes[1, 1].loglog(ns, [max(float(r["negative_number"]), 1e-18) for r in sharp],
                      marker=marker, color=color, label=label)

with (root / "rectangle_cut.csv").open() as handle:
    cut = list(csv.DictReader(handle))
xs = [float(r["x"]) for r in cut]
axes[1, 0].plot(xs, [float(r["exact"]) for r in cut], color="#343b42",
                linewidth=2.2, alpha=.75, drawstyle="steps-mid", label="Analytical bin average")
for key, label, color, marker in methods:
    axes[1, 0].plot(xs, [float(r[key]) for r in cut], color=color,
                    linewidth=1.4, linestyle="--" if key.startswith("Inguva") else "-",
                    drawstyle="steps-mid" if key.startswith("Inguva") else "default",
                    label=label)
axes[1, 0].axhline(0, color="#9aa3aa", linewidth=.8)
axes[1, 0].set(xlim=(0, 0.8), ylim=(-.3,1.28), xlabel="Internal coordinate x", ylabel=r"$f$")

axes[0, 0].set(xlabel="Bins per axis", ylabel=r"$L^1$ density error")
axes[0, 1].set(xlabel="Warmed solve time (ms)", ylabel=r"$L^1$ density error")
axes[0, 1].grid(False,which="both")
axes[1, 1].set(xlabel="Bins per axis", ylabel="Negative particle number")
for ax in (axes[0, 0], axes[1, 1]):
    ax.set_xticks([32, 64, 128], labels=["32", "64", "128"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(False,which="both")

axes[1, 1].text(.03, .14, r"Zero displayed at $10^{-18}$.",
                transform=axes[1, 1].transAxes, fontsize=8, color="#555e66")
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.52,.005), fontsize=7.5)
export_figure(fig, root, "benchmark_2d")
