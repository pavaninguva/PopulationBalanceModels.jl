"""Optional figure generation; Julia benchmark itself has no Python dependency."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

root = Path(__file__).resolve().parents[1] / "src" / "assets"
with (root / "results_2d.csv").open() as handle:
    rows = list(csv.DictReader(handle))
methods = [
    ("WENO5_SSPRK33", "WENO5 + SSPRK3", "#1769aa", "o"),
    ("HOCADI4_variant", "Compact–ADI variant", "#d26b15", "s"),
    ("InguvaBraatz_constant_shift", "Inguva–Braatz: constant growth", "#21845b", "^"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), layout="constrained")
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
axes[1, 0].set(xlim=(0, 0.8), ylim=(-.3,1.65), xlabel="Internal coordinate x", ylabel="Number density",
               title="C   Rectangle: cut through the final pulse (128 × 128)")
axes[1, 0].legend(fontsize=8, loc="upper left")
axes[0, 0].set(xlabel="Bins per axis", ylabel="L¹ density error",
               title="A   Smooth transport: analytical error")
axes[0, 1].set(xlabel="Warmed solve time (ms; minimum of 3)", ylabel="L¹ density error",
               title="B   Smooth transport: accuracy versus cost")
axes[0, 1].grid(which="major",alpha=.18)
axes[1, 1].set(xlabel="Bins per axis", ylabel="Negative particle number",
               title="D   Rectangle: undershoots, without clipping")
for ax in (axes[0, 0], axes[1, 1]):
    ax.set_xticks([32, 64, 128], labels=["32", "64", "128"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(which="major", alpha=.18)
axes[0, 0].legend(fontsize=8, loc="center right")
axes[1, 1].text(.03, .14, "Zero values shown at 10⁻¹⁸ for log display.",
                transform=axes[1, 1].transAxes, fontsize=8, color="#555e66")
fig.suptitle("2D population-balance transport: accuracy, cost and positivity", fontsize=16)
fig.supxlabel("Periodic constant-growth diagnostic; compact–ADI is a specified variant, not a reproduction of Li et al. (2026).",
              fontsize=9, color="#555e66")
fig.savefig(root / "benchmark_2d.png", dpi=190, bbox_inches="tight", pad_inches=.18)
fig.savefig(root / "benchmark_2d.svg", bbox_inches="tight", pad_inches=.18)
