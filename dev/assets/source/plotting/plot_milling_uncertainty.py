"""Plot retained milling UQ outputs; this command never runs Julia.

Final: python docs/benchmarks/plot_milling_uncertainty.py --usetex
Forward check: add --forward to plot the separate Barrasso Figure 7 comparison.
Preview: add --smoke (writes only .cache/milling_uncertainty/preview).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plot_style import configure, export_figure, panel_letters

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/src/assets/milling_uncertainty"
COLORS = {"mc": "#24679a", "pce1": "#bd6b28", "pce2": "#26816b"}
LABELS = {"mc": "MC", "pce1": "PCE, degree 1", "pce2": "PCE, degree 2"}


def table(folder, name):
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing numerical export {path}; run the Julia workflow first")
    return np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")


def checked_data(folder, smoke):
    path = folder / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed milling provenance: {path}")
    manifest = json.loads(path.read_text())
    required = "smoke" if smoke else "production"
    if not manifest["complete"] or manifest["mode"] != required:
        raise ValueError(f"Expected completed {required} results")
    for name, digest in manifest["numerical_files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed numerical export: {name}")
    if not smoke and manifest["settings"]["mc_samples"] != 12000:
        raise ValueError("Final Figure 4 requires the full 12,000-solve MC ensemble")
    return manifest


def draw(axis, name, data, nominal):
    t = data["time_s"]
    color = COLORS[name]
    axis.fill_between(t, data["q025_um"], data["q975_um"], color=color, alpha=.20, lw=0)
    axis.plot(t, data["gaussian_low_um"], color=color, ls=":", lw=.75, alpha=.75)
    axis.plot(t, data["gaussian_high_um"], color=color, ls=":", lw=.75, alpha=.75)
    axis.plot(t, data["mean_um"], color=color, lw=1.25)
    axis.plot(nominal["time_s"], nominal["d50_um"], color="#424242", ls="--", lw=.8)
    axis.axvline(30, color=".5", ls=(0, (3, 3)), lw=.6, zorder=0)
    axis.set_xlabel(r"Time, $t$ (s)")
    axis.set_xlim(0, 200)
    axis.tick_params(direction="out")


def plot_forward(*, usetex, preview=False):
    """Read only the separately produced constant-speed comparison, never MC data."""
    folder = ASSETS / "forward"
    manifest = json.loads((folder / "provenance.json").read_text())
    if not manifest["complete"] or manifest["mode"] != "forward_comparison":
        raise ValueError("Expected completed Barrasso forward-comparison results")
    for name, digest in manifest["numerical_files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed forward export: {name}")
    for name, digest in manifest["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed forward producer: {name}; rerun the forward check")
    data = table(folder, "trajectories.csv")
    points = np.genfromtxt(folder / "reference_comparison.csv", delimiter=",",
                           names=True, dtype=None, encoding="utf-8")
    output = ROOT / ".cache/milling_uncertainty/preview" if preview else ASSETS
    configure(usetex=usetex)
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.7), sharex=True)
    fig.subplots_adjust(left=.10, right=.985, top=.95, bottom=.23,
                        hspace=.29, wspace=.26)
    settings = [(100, "pivot_CDF_d50_um", "pivot_CDF", "--"),
                (100, "finite_bin_d50_um", "finite_bin_CDF", ":"),
                (800, "finite_bin_d50_um", "finite_bin_CDF", "-")]
    speed_colors = {3350: "#24679a", 4923: "#26816b"}
    # genfromtxt's default dtype is numeric: use explicit string fields for the
    # source labels while retaining their original, unrounded coordinates.
    reference = np.genfromtxt(ROOT / "examples/data/milling_uncertainty/barrasso_figure7.csv",
                             delimiter=",", names=True, dtype=None, encoding="utf-8")
    for rpm, color in speed_colors.items():
        for bins, field, statistic, linestyle in settings:
            d = data[(data["rpm"] == rpm) & (data["bins"] == bins)]
            axes[0, 0].plot(d["time_s"], d[field], color=color, ls=linestyle, lw=1.0)
            e = points[(points["rpm"] == rpm) & (points["bins"] == bins) &
                       (points["quantity"] == "d50_um") & (points["statistic"] == statistic)]
            axes[1, 0].plot(e["time_s"], e["signed_difference"], color=color,
                            ls=linestyle, lw=.9)
            # Both median observations use exactly the same inventory state.
            if statistic == "pivot_CDF" or bins == 800:
                axes[0, 1].plot(d["time_s"], d["holdup_g"], color=color, ls=linestyle, lw=1.)
                e = points[(points["rpm"] == rpm) & (points["bins"] == bins) &
                           (points["quantity"] == "holdup_g")]
                axes[1, 1].plot(e["time_s"], e["signed_difference"], color=color,
                                ls=linestyle, lw=.9)
        marker = "^" if rpm == 3350 else "s"
        for axis, quantity in zip(axes[0], ("d50_um", "holdup_g")):
            r = reference[(reference["rpm"] == rpm) & (reference["quantity"] == quantity)]
            axis.plot(r["time_s"], r["value"], ls="none", marker=marker,
                      ms=3.2, mfc="white", mec=color, mew=.7)
    axes[0, 0].set_ylabel(r"Outlet $d_{50}$ ($\mu$m)")
    axes[0, 1].set_ylabel(r"Mill hold-up (g)")
    axes[1, 0].set_ylabel(r"$d_{50}$ difference ($\mu$m)")
    axes[1, 1].set_ylabel(r"Hold-up difference (g)")
    axes[0, 0].set_ylim(0, 1000)
    axes[0, 1].set_ylim(0, 250)
    for axis in axes[1]:
        axis.axhline(0, color=".5", lw=.6)
        axis.set_xlabel(r"Time, $t$ (s)")
    for axis in axes.flat:
        axis.set_xlim(0, 180)
    panel_letters(axes)
    handles = [Line2D([], [], color=c, lw=1.2, label=f"{rpm} RPM")
               for rpm, c in speed_colors.items()]
    handles += [Line2D([], [], color=".2", ls="none", marker="s", mfc="white", ms=3.2,
                       label="Barrasso Figure 7"),
                Line2D([], [], color=".2", ls="--", label="100 pivots; pivot CDF"),
                Line2D([], [], color=".2", ls=":", label="100 pivots; finite-bin CDF"),
                Line2D([], [], color=".2", ls="-", label="800 pivots; finite-bin CDF")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(.52, .015), columnspacing=1.4, handlelength=2.4)
    export_figure(fig, output, "milling_barrasso_figure7")
    caption = (
        "Forward comparison with Barrasso et al. (2013), DOI 10.1007/s12247-013-9155-0, "
        "Figure 7. (a,b) Instantaneous volume-weighted outlet median and mill hold-up at "
        "3350/4923 RPM; symbols are published simulation markers extracted from the original "
        "vector PDF, not experimental data. (c,d) Signed package-minus-source differences at "
        "the 22 noninitial source times. The 100-pivot power grid and printed point-normalized "
        "feed are used without fitting kinetic parameters. An initial inventory of one second "
        "of feed (7.4 g) is a declared reconstruction. Dashed/dotted diameter curves change only "
        "the observation reconstruction; solid curves additionally refine to 800 pivots at "
        "fixed endpoint volumes. Hold-up is independent of the median reconstruction. "
        "All package curves use the shared conservative model and checked CVODE BDF adapter. "
        "Hold-up is close, but diameter startup and some later values differ: this is a partial "
        "reproduction, not verified equivalence to the original gPROMS implementation. "
        "This separate calculation does not replace the retained geometric-grid UQ ensemble."
    )
    (output / "forward_caption.txt").write_text(caption + "\n")
    (output / "forward_figure_provenance.json").write_text(json.dumps({
        "latex": usetex, "dpi": 300, "preview": preview,
        "numerical_producer": manifest["revision"],
        "numerical_files": manifest["numerical_files"],
        "source_reference_sha256": manifest["configuration"]["reference_sha256"],
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(Path(__file__).with_name("plot_style.py").read_bytes()).hexdigest(),
        "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "presentation": "Lines connect stored predictions; source points and signed errors are unadjusted",
    }, indent=2) + "\n")
    print(f"Wrote Barrasso forward PNG/PDF/SVG to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--forward", action="store_true", help="Plot the separate Barrasso Figure 7 check")
    args = parser.parse_args()
    if not args.smoke and not args.usetex:
        parser.error("Final figures require --usetex; mathtext is reserved for --smoke previews")
    if args.forward:
        plot_forward(usetex=args.usetex, preview=args.smoke)
        return
    folder = ROOT / ".cache/milling_uncertainty/smoke" if args.smoke else ASSETS / "data"
    output = ROOT / ".cache/milling_uncertainty/preview" if args.smoke else ASSETS
    manifest = checked_data(folder, args.smoke)
    configure(usetex=args.usetex)
    nominal = table(folder, "nominal.csv")
    summaries = {name: table(folder, f"{name}_summary.csv") for name in COLORS}
    for data in summaries.values():
        if not np.array_equal(data["time_s"], nominal["time_s"]):
            raise ValueError("Mismatched output times")
    low = min(float(np.min(d["gaussian_low_um"])) for d in summaries.values())
    high = max(float(np.max(d["gaussian_high_um"])) for d in summaries.values())
    low = min(low, *(float(np.min(d["q025_um"])) for d in summaries.values()), 0)
    high = max(high, *(float(np.max(d["q975_um"])) for d in summaries.values()))
    limits = (low - .03 * (high-low), high + .05 * (high-low))
    handles = [Line2D([], [], color="black", lw=1.25, label="Propagated mean"),
               Line2D([], [], color=".3", ls="--", lw=.8, label="Inputs at their means"),
               Patch(facecolor=".55", alpha=.25, label=r"Pointwise 95\% output quantiles"),
               Line2D([], [], color=".4", ls=":", lw=.8, label=r"Mean $\pm 1.96$ SD")]
    captions = []
    for name, number in (("mc", 4), ("pce1", 7), ("pce2", 8)):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), layout=None)
        fig.subplots_adjust(left=.10, right=.985, top=.91, bottom=.27, wspace=.25)
        for axis in axes:
            draw(axis, name, summaries[name], nominal)
        axes[0].set_ylim(limits)
        axes[0].set_ylabel(r"Outlet volume median, $d_{50}$ ($\mu$m)")
        # Explicit transient zoom; the left panel retains the complete startup.
        axes[1].set_xlim(25, 45)
        mask = (nominal["time_s"] >= 25) & (nominal["time_s"] <= 45)
        data = summaries[name]
        y = np.concatenate([data[k][mask] for k in ("q025_um", "q975_um", "gaussian_low_um", "gaussian_high_um")])
        pad = .08 * np.ptp(y)
        axes[1].set_ylim(np.min(y)-pad, np.max(y)+pad)
        axes[1].legend(handles=[Line2D([], [], color=COLORS[name], label=LABELS[name])],
                       loc="lower right", frameon=False)
        panel_letters(axes)
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(.52, .005), columnspacing=1.5, handlelength=2.4)
        export_figure(fig, output, f"milling_figure{number}")
        count = manifest["settings"]["mc_samples"] if name == "mc" else (12 if name == "pce1" else 32)
        captions.append(
            f"Figure {number} reconstruction of Bhonsale et al. (2018), DOI 10.3390/pr6120255 "
            f"({LABELS[name]}, {count} full-model realizations). "
            "(a) Complete startup and speed response; (b) zoom around the speed change at 30 s. "
            "The solid curve is the propagated mean, the dashed grey curve is the nominal solution "
            "at the parameter means. Shading gives pointwise 2.5/97.5 percentiles; dotted curves "
            "give the explicitly approximate mean +/-1.96 SD reference. These are output-uncertainty "
            "bands, not confidence intervals for a mean or simultaneous coverage. "
            "The outlet statistic is an instantaneous volume-weighted diameter median. "
            "Empty initial mill, geometric-volume grid (100 pivots, 6.54e-17 to 1e-3 m3), "
            "positive fixed-pivot daughter projection, and segmented CVODE BDF with reltol=1e-9, "
            "abstol=1e-14. Initial state, upper domain and median interpolation are declared "
            "reconstruction choices; original random designs and unique source band degrees "
            "of freedom are unavailable. This is not an exact replication of the published curves."
        )

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.985, top=.91, bottom=.30, wspace=.12)
    for axis, name in zip(axes, COLORS):
        draw(axis, name, summaries[name], nominal)
        axis.set_ylim(limits)
        axis.legend(handles=[Line2D([], [], color=COLORS[name], label=LABELS[name])],
                    loc="upper right", frameon=False)
        axis.set_xticks([0, 100, 200])
    axes[0].set_ylabel(r"$d_{50}$ ($\mu$m)")
    panel_letters(axes)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.52, .005), columnspacing=1.5, handlelength=2.4)
    export_figure(fig, output, "milling_comparison")
    captions.append("MC/PCE comparison: (a) MC; (b) degree-one PCE; (c) degree-two PCE. "
                    "Axes and interval definitions match. All output quantiles retain the full "
                    "Gaussian input law; bounded regression designs do not truncate that law. "
                    "PCE quantiles use 50,000 seeded surrogate draws in production. "
                    "Numerical surrogate predictions are not clipped. See independent-validation "
                    "errors and the separate nine-point degree-one design sensitivity.")
    (output / "captions.txt").write_text("\n\n".join(captions) + "\n")
    (output / "figure_provenance.json").write_text(json.dumps({
        "production": not args.smoke, "latex": args.usetex, "dpi": 300,
        "configuration_key": manifest["configuration_key"], "python": sys.version,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(Path(__file__).with_name("plot_style.py").read_bytes()).hexdigest(),
        "source_numerical_hashes": manifest["numerical_files"],
        "presentation": "Stored numerical summaries; no smoothing or additional solves",
    }, indent=2) + "\n")
    print(f"Wrote four PNG/PDF/SVG figure sets to {output}")


if __name__ == "__main__":
    main()
