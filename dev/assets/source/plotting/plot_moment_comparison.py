"""Plot retained Qamar moment families; incomplete trajectories remain visible.

No model solves, CSV exports or provenance updates are performed here.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import configure, export_figure, panel_letters, relative_percent

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs/src/assets/qamar_moments"
METHODS = ("QMOM", "RankAdaptive", "GammaEQMOM", "DQMOM", "LogNormal")
LABELS = ("QMOM", "Rank-adaptive QMOM", "Gamma EQMOM", "DQMOM", "Assumed lognormal")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222")
STYLES = ("-", "--", "-.", ":", "-")
MARKERS = ("o", "s", "^", "d", "x")


def style(i, *, markers=False, count=1):
    out = dict(color=COLORS[i], ls=STYLES[i], lw=1.15, label=LABELS[i])
    if markers:
        out.update(marker=MARKERS[i], markevery=max(1, count//9),
                   ms=3.2, mfc="white", mew=.8)
    return out


def read(folder, name):
    return np.atleast_1d(np.genfromtxt(folder/name, names=True, delimiter=",",
                                     dtype=None, encoding="utf8"))


def aggregation(folder, data, summary):
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.1))
    fig.subplots_adjust(left=.12, right=.98, bottom=.19, top=.95, hspace=.55, wspace=.45)
    panel_letters(axes)
    for i, method in enumerate(METHODS):
        d = data[(data["scenario"] == "aggregation_breakage") & (data["method"] == method)]
        for ax, k in zip(axes[0], (1, 2)):
            q = d[d["order"] == k]
            error = relative_percent(q["moment"], q["reference"], display_floor=True)
            ax.semilogy(q["time"][1:], error[1:], **style(i, markers=True, count=len(q)))
            ax.set(xlabel=r"$t\ (\mathrm{s})$", ylabel=rf"Relative $\mu_{k}$ error $(\%)$")
        q = d[d["order"] == 3]
        axes[1, 0].plot(q["time"], q["moment"]/q["reference"],
                        **style(i, markers=True, count=len(q)))
        report = summary[(summary["scenario"] == "aggregation_breakage") &
                         (summary["method"] == method)][0]
        axes[1, 1].bar(i, report["seconds"], color=COLORS[i], width=.6)
        axes[1, 1].annotate(f"{report['states']} states", (i, report["seconds"]),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5)
    axes[1, 0].axhline(1, color=".5", lw=.7)
    axes[1, 0].set(xlabel=r"$t\ (\mathrm{s})$", ylabel=r"$\mu_3(t)/V_0$")
    axes[1, 1].set(xticks=range(5), xticklabels=["QMOM", "RA-Q", "EQMOM", "DQMOM", "LN"],
                    yscale="log", ylabel=r"One measured solve $(\mathrm{s})$")
    axes[1, 1].tick_params(axis="x", labelsize=7)
    plt.setp(axes[1, 1].get_xticklabels(), rotation=25, ha="right")
    axes[1, 1].margins(y=.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.52, .01))
    export_figure(fig, folder, "method_aggregation")


def pulse(folder, data, summary, reference):
    fig, axes = plt.subplots(2, 3, figsize=(7, 5.2))
    fig.subplots_adjust(left=.1, right=.985, bottom=.19, top=.94, hspace=.55, wspace=.5)
    panel_letters(axes)
    for row, scenario in enumerate(("growth_nucleation", "growth_nucleation_collisions")):
        for col, k in enumerate((0, 1, 3)):
            ax = axes[row, col]
            maximum, minimum = 0.0, np.inf
            for i, method in enumerate(METHODS):
                q = data[(data["scenario"] == scenario) & (data["method"] == method) &
                         (data["order"] == k)]
                if scenario == "growth_nucleation":
                    truth = q["reference"]
                else:
                    ref = reference[(reference["bins"] == max(reference["bins"])) &
                                    (reference["order"] == k)]
                    truth = np.interp(q["time"], ref["time"], ref["moment"])
                error = relative_percent(q["moment"], truth, display_floor=True)
                maximum = max(maximum, np.nanmax(error[1:]))
                minimum = min(minimum, np.nanmin(error[1:]))
                ax.semilogy(q["time"][1:], error[1:], **style(i, markers=True, count=len(q)))
                status = summary[(summary["scenario"] == scenario) & (summary["method"] == method)][0]
                if status["status"] != "completed" and len(q) > 1:
                    ax.plot(q["time"][-1], error[-1], "x", ms=6, mew=1.3, color=COLORS[i])
            upper = 10**np.ceil(np.log10(max(maximum*1.2, 1e-12)))
            lower = 10**np.floor(np.log10(minimum/2))
            ax.set(xlabel=r"$t\ (\mathrm{s})$", ylabel=rf"Relative $\mu_{k}$ difference $(\%)$",
                   ylim=(lower, upper))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.52, .01))
    export_figure(fig, folder, "method_pulse")


def main(folder=FOLDER, usetex=False):
    configure(usetex)
    data = read(folder, "method_comparison.csv")
    summary = read(folder, "method_summary.csv")
    reference = read(folder, "sectional_reference.csv")
    aggregation(folder, data, summary)
    pulse(folder, data, summary, reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=FOLDER)
    parser.add_argument("--usetex", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.usetex)
