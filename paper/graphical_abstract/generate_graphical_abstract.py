#!/usr/bin/env python3
"""Draw the package graphical abstract without simulations or retained datasets.

Run from the repository root with the existing paper Python environment:
    python paper/graphical_abstract/generate_graphical_abstract.py
    python paper/graphical_abstract/generate_graphical_abstract.py --output paper/figures/00_graphical_abstract/graphical_abstract.pdf

All process icons and miniature plots are conceptual illustrations, not numerical
results. The default exports one 300-dpi PNG with soft sans-serif lettering and
an actual LaTeX equation, matching the package architecture schematic. An
explicit --preview writes a separate mathtext preview. No companion files or
unrequested formats are written; a .pdf output selects a vector export.
"""
import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
from architecture.generate_pbm_architecture import COLORS, configure_schematic
from plotting_style import configure, save_figure

CANVAS = (240.0, 94.0)
DEFAULT_OUTPUT = PAPER / "figures/00_graphical_abstract/graphical_abstract.png"


def label(ax, x, y, value, *, size=9.0, color="ink", align="left", bold=False,
          usetex=False):
    return ax.text(x, y, value, ha=align, va="center", fontsize=size,
                   color=COLORS[color], weight="bold" if bold else "normal",
                   fontfamily="serif" if usetex else None,
                   linespacing=1.25, usetex=usetex, zorder=8)


def card(ax, x, y, width, height, *, role, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0,rounding_size=2.0",
        facecolor=COLORS[role + "_fill"],
        edgecolor=COLORS[role] if dashed else COLORS["line"],
        linewidth=.65, linestyle=(0, (3, 2)) if dashed else "-", zorder=0))


def arrow(ax, start, end, *, role="model", dashed=False, both=False, width=.85):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->" if both else "-|>",
        mutation_scale=8, linewidth=width, color=COLORS[role],
        linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0, zorder=3))


def particle(ax, x, y, radius, *, alpha=1.0):
    ax.add_patch(Circle((x, y), radius, facecolor=COLORS["model"], alpha=alpha,
                        edgecolor="none", zorder=4))


def mechanisms(ax):
    label(ax, 6, 75.5, "Population mechanisms", size=10.5, color="model", bold=True)
    card(ax, 6, 4, 64, 68, role="model")
    for x, name in ((18, "Particles"), (38, "Cells"), (58, "Bubbles")):
        label(ax, x, 67, name, size=9.4, align="center")

    # Alternative population illustrations, not a mandatory coupled system.
    for x, y, radius, alpha in ((12.5, 59.5, 1.15, .50), (17, 61, 1.55, .85),
                                 (21, 59.2, 1.3, .65)):
        particle(ax, x, y, radius, alpha=alpha)
    crystal = np.array([(24, 58), (22.5, 61), (24.5, 63),
                        (27, 61.5), (27.5, 58.5)])
    ax.add_patch(Polygon(crystal, closed=True, facecolor=COLORS["model"],
                         edgecolor="white", linewidth=.65, alpha=.8))
    ax.plot([24, 24.5, 27], [58, 61, 61.5], color="white", lw=.55)
    for x, y, radius in ((36.2, 60.7, 2.5), (43, 59.7, 1.7)):
        ax.add_patch(Ellipse((x, y), 2.2 * radius, 1.8 * radius, angle=20,
            facecolor="white", edgecolor=COLORS["model"], linewidth=1.0))
        particle(ax, x + .25, y, .65, alpha=.7)
    for x, y, radius in ((55, 61, 2.6), (62, 59.6, 2.0)):
        ax.add_patch(Circle((x, y), radius, facecolor="white",
                            edgecolor=COLORS["model"], linewidth=1.0))
        ax.add_patch(Arc((x, y), 1.4 * radius, 1.4 * radius, theta1=55, theta2=145,
                         color=COLORS["model"], linewidth=.7, alpha=.5))
    ax.plot([11, 65], [54, 54], color=COLORS["line"], lw=.6)

    particle(ax, 13, 47.8, 1.2, alpha=.55)
    arrow(ax, (16.0, 47.8), (22.2, 47.8), width=.7)
    particle(ax, 26, 47.8, 2.1)
    label(ax, 33, 47.8, "Growth", size=9.4)

    label(ax, 11, 38.8, "+", size=13, color="model", align="center")
    particle(ax, 18, 38.8, 1.5, alpha=.8)
    label(ax, 26.5, 38.8, "−", size=13, color="model", align="center")
    label(ax, 33, 38.8, "Birth / Death", size=9.4)

    particle(ax, 12.2, 31.5, 1.25, alpha=.65)
    particle(ax, 12.2, 28.0, 1.25, alpha=.65)
    arrow(ax, (15.5, 29.8), (22.8, 29.8), both=True, width=.7)
    particle(ax, 26.3, 29.8, 1.25 * 2 ** (1 / 3))
    label(ax, 33, 29.8, "Aggregation /\nBreakage", size=9.1)

    x = np.linspace(0, 1, 200)
    narrow = .58 * np.exp(-.5 * ((x - .5) / .09) ** 2) / .09
    broad = .58 * np.exp(-.5 * ((x - .5) / .18) ** 2) / .18
    ax.plot(10 + 19 * x, 17.3 + narrow, color=COLORS["model"], lw=.8,
            ls=(0, (2, 1.5)), alpha=.45)
    ax.plot(10 + 19 * x, 17.3 + broad, color=COLORS["model"], lw=1.15)
    label(ax, 33, 20.8, "Growth Rate\nDispersion", size=9.1)

    ax.add_patch(FancyBboxPatch((14, 8.2), 10, 4.8,
        boxstyle="round,pad=0,rounding_size=1.1", facecolor="white",
        edgecolor=COLORS["model"], linewidth=.75, linestyle=(0, (2.5, 2))))
    label(ax, 19, 10.6, "+", size=12, color="model", align="center")
    label(ax, 33, 10.6, "Custom Terms", size=9.4)


def model_and_numerics(ax, *, equation_usetex):
    label(ax, 82, 75.5, "Shared forward model", size=10.5,
          color="numerics", bold=True)
    card(ax, 82, 22.5, 71, 49.5, role="numerics")
    label(ax, 117.5, 66.7, "Composable population balance", size=10.0,
          color="numerics", align="center", bold=True)
    equation = (r"$\displaystyle\frac{\partial n}{\partial t}=\sum_i f_i[n]$"
                if equation_usetex else r"$\frac{\partial n}{\partial t}=\sum_i f_i[n]$")
    label(ax, 117.5, 55.5, equation, size=17.5, align="center",
          usetex=equation_usetex)
    ax.plot([88, 147], [45, 45], color=COLORS["line"], lw=.6)
    label(ax, 117.5, 42.2, "Select compatible numerics", size=9.1, align="center")

    # Schematic bin populations and quadrature nodes identify two formulations;
    # they do not prescribe the solver-state layout of every moment family.
    edges = np.linspace(0, 1, 10)
    for left, right in zip(edges[:-1], edges[1:]):
        height = 6.0 * np.exp(-.5 * (((left + right) / 2 - .5) / .20) ** 2)
        ax.add_patch(plt.Rectangle((89 + 23 * left, 32.3), 23 * (right - left),
            height, facecolor=COLORS["numerics"], edgecolor="white", linewidth=.4,
            alpha=.58, zorder=2))
    ax.plot([89, 112], [32.3, 32.3], color=COLORS["numerics"], lw=.6)
    label(ax, 100.5, 27.2, "Direct\ndiscretization", size=8.7, align="center")
    nodes = np.array([128.5, 136, 144])
    heights = np.array([2.6, 6.1, 3.8])
    ax.vlines(nodes, 32.3, 32.3 + heights, colors=COLORS["numerics"], lw=1.2)
    ax.scatter(nodes, 32.3 + heights, s=[15, 24, 19], c=COLORS["numerics"], zorder=3)
    label(ax, 136, 27.2, "Moment\nformulations", size=8.7, align="center")

    label(ax, 117.5, 19.5, "Transient / Stationary", size=9.0,
          color="numerics", align="center")
    card(ax, 82, 4, 71, 12, role="numerics", dashed=True)
    label(ax, 117.5, 12, "Optional coupling", size=8.7, align="center", bold=True)
    label(ax, 117.5, 7.2, "Bulk balances · Reactor transport", size=8.4,
          align="center")
    arrow(ax, (146, 16), (146, 22.5), role="numerics", dashed=True, width=.7)


def miniature(fig, bounds, *, ylabel, xlabel):
    x, y, width, height = bounds
    ax = fig.add_axes([x / CANVAS[0], y / CANVAS[1],
                       width / CANVAS[0], height / CANVAS[1]])
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xticks=[], yticks=[])
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
        ax.spines[side].set_linewidth(.6)
    ax.text(-.14, .97, ylabel, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color=COLORS["ink"])
    ax.text(1., -.08, xlabel, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=COLORS["ink"])
    return ax


def analyses(fig, ax):
    label(ax, 170, 75.5, "Analysis", size=10.5, bold=True)
    rows = ((51.8, "numerics", "Simulate", "Population\ndynamics"),
            (28.5, "inference", "Estimate", "Parameters\nfrom data"),
            (5.2, "uq", "Quantify", "Uncertainty\nMC / PCE"))
    for bottom, role, title, subtitle in rows:
        card(ax, 170, bottom, 64, 20.7, role=role)
        label(ax, 174, bottom + 15.6, title, size=11.0, color=role, bold=True)
        label(ax, 174, bottom + 8.2, subtitle, size=8.6)

    x = np.linspace(0, 1, 240)
    dynamics = miniature(fig, (204.5, 56.1, 26.7, 12.8), ylabel=r"$n$", xlabel=r"$x$")
    for center, alpha in ((.22, .25), (.48, .55), (.74, 1.)):
        y = .90 * np.exp(-.5 * ((x - center) / .085) ** 2)
        dynamics.plot(x, y, color=COLORS["numerics"], lw=1.35, alpha=alpha)
    arrow(ax, (210, 70.3), (225.7, 70.3), role="numerics", width=.6)
    label(ax, 228, 70.3, r"$t$", size=8.0, color="numerics", align="center")

    fitted = miniature(fig, (204.5, 32.8, 26.7, 12.8), ylabel=r"$y$", xlabel=r"$t$")
    response = lambda t: .16 + .76 * np.exp(-2.6 * t)
    fitted.plot(x, response(x), color=COLORS["inference"], lw=1.4)
    # Fixed illustrative offsets: these are not experimental measurements or a fit.
    times = np.array([.06, .19, .34, .49, .65, .81, .96])
    observed = response(times) + np.array([.035, -.030, .025, -.035, .015, -.020, .025])
    fitted.scatter(times, observed, s=11, facecolors="white", edgecolors=COLORS["ink"],
                   linewidths=.65, zorder=4)

    uncertainty = miniature(fig, (204.5, 9.5, 26.7, 12.8), ylabel=r"$y$", xlabel=r"$t$")
    mean = .22 + .49 * (1 - np.exp(-2.2 * x))
    spread = .025 + .13 * x
    uncertainty.fill_between(x, mean - spread, mean + spread,
                             color=COLORS["uq"], alpha=.22, linewidth=0)
    uncertainty.plot(x, mean - spread, color=COLORS["uq"], alpha=.45, lw=.55)
    uncertainty.plot(x, mean + spread, color=COLORS["uq"], alpha=.45, lw=.55)
    uncertainty.plot(x, mean, color=COLORS["uq"], lw=1.35)


def build_figure(*, width=10.0, preview=False):
    """Compose independent schematic analysis branches using one shared model."""
    fig = plt.figure(figsize=(width, width * CANVAS[1] / CANVAS[0]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, CANVAS[0]), ylim=(0, CANVAS[1]))
    ax.axis("off")
    label(ax, 6, 89.5, "PopulationBalanceModels.jl", size=20.0, bold=True)
    tagline = "Define the physics once. Reuse the model."
    label(ax, 6, 81.8, tagline + ("  [Preview]" if preview else ""), size=11.0,
          color="muted")
    mechanisms(ax)
    model_and_numerics(ax, equation_usetex=not preview)
    arrow(ax, (70, 45.5), (81.5, 45.5), width=1.05)

    # Independent branches: UQ does not require a preceding estimation step.
    ax.plot([153, 162], [45.5, 45.5], color=COLORS["numerics"], lw=.9)
    ax.plot([162, 162], [15.5, 62.1], color=COLORS["line"], lw=.9)
    for y, role in ((62.1, "numerics"), (38.8, "inference"), (15.5, "uq")):
        ax.add_patch(Circle((162, y), .55, facecolor=COLORS[role], edgecolor="none"))
        arrow(ax, (162, y), (169.8, y), role=role, width=.85)
    analyses(fig, ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination PNG or vector PDF (default: paper/figures/00_graphical_abstract/graphical_abstract.png)")
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches; default: 10")
    parser.add_argument("--preview", action="store_true",
                        help="Use mathtext instead of LaTeX; write a separately named preview")
    args = parser.parse_args()
    if not np.isfinite(args.width) or args.width <= 0:
        parser.error("width must be finite and positive")
    if args.output.suffix.lower() not in (".png", ".pdf"):
        parser.error("the graphical abstract exports PNG or PDF")
    if not args.preview:
        missing = [tool for tool in ("latex", "dvipng") if not shutil.which(tool)]
        if missing:
            parser.error("the equation requires " + ", ".join(missing) + "; install it or use --preview")
    output = args.output
    if args.preview:
        output = output.with_stem(output.stem + "_preview")
    with configure(preview=True):
        configure_schematic()
        fig = build_figure(width=args.width, preview=args.preview)
        try:
            save_figure(fig, output, dpi=300, vector_first=output.suffix.lower() == ".png",
                        creator="PopulationBalanceModels.jl schematic graphical abstract")
        finally:
            plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
