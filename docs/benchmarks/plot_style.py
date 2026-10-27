"""Shared typography for exported scientific figures; no display backend required."""
import shutil
import numpy as np
import matplotlib.pyplot as plt

RELATIVE_DISPLAY_FLOOR = 1e-16  # fraction; 1e-14 when displayed as percent


def relative_percent(value, reference, *, display_floor=False):
    """Absolute relative difference against the stated reference, in percent.

    Relative error is undefined at a zero reference; mask it rather than divide by epsilon.
    The optional display floor does not alter numerical exports.
    """
    value, reference = np.broadcast_arrays(value, reference)
    out = np.full(value.shape, np.nan, dtype=float)
    np.divide(np.abs(value-reference), np.abs(reference), out=out, where=reference != 0)
    if display_floor:
        out = np.maximum(out, RELATIVE_DISPLAY_FLOOR)
    return 100*out


def panel_letters(axes):
    for letter, axis in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).flat):
        axis.text(-.16, 1.03, f"({letter})", transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=10)


def export_figure(fig, folder, name, *, vector=True):
    """Plot-only exports: 300-dpi PNG and vector PDF/SVG, with numerical data untouched."""
    folder.mkdir(parents=True, exist_ok=True)
    for axis in fig.axes:
        axis.grid(False, which="both")
    fig.savefig(folder/f"{name}.png", dpi=300)
    if vector:
        fig.savefig(folder/f"{name}.pdf", dpi=300,
                    metadata={"CreationDate": None, "ModDate": None})
        path = folder/f"{name}.svg"
        fig.savefig(path, dpi=300, metadata={"Date": None})
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def configure(usetex=False):
    if usetex and (not shutil.which("latex") or not shutil.which("dvipng")):
        raise RuntimeError("Final LaTeX figures require latex and dvipng; omit --usetex only for previews")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["serif"] if usetex else ["DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "cm",
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.use_mathtext": True,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.hashsalt": "PopulationBalanceModels-case-studies",
    })
