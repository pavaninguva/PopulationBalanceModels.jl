"""Shared typography for exported scientific figures; no display backend required."""
import matplotlib.pyplot as plt


def configure(usetex=False):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if usetex else ["DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "cm",
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.use_mathtext": True,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "svg.hashsalt": "PopulationBalanceModels-case-studies",
    })
