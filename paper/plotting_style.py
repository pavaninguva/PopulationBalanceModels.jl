"""Paper-only style: actual LaTeX, physical widths and no figure titles."""
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE = 3.35
DOUBLE = 7.0
METHODS = ["upwind","muscl","weno3","weno5","inguva2022","hoc"]
LABELS = {"upwind":"Upwind + Euler", "muscl":"MUSCL + SSPRK3", "weno3":"WENO3 + SSPRK3",
          "weno5":"WENO5 + RK4", "inguva2022":"Inguva2022 ($C=1$)", "hoc":"HOC + midpoint"}
COLORS = dict(zip(METHODS,["#0072B2","#D55E00","#009E73","#CC79A7","#222222","#6A3D9A"]))
MARKERS = dict(zip(METHODS,["o","s","^","v","x","D"]))
STYLES = dict(zip(METHODS,["-","--","-.","-",":","--"]))

def configure(preview=False, *, require_dvipng=True):
    if not preview and (not shutil.which("latex") or (require_dvipng and not shutil.which("dvipng"))):
        required="latex and dvipng" if require_dvipng else "latex"
        raise RuntimeError(f"Final figures require {required}; --preview is smoke-only")
    plt.rcParams.update({"text.usetex":not preview,"font.family":"serif",
        # LaTeX's default serif is Computer Modern. Generic family declarations
        # avoid adding the unrelated CM-Super/type1ec package to every figure.
        "font.serif":["serif"] if not preview else ["DejaVu Serif"],
        "font.sans-serif":["sans-serif"] if not preview else ["DejaVu Sans"],
        "font.monospace":["Courier"] if not preview else ["DejaVu Sans Mono"],
        "font.size":8.5,"axes.labelsize":9,"legend.fontsize":8,
        "xtick.labelsize":8,"ytick.labelsize":8,"axes.grid":False,
        "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,
        "savefig.facecolor":"white","figure.facecolor":"white",
        "savefig.bbox":None,"figure.constrained_layout.use":False,
        "text.latex.preamble":r"\usepackage{amsmath}","pdf.fonttype":42,
        "svg.hashsalt":"PopulationBalanceModels-paper"})

def panel(ax,letter):
    ax.text(-.12,1.02,f"({letter})",transform=ax.transAxes,ha="left",va="bottom",fontsize=10)

def method_style(name,markers=True):
    style = dict(color=COLORS[name],ls=STYLES[name],lw=1.2,label=LABELS[name])
    if markers:
        style.update(marker=MARKERS[name],ms=3.5,mew=.8,mfc="white")
    return style
