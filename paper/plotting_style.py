"""Paper-only style: actual LaTeX, physical widths and no figure titles."""
import shutil
import io
from pathlib import Path
import pymupdf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE = 3.35
DOUBLE = 7.0
METHODS = ["upwind","muscl","weno3","weno5_ssprk33","weno5","hoc","inguva2022"]
LABELS = {"upwind":"Upwind + Euler", "muscl":"MUSCL + SSPRK3", "weno3":"WENO3 + SSPRK3",
          "weno5_ssprk33":"WENO5 + SSPRK3", "weno5":"WENO5 + RK4",
          "inguva2022":"Inguva2022 ($C=1$)", "hoc":"HOC + midpoint"}
COLORS = dict(zip(METHODS,["#0072B2","#D55E00","#009E73","#56B4E9","#CC79A7","#6A3D9A","#222222"]))
MARKERS = dict(zip(METHODS,["o","s","^","P","v","D","x"]))
STYLES = dict(zip(METHODS,["-","--","-.","-.","-","--",":"]))

def save_png(fig,path, *, creator="PopulationBalanceModels.jl paper workflow"):
    """Render LaTeX artwork to a 300-dpi PNG without leaving vector derivatives.

    The intermediate PDF exists only in memory. This retains the established
    protein/schematic rasterization and requires no extra simulation work.
    """
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    vector=io.BytesIO()
    fig.savefig(vector,format="pdf",dpi=300,
                metadata={"Creator":creator,"CreationDate":None,"ModDate":None})
    with pymupdf.open(stream=vector.getvalue(),filetype="pdf") as document:
        document[0].get_pixmap(dpi=300,alpha=False).save(path)
    with Image.open(path) as image:
        dpi=image.info.get("dpi",())
        if len(dpi)!=2 or any(abs(v-300)>.1 for v in dpi):
            raise ValueError("PNG lacks 300-dpi metadata")
        return dict(png_pixels=list(image.size),png_dpi=list(dpi))

def configure(preview=False, *, require_dvipng=False):
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
        "text.latex.preamble":r"\usepackage{amsmath}","pdf.fonttype":42})

def panel(ax,letter):
    ax.text(-.12,1.02,f"({letter})",transform=ax.transAxes,ha="left",va="bottom",fontsize=10)

def method_style(name,markers=True):
    style = dict(color=COLORS[name],ls=STYLES[name],lw=1.2,label=LABELS[name])
    if markers:
        style.update(marker=MARKERS[name],ms=3.5,mew=.8,mfc="white")
    return style
