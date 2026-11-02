"""Compatibility imports for paper layouts; rendering lives in the package."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from pbm_plotting.style import (SINGLE, DOUBLE, METHODS, LABELS, COLORS, MARKERS,
    STYLES, method_style, panel, style, save_figure)


def configure(preview=False, *, require_dvipng=False):
    """Scoped style; use ``with configure(...):`` before creating figures."""
    return style("preview" if preview else "publication")


def save_png(fig,path, *, creator="PopulationBalanceModels.jl paper workflow"):
    return save_figure(fig,path,dpi=300,creator=creator,vector_first=True)
