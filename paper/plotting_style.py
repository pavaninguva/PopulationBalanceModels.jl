"""Compatibility imports for paper layouts; rendering lives in the package."""
from pathlib import Path
from contextlib import contextmanager
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from pbm_plotting.style import (SINGLE, DOUBLE, METHODS, LABELS, COLORS, MARKERS,
    STYLES, method_style, panel as _panel, style, save_figure)


_PANEL_NUMERALS = dict(zip("abcd", ("i", "ii", "iii", "iv")))


def panel_label(letter):
    """Format the paper's panel keys as lowercase Roman numerals."""
    return f"({_PANEL_NUMERALS[letter]})"


def panel(ax, letter, **properties):
    artist = _panel(ax, _PANEL_NUMERALS[letter])
    artist.set(**properties)
    return artist


@contextmanager
def alphabetic_panels(fig):
    """Preserve alphabetic labels in documentation copies of paper canvases."""
    from matplotlib.text import Text
    labels = {panel_label(letter): f"({letter})" for letter in _PANEL_NUMERALS}
    artists = [(item, item.get_text()) for item in fig.findobj(match=Text)
               if item.get_text() in labels]
    try:
        for item, value in artists:
            item.set_text(labels[value])
        yield
    finally:
        for item, value in artists:
            item.set_text(value)


def configure(preview=False, *, require_dvipng=False):
    """Scoped style; use ``with configure(...):`` before creating figures."""
    return style("preview" if preview else "publication")


def save_png(fig,path, *, creator="PopulationBalanceModels.jl paper workflow"):
    return save_figure(fig,path,dpi=300,creator=creator,vector_first=True)
