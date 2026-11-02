"""Compatibility imports for the package-owned Matplotlib renderer."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"python"))
from pbm_plotting.style import (style, panel_letters, relative_percent,
    RELATIVE_DISPLAY_FLOOR, save_figure)


def configure(usetex=False):
    return style("publication" if usetex else "preview")


def export_figure(fig, folder, name, *, vector=True):
    """Preserve this workflow's explicitly selected output formats."""
    folder=Path(folder)
    save_figure(fig,folder/f"{name}.png")
    if vector:
        save_figure(fig,folder/f"{name}.pdf")
        save_figure(fig,folder/f"{name}.svg")
    plt.close(fig)
