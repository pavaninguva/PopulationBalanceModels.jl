"""Scoped typography, method styles and physical-size exports (no backend choice)."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import io
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SINGLE, DOUBLE = 3.35, 7.0
METHODS = ["upwind", "muscl", "weno3", "weno5_ssprk33", "weno5", "hoc", "inguva2022"]
LABELS = dict(zip(METHODS, ["Upwind + Euler", "MUSCL + SSPRK3", "WENO3 + SSPRK3",
    "WENO5 + SSPRK3", "WENO5 + RK4", "HOC + midpoint", r"Inguva2022 ($\mathrm{CFL}=1$)"]))
COLORS = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#6A3D9A", "#222222"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "P", "v", "D", "x"]))
STYLES = dict(zip(METHODS, ["-", "--", "-.", "-.", "-", "--", ":"]))
RELATIVE_DISPLAY_FLOOR = 1e-16


def renderer_sources():
    """Renderer identity for plotting receipts, separate from numerical producers."""
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def settings(preset="default", *, font_size=8.5):
    if preset not in ("default", "preview", "publication"):
        raise ValueError("preset must be default, preview or publication")
    publication = preset == "publication"
    if publication and (not shutil.which("latex") or not shutil.which("dvipng")):
        raise RuntimeError("The publication preset requires latex and dvipng. Use preset='preview' for a labelled preview.")
    return {"text.usetex": publication, "font.family": "serif" if publication else "sans-serif",
        "font.serif": ["serif"] if publication else ["DejaVu Serif"],
        "font.sans-serif": ["sans-serif"] if publication else ["DejaVu Sans"],
        "font.monospace": ["Courier"] if publication else ["DejaVu Sans Mono"],
        "font.size": font_size, "axes.labelsize": font_size+.5,
        "legend.fontsize": font_size-.5, "xtick.labelsize": font_size-.5,
        "ytick.labelsize": font_size-.5, "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": None,
        "pdf.fonttype": 42, "svg.hashsalt": "PopulationBalanceModels"}


@contextmanager
def style(preset="default", *, font_size=8.5, **overrides):
    """Apply formatting before figure creation, restoring unrelated rcParams."""
    values = settings(preset, font_size=font_size)
    values.update(overrides)
    with mpl.rc_context(values):
        yield


def method_style(name, markers=True):
    result = dict(color=COLORS[name], ls=STYLES[name], lw=1.2, label=LABELS[name])
    if markers:
        result.update(marker=MARKERS[name], ms=3.5, mew=.8, mfc="white")
    return result


def panel(ax, letter):
    return ax.text(-.12, 1.02, f"({letter})", transform=ax.transAxes,
                   ha="left", va="bottom", fontsize=10)


def panel_letters(axes):
    for letter, ax in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).flat):
        panel(ax, letter)


def relative_percent(value, reference, *, display_floor=False):
    value, reference = np.broadcast_arrays(value, reference)
    result = np.full(value.shape, np.nan, dtype=float)
    np.divide(np.abs(value-reference), np.abs(reference), out=result, where=reference != 0)
    return 100*np.maximum(result, RELATIVE_DISPLAY_FLOOR) if display_floor else 100*result


def _clipped_text(fig):
    fig.canvas.draw()
    if hasattr(fig.canvas,'get_renderer'):
        renderer=fig.canvas.get_renderer();frame=fig.bbox
        texts=list(fig.texts)
        for ax in fig.axes:
            texts.extend(ax.texts)
            if ax.axison:
                texts.extend([ax.xaxis.label,ax.yaxis.label,ax.xaxis.offsetText,ax.yaxis.offsetText])
                # Locators can retain labels outside the view interval;
                # Matplotlib does not draw those ticks, even if their Text
                # object's visible flag is true.
                for coordinate,limits in ((ax.xaxis,ax.get_xbound()),(ax.yaxis,ax.get_ybound())):
                    lower,upper=sorted(limits)
                    for tick in coordinate.get_major_ticks()+coordinate.get_minor_ticks():
                        if lower<=tick.get_loc()<=upper:
                            texts.extend([tick.label1,tick.label2])
            legend=ax.get_legend()
            if legend is not None:texts.extend(legend.get_texts())
        for legend in fig.legends:texts.extend(legend.get_texts())
        clipped=[]
        for item in texts:
            if not item.get_visible() or not item.get_text():continue
            box=item.get_window_extent(renderer)
            if box.x0<frame.x0-2 or box.y0<frame.y0-2 or box.x1>frame.x1+2 or box.y1>frame.y1+2:
                clipped.append(item.get_text())
        return clipped
    return []


def save_figure(figure, path, *, dpi=300, width=None, height=None, font_size=None,
                layout=None, creator="PopulationBalanceModels.jl", vector_first=False,check_layout=True):
    """Write exactly the requested file, preserving inches/fonts when DPI changes.

    Optional vector-first PNG rasterization uses PyMuPDF. No vector sibling is
    written. Layout is recalculated after any formatting changes, before export.
    """
    fig = figure[0] if isinstance(figure, tuple) else figure
    path = Path(path)
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    size = fig.get_size_inches()
    if width is not None or height is not None:
        w = size[0] if width is None else width
        h = size[1]*w/size[0] if height is None else height
        if not w > 0 or not h > 0:
            raise ValueError("physical dimensions must be positive")
        fig.set_size_inches(w, h)
    if font_size is not None:
        from matplotlib.text import Text
        for item in fig.findobj(Text):
            item.set_fontsize(font_size)
    if layout == "tight":
        fig.tight_layout()
    elif layout is not None:
        fig.set_layout_engine(layout)
    elif width is not None or height is not None or font_size is not None:
        if fig.get_layout_engine() is None:
            fig.tight_layout()
    if check_layout:
        clipped=_clipped_text(fig)
        # An unmanaged ordinary subplot may simply lack calculated margins.
        # Resolve those once, preserving inches/fonts. Explicit colourbar axes
        # and figure-level publication legends retain their manual layout.
        if clipped and layout is None and fig.get_layout_engine() is None and not fig.legends and \
                all(ax.get_subplotspec() is not None for ax in fig.axes):
            fig.tight_layout()
            clipped=_clipped_text(fig)
        if clipped:
            raise ValueError('Figure layout clips text: '+repr(clipped[:3])+'. Increase figsize/change legend or spacing before saving; check_layout=False is an explicit override.')
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": creator}
    if path.suffix == ".pdf":
        metadata.update(CreationDate=None, ModDate=None)
    if path.suffix == ".svg":
        metadata.update(Date=None)
    if vector_first and path.suffix == ".png":
        try:
            import pymupdf
        except ImportError as error:
            raise RuntimeError("Vector-first PNG export requires the optional PyMuPDF package") from error
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", dpi=dpi, metadata={"Creator": creator, "CreationDate": None, "ModDate": None})
        with pymupdf.open(stream=buffer.getvalue(), filetype="pdf") as doc:
            doc[0].get_pixmap(dpi=dpi, alpha=False).save(path)
    else:
        fig.savefig(path, dpi=dpi, metadata=metadata, bbox_inches=None)
    from PIL import Image
    if path.suffix == ".png":
        with Image.open(path) as image:
            actual=image.info.get('dpi',())
            if len(actual)!=2 or any(abs(value-dpi)>.1 for value in actual):
                raise ValueError('PNG lacks the requested DPI metadata')
            return dict(png_pixels=list(image.size), png_dpi=list(actual),renderer_sources=renderer_sources())
    return dict(path=str(path), inches=list(fig.get_size_inches()),renderer_sources=renderer_sources())
