"""PopulationBalanceModels' optional, value-only Matplotlib renderer.

From a checkout: add ``<checkout>/python`` to PYTHONPATH. Julia's PythonPlot
extension locates this directory inside its installed package automatically.
"""
from .style import (style, settings, save_figure, method_style, panel, panel_letters,
                    relative_percent, SINGLE, DOUBLE, METHODS, LABELS, COLORS, MARKERS, STYLES)
from .render import (plot_distribution, plot_observable, plot_fit, plot_parameters,
                     plot_uq, plot_field)
