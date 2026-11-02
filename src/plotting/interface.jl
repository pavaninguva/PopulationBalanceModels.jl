"""
    plot_distribution(solution; times=nothing, population=nothing, cell=nothing,
                      marginal=nothing, reconstruction=nothing, ax=nothing, kwargs...)

Plot bin-average number densities at saved times, using actual bin widths.
Load PythonPlot to enable the shared Matplotlib renderer. No forward solve or
normalization is performed. For Grid2D select `marginal=:x`/`:y` explicitly, or
use plot_field. Moment states require an explicit reconstruction callback
`(state,operator,p,t)->(edges=...,density=...)`; no closure is invented.
`curve="centers"` connects stored bin averages for display; default is stairs.
Returns `(figure, axes)`; supply an existing `ax` for composition. Units/labels
may be supplied. `preset="publication"` requires real LaTeX; the default does not.
"""
function plot_distribution(args...;kwargs...)
    _plotting_unavailable()
end

"""
    plot_observable(solution, observable; times=nothing, measurement=nothing, ax=nothing, kwargs...)
    plot_observable(measurement; ax=nothing, kwargs...)

Render public observe results or measured data without solving. Observation
times must be saved exactly; no temporal interpolation is introduced. Fixed
Gaussian errors are marginal standard deviations; correlated covariance is
retained and identified, not presented as independent data. Callback noise
requires explicit evaluated errors or parameter coordinates in plot_fit.
Unknown units require caller labels. Returns `(figure, axes)`.
"""
function plot_observable(args...;kwargs...)
    _plotting_unavailable()
end

"""
    plot_fit(estimation, prepared_predictions; experiments=nothing, blocks=nothing,
             parameter_coordinates=nothing, axes=nothing, kwargs...)

Render values already computed by `predictions(estimation,z)`, grouped by
experiment and measurement block. This function NEVER calls predictions/fit,
sampling or a forward solver. MAP-only values require no posterior inputs.
Vector measurements at one time use their explicitly declared output coordinates
where available; a CDF remains a CDF. Other measurements use time trajectories.
Supply `abscissa=output_coordinates` to plot vector profiles at multiple times;
data, errors and covariance are permuted consistently, without guessing units.
`parameter_coordinates=z` permits evaluating declared noise callbacks without
solving. Supply optional prepared `bands` and an explicit `band_label` separately.
Returns `(figure, axes)`, one axis per selected block.
"""
function plot_fit(args...;kwargs...)
    _plotting_unavailable()
end

"""
    plot_parameters(estimation, result; coordinates=:inference, indices=nothing,
                    priors=true, axes=nothing, kwargs...)

Plot retained PosteriorSamples, a checked Laplace/GaussNewtonApproximation, or
MAP points (which are not a posterior distribution). `coordinates=:physical`,
`:inference` or `:log` explicitly selects display coordinates. Parameter ordering
includes selected shared/local/nuisance coordinates; no units or logarithms are
inferred from names. Priors retain their declared law and display Jacobian.
Gaussian contours map native probability ellipses; nonlinear maps generally do
not preserve Gaussianity or highest-density regions. Retained-draw KDE contours
use integrated grid probability. Diagnostic failures remain visible; no draws
are clipped, filtered or replaced. No PBM evaluations are performed.
"""
function plot_parameters(args...;kwargs...)
    _plotting_unavailable()
end

"""
    plot_uq(result; output, band=:quantiles, probabilities=(0.025,0.975), ax=nothing, kwargs...)

Plot a named MC/PCE output using uq_summary. MC quantiles describe pointwise
output variability, not Monte Carlo error. PCE analytical summaries supply only
mean/variance: use `band=:none` or explicitly `band=:sd`, or first call
evaluate_pce to obtain surrogate quantiles. No PCE training, propagation or
Gaussian quantile assumption occurs here. Failed samples are not silently
removed; pass a separately labelled, explicit success-conditioned summary if
that is the intended diagnostic. Returns `(figure, axes)`.
"""
function plot_uq(args...;kwargs...)
    _plotting_unavailable()
end

"""
    plot_field(solution; index=length(solution.u), population=nothing,
               bulk=nothing, orientation, ax=nothing, kwargs...)

Render a Grid2D density or a 1D-internal-coordinate/axial field on actual edges.
Specify `orientation=:internal_x` (first internal coordinate horizontal) or
`:axial_x` (axial coordinate horizontal). For bulk trajectories use
plot_observable with CellObservable(BulkObservable(name),cell).
No interpolation, inferred tensor marginal or hidden normalization is applied.
Tensor populations with an additional axial coordinate require an explicitly
selected local cell via plot_distribution/plot_field. Returns `(figure, axes)`.
"""
function plot_field(args...;kwargs...)
    _plotting_unavailable()
end

"""
    save_figure(figure_or_plot, path; dpi=300, width=nothing, height=nothing,
                font_size=nothing, layout=nothing)

Save exactly the requested PNG/PDF/SVG file through the optional PythonPlot
renderer. No sibling formats are created by default. Width/height are inches;
changing DPI preserves physical dimensions and fonts. Formatting changes precede
layout recomputation. A preview is not publication LaTeX output. Numerical
objects and retained producer records are never written by rendering.
"""
function save_figure(args...;kwargs...)
    _plotting_unavailable()
end

_plotting_unavailable()=throw(ArgumentError("Load the optional PythonPlot package (`import PythonPlot`) to enable plotting; simulation does not require it. See the plotting guide."))
