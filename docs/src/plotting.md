# Plotting numerical results

Compute once, then plot the result. The optional plotting interface reads package
solutions, observations, prepared predictions, parameter results and forward-UQ
summaries. It uses one package-owned Matplotlib renderer shared with the Python
case-study plotters. Rendering never fits, samples, solves a PBM or trains a PCE.
Numerical calculations need neither Python nor LaTeX.

## Optional setup and first plot

In the environment containing this checkout or the installed package:

```julia
using Pkg
Pkg.add("PythonPlot") # Optional; not a simulation dependency.
```

[PythonPlot](https://github.com/JuliaPy/PythonPlot.jl) uses PythonCall and normally
installs Matplotlib through CondaPkg. The package supports Julia 1.10 and newer;
PythonPlot 1.x itself supports Julia 1.6 and newer. For an existing Python,
configure `JULIA_PYTHONCALL_EXE` and `JULIA_CONDAPKG_BACKEND` **before** starting
Julia, following PythonPlot/PythonCall's instructions; that Python needs NumPy,
Matplotlib and Pillow. Repository batch scripts additionally use the environment
in `paper/requirements.txt` and Python 3.11 or newer. PyMuPDF is needed only for
their vector-first PNG export, not ordinary package plotting.

```julia
using PopulationBalanceModels
using Distributions: Normal
import PythonPlot

grid = Grid1D(range(0.0,2.0;length=81))
operator = compile(PBMModel(growth=Growth((x,p,t,ctx)->p.speed)),grid;
    methods=TermMethods(growth=MUSCL()))
initial = InitialDistribution(Normal(0.6,0.08);total=2.0,truncation=:physical)
problem = PBMProblem(operator,initial,(0.0,1.0);p=(speed=0.1,))
solution = simulate(problem;solver=ForwardSolver(SSPRK33();dt=0.01),saveat=[0.0,0.5,1.0])

picture = plot_distribution(solution; xlabel="Internal coordinate",ylabel="Number density")
save_figure(picture,"distribution.png") # Only this file, at 300 dpi.
```

Every plot returns `(figure, axes)` as a named tuple. Handles remain available
for Matplotlib customization, e.g. `picture.axes.set_xlim(0.2,1.2)`. Supply `ax`
to reuse an axis; multi-block fits/corners accept `axes`. Draw comparisons by
calling the function again on the same axis with prepared references or another
solution. Save explicitly after changing labels, legends or spacing.

For a generated minimum case-study script, numerical-only execution is unchanged:

```julia
include("docs/src/assets/source/tutorials/stem_cells.jl")
result = main(bins=40,times=[0.0,0.05,0.1])
import PythonPlot
pictures = plot_results(result) # No new forward evaluations.
save_figure(pictures.distribution,"stem_distribution.png")
```

## Scientific meaning of the plots

| Function | Values displayed and explicit choices |
|:--|:--|
| `plot_distribution(solution; times, population, cell)` | Number density from bin integrals divided by actual widths; no normalization. Times must already be saved. `curve="centers"` only connects bin averages for display. |
| `plot_observable(solution, observable; measurement)` | Public `observe` values; measurements align to saved times. Moment trajectories use their observables, including DQMOM. |
| `plot_observable(measurement)` | Data-only plot; fixed Gaussian errors use their declared marginal SD. A covariance is retained and labelled as correlated. |
| `plot_fit(estimation, prepared_predictions)` | One axis per selected experiment/block. Data and predictions keep the observation response, window and weighting. Count likelihoods use the existing expected-count conversion. A CDF is not converted into a density. |
| `plot_parameters(estimation, result; coordinates)` | Retained joint MCMC draws, a checked local Gaussian, or MAP points alone. Select `:inference`, `:physical` or `:log` explicitly. |
| `plot_uq(result; output, band)` | Named MC/PCE output summaries. Quantiles describe pointwise output variability, not MC error of a mean or simultaneous coverage. |
| `plot_field(solution; orientation)` | Actual rectangular mesh edges; `:internal_x` for Grid2D and `:axial_x` for axial fields. No interpolation onto a finer mesh. |

Name a population for `PBMSystem` and select an axial cell for an outlet density.
A joint Grid2D density uses `plot_field`; an explicit `marginal=:x` or `:y`
selects a one-coordinate density. Summing tensor bin integrals over the other
axis already integrates it: there is no extra transverse-width multiplication.
For an axial tensor population, select a local `cell`; a full three-coordinate
field requires a separately declared projection.

Moment states do not uniquely determine a density. Distribution plots require
an explicitly supplied reconstruction callback returning `(edges,density)`;
the renderer never selects a closure or interprets DQMOM entries as raw moments.
Signed diagnostic populations remain signed. No clipping or smoothing repairs
a numerical solution. Labels and units are caller supplied when absent from
metadata; parameter names do not imply units or logarithmic coordinates.

Converged stationary solutions support distribution and observable plots without
an artificial time axis. Stationary measurements retain their output/replicate
axes. An unconverged stationary state is not displayed as a checked solution.

## Prepared fits, parameters and uncertainty

Prediction work is explicit and reusable:

```julia
predicted = predictions(estimation, fitted.minimizer) # Explicit forward solves.
picture = plot_fit(estimation,predicted; parameter_coordinates=fitted.minimizer)
save_figure(picture,"fit.png")
```

The optional coordinates permit evaluation of declared noise callbacks without
another solve. Otherwise only fixed errors are displayed. Correlated Gaussian
error bars are marginal one-SD projections, not independent observations or a
joint confidence region. Contrast likelihoods and truncated-Gaussian location
scales are not silently turned into measurement SD bars. Prepared `bands` need
an explicit `band_label`; for count data they must already be in expected-count
units. A MAP-only figure needs no posterior draws or uncertainty exports.

For several measured profiles, supply `abscissa=measurement_coordinates` to
show one curve per saved measurement time instead of one time trajectory per
output. The coordinates must match the selected block's rows; data, SDs and the
full covariance ordering are transposed consistently. A callback observable
cannot reveal physical abscissae or units, so these remain explicit inputs.

`plot_parameters(estimation,samples;coordinates=:physical)` maps retained draws
and analytical priors consistently. `:log` means natural log of the numerical
physical values; supply dimensionally meaningful labels/reference units yourself.
Shared, experiment-local and nuisance coordinates retain their recorded order.
KDEs summarize the retained joint draws, not a fitted Gaussian. Their contours
use integrated grid mass; an explicit `raw_pairs` selection preserves thin curved
relationships that a KDE would blur. A displayed prior is never peak-scaled or
renormalized onto a cropped posterior-focused range.

For an accepted Laplace/Gauss–Newton result, diagonal densities are exact
pushforwards of its recorded joint Gaussian. Joint contours map its native
probability ellipses; under nonlinear coordinate changes these are generally
**not** highest-density contours in physical coordinates. MAP is the mapped
inference-density mode, not generally a physical posterior mean. A MAP result
alone shows only points and priors, not an invented uncertainty distribution.
Failed diagnostics remain visible; nonfinite transformations raise an error
instead of silently removing draws. Historical covariance conventions must be
rendered from their explicitly recorded coordinates, not relabelled as new results.

PCE analytical summaries contain mean and variance, not output quantiles. Select
`band=:none` or the explicitly labelled `band=:sd`, or first call `evaluate_pce`
and plot its prepared ensemble. Surrogate-validation failures remain visible.
Failed forward realizations require investigation; a success-conditioned summary
is not silently substituted for the original UQ distribution.

## Formatting and saving

The ordinary preset uses Matplotlib fonts and needs no external LaTeX. Set
`preset="publication"` before creating artists for real LaTeX; missing `latex`
or `dvipng` produces an error. `preset="preview"` is explicitly labelled as
mathtext preview. No preset adds figure titles or selects a GUI/headless backend.
For batch work, set `MPLBACKEND=Agg` before importing PythonPlot/Matplotlib.

```julia
picture = plot_distribution(solution; preset="publication",font_size=9.0,
    figsize=(7.0,3.2),xlabel=raw"$L$ ($\mu$m)",ylabel=raw"$n(L)$")
save_figure(picture,"distribution.png";dpi=300)
save_figure(picture,"distribution.pdf") # Explicit vector request.
```

`styles` supplies per-curve Matplotlib options, and `palette` supplies an explicit
colour sequence; `legend` accepts a Bool or legend
keyword dictionary, and `axis` supplies axis overrides. Fields accept shared
`norm`/colour limits and `colorbar=false` for a figure-level colour bar. Layout
uses Matplotlib's constrained layout by default. Existing axes preserve explicit
case-study panel spacing. `save_figure(...;width,height,font_size,layout)` applies
formatting before recomputing layout; width/height are inches. Changing DPI alone
preserves physical dimensions and fonts. Use `layout="tight"` or
`layout="constrained"` after custom legend/label changes as appropriate.
Styles are scoped, leaving unrelated Matplotlib state unchanged.
Exports check visible text for clipping and report an actionable layout error;
they do not silently reduce fonts. `check_layout=false` is an explicit override
for deliberately cropped artwork.

Python plotters import `pbm_plotting` with `<package>/python` on `PYTHONPATH`.
Its array-only functions are the **same** functions called by the Julia extension;
there is no second backend or universal result-file format. Lightweight local
readers retain each study's producer metadata. Redrawing validates recorded files,
not equivalence to current source. New numerical analyses retain their stricter
compatibility checks. Protein `paper/protein/plot.py --map-only` skips posterior
inputs; validated bands/corners remain available in the ordinary figure-only run.
Paper growth/precipitation `--tables` explicitly opts into derived analysis tables;
ordinary redraws only generate figures and plotting receipts.

See the [plotting API reference](api/plotting.md), the
[minimum-script checks](verification.md#Examples-and-figures), and
[optional integration tests](https://github.com/pavaninguva/PopulationBalanceModels.jl/blob/main/test/README.md) in the checkout.
