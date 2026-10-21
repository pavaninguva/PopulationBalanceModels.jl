# Running the case studies

From the package root, use these calculation/plot pairs:

```sh
julia --project=. examples/cooling_crystallization_benchmark.jl /path/to/gunawan_inputs.jl
python docs/benchmarks/plot_case_studies.py cooling

julia --project=/path/to/working_environment examples/shin_lnp_reproduction.jl /path/to/shin_inputs.jl
python docs/benchmarks/plot_case_studies.py lnp

julia --project=. examples/inguva_braatz_case2.jl
python docs/benchmarks/plot_case_studies.py transport
```

NumPy and Matplotlib are plotting dependencies only. `--data-dir PATH` selects
an alternative exported-data directory. The plotter requires the appropriate
CSV files; it does not fabricate missing curves or substitute demonstration data.

The Gunawan input file returns `initial_density`, `c0`, and `provenance`; it can
also set `grids`, `dt`, and `initial_moments`. The literal Eq.34 polynomial is
retained but rejected as negative on its support. Resolve that discrepancy before
claiming published-figure reproduction. The moderate published spacing is 0.5 µm
(3200 bins) and the fine reference spacing is 0.05 µm (32000 bins), both with a
0.01-second time step. Current comparisons use the package's Upwind/MUSCL,
not the paper's HR1/Lax–Wendroff algorithms.

The Shin input file returns `inputs`, `solver`, and `provenance`, optionally
`saveat`. `inputs` includes the grid and numerical material closures; the Figure 4
adapter supplies zero initial particles, the 10 kg/m³ organic feed, FRR 3,
instantaneous mixing, and the 4320-minute horizon by default. Use an adaptive stiff
solver for that long horizon. The output schedule includes 1, 40, 100, 170, and
4320 minutes. Experimental data are not needed to render the model-only panels.

Inguva–Braatz Case 2 is self-contained: it supplies the separable affine growth
laws, Gaussian initial condition, analytical solution, and a case-specific
travel-time transform for exact bin shifts. Uniform FV and nonuniform mapped
mesh differences are recorded in `errors.csv`.

New study results are written under `docs/src/assets/gunawan_example2/`,
`shin_case/`, and `inguva_case2/`. Rebuild the manual after rendering figures.
The docs build does not run these expensive simulations. When publication-input
results have not been generated, the page gives their commands. Separate
illustrations show explicitly labelled demonstration calculations.

`diagnostics/` retains earlier constant-growth and demonstration-initialization
data. The cooling and nanoparticle pages display labelled plots of those
demonstrations, with their provenance. The constant-growth Gaussian is now a
regression fixture in `test/fixtures/`. The original periodic rectangle/smooth
benchmark and compact/ADI stability audit remain numerical-method diagnostics,
not additional case-study pages.

## Typesetting and additional figures

`plot_case_studies.py` applies serif typography, LaTeX mathematical labels, and
no grid lines. `--usetex` uses an installed LaTeX distribution (including amsmath);
omitting it uses Matplotlib's Computer Modern MathText. The committed case-study
figures were rendered with LaTeX. PNG and compact SVG exports are retained.

```sh
python docs/benchmarks/plot_case_studies.py transport --usetex
python docs/benchmarks/plot_case_studies.py illustrations --usetex
julia --project=. examples/temperature_programmed_crystallization.jl
python docs/benchmarks/plot_case_studies.py moments --usetex
julia --project=/path/to/working_environment examples/kinetic_parameter_uncertainty.jl
python docs/benchmarks/plot_case_studies.py estimation --usetex
```

`illustrations` re-renders the retained cooling and LNP demonstration data into
`docs/src/assets/illustrations/`. Their input provenance is in the diagnostic
folders. They do not replace the unresolved publication inputs or represent
published-figure reproductions. The moment calculation records its reconstructed
monotonic protocol and material assumptions in `moment_crystallization/inputs.txt`;
it does not implement the extinction/distribution parts of Figure 13.

The older estimation example uses seeded synthetic Gaussian data. Its raw exports
include whitened residuals, mapped joint-region points, a nuisance-reoptimized
profile, and numerical diagnostics. The output is a local uncertainty example,
not a coverage validation for physical models.

## Bayesian growth inference

Use a working environment with Optim, Distributions, AdvancedHMC 0.8 and
MCMCDiagnosticTools 0.3. The new manual example replaces the older trivial
kinetic fit, whose code remains available for regression purposes.

```sh
julia --project=/path/to/working_environment examples/estimate_size_dependent_growth.jl
python docs/benchmarks/plot_bayesian_growth.py --usetex
```

The default budget is three seeded MAP starts, four chains of 500 warmup and
1000 retained NUTS draws, forty posterior predictive solves, two profiles, an
alternative-prior fit and four numerical refits. Sampling uses the inexpensive
case-specific characteristic solution, not a sectional solve at every leapfrog
stage. Its reported forward count includes those analytical trajectories.
The numerical refits use sectional MUSCL. CSVs retain convergence flags; failed
strict stationarity checks must remain visible in plots and interpretation.
Outputs, raw draws and provenance are in docs/src/assets/bayesian_growth.
