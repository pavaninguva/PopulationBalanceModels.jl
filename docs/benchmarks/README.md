# Running the case studies

From the package root, use these calculation/plot pairs:

```sh
julia --project=. examples/cooling_crystallization_benchmark.jl
python docs/benchmarks/plot_transport_comparisons.py cooling --usetex

julia --project=/path/to/working_environment examples/shin_lnp_reproduction.jl /path/to/shin_inputs.jl
python docs/benchmarks/plot_case_studies.py lnp

julia --project=. examples/inguva_braatz_case2.jl
python docs/benchmarks/plot_case_studies.py transport
```

NumPy and Matplotlib are plotting dependencies only. `--data-dir PATH` selects
an alternative exported-data directory. The plotter requires the appropriate
CSV files; it does not fabricate missing curves or substitute demonstration data.

The cooling benchmark chooses and records a nonnegative parabolic initial
condition and a saturated initial solution. The printed Eq.34 is retained
literally as `reported_initial_density`, not used as a density or silently
repaired. Upwind and MUSCL at 400–3200 nominal bins are compared with a
12800-bin MUSCL reference. A 6400-bin comparison and a time-step check quantify
reference error; independent continuum moment ODEs check its moments. An optional
input file can override `initial_density`, `c0`, `provenance`, `grids`,
`reference_bins`, `check_bins`, `dt`, and `tf`. The default is dt=0.1 s, tf=1000 s.
This is not a claim to implement the paper's HR1 or reproduce its original
initial-condition curves. Its published reference used 32000 bins and dt=0.01 s.

The Shin input file returns `inputs`, `solver`, and `provenance`, optionally
`saveat`. `inputs` includes the grid and numerical material closures; the Figure 4
adapter supplies zero initial particles, the 10 kg/m³ organic feed, FRR 3,
instantaneous mixing, and the 4320-minute horizon by default. Use an adaptive stiff
solver for that long horizon. The output schedule includes 1, 40, 100, 170, and
4320 minutes. Experimental data are not needed to render the model-only panels.

Inguva–Braatz Case 2 is self-contained: it supplies the separable affine growth
laws, Gaussian initial condition and analytical solution, and uses the public
InguvaBraatz2023 compiler with characteristic_grid. Uniform FV and nonuniform
mapped mesh differences are recorded in `errors.csv`.

New study results are written under `docs/src/assets/gunawan_example2/`,
`shin_case/`, and `inguva_case2/`. Rebuild the manual after rendering figures.
The docs build does not run these expensive simulations. When publication-input
results have not been generated, the page gives their commands. Separate
illustrations show explicitly labelled demonstration calculations.

`diagnostics/` retains earlier constant-growth and demonstration-initialization
data. The nanoparticle page retains labelled demonstration plots; the cooling page
uses the new reference-checked numerical benchmark. The constant-growth Gaussian is now a
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

The study evaluates the analytical density at 281 sizes and six times and adds
independent Gaussian noise with known standard deviation one. It fits only g0
and g1. Each of two prior scenarios (lognormal and physical uniform bounds)
uses three MAP starts and four chains with 1000 warmup and 1500 retained draws.
Sampling uses the analytical solution, not MUSCL at every leapfrog step.
The runner checks R-hat, bulk/tail ESS, divergences, tree depth and per-chain
BFMI before reporting posterior results. Raw draws preserve chain identifiers;
CSV reports include MCSE, energy, acceptance, step sizes and warmup statistics.
Six numerical refits compare Inguva2022 and sectional MUSCL. CFL=1 requires
refining the characteristic mesh and time step together. Failed stationarity
checks remain visible. Outputs and provenance are in docs/src/assets/bayesian_growth.

## Growth/time combinations and axial transport

```sh
julia --project=. examples/constant_growth_methods.jl
python docs/benchmarks/plot_transport_comparisons.py constant --usetex
julia --project=. examples/axially_dispersed_growth.jl
python docs/benchmarks/plot_transport_comparisons.py axial --usetex
```

The constant-growth comparison follows the 2022 Case 1 equation, initial
exponential, and t=4. It compares analytical bin integrals with several
reconstruction/integrator combinations at C=0.5 and C=1. Negative concentrations
are retained deliberately; unstable LinearUpwind/C=1 runs are reported rather
than interpreted as valid solutions. Timings are single warmed runs. The axial
example feeds a growing population into an initially empty dispersed tube; the
PDF-independent verification uses analytical periodic advection–diffusion and
nonuniform-grid inlet/outlet balances in `test/axial.jl`.
