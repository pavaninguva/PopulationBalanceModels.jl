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
julia --project=. examples/qamar_moment_benchmarks.jl
python docs/benchmarks/plot_case_studies.py moments --usetex
julia --project=/path/to/working_environment examples/kinetic_parameter_uncertainty.jl
python docs/benchmarks/plot_case_studies.py estimation --usetex
```

`illustrations` re-renders the retained cooling and LNP demonstration data into
`docs/src/assets/illustrations/`. Their input provenance is in the diagnostic
folders. They do not replace the unresolved publication inputs or represent
published-figure reproductions. The active moment case uses Qamar problems 4 and 5 and records analytical
references, quadrature/time refinement and a sectional reference in
`qamar_moments/`. The previous temperature-programmed recipe remains runnable
for compatibility; it is no longer the active manual case study.

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

The study evaluates the analytical density at 281 sizes and six times and draws
independent Gaussian measurements conditioned to be nonnegative, with underlying
standard deviation one. Its likelihood includes that truncation; no values are
clipped. Normal quantile residuals account for the asymmetric errors near zero.
It fits only g0
and g1. Each of two prior scenarios (lognormal and physical uniform bounds)
uses three MAP starts and four chains with 1000 warmup and 1500 retained draws.
Sampling uses ForwardDiff through the analytical solution, not MUSCL at every
leapfrog step. Finite differences independently verify gradients. The mesh-changing
Inguva refits explicitly use finite differences; fixed-grid MUSCL refits use AD.
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

The constant-growth comparison uses six representative pairings for the 2022
Case 1 front and a smooth periodic input. Each configuration has one measured
solve following a small compilation warmup. L1/RMSE panels and measured
refinement orders replace arbitrary h-power guides and repeated-run bands.
Inguva2022 appears in the timing and error plots at its required C=1, labelled
separately from the other methods at C=.5. Run `examples/audit_lax_wendroff.jl`
to regenerate the coefficient, boundary and final-time comparison with the
PBM_Schemes reference. The reference also oscillates at C=.5; whole-cell C=1
shifts remove that error. The axial example retains analytical periodic
advection–diffusion and nonuniform inlet/outlet tests in `test/axial.jl`.

All supplied growth and time methods remain available in the package;
the case-study figures deliberately show only the representative subset.
HOCADI and InguvaBraatz2023 belong to the separate two-coordinate studies.

For the additional moment and reactor calculations:

```sh
julia --project=. examples/quadrature_moment_methods.jl
python docs/benchmarks/plot_new_case_studies.py moments --usetex

julia --project=. examples/continuous_precipitation.jl
python docs/benchmarks/plot_new_case_studies.py precipitation --usetex

# Use an environment containing PopulationBalanceModels and optional Optim:
julia --project=/path/to/working_environment examples/protein_crystallization.jl
python docs/benchmarks/plot_new_case_studies.py protein --usetex
```

The precipitation default uses a small documentation budget for Zhang2024 Case
7A. `--full` requests expensive SI-scale mesh refinement. Its reference remains
numerical, and its methods/timings are not CADET's. Protein observations are
experimental markers digitized from Mitchell2023 Figure 4; the committed CSV and
extraction provenance are in `examples/data/protein_crystallization/`. The
protein plotter consumes the actual fitted predictions and does not invent
uncertainty intervals when the Laplace verification fails. Every case-study
page embeds its full Julia source during the documentation build.
