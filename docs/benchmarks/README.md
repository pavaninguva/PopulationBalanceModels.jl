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
python docs/benchmarks/plot_moment_comparison.py --usetex
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

The moment comparison includes QMOM, rank-adaptive QMOM, Gamma EQMOM, DQMOM
and an assumed-lognormal closure. The CSVs record state counts, single-run
times, time-step checks, conditioning and the last completed time when a
method fails its state checks. Higher moments inferred from the lognormal
shape are marked separately from evolved moments. Reuse the committed
sectional reference and regenerate just this comparison with
`julia --project=. examples/qamar_moment_benchmarks.jl --methods-only`.

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

The constant-growth comparison uses eight representative pairings (including WENO3 + SSPRK33 and HOCFiniteVolume + midpoint) for the 2022
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

The precipitation default separates size and axial refinement with four points
per path. It compares Upwind, MUSCL and WENO5 in size with axial MUSCL/RK4 fixed,
against one uniform-size WENO5/axial-MUSCL/RK4 reference on 144×128 cells.
Fields save 0, 40, 80 and 200 s. Same-mesh startup field/outlet/bulk temporal
checks accompany independent time/size/axial reference checks
and logarithmic-size sensitivity runs are exported. WENO5 in both directions
still fails the empty-start bulk check; those failures are retained without clipping.
`--full` requests the SI Cartesian mesh arrangement with axial MUSCL fixed
(25 pairs per eligible size scheme/grid family), and a 250×750 reference.
It was not executed for the documentation. This is a numerical reference and
package comparison, not CADET's WENO23/WENO35 or timing reproduction.
Protein observations are
experimental markers digitized from Mitchell2023 Figure 4; the committed CSV and
extraction provenance are in `examples/data/protein_crystallization/`. The
protein plotter consumes the actual fitted predictions and does not invent
uncertainty intervals when the Laplace verification fails. Each case-study page ends with a short complete run and downloadable full
sources. The build copies sources into ignored download assets; it does not
embed export drivers or rerun expensive studies.

For the joint protein posterior, add AdvancedHMC 0.8 and MCMCDiagnosticTools 0.3
to the working environment, alongside Optim and the package:

```sh
julia --project=/path/to/working_environment examples/protein_crystallization.jl --posterior-parallel
python docs/benchmarks/plot_protein_posterior.py --usetex
```

The default broad priors expand logAJ/j/g support and double the log-growth
prior scale; `priors=:previous` is available for explicit sensitivity reruns.
Changing priors requires new posterior sampling; never relabel the previous draws.
`--posterior-only` reuses physical saved MAP values as starts, retransforming them
when bounds change. `--posterior` runs the MAP calculation
first. The posterior runner uses four chains, 750 warmup and 1,500 retained draws
per chain, base seed 2027 (isolated chain seeds 2027–2030), and 100 additional
predictive solves. The initial 750-draw broad-prior run failed the unchanged
convergence screen and is retained under `sampling_budget_750/`. The larger
budget uses isolated processes; `--posterior-only` keeps sequential execution. This repeatedly
solves the coupled PBM and can take tens of minutes on a CPU.
Rate-based sampling coordinates improve mixing without changing the
physical model, priors or posterior. Retained draws, prior densities, chain
statistics, standardized residuals and temporal-refinement checks are exported
to `protein_crystallization/`. The plotting script renders the corner, prior
comparison, chain/rank and predictive figures, and visibly marks samples as
provisional if the diagnostic screen fails. No posterior covariance is
manufactured from the rejected Laplace calculation.

The native dissolution verification and figure use:

```sh
julia --project=. examples/realizable_dissolution.jl
python docs/benchmarks/plot_dissolution.py --usetex
julia --project=. examples/realizable_startup.jl
julia --project=. examples/realizable_axial_transport.jl
```

These compare analytical uniform-population dissolution, empty point-source
startup and conservative nonuniform axial transport. Moment dissolution uses an
explicit atomic boundary approximation; rank and sectional-grid errors remain
separate from the discrete material balance error.

For plot-only validation, run `python test/plotting_summaries.py` in the plotting
environment. It checks enclosed-probability contours against a 2D Gaussian,
resolves a synthetic thin ridge, enforces bounded support, and checks Pareto
selection. A fifth test independently recomputes all 36 outlet errors through
cumulative bin integrals and verifies the exported volume normalization. The protein KDEs and rate bands reuse retained NUTS draws; they do
not rerun sampling or change the posterior target. `posterior_summaries.csv`
records recalculated correlations, quantiles and the saved MAP supersaturation
range. The LNP long-time outputs remain historical: the audited collision checks
currently reject an internal state in the optional stiff startup driver. A short
native startup is runnable without clipping, as documented on the case page.

`examples/protein_crystallization.jl --identifiability-only` reuses the saved MAP
to export the existing package's local prediction sensitivities, a noise-scaled
SVD and separate likelihood/prior curvature. It performs no fit or sampling.
These local Hessians are not verified posterior covariance estimates.

Executed checks, environment versions and historical results are consolidated
in the [audit register](../src/audit.md). Current case pages link their actual
CSV exports, settings and limitations. Broader-prior protein results require a
new fit and NUTS run; older draws must not be relabelled as the new posterior.
