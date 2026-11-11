# Verification

Run from the repository root. The core suite includes Aqua and independent
analytical, conservation, refinement, admissibility and failure checks:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Files are grouped by behaviour, rather than by development milestone:

| Area | Main files |
|:--|:--|
| Physical terms and coupling | `sources_moments.jl`, `systems.jl`, `papers.jl`, `size_diffusion.jl`, `tensor_diffusion.jl` |
| Transport and time methods | `transport_integrators.jl`, `nonuniform_weno.jl`, `compact_finite_volume.jl`, `tensor.jl` |
| Run preparation, physical initialization and execution | `forward_execution.jl`; optional checked ODE execution: `forward_execution_integration.jl` |
| Moment representations and accounting | `moment_compilation.jl`, `extended_moments.jl`, `measure_evolution.jl`, `qamar_moments.jl` |
| Axial and stationary balances | `axial.jl`, `axial_inlets.jl`, `axial_system.jl`, `stationary.jl` |
| Calibration, Bayesian targets and derivatives | `uncertainty.jl`, `bayesian.jl`, `physical_parameters.jl`, `differentiation.jl`, `isothermal_protein.jl` |
| Data-free forward UQ and milling physics | `forward_uq.jl`, `pce_construction.jl`, `milling_model.jl`; optional BDF adapter: `milling_integration.jl` |
| Plot extraction and optional rendering | `plotting.jl`; PythonPlot: `plotting_integration.jl`; Python: `shared_plotting.py` |
| Scientific regression cases | `audit_regressions.jl`, `numerical_contracts.jl`, `stem_cells.jl`, `li_supplement.jl` |

Documentation CI runs `julia --project=docs test/docs_reference.jl`, the strict
Documenter build and `python test/docs_links.py docs/build`. Source-routing tests
cover new subsystem files, overlap precedence, unknown sources and empty groups;
new API documentation belongs in topic reference pages, not in walkthroughs.

`fixtures/` holds independent references and reusable setup.
`fixtures/example_modules.jl` loads each example module once per enclosing test
module, so the same builders can be used in several files without replacing
modules and repeating compilation. It also supports standalone test-file runs.
The cleanup retains the numerical assertions and their independent references.

For an interface-only example check, run `julia --project=. test/example_interfaces.jl`.
It constructs problems, checks priors/state layouts and evaluates selected initial
residuals; it does not fit, sample, advance a case study or write results. In an
environment with Optim and NonlinearSolve it also constructs the stationary
estimation example. Long benchmark/reproduction commands are separate.

`julia --project=docs test/case_study_scripts.jl` checks all nine authored
walkthroughs, verbatim display/download copies, portable relative companions,
physical inputs, initial integrals and representative rates against maintained
examples. Protein additionally checks calibration/test exclusion, normalized
CDF likelihood equivalence (including the known whitening determinant), priors
and posterior gradients at fixed parameters. No fit or posterior chain runs.

`julia --project=docs test/case_study_migration.jl` includes those checks and
executes short numerical paths with explicitly reduced settings. It verifies
independent initial/CDF/pivot-hat integrals, population/bulk/material balances,
unchanged precipitation controls and the stem population with/without diagnostic
counters. In an environment also containing Sundials and SciMLBase, it executes
one nominal milling solve and the separately invoked MC/PCE stage (16 MC and
12 training solves; existing MC supplies validation). No production ensemble,
benchmark export or publication figure is regenerated. PythonPlot recipes are
checked separately by `test/plotting_integration.jl`. The legacy stem producer's
12-node rule remains separately tested by `stem_cells.jl`.

## Optional integrations

Use an isolated working environment containing this checkout and the dependencies
listed in [CI](../.github/workflows/CI.yml). Then run the corresponding entry points:

```sh
julia --project=/path/to/optional-env test/integration.jl
julia --project=/path/to/optional-env test/bayesian_integration.jl
julia --project=/path/to/optional-env test/protein_parallel.jl
julia --project=/path/to/optional-env test/stationary_integration.jl
julia --project=/path/to/optional-env test/stationary_inference.jl
julia --project=/path/to/optional-env test/tensor_integration.jl
julia --project=/path/to/sensitivity-env test/sensitivity_integration.jl
```

The inference tests use small analytical targets; they do not regenerate the
long protein posterior. `physical_parameters.jl` and its optional integration
companion verify explicit transforms, physical priors, z-density modes/curvature,
joint nuisance covariance, mapped Gaussian draws and convention guards.
`sampling_control.jl`, included by `bayesian_integration.jl`, checks diagnostic stopping against conjugate Gaussian
and transformed lognormal posteriors, exact continuation against upstream
AdvancedHMC, independent batch sizes, MCSE requirements, budgets and failure
propagation. `protein_parallel.jl` checks joint reassessment and retained-output
protection without running protein calibration or sampling. Minimum/current Julia jobs, normal compiled-module
sensitivity checks, and selective JET analysis remain in CI.

The minimum Julia 1.10.0 Linux job uses Ubuntu 24.04: its bundled libopenlibm
requests an executable stack that newer glibc releases reject before tests
start ([upstream issue](https://github.com/JuliaLang/julia/issues/57250)). The
other Linux core jobs keep `ubuntu-latest`; all six core combinations remain.

Sensitivity CI resolves and precompiles its dependencies in one process, then
runs the tests in a fresh process with the same project. Its cache keeps package
sources and artifacts, but excludes compiled images; normal Julia precompilation
remains enabled. This job uses `--cpu-target=generic` for preparation and tests,
and `JULIA_CPU_TARGET=generic` for package images. This avoids the observed
Julia 1.13.3/LLVM 20 `VPScalarPHIRecipe` compilation crash on Ice Lake when
differentiating parameter-dependent initialization. Optimization stays at its
normal level; no tests, derivative paths or tolerances are disabled. The core
and other extension jobs retain native compilation coverage. This is a CI
compiler workaround, not a change to initialization or sensitivity algorithms.
Each job prints Julia/dependency versions and retains its
`Project.toml` and `Manifest.toml`, including on failure. This distinguishes
environment/precompilation failures from failed sensitivity assertions.

To reproduce that separation from the repository root:

```sh
PBM_SENSITIVITY_ENV=/tmp/pbm-sensitivity-env
JULIA_CPU_TARGET=generic JULIA_NUM_PRECOMPILE_TASKS=1 julia --cpu-target=generic --project="$PBM_SENSITIVITY_ENV" -e 'import Pkg; Pkg.develop(path=pwd()); Pkg.add(["Optim", "SciMLBase", "SciMLSensitivity", "ReverseDiff", "OrdinaryDiffEqTsit5", "Distributions", "ForwardDiff"]); Pkg.status(;mode=Pkg.PKGMODE_MANIFEST); Pkg.precompile(;strict=true)'
JULIA_CPU_TARGET=generic julia --cpu-target=generic --project="$PBM_SENSITIVITY_ENV" test/sensitivity_integration.jl
```

Use selective imports in shared test processes. For example, importing all of
Distributions also exports `moment`, which conflicts with the package's
population-moment function. The complete optional-CI sequence is checked together,
as well as through its standalone entry points.

## Retained results and utilities

For optional plotting checks, develop this checkout into an isolated environment
containing `PythonPlot`, `Distributions`, `ForwardDiff`, `SpecialFunctions` and
`Optim`; add `SciMLBase`/`Sundials` for the small milling walkthrough. PythonPlot
normally installs its own Python/Matplotlib. To reuse the paper's pinned Python,
set `JULIA_PYTHONCALL_EXE` to that environment's Python and
`JULIA_CONDAPKG_BACKEND=Null` **before** starting Julia. The repository Python
plotters require Python 3.11 or newer.

```sh
MPLBACKEND=Agg julia --project=/path/to/plotting-env test/plotting_integration.jl
MPLBACKEND=Agg python test/shared_plotting.py
MPLBACKEND=Agg python test/render_plotting_fixtures.py --output /tmp/pbm-plot-checks
```

The last command renders retained growth/protein/milling values and a small field
fixture with real LaTeX. Its required output directory prevents accidental writes
to retained assets. Numerical-only extraction tests are part of `Pkg.test()`.
The optional tests exercise callable/tutorial wiring with small solves, not
production fits, MCMC or ensembles. Publication rendering needs `latex`/`dvipng`;
ordinary plotting and simulation do not. API routing/link checks remain separate.

```sh
python tools/retained_data.py fetch --profile all
python test/protein_results.py --retained
python test/paper_data.py --case 1 --retained
python test/paper_data.py --case 2 --retained
python test/plotting_summaries.py
python test/stem_cell_results.py
python test/docs_links.py docs/build
julia --project=paper test/tools/paper.jl
julia --project=. test/tools/run_examples.jl
julia --project=. test/tools/environment.jl
julia --project=. test/tools/profile_paths.jl working_tree
```

Use the plotting environment in `paper/requirements.txt` for the Python checks.
`--retained` checks stored data, normalizations, balances and posterior diagnostics;
it explicitly does **not** assert that today's checkout is their producer.
Without it, other benchmarks require an exact source match; protein also accepts
the separately verified compatibility receipt described below. A source mismatch
is not a reason to overwrite posterior draws or their provenance. The longer standalone
`audit_protein.jl` and `audit_qamar_diagnosis.jl` retain their historical numerical
questions and are not run as part of a directory-cleanup smoke check.

`tools/` contains executable reporting utilities, not extra core test suites.
The active logarithmic-grid tubular dataset includes its same-mesh
time-refinement run; check it with `python test/paper_data.py --case 3 --retained`.
The superseded 400 × 400 study was incomplete and is preserved as such in the
external recovery archive described in the
[archive catalogue](fixtures/archives/README.md#relocated-legacy-outputs).
Superseded manual figures and older protein draws are preserved in
[`fixtures/archives/`](fixtures/archives/README.md); its README explains extraction
for the optional historical protein audit. Current protein samples remain in the
paper production results and are not archived or regenerated by these checks.
Performance probes report warmed allocations/work separately from the paper's
single-solve timing protocol. Build documentation before checking local links.

Publication analysis adds focused, inexpensive contracts (no protein resampling):

```sh
julia --project=paper test/protein_analysis_modes.jl
python test/protein_publication.py
python test/protein_results.py --retained
```

The active protein case fixes gamma and compares CNT, constant-D0 GRD and
growth-dependent GRD (three/four inferred coordinates). Its producer and
posterior checks are separate from the
archived four/five-coordinate targets. `protein_publication.py` verifies and
extracts the historical uncertainty receipts into a temporary directory for
its Laplace/Gauss–Newton artifact tests; this does not assert equivalence of
those targets to the active model. Analytical legacy-coordinate tests use a
small parameter-observation fixture, not a second case-specific PBM engine.
`protein_analysis_modes.jl` also rebuilds the saved draw arrays by chain/draw
identity and independently reassesses rank R-hat, bulk/tail ESS and BFMI.

The protein check can accept a separately recorded compatibility receipt after
an explicit source review and new posterior/gradient evaluations. It still
validates the original producer manifest and fails on changes since that
receipt. Other benchmarks retain their strict source matching. Read-only
analysis-mode tests cover invalid flag combinations, changed mesh settings and
preservation of every retained production file. Transformation tests use
independent Normal/lognormal/logit-normal integrals and unfiltered support checks.

Sensitivity-based Bayesian covariance has independent analytical tests in
`gauss_newton.jl` (included in the core suite). They cover correlated Gaussian
and shared/local parameters, prior Jacobians, joint nuisance marginalization,
parameter-unit rescaling, rank/conditioning failures, support reporting and a
native PBM derivative. `gauss_newton_optim.jl` is included by the optional Bayesian
integration suite and verifies opt-in `fit(...;uncertainty=:gauss_newton)` without
changing the fitted target. Run these alone in an Optim-enabled environment:

```sh
julia --project=paper -e 'using Test, PopulationBalanceModels, LinearAlgebra; include("test/gauss_newton.jl"); include("test/gauss_newton_optim.jl")'
```
