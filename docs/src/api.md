# API reference

The constructors and lifecycle below connect the manual to the public interface.
Arguments in callback signatures are physical parameters `p`, time `t`, and
current stage context `ctx`. Read each method's restrictions in the linked manual.

## Model and solve

```julia
Grid1D(edges; coordinate=:volume)
PBMModel(; named_terms...)
TermMethods(; by_term=NamedTuple(), mechanism_methods...)
compile(model, grid; methods=TermMethods(), bc=SizeBC(), growth_scheme=Upwind())
compile(model, MomentSet(0:K); method=ExactClosure(), methods=TermMethods(),
        boundary=MomentBoundary())
characteristic_grid(AffineGrowth(:g0,:g1),(0.0,1.4);dt,p)
compile(model, grid; evolution=Inguva2022(),p,bc)
PBMProblem(operator, initial, (t0,tf); p=NamedTuple())
simulate(problem; dt, saveat=[t0,tf], method=SSPRK33())
compile(model, grid; evolution=LaxWendroff(),bc)
compile(model, grid; domain=AxialDispersion(AxialGrid(z_edges);
    velocity=u,dispersion=D,inlet=(p,t)->feed_bins))
```

The displayed calls summarize valid argument shapes, rather than forming one
standalone program. A [complete first model](getting_started.md) is executed
in the documentation build. `solution.t` and `solution.u` hold saved times and
states; standalone 1D sectional and moment states have one column. Use
`density(grid,N)`, `moment(grid,N,k)`, and `term_methods(operator)` to interpret
what was stored and how it was computed.

Residual-based time methods are `ForwardEuler`, `SSPRK22`, `SSPRK33`, `SSPRK54`, and `RK4`.
`VanLeer` and geometry-aware `WENO3` support nonuniform 1D grids.
`Centered` and `LinearUpwind` add unlimited uniform-grid 1D reconstructions.
`LaxWendroff` and `BeamWarming` are complete constant-growth space–time updates;
they require an eligible model compiled with `evolution`.
For axial states, `axial_population(operator,u,cell)` restores a local intrinsic
view and `axial_fluxes(operator,u,p,t)` returns physical face fluxes. See
[Axially dispersed reactors](axial_reactors.md) for units and boundary contracts.

`RankAdaptiveQMOM(N)` evolves 2N raw moments with a checked variable-rank
quadrature. `EQMOM(N)` evolves 2N+1 raw moments with a Gamma-kernel
reconstruction; `reconstructed_density` and `reconstructed_bins` make that
assumption explicit. `DQMOM(N)` evolves positive weights and weighted abscissas;
initialize it with `quadrature_state(method,nodes,weights)`. Read
[Moment methods](moment_methods.md) for rank, conditioning and boundary limits.

## Couple and observe

`PBMSystem(populations; bulk_rhs, interactions, exchanges, fields, nonnegative_bulk, accounting)`
accepts a named tuple of compiled populations. Its problem adds named initial
populations and `bulk0`; use `stateview(problem,u)` to obtain named views.
Use `population(problem,u,name)`, `bulk(problem,u,name)` and
`cumulative(problem,u,population,mechanism,property=:number)` for packed states.
`accounting_rates(problem,u,p,t)` returns a named tree of instantaneous rates.
See [Coupled systems](coupled_systems.md) for callback and exchange contracts.
`axial_problem(local_problem,reactor;axial_scheme=...)` adds axial transport to
local sectional/raw-moment systems, including every scalar bulk state.
`axial_stateview(problem,u,cell)` exposes a named local state; population and
bulk accessors return axial arrays. Direct moment states are rejected.

`Observation(times,data,observable; sigma)` or a fixed `covariance` block becomes part of
`Experiment(problem,observations...)` and then
`CalibrationProblem(experiments; parameters,dt,solver)`. `residuals`, `objective`,
and `sensitivity_report` use this shared forward model.
`fit(calibration,z0; ...)` needs Optim; `odeproblem(problem; ...)` needs SciMLBase.
`predictions` returns the observation matrices. `fit_diagnostics` assesses rank,
residuals, and numerical sensitivity; `confidence_ellipsoid` and `ellipse_points`
construct local joint regions. `fit_multistart` and `profile_likelihood` use the
fitting backend for multiple starts and nuisance-parameter refits.
See [Parameter estimation](inference.md) for assumptions and examples.

New Bayesian calculations use `ParameterSpec`, explicitly coordinated `Prior`
declarations, `Measurement`, `InferenceExperiment`, and `EstimationProblem`.
`fit(problem)` performs MAP using Optim. `loglikelihood`, `logprior`, and
`logposterior` share the density used by `laplace_approximation` and
`sample_posterior`. Sampling requires both AdvancedHMC and MCMCDiagnosticTools.
`parameter_draws` transforms draws, and `posterior_predictive` requires an explicit
forward-solve budget. The Bayesian and legacy frequentist uncertainty APIs have
different interpretations; see the estimation chapter.
`diagnostics(samples)` exposes per-chain sampling reports, including BFMI,
acceptance, step sizes, warmup statistics and Monte Carlo errors, alongside
rank R-hat and bulk/tail ESS.

`posterior_gradient` and `likelihood_gradient` default to ForwardDiff.
`check_gradient` compares against independent finite differences. Both inference
problem constructors accept `derivatives=:finite_difference` when explicitly
needed. `SciMLSensitivitySolver(algorithm;sensitivity=:auto)` enables optional
continuous forward/adjoint ODE sensitivities; see the estimation chapter for
dependencies and compatibility. `TruncatedGaussianLikelihood(sigma=...)`
provides normalized independent nonnegative measurements.

## Numerical extensions

A sectional policy supplies `prepare_term(term,method,representation; kwargs...)`
and `add_term!(du,u,term,method,cache,operator,p,t,ctx)`.
A moment family additionally supplies state mapping, stage validation,
reconstruction, and weak-form accumulation. See [Development](development.md)
for full contracts and a working delegation example.

## Exported docstrings

```@autodocs
Modules = [PopulationBalanceModels]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
