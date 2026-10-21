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
compile(model, MomentSet(0:K); method=ExactClosure(), methods=TermMethods())
PBMProblem(operator, initial, (t0,tf); p=NamedTuple())
simulate(problem; dt, saveat=[t0,tf], method=SSPRK33())
```

The displayed calls summarize valid argument shapes, rather than forming one
standalone program. A [complete first model](getting_started.md) is executed
in the documentation build. `solution.t` and `solution.u` hold saved times and
states; standalone 1D sectional and moment states have one column. Use
`density(grid,N)`, `moment(grid,N,k)`, and `term_methods(operator)` to interpret
what was stored and how it was computed.

## Couple and observe

`PBMSystem(populations; bulk_rhs, interactions, exchanges, fields, nonnegative_bulk)`
accepts a named tuple of compiled populations. Its problem adds named initial
populations and `bulk0`; use `stateview(problem,u)` to obtain named views.
See [Coupled systems](coupled_systems.md) for callback and exchange contracts.

`Observation(times,data,observable; sigma)` or a fixed `covariance` block becomes part of
`Experiment(problem,observations...)` and then
`CalibrationProblem(experiments; parameters,dt,solver)`. `residuals`, `objective`,
and `sensitivity_report` use this shared forward model.
`fit(calibration,z0; ...)` needs Optim; `odeproblem(problem; ...)` needs SciMLBase.
`predictions` returns the observation matrices. `fit_diagnostics` assesses rank,
residuals, and numerical sensitivity; `confidence_ellipsoid` and `ellipse_points`
construct local joint regions. `fit_multistart` and `profile_likelihood` use the
fitting backend for multiple starts and nuisance-parameter refits.
See [Observations and estimation](inference.md) for assumptions and examples.

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
