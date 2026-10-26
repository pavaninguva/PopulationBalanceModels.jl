# Contributing, internals, and extensions

## Source organization

The module is assembled in `src/PopulationBalanceModels.jl`. `modeling/` holds
physical terms, laws, sources, and boundaries; `representations/` defines grids
and moment bases. `numerics/` contains selectable algorithms, and `compilation/`
validates combinations and creates prepared terms. `coupling/` packs populations
and bulk balances; `time_integration/` advances them; `inference/` maps results
to observations and estimation. `stationary/` assembles stationary problems and
parameter paths from the same residuals; its optional NonlinearSolve adapter
lives alongside other integrations in `ext/`.

This separation lets physical laws survive changes of representation and
numerical method. `core/interfaces.jl` defines public dispatch contracts and
method selection. `BatchDomain` and `AxialDispersion` implement well-mixed and
prescribed axial advection–dispersion domains. The axial compiler wraps local
sectional residuals and adds shared physical face fluxes; it does not treat
position as another intrinsic coordinate. General CFD remains outside the scope.

## Add a numerical algorithm

The public sectional extension contract is:

1. Define a subtype of AbstractTermMethod.
2. Extend prepare_term(term,method,representation;...) to validate applicability
   and construct reusable coefficients, quadrature or deposition caches.
3. Extend add_term!(du,u,term,method,cache,operator,p,t,ctx) to **add** its residual.
   Sectional methods return unresolved birth number/property rates for diagnostics.
4. Test the method's actual conservation, positivity assumptions, boundary
   behavior, accuracy and refinement. Implementing an interface is not validation.

Here is a complete extension skeleton that delegates to the existing algorithm:

```julia
using PopulationBalanceModels
struct MyAggregationMethod <: AbstractTermMethod end

PopulationBalanceModels.prepare_term(term::Aggregation,::MyAggregationMethod,grid::Grid1D;kwargs...) =
    prepare_term(term,FixedPivot(),grid;kwargs...)

PopulationBalanceModels.add_term!(du,u,term::Aggregation,::MyAggregationMethod,cache,op,p,t,ctx) =
    add_term!(du,u,term,FixedPivot(),cache,op,p,t,ctx)
```

Replacing those preparation/residual calculations with the new algorithm
does not require editing a central list of aggregation method names. A new
physical term can likewise subtype AbstractPBMTerm, declare mechanism and
provide compatible preparation/residual methods. Common model validation
does not hard-code a whitelist of user term types.

CustomTerm remains a convenient model-specific escape hatch. A reusable
numerical method should use the explicit policy contract so its identity and
cache are inspectable and selectable. ConservativeExchange and diagnostics
call the **selected** residual, including external methods.

## Extend moment evolution

The type hierarchy exposes AbstractMomentMethod, with AbstractMomentClosure
for raw-moment states and AbstractDirectMomentMethod for methods with other
states. Numerical weak-form evaluation remains a separate per-term policy:

```julia
op = compile(model,MomentSet(0:3);method=QMOM(2),
    methods=TermMethods(growth=ExactMomentRule(),
                        aggregation=QuadratureMomentRule()))
```

Thus analytic affine-growth moments can coexist with quadrature-based
collision integration. A global moment reconstruction is shared within one
residual evaluation rather than inverted independently for every term.
`examples/moment_closures.jl` exercises exact, quadrature and lognormal closure
selection; `examples/quadrature_moment_methods.jl` exercises the extended families.

A new moment family supplies:

| Public hook | Responsibility |
|---|---|
| validate_moment_method | Check the supported physical laws, moment basis and family options |
| moment_state_length | Define the packed state layout; it need not equal the number of tracked moments |
| moment_value | Convert a state into an experimental or coupled-model moment |
| moment_rate | Apply the chain rule from the method's derivative to a moment rate |
| validate_moment_state / check_moment_state | Family-specific initial/stage validity |
| prepare_moment_state | Reconstruct the density/quadrature/closure information needed at the current stage |
| prepare_term / moment_term! | Prepare each selected weak-form rule and add its contribution in that family's state variables |

The concrete signatures are:

```julia
validate_moment_method(method, model, basis)
moment_state_length(method, basis)
moment_value(method, basis, state, k)
moment_rate(method, basis, state, derivative, k)
validate_moment_state(method, basis, state)
check_moment_state(method, basis, state, tolerance)
prepare_moment_state(method, basis, state, p, t, ctx)
moment_term!(du, u, term, rule, cache, method, reconstructed, op, p, t, ctx)
```

`moment_term!` adds a contribution rather than resetting the derivative. The
compiled moment residual prepares reconstruction once per evaluation. The
external examples in `test/method_interfaces.jl` show both term dispatch and
a direct-state family with explicit moment chain rules.

Underscored helpers such as `_prepare_checked_moment_state`,
`_gamma_primary_precise` and `_ledger_index` are private. Third-party families
should implement the public hooks above; native accepted states are checked
through `validate_moment_state` as well as the necessary stage checks. Rate and
reconstruction callbacks must be pure: rejected trials must leave no state,
ledger or externally visible cache changes.

The moment_rate hook lets MomentExchange remain correct for nonlinear
weight/node-to-moment maps. Direct-state families must implement their
evolution explicitly; the raw-moment residual is never silently applied to
node variables. The external one-node growth test verifies a state layout
different from the moment basis, shared bulk exchange and both moment
observation adapters. The supplied `DQMOM` implementation uses these same
hooks for a scaled weight/weighted-abscissa source solve; its conditioning and
positive-weight restrictions are described in [Moment methods](moment_methods.md).

MomentSet currently holds contiguous nonnegative integer orders in one
coordinate. AbstractMomentBasis leaves room for sparse, fractional or
multivariate moment sets; those require additional concrete representations
and compilation rules. A conditional algorithm cannot be enabled merely by
renaming a one-coordinate QMOM closure.

Use `moment(ctx,k)` and `moment(ctx.system,:population,k)` in physical laws
when portability across representations matters. Directly indexing ctx.N
couples that law to a specific state layout. Observation and bulk-exchange
adapters use the method's moment mapping.



## Verification and documentation changes

Run the core scientific suite with `julia --project=. -e 'using Pkg; Pkg.test()'`.
The suite checks event factors, additive-property conservation, source integrals,
boundaries, signed growth, moment weak forms, coupling, inference, and restricted
2D refinement. The external method tests exercise dispatch and direct-state
moment chain rules. Passing these tests establishes those configurations,
not universal positivity or physical validation.

Use a separate working environment for optional integrations:

```julia
using Pkg
root = pwd() # package directory
Pkg.activate(mktempdir())
Pkg.develop(path=root)
Pkg.add(["Optim", "SciMLBase", "OrdinaryDiffEqTsit5", "Distributions", "ForwardDiff", "AdvancedHMC", "MCMCDiagnosticTools"])
include(joinpath(root,"test","integration.jl"))
include(joinpath(root,"test","bayesian_integration.jl"))
```

`docs/src/` is the authoritative manual; `docs/make.jl` never copies development
reports into it. Add focused explanations with equations, units, assumptions,
restrictions, and a runnable implementation. Update the relevant case study
when modifying an example. Keep benchmark inputs, raw CSV data, and environment
metadata with any new figures. Label historical evidence and synthetic inputs.

Core tests include [Aqua](https://juliatesting.github.io/Aqua.jl/stable/) checks
for ambiguities, exports, dependency metadata, piracy and persistent tasks.
Aqua is a test dependency. The separate JET job targets two concrete residuals
on Julia 1.13 with JET 0.12; it does not certify every callback or extension.
`test/profile_paths.jl` measures warmed residual allocations and work without
adding a runtime profiling dependency.

The sensitivity job uses normal compiled modules on maintained Julia 1.10 and
current Julia. The audited older Julia 1.10.3 sensitivity stack failed upstream
Enzyme/GPUCompiler precompilation. Its core package checks pass; optional
continuous sensitivities were verified with normal loading on 1.10.12. Use a
maintained Julia patch for that stack rather than relying on disabled compiled
modules. See the [audit register](audit.md) for exact environments and outcomes.

Both Distributions and this package export `moment`. Use selective imports
(`using Distributions: Normal, logpdf`, for example), or explicitly qualify
`PopulationBalanceModels.moment` when using both packages. The PBM operation
integrates raw powers of particle size; StatsBase's array operation computes a
sample central moment. Their statistical meanings remain distinct.

Build with the commands in the README. The build executes the model tutorials and selected case-study code blocks,
checks references, and treats warnings as failures. The [publishing guide](publishing.md) describes deployment.


## Updating this checkout

Edits made in this working directory are ordinary local Git changes. Review
`git status --short` and `git diff` before committing. Run the package tests and
build the manual, then commit the source, tests, documentation, and benchmark
evidence together. No archive extraction or replacement of the `.git` directory
is needed.

```sh
git diff --stat
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
git add src examples test docs CHANGELOG.md
git diff --cached --stat
git commit -m "Clarify model tutorials and numerical interfaces"
git push origin main
```

Use the appropriate branch instead of `main` when working through a pull request.
A normal push to `main` triggers the existing documentation workflow and updates
`/dev/`. Builds and manifests remain ignored. Keep downloaded PDFs and temporary
research environments outside the tracked tree unless redistribution is intended.
