# Preparing and executing forward problems

Declare population mechanisms in `PBMModel`, choose a representation and methods
with `compile`, bind initial data and operating inputs in `PBMProblem`, and execute
with `simulate`. The package performs numerical conversion, run preparation,
stepping, segmentation and completion checks. The physical laws remain ordinary
callbacks evaluated with current parameters, time and stage context.

## A distribution, a daughter law and an operating change

```@example prepared_forward
using PopulationBalanceModels
using Distributions: Normal, truncated

grid = Grid1D(range(0.0, 2.0; length=41); coordinate=:volume)
daughters = ContinuousDaughters((r,parent,p,t,ctx) -> 2.0;
    coordinates=:ratio, support=(0.0, 1.0), count=2.0)
model = PBMModel(fragmentation=Breakage((v,p,t,ctx) -> p.k,
    daughters; overflow=:discard))
operator = compile(model, grid; methods=TermMethods(breakage=FixedPivot()))
initial = InitialDistribution(truncated(Normal(0.8, 0.15), 0.0, 2.0); total=3.0)
history = OperatingHistory([0.5]; parameters=((k=0.2,), (k=0.6,)))
problem = PBMProblem(operator, initial, (0.0, 1.0); p=(k=0.2,), history)
solver = ForwardSolver(SSPRK33(); dt=0.01)
solution = simulate(problem; solver, saveat=[0.0, 0.25, 0.5, 0.75, 1.0])
initialization_report(problem)
```

Here `n(v)` is number concentration per volume-coordinate unit, `total` is
number concentration, and `k` is a hazard in inverse time. The binary daughter
measure is uniform in the volume ratio: ∫b(r)dr=2 and ∫r b(r)dr=1. The numerical
state contains bin integrals. Daughters below the first pivot are explicitly
unresolved: `overflow=:discard` permits their exit rather than redistributing
them. Add `CumulativeAccounting` in a `PBMSystem` when their number/material must
be recorded; the [full example](assets/source/examples/prepared_forward.jl)
does this without adding new physical mechanisms.

## Initial data and their interpretation

| Declaration | Meaning and default |
|:--|:--|
| Numerical array | Fixed solver state, copied; `remake(pb;p=...)` preserves it |
| `InitialDensity((x,p,t)->n; support=...)` | Physical number density; never normalized automatically |
| `InitialDistribution(law; total=...)` | Probability law times total number; omitted domain tails are retained as exclusions |
| `truncation=:renormalize` | Explicit conditioning on the represented domain and declared support |

`law` and `total` can be `(p,t)` callbacks. A description remains attached to the
problem; parameter changes through `remake` rebuild the initial state. An explicit
`u0` override replaces that description. Initial descriptions use the problem's
base parameters at its starting time; operating overrides control the evolution.
Neither initialization nor solving mutates the base problem.

Univariate probability laws use CDF/CCDF differences for bin probabilities,
including stable upper-tail differences. General densities use positive
Gauss–Legendre quadrature, checked by increasing its order; `quadrature`, `rtol`,
`atol` and `maxorder` are inspectable controls. The report records the applied
policy, retained number and excluded number where known. For a density with no
declared support, the grid domain defines the integration range and the external
population measure is **unknown**, not zero.

For `Grid2D`, supply a joint density `((x,y),p,t)->n` or an explicit two-dimensional
joint distribution. Support is a pair of intervals. The package integrates each
rectangle; it does not infer statistical independence. Numerical axial arrays
remain supported. A continuous axial initial profile still requires an explicit
matrix: initialize the local population and declare how it varies along the
reactor. This differs from a continuous [density inlet](axial_reactors.md).

For `MomentSet`, initialization integrates the **continuous** distribution on its
support. These moments generally differ from moments of a finite centroid
projection. Standard full-support laws have analytical raw moments; other laws
use checked quadrature. Density callbacks require finite explicit support.
The selected raw-moment family receives raw moments; DQMOM receives weights and
weighted abscissas from its existing inversion. Empty/rank-deficient states are
accepted only where the selected family already supports them. No artificial
seed, new moment, family switch or realizability projection is introduced.

```julia
using Distributions: Gamma
basis = MomentSet(0:3; coordinate=:volume)
moment_operator = compile(PBMModel(), basis; moment_method=DQMOM())
moment_problem = PBMProblem(moment_operator,
    InitialDistribution(Gamma(2.0, 0.5); total=3.0), (0.0, 1.0))
```

Named `PBMSystem` populations accept descriptions independently, e.g.
`PBMProblem(system, (cells=initial,crystals=other_initial), span; bulk0, p)`.
Each population converts through its own representation. Packed numeric states
and scalar bulk initial values retain their existing meanings.

## Solver configurations and controls

`ForwardSolver` is callable as `solver(problem,times)`. Pass the same object to
`InferenceExperiment(...;solver)` and `UQProblem(...;solver)`. Their solve budgets
still count full forward calls; accepted/rejected integration steps are separate
inner work. Existing custom callable solvers remain valid in these consumers.
UQ metadata retains per-sample inner integration work; MAP/MCMC diagnostics
aggregate it separately from forward-solve and derivative-evaluation counts.

| Configuration | Contract |
|:--|:--|
| `ForwardSolver(SSPRK33();dt=...)` | Native fixed maximum step; shorten to each output |
| `control=StepControl(cfl=0.4)` | Also recompute the applicable local `suggest_dt` recommendation |
| `StepControl(cfl=nothing,bound=...,guard=...)` | Explicit model-specific step bound/guard when no general recommendation applies |
| `ForwardSolver(algorithm;backend=:external,...)` | Optional checked external residual solve |
| `ForwardSolver(SciMLSensitivitySolver(algorithm;...);backend=:sensitivity)` | Delegate eligible continuous sensitivity dispatch |

Legacy `simulate(pb;method=RK4(),dt=...,saveat=...)` is the fixed-maximum-step
form. Complete space–time updates and `MeasureEuler` keep their own algorithms,
alignment and admissibility restrictions. They reject `StepControl`; they are
not recast as ODE residuals.

`bound(pb,u,t)` returns an additional positive maximum step; `Inf` is allowed.
`guard(pb,u,t)` returns Bool and is checked at all native RK stages and the final
combination. Both callbacks must be pure; use public `rhs!`, `bulk`, `population`
and `stateview` to express physical bounds. An invalid initial guard is an input
error. A rejected **trial state** is retried with a smaller step, preserving the
accepted population, bulk, accounting and working storage. Rate-law declaration
errors, invalid coefficients and programming errors are not retryable failures.
`min_dt`, `max_rejections` and the solver's `maxsteps` bound computational work.

A current-state CFL or depletion estimate neither bounds all later stages nor
estimates integration error. WENO has no positivity limiter, and arbitrary bulk
balances need their own justified constraints. Check temporal refinement
independently. No negative-value clipping, numerical floors or moment projection
are performed. Diagnostics record accepted/rejected steps, rejection reasons,
segment/backend work and stopping reason.

## Prescribed operating inputs

`OperatingHistory` has one parameter/field entry per interval. Parameter entries
are NamedTuple overrides of declared parameters or `p->overrides`; field entries
are NamedTuples or `(p,t)->fields`. Existing compiled field callbacks remain in
effect; history fields override names they explicitly supply. Fixed transition
times are strictly interior to the problem span. State resets, triggered events
and fitted transition times are outside this contract.

The package holds each interval's conditions through **all** integration
evaluations, including endpoint RK stages. It transfers every state component,
restarts external multistep methods, and assembles requested outputs once.
`operating_inputs(pb,t)` and public observations select the new/right interval
at a transition; `side=:left` inspects the previous conditions. State is
continuous even when a flux-dependent observation jumps. A solution remains
associated with the full problem/history. Methods requiring aligned travel
times reject incompatible transitions or outputs instead of taking a partial
characteristic step.

`accounting_rates(problem,state;time=t)` follows the same right-hand convention
for instantaneous escape rates. Cumulative ledger values remain continuous.

## Checked optional external integration

In an environment containing this checkout, SciMLBase and OrdinaryDiffEqTsit5:

```julia
using SciMLBase, OrdinaryDiffEqTsit5
solver = ForwardSolver(Tsit5(); backend=:external, reltol=1e-8, abstol=1e-11)
solution = simulate(problem; solver, saveat=[0.0, 0.5, 1.0])
```

This returns `PBMSolution` automatically. The adapter checks return status,
completion through the final requested time, requested output coverage, state
shape, finiteness and saved-state admissibility. These checks do **not** establish
native internal-stage validity or a general external positivity guarantee.
Use `simulate` for histories: a single raw `odeproblem` cannot encode their
endpoint convention. Rebuild an ODE problem from a remade PBM problem when its
parameter-dependent preparation changes.

## Preparation lifetimes and derivatives

1. `compile` validates methods/representations and prepares fixed geometry.
2. `prepare_run(compiled,p)` binds parameter-dependent projections and creates
   solve-local working storage. `simulate` invokes it automatically per interval.
3. Residual callbacks evaluate state/time/bulk-dependent quantities at current
   integration inputs. They must not be frozen into a parameter cache.

The default term-level `prepare_run(term,method,cache,representation,p)` copies
the structural cache. Numerical extensions can override it to rebuild explicitly
declared parameter-dependent information; their `add_term!` interface is unchanged.
See the [extension guide](development.md#Preparation-and-execution-contracts).
Structural mesh changes still belong in explicit setup/recompile callbacks.

Native fixed schedules preserve supported ForwardDiff arithmetic. A controlled
state/parameter-dependent schedule rejects dual arithmetic; select finite
differences explicitly. Eligible SciMLSensitivity solvers retain their existing
derivative restrictions and are delegated through the configured wrapper.
Continuous sensitivities across operating changes are not yet implemented;
prescribed fixed transition times can use native fixed-schedule ForwardDiff.
The generic `backend=:external` path rejects dual inputs, rather than silently
differentiating external step selection. Choose the continuous-sensitivity backend
for eligible inference, or explicitly select finite differences.
No adaptive-AD or event-sensitivity guarantee is claimed.

## Continuous daughter preparation

`ContinuousDaughters` declares expected count, support, coordinate convention and
conserved power. `Breakage` still declares its additive property and unresolved
birth policy. `FixedPivot` chooses positive quadrature with
`daughter_quadrature`, `daughter_rtol`, `daughter_atol`, `daughter_maxorder`.
Integration is split at pivot hats; lower/upper tails retain their own number
and additive property. The package checks quadrature refinement and count/property
balances without normalization or rescaling to force conservation.
An unresolved tail is represented by its count and mean additive property;
this does not preserve every higher tail moment independently.

For a length-coordinate law preserving volume, declare `conserved_order=3` and
`coordinate_map=AdditiveCoordinate(x->x^3,cbrt)` on `Breakage`. A ratio density
includes the appropriate inverse-parent change of variables when integrated
in daughter coordinates. Supplied analytical `daughter_moment` integrals are
reused by moment weak forms; otherwise inner quadrature is checked numerically.
Strict `ExactClosure` does not become available merely because an inner integral
is analytical.
For a general additive map that is not a raw power, explicitly use
`conserved_order=nothing`; Breakage checks the supplied map's property balance.

Parameter-dependent fixed-pivot projections rebuild automatically after `remake`.
State/time-dependent daughter laws must declare those dependencies: moment
quadrature evaluates them at current inputs; the precomputed direct projection
currently rejects them clearly. Discrete laws, `SelfSimilarDaughters` and
`UniformVolumeDaughters` retain their existing treatment.

## Reproduction and migration

```sh
julia --project=. examples/prepared_forward.jl
julia --project=. test/forward_execution.jl
```

The [test setup](verification.md#Optional-integrations) describes optional
environments; run `test/forward_execution_integration.jl` there for real ODE
completion/failure checks. No production fit or ensemble is needed.

The precipitation example now declares its physical transport/depletion bound
while the package owns execution. The retained protein controller additionally
uses a locally constant bucketed derivative schedule; the milling adapter uses
an exact algebraic reduction, material-coordinate scaling and backend inequality
constraints. These genuine specializations have not been silently replaced by
the new controller. Retained figures/data and producer records are unchanged;
new source hashes require a later equivalence review or rerun before claiming
that historical datasets were produced by this implementation.
