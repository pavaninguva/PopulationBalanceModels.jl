# Stationary population balances

A continuous crystallizer can operate at a constant size distribution even
though crystals continue to nucleate, grow and leave the vessel. Its stationary
equations balance those rates directly. There is no time span or physical initial
condition: the nonlinear solver needs a numerical guess for the unknown state.

For a well-mixed vessel with residence time ``\tau``, constant growth ``G>0``
and boundary nucleation ``J``, the model is

```math
\frac{d}{dL}\left(Gn(L)\right)+\frac{n(L)}{\tau}=0,
\qquad G n(0)=J.
```

On the nonnegative half-line its analytical solution and moments are

```math
n(L)=\frac{J}{G}\exp\left(-\frac{L}{G\tau}\right),
\qquad M_k=k!\,J G^k\tau^{k+1}.
```

Here ``n`` is number per process volume per size; ``J`` is number per process
volume per time. The package's sectional unknown remains the integral of ``n``
over each bin. Particle withdrawal is `Dilution`, not an extra size-boundary
condition. A finite computational upper boundary introduces an additional escape
flux, which must be included in the number balance.

## Define and solve

Load the optional NonlinearSolve dependency in your working environment. The
physical declaration and spatial methods are the same ones used for a transient
calculation.

```julia
using PopulationBalanceModels, NonlinearSolve

grid = Grid1D(range(0,12; length=161); coordinate=:length)
model = PBMModel(
    growth = Growth((L,p,t,ctx) -> p.G),
    births = Nucleation((p,t,ctx) -> p.J, 0.0),
    withdrawal = Dilution((L,p,t,ctx) -> 1/p.tau),
)
op = compile(model, grid;
    methods=TermMethods(nucleation=BoundaryInjection()),
    bc=SizeBC(NoFlux(), OpenBoundary()))
problem = StationaryPBMProblem(op, zeros(length(grid));
    p=(G=0.4,J=1.2,tau=2.0), autonomous=true)
solution = solve_stationary(problem) # RobustMultiNewton()
solution.converged || error("stationary solve failed: $(solution.retcode)")
bin_density = density(grid, solution.u)
diagnostics(solution)
```

`autonomous=true` declares that all callbacks, fields and inlet conditions are
time-independent. The package cannot prove that property for arbitrary Julia
callbacks. Alternatively, `time=t` explicitly freezes prescribed conditions at
one time. That calculation does not describe the response to a changing history.
Neither choice inserts physical time integration.

`solve_stationary(problem, algorithm; abstol, reltol, maxiters, ...)` accepts a
NonlinearSolve algorithm. `nonlinearproblem(problem)` exposes the same problem
to that library directly, with flattened physical unknowns. Use
`solve_stationary` when you want the package's independent final checks and named
result accessors. Ordinary callback, closure and coefficient restrictions remain
in force. Invalid trial states raise an error; no negative values are clipped.
RobustMultiNewton does not itself guarantee nonnegative or realizable iterates.
For quadrature moment families the default uses finite-difference Jacobians,
because inversion eigensystems need not support dual numbers. Sectional and
exact-moment residuals retain NonlinearSolve's default differentiation. An
explicit algorithm controls its own differentiation backend; the physical
closure is unchanged.

## Coupling, constraints and scaling

For a `PBMSystem`, supply named population guesses and named bulk guesses:

```julia
problem = StationaryPBMProblem(system, (crystals=population_guess,);
    bulk=(c=concentration_guess,), p=parameters, autonomous=true)
solution = solve_stationary(problem)
population(solution, :crystals)
bulk(solution, :c)
```

All physical residuals are assembled together, including the usual same-state
bulk callbacks and selected-mechanism exchanges. Do not couple solute uptake to
the **total stationary population derivative**, which is zero: growth and births
can consume solute while withdrawal removes the resulting crystals.
`ConservativeExchange` and `MomentExchange` retain that distinction.

Native cumulative loss ledgers are historical integrals and are excluded from
the stationary unknowns. `accounting_rates(solution)` instead returns removal
rates, including nonzero escape at a stationary distribution. Physical reservoirs
with their own balances remain unknowns. User-defined passive counters must be
removed from the stationary model; the package cannot infer their role.

For a coupled axial reactor, `axial_problem(problem, domain; axial_scheme=Upwind())`
accepts a local stationary system directly. Supply `guess` for a spatially varying
state or let it repeat the local guess in every axial cell. Its inlet and fields
must obey the same autonomy/frozen-condition declaration. This uses the existing
axial flux discretization without a fabricated transient time span.

Closed conservative populations often have a family of stationary states. Supply
a conserved inventory by replacing a redundant equation, for example
`constraints=(1 => (u,p,t)->sum(u)/target_number-1,)` for one closed population.
The callback returns a dimensionless constraint and receives the state in its
ordinary shape. Rows use linear physical-state indices. The solver also checks
every original physical balance, so replacing an independent equation cannot
conceal an unbalanced source. Use named population/bulk accessors in callbacks
for coupled systems instead of assuming offsets.

`residual_scale` is a positive scalar or state-shaped array of characteristic
physical rates. Physical residuals are divided by it; constraint callbacks are
already dimensionless. Choose scales to make the requested tolerance meaningful
across population and bulk units. A result is accepted only when the backend
reports success and both constrained and original scaled residuals satisfy
`abstol + reltol*r0`, with `r0` the initial constrained infinity norm. The default
`reltol=0` makes this an absolute scaled-balance test.

## Observe and estimate parameters

Observation operators apply to the converged state, with no invented time
history. For a coupled sectional system, for example,
`observe(PopulationMoment(:crystals, 0), solution)` returns number concentration
and `observe(BulkObservable(:c), solution)` returns dissolved concentration.
For an axial system, wrap a local observation in
`CellObservable(BulkObservable(:c), outlet_cell)`.

Use the same Bayesian estimation API as for transient models. Omit timestamps
when declaring stationary data:

```julia
measurement = Measurement([observed_number], MomentObservable(0);
    likelihood=GaussianLikelihood(sigma=number_sd))
experiment = InferenceExperiment(:vessel, problem, measurement;
    solver=StationarySolver(abstol=1e-12))
parameter = ParameterSpec(:J; initial=1.0, transform=LogTransform(),
    prior=Prior(Normal(0, 1); coordinates=:inference)) # log(J) prior
estimation = EstimationProblem((parameter,), (experiment,))
result = fit(estimation) # load Optim; MAP by default
```

Here `problem` is a `StationaryPBMProblem` with a parameter `p.J`; load
Distributions for `Normal`. A stationary data vector contains outputs at one
operating point. Matrix columns are repeat measurements of that same state.
Different feeds, residence times or frozen conditions need separate experiments;
kinetic parameters can be shared, while calibration/noise parameters can be
experiment-specific. Time-labelled measurements are rejected for this overload.

`predictions`, likelihood/prior evaluations, MAP, profiles, Gauss–Newton and
Laplace approximations, posterior prediction and optional NUTS all use this same
statistical model. Failed equilibria become visible `NumericalFailure`s, never
valid predictions. The legacy `Observation(data, observable)` and
`CalibrationProblem` interface also supports stationary calibration; its
frequentist likelihood/discrepancy interpretation remains distinct from priors.

For a smooth isolated equilibrium ``R(u,\theta)=0``, sensitivities solve

```math
\frac{\partial R}{\partial u}\frac{\partial u}{\partial\theta}
=-\frac{\partial R}{\partial\theta}.
```

The default ForwardDiff path differentiates these equations, including parameter
transforms, bulk coupling, constraints and the observation response. Nested
derivatives retain the full posterior Hessian; differentiating Newton's iteration
history is unnecessary. The implementation currently forms dense Jacobians and
linear factorizations. `StationarySolver(condition_limit=...)` checks the
equilibrated state Jacobian without jitter or a pseudoinverse. Singular roots
and unresolved conditioning reject implicit sensitivities. Constraints must
remove genuine inventory nullspaces, as in the forward problem.

An eigen-based moment inversion may not support dual numbers. Explicit
`derivatives=:finite_difference` fits the same posterior by resolving each
perturbed equilibrium; verify gradient and solve-tolerance refinement. There is
no silent derivative fallback. Root branch changes, rank changes and extinction
events are not smooth sensitivity paths. A numerical guess selects a branch;
it is not a physical initial condition or evidence of uniqueness. The solver
restarts from the declared guess at each evaluation, without hidden continuation.

Run the small synthetic MSMPR calibration in
[estimate_stationary_crystallizer.jl](assets/source/examples/estimate_stationary_crystallizer.jl):

```sh
julia -e 'using Pkg; Pkg.activate("/tmp/pbm-steady"); Pkg.develop(path=pwd()); Pkg.add(["NonlinearSolve", "Optim", "Distributions"])'
julia --project=/tmp/pbm-steady examples/estimate_stationary_crystallizer.jl
# AdvancedHMC and MCMCDiagnosticTools additionally enable --sample.
```

The example compares exact stationary moments with a joint growth/nucleation
fit, verifies gradients and reports local sensitivity-based uncertainty. It uses
synthetic measurements and imposes no experimental interpretation.

## Continue an operating point

Continuation repeatedly solves the same stationary equations with new physical
parameters. It can help reach a difficult operating point or follow a branch:

```julia
path = continue_stationary(problem,
    lambda -> merge(problem.p, (feed=3.5+1.5lambda,)), (0.0,1.0))
path.converged || error("requested endpoint was not reached")
last_solution = path.solutions[end]
```

The parameter callback returns the complete parameter object. The helper solves
the anchor, then uses each accepted state as the next guess. It halves failed
increments and grows successful ones by a factor of 1.5, within the selected
limits. `path.attempts` retains rejected corrections; partial paths are returned
with `converged=false`. Pure callbacks are required so rejected evaluations do
not mutate user state. This is natural-parameter continuation: it cannot follow
a fold that reverses the parameter. Pseudo-arclength, automatic mesh transfer,
stability analysis and enumeration of all roots are not implemented.

## Supported operators and verification

| Operator | Stationary support |
|:--|:--|
| Grid1D sectional residuals | Growth, sources, removal, diffusion and collision residuals with their existing restrictions |
| Moment residuals | Existing weak forms and closure acceptance; no new realizability guarantee for Newton iterates |
| Grid2D residual transport | Existing Upwind/WENO5 and source restrictions |
| Axial residuals and coupled axial systems | Existing physical fluxes and local balances; no new boundary interpretation |
| HOC, characteristic and other complete direct updates | Rejected: these do not define a spatial residual |
| Moment extinction/escape events | Rejected: these require native MeasureEuler events |

The analytical MSMPR verification independently checks the finite-volume
bidiagonal system, grid convergence against exact bin integrals, and exact
unbounded moment equations. A concentration-dependent-growth version checks the
coupled material balance against a scalar analytical reduction. Additional tests
cover axial feed inventories, diffusion equilibrium, loss rates, conservation
constraints and continuation failure. A small residual is a statement about
the discretized equations; check size/domain refinement separately. It does not
establish uniqueness or dynamical stability. Estimation additionally requires a
compatible observation model and a resolved, consistent solution branch.

Run the complete verification example after installing NonlinearSolve in your
working environment:

```sh
julia --project=. -e 'using Pkg; Pkg.add("NonlinearSolve")'
julia --project=. examples/steady_state_crystallizer.jl
```

The [full runnable source](assets/source/examples/steady_state_crystallizer.jl)
prints spatial errors and number-balance residuals, solves the coupled moment
model, and continues its feed concentration. For an isolated environment that
does not change the package's Project.toml:

```sh
julia -e 'using Pkg; Pkg.activate("/tmp/pbm-steady"); Pkg.develop(path=pwd()); Pkg.add("NonlinearSolve")'
julia --project=/tmp/pbm-steady examples/steady_state_crystallizer.jl
julia --project=/tmp/pbm-steady test/stationary_integration.jl
```
