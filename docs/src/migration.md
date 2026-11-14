# Migrating existing workflows

The first public release, 0.1.0, consolidates internal development changes to
parameter declarations, moment compilation and reusable forward execution.
Existing numerical arrays, physical laws, direct-discretization method selection
and custom `solver(problem,times)`
callbacks remain supported. This page describes the final interfaces; individual
guides give their scientific and numerical restrictions.

## Bayesian parameter declarations and result coordinates

Declare the physical starting guess and physical prior, together with an
explicit internal transformation:

```julia
using PopulationBalanceModels
using Distributions: Normal, truncated

growth = ParameterSpec(:g0; initial=0.2,
    prior=truncated(Normal(0.2, 0.05), 0.0, Inf),
    transform=LogTransform())
```

Use `IdentityTransform()` explicitly for untransformed parameters. Different
parameters can use different transforms. An unrestricted physical Normal does
not become positive merely because log coordinates are selected: truncate the
physical law explicitly when that is the intended belief. A Normal belief about
`log(g0)` is expressed by a physical `LogNormal` prior.

`initial` is neither a prior statistic nor a fixed forward parameter. It is passed
to the estimation setup in physical units, and does not mutate the base problem.
Physical rate callbacks continue to receive physical parameters.

For ``\theta=T(z)``, all inference uses

```math
\ell_z(z)=\log L(y\mid T(z))+\log\pi_\theta(T(z))+
\log|\det DT(z)|.
```

MAP maximizes this declared-coordinate density. A nonlinear transformation can
change the mapped MAP; it preserves the represented posterior probability law
when Jacobians are handled consistently. Laplace and Gauss–Newton covariances
and Gaussian draws are in ``z``. Physical summaries transform those draws; a
mapped MAP is not generally a posterior mean. An auxiliary sampling map adds
its own determinant and does not redefine the MAP target.

The result convention is `inference_map_v2`. Interim `physical_map_v1` and
older/unversioned MAP or local-Gaussian results require compatibility checking
and potentially recomputation. Existing MCMC draws can remain valid when they
represent the same physical posterior. Keep their original producer records;
do not relabel retained figures or rewrite hashes. Tagged `Prior` wrappers are
compatibility declarations whose tagged probability laws are preserved.
See [parameter estimation](inference.md) for gradient and support checks.

## Moment compilation

```julia
model = PBMModel(growth=Growth(PolynomialRate((0.0, 0.0, 0.1))))
basis = MomentSet(0:3; coordinate=:volume)
operator = compile(model, basis; moment_method=QMOM())
moment_plan(operator)
```

`moment_method` replaces ordinary uses of the older `method`/`closure` aliases.
Those aliases and deliberate rule overrides remain supported; conflicting
declarations reject. Node counts are inferred only from a compatible basis.
Tracked contributions are analytical automatically, while missing moments use
the selected family. `ExactClosure()` requests analytical evolution only.

`MomentSet` defines the tracked basis, not every family's solver-state layout.
Raw-moment methods advance moments; DQMOM advances weights and weighted nodes.
Analytical contributions do not make closure-dependent trajectories or time
integration exact. See [moment methods](moment_methods.md).

## Initialization and reusable execution

For a physical number density use `InitialDensity`; for a probability law times
total concentration use `InitialDistribution`. Grid conversion and eligible
moment/DQMOM initialization are performed by the package. The default excludes
unrepresented tails without redistribution. `truncation=:renormalize` is an
explicit conditioning choice, not a numerical repair. Parameter-dependent
descriptions are rebuilt by `remake`; numerical arrays remain fixed states.

Simple runs can retain `simulate(problem;method=SSPRK33(),dt=0.01,saveat=times)`.
Use `ForwardSolver` when reusing a configuration in estimation/UQ or requesting
supported execution controls. `StepControl` concerns native stability and
admissibility, not an integration-error estimate. Specialized derivative schedules
and model-specific reductions remain appropriate where documented.

`OperatingHistory` owns supported fixed transitions and endpoint conventions;
`ContinuousDaughters` owns supported expected-count daughter integration.
Run preparation is automatic. See [forward execution](forward_execution.md) for
supported combinations, external completion checks and extension contracts.

## Optional plotting and retained case-study data

The [plotting interface](plotting.md) operates on existing numerical objects.
Compute `predicted = predictions(estimation, result.minimizer)` explicitly,
then call `plot_fit(estimation, predicted)` after loading the optional plotting
integration. Rendering does not simulate or fit.

Large retained case-study exports are optional downloads, separate from numerical
execution. From a source checkout, prepare the manual with:

```sh
python tools/retained_data.py fetch --profile docs
julia --project=docs docs/make.jl
```

Use `--profile paper` for the four publication cases or `--profile all` for
all retained checks. Every fetched payload is checked against its original byte
count and SHA-256. Existing differing files are never overwritten. See
[verification and reproduction](verification.md) for integrity versus current-source
equivalence, and [publishing](publishing.md) for the complete setup commands.
