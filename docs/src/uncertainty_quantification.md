# Forward uncertainty quantification

An uncertain kinetic rate, feed or initial population gives a range of possible
model predictions. Forward uncertainty quantification propagates a declared
input law through a PBM; it needs no observations, likelihood or fitting step.
This differs from the size distribution *within* a population. A broad PSD and
uncertainty in its predicted median describe different kinds of variability.

Constructor signatures and keyword contracts are in the separate
[UQ API reference](api/uncertainty.md).

```@raw html
<!-- Preserve the former guide-section URL after moving its reference entries. -->
<span id="Forward-UQ-reference"></span>
```

## A first calculation

Consider removal at an uncertain, constant rate. Each realization solves

```math
\frac{\partial n(x,t)}{\partial t}=-k n(x,t),
\qquad k\sim\mathcal U(0.2,0.6).
```

Use the ordinary problem, solver and observation interfaces:

```julia
using PopulationBalanceModels
import Distributions

grid = Grid1D(range(0, 1; length=17))
model = PBMModel(removal=Death((x,p,t,ctx) -> p.k))
base = PBMProblem(compile(model,grid), 2 .* grid.widths, (0.,2.);
                  p=(k=0.4,))
inputs = UncertainInputs((k=Distributions.Uniform(0.2,0.6),))
uq = UQProblem(base,inputs;
    outputs=(number=MomentObservable(0),),
    times=[0.,1.,2.], dt=0.02, method=RK4())

mc = propagate(uq,MonteCarlo(500; seed=42); max_solves=500)
summary = uq_summary(mc; probabilities=(0.025,0.5,0.975),
    events=(low_number = y -> y.number[1,end] < 0.8,))
summary.mean.number       # one row, three output times
summary.events.low_number

training = pce_design(inputs; n=12, seed=43)
validation = propagate(uq,MonteCarlo(100; seed=44); max_solves=100)
pce = fit_pce(uq,PolynomialChaos(2);
              design=training, validation, max_solves=12)
uq_summary(pce).variance.number
diagnostics(pce).condition
pce.validation.rmse
diagnostics(pce).accuracy_status  # :passed, :failed, or :not_run
surrogate = evaluate_pce(pce,MonteCarlo(10_000; seed=45))
uq_summary(surrogate).quantiles.number
```

The default scenario builder merges named inputs into `base.p`. Supply
`setup(theta,base)` to change an initial state, a bulk state, prescribed fields,
or a mesh. Rebuild numerical preparation when its coefficients depend on an
uncertain input. Setup must return a fresh valid problem and must not mutate
the base. `UncertainInputs(...; valid=...)` declares physical input restrictions;
invalid draws are recorded, never clipped or redrawn.

## Inputs, problems and observations

| Input source | Meaning |
|:--|:--|
| Named univariate distributions | Independent reference marginals |
| A joint distribution, or `rng -> joint_sample` | One joint draw; correlations are preserved |
| Matrix columns or a vector of supplied samples | Recorded realizations in their supplied order |
| `UncertainInputs(parameter_draws(...))` | Physical posterior draws, retaining inference coordinates and sample identities |

`map` transforms a **whole joint draw** into physical inputs. Record its meaning
with `coordinates` and `metadata`. A deterministic map can produce correlated
physical parameters from independent reference variables; the user must justify
that it represents the intended joint law. Independent parameter priors are
not a substitute for a correlated joint posterior.
The map acts on samples; an extra density Jacobian is not multiplied into the
output values. PCE orthogonality is defined under the declared reference law.

For example, `UncertainInputs(draws)` accepts the result of the existing
[`parameter_draws`](@ref) interface. The overload
`UncertainInputs(estimation,posterior; n,seed)` selects MCMC draws through that
same interface. MCMC draws are not labelled iid. Custom samplers and supplied
sample sets also default to `iid=false`; an explicit `iid=true` is a declaration
about their production, not a diagnostic test of independence.

Actual inference draws retain `shared` and experiment-local parameter groups.
Select the intended experiment explicitly, for example
`UncertainInputs(draws; map=θ -> merge(θ.shared,θ.experiments.run))`.
This preserves the joint draws while making ordinary forward parameters for
the experiment named `:run`; it does not replace its setup callback.

`outputs` is a named tuple of existing observables. Scalars, vectors and multiple
times share one solve per realization. Output blocks use output-component rows
and time columns. `UQResult.layout` records their flattening order and declared
coordinates. The number of components and output times must agree across runs.
An output's `output_coordinates` metadata should identify units, weighting and
any fixed measurement window; it does not certify a callback's mathematics.

Grid, raw-moment, DQMOM and coupled populations retain their usual observation
contracts. `MomentObservable((1,1))` requests a joint moment of a standalone
tensor population. Use explicit `SystemObservable` callbacks for coupled
tensor measurements. DQMOM state entries remain weights and weighted abscissas;
no measured PSD is inferred from moment states automatically.

For an external transient solver, pass `solver=(problem,times)->solution`.
It must check successful completion and return `PBMSolution` for the supplied
problem at exactly those times. UQ checks saved states and observations; it
does not add an adaptive controller or transfer native positivity guarantees to
external ODE methods. For a stationary problem, omit `times` and `dt` and use
`StationarySolver()` or another checked `solver(problem)`. The ordinary
[stationary compatibility restrictions](steady_state.md) still apply.
Choose numerical controls that resolve the uncertain scenarios, not only the
nominal parameters. A fixed native `dt` is not adjusted by this wrapper;
`suggest_dt` remains a local recommendation. Grid/time error and closure error
must be checked separately from input uncertainty and surrogate error.

## Monte Carlo results and failures

`propagate` requires an explicit budget before starting. `diagnostics(result)`
reports actual solver invocations, separate setup/integration/observation times,
and failures by sample, identity and stage. A full forward solve supplies all
outputs; internal solver substeps or operating segments are not separate input
realizations. Only declared `NumericalFailure`s are captured. Programming errors
and inconsistent output layouts remain errors.

Failed columns remain visible as NaNs. Summaries reject an incomplete ensemble
unless `conditional_on_success=true` is explicitly requested. Such a conditional
summary describes successful runs, not the original input law.

`uq_summary` returns means, sample variances with an `n-1` denominator, optional joint
output/time covariance, pointwise empirical quantiles, and event probabilities.
The variance estimator is unbiased under iid sampling; that assertion does not
extend automatically to correlated imported draws.
For declared iid MC it also reports mean Monte Carlo standard errors and exact
binomial intervals for event probabilities. These quantify estimation error in
a statistic, whereas output SD and output quantiles describe propagated input
uncertainty. Variance and quantile MC errors are not estimated by this API.
Imported MCMC draws receive no unjustified iid error bars. Pointwise intervals
are not simultaneous trajectory bands.

## Non-intrusive polynomial chaos

PCE approximates each requested output with a polynomial in reference inputs:

```math
Y(\xi)\approx\sum_{\alpha}c_\alpha\Psi_\alpha(\xi),
\qquad |\alpha|\le p.
```

The backend supports independent normal, uniform and beta marginals, including
mixed products. It uses normalized probabilists' Hermite polynomials
`He_k(z)/sqrt(k!)` for standard normal variables and
`sqrt(2k+1) P_k(z)` for uniform variables on `[-1,1]`. Other normal/uniform
locations and scales are transformed to these coordinates. For `Beta(α,β)`,
the coordinate `z=2x-1` uses Jacobi polynomials with weight proportional to
`(1-z)^(β-1)*(1+z)^(α-1)`, normalized under the beta **probability** law. Thus
`E[Ψα Ψβ] = δαβ`. Four inputs give five terms at degree one and fifteen at
degree two.

One recurrence interface supplies basis evaluation and quadrature. The built-in
families need only existing dependencies. [`pce_basis`](@ref) is the public hook
for another distribution or a basis-library adapter: return a [`PCEBasis`](@ref)
with probability-orthonormal recurrence coefficients. Monic polynomials or
polynomials normalized under an unnormalized weight need conversion first.
This contract also keeps future basis backends separate from PBM solving.
The classical recurrences follow the [NIST DLMF](https://dlmf.nist.gov/18.9);
PoCET illustrates their use for dynamic uncertainty propagation
([Petzke et al., 2020](https://doi.org/10.1016/j.ifacol.2020.12.560)).
A basis for an arbitrary empirical or correlated posterior is not constructed
automatically. Correlated physical parameters still require an explicit map.

`pce_design` records actual LHS or random training points. By default LHS
stratifies the marginal CDFs over their full support. Optional training `bounds`
instead define a finite box; this changes the training design **without changing
the propagated law**. Caller-supplied designs and reusable `UQResult`s are also
accepted. Keep training and validation roles distinct.

By default, `fit_pce` uses an SVD least-squares solve for all output components/times, with
rank and conditioning checks. It neither forms normal equations nor introduces
a ridge. Failed training points are retained in `PCEFailure.evaluations`, not
deleted. Validation inputs must be disjoint from training; disjointness alone
does not establish independent sampling. Reusing already evaluated validation
data costs no additional PBM solves.
Surrogate predictions from `evaluate_pce` cannot serve as full-model training
or validation references, even though they share the `UQResult` container.

Orthonormal coefficients give the surrogate mean `c₀` and covariance from its
nonconstant coefficients. Quantiles instead require surrogate evaluation and
are not presumed Gaussian. Negative surrogate predictions are retained.
Independent full-model validation reports per-output/time errors separately
from propagated uncertainty. Increasing polynomial degree, training size, MC
size and PBM numerical resolution answer different convergence questions.

### Quadrature construction

For a few independent inputs, Gaussian quadrature offers a second way to
calculate coefficients. With an orthonormal basis,

```math
c_\alpha=\mathbb E[Y\Psi_\alpha]
\approx\sum_{r=1}^{Q}w_rY(\xi_r)\Psi_\alpha(\xi_r).
```

Using the same `uq` and full-model `validation` from the first calculation:

```julia
rule = pce_quadrature(inputs; order=5)
projection = fit_pce(uq,PolynomialChaos(3; construction=:projection);
    design=rule, validation, max_solves=5,
    validation_atol=1e-8, validation_rtol=0.01)
diagnostics(projection).accuracy_status
```

`order=q` means q nodes in **each** coordinate; a tuple selects individual
orders. The probability weights sum to one. Golub–Welsch nodes and Christoffel
weights retain small tail contributions without a weight floor. A q-point
univariate rule integrates polynomials through degree `2q-1`; resolving the
basis products requires at least `q=p+1` for degree p. This does not establish
accuracy for a nonpolynomial PBM output. Refine quadrature order separately.
Full tensor cost grows as the product of the node counts; `max_points` checks
that size before allocating. This is neither sparse quadrature nor intrusive PCE.

Projection rejects unresolved basis products, invalid weights and weights that
do not sum to one. They are never silently renormalized. Weighted designs must
select `construction=:projection`; explicitly pass `rule.samples` for an
unweighted regression comparison. Quadrature nodes are not a random validation
sample. Reusing `projection.training` retains its recorded weights and incurs
zero new solves.

### Validation and refinement

A small training residual and a well-conditioned matrix do not establish PCE
accuracy. For example, fitting a linear polynomial to `Y=X^3` at `X=-1,0,1`
interpolates perfectly. Under a standard normal input it predicts variance 1,
where the exact variance is 15.

[`validate_pce`](@ref) checks every output and time against disjoint full-model
evaluations. Its acceptance criterion is

```math
\mathrm{RMSE}_j\le a_j+r_j\,\mathrm{SD}_{j,\mathrm{validation}}.
```

`validation_atol` has output units; `validation_rtol` is relative to the reference
output SD. Named tolerances can distinguish outputs with different units. A
constant output has zero SD and needs an appropriate absolute tolerance.
Failures retain all residual diagnostics; `fit_pce` does not throw away an
inaccurate candidate. Missing validation is explicitly `:not_run`.

[`refine_pce`](@ref) adds a selection rule. It requires validation and at least
two distinct candidates, then stops when two successive candidates pass
validation **and** their analytical means/variances meet the requested stability
limits. Continue the first example with:

```julia
sizes = (8,12,16,20)
study = refine_pce(uq,PolynomialChaos.((1,2,3,4));
    designs=[pce_design(inputs; n,seed=100+n) for n in sizes],
    validation, max_solves=sum(sizes),
    validation_atol=1e-8, validation_rtol=0.01,
    mean_atol=1e-8, mean_rtol=0.001,
    variance_atol=1e-8, variance_rtol=0.01)
study.converged || error("Inspect study.history and the retained candidates")
selected = study.surrogate
test = propagate(uq,MonteCarlo(100; seed=46); max_solves=100)
validate_pce(selected,test; atol=1e-8,rtol=0.01).passed
```

Mean/variance stability uses `atol + rtol*max(abs(previous),abs(current))`;
variance absolute tolerances have squared-output units. Specify absolute limits
near zero. Hold degree fixed while enlarging a regression design or increasing
quadrature order to isolate those errors; then increase degree. A prescribed
list is not an adaptive/sparse basis algorithm. All candidates, criteria,
coordinates, input points and actual solve counts remain available. Existing
`UQResult` inputs can be reused at zero cost; ordinary repeated designs each
incur their stated solve count. The worst-case budget is checked before solving.

If the plan is exhausted, `converged=false` and `surrogate=nothing`; incomplete
evidence is not returned as a selected surrogate. A forward failure also stops
selection and retains earlier candidates plus the failed sample-level
evaluations in `diagnostics(study).failure`. Programming errors remain errors.
This empirical rule covers the
specified outputs and validation set, not every tail or event. A set used for
selection is no longer an untouched test set, which is why the example uses
fresh final inputs. Forward discretization/closure errors and validity of the
input law remain separate checks.

A complete beta-removal calculation compares both constructions with the
analytical moments of `N(t)=2 exp(-k t)`, `k~Beta(2,2)`:

```sh
julia --project=. examples/pce_refinement.jl
```

[Download the complete example](assets/source/examples/pce_refinement.jl).

The [milling case](case_studies/milling_uncertainty.md) demonstrates all these
steps. Bhonsale et al. (2018), [DOI 10.3390/pr6120255](https://doi.org/10.3390/pr6120255),
provides the physical UQ example; Makrygiorgos, Maggioni and Mesbah (2020),
[DOI 10.1016/j.compchemeng.2020.106814](https://doi.org/10.1016/j.compchemeng.2020.106814),
provides context for non-intrusive surrogates of two-coordinate PBMs. This change
does not reproduce the latter study or add its sparse/Kriging methods.
