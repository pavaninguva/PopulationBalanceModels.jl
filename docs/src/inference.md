# Parameter estimation

An estimation problem combines a forward PBM, observations, a measurement
likelihood and parameter priors. `fit` computes a maximum a posteriori (MAP)
point using those definitions. Optional local uncertainty and posterior sampling
reuse the same physical probability model; neither runs automatically with a basic fit.
Declare a physical starting guess and a physical prior; internal coordinates are
chosen from the prior support:

```julia
ParameterSpec(:g0; initial=0.2, prior=LogNormal(log(0.2), 0.5))
```

**MAP convention changed (`physical_map_v1`).** `fit` now finds the mode of the
physical parameter density. Earlier releases optimized the internal-coordinate
density. MCMC retains the intended physical probability law, with coordinate
Jacobians handled internally. Older MAP, Laplace and Gauss–Newton results must
not be relabelled as new results; see [migration](#Migration-and-retained-results).

## A complete first fit

This small **synthetic** experiment observes total population number during
constant-rate removal. Initially the integrated number is 1.5 per unit process
volume, and the independent solution is ``M_0(t)=1.5\exp(-kt)``. Only k is
unknown. Work in the [optional environment](verification.md#Optional-integrations)
with Optim and Distributions available:

```@example first_map
using PopulationBalanceModels, Optim
using Distributions: LogNormal

grid = Grid1D([0.0,0.5,1.0])
model = PBMModel(removal=Death((x,p,t,ctx)->p.k))
operator = compile(model,grid)
forward_problem = PBMProblem(operator,[1.0,0.5],(0.0,2.0);p=(k=0.3,))

times = [0.5,1.0,2.0]
data = [1.24,1.00,0.68] # synthetic number measurements
measurement = Measurement(times,data,MomentObservable(0);
    likelihood=GaussianLikelihood(sigma=0.03))
parameter = ParameterSpec(:k; initial=0.3, prior=LogNormal(log(0.4), 0.5))
experiment = InferenceExperiment(:batch,forward_problem,measurement;dt=0.02)
estimation = EstimationProblem((parameter,),(experiment,))

fitted = fit(estimation;nstarts=2,seed=17)
@assert fitted.converged
estimate = physical_parameters(estimation,fitted.minimizer,:batch)
predicted = vec(only(only(predictions(estimation,fitted.minimizer))))
@assert isapprox(predicted,1.5exp.(-estimate.k .* times);rtol=1e-7)
(removal_rate=estimate.k, predictions=predicted,
 posterior=posterior_components(estimation,fitted.minimizer))
```

The Gaussian standard deviation is a declared synthetic observation assumption.
`initial=0.3` is a starting guess in the same units as the removal coefficient.
It is neither a prior mean nor a fixed rate. The standalone `forward_problem`
still has its nominal `p.k=0.3`; estimation sends each current physical candidate
to a remade problem without mutating that base. To match another nominal tuple,
write `initial=p.k` explicitly.

The LogNormal prior selects log coordinates automatically. Those coordinates
help optimization but do not change the physical MAP objective. For a constant
likelihood and `LogNormal(log(0.2),0.5)`, the physical mode is
``0.2\exp(-0.5^2)=0.1557601566``, whereas the median is 0.2. Inspect convergence
and physical stationarity diagnostics before reporting an estimate. Multistart
searches do not prove a global optimum.

For a distribution-based physical example, continue with
[estimating size-dependent growth](case_studies/estimating_growth.md). The sections
below extend the same assembly to shared/local and nuisance parameters,
likelihood choices, derivatives, [Gauss–Newton](#Request-sensitivity-based-uncertainty),
[Laplace](#Request-full-Hessian-Laplace-uncertainty) and
[MCMC](#Sample-the-same-posterior). Deterministic discrepancy and legacy Gaussian
calibration are described [separately](#Existing-Gaussian-calibration-interface).

## Forward models and observations

Transient and stationary problems share observation operators and the statistical
model. `InferenceExperiment` dispatches to `simulate` or a checked stationary
solve; parameter declarations, priors, likelihoods, nuisance parameters, MAP,
uncertainty and sampling remain the same.

| Forward model | Measurement declaration | Sensitivity path |
|:--|:--|:--|
| Transient sectional, exact-moment and differentiable coupled models | `Measurement(times, data, observable; likelihood)` | Native ForwardDiff; optional continuous ODE sensitivities for eligible residual solvers |
| Transient tensor, axial and eligible complete transport updates | Same timed declaration; select the actual forward solver | ForwardDiff where supported, or explicitly selected finite differences |
| Stationary sectional, exact-moment, tensor, coupled and axial residuals | `Measurement(data, observable; likelihood)` | Implicit equilibrium sensitivities for smooth isolated roots |
| Quadrature/assumed-shape moments and other AD-incompatible residuals | Same declaration for the relevant problem | Explicit finite differences, with resolved perturbed solves and unchanged closure |

This is support for fitting eligible forward models, not every combination of
observable, closure, boundary policy and integrator. A moment model does not
automatically predict a PSD; a complete HOC/characteristic time update does not
define a stationary residual. Rank changes, branch changes and extinction events
do not acquire differentiable sensitivities merely by entering estimation.

`observe(observable, solution)` evaluates native transient or converged stationary
results directly. `CellObservable(observable, cell)` selects a local axial state,
including named populations and bulk variables. Global reactor inventories need
an explicit `FunctionObservable` with the appropriate cell-volume weighting.

For stationary measurements, a vector is one output column; matrix columns are
repeat measurements of one equilibrium. Distinct operating points use separate
experiments, which may share kinetic parameters. `StationarySolver` accepts the
NonlinearSolve algorithm and solve tolerances and uses no artificial observation
times or timesteps. Its implicit derivative solves the linearized equilibrium
equations, including constraints and bulk exchange, rather than differentiating
Newton's iterations. Singular/ill-conditioned state Jacobians reject that path;
there is no automatic regularization. See
[stationary estimation](steady_state.md#Observe-and-estimate-parameters) for a
complete synthetic crystallizer example and the dense-factorization limitation.

The legacy `CalibrationProblem` accepts `Observation(data, observable)` for
stationary Gaussian calibration and timeless `ResidualObservation` or
`WassersteinObservation(x, F, population; ...)` blocks for deterministic fitting.
Their statistical meanings are unchanged. No solver failure is a valid fit.

## Describe the measurements

An observation operator maps a solved PBM state to predicted measurements.
MomentObservable, BulkObservable, FunctionObservable and SystemObservable
can all be used in a Measurement.

Other instruments produce histograms or bulk measurements. For a sectional
distribution, DistributionObservable integrates over the **measurement** bins:

```julia
histogram = DistributionObservable(measured_edges;
    coordinate=:length, population=:crystals, weight=L->1.0)
sizes = Measurement(times, counts, histogram;
    likelihood=MultinomialLikelihood(), source=:particle_counter)
concentration = Measurement(times, measured_concentration, BulkObservable(:c);
    likelihood=GaussianLikelihood(sigma=(theta,experiment)->theta.noise),
    source=:concentration_instrument)
```

Data matrices have one output per row and one time per column. Each experiment
solves at the union of its measurement times. The likelihood sums all bins,
times, and experiments; it is never divided by the number of observations.

The coordinate must agree with the model. Diameter and volume densities differ
by a change-of-variable Jacobian. Weighting is also explicit: number uses
`L->1.0`, volume can use `L->π*L^3/6`, and the nanoparticle
scattering approximation uses `L->L^6`. Bin values are integrals, not density
values sampled at centres. Weighted sectional integrals use eight-point Gauss
quadrature of the piecewise-constant solver density.

A response callback `response(lo,hi,L,p,t)` can supply the probability that a
particle of true size L is detected in a measured bin. This is the hook for
sizing uncertainty and detection efficiency. Its probabilities must be finite
and between zero and one. Resolve discontinuities explicitly; quadrature through
an unresolved detection threshold is another numerical approximation.

Moment states can predict compatible moments and explicit observation integrals.
A histogram requires an independently specified reconstruction, supplied through
a custom observable. QMOM nodes and weights are quadrature data; they are not
automatically a uniquely determined measured PSD.

## Choose the likelihood

The choice follows how the instrument produced the data.

| Data | Likelihood | Interpretation |
|:--|:--|:--|
| Actual particle counts in bins, with a sampled total at each time | MultinomialLikelihood() | Conditional size probabilities in the measurement window |
| Absolute counts from a known sampled volume or exposure | PoissonLikelihood(exposure=...) | Expected counts equal predicted bin concentration times exposure |
| Suitable continuous outputs | GaussianLikelihood(sigma=...) or covariance=... | Absolute errors or an explicitly fitted noise model |
| Nonnegative continuous measurements | TruncatedGaussianLikelihood(sigma=...,lower=0) | Independent Gaussian measurements conditioned above the specified lower bound; not a clipping model |
| Normalized continuous fractions | GaussianLikelihood with independent contrasts | Continuous instrument fractions, without an invented particle count |

For sampled counts ``y_i``, let ``N=\sum_i y_i`` and normalize the predicted
bin integrals to probabilities ``p_i`` over the stated window. The multinomial
contribution is

```math
\log p(\boldsymbol y\mid\theta)
=\log(N!)-\sum_i\log(y_i!)+\sum_i y_i\log p_i(\theta).
```

The measured total supplies the statistical sample size. Twice as many sampled
particles generally supplies more information. A normalized instrument trace
does not supply that count, and must not be multiplied by an arbitrary N.
Zero counts are valid. A nonzero count in a zero-probability bin has zero
likelihood.

For absolute Poisson counts, the contribution retains both
``y_i\log\lambda_i-\lambda_i`` and the factorial normalization.
For Gaussian outputs with covariance ``\Sigma(\theta)``,

```math
\log p(\boldsymbol y\mid\theta)
=-\frac12[\widehat{\boldsymbol y}(\theta)-\boldsymbol y]^\mathsf T
\Sigma(\theta)^{-1}[\widehat{\boldsymbol y}(\theta)-\boldsymbol y]
-\frac12\log\det\Sigma(\theta)-\frac n2\log(2\pi).
```

The determinant is essential when noise parameters are inferred. Covariance
may be a fixed matrix or `(theta,experiment)->matrix` and applies to
`vec(data)`: outputs within a time, then successive times.

For independent truncated-Gaussian measurements with lower bound zero, retain
the normalization for every observation:

```math
\log p(y_i\mid\theta)
=-\frac{[y_i-\mu_i(\theta)]^2}{2\sigma_i^2}
-\log\sigma_i-\frac12\log(2\pi)
-\log\Phi\!\left(\frac{\mu_i(\theta)}{\sigma_i}\right),\qquad y_i\ge0.
```

The location ``\mu_i`` is the model's latent prediction; the conditional
measurement mean differs near zero. `sigma` may be fixed or a
`(theta,experiment)->scales` callback. Correlated truncation is not implemented.
Use this likelihood when measurements follow that model; censored instruments
and clipped outputs require different likelihoods.

Normalized fractions have a sum constraint. Supply a full-row-rank contrast
matrix A with ``A\boldsymbol1=0`` and their covariance:

```julia
fractions = Measurement(times, fractions_data,
    DistributionObservable(edges; coordinate=:length, normalize=true);
    likelihood=GaussianLikelihood(covariance=Sigma, contrasts=A))
```

The likelihood projects data, predictions and covariance before factorization.
It does not invert the singular full covariance or add a diagonal ridge.
Predictive outputs from this likelihood are also in contrast coordinates.
The Gaussian contrast model is an unbounded approximation. Choose it where
noise outside the physical simplex is negligible; bounded-simplex measurements
need a different response/likelihood model.
Declared normalized fractions and post-response predictions must lie in
``[0,1]`` and sum to one at every time (absolute normalization tolerance
``10^{-10}``). Invalid data are rejected; they are never renormalized or clipped.

If moments were calculated from the measured histogram, they share its
measurement errors. Use the histogram alone, or a joint vector observable with
the appropriate joint covariance; do not count both as independent evidence.
Use the same source tag on dependent blocks: InferenceExperiment rejects duplicate
tags. Untagged dependence cannot be inferred automatically.

## Declare fitted and fixed parameters

Each scalar declaration supplies a physical starting value and prior:

```julia
using Distributions: LogNormal
g0 = ParameterSpec(:g0; initial=0.4, prior=LogNormal(log(0.4), 0.4))
noise = ParameterSpec(:noise; initial=0.003,
    prior=LogNormal(log(0.003), 0.4), role=:nuisance)
```

The noise declaration is optional when measurement noise is unknown; it is not
needed for a fixed-error study. All fitted nuisance parameters enter the joint
posterior and covariance. Priors must be proper continuous univariate
Distributions.jl distributions; discrete or multivariate priors are not supplied
by this scalar interface. There is no implicit improper prior.

Automatic selection uses declared **support**, never sampled values or a prior
mean. It resolves to a concrete transform stored in the specification:

| Physical prior support | Automatic internal coordinates |
|:--|:--|
| Entire real line | `IdentityTransform()` |
| Positive half-line | `LogTransform()` |
| Finite interval | `LogitTransform(lower,upper)` |
| Other lower/upper half-line | `ShiftedLogTransform(bound; upper=false/true)` |

Normal, LogNormal, Exponential, Gamma, Beta, Uniform and eligible truncated
versions are included, along with other registered interval families. An
unclassified/custom or disconnected support requires an explicit compatible
transform; the generic minimum/maximum of a distribution does not certify an
interval. `IdentityTransform()` covers the real line and lets the prior reject
outside-support proposals. Initial values must be finite, in prior support and
representable by the transform. They are never clamped.

Advanced users may override the numerical coordinates:

```julia
g0 = ParameterSpec(:g0; initial=0.4, prior=LogNormal(log(0.4),0.4),
    transform=LogTransform(scale=0.1))
parameterization(estimation) # resolved transforms, physical initials and prior declarations
```

An explicit identity remains distinct from `transform=:auto`. A transform may
cover a larger domain than the prior, but must not silently remove allowed
physical support. For example, a log transform cannot turn an untruncated
Normal into a positive prior; declare the intended truncated distribution.
Additional constraints are checked on physical candidates. Arbitrary constraint
callbacks do not certify a regular boundary optimum or a global support map.

A shared parameter has no experiment name. Add `experiment=:batch_B` to make
a fitted parameter local to that experiment. Put known values in the problem's
fixed tuple or the experiment's fixed tuple. A name cannot be fixed and fitted,
or shadow a shared parameter locally.

An experiment's setup callback constructs its actual initial-value problem:

```julia
setup = function (theta, base)
    u0 = copy(base.u0)
    population(base,u0,:crystals) .*= theta.initial_scale
    remake(base; u0, p=theta)
end
experiment = InferenceExperiment(:batch, base_problem, sizes, concentration;
    setup, dt=0.002)
problem = EstimationProblem(parameters, (experiment,); fixed=(known_value=1.0,))
```

This permits uncertain initial populations, bulk initial states, calibration
factors and noise parameters. A Measurement response callback
`(prediction,theta,experiment)->...` can apply a calibration factor.
Operating histories remain in the physical problem or are rebuilt by setup.
The default setup merges physical parameters into a NamedTuple p.

## Fit the posterior mode

Load Optim in a working environment. For a constructed EstimationProblem:

```julia
using Optim
result = fit(problem) # MAP by default, starting from ParameterSpec initial values
result.parameters
result.diagnostics.gradient_norm

result = fit(problem; nstarts=3, seed=22, scaling=ones(length(problem.parameters)))
# Or supply explicit inference-coordinate vectors:
result = fit(problem; starts=[z_start_1,z_start_2])
```

The default is BFGS with ForwardDiff derivatives of the shared posterior.
The supplied scaling affects optimizer coordinates, not the physical density
being optimized. Convergence also checks the physical gradient in parameter-scaled
units (`diagnostics.stationarity`). Raw internal and optimizer-scaled gradients
remain labelled in diagnostics. A small internal gradient caused by a vanishing
transform derivative at an endpoint does not establish physical stationarity.
Allowed endpoint or singular-density modes are reported as boundary-limited or
nonregular; this interface does not manufacture an interior MAP or a Gaussian
uncertainty there. A seeded perturbation generates extra starts.
Multiple starts improve exploration but do not establish a global optimum.

The selected gradient is checked against step-halved central differences before
fitting and sampling. Those differences verify the derivative; they are not the
default optimizer or NUTS gradient. Limiters can produce nonsmooth objectives;
a local derivative check does not establish global smoothness.

Results retain separate likelihood/prior contributions, convergence, gradients,
objective history, start outcomes, instrumented forward-solve counts and timing.
NumericalFailure rejects an evaluation and records a message. Programming errors
propagate. An external solver must check its success code and return exactly
the requested output times. Invalid derivatives are reported rather than replaced
silently with finite differences.

## Choose how derivatives are calculated

`EstimationProblem(...;derivatives=:auto)` and the legacy CalibrationProblem use
ForwardDiff by default. This propagates parameter derivatives through the
actual calculation, including transforms, the appropriate target-specific prior
terms, observation responses and inferred noise. For a small parameter vector it usually requires one
dual-number solve per experiment, rather than two perturbed solves per parameter.
Larger vectors are processed in ForwardDiff chunks. Evaluation counts describe
these solves, not the number of underlying arithmetic operations.

```julia
posterior_gradient(problem,z) # physical log posterior, differentiated with respect to z
sampling_gradient(problem,z)  # additionally differentiates the coordinate Jacobian
likelihood_gradient(problem,z) # excludes the prior
check_gradient(problem,z; target=:map)
check_gradient(problem,z; target=:sampling)
result.diagnostics.derivative_backend
result.diagnostics.gradient_evaluations
result.diagnostics.forward_solves
```

Callbacks must preserve their numeric input types: use `similar(u)` or
`zeros(eltype(u),...)` for state-dependent storage, and avoid converting fitted
values to Float64. See [ForwardDiff's limitations](https://juliadiff.org/ForwardDiff.jl/dev/user/limitations/).
AD has been checked for the analytical growth observable, native sectional
growth/removal, exactly closed moments, and coupled bulk balances. A fixed grid
is important. The Inguva fitting recipe rebuilds a mesh whose number of cells can
change with parameters; it explicitly requests finite differences. Eigen-based
QMOM inversion is also outside the verified AD path. Select
`derivatives=:finite_difference` for such a calculation and inspect step-size
checks. An incompatible AD callback raises DifferentiationFailure.

For a differentiable ODE residual, the optional adapter instead integrates
continuous forward sensitivities or an interpolating adjoint:

```julia
import SciMLBase
using SciMLSensitivity, OrdinaryDiffEqTsit5
solver = SciMLSensitivitySolver(Tsit5();sensitivity=:auto,
    reltol=1e-8,abstol=1e-10)
experiment = InferenceExperiment(:batch,physical_problem,measurements...;
    setup,solver)
problem = EstimationProblem(parameters,(experiment,))
```

Install these optional packages in the working environment. SciMLSensitivity
loads ReverseDiff, which the adapter uses for the scalar likelihood's state
gradient. No sensitivity packages are loaded for a native fit.
The automatic rule chooses forward sensitivities for at most 16 fitted
coordinates and the adjoint otherwise. This is a starting heuristic, not a cost
guarantee; choose `sensitivity=:forward` or `:adjoint` and compare timings for the
actual problem. Both include parameter-dependent initial states, shared and
experiment-specific parameters, and direct observation/noise dependence.
Correlations across measurement times enter one joint likelihood.

Forward sensitivities integrate ``\partial u/\partial z`` alongside the state.
The adjoint retains a dense forward trajectory, then integrates backward with
measurement contributions. It avoids an augmented state for every parameter,
but the current adapter forms local residual Jacobians and retains the
trajectory; it is not a memory-free or matrix-free implementation. Tighten
solver tolerances and check gradients when changing the residual or integrator.
ODE adapters do not enforce the native CFL or positivity stage checks.
Event callbacks are rejected because sensitivity jump rules are not implemented;
prescribed time histories can use fixed `tstops` instead.
Direct cell-shift methods are not ODE residuals and cannot use this adapter.

This uses the [direct forward](https://docs.sciml.ai/SciMLSensitivity/stable/tutorials/direct_sensitivity/)
and [direct adjoint](https://docs.sciml.ai/SciMLSensitivity/stable/manual/direct_adjoint_sensitivities/)
interfaces. Merely adding `sensealg` to an ODE solve differentiated with
ForwardDiff does not select an adjoint. The adapter explicitly invokes the
sensitivity equations while reusing the package's likelihood.

## Request sensitivity-based uncertainty

For fixed-error Gaussian measurements, a useful first calculation is a local
Gauss–Newton approximation at the MAP:

```julia
using PopulationBalanceModels, Optim
result = fit(problem) # MAP only; the statistical target is unchanged
local_gaussian = gauss_newton_approximation(problem,result)
# Equivalent opt-in convenience:
result = fit(problem; uncertainty=:gauss_newton)
kinetics = marginal_covariance(problem,local_gaussian; role=:kinetic)
draws = parameter_draws(problem,local_gaussian; n=1000,seed=23)
```

Let ``r(\theta)`` contain all observation residuals after covariance whitening and
any independent contrasts. The response, times, experiments, parameter sharing
and uncertain initial conditions are the same as in the likelihood. Then

```math
J_\theta=\left.\frac{\partial r}{\partial\theta}\right|_{\widehat\theta},\qquad
P_{\mathrm{GN}}=J_\theta^\mathsf TJ_\theta+
\left.\frac{\partial^2[-\log\pi_\theta(\theta)]}{\partial\theta\,\partial\theta^\mathsf T}
\right|_{\widehat\theta},\qquad
\Sigma_{\mathrm{GN}}=P_{\mathrm{GN}}^{-1}.
```

This is a **Gaussian in physical parameter coordinates**. The prior curvature
is that of the physical density, including the inverse Jacobian only for an
explicit advanced internal-coordinate prior. Numerical-transform Jacobians do
not enter the MAP objective or this covariance. No averaging
over bins/times or residual-based noise rescaling is introduced. Factorizations
and solves produce the covariance. A kinetic marginal uses the relevant block
of the full joint covariance, including calibration and initial-state nuisance
parameters; it does not fix those parameters at their MAP values.
The [Ceres covariance guide](https://ceres-solver.readthedocs.io/latest/nnls_covariance.html)
explains the measurement-weighted sensitivity construction for least squares;
the prior term above extends that local construction to this declared posterior.

Initial support is `GaussianLikelihood` with fixed scalar/array standard
deviations or fixed covariance matrices, including projected fraction covariance.
Noise callbacks, inferred covariance, count likelihoods and truncated Gaussians
are rejected explicitly: their information cannot be obtained just by weighting
the mean-prediction Jacobian. Use full-posterior Laplace or MCMC for those targets.
The legacy deterministic discrepancy blocks are not statistical uncertainties.

ForwardDiff computes prediction Jacobians by default. The independently
step-halved checks also verify prior curvature and the assembled precision in
its covariance-scaled directions, so a large error confined to a strong direction
is not confused with unresolved weak curvature. An information discrepancy
``e<1`` bounds the checked candidate covariances between
``C_{\mathrm{GN}}/(1+e)`` and ``C_{\mathrm{GN}}/(1-e)``. This checks derivatives
of the **chosen discrete model**, not continuum accuracy or the omitted Hessian
terms. Grid, domain and time refinement remain separate checks.
Independent residual/prior gradients are also checked in the covariance metric;
small Jacobian errors are not allowed to hide cancellation in ``J^\mathsf Tr``
near the mode. These checks do not rescale errors by the observed residual size.

`GaussNewtonFailure.report` records nonstationarity, unresolved sensitivities,
nonpositive precision or excessive conditioning. No ridge or pseudoinverse
inserts apparent certainty. Data rank and prior curvature are reported
separately: a proper prior can support a data-null direction without making it
identified by measurements. An external solver must support the prediction
Jacobian's AD path. This does not convert an adjoint scalar-gradient callback
into a prediction Jacobian; `derivatives=:finite_difference` is an explicit
alternative, subject to the same checks.

Parameter draws cost no PBM solves. They are physical Gaussian draws: `theta`
and `physical` retain every value, including negative or out-of-support values.
`in_support` and `transform_valid` report validity; invalid inverse transforms
are not attempted (`z` is NaN for those columns). No draw is clipped, filtered,
resampled or silently truncated. `posterior_predictive`
still needs its explicit solve budget. Label resulting intervals **Gauss–Newton
approximations**. The construction omits ``\sum_i r_i\nabla^2r_i`` from the full
posterior Hessian; it need not capture bounds, curved compensation, multiple
basins or systematic model error. Positive information and optimizer convergence
also do not independently certify a minimum of the full posterior.

## Request full-Hessian Laplace uncertainty

```julia
laplace = laplace_approximation(problem,result)
# Equivalent opt-in convenience:
result = fit(problem; uncertainty=:laplace)
kinetics = marginal_covariance(problem,laplace; role=:kinetic)
draws = parameter_draws(problem,laplace; n=1000,seed=23)
```

The Hessian uses nested ForwardDiff on the actual negative log posterior.
Independent step-halved differences of its AD gradient verify the curvature:

```math
H_\theta=\left.\frac{\partial^2[-\ell_\theta(\theta)]}{\partial\theta\,\partial\theta^\mathsf T}
\right|_{\widehat\theta},\qquad
p(\theta\mid y)\approx\mathcal N(\widehat\theta,H_\theta^{-1}).
```

All nuisance parameters enter this joint curvature. A kinetic marginal uses
the corresponding block of the full covariance; inverting the kinetic Hessian
subblock would instead condition on fixed nuisance parameters.
With a SciML sensitivity adapter, this second derivative passes through the
primal ODE solve; it does not differentiate a first-order adjoint. An explicitly
selected finite-difference backend also uses finite-difference curvature.

Factorizations and solves form the covariance and draws. Nonconvergence,
nonstationarity, unstable derivatives, nonpositive curvature or excessive
conditioning reject the approximation with a LaplaceFailure report. No
regularization converts an invalid Hessian into reported certainty.
Positive local curvature does not establish unimodality or Gaussian tails.
Centers, Hessians, covariances, marginal blocks and Gaussian draws are in
physical parameter units. Internal verification may use an affine scaling of
physical values; it does not replace the density with an unconstrained one.
Intervals and parameter regions are labelled physical-coordinate Laplace
approximations. A Gaussian may extend outside physical support.
Draws include an in_support flag. Predictive propagation reports draws outside
the declared prior/constraint support as failures; it does not silently truncate
the Gaussian approximation.

The retained growth case study compares lognormal growth priors with physical
uniform priors bounded by stated growth timescales. These choices express
different assumptions; neither is a universal default. Both are shown here,
and their fitted posteriors are compared below.

![Physical lognormal and timescale-bounded uniform growth priors](assets/bayesian_growth/priors.png)

## Sample the same posterior

Sampling is requested separately and requires AdvancedHMC and
MCMCDiagnosticTools in the working environment. Continuing the first fit above:

```julia
using AdvancedHMC, MCMCDiagnosticTools
samples = sample_posterior(estimation; initial=fitted,
    chains=4, warmup=500, seed=24, verbose=true)
report = diagnostics(samples)
(report.stopping_reason, report.samples, report.diagnostics_passed)
report.check_history
```

The default warms up each of four chains once, freezes adaptation, then continues
those same chains in batches of 250 retained draws. Checks use **all retained
chains jointly**. After at least 500 draws per chain, two successive checkpoints
must pass rank-normalized split R-hat ≤ 1.01, bulk and tail ESS ≥ 400, BFMI ≥ 0.3,
finite mean MCSE, and zero retained divergences or maximum-depth hits. There is
a finite budget of 10,000 draws per chain. Set `min_samples`, `batch_samples`,
`max_samples` and `consecutive_passes` to change this policy. These are draw
budgets, not limits on elapsed time or PBM evaluations.

`stopping_reason` records `:diagnostics_passed`, `:max_samples` or
`:sampler_pathology`. Exhausting the budget without the required consecutive
passes returns `diagnostics_passed=false`. A retained divergence or depth hit
ends diagnostic mode unsuccessfully: appending draws cannot undo that event.
The draws and failed checks remain available; the adapter does not discard them
or restart adaptation. Inspect the diagnostics and change tuning deliberately
before starting a new run. Warmup divergences are reported separately.

For an explicitly fixed run, supply a count:

```julia
samples = sample_posterior(estimation; initial=fitted,
    stopping=:fixed, samples=1000, chains=4, warmup=500, seed=24)
```

Existing calls with `samples=N` also select fixed mode and retain that sampling
policy. The Jacobian-corrected target represents the same physical posterior;
bitwise transition sequences across code changes are not promised. They report diagnostics but do not stop early. Fixed mode
allows one chain; diagnostic stopping requires at least two. Four dispersed
chains are a useful default, especially when different plausible modes need to
be explored. Joint chains are currently advanced serially. Batching preserves
the adapted kernel, phase point and private RNG; it is not a persistent
checkpoint/restart format.

R-hat and ESS assess mixing, not a specified accuracy for every reported
quantity. Optionally monitor cheap scalar functions of physical parameters and
require absolute Monte Carlo precision:

```julia
samples = sample_posterior(estimation; initial=fitted, seed=24,
    diagnostic_quantities=(rate=theta -> theta.shared.k,),
    mcse_limits=(rate=(mean=0.005, quantiles=(0.025=>0.01, 0.975=>0.01)),))
diagnostics(samples).quantity_diagnostics.rate
```

These functions receive `(shared, experiments)` and must return deterministic,
finite real scalars without additional PBM solves. Every monitored quantity
also enters the R-hat/ESS screen. MCSE limits have the quantity's units;
quantile MCSE measures uncertainty in an estimated interval endpoint, **not**
the posterior interval width. Constant or unresolved diagnostics cannot pass.
Repeated checks are a computational stopping rule, not proof of global mixing,
a rigorous sequential-coverage guarantee, identifiability or model adequacy.
Inspect traces and posterior shape before interpreting intervals.

NUTS uses the gradient of `sampling_logposterior`, not the physical-MAP
objective alone. A supplied physical `laplace` covariance is locally converted
through the inverse parameter Jacobian, and through any auxiliary sampling map,
for metric initialization and dispersed starts. Diagnostics label this conversion
as **tuning**; it is not another posterior covariance estimate. Otherwise
adaptation is diagonal by default.
An explicit positive-definite `initial_covariance` also selects dense adaptation.
It is tuning, not a verified uncertainty estimate. Supply one matrix or a
tuple/vector of matrices for different chain starts. A Laplace approximation can
also inform dispersed starts; explicit inference-coordinate `starts` are
accepted. Warmup draws are excluded from retained diagnostics.

The report retains warmup and post-warmup statistics, acceptance rates, adapted
inverse masses, step sizes, divergences, depth hits, gradient checks, seeds,
backend versions, evaluation counts and elapsed time. `mcse_mean` concerns
inference-coordinate means. Use `diagnostic_quantities` for physical-coordinate
precision. `diagnostics(z, statistics; ...)` can reassess existing draws with
the same joint screen; it does not apply the controller's minimum-length or
consecutive-pass rules.

Strongly correlated parameters can benefit from a bijective `sampling_map`
with `to_sampler`, `from_sampler`, `logabsdetjac` and `label` fields. Here
`logabsdetjac(y)` is the log absolute determinant of `from_sampler` at `y`.
The adapter samples
``\log p_z(\mathrm{from\_sampler}(y))+\log|\det J_{\mathrm{from\_sampler}}(y)|``.
The inverse and determinant are checked at every mapped target evaluation.
`check_sampling_map(problem,map,z)` also checks both inverse directions and the
transformed gradient at a supplied point in the **original** posterior support.
Use it at representative points and tails. The `step` argument controls the
independent finite-difference verification stencil; check its halving where
limiter switches or sharp likelihood curvature affect the comparison. Finite
tests cannot prove that arbitrary callbacks are globally bijective or cover the
whole posterior support; that remains a requirement of the supplied map.
The prior probability laws remain unchanged. Returned `z` draws and
`statistics.log_density` refer to the internal-coordinate sampling density,
including the parameter-transform Jacobian exactly once. `sampler_log_density`
also includes the auxiliary-map Jacobian. A tuning
matrix supplied with a map must be expressed in **sampler** coordinates.
AdvancedHMC's `DenseEuclideanMetric` takes the inverse mass matrix: a supplied
`initial_covariance` initializes that matrix, following its
[metric convention](https://turinglang.org/AdvancedHMC.jl/stable/api/).
Hamiltonian energies, BFMI and transition statistics remain those of the
sampler coordinates, including the map Jacobian. They are not relabelled as
inference-coordinate energies or covariance estimates.
The [protein tutorial](case_studies/protein_crystallization.md) uses a
unit-determinant growth-rate shear for its three-/four-parameter targets with
fixed interfacial energy. Earlier four-/five-parameter targets used different
nonlinear maps and are preserved as historical results.
Archived recipes have different statistical targets; their inputs and producing
records are retained with the
[historical fixtures](https://github.com/pavaninguva/PopulationBalanceModels.jl/blob/main/test/fixtures/archives/README.md).

See the upstream [AdvancedHMC](https://turinglang.org/AdvancedHMC.jl/stable/get_started/)
and [MCMCDiagnosticTools](https://turinglang.org/MCMCDiagnosticTools.jl/stable/)
documentation for sampler and diagnostic definitions. NUTS reuses the selected AD or ODE-sensitivity backend and adds the required
coordinate-Jacobian gradient; diagnostics record the target and backend.

The [growth-estimation case](case_studies/estimating_growth.md) compares priors,
Laplace intervals and NUTS draws for synthetic density measurements. The
[protein case](case_studies/protein_crystallization.md) illustrates coefficient–
exponent tradeoffs and predictive mismatch despite a passed sampling screen.

MCMC `parameter_draws` also returns `indices` into
`reshape(samples.z, number_of_parameters, :)`, with draws varying before chains.
Retain these identities when exporting predictive curves. A seed alone need not
reproduce `randperm` selection across Julia versions; the historical protein
example exports `posterior_predictive_indices.csv` for that purpose. Sampling tuning
matrices and transformed Hamiltonian energies retain their separate meanings.

## Keep the density coordinates explicit

This advanced section states the change of variables; ordinary declarations need
only physical initials and distributions. Let ``\theta=T(z)``. The physical
posterior is

```math
\ell_\theta(\theta)=\sum_{e,b}\log L_{e,b}(\theta)
+\sum_j\log\pi_j(\theta_j).
```

Each ParameterSpec contributes its prior once, regardless of the number of
experiments or blocks. Optimization over ``z`` maximizes
``\ell_\theta(T(z))`` **without** a coordinate correction. Sampling instead uses

```math
\ell_z(z)=\ell_\theta(T(z))+\log|\det DT(z)|.
```

With an auxiliary ``z=S(y)``, the sampler adds ``\log|\det DS(y)|`` once more.
These changes preserve probabilities, while modes depend on the reference density.

| Function | Argument | Returned density / derivative |
|:--|:--|:--|
| `loglikelihood(problem,z)` | Internal z | Likelihood only |
| `logprior`, `logposterior`, `posterior_components` | Internal z by default | Physical density at T(z) |
| Same functions with `coordinates=:physical` | Ordered physical vector | Physical density |
| `posterior_gradient(problem,z)` | Internal z | Derivative of the physical-density objective with respect to z |
| `sampling_logprior`, `sampling_logposterior`, `sampling_gradient` | Internal z | Density with respect to dz, or its gradient |

`Prior(d; coordinates=:physical)` remains equivalent to a direct distribution.
`Prior(d; coordinates=:inference)` is advanced and requires an explicit transform.
It describes a law for z: physical MAP uses
``\log\pi_\theta(\theta)=\log\pi_z(T^{-1}(\theta))-\log|\det DT|``.
Sampling restores the forward Jacobian, leaving that declared law intact.
For example a Normal prior on log(k) implies a physical LogNormal, not a Normal
prior on k. An explicit identity is a valid advanced declaration too. For an
advanced inference prior, changing the transform while holding its z-density
fixed changes the implied physical law; transform invariance applies when the
physical prior law is held fixed.

A custom `gradient(z)` must declare `gradient_target=:physical` or `:sampling`.
Both mean derivatives with respect to z; the package adds/subtracts the Jacobian
gradient for the other target. Both targets have independent finite-difference
checks. An old callback is never silently used for a different density.

## Migration and retained results

- Prefer `ParameterSpec(:k; initial=p.k, prior=LogNormal(...))`.
  Legacy physical Prior wrappers remain valid. Automatic transforms can differ
  from the old identity default; use `initial_coordinates` rather than assuming
  that a start vector contains physical values.
- Existing inference-coordinate priors must specify their transform explicitly.
  Explicit transforms must cover the declared physical prior support; choose an
  explicitly truncated prior when that is the intended physical domain.
- MAP components and local covariances now use `physical_map_v1`. Older modes,
  full-Hessian Laplace and Gauss–Newton results require recomputation. A native
  result with unspecified/old convention is rejected for new uncertainty or
  MAP-based sampler initialization, rather than silently relabelled.
- MCMC draws may remain scientifically valid when the forward model, observations,
  likelihood and physical prior law are unchanged. Their original producer and
  compatibility records remain authoritative. This refactor alone is not a
  current-source replay receipt.
- The retained growth and protein figures/tables still use their recorded
  inference-density MAP/local approximations. Their historical labels remain;
  current scripts use the new convention and must write a new output directory.
  No production fits or long posterior chains were rerun for this migration.

## Check prediction, identifiability and numerical error

Numerical-refinement and prior-sensitivity comparisons retain data, times,
likelihood settings, fixed parameters, constraints and measurement responses.
Changing a noise scale is a statistical-model change, not numerical refinement.
The checks cannot establish that arbitrary model/setup callbacks describe the
same physical initial conditions or observable; keep those definitions fixed
when changing the solver, grid or domain.

Drawing approximate parameters does not solve the PBM. Predictive propagation
has an explicit forward-solve budget:

```julia
draws = parameter_draws(problem,samples; n=40,seed=25)
prediction = posterior_predictive(problem,draws;
    max_solves=40, measurement_noise=true, seed=26)
```

One experiment per draw requires one solve; additional experiments multiply
the cost. The budget is checked before execution. Returned blocks separate raw
model outputs, latent measurement means and fresh measurement noise. Failures
remain visible by draw. Poisson predictions include known exposure; multinomial
predictive counts condition on the observed snapshot totals.

identifiability_report separates likelihood prediction sensitivities and
observed likelihood curvature from physical-prior curvature. By default its
parameter derivatives are physical; `coordinates=:inference` is explicit. Prediction rank uses the
unscaled measurement units and omits noise-only information; inferred noise
also enters the likelihood Hessian. These are local diagnostics, not proof of
structural identifiability.

```julia
likelihood = estimation_profile(problem,result,index,values; target=:likelihood)
posterior  = estimation_profile(problem,result,index,values; target=:posterior)
```

Each profile reoptimizes every remaining coordinate, including nuisances.
The likelihood profile obtains its own unpenalized reference optimum.
Posterior profiles describe physical-posterior density shape (their supplied
profile grid still uses internal coordinates) and have no ordinary
likelihood-confidence threshold. Check every conditional convergence flag.

replace_priors retains the physical experiment and changes explicit prior
declarations. prior_sensitivity fits named alternatives and reports mode shifts
in reference posterior SD units and changes in Laplace uncertainty. Its
prior_sensitive flag uses explicit, configurable shift/scale-ratio thresholds;
it is a screening rule, not an identifiability theorem.
Prior dependence is evidence to examine, even when MAP is stable.

For the growth study, the residuals below apply the fitted truncated-Gaussian
CDF to each measurement, then the inverse standard-normal CDF. This accounts
for the asymmetric errors near zero density. The quantile plot compares these
normal quantile residuals with a standard Gaussian. The final panel shows how much
the rate estimates change when lognormal priors are replaced by uniform priors.

![Truncated-Gaussian quantile residuals and sensitivity to physical prior choices](assets/bayesian_growth/checks.png)

refinement_study refits explicitly named numerical variants. Keep measurements,
parameter coordinates and priors consistent. Compare spatial resolution, size
domain and time step separately; compare moment closures separately again,
because closure error is a model approximation. The growth case study records
such refits alongside reference posterior uncertainty. Parameter uncertainty,
measurement noise, numerical error and physical model inadequacy remain
different sources of discrepancy.

The growth example refits with Inguva2022 and MUSCL. The first method requires
a matched travel-time mesh and time step, so they must be refined together.
The MUSCL checks refine space, time and domain separately. The analytical
solution supplies the reference; interpolating bin averages to point density
measurements adds a separate observation-reconstruction error.

![Growth-parameter shifts and prediction changes under grid, time-step, and size-domain refinement](assets/bayesian_growth/refinement.png)

## Existing Gaussian calibration interface

A deterministic comparison is available when no probabilistic observation
model is declared. `ResidualObservation` contributes
`weight * mean(((prediction-data)/scale).^2)`; `WassersteinObservation`
integrates squared differences between conditional number quantiles. The
comparison scales are not sensor standard deviations. These blocks use
`CalibrationProblem` and the optional Optim extension, but cannot be mixed
with Gaussian observations or used to manufacture confidence intervals.
`loss_contributions` separates their contributions; `sensitivity_report`
reports local sensitivity rank without a covariance estimate.


CalibrationProblem, Observation, Experiment and fit(calibration,z0) remain
available for existing code. With `Observation` blocks they implement the
Gaussian maximum likelihood workflow. `ResidualObservation` instead selects
the explicit deterministic discrepancy described above. `confidence_ellipsoid`
is the Gaussian workflow's local frequentist calculation and rejects discrepancy
weights. It is separate from the Bayesian Laplace API. The older small kinetic examples remain
regression examples. New Bayesian work should use EstimationProblem so all
inference methods share priors, transforms and likelihood definitions.
