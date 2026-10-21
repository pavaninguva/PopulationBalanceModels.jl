# Observations and Bayesian parameter estimation

A measured size distribution depends on the kinetics, the initial population,
and the instrument. Estimating a growth rate therefore means fitting a complete
measurement model. A concentration trace can provide different information from
a normalized size histogram, even when both observe the same batch.

The package uses one statistical model for three calculations: MAP estimation,
an explicitly requested Laplace approximation, and optional posterior sampling.
Start with the [size-dependent growth example](case_studies/estimating_growth.md).
It fits two growth coefficients from dynamic distribution measurements, together
with uncertain initial population amplitude and concentration noise.

## Describe the measurements

An observation operator maps a solved PBM state to a predicted measurement.
The existing MomentObservable, BulkObservable, FunctionObservable and
SystemObservable can all be used in a Measurement. For a sectional distribution,
DistributionObservable integrates over the **measurement** bins:

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

If moments were calculated from the measured histogram, they share its
measurement errors. Use the histogram alone, or a joint vector observable with
the appropriate joint covariance; do not count both as independent evidence.
Use the same source tag on dependent blocks: InferenceExperiment rejects duplicate
tags. Untagged dependence cannot be inferred automatically.

## Declare fitted and fixed parameters

Each scalar declaration states its physical initial value, transform and prior:

```julia
using Distributions
g0 = ParameterSpec(:g0; initial=0.4, transform=LogTransform(),
    prior=Prior(LogNormal(log(0.4),0.4); coordinates=:physical))
noise = ParameterSpec(:noise; initial=0.003, transform=LogTransform(),
    role=:nuisance,
    prior=Prior(Normal(log(0.003),0.4); coordinates=:inference))
```

Both priors above imply Gaussian log coordinates; the coordinate declaration
makes their density measures unambiguous. There is no implicit prior-coordinate
default. IdentityTransform supports a location and scale; LogTransform supports
a positive reference scale. LogitTransform(lower,upper) gives an open bounded
physical interval. Additional physical restrictions belong in the estimation
problem's constraints callback; truncating admissible support changes the prior.

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

## Keep the density coordinates explicit

The inference vector is ``z``, with physical parameters ``\theta=T(z)``.
For each physical-coordinate prior the package evaluates

```math
\log p_z(z)=\log p_\theta(T(z))+\log|\det J_T(z)|.
```

An inference-coordinate prior needs no additional Jacobian. This adjustment
occurs exactly once in logprior. MAP, Hessian calculations and NUTS all use

```math
\log p(z\mid y)=\sum_e\log p(y_e\mid T(z))+\log p_z(z)+\text{constant}.
```

The functions loglikelihood, logprior and logposterior expose these separate
evaluations. posterior_components returns all three at one coordinate vector.
Returned physical parameter values are the transformed z-density mode.
Modes depend on density coordinates: transforming that mode does not generally
give the mode of the physical-coordinate density. Every result records its
parameter declarations and coordinates.

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

The reference method is BFGS with central finite-difference derivatives.
The supplied scaling affects the optimizer coordinates, not the statistical
density being optimized. Convergence uses the gradient in those optimizer
coordinates; diagnostics retain both raw z gradients and scaled gradients,
together with the requested tolerance. Choosing scales is therefore part of
the numerical specification. A seeded perturbation generates extra starts.
Multiple starts improve exploration but do not establish a global optimum.

A supplied posterior-gradient callback is checked against step-halved central
differences. Solver/representation combinations are not presumed to support
automatic differentiation. The verified combinations include affine-growth
characteristics, the native moment reference, a specified sectional MUSCL
calculation, and a linear Gaussian reference. Limiters can produce nonsmooth
objectives; a local derivative check does not establish global smoothness.

Results retain separate likelihood/prior contributions, convergence, gradients,
objective history, start outcomes, instrumented forward-solve counts and timing.
NumericalFailure rejects an evaluation and records a message. Programming errors
propagate. An external solver must check its success code and return exactly
the requested output times. Gradient stencils crossing failed solves are reported.

## Request Laplace uncertainty

```julia
laplace = laplace_approximation(problem,result)
# Equivalent opt-in convenience:
result = fit(problem; uncertainty=:laplace)
kinetics = marginal_covariance(problem,laplace; role=:kinetic)
draws = parameter_draws(problem,laplace; n=1000,seed=23)
```

The Hessian is computed from the actual negative log posterior and checked
under step halving:

```math
H=\left.\frac{\partial^2[-\log p(z\mid y)]}{\partial z\,\partial z^\mathsf T}
\right|_{\widehat z},\qquad
p(z\mid y)\approx\mathcal N(\widehat z,H^{-1}).
```

All nuisance parameters enter this joint curvature. A kinetic marginal uses
the corresponding block of the full covariance; inverting the kinetic Hessian
subblock would instead condition on fixed nuisance parameters.

Factorizations and solves form the covariance and draws. Nonconvergence,
nonstationarity, unstable derivatives, nonpositive curvature or excessive
conditioning reject the approximation with a LaplaceFailure report. No
regularization converts an invalid Hessian into reported certainty.
Positive local curvature does not establish unimodality or Gaussian tails.
Intervals and parameter regions from these draws are labelled Laplace
approximations and are transformed to physical coordinates explicitly.
Draws include an in_support flag. Predictive propagation reports draws outside
the declared prior/constraint support as failures; it does not silently truncate
the Gaussian approximation.

## Sample the same posterior

Sampling is requested separately and requires AdvancedHMC and
MCMCDiagnosticTools in the working environment:

```julia
using AdvancedHMC, MCMCDiagnosticTools
samples = sample_posterior(problem; initial=result, laplace,
    warmup=500, samples=1000, chains=4, seed=24)
samples.diagnostics
```

The adapter uses NUTS and explicit posterior gradients. With a supplied Laplace
approximation it adapts a dense metric initialized from its covariance;
otherwise it uses a diagonal metric. Laplace can inform dispersed chain starts
and initial scale; explicit
starts are also accepted. Warmup draws are excluded from the retained sample.
Diagnostics include rank-normalized split R-hat, bulk and tail effective sample
sizes, divergences, warmup divergences, maximum-depth hits, gradient checks,
evaluation counts and time. diagnostics_passed uses the reported thresholds;
it is a screening flag, not a proof of convergence. A single chain cannot pass
that screen. Inspect chains and posterior shape before interpreting results.

See the upstream [AdvancedHMC](https://turinglang.org/AdvancedHMC.jl/stable/get_started/)
and [MCMCDiagnosticTools](https://turinglang.org/MCMCDiagnosticTools.jl/stable/)
documentation for sampler and diagnostic definitions. Finite differences cost
repeated forward solves; adjoint or AD acceleration remains a separately
verified extension opportunity.

## Check prediction, identifiability and numerical error

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
observed likelihood curvature from prior curvature. Prediction rank uses the
unscaled measurement units and omits noise-only information; inferred noise
also enters the likelihood Hessian. These are local diagnostics, not proof of
structural identifiability.

```julia
likelihood = estimation_profile(problem,result,index,values; target=:likelihood)
posterior  = estimation_profile(problem,result,index,values; target=:posterior)
```

Each profile reoptimizes every remaining coordinate, including nuisances.
The likelihood profile obtains its own unpenalized reference optimum.
Posterior profiles describe penalized posterior shape and have no ordinary
likelihood-confidence threshold. Check every conditional convergence flag.

replace_priors retains the physical experiment and changes explicit prior
declarations. prior_sensitivity fits named alternatives and reports mode shifts
in reference posterior SD units and changes in Laplace uncertainty. Its
prior_sensitive flag uses explicit, configurable shift/scale-ratio thresholds;
it is a screening rule, not an identifiability theorem.
Prior dependence is evidence to examine, even when MAP is stable.

refinement_study refits explicitly named numerical variants. Keep measurements,
parameter coordinates and priors consistent. Compare spatial resolution, size
domain and time step separately; compare moment closures separately again,
because closure error is a model approximation. The growth case study records
such refits alongside reference posterior uncertainty. Parameter uncertainty,
measurement noise, numerical error and physical model inadequacy remain
different sources of discrepancy.

## Existing Gaussian calibration interface

CalibrationProblem, Observation, Experiment and fit(calibration,z0) remain
available for existing code. They implement the previous Gaussian maximum
likelihood workflow. confidence_ellipsoid is its local frequentist calculation;
it is not the Bayesian Laplace API. The older small kinetic examples remain
regression examples. New Bayesian work should use EstimationProblem so all
inference methods share priors, transforms and likelihood definitions.
