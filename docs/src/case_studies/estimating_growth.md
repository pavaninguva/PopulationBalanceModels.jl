# Estimating size-dependent growth

A seed population moves toward larger sizes as crystals grow. A single snapshot
can confound the overall growth speed with its dependence on size. Following
the changing distribution at several times gives more information. A separate
concentration measurement can also constrain how much solid is present.

This example generates synthetic measurements, fits growth coefficients and two
nuisance parameters, then compares MAP, Laplace uncertainty and MCMC. The growth
problem and initial distribution come from Case 2 of
[Inguva, Schickel and Braatz (2022)](https://doi.org/10.1016/j.compchemeng.2022.107808).
The sampling experiment and scaled bulk balance are additions for this inference
demonstration. No experimental observations or published parameter estimates
are being reproduced.

## The population and material balances

Let a be the intrinsic size coordinate. The population equation is

```math
\frac{\partial f(a,t)}{\partial t}
+\frac{\partial[(g_0+g_1a)f(a,t)]}{\partial a}=0,
\qquad a\ge0.
```

The paper specifies

```math
f(a,0)=50\exp\left[-\frac{(a-0.2)^2}{0.0005}\right],
\qquad g_0=0.434,\quad g_1=0.2604.
```

We retain these scaled size and time coordinates. For fixed units, g₀ has
size/time units and g₁ has inverse-time units. There is no nucleation or inflow.
For the numerical calculation, both size boundaries are closed and the upper
boundary must remain unpopulated. This differs from the paper's outflow-boundary
discretization; domain enlargement checks whether the difference matters here.

We add an uncertain seed amplitude A and a scaled dissolved concentration:

```math
f(a,0;A)=A f_0(a),\qquad
\frac{dc(t)}{dt}=-q\frac{d\mu_3(t)}{dt},\qquad
\mu_3(t)=\int_0^\infty a^3f(a,t)\,da,
```

with c(0)=2 and q=0.2 in the chosen scaled material units. This is a controlled
growth experiment: the growth law is prescribed and does not depend on c.
The invariant is c+qμ₃. We use the concentration as an independent observation
of material uptake; it is not a moment calculated from the measured histogram.

The executable builder uses the normal term and coupling interfaces:

```julia
model = PBMModel(growth=Growth(rate=(a,p,t,ctx)->p.g0+p.g1*a))
operator = compile(model,grid; methods=TermMethods(growth=MUSCL()),
    bc=SizeBC(NoFlux(),NoFlux()))
system = PBMSystem((crystals=operator,); nonnegative_bulk=(:c,),
    exchanges=(ConservativeExchange(:crystals,:c,
        (a,p,t,ctx)->0.2*a^3; mechanisms=(:growth,)),))
```

## Generate measurements independently of the sectional solver

The size characteristics are

```math
a(t)=a_0e^{g_1t}+\frac{g_0}{g_1}(e^{g_1t}-1),\qquad
a_0(a,t)=ae^{-g_1t}+\frac{g_0}{g_1}(e^{-g_1t}-1).
```

For a characteristic entering from a negative initial size, the density is zero.
Elsewhere,

```math
f(a,t)=e^{-g_1t}A f_0(a_0(a,t)).
```

Integrating f over a measurement bin is equivalent to integrating f₀ between
its inverse-characteristic edges, restricted to a₀≥0. The recipe uses Gaussian
CDF integrals and stable tail differences. Initial moments use the Gaussian
truncated at zero; affine-growth moment equations provide another independent
time-integration check. This case-specific solution does not extend the package's
general ExactShift algorithm to arbitrary variable growth.

At each of six times from 0.1 to 1, sample 400 particles conditional on the
measurement window [0,1.4]. Separately add independent Gaussian concentration
noise with true standard deviation 0.003. Random seeds are fixed and recorded.

```@example growth_inference
using PopulationBalanceModels, Distributions
include(joinpath(pkgdir(PopulationBalanceModels),"examples","estimate_size_dependent_growth.jl"))
import .EstimateSizeDependentGrowth
const recipe = EstimateSizeDependentGrowth
data = recipe.synthetic_measurements()
problem = recipe.estimation_problem(data)
z0 = initial_coordinates(problem)
posterior_components(problem,z0)
```

The synthetic distributions and fitted bin-integral probabilities are shown
below. Histogram heights are divided by the measurement-bin widths.

![Synthetic dynamic particle counts and fitted size densities](../assets/bayesian_growth/distributions.png)

## Fit the growth law and nuisance parameters

The inferred physical parameters are g₀, g₁, A, and σc, all represented by
logarithmic coordinates. The priors are physical-coordinate LogNormal densities:

| Parameter | LogNormal location | LogNormal scale | Role |
|:--|--:|--:|:--|
| g₀ | log(0.4) | 0.4 | growth |
| g₁ | log(0.25) | 0.6 | size dependence |
| A | 0 | 0.2 | initial-condition nuisance |
| σc | log(0.003) | 0.4 | noise nuisance |

For example, the growth intercept is declared as

```julia
ParameterSpec(:g0; initial=0.4, transform=LogTransform(),
    prior=Prior(LogNormal(log(0.4),0.4); coordinates=:physical))
```

The experiment setup scales the initial population by A. The measurement
declarations are

```julia
sizes = Measurement(data.times,data.counts,
    DistributionObservable(data.edges;coordinate=:length,population=:crystals);
    likelihood=MultinomialLikelihood(),source=:particle_counter)
bulk = Measurement(data.times,data.bulk,BulkObservable(:c);
    likelihood=GaussianLikelihood(sigma=(theta,experiment)->theta.noise),
    source=:independent_concentration_instrument)
```

The multinomial likelihood supplies shape information, conditional on the
sampled counts. It cannot determine A by itself. The independent concentration
observations and declared prior constrain A. The Gaussian determinant changes
with σc and remains in the likelihood.

```julia
using Optim
result = fit(problem;nstarts=3,seed=22,uncertainty=:laplace)
result.parameters.shared
kinetic_covariance = marginal_covariance(problem,result.uncertainty)
```

The covariance includes A and σc before extracting its kinetic marginal. A
strong g₀–g₁ correlation reflects the limited size range traversed by the seeds;
neither optimizer convergence nor a bounded posterior establishes independent
identification of the two growth coefficients.

![Growth-law uncertainty, parameter correlations and marginal distributions](../assets/bayesian_growth/posterior.png)

The following summaries come from the recorded run. The intervals are marginal
equal-tail MCMC credible intervals, conditional on this model, data and priors.
The MAP column is the physical transform of the log-coordinate posterior mode.

```@eval
using Markdown, PopulationBalanceModels
folder=joinpath(pkgdir(PopulationBalanceModels),"docs","src","assets","bayesian_growth")
rows=readlines(joinpath(folder,"credible_summary.csv"))[2:end]
table=["| Parameter | MAP | MCMC mean | MCMC 95% interval |",
       "|:--|--:|--:|:--|"]
for row in rows
    fields=split(row,',')
    values=round.(parse.(Float64,fields[2:5]);sigdigits=5)
    push!(table,"| $(fields[1]) | $(values[1]) | $(values[2]) | [$(values[3]), $(values[4])] |")
end
Markdown.parse(join(table,'\n'))
```

Here the Laplace and MCMC intervals for the growth coefficients are similar,
while the noise parameter has a longer upper tail under MCMC: its 95% upper
endpoint is about 0.00425, compared with 0.00340 from the Laplace draws. Six
concentration observations leave appreciable uncertainty in the noise scale.
This is a useful distinction between local curvature and the sampled posterior,
even when a Laplace Hessian passes all numerical checks.

## Compare local uncertainty with sampling

The default runner uses four NUTS chains, each with 500 warmup iterations and
1000 retained draws. Starts are dispersed using the Laplace scale. MAP, Laplace
and NUTS use the same posterior in log coordinates, including all Jacobians.
The Laplace plots use draws from the full four-parameter Gaussian, transformed
back to physical values.

```julia
using AdvancedHMC, MCMCDiagnosticTools
samples = sample_posterior(problem;initial=result,laplace=result.uncertainty,
    warmup=500,samples=1000,chains=4,seed=24)
samples.diagnostics
```

Raw draws, diagnostic thresholds, R-hat, bulk/tail ESS, divergences, forward
evaluation counts and timing are exported beside the figures. Inspect those
diagnostics before treating the retained draws as posterior inference.
This is one synthetic data realization, not a repeated-experiment coverage study.
In the recorded run, maximum R-hat was 1.0024, minimum bulk ESS was 1797, minimum
tail ESS was 1545, and the retained samples had no divergences or maximum-depth
hits. There were 73 divergences during warmup, when the step size and metric
were being adapted; these are recorded separately and those draws are discarded.
About 728,000 analytical forward evaluations were used. A comparable chain budget
with sectional PBM solves would cost substantially more; it is not run here.

## Predict measurements and examine the priors

Forty posterior draws are propagated with a budget of forty reference solves.
Latent concentration uncertainty and additional measurement noise are returned
separately. Displayed predictive bands are empirical quantiles of that limited
draw set; they are a budgeted illustration, not precise tail estimates.

The likelihood and posterior profiles below refit all three remaining
parameters at each fixed log(g₁). They have separate reference optima. The
posterior curve has no ordinary likelihood-confidence interpretation. The
prior-sensitivity calculation doubles the log-scale widths of the two kinetic
priors while leaving the nuisance priors unchanged, then refits.

![Predictive checks, nuisance-reoptimized profiles and prior sensitivity](../assets/bayesian_growth/checks.png)

## Separate discretization from posterior uncertainty

The characteristic calculation avoids fitting synthetic data with the same
sectional approximation that generated them. The runner also refits the model
with MUSCL on 280 and 560 bins, halves the time step on the finer grid, and
enlarges the size domain at the same fine spacing. These are separate numerical
experiments, with identical data, likelihoods, priors and coordinate choices.
The optimizer scales for these refits are the reference posterior marginal
SDs in log coordinates. The central-difference step is 10⁻⁶. Both raw and
scaled gradients are retained, and the scaled infinity-norm tolerance is 10⁻³.

![Changes in inferred growth coefficients under numerical refinement](../assets/bayesian_growth/refinement.png)

Mode changes are expressed in the reference joint posterior's marginal SD
units in log coordinates. A small material-balance drift alone would not
establish an accurate distribution fit. Each numerical refit retains its
convergence flag. All four recorded refits passed their declared convergence
checks. The 280-bin fit moves g₀ by about 1.68 reference SDs. At 560 bins, the
shifts are about −0.13 SD for g₀ and +0.30 SD for g₁. Halving the time step changes
these shifts by about 0.003 SD, and the enlarged-domain calculation agrees with
that fine-step result to the displayed precision. Spatial error remains
measurable: a stricter requirement such as 0.1 SD would need further refinement.

The raw tables also report a sample-size-scaled Hellinger difference between
predicted histogram probabilities and concentration changes relative to its
noise scale, evaluated at common parameter values. These measure prediction
changes separately from changes in the fitted parameters. The analytical prior
sensitivity and MCMC comparison do not
include a moment closure: a future closure-based fit needs a separate closure
error assessment, beyond time-step refinement.

## Run the complete calculation

Use a working Julia environment containing the local package and Optim,
Distributions, AdvancedHMC 0.8, and MCMCDiagnosticTools 0.3:

```sh
julia --project=/path/to/working_environment examples/estimate_size_dependent_growth.jl
python docs/benchmarks/plot_bayesian_growth.py --usetex
```

The runner records its inputs and results under docs/src/assets/bayesian_growth.
The documentation build reads those assets and executes only the short setup
block above. It does not run optimization or sampling on every build. For a
short local smoke check, call run_example with smaller samples/warmup and
refine=false; a small chain budget may fail the diagnostic thresholds and must
remain labelled accordingly.
