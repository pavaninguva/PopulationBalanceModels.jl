# Observations and parameter estimation

A particle counter, a concentration measurement, and a scattering instrument
observe different features of a population. Before fitting kinetics, specify
how a model state becomes the reported measurement. Otherwise a change in
observation weighting can be mistaken for a change in the physical rate law.

## Fit a removal rate from number measurements

For size-independent removal,

```math
\frac{\partial n(v,t)}{\partial t}=-k n(v,t),\qquad
\frac{dM_0(t)}{dt}=-kM_0(t).
```

We can generate synthetic observations from the analytical solution, independently
of the numerical solver used in fitting:

```@example inference
using PopulationBalanceModels

grid = Grid1D(range(0,8;length=41))
model = PBMModel(loss=Death(rate=(v,p,t,ctx)->p.k))
operator = compile(model,grid)
initial = exp.(-grid.centers).*grid.widths
problem = PBMProblem(operator,initial,(0.0,2.0);p=(k=0.4,))
times = [0.0,0.2,0.6,1.2,2.0]
data = sum(initial).*exp.(-0.4.*times)
observation = Observation(times,data,MomentObservable(0);sigma=0.02)
experiment = Experiment(problem,observation)
nothing # hide
```

MomentObservable(0) maps the computed distribution to total number. `sigma` is
the assumed standard deviation of independent Gaussian measurement errors. Here
the data contain no added noise; they are a check of the fitting workflow.

To keep the fitted hazard positive, optimize its logarithm:

```@example inference
calibration = CalibrationProblem((experiment,);
    parameters=(z,experiment)->(k=exp(z[1]),),dt=0.01)
residuals(calibration,[log(0.4)])
```

CalibrationProblem solves the experiment at the requested observation times and
forms standardized residuals,

```math
r_j=\frac{\widehat M_0(t_j)-y_j}{\sigma_j}.
```

`objective(calibration,z)` is the independent Gaussian negative log likelihood,
including its known-variance normalization. The small residual at the true
parameter is numerical time-integration error, not a discrepancy in the
analytical data.

Load Optim in a working environment to enable fitting:

```julia
using Optim
result = fit(calibration,[log(0.15)];maxiters=100)
estimated_rate = exp(result.minimizer[1])
result.converged
```

`examples/estimate_kinetic_parameters.jl` runs this calculation jointly for two
initial population levels. Each experiment can have its own initial conditions
and operating history while sharing the fitted parameters. FitResult retains
the convergence flag, objective, raw optimizer result, and numerical failures.

## Observe other quantities

| Measurement | Declaration |
|:--|:--|
| A moment of a standalone population | `MomentObservable(k)` |
| A moment of a named coupled population | `PopulationMoment(:particles,k)` |
| A dissolved concentration | `BulkObservable(:c)` |
| Counts in specified size intervals | `HistogramObservable(edges)` |
| A weighted histogram of a coupled population | `PopulationHistogram(:particles,edges;weight=x->1,normalize=false)` |
| A custom standalone or system output | `FunctionObservable((N,op,p,t)->...)` or `SystemObservable((views,op,p,t)->...)` |

Histogram observations integrate the model's piecewise-constant density over
measurement intervals, which must lie inside the model domain. Experimental and
numerical bin edges need not coincide. Normalization is over the measured
window; normalizing an empty window raises a numerical failure.

A two-coordinate moment uses `MomentObservable((kx,ky))`; custom functions can
extract marginals or two-dimensional bins. A moment-state model needs an explicit
distribution reconstruction before predicting a histogram: a finite set of
moments does not uniquely determine a distribution.

The [nanoparticle calculation](case_studies/nanoparticles.md) weights number
density by ``L^6`` and reports ``M_7/M_6``. Its observation definition is part of
the model comparison, separate from the growth or collision discretization.

## Describe errors and assess sensitivity

Observation data use rows for outputs and columns for times. A scalar observable
also accepts a data vector, as above. Times must increase within the problem
interval; standard deviations must be finite and positive. Select missing
measurements before constructing an Observation. Correlated errors and normalized
histogram errors require a custom likelihood, since normalization introduces
dependence between bins.

A local sensitivity report is available without an optimizer:

```@example inference
report = sensitivity_report(calibration,[log(0.4)])
(rank=report.rank,singular_values=report.singular_values)
```

The report finite-differences standardized residuals in **optimization
coordinates**. It returns a residual Jacobian, singular values, numerical rank,
condition number, and a covariance approximation only at full column rank.
For this example its covariance is in ``\log k``, not ``k``. This local Gaussian
approximation does not establish structural identifiability or confidence
intervals. Check differencing steps, grid/time convergence, starting points,
and likelihood profiles or resampling before interpreting uncertainty.

By default solver failures propagate. `objective(...;on_failure=:infinity)` can
map NumericalFailure to infinity; programming and invalid-model errors still
propagate. The reference fitting backend uses finite differences. Automatic
differentiation and adjoints need a separately verified integration.
