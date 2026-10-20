# Observations and parameter estimation

A forward solution stores particle numbers or moments and bulk states. A sensor
usually reports a different quantity: a windowed histogram, a weighted size
average, a dissolved concentration, or a moment. Define that measurement map
before assigning kinetic parameters.

`MomentObservable(k)` measures a standalone population; for a 2D distribution
use a tuple `(kx,ky)`. `PopulationMoment(:particles,k)` and `BulkObservable(:c)`
measure coupled states. `HistogramObservable(edges)` integrates a piecewise
constant density over experimental bins, which must lie inside the model domain.
`PopulationHistogram(:particles,edges; weight=x->1, normalize=false)` additionally
weights pivot numbers and can normalize over the measured window. Normalizing
an empty measured distribution raises a numerical failure. An initially empty
population is allowed if it is nonempty at the observation time.

`FunctionObservable((N,op,p,t)->...)` and
`SystemObservable((views,op,p,t)->...)` describe custom scalar or vector outputs.
Finite moments do not uniquely specify a histogram: moment populations need an
explicit reconstruction/measurement model. Use a FunctionObservable for 2D
bins or marginals. An intensity weighting and an instrument's reported average
must be defined separately; the CEJ example explicitly uses ``M_7/M_6``.

## Experiments and likelihood

`Observation(times,data,observable; sigma=...)` uses rows for outputs and columns
for times; a scalar observable accepts a data vector. Times must increase and
lie inside the problem interval. Standard deviations are positive, finite,
known independent Gaussian errors, supplied as a scalar or matching matrix.
Select missing observations before construction. Correlated or normalized
histogram errors require a custom likelihood; normalization creates dependence.

An `Experiment(problem, observations...)` binds operating conditions and
initial states to measurements. A `CalibrationProblem(experiments;
parameters=(z,experiment)->physical_parameters, dt)` fits common or
experiment-specific parameters through the same forward model. Transform a
positive rate as `exp(z[1])` and keep its optimization coordinate distinct from
the physical rate.

`residuals(cal,z)` concatenates ``(prediction-data)/sigma`` after solving at all
measurement times. `objective(cal,z)` is the corresponding known-variance
Gaussian negative log likelihood, including its normalization constant.
`on_failure=:infinity` optionally maps `NumericalFailure` to infinity; programming
or invalid-model errors still propagate. Inspect failure behavior instead of
masking errors with arbitrary penalties.

## Fitting and interpretation

Load Optim in a separate working environment to activate `fit(cal,z0; ...)`.
The adapter returns a `FitResult` with minimizer, convergence flag, minimum,
raw optimizer output, and numerical failure records. See the
[kinetic estimation case study](case_studies/estimation.md) for the runnable fit.

`sensitivity_report(cal,z)` finite-differences standardized residuals in
optimization coordinates and reports singular values, numerical rank, condition
number, and a covariance approximation only at full column rank. This local
Gaussian calculation is not a structural identifiability proof or confidence
interval. Study step sensitivity, mesh/time error, starting points, and likelihood
profiles or resampling before making uncertainty claims. Validated automatic
differentiation and adjoints are outside this reference fitting backend.

Load SciMLBase to activate `odeproblem(problem; ...)`; select a compatible solver
in the working environment. The adapter exposes the compiled residual, including
bulk balances, rather than the direct `HOCADI` or `ExactShift` updates. Solver
choice does not remove closure, boundary, or positivity restrictions.
