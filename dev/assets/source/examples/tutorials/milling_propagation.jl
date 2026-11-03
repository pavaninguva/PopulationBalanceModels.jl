# Execute after including milling_uncertainty.jl; no nominal solve is repeated.
# These are small demonstration budgets, not the 12,000-realization study.

# Stage 2: MC. Failed physical inputs/solves remain visible in this result.
mc = propagate(uq,MonteCarlo(16;seed=2018255);max_solves=16)
mc_summary = uq_summary(mc;covariance=false)
@show mc_summary.mean.d50 mc_summary.quantiles.d50

# Stage 3: first-order PCE. The 12 LHS training points are disjoint from MC;
# the EXISTING 16 MC realizations supply validation, without another ensemble.
box = hcat(fill(-2.0,4),fill(2.0,4)) # Design bounds, not truncation of the input law.
design = pce_design(uq.inputs;n=12,seed=2018257,bounds=box)
pce = fit_pce(uq,PolynomialChaos(1);design,validation=mc,max_solves=12)
@show pce.validation.rmse pce.validation.passed
pce.validation.passed || @warn "PCE validation failed: refine degree/design before using uncertainty predictions"

# Stage 4: analytical PCE mean/variance, not an assumption of Gaussian outputs.
pce_summary = uq_summary(pce;covariance=false)
@show pce_summary.mean.d50 pce_summary.variance.d50
# Optional cheap polynomial evaluations for output quantiles, AFTER validation:
# cheap = evaluate_pce(pce,MonteCarlo(10_000;seed=2018259))
# intervals = uq_summary(cheap;covariance=false).quantiles.d50
# Optional: import PythonPlot
# plot_uq(mc;output=:d50,xlabel="Time (s)",ylabel="Outlet volume median (micrometres)")
# plot_uq(pce;output=:d50,band=:sd) # Explicit mean +/- one SD; not output quantiles.
