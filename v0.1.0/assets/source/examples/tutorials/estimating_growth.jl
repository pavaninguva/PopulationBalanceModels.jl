# Fit prepared point-density data with an analytical affine-growth forward law.
using PopulationBalanceModels, Optim, Random, LinearAlgebra
using Distributions: Normal, LogNormal, Uniform, pdf, cdf, ccdf, truncated
include(joinpath(@__DIR__,"..","support","affine_growth_reference.jl"))

# Separate, reproducible synthetic-data generation; no PBM solve is involved.
data = synthetic_measurements(seed=2022)
model = PBMModel(growth=Growth(LAW)) # G(x)=g0+g1*x; the reference companion defines LAW.
# The reference companion supplies the required PBM state/solution context.
# It is not the forward approximation used by this analytical likelihood.
problem = analytical_context(model)
observable = FunctionObservable((N,op,p,t)->exact_density.(data.sizes,t,Ref(p)))
measurement = Measurement(data.times,data.values,observable;
    likelihood=TruncatedGaussianLikelihood(sigma=data.noise),source=:density_measurements)
experiment = InferenceExperiment(:growth,problem,measurement;solver=analytical_solver)
parameters = [
    ParameterSpec(:g0;initial=0.40,prior=LogNormal(log(0.4),0.4),transform=LogTransform()),
    ParameterSpec(:g1;initial=0.23,prior=LogNormal(log(0.25),0.6),transform=LogTransform()),
]
estimation = EstimationProblem(parameters,(experiment,))

if abspath(PROGRAM_FILE) == @__FILE__
    fitted = fit(estimation;nstarts=3,seed=22,maxiters=500)
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    predicted = predictions(estimation,fitted.minimizer)
    @show fitted.parameters.shared fitted.convention
    # Optional: import PythonPlot
    # plot_fit(estimation,predicted;abscissa=data.sizes,xlabel="Size coordinate",ylabel="Number density")
    # plot_parameters(estimation,fitted;coordinates=:physical)
end
# Numerical-forward estimation with an explicit point reconstruction, refinement
# and local uncertainty are follow-on calculations in estimate_size_dependent_growth.jl.
