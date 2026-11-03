# Fixed-error Tang lysozyme study: length in micrometres, time in minutes,
# concentration in g/L, particle number in particles/L. No seed population.
using PopulationBalanceModels, Optim, LinearAlgebra, DelimitedFiles, ForwardDiff
using Distributions: Uniform, Exponential
include(joinpath(@__DIR__,"..","support","protein_data.jl"))

const C0 = 19.965841161400512 # Fixed initial condition, excluded from likelihood.
const FIXED_GAMMA = 0.64
const WINDOW = (2.5,25.0)
const SIGMA_F = 0.03
const CORRELATION_LENGTH = 2.0
const AJ_BOUNDS = (log(60/1000),100+log(60/1000))
const D0_PRIOR_MEAN = 0.01 # Declared regularizing mean, micrometre^2/min.
parameters(;logAJ=30.5,gamma=FIXED_GAMMA,AG=0.0002,g=3.3,cstar=2.16,grd_strength=0.0) =
    (;logAJ,gamma,AG,g,cstar,grd_strength,c0=C0,rho=1.24e-12,kv=0.81)
barrier(gamma) = 16pi*(1e-3gamma)^3*(2.97e-26)^2/(3*(1.380649e-23*293.15)^3)
growth_rate(p,c) = c<=p.cstar ? zero(c) : p.AG*(c/p.cstar-1)^p.g
function nucleation_rate(p,c)
    c<=p.cstar && return zero(c)
    logJ = p.logAJ-barrier(p.gamma)/log(c/p.cstar)^2
    logJ<log(floatmin(Float64)) && return zero(c) # Machine underflow, not a physical cutoff.
    logJ<log(floatmax(Float64)) || throw(NumericalFailure("CNT nucleation overflow"))
    exp(logJ)
end
# Current-stage concentration caps, the bucketed SSPRK33 interval schedule and
# the three prescribed calibration starts are shared with the maintained example.
include(joinpath(@__DIR__,"..","support","protein_execution.jl"))
growth = Growth(PolynomialRate(((p,t,ctx)->growth_rate(p,stage_concentration(p,ctx)),)))
primary = Nucleation((p,t,ctx)->nucleation_rate(p,stage_concentration(p,ctx)),0.0)
cnt_model = PBMModel(;growth,primary)

# Repeated parameter-dependent construction is useful for fitting/prediction.
function batch_problem(model;p=parameters(),bins=200,tf=660.0)
    0<p.cstar<C0 || throw(ArgumentError("require 0 < solubility < initial concentration"))
    p.grd_strength>=0 || throw(ArgumentError("diffusivity must be nonnegative"))
    grid = Grid1D(range(0.0,50.0;length=bins+1);coordinate=:length)
    operator = compile(model,grid;bc=SizeBC(),methods=TermMethods(
        growth=MUSCL(),nucleation=BoundaryInjection(:left)))
    exchange = ConservativeExchange(:crystals,:c,(L,p,t,ctx)->p.rho*p.kv*L^3)
    system = PBMSystem((crystals=operator,);exchanges=(exchange,),nonnegative_bulk=(:c,))
    PBMProblem(system,(crystals=zeros(bins),),(0.0,tf);bulk0=(c=C0,),p)
end
problem = batch_problem(cnt_model)
solver = cfl_solve # Reused unchanged in inference; generic StepControl is not equivalent.
if abspath(PROGRAM_FILE) == @__FILE__
    nominal = solver(problem,[0.0,74.0,180.0,420.0,660.0])
    @show observe(BulkObservable(:c),nominal)
end

# Measurements: 11 concentrations and 74/180/420-min conditional number CDFs.
# Four later concentrations and the 660-min CDF are held out. Quantiles are not fitted.
data = load_data()
calibration_concentration = select_concentration(data,:calibration)
test_concentration = select_concentration(data,:validation)
calibration_psds = select_psds(data,:calibration)
test_psds = select_psds(data,:validation)
observed = [r.concentration_g_L for r in calibration_concentration]
measurements = Measurement[Measurement([r.model_time_min for r in calibration_concentration],
    observed,BulkObservable(:c);likelihood=GaussianLikelihood(sigma=reshape(0.05 .*observed,1,:)),
    source=:concentration)]
for rows in calibration_psds
    selected = rows[2:2:end-1] # The same 22 interior ordinates, with endpoints excluded.
    x = [r.size_um for r in selected]
    F = reshape([r.conditional_cdf for r in selected],:,1)
    covariance = SIGMA_F^2 .* [exp(-abs(a-b)/CORRELATION_LENGTH) for a in x,b in x]
    push!(measurements,Measurement([first(rows).time_min],F,
        PopulationCDF(:crystals,x;window=WINDOW);likelihood=GaussianLikelihood(;covariance),
        source=Symbol("CDF_",Int(first(rows).time_min))))
end

# Physical prior laws with explicit inference transforms. The AG compatibility
# declaration is log-uniform physically; the shared calibration-start helper
# uses this same specification to construct its three starts.
function parameter_specs(;p=parameters(),model=:cnt)
    common = (ParameterSpec(:logAJ;initial=p.logAJ,prior=Uniform(AJ_BOUNDS...),
                  transform=LogitTransform(AJ_BOUNDS...)),
              ParameterSpec(:AG;initial=p.AG,transform=LogTransform(),
                  prior=Prior(Uniform(log(1e-5),log(1.5));coordinates=:inference)),
              ParameterSpec(:g;initial=p.g,prior=Uniform(1.0,5.0),transform=LogitTransform(1.0,5.0)))
    model===:cnt ? common : (common...,ParameterSpec(:grd_strength;
        initial=p.grd_strength>0 ? p.grd_strength : D0_PRIOR_MEAN,
        prior=Exponential(D0_PRIOR_MEAN),transform=LogTransform()))
end
experiment = InferenceExperiment(:batch,batch_problem(cnt_model;tf=420.0),measurements...;solver)
estimation = EstimationProblem(parameter_specs(),(experiment,);
    fixed=(cstar=2.16,gamma=FIXED_GAMMA,grd_strength=0.0))
if abspath(PROGRAM_FILE) == @__FILE__
    fitted = fit_model(estimation) # Prescribed calibration starts, no MCMC/local Gaussian calculation.
    fitted.converged || error("MAP failed; inspect the diagnostics")
    frozen = merge(parameters(),physical_parameters(estimation,fitted.minimizer,:batch))
    heldout_times = sort!(unique(vcat(420.0,[r.model_time_min for r in test_concentration],660.0)))
    prediction = solver(batch_problem(cnt_model;p=frozen),heldout_times)
    @show observe(BulkObservable(:c),prediction)
    heldout_cdf = PopulationCDF(:crystals,[r.size_um for r in only(test_psds)];window=WINDOW)
    @show observe(heldout_cdf,prediction;index=length(prediction.t)) # 660-min conditional CDF.
    @show observe(PopulationQuantiles(:crystals,[0.1,0.5,0.9];window=WINDOW),prediction)
    # Optional: import PythonPlot
    # plot_observable(prediction,BulkObservable(:c);xlabel="Time (min)",ylabel="Concentration (g/L)")
end

# Next stage: keep the kinetics and observations; add constant native Ito diffusion.
grd_model = PBMModel(;growth,primary,dispersion=SizeDiffusion(
    PolynomialRate(((p,t,ctx)->p.grd_strength,))))
grd_parameters = parameters(grd_strength=D0_PRIOR_MEAN)
grd_experiment = InferenceExperiment(:batch,batch_problem(grd_model;p=grd_parameters,tf=420.0),
    measurements...;solver)
grd_estimation = EstimationProblem(parameter_specs(p=grd_parameters,model=:cnt_grd),
    (grd_experiment,);fixed=(cstar=2.16,gamma=FIXED_GAMMA))
# Run explicitly when ready: grd_fitted = fit_model(grd_estimation)
# Local uncertainty and posterior sampling remain separate subsequent analyses.
