# Minimal protein crystallization tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using Optim
using Distributions: Uniform, Exponential
# Import CSV readers, the checked interval solver and calibration starts.
include(joinpath(pkgdir(PopulationBalanceModels), "examples", "isothermal_protein_crystallization.jl"))
using .IsothermalProteinCrystallization: load_data, select_concentration, select_psds,
    model_name, stage_concentration, cfl_solve, fit_model

# From examples/isothermal_protein_crystallization.jl (constants).
const C0 = 19.965841161400512
const TEMPERATURE = 293.15                 # K
const BOLTZMANN = 1.380649e-23             # J K⁻¹
const WINDOW = (2.5,25.0)
const SIGMA_F = 0.03
const CORRELATION_LENGTH = 2.0
const FIXED_GAMMA = 0.64                  # mJ/m²; fixed Mitchell prior centre
const RHO = 1.24e-12                      # g/μm³ (= 1240 kg/m³)
const LOG_AJ_SHIFT = log(60/1000)          # old m⁻³ s⁻¹ -> L⁻¹ min⁻¹
const AJ_BOUNDS = (LOG_AJ_SHIFT,100+LOG_AJ_SHIFT)
const D0_PRIOR_MEAN = 0.01                # μm²/min; declared regularization
const MOLECULAR_VOLUME = 2.97e-26          # m³ molecule⁻¹; Lin et al. (2017)

# Physical definitions from examples/isothermal_protein_crystallization.jl.
parameters(; logAJ=30.5, gamma=FIXED_GAMMA, AG=.0002, g=3.3, cstar=2.16,
           grd_strength=0.) =
    (; logAJ, gamma, AG, g, cstar, grd_strength, c0=C0, rho=RHO, kv=.81)

barrier(gamma) = 16pi * (1e-3gamma)^3 * MOLECULAR_VOLUME^2 /
                 (3 * BOLTZMANN^3 * TEMPERATURE^3)

function growth_rate(p, c)
    c <= p.cstar && return zero(c)
    p.AG * (c/p.cstar - 1)^p.g          # μm/min
end

function nucleation_rate(p, c)
    c <= p.cstar && return zero(c)
    logJ = p.logAJ - barrier(p.gamma)/log(c/p.cstar)^2
    # Avoid underflowed Dual values with nonzero subnormal partials. This is
    # machine underflow, not a physical nucleation cutoff or a seed population.
    logJ < log(floatmin(Float64)) && return zero(c)
    logJ < log(floatmax(Float64)) || throw(NumericalFailure("CNT nucleation overflow"))
    exp(logJ) # particles/(L min)
end

nucleation(p,t,ctx) = nucleation_rate(p,stage_concentration(p,ctx))

# Physical definitions from examples/isothermal_protein_crystallization.jl.
function parameter_specs(; p=parameters(),model=:cnt)
    model=model_name(model)
    common=(ParameterSpec(:logAJ;initial=p.logAJ,transform=LogitTransform(AJ_BOUNDS...),
         prior=Uniform(AJ_BOUNDS...)),
     ParameterSpec(:AG;initial=p.AG,transform=LogTransform(),
         prior=Prior(Uniform(log(1e-5),log(1.5));coordinates=:inference)),
     ParameterSpec(:g;initial=p.g,transform=LogitTransform(1.,5.),
         prior=Uniform(1.,5.)))
    model===:cnt ? common : (common...,ParameterSpec(:grd_strength;
        initial=p.grd_strength>0 ? p.grd_strength : D0_PRIOR_MEAN,transform=LogTransform(),
        prior=Exponential(D0_PRIOR_MEAN)))
end

function build(; p=parameters(), model=:cnt, bins=200, Lmax=50.0, tf=660.0)
    model = model_name(model)
    model === :cnt && (p=merge(p, (grd_strength=zero(p.grd_strength),)))
    0 < p.cstar < C0 || throw(ArgumentError("require 0 < solubility < initial concentration"))
    p.grd_strength >= 0 || throw(ArgumentError("diffusivity must be nonnegative"))
    grid = Grid1D(range(0.0, Lmax; length=bins+1); coordinate=:length)
    terms = (
        growth=Growth((L,p,t,ctx) -> growth_rate(p, stage_concentration(p,ctx))),
        primary=Nucleation(nucleation, 0.0),
    )
    if model === :cnt_grd
        terms = merge(terms, (dispersion=SizeDiffusion((L,p,t,ctx) -> p.grd_strength),))
    end
    methods = TermMethods(growth=MUSCL(), by_term=(primary=BoundaryInjection(:left),))
    operator = compile(PBMModel(; terms...), grid; methods, bc=SizeBC())
    exchange = ConservativeExchange(:crystals, :c, (L,p,t,ctx) -> p.rho*p.kv*L^3)
    system = PBMSystem((crystals=operator,); exchanges=(exchange,), nonnegative_bulk=(:c,))
    PBMProblem(system, (crystals=zeros(bins),), (0.0, tf); bulk0=(c=C0,), p)
end

function tutorial_problem(; model=:cnt, bins=200)
    model = model_name(model)
    data = load_data()
    concentration = select_concentration(data, :calibration)
    distributions = select_psds(data, :calibration)
    observed = [r.concentration_g_L for r in concentration]
    measurements = Measurement[Measurement([r.model_time_min for r in concentration], observed,
        BulkObservable(:c); likelihood=GaussianLikelihood(sigma=reshape(0.05 .* observed,1,:)),
        source=:concentration)]
    for rows in distributions
        selected = rows[2:2:end-1] # Same 22 interior CDF ordinates; endpoints excluded.
        x = [r.size_um for r in selected]
        F = reshape([r.conditional_cdf for r in selected], :, 1)
        covariance = SIGMA_F^2 .* [exp(-abs(a-b)/CORRELATION_LENGTH) for a in x,b in x]
        push!(measurements, Measurement([first(rows).time_min], F,
            PopulationCDF(:crystals, x; window=WINDOW);
            likelihood=GaussianLikelihood(; covariance),
            source=Symbol("CDF_", Int(first(rows).time_min))))
    end
    p = parameters(grd_strength=model === :cnt ? 0.0 : D0_PRIOR_MEAN)
    problem = build(; p, model, bins, tf=420.0)
    # This specialized bucketed schedule preserves the study's derivative contract.
    # StepControl rejects AD through state-dependent step decisions.
    solver = cfl_solve
    experiment = InferenceExperiment(:batch, problem, measurements...; solver)
    fixed = model === :cnt ? (cstar=p.cstar, gamma=FIXED_GAMMA, grd_strength=0.0) :
        (cstar=p.cstar, gamma=FIXED_GAMMA)
    EstimationProblem(parameter_specs(; p, model), (experiment,); fixed)
end

function main(; model=:cnt)
    estimation = tutorial_problem(; model)
    fitted = fit_model(estimation) # Three calibration-only starts; no MCMC.
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    theta = physical_parameters(estimation, fitted.minimizer, :batch)
    # Freeze calibration before predicting the 660-min held-out distribution.
    p = merge(parameters(), theta)
    solver = only(estimation.experiments).solver
    solution = solver(build(; p, model), [0.0, 74.0, 180.0, 420.0, 660.0])
    @show theta [bulk(solution.problem,u,:c) for u in solution.u]
    @show observe(PopulationQuantiles(:crystals, [0.1,0.5,0.9]; window=WINDOW), solution)
    (; estimation, fitted, solution)
end

# Optional after import PythonPlot. These MAP plots require no MCMC or bands.
function plot_results(result)
    concentration = plot_observable(result.solution, BulkObservable(:c);
        xlabel="Time (min)", ylabel="Concentration (g/L)")
    parameters = plot_parameters(result.estimation, result.fitted; coordinates=:physical)
    (; concentration, parameters)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
