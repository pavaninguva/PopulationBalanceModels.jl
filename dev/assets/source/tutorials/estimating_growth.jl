# Minimal estimating growth tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using Optim, Random
using Distributions: Normal, LogNormal, Uniform, pdf, cdf, ccdf, quantile, truncated

# Physical definitions from examples/estimate_size_dependent_growth.jl.
const TRUTH=(g0=0.434,g1=0.2604)

const TIMES=[0.1,0.25,0.4,0.6,0.8,1.0]

const SIZES=collect(range(0.0,1.4;length=281))

const CENTER=0.2

const WIDTH=sqrt(0.0005/2)

const INITIAL_NUMBER=50sqrt(π*0.0005)

const UNIFORM_BOUNDS=((0.05,1.0),(0.02,1.0))

const LAW=AffineGrowth(:g0,:g1)

initial_density(a)=50exp(-(a-CENTER)^2/0.0005)

initial_profile()=InitialDistribution(Normal(CENTER,WIDTH);
    total=INITIAL_NUMBER,truncation=:physical)
# Independent closed-form reference for data generation, not a discretized solve.

shift(t,p)=iszero(p.g1) ? p.g0*t : p.g0*expm1(p.g1*t)/p.g1

function exact_density(a,t,p=TRUTH)
    foot=(a-shift(t,p))*exp(-p.g1*t)
    foot<0 ? 0.0 : initial_density(foot)*exp(-p.g1*t)
end

function initial_integral(lo,hi)
    hi<=lo && return 0.0
    d=Normal(CENTER,WIDTH)
    INITIAL_NUMBER*(lo>CENTER ? ccdf(d,lo)-ccdf(d,hi) : cdf(d,hi)-cdf(d,lo))
end

function exact_counts(edges,t,p=TRUTH)
    inverse(a)=max(0.0,(a-shift(t,p))*exp(-p.g1*t))
    [initial_integral(inverse(a),inverse(b)) for (a,b) in zip(edges[1:end-1],edges[2:end])]
end

function synthetic_measurements(;seed=2022,noise=1.0,sizes=SIZES,times=TIMES)
    noise>0 && isfinite(noise) || throw(ArgumentError("noise must be positive"))
    truth=[exact_density(a,t) for a in sizes,t in times]
    rng=MersenneTwister(seed)
    values=[rand(rng,truncated(Normal(truth[i,j],noise),0.0,Inf)) for i in axes(truth,1),j in axes(truth,2)]
    (;values,truth,noise,sizes=collect(sizes),times=collect(times),seed,noise_model=:truncated_gaussian)
end

function analytical_solver(pb,times)
    states=[reshape(exact_counts(pb.operator.grid.edges,t,pb.p),:,1) for t in times]
    PBMSolution(times,states,pb,0)
end
# Point observations from a bin solver require an explicit reconstruction. This
# linear interpolation of bin-average densities is NOT an exact continuous PSD.

function tutorial_problem(; prior=:lognormal)
    data = synthetic_measurements(seed=2022)
    grid = characteristic_grid(LAW, (0.0,1.4); dt=0.005, p=TRUTH)
    model = PBMModel(growth=Growth(LAW))
    operator = compile(model, grid; evolution=Inguva2022(), p=TRUTH,
        bc=SizeBC(OpenBoundary(),OpenBoundary()))
    base = PBMProblem(operator, initial_profile(), (0.0,1.0); p=TRUTH)
    observable = FunctionObservable((N,op,p,t) -> exact_density.(data.sizes,t,Ref(p)))
    measurement = Measurement(data.times, data.values, observable;
        likelihood=TruncatedGaussianLikelihood(sigma=data.noise), source=:density_measurements)
    # Closed-form forward values isolate estimation from transport error.
    solver = analytical_solver # Deliberate exact forward reference, not an ODE wrapper.
    experiment = InferenceExperiment(:growth, base, measurement; solver)
    distributions = prior === :lognormal ?
        (LogNormal(log(0.4),0.4), LogNormal(log(0.25),0.6)) :
        (Uniform(UNIFORM_BOUNDS[1]...), Uniform(UNIFORM_BOUNDS[2]...))
    parameters = [ParameterSpec(name; initial, prior=d, transform=LogTransform())
        for (name,initial,d) in zip((:g0,:g1),(0.40,0.23),distributions)]
    EstimationProblem(parameters, (experiment,))
end

function main()
    problem = tutorial_problem()
    fitted = fit(problem; nstarts=3, seed=22, maxiters=500)
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    @show fitted.parameters.shared fitted.convention
    @show posterior_components(problem, fitted.minimizer)
    predicted = predictions(problem, fitted.minimizer) # Explicit forward work.
    (; estimation=problem, fitted, predicted)
end

# Optional after import PythonPlot; predictions were computed explicitly in main.
function plot_results(result)
    fit_figure = plot_fit(result.estimation, result.predicted; abscissa=SIZES,
        xlabel="Size coordinate", ylabel="Number density")
    parameters = plot_parameters(result.estimation, result.fitted; coordinates=:physical)
    (; fit_figure, parameters)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
