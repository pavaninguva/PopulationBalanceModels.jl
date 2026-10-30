"""Synthetic stationary MSMPR calibration; no experimental data or time integration."""
module EstimateStationaryCrystallizer
using PopulationBalanceModels, NonlinearSolve, Optim, Distributions

function estimation_problem()
    # Length in micrometres, time in minutes, moments per process volume.
    # Constant growth, point nucleation at zero, and residence-time withdrawal.
    model = PBMModel(
        growth = Growth(PolynomialRate(((p,t,ctx) -> p.G,))),
        births = Nucleation((p,t,ctx) -> p.J, 0.0),
        withdrawal = Dilution(PolynomialRate(((p,t,ctx) -> 1/p.tau,))),
    )
    operator = compile(model, MomentSet(0:3; coordinate=:length); method=ExactClosure())
    base = StationaryPBMProblem(operator, [2.0, 1.0, 1.0, 2.0];
        p=(G=0.3, J=1.0, tau=2.0), autonomous=true)
    # Independently generated synthetic data: M_k = k! J G^k tau^(k+1).
    truth = (G=0.4, J=1.2, tau=2.0)
    exact(k) = factorial(k) * truth.J * truth.G^k * truth.tau^(k+1)
    measurements = (
        Measurement([exact(0)], MomentObservable(0); likelihood=GaussianLikelihood(sigma=0.03)),
        Measurement([exact(1)], MomentObservable(1); likelihood=GaussianLikelihood(sigma=0.03)),
        Measurement([exact(3)], MomentObservable(3); likelihood=GaussianLikelihood(sigma=0.10)),
    )
    parameters = (
        ParameterSpec(:G; initial=0.3, prior=LogNormal(log(0.3),0.7)),
        ParameterSpec(:J; initial=1.0, prior=LogNormal(log(1.0),0.7)),
    )
    experiment = InferenceExperiment(:crystallizer, base, measurements...;
        solver=StationarySolver(abstol=1e-12))
    EstimationProblem(parameters, (experiment,)), truth
end

function run(; sample=false)
    problem, truth = estimation_problem()
    result = fit(problem; nstarts=3, seed=42, uncertainty=:gauss_newton)
    result.converged || error("stationary MAP did not pass convergence checks")
    println("Synthetic truth: ", truth)
    println("MAP in physical units: ", result.parameters.shared)
    println("Gradient verification: ", check_gradient(problem,result.minimizer))
    println("Predicted stationary moments: ", predictions(problem,result.minimizer))
    println("Local covariance in physical G, J units: ", result.uncertainty.covariance)
    if sample
        posterior = sample_posterior(problem; initial=result, warmup=300,
            samples=1000, chains=2, seed=43)
        println("NUTS diagnostics: ", diagnostics(posterior))
        return (; problem, truth, result, posterior)
    end
    (; problem, truth, result)
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--sample" in ARGS
        using AdvancedHMC, MCMCDiagnosticTools
    end
    EstimateStationaryCrystallizer.run(; sample="--sample" in ARGS)
end
