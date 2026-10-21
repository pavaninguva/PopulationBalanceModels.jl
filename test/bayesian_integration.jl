using Test, LinearAlgebra, Random, Statistics, Distributions
using PopulationBalanceModels, Optim, AdvancedHMC, MCMCDiagnosticTools
include("fixtures/bayesian_linear.jl")

@testset "MAP, full Laplace and shared sampler density" begin
    f=bayesian_linear_fixture();p=f.problem
    result=fit(p;nstarts=2,seed=41,uncertainty=:laplace)
    @test result.converged
    @test result.minimizer≈f.mode rtol=1e-6
    @test result.uncertainty.covariance≈f.covariance rtol=2e-6
    @test result.logposterior≈logposterior(p,result.minimizer)
    @test result.diagnostics.forward_solves>0 && !isempty(result.diagnostics.history)
    @test result.diagnostics.gradient_norm<1e-3
    again=fit(p;nstarts=2,seed=41)
    @test [r.start for r in result.diagnostics.starts]==[r.start for r in again.diagnostics.starts]
    @test_throws ArgumentError fit(bayesian_linear_fixture(gradient=z->ones(2)).problem)
    exact_gradient=bayesian_linear_fixture(gradient=z->-f.precision*(z-f.mode)).problem
    @test fit(exact_gradient).minimizer≈f.mode rtol=1e-6
    scaled=fit(p;scaling=[0.1,0.01])
    @test scaled.converged
    @test scaled.minimizer≈f.mode rtol=1e-6
    @test scaled.diagnostics.scaled_gradient≈scaled.diagnostics.gradient.*[0.1,0.01]
    ex=first(p.experiments)
    failing=InferenceExperiment(:run,ex.problem,ex.measurements...;
        solver=(pb,ts)->pb.p.a>3 ? throw(NumericalFailure("deliberate invalid start")) : f.solver(pb,ts))
    recovered=fit(EstimationProblem(p.parameters,(failing,));starts=[[4.,0.5],initial_coordinates(p)])
    @test recovered.converged && recovered.diagnostics.failed_evaluations>0
    @test any(occursin("deliberate invalid start",s) for s in recovered.diagnostics.failure_messages)
    saved_covariance=copy(result.uncertainty.covariance)
    draws=sample_posterior(p;initial=result,laplace=result.uncertainty,warmup=250,samples=700,chains=2,seed=42)
    @test result.uncertainty.covariance==saved_covariance
    Z=reshape(draws.z,2,:)
    @test size(draws.z)==(2,700,2)
    @test all(abs.(vec(mean(Z;dims=2))-f.mode).<0.25sqrt.(diag(f.covariance)))
    @test norm(cov(permutedims(Z))-f.covariance)/norm(f.covariance)<0.18
    @test maximum(draws.diagnostics.rhat)<1.05
    @test minimum(draws.diagnostics.ess_bulk)>100
    @test draws.diagnostics.divergences==0
    @test draws.diagnostics.forward_solves>0
    @test diagnostics(draws)===draws.diagnostics
    @test all(isfinite,draws.diagnostics.mcse_mean)
    @test minimum(draws.diagnostics.ebfmi)>0.3
    @test size(draws.diagnostics.energy)==(700,2)
    @test length(draws.diagnostics.warmup_statistics[1])==250
    @test sum(c.divergences for c in draws.diagnostics.per_chain)==draws.diagnostics.divergences
    @test all(c->0<c.acceptance_rate<=1 && c.step_size>0,draws.diagnostics.per_chain)
    @test parameter_draws(p,draws;n=12,seed=2).z==parameter_draws(p,draws;n=12,seed=2).z
    short=sample_posterior(p;initial=result,laplace=result.uncertainty,warmup=50,samples=24,chains=2,seed=43)
    repeat=sample_posterior(p;initial=result,laplace=result.uncertainty,warmup=50,samples=24,chains=2,seed=43)
    @test short.z==repeat.z
    @test !short.diagnostics.diagnostics_passed
    values=range(f.mode[1]-0.15,f.mode[1]+0.15;length=5)
    posterior=estimation_profile(p,result,1,values;target=:posterior)
    @test all(posterior.converged)
    @test posterior.delta≈(collect(values).-f.mode[1]).^2/(2f.covariance[1,1]) atol=1e-6
    likelihood=estimation_profile(p,result,1,values;target=:likelihood)
    mle=(f.A'*(f.C\f.A))\(f.A'*(f.C\f.y))
    @test likelihood.reference≈mle rtol=1e-5
    @test likelihood.interpretation==:likelihood_profile && posterior.interpretation==:posterior_shape_no_likelihood_cutoff
    @test_throws UndefKeywordError estimation_profile(p,result,1,values)
    positive=ParameterSpec(:rate;initial=0.4,transform=LogTransform(),prior=Prior(Exponential(0.5);coordinates=:physical))
    unused=EstimationProblem((positive,),p.experiments;fixed=(a=0.0,b=0.0))
    shifted=fit(unused;uncertainty=:laplace)
    @test shifted.parameters.shared.rate≈0.5 rtol=1e-6 # physical exponential density has its mode at zero.
    @test shifted.uncertainty.covariance[1,1]≈1.0 rtol=1e-5
    @test identifiability_report(unused,shifted.minimizer).prediction_rank==0
    prior_only=prior_sensitivity((base=unused,changed=replace_priors(unused,[Prior(Exponential(5.0);coordinates=:physical)])))
    @test prior_only.comparisons.changed.prior_sensitive
    alternatives=(base=p,weaker=replace_priors(p,[Prior(Normal(0,4);coordinates=:physical),Prior(Normal(0,2);coordinates=:inference)]))
    comparison=prior_sensitivity(alternatives)
    @test comparison.comparisons.base.shift_in_reference_sd≈zeros(2)
    @test norm(comparison.comparisons.weaker.shift_in_reference_sd)>0
    refinement=refinement_study((base=p,equivalent=p))
    @test refinement.comparisons.equivalent.max_prediction_change==0
end

include("../examples/estimate_size_dependent_growth.jl")
@testset "Size-dependent growth: analytical and numerical inference" begin
    recipe=EstimateSizeDependentGrowth
    data=recipe.synthetic_measurements()
    @test data.values==recipe.synthetic_measurements().values
    @test size(data.values)==(281,6)
    @test any(data.values.<0) # Additive Gaussian noise is neither clipped nor normalized.
    p=recipe.estimation_problem(data)
    z=initial_coordinates(p)
    @test check_gradient(p,z).valid
    result=fit(p;uncertainty=:laplace)
    @test result.converged
    @test abs(result.parameters.shared.g0-recipe.TRUTH.g0)<0.01
    @test abs(result.parameters.shared.g1-recipe.TRUTH.g1)<0.03
    uniform=fit(recipe.estimation_problem(data;prior=:uniform);uncertainty=:laplace)
    @test uniform.converged
    @test norm(uniform.minimizer-result.minimizer)<0.01
    @test logprior(recipe.estimation_problem(data;prior=:uniform),log.([0.01,0.2]))==-Inf
    # Native exact moment evolution agrees with the independent characteristic moments.
    model=PBMModel(growth=Growth(AffineGrowth(:g0,:g1)))
    op=compile(model,MomentSet(0:3);method=ExactClosure())
    initial=[recipe.exact_moment(k,0.,recipe.TRUTH) for k in 0:3]
    moment_problem=PBMProblem(op,initial,(0.,1.);p=recipe.TRUTH)
    solution=simulate(moment_problem;dt=0.01,saveat=[0.5,1.0])
    @test vec(solution.u[end])≈[recipe.exact_moment(k,1.,recipe.TRUTH) for k in 0:3] rtol=1e-7
    @test_throws ArgumentError PopulationBalanceModels._observe(DistributionObservable([0.,0.5,1.];coordinate=:volume),solution.u[end],op,recipe.TRUTH,1.)
    fv=recipe.estimation_problem(data;forward=:sectional,bins=140,dt=0.002)
    @test check_gradient(fv,result.minimizer;tolerance=0.01).valid
    @test isfinite(logposterior(fv,result.minimizer))
    mapped=recipe.estimation_problem(data;forward=:inguva,dt=0.005)
    @test isfinite(logposterior(mapped,result.minimizer))
    @test maximum(abs,predictions(mapped,result.minimizer)[1][1]-predictions(p,result.minimizer)[1][1])<2.0
    moment_data=[recipe.exact_moment(3,t,recipe.TRUTH) for t in [0.5,1.0]]
    measurement=Measurement([0.5,1.0],moment_data,MomentObservable(3);likelihood=GaussianLikelihood(covariance=[1e-5 2e-6;2e-6 1e-5]))
    moment_estimation=EstimationProblem(p.parameters[1:2],(InferenceExperiment(:batch,moment_problem,measurement;dt=0.01),))
    @test check_gradient(moment_estimation,log.([recipe.TRUTH.g0,recipe.TRUTH.g1])).valid
    @test isfinite(loglikelihood(moment_estimation,log.([recipe.TRUTH.g0,recipe.TRUTH.g1])))
end
