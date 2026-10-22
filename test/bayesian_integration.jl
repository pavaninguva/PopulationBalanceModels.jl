using Test, LinearAlgebra, Random, Statistics, ForwardDiff
using Distributions: Normal, Exponential
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
    @test all(data.values.>=0) # Draws and the likelihood use the same nonnegative truncation.
    p=recipe.estimation_problem(data)
    z=initial_coordinates(p)
    @test check_gradient(p,z).valid
    result=fit(p;nstarts=3,seed=22,uncertainty=:laplace)
    @test result.converged
    @test result.diagnostics.scaled_gradient_norm<=result.diagnostics.gradient_tolerance
    @test all(r->!r.converged || r.scaled_gradient_norm<=result.diagnostics.gradient_tolerance,result.diagnostics.starts)
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

@testset "Sampler tuning preserves the shared posterior" begin
    f=bayesian_linear_fixture();p=f.problem
    center=fit(p)
    tuning=Matrix(Symmetric(f.covariance));saved_tuning=copy(tuning)
    samples=sample_posterior(p;initial=center,initial_covariance=tuning,
        warmup=200,samples=600,chains=2,seed=153,min_ess=100)
    Z=reshape(samples.z,2,:)
    @test tuning==saved_tuning
    @test samples.diagnostics.metric_initialization===:user_tuning
    @test samples.diagnostics.metric===:dense
    @test maximum(samples.diagnostics.rhat)<1.05
    @test all(abs.(vec(mean(Z;dims=2))-f.mode).<.3sqrt.(diag(f.covariance)))
    @test norm(cov(permutedims(Z))-f.covariance)/norm(f.covariance)<.2
    @test_throws ArgumentError sample_posterior(p;initial_covariance=[1. 2.;2. 1.],warmup=50,samples=8,chains=2)
    @test_throws ArgumentError sample_posterior(p;initial_covariance=ones(3,3),warmup=50,samples=8,chains=2)
end

@testset "Auxiliary sampling map keeps inference density and Jacobian" begin
    f=bayesian_linear_fixture();p=f.problem;r=fit(p)
    # A nonlinear triangular map with determinant one straightens a curved ridge.
    to_sampler=z->[z[1]+.7z[2]^2,z[2]]
    from_sampler=y->[y[1]-.7y[2]^2,y[2]]
    map=(;to_sampler,from_sampler,logabsdetjac=y->zero(first(y)),label=:nonlinear_test)
    B=ForwardDiff.jacobian(to_sampler,r.minimizer)
    C=Matrix(Symmetric(B*f.covariance*B'))
    s=sample_posterior(p;initial=r,sampling_map=map,initial_covariance=C,
        warmup=200,samples=600,chains=2,seed=166,min_ess=100)
    Z=reshape(s.z,2,:)
    @test s.diagnostics.sampling_coordinates===:nonlinear_test
    @test s.diagnostics.z_coordinates===:inference
    @test maximum(s.diagnostics.rhat)<1.05
    @test all(abs.(vec(mean(Z;dims=2))-f.mode).<.3sqrt.(diag(f.covariance)))
    @test norm(cov(permutedims(Z))-f.covariance)/norm(f.covariance)<.2
    @test s.diagnostics.statistics[1][1].log_density≈logposterior(p,s.z[:,1,1]) atol=1e-10
    wrong=(;to_sampler,from_sampler,logabsdetjac=y->one(first(y)),label=:wrong)
    @test_throws ArgumentError sample_posterior(p;initial=r,sampling_map=wrong,warmup=50,samples=8,chains=2)

    # A global nonlinear bijection with NONconstant determinant tests the
    # transformed density, not just a rotation of Gaussian coordinates.
    sinh_map=(to_sampler=z->[asinh(z[1]),z[2]],
        from_sampler=y->[sinh(y[1]),y[2]],
        logabsdetjac=y->log(cosh(y[1])),label=:sinh_test)
    B=ForwardDiff.jacobian(sinh_map.to_sampler,r.minimizer)
    C=Matrix(Symmetric(B*f.covariance*B'))
    nonlinear=sample_posterior(p;initial=r,sampling_map=sinh_map,initial_covariance=C,
        warmup=250,samples=700,chains=2,seed=167,min_ess=100)
    Z=reshape(nonlinear.z,2,:)
    @test maximum(nonlinear.diagnostics.rhat)<1.05
    @test all(abs.(vec(mean(Z;dims=2))-f.mode).<.3sqrt.(diag(f.covariance)))
    @test norm(cov(permutedims(Z))-f.covariance)/norm(f.covariance)<.2
    @test nonlinear.diagnostics.statistics[1][1].log_density≈logposterior(p,nonlinear.z[:,1,1]) atol=1e-10
    for chain in 1:2,draw in (1,97,350,700)
        @test nonlinear.diagnostics.statistics[chain][draw].log_density≈logposterior(p,nonlinear.z[:,draw,chain]) atol=1e-9
    end
    @test nonlinear.diagnostics.sampling_map_validation===:every_evaluation
    @test nonlinear.diagnostics.energy==hcat([[s.hamiltonian_energy for s in chain] for chain in nonlinear.diagnostics.statistics]...)
    # This wrong Jacobian agrees at every chain start. It must fail when NUTS
    # moves away, rather than silently sampling a different target.
    center=r.minimizer
    wrong_away=(to_sampler=identity,from_sampler=identity,
        logabsdetjac=y->.1*(y[1]-center[1])^4,label=:wrong_away)
    @test_throws ArgumentError sample_posterior(p;initial=center,starts=[copy(center),copy(center)],
        sampling_map=wrong_away,warmup=50,samples=8,chains=2,seed=168)
end

@testset "Profile curvature and nuisance stationarity" begin
    f=bayesian_linear_fixture();p=f.problem;r=fit(p)
    likelihood_precision=f.A'*(f.C\f.A)
    likelihood_mode=likelihood_precision\(f.A'*(f.C\f.y))
    for (target,mode,H,C) in ((:posterior,f.mode,f.precision,f.covariance),
            (:likelihood,likelihood_mode,likelihood_precision,inv(likelihood_precision)))
        values=mode[1].+[-.15,0.,.15]
        profile=estimation_profile(p,r,1,values;target)
        @test all(profile.converged)
        @test profile.delta≈.5 .* (values.-mode[1]).^2 ./C[1,1] atol=1e-7
        expected=mode[2].-H[2,1]/H[2,2].*(values.-mode[1])
        @test [v[2] for v in profile.minimizers]≈expected atol=1e-6
    end
end

@testset "Upstream sampler metric and mapped Hamiltonian convention" begin
    # Check the installed official implementation against the analytic
    # Hamiltonian; a tuning covariance is the inverse MASS, not its inverse.
    C=[2. .3;.3 .7]
    target(y)=-sum(abs2,[sinh(y[1]),y[2]])/2+log(cosh(y[1]))
    gradient(y)=[-sinh(y[1])*cosh(y[1])+tanh(y[1]),-y[2]]
    h=AdvancedHMC.Hamiltonian(AdvancedHMC.DenseEuclideanMetric(C),target,y->(target(y),gradient(y)))
    for y in ([.3,-.7],[-2.,1.]),momentum in ([.2,.8],[-1.,.4])
        point=AdvancedHMC.phasepoint(h,y,momentum)
        @test AdvancedHMC.energy(point)≈-target(y)+dot(momentum,C*momentum)/2 rtol=1e-13
        @test AdvancedHMC.∂H∂r(h,momentum)≈C*momentum rtol=1e-13
    end
end
