using Test, PopulationBalanceModels, Optim, AdvancedHMC, MCMCDiagnosticTools
using LinearAlgebra, Statistics
using Distributions: Normal, LogNormal, Exponential, Beta, truncated, logpdf
isdefined(@__MODULE__,:physical_prior_fixture) || include("physical_parameters.jl")

@testset "Physical MAP, coordinate covariance and sampler target integration" begin
    mode=.2exp(-.25)
    for transform in (:auto,IdentityTransform(),LogTransform(),LogTransform(scale=7.))
        p=physical_prior_fixture(;transform).problem
        result=fit(p;nstarts=3,seed=51,gradient_tolerance=1e-7)
        @test result.converged
        @test result.parameters.shared.g0≈mode rtol=1e-7
        @test result.logprior≈logpdf(LogNormal(log(.2),.5),mode) atol=1e-12
        @test result.convention===:physical_map_v1
        @test result.diagnostics.density_coordinates===:physical
        lap=laplace_approximation(p,result)
        @test lap.covariance[1]≈.25mode^2 rtol=1e-7
    end
    # Known interior modes under both reflected/shifted and bounded support.
    for (prior,initial,expected,variance) in (
            (truncated(Normal(4.,1.),3.,Inf),5.,4.,1.),
            (truncated(Normal(-4.,1.),-Inf,-3.),-5.,-4.,1.),
            (Beta(2.,3.),.6,1/3,2/27))
        p=physical_prior_fixture(prior;initial).problem
        result=fit(p;gradient_tolerance=1e-7)
        @test result.converged
        @test result.parameters.shared.g0≈expected rtol=1e-7
        @test laplace_approximation(p,result).covariance[1]≈variance rtol=2e-7
    end
    boundary=fit(physical_prior_fixture(Exponential(.5)).problem)
    @test !boundary.converged
    @test !boundary.diagnostics.stationarity.valid
    @test occursin("boundary",boundary.diagnostics.stationarity.reason)
    @test_throws LaplaceFailure laplace_approximation(physical_prior_fixture(Exponential(.5)).problem,boundary)

    # A weak exponential prior and direct Gaussian observation have the exact
    # physical mode y-sigma^2/scale and Gaussian local variance sigma^2. The
    # remote prior scale must not make this ordinary interior fit fail checks.
    f=physical_prior_fixture(Exponential(1e12))
    observed=Measurement([0.],[.2],FunctionObservable((u,op,p,t)->p.g0);
        likelihood=GaussianLikelihood(sigma=.01))
    ex=InferenceExperiment(:batch,f.base,observed;solver=f.solver)
    broad=EstimationProblem(f.problem.parameters,(ex,))
    result=fit(broad;uncertainty=:gauss_newton)
    @test result.converged
    @test result.parameters.shared.g0≈.2-.01^2/1e12 atol=1e-14
    @test result.uncertainty.covariance[1]≈.01^2 rtol=1e-10

    # Independently derived conjugate physical Gaussian mode/covariance. All
    # coordinates are positive, but the local precision is unaltered by truncation.
    f=physical_prior_fixture()
    A=[1. 1.;2. -1.];y=[3.2,3.]
    prior_mean=[2.,1.];prior_variance=[1.,4.]
    H=A'*A/.25+Diagonal(1 ./prior_variance)
    C=H\Matrix{Float64}(I,2,2)
    theta=H\(A'*y/.25+prior_mean./prior_variance)
    measurement=Measurement([0.],reshape(y,:,1),FunctionObservable((u,op,p,t)->A*[p.a,p.b]);
        likelihood=GaussianLikelihood(sigma=.5))
    ex=InferenceExperiment(:batch,f.base,measurement;solver=f.solver)
    for transforms in ((:auto,:auto),(IdentityTransform(),IdentityTransform()),
            (LogTransform(scale=3.),LogTransform(scale=.1)))
        ps=[ParameterSpec(:a;initial=theta[1],prior=truncated(Normal(2.,1.),0.,Inf),transform=transforms[1]),
            ParameterSpec(:b;initial=theta[2],prior=truncated(Normal(1.,2.),0.,Inf),transform=transforms[2],role=:nuisance)]
        p=EstimationProblem(ps,(ex,))
        result=fit(p;gradient_tolerance=1e-7)
        @test result.converged
        @test [result.parameters.shared.a,result.parameters.shared.b]≈theta rtol=1e-8
        lap=laplace_approximation(p,result);gn=gauss_newton_approximation(p,result)
        @test lap.hessian≈H rtol=2e-10
        @test lap.covariance≈C rtol=2e-10
        @test gn.covariance≈C rtol=2e-10
        @test marginal_covariance(p,gn).covariance[1]≈C[1,1]
        @test abs(C[1,1]-inv(H[1,1]))>1e-4
    end

    # A modest exact Normal-in-log-coordinate target, including a nonunit
    # reference and a nonlinear auxiliary coordinate map. No PBM campaign.
    p=physical_prior_fixture(;transform=LogTransform(scale=3.)).problem
    result=fit(p);lap=laplace_approximation(p,result)
    for (map,seed) in ((nothing,705),
            ((to_sampler=z->asinh.(z),from_sampler=y->sinh.(y),
              logabsdetjac=y->sum(log.(cosh.(y))),label=:sinh),706))
        posterior=sample_posterior(p;initial=result,laplace=lap,sampling_map=map,
            warmup=250,samples=800,chains=2,seed,min_ess=40,target_acceptance=.9,max_depth=9)
        values=vec(posterior.z)
        @test abs(mean(values)-log(.2/3))<.08
        @test abs(var(values)-.25)<.05
        target_variance=map===nothing ? .25 : .25/(1+result.minimizer[1]^2)
        @test posterior.diagnostics.laplace_tuning_covariance[1]≈target_variance rtol=1e-7
        @test posterior.diagnostics.metric_initialization===:physical_laplace_local_tuning
        for c in 1:2,i in (1,200,800)
            z=posterior.z[:,i,c];stat=posterior.diagnostics.statistics[c][i]
            @test stat.log_density≈logpdf(Normal(log(.2/3),.5),z[1])-log(2pi)/2 atol=1e-10
            y=map===nothing ? z : map.to_sampler(z)
            jac=map===nothing ? 0. : map.logabsdetjac(y)
            @test stat.sampler_log_density≈stat.log_density+jac atol=1e-10
        end
    end
end
