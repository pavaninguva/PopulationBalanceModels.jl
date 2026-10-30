using Test, PopulationBalanceModels, Optim, AdvancedHMC, MCMCDiagnosticTools
using LinearAlgebra, Statistics
using Distributions: Normal, LogNormal, Exponential, Beta, truncated, logpdf
isdefined(@__MODULE__,:coordinate_prior_fixture) || include("physical_parameters.jl")

@testset "Declared-coordinate MAP, joint covariance and NUTS integration" begin
    for (transform,expected_z,expected_theta,variance) in (
            (LogTransform(),log(.2),.2,.25),
            (LogTransform(scale=7.),log(.2/7),.2,.25),
            (IdentityTransform(),.2exp(-.25),.2exp(-.25),.25*(.2exp(-.25))^2))
        p=coordinate_prior_fixture(;transform).problem
        result=fit(p;nstarts=3,seed=51,gradient_tolerance=1e-7)
        @test result.converged
        @test result.minimizer≈[expected_z] rtol=1e-7
        @test result.parameters.shared.g0≈expected_theta rtol=1e-7
        @test result.convention===:inference_map_v2
        @test result.diagnostics.density_coordinates===:inference
        @test laplace_approximation(p,result).covariance[1]≈variance rtol=2e-7
    end
    mu=.2;sigma=.05;mode=(mu+sqrt(mu^2+4sigma^2))/2
    p=coordinate_prior_fixture(truncated(Normal(mu,sigma),0.,Inf)).problem
    result=fit(p;gradient_tolerance=1e-7)
    @test result.converged
    @test result.parameters.shared.g0≈mode rtol=1e-7
    @test laplace_approximation(p,result).covariance[1]≈sigma^2/(2mode^2-mu*mode) rtol=1e-7

    # Under log coordinates, an exponential prior has a regular z-mode at its
    # physical mean. Identity coordinates retain its nonregular endpoint mode.
    log_problem=coordinate_prior_fixture(Exponential(.5)).problem
    log_fit=fit(log_problem;uncertainty=:laplace)
    @test log_fit.converged
    @test log_fit.parameters.shared.g0≈.5 rtol=1e-6
    @test log_fit.uncertainty.covariance[1]≈1. rtol=1e-6
    identity=coordinate_prior_fixture(Exponential(.5);transform=IdentityTransform()).problem
    boundary=fit(identity)
    @test !boundary.converged
    @test_throws LaplaceFailure laplace_approximation(identity,boundary)

    # A Gaussian likelihood in PHYSICAL g0 is nonlinear in z. Its full Hessian
    # and GN information differ by the independently known residual curvature.
    f=coordinate_prior_fixture(Exponential(1e12))
    observed=Measurement([0.],[.2],FunctionObservable((u,op,p,t)->p.g0);
        likelihood=GaussianLikelihood(sigma=.01))
    ex=InferenceExperiment(:batch,f.base,observed;solver=f.solver)
    problem=EstimationProblem(f.problem.parameters,(ex,))
    result=fit(problem;gradient_tolerance=1e-7)
    a=.2-.01^2/1e12;theta=(a+sqrt(a^2+4*.01^2))/2
    @test result.converged
    @test result.parameters.shared.g0≈theta rtol=1e-8
    @test gauss_newton_approximation(problem,result).precision[1]≈theta^2/.01^2+theta/1e12 rtol=1e-8
    @test laplace_approximation(problem,result).hessian[1]≈(2theta^2-.2theta)/.01^2+theta/1e12 rtol=1e-8

    # Exact joint Gaussian in mixed, nonunit declared coordinates. The physical
    # g0 prior is LogNormal; the physical nuisance b prior is Normal(0,2).
    A=[1. 1.;2. -1.];y=[-1.,-6.];prior_mean=[log(.2/3),0.]
    H=A'*A/.25+Diagonal([4.,1.]);C=H\Matrix{Float64}(I,2,2)
    mode=H\(A'*y/.25+[4prior_mean[1],0.])
    measurement=Measurement([0.],reshape(y,:,1),
        FunctionObservable((u,op,p,t)->A*[log(p.g0/3),p.b/2]);likelihood=GaussianLikelihood(sigma=.5))
    ex=InferenceExperiment(:batch,f.base,measurement;solver=f.solver)
    ps=[ParameterSpec(:g0;initial=.2,prior=LogNormal(log(.2),.5),transform=LogTransform(scale=3.)),
        ParameterSpec(:b;initial=0.,prior=Normal(0.,2.),transform=IdentityTransform(scale=2.),role=:nuisance)]
    p=EstimationProblem(ps,(ex,));result=fit(p;gradient_tolerance=1e-7)
    @test result.converged
    @test result.minimizer≈mode rtol=1e-7
    lap=laplace_approximation(p,result);gn=gauss_newton_approximation(p,result)
    @test lap.hessian≈H rtol=1e-10
    @test lap.covariance≈C rtol=1e-10
    @test gn.covariance≈C rtol=1e-10
    @test gn.prior_hessian≈Diagonal([4.,1.]) rtol=1e-10
    @test marginal_covariance(p,gn).covariance[1]≈C[1,1]
    @test abs(C[1,1]-inv(H[1,1]))>1e-4
    @test likelihood_gradient(p,result.minimizer)≈-(A'*A*mode-A'*y)/.25 atol=1e-7
    @test identifiability_report(p,result.minimizer).likelihood_hessian≈A'*A/.25 rtol=1e-9

    # Likelihood-only profiling may find its optimum outside prior support
    # when the declared transform and physical model permit that value.
    f=coordinate_prior_fixture(Beta(2.,2.);initial=.6,transform=IdentityTransform())
    m=Measurement([0.],[2.],FunctionObservable((u,op,p,t)->p.g0);likelihood=GaussianLikelihood(sigma=1.))
    p=EstimationProblem(f.problem.parameters,(InferenceExperiment(:batch,f.base,m;solver=f.solver),))
    result=fit(p)
    @test result.converged
    profile=estimation_profile(p,result,1,[0.,1.,2.];target=:likelihood)
    @test profile.reference≈[2.] atol=1e-6
    @test profile.delta≈[2.,.5,0.] atol=1e-7
    @test profile.density_coordinates===:measurement

    # Actual short NUTS runs against a known log-coordinate Normal, with and
    # without a nonlinear auxiliary map. These are tests, not a PBM campaign.
    p=coordinate_prior_fixture(;transform=LogTransform(scale=3.)).problem
    result=fit(p);lap=laplace_approximation(p,result)
    for (mapping,seed) in ((nothing,705),
            ((to_sampler=z->asinh.(z),from_sampler=y->sinh.(y),
              logabsdetjac=y->sum(log.(cosh.(y))),label=:sinh),706))
        posterior=sample_posterior(p;initial=result,laplace=lap,sampling_map=mapping,
            warmup=250,samples=800,chains=2,seed,min_ess=40,target_acceptance=.9,max_depth=9)
        values=vec(posterior.z)
        @test abs(mean(values)-log(.2/3))<.08
        @test abs(var(values)-.25)<.05
        tuning=mapping===nothing ? .25 : .25/(1+result.minimizer[1]^2)
        @test posterior.diagnostics.laplace_tuning_covariance[1]≈tuning rtol=1e-7
        @test posterior.diagnostics.metric_initialization===:inference_laplace_tuning
        for c in 1:2,i in (1,200,800)
            z=posterior.z[:,i,c];stat=posterior.diagnostics.statistics[c][i]
            @test stat.log_density≈logpdf(Normal(log(.2/3),.5),z[1])-log(2pi)/2 atol=1e-10
            jac=mapping===nothing ? 0. : mapping.logabsdetjac(mapping.to_sampler(z))
            @test stat.sampler_log_density≈stat.log_density+jac atol=1e-10
        end
    end
end
