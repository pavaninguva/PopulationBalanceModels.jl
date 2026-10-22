using Test, LinearAlgebra, ForwardDiff
using Distributions: LogNormal, MvNormal, logpdf
using PopulationBalanceModels, SciMLBase, SciMLSensitivity, OrdinaryDiffEqTsit5, Optim

function sensitivity_fixture(mode)
    grid=Grid1D([0.,0.5,1.])
    pb=PBMProblem(compile(PBMModel(Death((a,p,t,c)->p.k)),grid),[1.,0.5],(0.,1.);p=(k=0.4,))
    ts=[0.,0.3,0.7,1.]
    specs=[ParameterSpec(:k;initial=0.3,transform=LogTransform(),prior=Prior(LogNormal(log(0.4),0.5);coordinates=:physical)),
        ParameterSpec(:initial;initial=1.1,transform=LogTransform(),prior=Prior(LogNormal(0.,0.2);coordinates=:physical),role=:nuisance),
        ParameterSpec(:gain;initial=0.9,transform=LogTransform(),prior=Prior(LogNormal(0.,0.2);coordinates=:physical),role=:nuisance,experiment=:b),
        ParameterSpec(:sigma;initial=0.12,transform=LogTransform(),prior=Prior(LogNormal(log(0.1),0.5);coordinates=:physical),role=:nuisance)]
    C=[0.2^abs(i-j) for i in eachindex(ts),j in eachindex(ts)]
    m=Measurement(ts,1.5exp.(-0.4.*ts),MomentObservable(0);
        likelihood=GaussianLikelihood(covariance=(theta,ex)->theta.sigma^2 .*C),
        response=(P,theta,ex)->theta.gain.*P)
    setup=(theta,base)->PopulationBalanceModels.remake(base;p=theta,u0=theta.initial.*base.u0)
    solver=SciMLSensitivitySolver(Tsit5();sensitivity=mode,reltol=1e-10,abstol=1e-12)
    a=InferenceExperiment(:a,pb,m;setup,solver,fixed=(gain=1.,))
    b=InferenceExperiment(:b,pb,m;setup,solver)
    EstimationProblem(specs,(a,b))
end

@testset "Continuous forward and adjoint sensitivities" begin
    forward=sensitivity_fixture(:forward);adjoint=sensitivity_fixture(:adjoint)
    z=initial_coordinates(forward)
    function exact_posterior(q)
        value=logprior(forward,q)
        for ex in forward.experiments
            theta=physical_parameters(forward,q,ex.name);m=only(ex.measurements)
            mu=theta.gain*theta.initial*1.5 .*exp.(-theta.k.*m.times)
            C=[0.2^abs(i-j) for i in eachindex(m.times),j in eachindex(m.times)]
            value+=logpdf(MvNormal(mu,theta.sigma^2 .*C),vec(m.data))
        end
        value
    end
    reference=ForwardDiff.gradient(exact_posterior,z)
    @test posterior_gradient(forward,z;derivatives=:forwarddiff)≈reference rtol=1e-7 atol=1e-7
    @test check_gradient(forward,z;derivatives=:forwarddiff).valid
    for problem in (forward,adjoint,sensitivity_fixture(:auto))
        counts=PopulationBalanceModels._InferenceCounts()
        gradient=posterior_gradient(problem,z;_counts=counts)
        @test gradient≈reference rtol=1e-6 atol=1e-6
        @test check_gradient(problem,z).valid
        @test counts.solves==2
        @test counts.adjoint_solves==(problem===adjoint ? 2 : 0)
    end
    # The Hessian differentiates the primal solve, including nuisance blocks.
    H=ForwardDiff.hessian(v->-logposterior(forward,v),z)
    @test H≈ForwardDiff.hessian(v->-logposterior(adjoint,v),z) rtol=1e-8
    fitforward=fit(forward;uncertainty=:laplace,maxiters=150)
    fitadjoint=fit(adjoint;maxiters=150)
    @test fitforward.converged && fitadjoint.converged
    @test fitforward.minimizer≈fitadjoint.minimizer rtol=1e-5 atol=1e-6
    @test fitforward.uncertainty.diagnostics.valid
    @test norm(fitforward.uncertainty.covariance[1,2:4])>0
    @test_throws ArgumentError SciMLSensitivitySolver(Tsit5();sensitivity=:bad)
    @test_throws ArgumentError SciMLSensitivitySolver(Tsit5();callback=nothing)
end
