using Test, PopulationBalanceModels, ForwardDiff, LinearAlgebra
import Distributions
using Distributions: Normal, LogNormal, Exponential, Gamma, Beta, Uniform,
    Poisson, truncated, ContinuousUnivariateDistribution, insupport, logpdf

# An independent constant-likelihood experiment: no kinetic contribution to the
# known scalar prior density. The generic forward/observation lifecycle is used.
function physical_prior_fixture(prior=LogNormal(log(.2),.5);initial=.2,transform=:auto,
        gradient=nothing,gradient_target=nothing)
    grid=Grid1D([0.,.5,1.])
    base=PBMProblem(compile(PBMModel(),grid),[1.,0.],(0.,1.);p=(g0=.9,))
    measurement=Measurement([0.],[0.],FunctionObservable((u,op,p,t)->zero(p.g0));
        likelihood=GaussianLikelihood(sigma=1.))
    solver=(pb,times)->PBMSolution(times,[copy(pb.u0) for _ in times],pb,0)
    experiment=InferenceExperiment(:batch,base,measurement;solver)
    spec=ParameterSpec(:g0;initial,prior,transform)
    problem=EstimationProblem((spec,),(experiment,);gradient,gradient_target)
    (;problem,base,solver,experiment)
end
function physical_analytic_map(problem,theta;convention=:physical_map_v1)
    z=PopulationBalanceModels._inference_coordinates(problem,theta)
    parts=posterior_components(problem,z)
    MAPResult(Float64.(z),physical_parameters(problem,z),parts.loglikelihood,
        parts.logprior,parts.logposterior,true,(source=:independent_analytic_mode,),
        ParameterSpec[problem.parameters...],nothing;convention)
end

# A generic support(min,max) fallback would falsely certify the gap (1,2).
struct DisconnectedPhysicalPrior <: ContinuousUnivariateDistribution end
Base.minimum(::DisconnectedPhysicalPrior)=0.
Base.maximum(::DisconnectedPhysicalPrior)=3.
Distributions.insupport(::DisconnectedPhysicalPrior,x::Real)=(0<=x<=1 || 2<=x<=3)
Distributions.logpdf(d::DisconnectedPhysicalPrior,x::Real)=insupport(d,x) ? -log(2.)+zero(x) : -Inf

@testset "Physical priors, automatic coordinates and invariant MAP target" begin
    @testset "Support contracts and physical starts" begin
        for (d,initial,T) in ((Normal(),.2,IdentityTransform),
                (LogNormal(),.2,LogTransform),(Exponential(2.),.2,LogTransform),
                (Gamma(2.,1.),.2,LogTransform),(Beta(2.,3.),.2,LogitTransform),
                (Uniform(-2.,3.),.2,LogitTransform),
                (truncated(Normal(),2.,Inf),3.,ShiftedLogTransform),
                (truncated(Normal(),-Inf,-2.),-3.,ShiftedLogTransform),
                (truncated(Normal(),-2.,3.),.2,LogitTransform))
            spec=ParameterSpec(:x;initial,prior=d)
            @test spec.transform isa T
            z=PopulationBalanceModels._coordinate(spec.transform,initial)
            @test PopulationBalanceModels._physical(spec.transform,z)≈initial
            @test abs(ForwardDiff.derivative(x->PopulationBalanceModels._physical(spec.transform,x),z)) ≈ exp(PopulationBalanceModels._logjac(spec.transform,z))
        end
        p=ParameterSpec(:x;initial=.2,prior=LogNormal(),transform=IdentityTransform())
        @test !p.automatic && p.transform isa IdentityTransform
        @test ParameterSpec(:x;initial=.2,prior=Prior(LogNormal();coordinates=:physical)).automatic
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=Prior(Normal();coordinates=:inference))
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=Normal(),transform=LogTransform())
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=Exponential(),transform=LogitTransform(0.,1.))
        @test_throws ArgumentError ParameterSpec(:x;initial=NaN,prior=Normal())
        @test_throws ArgumentError ParameterSpec(:x;initial=-.2,prior=LogNormal())
        @test_throws ArgumentError ParameterSpec(:x;initial=1.,prior=Uniform(0.,1.))
        @test_throws ArgumentError ParameterSpec(:x;initial=0.,prior=Beta(.5,2.),transform=IdentityTransform())
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=Poisson(2.))
        @test_throws ArgumentError ParameterSpec(:x;initial=.5,prior=DisconnectedPhysicalPrior())
        @test ParameterSpec(:x;initial=.5,prior=DisconnectedPhysicalPrior(),transform=IdentityTransform()).initial==.5
        @test_throws ArgumentError ParameterSpec(:x;initial=1.5,prior=DisconnectedPhysicalPrior(),transform=IdentityTransform())
        transform=LogTransform(scale=1e-300)
        @test isfinite(PopulationBalanceModels._physical(transform,700.))
        @test PopulationBalanceModels._coordinate(transform,
            PopulationBalanceModels._physical(transform,700.))≈700.
        @test !PopulationBalanceModels._transform_valid(LogTransform(),-1000.)
        @test !PopulationBalanceModels._transform_valid(LogitTransform(0.,1.),1000.)
        # A broad prior's SD is not a boundary-distance or stationarity scale.
        broad=physical_prior_fixture(Exponential(1e12)).problem
        @test PopulationBalanceModels._interior_start(broad,initial_coordinates(broad))
        report=PopulationBalanceModels._physical_gradient(broad,initial_coordinates(broad),[1.])
        @test report.scales==[.2]
    end

    @testset "Lognormal physical mode and both gradients" begin
        mu=log(.2);sigma=.5;mode=exp(mu-sigma^2)
        for transform in (:auto,IdentityTransform(),LogTransform(),LogTransform(scale=7.))
            f=physical_prior_fixture(;transform);p=f.problem
            @test f.base.p.g0==.9
            @test physical_parameters(p,initial_coordinates(p),:batch).g0≈.2
            @test parameterization(p)[1].map_density===:physical
            for x in (.07,mode,.2,.6)
                z=PopulationBalanceModels._inference_coordinates(p,[x])
                @test logposterior(p,z)≈logpdf(LogNormal(mu,sigma),x)-log(2pi)/2 atol=2e-13
                @test logposterior(p,[x];coordinates=:physical)≈logposterior(p,z) atol=2e-13
                @test check_gradient(p,z;target=:map).valid
                @test check_gradient(p,z;target=:sampling).valid
            end
            @test abs(only(posterior_gradient(p,[mode];coordinates=:physical)))<1e-12
            z=PopulationBalanceModels._inference_coordinates(p,[mode])
            @test logposterior(p,z)>logposterior(p,initial_coordinates(p))
            @test f.base.p.g0==.9 && f.base.u0==reshape([1.,0.],2,1)
        end
        @test mode≈.155760156614281 atol=1e-15
        for reference in (1.,3.)
            p=physical_prior_fixture(;transform=LogTransform(scale=reference)).problem
            for z in (-4.,-1.5,.3)
                @test sampling_logposterior(p,[z])≈logpdf(Normal(mu-log(reference),sigma),z)-log(2pi)/2 atol=1e-12
                @test only(sampling_gradient(p,[z]))≈-(z+log(reference)-mu)/sigma^2 atol=1e-12
            end
            map=(to_sampler=z->asinh.(z),from_sampler=y->sinh.(y),
                logabsdetjac=y->sum(log.(cosh.(y))),label=:nonlinear)
            for z in (-4.,-1.,1.)
                report=check_sampling_map(p,map,[z])
                y=asinh(z)
                expected=logpdf(Normal(mu-log(reference),sigma),sinh(y))+log(cosh(y))-log(2pi)/2
                @test report.valid
                @test report.logdensity_sampler≈expected atol=1e-12
                @test only(report.gradient)≈-(sinh(y)+log(reference)-mu)/sigma^2*cosh(y)+tanh(y) atol=1e-11
            end
        end
        advanced=physical_prior_fixture(Prior(Normal(mu,sigma);coordinates=:inference);
            transform=LogTransform()).problem
        @test logprior(advanced,[mu])≈logpdf(LogNormal(mu,sigma),.2)
        @test sampling_logposterior(advanced,[mu])≈logpdf(Normal(mu,sigma),mu)-log(2pi)/2
        @test abs(only(posterior_gradient(advanced,[mode];coordinates=:physical)))<1e-12
        old_gradient=z->[-(z[1]-mu)/sigma^2]
        p=physical_prior_fixture(;gradient=old_gradient,gradient_target=:sampling).problem
        @test check_gradient(p,[mu];target=:map).valid
        @test check_gradient(p,[mu];target=:sampling).valid
        @test only(posterior_gradient(p,[mu]))≈-1
        @test_throws ArgumentError physical_prior_fixture(;gradient=old_gradient)
    end

    @testset "Candidate parameters and once-only prior factorization" begin
        f=physical_prior_fixture()
        m=Measurement([0.,1.],[.2,.2],FunctionObservable((u,op,p,t)->p.g0);
            likelihood=GaussianLikelihood(sigma=.1))
        a=InferenceExperiment(:a,f.base,m;solver=f.solver)
        b=InferenceExperiment(:b,f.base,m;solver=f.solver)
        p=EstimationProblem(f.problem.parameters,(a,b))
        @test all(P->P==fill(.2,1,2),(predictions(p,initial_coordinates(p))[1][1],
            predictions(p,initial_coordinates(p))[2][1]))
        z=[log(.3)]
        @test all(P->P≈fill(.3,1,2),(predictions(p,z)[1][1],predictions(p,z)[2][1]))
        @test logposterior(p,z)≈4logpdf(Normal(.3,.1),.2)+logpdf(LogNormal(log(.2),.5),.3)
        @test f.base.p.g0==.9
    end

    @testset "Physical Laplace/GN curvature and unchanged numerical transforms" begin
        f=physical_prior_fixture();sigma=.5;mode=.2exp(-sigma^2)
        expected=sigma^2*mode^2
        for transform in (:auto,IdentityTransform(),LogTransform(scale=3.))
            p=physical_prior_fixture(;transform).problem
            result=physical_analytic_map(p,[mode])
            lap=laplace_approximation(p,result)
            gn=gauss_newton_approximation(p,result)
            @test lap.center≈[mode] && gn.center≈[mode]
            @test lap.covariance[1]≈expected rtol=2e-10
            @test gn.covariance≈lap.covariance rtol=2e-10
            @test marginal_covariance(p,lap).coordinates===:physical
            @test lap.hessian[1]≈inv(expected) rtol=1e-12
        end
        p=f.problem
        old=physical_analytic_map(p,[mode];convention=:unspecified)
        @test_throws ArgumentError laplace_approximation(p,old)
        @test_throws ArgumentError gauss_newton_approximation(p,old)
        # A vanishing transform derivative cannot certify an endpoint optimum.
        endpoint=physical_prior_fixture(Exponential(.5)).problem
        check=PopulationBalanceModels._map_stationarity(endpoint,[-30.],posterior_gradient(endpoint,[-30.]))
        @test !check.valid && check.boundary_limited
        # At a zero-valued physical mode, a huge prior SD must not expand a
        # local derivative stencil into the nonlinear forward model's tails.
        # y=exp(theta), y_obs=1, sigma=.1 has mode 0 and precision 100+1e-24.
        f=physical_prior_fixture(Normal(0.,1e12);initial=0.)
        m=Measurement([0.],[1.],FunctionObservable((u,op,p,t)->exp(p.g0));
            likelihood=GaussianLikelihood(sigma=.1))
        ex=InferenceExperiment(:batch,f.base,m;solver=f.solver)
        p=EstimationProblem(f.problem.parameters,(ex,))
        result=physical_analytic_map(p,[0.])
        @test laplace_approximation(p,result).covariance[1]≈inv(100+1e-24) rtol=1e-10
        @test gauss_newton_approximation(p,result).covariance[1]≈inv(100+1e-24) rtol=1e-10
    end

    @testset "Physical Gaussian draws retain support failures" begin
        p=physical_prior_fixture(LogNormal(log(.2),2.)).problem
        mode=.2exp(-4.)
        lap=laplace_approximation(p,physical_analytic_map(p,[mode]))
        draws=parameter_draws(p,lap;n=1000,seed=39)
        invalid=findall(!,draws.in_support)
        @test length(invalid)>100
        @test all(draws.theta[1,invalid].<0)
        @test all(isnan,draws.z[:,invalid])
        @test all(draws.physical[i].shared.g0<0 for i in invalid)
        @test draws.draw_coordinates===:physical
        prediction=posterior_predictive(p,draws;max_solves=1000)
        @test length(prediction.results)==1000
        @test prediction.failed_draws==length(invalid)
        @test prediction.solves==1000-length(invalid)
        @test_throws ArgumentError posterior_predictive(p,draws;max_solves=999)
    end
end
