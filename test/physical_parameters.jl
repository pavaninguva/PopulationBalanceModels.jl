using Test, PopulationBalanceModels, ForwardDiff, LinearAlgebra
import Distributions
using Distributions: Normal, LogNormal, Exponential, Gamma, Beta, Uniform,
    Poisson, truncated, ContinuousUnivariateDistribution, insupport, logpdf, logcdf

# Parameter-independent forward likelihood, with a distinct standalone nominal p.
function coordinate_prior_fixture(prior=LogNormal(log(.2),.5);initial=.2,
        transform=LogTransform(),gradient=nothing,gradient_target=nothing)
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
function coordinate_analytic_map(problem,z;convention=:inference_map_v2)
    parts=posterior_components(problem,z)
    MAPResult(Float64.(z),physical_parameters(problem,z),parts.loglikelihood,
        parts.logprior,parts.logposterior,true,(source=:independent_analytic_mode,),
        ParameterSpec[problem.parameters...],nothing;convention)
end

struct DisconnectedPhysicalPrior <: ContinuousUnivariateDistribution end
Base.minimum(::DisconnectedPhysicalPrior)=0.
Base.maximum(::DisconnectedPhysicalPrior)=3.
Distributions.insupport(::DisconnectedPhysicalPrior,x::Real)=(0<=x<=1 || 2<=x<=3)
Distributions.logpdf(d::DisconnectedPhysicalPrior,x::Real)=insupport(d,x) ? -log(2.)+zero(x) : -Inf

@testset "Explicit physical priors and one declared-coordinate posterior" begin
    @testset "Required transforms, support and physical starts" begin
        @test_throws UndefKeywordError ParameterSpec(:x;initial=.2,prior=LogNormal())
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=LogNormal(),transform=:auto)
        for (d,initial,transform) in ((Normal(),.2,IdentityTransform()),
                (LogNormal(),.2,LogTransform()),(Exponential(2.),.2,LogTransform()),
                (Gamma(2.,1.),.2,LogTransform()),(Beta(2.,3.),.2,LogitTransform(0.,1.)),
                (Uniform(-2.,3.),.2,LogitTransform(-2.,3.)),
                (truncated(Normal(),2.,Inf),3.,ShiftedLogTransform(2.)),
                (truncated(Normal(),-Inf,-2.),-3.,ShiftedLogTransform(-2.;upper=true)))
            spec=ParameterSpec(:x;initial,prior=d,transform)
            z=PopulationBalanceModels._coordinate(transform,initial)
            @test PopulationBalanceModels._physical(transform,z)≈initial
            @test abs(ForwardDiff.derivative(x->PopulationBalanceModels._physical(transform,x),z))≈exp(PopulationBalanceModels._logjac(transform,z))
        end
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=Normal(),transform=LogTransform())
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=Exponential(),transform=LogitTransform(0.,1.))
        @test_throws ArgumentError ParameterSpec(:x;initial=NaN,prior=Normal(),transform=IdentityTransform())
        @test_throws ArgumentError ParameterSpec(:x;initial=-.2,prior=LogNormal(),transform=LogTransform())
        @test_throws ArgumentError ParameterSpec(:x;initial=1.,prior=Uniform(0.,1.),transform=LogitTransform(0.,1.))
        @test_throws ArgumentError ParameterSpec(:x;initial=.2,prior=Poisson(2.),transform=IdentityTransform())
        @test ParameterSpec(:x;initial=.5,prior=DisconnectedPhysicalPrior(),transform=IdentityTransform()).initial==.5
        @test_throws ArgumentError ParameterSpec(:x;initial=1.5,prior=DisconnectedPhysicalPrior(),transform=IdentityTransform())
        transform=LogTransform(scale=1e-300)
        @test isfinite(PopulationBalanceModels._physical(transform,700.))
        @test PopulationBalanceModels._coordinate(transform,PopulationBalanceModels._physical(transform,700.))≈700.
        @test !PopulationBalanceModels._transform_valid(LogTransform(),-1000.)
        @test !PopulationBalanceModels._transform_valid(LogitTransform(0.,1.),1000.)
    end

    @testset "Known modes, densities, gradients and curvature" begin
        mu=log(.2);sigma=.5
        for scale in (1.,3.)
            p=coordinate_prior_fixture(;transform=LogTransform(;scale)).problem
            mode=mu-log(scale)
            @test parameterization(p)[1].map_density===:inference
            for z in (-4.,mode,.3)
                exact=logpdf(Normal(mu-log(scale),sigma),z)-log(2pi)/2
                @test logposterior(p,[z])≈exact atol=1e-12
                @test sampling_logposterior(p,[z])==logposterior(p,[z])
                @test logprior(p,[z])≈logpdf(Normal(mu-log(scale),sigma),z) atol=1e-12
                @test only(posterior_gradient(p,[z]))≈-(z-mode)/sigma^2 atol=1e-12
                @test sampling_gradient(p,[z])==posterior_gradient(p,[z])
                @test check_gradient(p,[z]).valid
                @test check_gradient(p,[z];target=:sampling).valid
                theta=scale*exp(z)
                @test logposterior(p,[theta];coordinates=:physical)≈exact atol=1e-12
                @test posterior_components(p,[theta];coordinates=:physical).density_coordinates===:inference
            end
            result=coordinate_analytic_map(p,[mode])
            @test result.parameters.shared.g0≈.2
            @test PopulationBalanceModels._map_stationarity(p,[mode],posterior_gradient(p,[mode])).valid
            lap=laplace_approximation(p,result);gn=gauss_newton_approximation(p,result)
            @test lap.center≈[mode] && gn.center≈[mode]
            @test lap.covariance[1]≈.25 rtol=1e-10
            @test gn.covariance[1]≈.25 rtol=1e-10
            @test marginal_covariance(p,lap).coordinates===:inference
        end
        # Identity means a different reference density and hence a different mode.
        p=coordinate_prior_fixture(;transform=IdentityTransform()).problem
        mode=.2exp(-.25)
        @test mode≈.155760156614281 atol=1e-15
        @test abs(only(posterior_gradient(p,[mode])))<1e-12
        @test laplace_approximation(p,coordinate_analytic_map(p,[mode])).covariance[1]≈.25mode^2 rtol=1e-10

        # Positive-truncated physical Normal: solve x(x-mu)=sigma^2.
        mu=.2;sigma=.05;x=(mu+sqrt(mu^2+4sigma^2))/2
        @test x≈.2118033988749895 atol=1e-15
        p=coordinate_prior_fixture(truncated(Normal(mu,sigma),0.,Inf)).problem
        for z in (-3.,log(x),-.5)
            theta=exp(z)
            exact=logpdf(Normal(mu,sigma),theta)-logcdf(Normal(),mu/sigma)+z
            @test logprior(p,[z])≈exact atol=2e-12
            @test only(posterior_gradient(p,[z]))≈1-theta*(theta-mu)/sigma^2 atol=1e-11
        end
        result=coordinate_analytic_map(p,[log(x)])
        precision=(2x^2-mu*x)/sigma^2
        @test laplace_approximation(p,result).hessian[1]≈precision rtol=1e-10
        @test gauss_newton_approximation(p,result).prior_hessian[1]≈precision rtol=1e-10
        @test_throws ArgumentError laplace_approximation(p,coordinate_analytic_map(p,[log(x)];convention=:physical_map_v1))
        @test_throws ArgumentError gauss_newton_approximation(p,coordinate_analytic_map(p,[log(x)];convention=:unspecified))
        @test !PopulationBalanceModels._map_stationarity(p,[-1000.],[0.]).valid
    end

    @testset "Equivalent priors, callback contracts and auxiliary maps" begin
        mu=log(.2);sigma=.5;scale=3.
        priors=(LogNormal(mu,sigma),Prior(LogNormal(mu,sigma);coordinates=:physical),
            Prior(Normal(mu-log(scale),sigma);coordinates=:inference))
        for prior in priors
            p=coordinate_prior_fixture(prior;transform=LogTransform(;scale)).problem
            mapping=(to_sampler=z->asinh.(z),from_sampler=y->sinh.(y),
                logabsdetjac=y->sum(log.(cosh.(y))),label=:nonlinear)
            for z in (-4.,-1.,1.)
                report=check_sampling_map(p,mapping,[z]);y=asinh(z)
                @test report.valid
                @test report.logdensity_sampler≈logpdf(Normal(mu-log(scale),sigma),sinh(y))+log(cosh(y))-log(2pi)/2 atol=1e-12
                @test only(report.gradient)≈-(sinh(y)-mu+log(scale))/sigma^2*cosh(y)+tanh(y) atol=1e-11
            end
        end
        gradient=z->[-(z[1]-mu)/sigma^2]
        p=coordinate_prior_fixture(;gradient).problem
        @test check_gradient(p,[mu]).valid
        legacy=z->gradient(z).-1
        p=coordinate_prior_fixture(;gradient=legacy,gradient_target=:physical).problem
        @test check_gradient(p,[mu]).valid
        @test only(posterior_gradient(p,[mu]))≈0 atol=1e-12
        bad=coordinate_prior_fixture(;gradient=legacy).problem
        @test !check_gradient(bad,[mu]).valid
        @test_throws ArgumentError coordinate_prior_fixture(;gradient,gradient_target=:unknown)
    end

    @testset "Physical candidates, joint experiments and once-only priors" begin
        f=coordinate_prior_fixture()
        m=Measurement([0.,1.],[.2,.2],FunctionObservable((u,op,p,t)->p.g0);
            likelihood=GaussianLikelihood(sigma=.1))
        a=InferenceExperiment(:a,f.base,m;solver=f.solver);b=InferenceExperiment(:b,f.base,m;solver=f.solver)
        p=EstimationProblem(f.problem.parameters,(a,b))
        @test all(P->P==fill(.2,1,2),(predictions(p,initial_coordinates(p))[1][1],predictions(p,initial_coordinates(p))[2][1]))
        z=[log(.3)]
        @test all(P->P≈fill(.3,1,2),(predictions(p,z)[1][1],predictions(p,z)[2][1]))
        @test logposterior(p,z)≈4logpdf(Normal(.3,.1),.2)+logpdf(LogNormal(log(.2),.5),.3)+log(.3)
        @test f.base.p.g0==.9 && f.base.u0==reshape([1.,0.],2,1)
    end

    @testset "Gaussian z draws preserve support and transform failures" begin
        p=coordinate_prior_fixture().problem
        lap=laplace_approximation(p,coordinate_analytic_map(p,[log(.2)]))
        draws=parameter_draws(p,lap;n=500,seed=39)
        @test all(draws.in_support) && all(draws.transform_valid)
        @test draws.theta≈exp.(draws.z)
        @test draws.draw_coordinates===:inference
        @test posterior_predictive(p,draws;max_solves=500).failed_draws==0
        @test_throws ArgumentError posterior_predictive(p,draws;max_solves=499)
        f=coordinate_prior_fixture(Uniform(-1.,1.);initial=0.,transform=IdentityTransform())
        m=Measurement([0.],[0.],FunctionObservable((u,op,p,t)->p.g0);likelihood=GaussianLikelihood(sigma=2.))
        p=EstimationProblem(f.problem.parameters,(InferenceExperiment(:batch,f.base,m;solver=f.solver),))
        draws=parameter_draws(p,laplace_approximation(p,coordinate_analytic_map(p,[0.]));n=1000,seed=41)
        @test count(!,draws.in_support)>500
        @test all(draws.transform_valid) && draws.theta==draws.z
        @test posterior_predictive(p,draws;max_solves=1000).failed_draws==count(!,draws.in_support)
        # Very wide log coordinates expose overflow/underflow, without redraws.
        p=coordinate_prior_fixture(LogNormal(0.,1000.);initial=1.).problem
        draws=parameter_draws(p,laplace_approximation(p,coordinate_analytic_map(p,[0.]));n=1000,seed=42)
        @test count(!,draws.transform_valid)>200
        @test any(!isfinite,draws.theta) && all(isfinite,draws.z)
        @test size(draws.z,2)==1000
    end
end
