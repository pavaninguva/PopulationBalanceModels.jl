using Distributions: Normal, LogNormal, Exponential, Uniform, MvNormal, logpdf
using Random
using Distributions: Gamma, Beta, truncated

isdefined(@__MODULE__,:bayesian_linear_fixture) || include("fixtures/bayesian_linear.jl")

@testset "General continuous prior families and analytical AD" begin
    f=bayesian_linear_fixture()
    cases=(
        (Gamma(2.,.3),LogTransform(),(.1,.4,.8),
            x->2log(x)-x/.3-2log(.3),x->2-x/.3),
        (Beta(2.,3.),LogitTransform(0.,1.),(.1,.4,.8),
            x->log(12.)+2log(x)+3log1p(-x),x->2-5x),
        (truncated(Normal(0.,.3),0.,Inf),LogTransform(),(.1,.4,.8),
            x->log(2.)-log(.3sqrt(2pi))+log(x)-x^2/(2*.3^2),x->1-x^2/.3^2))
    for (distribution,transform,points,value,derivative) in cases
        spec=ParameterSpec(:x;initial=.4,transform,
            prior=Prior(distribution;coordinates=:physical))
        p=EstimationProblem((spec,),f.problem.experiments;fixed=(a=0.,b=0.))
        for x in points
            z=transform isa LogTransform ? log(x) : log(x/(1-x))
            @test logprior(p,[z])≈value(x) atol=2e-13
            @test ForwardDiff.derivative(v->logprior(p,[v]),z)≈derivative(x) atol=2e-13
        end
    end
end

@testset "One Bayesian density and coordinate conventions" begin
    f=bayesian_linear_fixture();p=f.problem;z=f.mode
    c=posterior_components(p,z)
    @test c.logposterior≈c.logprior+c.loglikelihood
    @test logposterior(p,z)≈logprior(p,z)+loglikelihood(p,z)
    @test loglikelihood(p,z)≈logpdf(MvNormal(f.A*z,f.C),f.y)
    @test check_gradient(p,[0.7,0.4]).valid
    @test posterior_gradient(p,[0.7,0.4])≈-(f.precision*([0.7,0.4]-f.mode)) rtol=1e-7
    bad=bayesian_linear_fixture(gradient=z->ones(2)).problem
    @test !check_gradient(bad,[0.7,0.4]).valid
    positive=[ParameterSpec(:rate;initial=1.0,transform=LogTransform(scale=2),prior=Prior(Exponential(0.5);coordinates=:physical))]
    ep=EstimationProblem(positive,p.experiments;fixed=(a=0.0,b=0.0))
    for x in (0.2,0.5,2.0)
        coordinate=log(x/2)
        @test logprior(ep,[coordinate])≈logpdf(Exponential(0.5),x)+log(x)
    end
    @test logprior(ep,[log(0.5/2)])>logprior(ep,[log(0.01/2)]) # z-mode θ=.5, physical mode θ=0.
    direct=replace_priors(ep,[Prior(Normal();coordinates=:inference)])
    @test logprior(direct,[0.4])≈logpdf(Normal(),0.4)
    @test_throws UndefKeywordError Prior(Normal())
    @test_throws ArgumentError ParameterSpec(:x;initial=0.0,transform=LogTransform(),prior=Prior(Normal();coordinates=:physical))
    t=LogitTransform(2,5)
    for x in (-100.0,0.0,100.0)
        @test isfinite(PopulationBalanceModels._logjac(t,x))
    end
    @test PopulationBalanceModels._physical(t,0.0)==3.5
end

@testset "Likelihood normalization, counts and fraction constraints" begin
    f=bayesian_linear_fixture();ex=first(f.problem.experiments);P=[0.0 1.0;0.0 3.0];Y=[0.0 0.0;0.0 4.0]
    ll=PopulationBalanceModels._measurement_loglikelihood
    @test ll(MultinomialLikelihood(),Y,P,(;),ex)≈4log(0.75)
    @test ll(MultinomialLikelihood(),2Y,P,(;),ex)≈8log(0.75)
    @test ll(PoissonLikelihood(exposure=2.0),zeros(2,2),P,(;),ex)≈-8.0
    @test ll(PoissonLikelihood(exposure=1.0),ones(2,2),P,(;),ex)==-Inf
    @test_throws NumericalFailure ll(MultinomialLikelihood(),Y,-P,(;),ex)
    g=DistributionObservable([0.0,0.5,1.0];coordinate=:volume,normalize=true)
    fractions=[0.3 0.4;0.7 0.6]
    @test_throws ArgumentError Measurement([0.0,1.0],fractions,g;likelihood=GaussianLikelihood(sigma=0.1))
    @test_throws ArgumentError Measurement([0.0,1.0],fractions,g;likelihood=MultinomialLikelihood())
    A=[1.0 -1.0];C=kron(Matrix{Float64}(I,2,2),[0.01 -0.01;-0.01 0.01])
    normal=GaussianLikelihood(covariance=C,contrasts=A)
    @test isfinite(ll(normal,fractions,fractions,(;),ex))
    noise=GaussianLikelihood(sigma=(theta,experiment)->theta.s)
    @test ll(noise,zeros(2,2),zeros(2,2),(s=2.0,),ex)-ll(noise,zeros(2,2),zeros(2,2),(s=1.0,),ex)≈-4log(2)
    # Independent density measurements must not allocate/factor a dense covariance.
    values=reshape(collect(range(-1.,1.;length=512)),:,1)
    gaussian=GaussianLikelihood(sigma=0.7)
    @test ll(gaussian,values,zeros(size(values)),(;),ex)≈sum(logpdf.(Normal(0,0.7),values))
    @test PopulationBalanceModels._gaussian(gaussian,values,values,(;),ex).factor.factors isa Diagonal
    a=Measurement([0.0],reshape([1.,0.],2,1),g;likelihood=MultinomialLikelihood(),source=:same)
    @test_throws ArgumentError InferenceExperiment(:run,f.pb,a,a;solver=f.solver)
end

@testset "Experiment-local nuisances, initial data and observation integrals" begin
    f=bayesian_linear_fixture();pb=f.pb
    obs=Measurement([0.0],[2.0],MomentObservable(0);likelihood=GaussianLikelihood(sigma=0.1),response=(P,theta,ex)->theta.gain.*P)
    setup=(theta,base)->remake(base;u0=theta.initial.*base.u0,p=theta)
    a=InferenceExperiment(:a,pb,obs;setup,solver=f.solver,fixed=(gain=2.0,))
    b=InferenceExperiment(:b,pb,obs;setup,solver=f.solver)
    specs=[ParameterSpec(:initial;initial=1.0,prior=Prior(LogNormal();coordinates=:physical),transform=LogTransform(),role=:nuisance),
        ParameterSpec(:gain;initial=3.0,prior=Prior(LogNormal();coordinates=:physical),transform=LogTransform(),experiment=:b,role=:nuisance)]
    ep=EstimationProblem(specs,(a,b));z=initial_coordinates(ep)
    @test [P[1][1] for P in predictions(ep,z)]≈[2.0,3.0]
    @test physical_parameters(ep,z,:a).gain==2.0
    @test physical_parameters(ep,z,:b).gain≈3.0
    @test_throws ArgumentError EstimationProblem([specs[1],ParameterSpec(:initial;initial=2.,prior=Prior(Normal();coordinates=:physical),experiment=:a)],(a,b))
    observable=DistributionObservable([0.0,0.25,1.0];coordinate=:volume,weight=x->x^3)
    values=PopulationBalanceModels._observe(observable,reshape([1.,0.],2,1),pb.operator,(;),0.0)
    @test values≈[0.25^4/2,(0.5^4-0.25^4)/2] rtol=1e-12
    @test_throws ArgumentError PopulationBalanceModels._observe(DistributionObservable([0.,0.5,1.];coordinate=:length),pb.u0,pb.operator,(;),0.)
    failed=InferenceExperiment(:run,pb,first(f.problem.experiments).measurements...;solver=(p,ts)->throw(NumericalFailure("test failure")))
    broken=EstimationProblem(f.problem.parameters,(failed,))
    @test logposterior(broken,[1.,0.5];on_failure=:reject)==-Inf
    @test_throws NumericalFailure logposterior(broken,[1.,0.5])
end

@testset "Actual posterior Hessian and nuisance marginal covariance" begin
    f=bayesian_linear_fixture();p=f.problem;parts=posterior_components(p,f.mode)
    result=MAPResult(f.mode,physical_parameters(p,f.mode),parts.loglikelihood,parts.logprior,parts.logposterior,true,(;),ParameterSpec[p.parameters...],nothing)
    lap=laplace_approximation(p,result)
    @test lap.physical_center==result.parameters
    @test [(p.name,p.experiment) for p in lap.coordinates]==[(:a,nothing),(:b,nothing)]
    @test lap.hessian≈f.precision rtol=2e-6
    @test lap.covariance≈f.covariance rtol=2e-6
    @test marginal_covariance(p,lap).covariance[1,1]≈f.covariance[1,1] rtol=2e-6
    @test abs(lap.covariance[1,1]-1/lap.hessian[1,1])>0.001
    draws=parameter_draws(p,lap;n=20,seed=10)
    @test draws.z==parameter_draws(p,lap;n=20,seed=10).z
    @test_throws ArgumentError posterior_predictive(p,draws;max_solves=19)
    pred=posterior_predictive(p,draws;max_solves=20,measurement_noise=true)
    @test pred.solves==20 && pred.failed_draws==0
    @test first(pred.results).measurements[1][1].observed!=first(pred.results).measurements[1][1].latent
    id=identifiability_report(p,f.mode)
    @test id.likelihood_hessian≈f.A'*(f.C\f.A) rtol=2e-6
    @test id.prior_hessian≈Matrix(Diagonal([0.25,1.0])) rtol=1e-6
    obs=Measurement([0.0],[1.0],FunctionObservable((N,op,p,t)->p.x^2);likelihood=GaussianLikelihood(sigma=0.1))
    ep=EstimationProblem([ParameterSpec(:x;initial=0.,prior=Prior(Normal();coordinates=:physical))],(InferenceExperiment(:run,f.pb,obs;solver=f.solver),))
    c=posterior_components(ep,[0.0])
    saddle=MAPResult([0.],physical_parameters(ep,[0.]),c.loglikelihood,c.logprior,c.logposterior,true,(;),ParameterSpec[ep.parameters...],nothing)
    @test_throws LaplaceFailure laplace_approximation(ep,saddle)
    uniform=Prior(Uniform(-1e6,1e6);coordinates=:physical)
    flat_obs=Measurement([0.0],[0.0],FunctionObservable((N,op,p,t)->p.a+p.b);likelihood=GaussianLikelihood(sigma=1.0))
    flat=EstimationProblem([ParameterSpec(:a;initial=0.,prior=uniform),ParameterSpec(:b;initial=0.,prior=uniform)],
        (InferenceExperiment(:run,f.pb,flat_obs;solver=f.solver),))
    c=posterior_components(flat,[0.,0.])
    ridge=MAPResult([0.,0.],physical_parameters(flat,[0.,0.]),c.loglikelihood,c.logprior,c.logposterior,true,(;),ParameterSpec[flat.parameters...],nothing)
    @test_throws LaplaceFailure laplace_approximation(flat,ridge)
end
