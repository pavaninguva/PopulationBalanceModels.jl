using ForwardDiff
using Distributions: Normal, LogNormal, logpdf, truncated

@testset "AD defaults and nonnegative density noise" begin
    f=bayesian_linear_fixture();p=f.problem;z=[0.7,0.4]
    @test p.derivatives===:auto
    @test check_gradient(p,z).backend===:forwarddiff
    @test posterior_gradient(p,z)≈f.precision*(f.mode-z) rtol=1e-12
    @test posterior_gradient(p,z;derivatives=:finite_difference)≈posterior_gradient(p,z) rtol=1e-8
    @test PopulationBalanceModels._differentiate_hessian(v->-logposterior(p,v),z)≈f.precision rtol=1e-12
    # AD traverses one likelihood rather than the 2n central-difference solves.
    ad=PopulationBalanceModels._InferenceCounts();fd=PopulationBalanceModels._InferenceCounts()
    posterior_gradient(p,z;_counts=ad)
    posterior_gradient(p,z;derivatives=:finite_difference,_counts=fd)
    @test ad.solves==1
    @test fd.solves==2length(z)
    # Storage that strips dual numbers must fail explicitly, not return zero.
    bad(v)=Float64(v[1])^2
    @test_throws DifferentiationFailure PopulationBalanceModels._differentiate_gradient(bad,[1.])
    @test PopulationBalanceModels._differentiate_gradient(bad,[1.];derivatives=:finite_difference)≈[2.]
    @test_throws ArgumentError EstimationProblem(p.parameters,p.experiments;derivatives=:unknown)

    ll=PopulationBalanceModels._measurement_loglikelihood
    ex=first(p.experiments);Y=reshape([0.,0.2,1.4],:,1);P=reshape([0.,0.1,1.2],:,1)
    model=TruncatedGaussianLikelihood(sigma=(theta,ex)->theta.s)
    actual(s)=ll(model,Y,P,(s=s,),ex)
    reference(s)=sum(logpdf(truncated(Normal(P[i],s),0.,Inf),Y[i]) for i in eachindex(Y))
    @test actual(0.4)≈reference(0.4)
    @test ForwardDiff.derivative(actual,0.4)≈(reference(0.400001)-reference(0.399999))/2e-6 rtol=1e-7
    @test ll(TruncatedGaussianLikelihood(sigma=1.),zeros(1,1),zeros(1,1),(;),ex)≈log(2)-log(2π)/2
    @test_throws ArgumentError Measurement([0.],[-0.1],MomentObservable(0);likelihood=TruncatedGaussianLikelihood(sigma=1.))
    # Zero-count bins retain the correct information under AD too.
    multinomial(v)=ll(MultinomialLikelihood(),reshape([0.,4.,0.],:,1),reshape([exp(v),one(v),zero(v)],:,1),(;),ex)
    poisson(v)=ll(PoissonLikelihood(exposure=2.),zeros(2,1),reshape([exp(v),zero(v)],:,1),(;),ex)
    @test ForwardDiff.derivative(multinomial,0.3)≈-4exp(0.3)/(1+exp(0.3))
    @test ForwardDiff.derivative(poisson,0.3)≈-2exp(0.3)
end

@testset "Differentiating exact moments and initial conditions" begin
    g=MomentSet(0:3)
    op=compile(PBMModel(Death(PolynomialRate(((p,t,c)->p.k,)))),g)
    pb=PBMProblem(op,[1.,1.,2.,6.],(0.,1.);p=(k=0.4,))
    ts=[0.,0.4,1.]
    obs=Measurement(ts,exp.(-0.4.*ts),MomentObservable(0);likelihood=GaussianLikelihood(sigma=0.1))
    setup=(theta,base)->remake(base;p=theta,u0=theta.initial.*base.u0)
    ex=InferenceExperiment(:a,pb,obs;setup,dt=0.001)
    specs=[ParameterSpec(:k;initial=0.3,prior=Prior(Normal(0.4,1.);coordinates=:physical),transform=IdentityTransform()),
        ParameterSpec(:initial;initial=1.1,prior=Prior(Normal(1.,1.);coordinates=:physical),role=:nuisance,transform=IdentityTransform())]
    problem=EstimationProblem(specs,(ex,));z=initial_coordinates(problem)
    expected=[sum((exp.(-z[1].*ts).*z[2].-exp.(-0.4.*ts)).*ts.*z[2].*exp.(-z[1].*ts))/0.01,
        -sum((exp.(-z[1].*ts).*z[2].-exp.(-0.4.*ts)).*exp.(-z[1].*ts))/0.01]
    @test likelihood_gradient(problem,z)≈expected rtol=1e-8
    @test check_gradient(problem,z).valid
end

@testset "Fixed-grid size-dependent growth sensitivities" begin
    g=Grid1D(range(0.,2.;length=31);coordinate=:length)
    op=compile(PBMModel(growth=Growth((a,p,t,c)->p.k*(1+a))),g;
        growth_scheme=MUSCL(),bc=SizeBC(OpenBoundary(),OpenBoundary()))
    pb=PBMProblem(op,exp.(-20 .* (g.centers.-0.5).^2).*g.widths,(0.,0.2);p=(k=0.3,))
    m=Measurement([0.1,0.2],[0.4,0.4],MomentObservable(1);likelihood=GaussianLikelihood(sigma=0.1))
    par=ParameterSpec(:k;initial=0.3,transform=LogTransform(),prior=Prior(LogNormal();coordinates=:physical))
    p=EstimationProblem((par,),(InferenceExperiment(:batch,pb,m;dt=0.001),))
    z=initial_coordinates(p)
    @test check_gradient(p,z).valid
    @test abs(likelihood_gradient(p,z)[1])>0
end
