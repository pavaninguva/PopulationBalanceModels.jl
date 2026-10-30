using Test, LinearAlgebra, Random, Statistics
using Distributions: Normal, LogNormal, Exponential, Uniform
isdefined(@__MODULE__, :bayesian_linear_fixture) || include("fixtures/bayesian_linear.jl")

# Test-only construction at analytically derived modes, not simulated optimizer
# convergence. Integration tests below exercise actual Optim results separately.
function analytic_map(problem,z; converged=true)
    parts=posterior_components(problem,z)
    MAPResult(Float64.(z),physical_parameters(problem,z),parts.loglikelihood,
        parts.logprior,parts.logposterior,converged,(origin=:analytic_reference,),
        ParameterSpec[problem.parameters...],nothing;convention=:inference_map_v2)
end

@testset "Sensitivity covariance equals the independent linear Gaussian posterior" begin
    f=bayesian_linear_fixture();p=f.problem;mode=analytic_map(p,f.mode)
    a=gauss_newton_approximation(p,mode)
    @test a isa GaussNewtonApproximation
    @test a.center==f.mode && a.physical_center==mode.parameters
    @test a.likelihood_information ≈ f.A'*(f.C\f.A) rtol=1e-12
    @test a.prior_hessian ≈ Matrix(Diagonal([.25,1.])) rtol=1e-12
    @test a.precision ≈ f.precision rtol=1e-12
    @test a.covariance ≈ f.covariance rtol=1e-12
    @test a.jacobian'*a.jacobian ≈ a.likelihood_information
    @test a.diagnostics.valid && a.diagnostics.data_rank==2
    @test a.diagnostics.information_error<1e-7
    @test a.diagnostics.gradient_verification<1e-7
    @test marginal_covariance(p,a).covariance[1,1]≈f.covariance[1,1]
    @test abs(a.covariance[1,1]-1/a.precision[1,1])>.001
    @test a.covariance≈laplace_approximation(p,mode).covariance rtol=2e-6
    fd=gauss_newton_approximation(p,mode;derivatives=:finite_difference,step=1e-4)
    @test fd.covariance≈f.covariance rtol=2e-5
    @test fd.diagnostics.derivative_backend===:finite_difference
    draws=parameter_draws(p,a;n=20000,seed=123)
    @test draws.method===:gauss_newton
    @test draws.z==parameter_draws(p,a;n=20000,seed=123).z
    @test norm(cov(permutedims(draws.z))-f.covariance)/norm(f.covariance)<.04
    small=parameter_draws(p,a;n=3)
    @test posterior_predictive(p,small;max_solves=3).solves==3
    @test_throws ArgumentError posterior_predictive(p,small;max_solves=2)
    @test_throws GaussNewtonFailure gauss_newton_approximation(p,analytic_map(p,f.mode;converged=false))
    @test_throws GaussNewtonFailure gauss_newton_approximation(p,analytic_map(p,f.mode.+1))
    @test_throws ArgumentError gauss_newton_approximation(p,mode;step=0.)
    @test_throws ArgumentError gauss_newton_approximation(p,mode;information_tolerance=1.)
    wrong=bayesian_linear_fixture(gradient=z->ones(2)).problem
    @test_throws GaussNewtonFailure gauss_newton_approximation(wrong,mode)
    # A change of parameter units rescales covariance, not its statistical meaning.
    obs=Measurement(first(p.experiments).measurements[1].times,f.y,
        FunctionObservable((u,op,p,t)->p.a/1000+p.b*t);likelihood=GaussianLikelihood(covariance=f.C))
    specs=[ParameterSpec(:a;initial=1000f.mode[1],prior=Prior(Normal(0,2000);coordinates=:physical),transform=IdentityTransform()),
        ParameterSpec(:b;initial=f.mode[2],role=:nuisance,prior=Prior(Normal();coordinates=:inference),transform=IdentityTransform())]
    scaled=EstimationProblem(specs,(InferenceExperiment(:run,f.pb,obs;solver=f.solver),))
    z=[1000f.mode[1],f.mode[2]]
    scale=Diagonal([1000.,1.])
    @test gauss_newton_approximation(scaled,analytic_map(scaled,z)).covariance≈scale*f.covariance*scale rtol=1e-11
    # Uniform unit rescaling can make every precision entry far below one;
    # that does not make an otherwise conditioned scalar problem singular.
    huge_units=EstimationProblem([ParameterSpec(:x;initial=0.,prior=Prior(Normal(0,1e12);coordinates=:physical),transform=IdentityTransform())],
        (InferenceExperiment(:run,f.pb,Measurement([0.],[0.],FunctionObservable((u,op,p,t)->p.x/1e12);
            likelihood=GaussianLikelihood(sigma=1.));solver=f.solver),))
    @test gauss_newton_approximation(huge_units,analytic_map(huge_units,[0.])).covariance[1,1]≈5e23 rtol=1e-12
end

@testset "Derivative verification resolves covariance directions rather than a global norm" begin
    J=Matrix(Diagonal([1e4,1e-4]));Q=Matrix(Diagonal([0.,.1]))
    U=cholesky(Symmetric(J'*J+Q)).U
    strong=copy(J);strong[1,1]+=.1
    # Independently known generalized eigenvalues of two diagonal precisions.
    expected=(strong[1,1]/J[1,1])^2-1
    @test PopulationBalanceModels._information_perturbation(strong,Q,J,Q,U)≈expected rtol=1e-10
    @test expected<1e-3
    weak=copy(J);weak[2,2]+=.1
    expected_weak=(weak[2,2]^2+.1)/(J[2,2]^2+.1)-1
    @test PopulationBalanceModels._information_perturbation(weak,Q,J,Q,U)≈expected_weak rtol=1e-12
    @test expected_weak>.01
end

@testset "Multiple experiments, joint nuisance covariance and observation responses" begin
    f=bayesian_linear_fixture();ts=[0.,1.,2.]
    specs=[ParameterSpec(:a;initial=0.,prior=Prior(Normal(0,2);coordinates=:physical),transform=IdentityTransform()),
        ParameterSpec(:b;initial=0.,experiment=:first,role=:nuisance,prior=Prior(Normal();coordinates=:inference),transform=IdentityTransform()),
        ParameterSpec(:b;initial=0.,experiment=:second,role=:nuisance,prior=Prior(Normal();coordinates=:inference),transform=IdentityTransform())]
    C=[.04 .01 0.;.01 .09 .01;0. .01 .16]
    y1=[.8,1.3,2.];y2=[1.1,.6,.2]
    # Response adds a shared offset; local observable contains the experiment's slope.
    observable=FunctionObservable((u,op,p,t)->p.b*t)
    m1=Measurement(ts,y1,observable;likelihood=GaussianLikelihood(covariance=C),response=(P,theta,ex)->P.+theta.a)
    m2=Measurement(ts,y2,observable;likelihood=GaussianLikelihood(sigma=[.2 .3 .4]),response=(P,theta,ex)->P.+theta.a)
    ex1=InferenceExperiment(:first,f.pb,m1;solver=f.solver)
    ex2=InferenceExperiment(:second,f.pb,m2;solver=f.solver)
    p=EstimationProblem(specs,(ex1,ex2))
    A=vcat(hcat(ones(3),ts,zeros(3)),hcat(ones(3),zeros(3),ts))
    V=zeros(6,6);V[1:3,1:3]=C;V[4:6,4:6]=Diagonal([.04,.09,.16])
    H=A'*(V\A)+Diagonal([.25,1.,1.]);z=H\(A'*(V\vcat(y1,y2)))
    a=gauss_newton_approximation(p,analytic_map(p,z))
    @test a.precision≈H rtol=1e-12
    @test a.covariance≈inv(H) rtol=1e-12
    @test a.diagnostics.observations==6
    @test marginal_covariance(p,a;role=:nuisance).indices==[2,3]
    @test a.covariance[1,1]>1/H[1,1]
end

@testset "Normalized fractions use independent covariance contrasts" begin
    f=bayesian_linear_fixture()
    observable=DistributionObservable([0.,.5,1.];coordinate=:volume,normalize=true)
    # Parameter-dependent response is a valid normalized number-fraction model.
    m=Measurement([0.],reshape([.7,.3],2,1),observable;
        likelihood=GaussianLikelihood(covariance=[.01 -.01;-.01 .01],contrasts=[1. -1.]),
        response=(P,theta,ex)->reshape([.5+theta.a,.5-theta.a],2,1))
    p=EstimationProblem([ParameterSpec(:a;initial=.1,prior=Prior(Normal();coordinates=:physical),transform=IdentityTransform())],
        (InferenceExperiment(:run,f.pb,m;solver=f.solver),))
    z=[20/101];a=gauss_newton_approximation(p,analytic_map(p,z))
    @test a.likelihood_information[1,1]≈100
    @test a.covariance[1,1]≈1/101
    @test a.diagnostics.observations==1
end

@testset "Physical priors, transformation Jacobians, support and nonlinear curvature" begin
    f=bayesian_linear_fixture()
    constant=Measurement([0.],[0.],FunctionObservable((u,op,p,t)->zero(p.rate));likelihood=GaussianLikelihood(sigma=1.))
    specs=[ParameterSpec(:rate;initial=2.,transform=LogTransform(scale=2.),prior=Prior(Exponential(2.);coordinates=:physical))]
    p=EstimationProblem(specs,(InferenceExperiment(:run,f.pb,constant;solver=f.solver),))
    a=gauss_newton_approximation(p,analytic_map(p,[0.]))
    @test a.prior_hessian[1,1]≈1
    @test a.covariance[1,1]≈1
    @test a.physical_center.shared.rate≈2
    @test a.diagnostics.data_rank==0 && a.diagnostics.prior_regularized
    # Real bounded physical priors contribute their logit-Jacobian curvature.
    ps=[ParameterSpec(:rate;initial=.5,transform=LogitTransform(0.,1.),prior=Prior(Uniform(0,1);coordinates=:physical))]
    bounded=EstimationProblem(ps,p.experiments)
    b=gauss_newton_approximation(bounded,analytic_map(bounded,[0.]))
    @test b.prior_hessian[1,1]≈.5
    # Identity-coordinate bounded prior: Gaussian draws outside support stay present.
    flat=EstimationProblem([ParameterSpec(:rate;initial=0.,prior=Prior(Uniform(-1,1);coordinates=:physical),transform=IdentityTransform())],
        (InferenceExperiment(:run,f.pb,Measurement([0.],[0.],FunctionObservable((u,op,p,t)->p.rate);
            likelihood=GaussianLikelihood(sigma=2.));solver=f.solver),))
    c=gauss_newton_approximation(flat,analytic_map(flat,[0.]))
    draws=parameter_draws(flat,c;n=1000,seed=7)
    @test size(draws.z,2)==1000
    @test count(!,draws.in_support)>500
    # At this exact nonlinear MAP, residual curvature is nonzero: H_full=4,
    # H_GN=4*x²+1=5. The two methods must not be silently conflated.
    nonlinear=Measurement([0.],[1.5],FunctionObservable((u,op,p,t)->p.x^2);likelihood=GaussianLikelihood(sigma=1.))
    q=EstimationProblem([ParameterSpec(:x;initial=1.,prior=Prior(Normal();coordinates=:physical),transform=IdentityTransform())],
        (InferenceExperiment(:run,f.pb,nonlinear;solver=f.solver),))
    g=gauss_newton_approximation(q,analytic_map(q,[1.]))
    @test g.precision[1,1]≈5
    @test laplace_approximation(q,analytic_map(q,[1.])).hessian[1,1]≈4 rtol=1e-5
end

@testset "Unavailable information and failed forward calculations stay visible" begin
    f=bayesian_linear_fixture();prior=Prior(Uniform(-100,100);coordinates=:physical)
    m=Measurement([0.],[0.],FunctionObservable((u,op,p,t)->p.a+p.b);likelihood=GaussianLikelihood(sigma=1.))
    p=EstimationProblem([ParameterSpec(:a;initial=0.,prior,transform=IdentityTransform()),ParameterSpec(:b;initial=0.,prior,transform=IdentityTransform())],
        (InferenceExperiment(:run,f.pb,m;solver=f.solver),))
    @test_throws GaussNewtonFailure gauss_newton_approximation(p,analytic_map(p,[0.,0.]))
    for likelihood in (GaussianLikelihood(sigma=(theta,ex)->theta.a),PoissonLikelihood(exposure=1.),
                       MultinomialLikelihood(),TruncatedGaussianLikelihood(sigma=1.))
        m=Measurement([0.],[1.],FunctionObservable((u,op,p,t)->p.a);likelihood)
        q=EstimationProblem([ParameterSpec(:a;initial=1.,prior=Prior(Normal();coordinates=:physical),transform=IdentityTransform())],
            (InferenceExperiment(:run,f.pb,m;solver=f.solver),))
        @test_throws ArgumentError gauss_newton_approximation(q,analytic_map(q,[1.]))
    end
    mode=analytic_map(f.problem,f.mode)
    failure=InferenceExperiment(:run,f.pb,first(f.problem.experiments).measurements...;
        solver=(pb,ts)->throw(NumericalFailure("intentional forward failure")))
    broken=EstimationProblem(f.problem.parameters,(failure,))
    @test_throws NumericalFailure gauss_newton_approximation(broken,mode)
end

@testset "Sensitivities pass through a native PBM integration" begin
    k=.3;times=[.2,.5,1.];sigma=.02
    grid=Grid1D([0.,.5,1.])
    pb=PBMProblem(compile(PBMModel(Death((L,p,t,ctx)->p.k)),grid),[1.,0.],(0.,1.);p=(k=k,))
    m=Measurement(times,exp.(-k.*times),MomentObservable(0);likelihood=GaussianLikelihood(;sigma))
    p=EstimationProblem([ParameterSpec(:k;initial=k,transform=LogTransform(),prior=Prior(Normal(log(k),1);coordinates=:inference))],
        (InferenceExperiment(:run,pb,m;dt=.001),))
    a=gauss_newton_approximation(p,analytic_map(p,[log(k)]))
    exact=-k.*times.*exp.(-k.*times)./sigma
    @test vec(a.jacobian)≈exact rtol=1e-8
    @test a.covariance[1,1]≈1/(sum(abs2,exact)+1) rtol=1e-8
end
