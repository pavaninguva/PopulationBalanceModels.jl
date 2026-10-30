using Test, PopulationBalanceModels, NonlinearSolve, Optim, ForwardDiff
using Distributions: Normal, logpdf
using LinearAlgebra, Statistics
using AdvancedHMC, MCMCDiagnosticTools

function stationary_normal_fixture(;bins=2)
    grid=Grid1D(range(0,1;length=bins+1))
    model=PBMModel(Birth((x,p,t,c)->p.rate)+Death((x,p,t,c)->1.))
    pb=StationaryPBMProblem(compile(model,grid),ones(bins);p=(rate=1.,),autonomous=true)
    measurement=Measurement([1.3],FunctionObservable((u,op,p,t)->log(sum(u)));
        likelihood=GaussianLikelihood(sigma=.2))
    spec=ParameterSpec(:rate;initial=1.,transform=LogTransform(),prior=Prior(Normal();coordinates=:inference))
    experiment=InferenceExperiment(:steady,pb,measurement;solver=StationarySolver(NewtonRaphson()))
    (;pb,measurement,spec,experiment,problem=EstimationProblem((spec,),(experiment,)),mode=1.25,variance=1/26)
end

@testset "Stationary Bayesian target, implicit derivatives and uncertainty" begin
    f=stationary_normal_fixture();p=f.problem
    for value in (-1.,.7,2.)
        z=[value]
        @test predictions(p,z)[1][1]≈reshape(z,1,1) atol=1e-10
        @test loglikelihood(p,z)≈logpdf(Normal(value,.2),1.3) atol=1e-9
        @test posterior_gradient(p,z)≈[-value-(value-1.3)/.2^2] atol=1e-8
        @test ForwardDiff.hessian(v->-logposterior(p,v),z)≈reshape([26.],1,1) atol=1e-7
        @test check_gradient(p,z).valid
    end
    result=fit(p;nstarts=2,seed=18,uncertainty=:laplace)
    @test result.converged
    @test result.minimizer≈[f.mode] atol=1e-7
    @test result.uncertainty.covariance≈reshape([f.variance],1,1) atol=1e-8
    @test result.diagnostics.derivative_backend===:implicit_stationary
    gn=gauss_newton_approximation(p,result)
    @test gn.covariance≈result.uncertainty.covariance atol=1e-8
    report=identifiability_report(p,result.minimizer)
    @test report.prediction_rank==1
    @test report.prediction_jacobian≈ones(1,1) atol=1e-10
    profile=estimation_profile(p,result,1,[1.15,1.25,1.35];target=:posterior)
    @test all(profile.converged)
    @test profile.delta≈([1.15,1.25,1.35] .- f.mode).^2 ./ (2f.variance) atol=1e-7
    propagated=posterior_predictive(p,reshape([1.1,1.2,1.3],1,:);max_solves=3)
    @test propagated.solves==3 && propagated.failed_draws==0
    @test [x.raw[1][1][1] for x in propagated.results]≈[1.1,1.2,1.3]
    @test_throws ArgumentError posterior_predictive(p,ones(1,3);max_solves=2)
    fine=stationary_normal_fixture(bins=5)
    refinement=refinement_study((coarse=p,fine=fine.problem))
    @test refinement.comparisons.fine.max_prediction_change<1e-10
    @test maximum(abs,refinement.comparisons.fine.shift_in_reference_sd)<1e-6
    @test observe(MomentObservable(0),solve_stationary(f.pb))≈[1.]

    legacy=CalibrationProblem((Experiment(f.pb,Observation([1.3],f.measurement.observable;sigma=.2)),);
        parameters=(z,e)->(rate=exp(z[1]),),solver=StationarySolver(NewtonRaphson()))
    calibrated=fit(legacy,[.2])
    @test calibrated.converged
    @test calibrated.minimizer≈[1.3] atol=1e-6
end

@testset "Stationary shared/local parameters and nuisance marginal covariance" begin
    f=stationary_normal_fixture()
    offset=ParameterSpec(:offset;initial=0.,prior=Prior(Normal(0,.3);coordinates=:physical),
        role=:nuisance,experiment=:other,transform=IdentityTransform())
    measurement=Measurement([1.0],f.measurement.observable;likelihood=GaussianLikelihood(sigma=.1),
        response=(y,theta,ex)->y .+theta.offset)
    other=InferenceExperiment(:other,f.pb,measurement;solver=StationarySolver(NewtonRaphson()))
    problem=EstimationProblem((f.spec,offset),(f.experiment,other))
    A=[1. 0.;1. 1.];C=Diagonal([.2^2,.1^2])
    H=Diagonal([1.,1/.3^2])+A'*(C\A)
    reference=H\(A'*(C\[1.3,1.0]))
    result=fit(problem;uncertainty=:laplace)
    @test result.converged
    @test result.minimizer≈reference atol=1e-6
    @test result.uncertainty.covariance≈H\Matrix{Float64}(I,2,2) rtol=1e-6
    @test marginal_covariance(problem,result.uncertainty).covariance≈result.uncertainty.covariance[1:1,1:1]
    @test abs(result.uncertainty.covariance[1,2])>1e-3
end

@testset "Nonlinear stationary curvature and unit rescaling" begin
    # M0 solves J-d*M0-K*M0^2/2=0. Its second J derivative is nonzero.
    d=.4;K=.7
    op=compile(PBMModel(Nucleation((p,t,c)->p.J,1.)+
        Aggregation(ConstantKernel((p,t,c)->K))+Death(PolynomialRate((d,)))),MomentSet(0:0))
    pb=StationaryPBMProblem(op,[1.];p=(J=.8,),autonomous=true)
    m=Measurement([1.1],MomentObservable(0);likelihood=GaussianLikelihood(sigma=.2))
    spec=ParameterSpec(:J;initial=.8,prior=Prior(Normal(1.,.4);coordinates=:physical),transform=IdentityTransform())
    p=EstimationProblem((spec,),(InferenceExperiment(:nonlinear,pb,m),))
    value=.8;M=2value/(d+sqrt(d^2+2K*value));a=1/(d+K*M);b=-K/(d+K*M)^3
    @test predictions(p,[value])[1][1][1]≈M atol=1e-9
    @test posterior_gradient(p,[value])[1]≈-(M-1.1)*a/.2^2-(value-1)/.4^2 atol=1e-7
    exact=(a*a+(M-1.1)*b)/.2^2+1/.4^2
    @test ForwardDiff.hessian(z->-logposterior(p,z),[value])[1]≈exact rtol=1e-7
    @test abs(exact-(a*a/.2^2+1/.4^2))>1e-2 # distinguishes full Hessian from GN
    # Changing the coordinate unit changes integrated counts by its domain
    # length, cancelled by the declared instrument response, not by a repair.
    for extent in (1e-6,1e6)
        grid=Grid1D([0.,.3extent,extent])
        operator=compile(PBMModel(Birth((x,p,t,c)->p.k)+Death((x,p,t,c)->1.)),grid)
        problem=StationaryPBMProblem(operator,grid.widths;p=(k=1.,),autonomous=true)
        observation=Measurement([2.],FunctionObservable((u,op,p,t)->sum(u)/extent);
            likelihood=GaussianLikelihood(sigma=.1))
        parameter=ParameterSpec(:k;initial=2.,prior=Prior(Normal(2.,1.);coordinates=:physical),transform=IdentityTransform())
        estimation=EstimationProblem((parameter,),(InferenceExperiment(:units,problem,observation),))
        @test ForwardDiff.jacobian(z->vec(predictions(estimation,z)[1][1]),[2.])≈ones(1,1) rtol=1e-8
    end
end

@testset "Stationary tensor and axial likelihood sensitivities" begin
    spec=ParameterSpec(:d;initial=.4,transform=LogTransform(),prior=Prior(Normal();coordinates=:inference))
    grid=Grid2D(Grid1D([0.,.3,1.]),Grid1D([0.,.5,1.2,2.]))
    op=compile(PBMModel(Birth((xy,p,t,c)->2.)+Death((xy,p,t,c)->p.d)),grid)
    pb=StationaryPBMProblem(op,ones(size(grid));p=(d=.4,),autonomous=true)
    m=Measurement([10.],MomentObservable((0,0));likelihood=GaussianLikelihood(sigma=.1))
    tensor=EstimationProblem((spec,),(InferenceExperiment(:tensor,pb,m),))
    @test predictions(tensor,[log(.4)])[1][1][1]≈10. atol=1e-8
    @test ForwardDiff.jacobian(z->vec(predictions(tensor,z)[1][1]),[log(.4)])[1]≈-10. atol=1e-7

    g=Grid1D([0.,.5,1.]);z=AxialGrid([0.,.05,.2,.5,1.]);v=.7
    feed=[1.,2.]
    localop=compile(PBMModel(Death((x,p,t,c)->p.d)),g)
    domain=AxialDispersion(z;velocity=v,dispersion=0.,inlet=(p,t)->feed)
    axial=StationaryPBMProblem(compile(PBMModel(Death((x,p,t,c)->p.d)),g;domain),
        repeat(feed,1,length(z));p=(d=.4,),autonomous=true)
    exact=sum(feed)*prod(v./(v.+.4z.widths))
    derivative=-exact*sum(.4z.widths./(v.+.4z.widths))
    measurement=Measurement([exact],CellObservable(MomentObservable(0),length(z));likelihood=GaussianLikelihood(sigma=.1))
    estimation=EstimationProblem((spec,),(InferenceExperiment(:outlet,axial,measurement),))
    @test predictions(estimation,[log(.4)])[1][1][1]≈exact atol=1e-8
    @test ForwardDiff.jacobian(q->vec(predictions(estimation,q)[1][1]),[log(.4)])[1]≈derivative atol=1e-7
    sys=PBMSystem((cells=localop,);bulk_rhs=(ctx,p,t)->(c=-p.d*ctx.bulk.c,))
    localpb=StationaryPBMProblem(sys,(cells=feed,);bulk=(c=4.,),p=(d=.4,),autonomous=true)
    coupled=axial_problem(localpb,AxialDispersion(z;velocity=v,dispersion=0.,
        inlet=(p,t)->(populations=(cells=feed,),bulk=(c=4.,))))
    bulk_measurement=Measurement([4prod(v./(v.+.4z.widths))],CellObservable(BulkObservable(:c),length(z));
        likelihood=GaussianLikelihood(sigma=.1))
    coupled_fit=EstimationProblem((spec,),(InferenceExperiment(:reactor,coupled,bulk_measurement),))
    @test check_gradient(coupled_fit,[log(.35)]).valid
    @test fit(coupled_fit).parameters.shared.d≈.4 rtol=.01
end

@testset "Stationary quadrature families retain explicit finite-difference fitting" begin
    J=1.;K=.4;d=.3
    number=2J/(d+sqrt(d*d+2K*J));M1=J/d;M2=(J+K*M1^2)/d
    for (method,raw) in ((QMOM(1),[number,M1]),(RankAdaptiveQMOM(1),[number,M1]),
            (EQMOM(1),[number,M1,M2]),(LogNormalClosure(),[number,M1,M2]))
        model=PBMModel(Nucleation((p,t,c)->p.J,1.)+Aggregation(ConstantKernel((p,t,c)->K))+
            Death((x,p,t,c)->d))
        op=compile(model,MomentSet(0:length(raw)-1);method)
        pb=StationaryPBMProblem(op,.9raw;p=(J=J,),autonomous=true)
        measurement=Measurement([M1],MomentObservable(1);likelihood=GaussianLikelihood(sigma=.02))
        spec=ParameterSpec(:J;initial=.9,transform=LogTransform(),prior=Prior(Normal();coordinates=:inference))
        problem=EstimationProblem((spec,),(InferenceExperiment(:moments,pb,measurement),);derivatives=:finite_difference)
        result=fit(problem;maxiters=80)
        @test result.converged
        @test result.parameters.shared.J≈J rtol=.002
        @test check_gradient(problem,result.minimizer).valid
    end
    method=DQMOM(1)
    op=compile(PBMModel(Nucleation((p,t,c)->p.J,2.)+Death((x,p,t,c)->d)),MomentSet(0:1);method)
    pb=StationaryPBMProblem(op,quadrature_state(method,[2.],[J/d]);p=(J=J,),autonomous=true)
    m=Measurement([2J/d],MomentObservable(1);likelihood=GaussianLikelihood(sigma=.02))
    spec=ParameterSpec(:J;initial=.9,transform=LogTransform(),prior=Prior(Normal();coordinates=:inference))
    p=EstimationProblem((spec,),(InferenceExperiment(:dqmom,pb,m),);derivatives=:finite_difference)
    @test predictions(p,[0.])[1][1][1]≈2J/d atol=1e-8
    @test fit(p).parameters.shared.J≈J rtol=.002
end

@testset "Stationary diffusion constraints and coupled material sensitivities" begin
    grid=Grid1D([0.,.1,.3,.6,1.])
    model=PBMModel(SizeDiffusion((x,p,t,ctx)->1+p.a*x))
    pb=StationaryPBMProblem(compile(model,grid),grid.widths;p=(a=2.,),autonomous=true,
        constraints=(1=>(u,p,t)->sum(u)-1,))
    weights=grid.widths./(1 .+2grid.centers)
    derivative=-grid.widths.*grid.centers./(1 .+2grid.centers).^2
    mean=dot(grid.centers,weights)/sum(weights)
    slope=2*(dot(grid.centers,derivative)*sum(weights)-dot(grid.centers,weights)*sum(derivative))/sum(weights)^2
    observation=Measurement([mean],MomentObservable(1);likelihood=GaussianLikelihood(sigma=.001))
    spec=ParameterSpec(:a;initial=1.8,transform=LogTransform(),prior=Prior(Normal();coordinates=:inference))
    p=EstimationProblem((spec,),(InferenceExperiment(:diffusion,pb,observation),))
    @test predictions(p,[log(2.)])[1][1][1]≈mean atol=1e-9
    @test ForwardDiff.jacobian(z->vec(predictions(p,z)[1][1]),[log(2.)])[1]≈slope atol=1e-8
    @test check_gradient(p,[log(1.8)]).valid

    grid=Grid1D([0.,.5,1.5]) # nucleus at the second pivot L=1
    parameters=(k=.6,d=.4,F=.7,feed=3.,alpha=.2)
    pop=compile(PBMModel(Nucleation((p,t,c)->p.k*c.bulk.c,1.)+Death((x,p,t,c)->p.d)),grid)
    system=PBMSystem((crystals=pop,);bulk_rhs=(ctx,p,t)->(c=p.F*(p.feed-ctx.bulk.c),),
        exchanges=(ConservativeExchange(:crystals,:c,(x,p,t,c)->p.alpha*x^3),),nonnegative_bulk=(:c,))
    coupled=StationaryPBMProblem(system,(crystals=[.1,1.],);bulk=(c=2.,),p=parameters,autonomous=true)
    k=parameters.k;F=parameters.F;feed=parameters.feed;alpha=parameters.alpha;d=parameters.d
    c=F*feed/(F+alpha*k);dc=-F*feed*alpha/(F+alpha*k)^2
    number=k*c/d;dn=(c+k*dc)/d
    measurements=(Measurement([number],PopulationMoment(:crystals,0);likelihood=GaussianLikelihood(sigma=.01)),
        Measurement([c],BulkObservable(:c);likelihood=GaussianLikelihood(sigma=.01)))
    spec=ParameterSpec(:k;initial=k,transform=LogTransform(),prior=Prior(Normal();coordinates=:inference))
    estimation=EstimationProblem((spec,),(InferenceExperiment(:coupled,coupled,measurements...),))
    @test [a[1] for a in predictions(estimation,[log(k)])[1]]≈[number,c] atol=1e-8
    J=ForwardDiff.jacobian(z->vcat((vec(a) for a in predictions(estimation,z)[1])...),[log(k)])
    @test vec(J)≈k.*[dn,dc] atol=1e-7
end

@testset "Failed stationary solves, singular roots and mixed experiments" begin
    f=stationary_normal_fixture()
    failure=pb->solve_stationary(pb,NewtonRaphson();maxiters=1,abstol=1e-30)
    nonlinear=compile(PBMModel(Nucleation((p,t,c)->p.rate,1.)+
        Aggregation(ConstantKernel((p,t,c)->1.2))+Death(PolynomialRate((.5,)))),MomentSet(0:0))
    failure_problem=StationaryPBMProblem(nonlinear,[.1];p=(rate=1.,),autonomous=true)
    ex=InferenceExperiment(:failed,failure_problem,f.measurement;solver=failure)
    p=EstimationProblem((f.spec,),(ex,))
    @test loglikelihood(p,[.3];on_failure=:reject)==-Inf
    @test_throws NumericalFailure predictions(p,[.3])
    @test f.pb.guess==ones(2,1)
    invalid=InferenceExperiment(:invalid,f.pb,f.measurement;solver=pb->(;u=pb.guess,t=[0.]))
    @test_throws ArgumentError predictions(EstimationProblem((f.spec,),(invalid,)),[0.])
    empty=StationaryPBMProblem(compile(PBMModel(),Grid1D([0.,.5,1.])),[.5,.5];p=(rate=1.,),autonomous=true)
    singular=EstimationProblem((f.spec,),(InferenceExperiment(:singular,empty,f.measurement;solver=StationarySolver(NewtonRaphson())),))
    @test_throws DifferentiationFailure posterior_gradient(singular,[0.])
    op=compile(PBMModel(Death((x,p,t,c)->p.rate)),Grid1D([0.,.5,1.]))
    transient=PBMProblem(op,[.5,.5],(0.,.2);p=(rate=1.,))
    m=Measurement([.1,.2],exp.(-[.1,.2]),MomentObservable(0);likelihood=GaussianLikelihood(sigma=.1))
    mixed=EstimationProblem((f.spec,),(f.experiment,InferenceExperiment(:transient,transient,m;dt=.005)))
    @test check_gradient(mixed,[.1]).valid
    @test posterior_gradient(mixed,[.1])≈posterior_gradient(mixed,[.1];derivatives=:finite_difference) rtol=1e-5
end

@testset "Transient direct, tensor and axial models retain the same estimation API" begin
    grid=Grid1D(range(0,1;length=11))
    parameter=ParameterSpec(:amplitude;initial=1.7,transform=LogTransform(),
        prior=Prior(Normal(log(2.),1.);coordinates=:inference))
    setup=(theta,base)->PopulationBalanceModels.remake(base;u0=theta.amplitude.*base.u0)
    histogram=DistributionObservable([0.,.3,1.];coordinate=:volume)
    measurement=Measurement([1.],reshape([.6,1.4],:,1),histogram;likelihood=GaussianLikelihood(sigma=.01))
    for method in (HOCFiniteVolume(),Inguva2022())
        operator=compile(PBMModel(Growth(AffineGrowth(.2,0.))),grid;
            evolution=method,bc=SizeBC(Periodic(),Periodic()))
        pb=PBMProblem(operator,grid.widths,(0.,1.))
        step=method isa Inguva2022 ? .5 : .1
        solver=(problem,times)->simulate(problem;method,dt=step,saveat=times)
        problem=EstimationProblem((parameter,),(InferenceExperiment(:direct,pb,measurement;setup,solver),);
            derivatives=:finite_difference)
        @test check_gradient(problem,[log(1.7)]).valid
        result=fit(problem;maxiters=80)
        @test result.converged
        @test result.parameters.shared.amplitude≈2. rtol=1e-5
    end
    tensor=Grid2D(Grid1D([0.,.3,1.]),Grid1D([0.,.5,2.]))
    operator=compile(PBMModel(Birth((xy,p,t,c)->p.amplitude)),tensor)
    pb=PBMProblem(operator,zeros(size(tensor)),(0.,1.);p=(amplitude=1.7,))
    m=Measurement([1.],[4.],MomentObservable((0,0));likelihood=GaussianLikelihood(sigma=.01))
    problem=EstimationProblem((parameter,),(InferenceExperiment(:tensor,pb,m;dt=.1),))
    @test check_gradient(problem,[log(1.7)]).valid
    @test fit(problem).parameters.shared.amplitude≈2. rtol=1e-5

    z=AxialGrid([0.,.2,.5,1.]);domain=AxialDispersion(z;velocity=.2,dispersion=.01,boundary=:periodic)
    grid=Grid1D([0.,.5,1.])
    pb=PBMProblem(compile(PBMModel(Death((x,p,t,c)->p.d)),grid;domain),
        repeat(grid.widths,1,length(z)),(0.,1.);p=(d=.3,))
    spec=ParameterSpec(:d;initial=.25,transform=LogTransform(),prior=Prior(Normal(log(.3),1.);coordinates=:inference))
    m=Measurement([.5,1.],exp.(-0.3 .* [.5,1.]),CellObservable(MomentObservable(0),length(z));
        likelihood=GaussianLikelihood(sigma=.01))
    problem=EstimationProblem((spec,),(InferenceExperiment(:axial,pb,m;dt=.005),))
    @test check_gradient(problem,[log(.25)]).valid
    @test fit(problem).parameters.shared.d≈.3 rtol=1e-5
end

@testset "Stationary NUTS shares the analytically known posterior" begin
    f=stationary_normal_fixture();mode=fit(f.problem;uncertainty=:laplace)
    posterior=sample_posterior(f.problem;initial=mode,laplace=mode.uncertainty,
        warmup=200,samples=700,chains=2,seed=2041,min_ess=400)
    values=vec(posterior.z)
    @test abs(mean(values)-f.mode)<.2sqrt(f.variance)
    @test abs(var(values)/f.variance-1)<.2
    @test maximum(diagnostics(posterior).rhat)<=1.01
    @test minimum(diagnostics(posterior).ess_bulk)>=400
    @test minimum(diagnostics(posterior).ess_tail)>=400
    @test diagnostics(posterior).divergences==0
    @test diagnostics(posterior).diagnostics_passed
    println("Stationary NUTS: ",(mean=mean(values),variance=var(values),
        rhat=diagnostics(posterior).rhat,ess_bulk=diagnostics(posterior).ess_bulk,
        ess_tail=diagnostics(posterior).ess_tail,divergences=diagnostics(posterior).divergences))
    for chain in 1:2,draw in (1,300,700)
        @test diagnostics(posterior).statistics[chain][draw].log_density≈
            logposterior(f.problem,posterior.z[:,draw,chain]) atol=1e-9
    end
end
