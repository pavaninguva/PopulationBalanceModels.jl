using Test, LinearAlgebra, ForwardDiff
using Distributions: Normal

@testset "Audit: unit-independent moment reconstruction" begin
    basis=MomentSet(0:2;coordinate=:length)
    for scale in (1.,1e-6,1e-12,1e3)
        mu=[1.,scale,exp(10.)*scale^2]
        @test_throws NumericalFailure quadrature(basis,mu,LogNormalClosure(quadrature_order=16))
        q=quadrature(basis,mu,LogNormalClosure(quadrature_order=64))
        @test [sum(q.weights.*q.nodes.^k)/mu[k+1] for k in 0:2]≈ones(3) rtol=1e-8
        op=compile(PBMModel(Growth(AffineGrowth(0.,.2))),basis;method=LogNormalClosure(quadrature_order=64))
        pb=PBMProblem(op,mu,(0.,.1));du=zero(pb.u0)
        rhs!(du,pb.u0,op,pb.p,0.)
        @test vec(du)./[1.,scale,scale^2]≈[0.,.2,.4exp(10.)] rtol=1e-8
    end
    # The stable helper must have the correct AD derivative even at its branch.
    @test ForwardDiff.derivative(PopulationBalanceModels._softplus,0.)≈.5
    @test ForwardDiff.hessian(z->PopulationBalanceModels._logjac(LogitTransform(0,1),z[1]),[0.])[1]≈-.5
    for n in (32,64,128)
        q=LogNormalClosure(quadrature_order=n)
        @test all(>(0),q.weights)
        @test [sum(q.weights.*q.nodes.^k) for k in (0,2,4,6)]≈[1.,1.,3.,15.] rtol=1e-12
    end
    # Check the transformed non-polynomial integrand as well as Gaussian
    # polynomial exactness. Both 64 and 128 points must resolve this tail.
    for n in (64,128)
        mu=[1.,1.,exp(14.)]
        q=quadrature(basis,mu,LogNormalClosure(quadrature_order=n))
        @test sum(q.weights.*q.nodes.^2)≈exp(14.) rtol=1e-8
    end
end

@testset "Audit: effective deposition policy and loss accounting" begin
    grid=Grid1D([0.,1.,2.])
    for (term,initial,loss,number,material) in (
            (Aggregation(ConstantKernel(1.)),[0.,1.],:aggregation_overflow,.5,1.5),
            (Breakage((x,p,t,c)->1.,SelfSimilarDaughters((.5,),(2.,))),[1.,0.],:breakage_underflow,2.,.5))
        op=compile(PBMModel(term),grid;methods=TermMethods(;NamedTuple{(mechanism(term),)}((FixedPivot(overflow=:discard),))...))
        system=PBMSystem((particles=op,);
            accounting=(CumulativeAccounting(:particles;properties=(material=1,)),),
            exchanges=(ConservativeExchange(:particles,:c,(x,p,t,ctx)->x;mechanisms=(mechanism(term),)),))
        pb=PBMProblem(system,(particles=initial,),(0.,.01);bulk0=(c=2.,))
        before=copy(pb.u0);du=zero(pb.u0);rhs!(du,pb.u0,pb.operator,pb.p,0.)
        @test cumulative(pb,du,:particles,loss)≈number
        @test cumulative(pb,du,:particles,loss,:material)≈material
        @test bulk(pb,du,:c)≈0 atol=1e-14
        total(u)=bulk(pb,u,:c)+sum(grid.centers.*population(pb,u,:particles))+cumulative(pb,u,:particles,loss,:material)
        for method in (SSPRK33(),MeasureEuler())
            sol=simulate(pb;dt=.005,method)
            @test total(last(sol.u))≈total(before) rtol=1e-13
        end
        @test pb.u0==before
        default=compile(PBMModel(term),grid)
        raw=PBMProblem(default,initial,(0.,.01))
        @test_throws NumericalFailure rhs!(zero(raw.u0),raw.u0,default,raw.p,0.)
    end
    # An explicit error policy wins over a term's discard setting, too.
    op=compile(PBMModel(Aggregation(ConstantKernel(1.);overflow=:discard)),grid;
        methods=TermMethods(aggregation=FixedPivot(overflow=:error)))
    pb=PBMProblem(op,[0.,1.],(0.,.01))
    @test_throws NumericalFailure rhs!(zero(pb.u0),pb.u0,op,pb.p,0.)
    @test_throws NumericalFailure PopulationBalanceModels._deposit!(zeros(2),grid.centers,NaN,1.,:discard)
end

@testset "Audit: unresolved nucleation consumes material once" begin
    grid=Grid1D([0.,1.,2.];coordinate=:length)
    term=Nucleation((p,t,c)->2.,3.)
    @test_throws ArgumentError compile(PBMModel(term),grid)
    op=compile(PBMModel(term),grid;methods=TermMethods(nucleation=FixedPivot(overflow=:discard)))
    system=PBMSystem((particles=op,);accounting=(CumulativeAccounting(:particles),),
        exchanges=(ConservativeExchange(:particles,:c,(x,p,t,c)->x^3),),nonnegative_bulk=(:c,))
    pb=PBMProblem(system,(particles=zeros(2),),(0.,.01);bulk0=(c=100.,))
    for method in (SSPRK33(),MeasureEuler())
        sol=simulate(pb;dt=.005,method);u=last(sol.u)
        @test all(iszero,population(pb,u,:particles))
        @test cumulative(pb,u,:particles,:nucleation_overflow)≈.02
        @test cumulative(pb,u,:particles,:nucleation_overflow,:material)≈.54
        @test bulk(pb,u,:c)+cumulative(pb,u,:particles,:nucleation_overflow,:material)≈100.
    end
end

@testset "Audit: normalized measurements and unchanged statistical models" begin
    obs=DistributionObservable([0.,1.,2.];coordinate=:volume,normalize=true)
    likelihood=GaussianLikelihood(sigma=.1,contrasts=[1. -1.])
    @test_throws ArgumentError Measurement([0.],reshape([.4,.7],2,1),obs;likelihood)
    @test_throws ArgumentError Measurement([0.],reshape([-.1,1.1],2,1),obs;likelihood)
    counts=Measurement([0.],reshape([4,6],2,1),obs;likelihood=MultinomialLikelihood())
    @test sum(counts.data)==10
    pb=PBMProblem(compile(PBMModel(),Grid1D([0.,1.,2.])),ones(2),(0.,1.))
    response=(P,theta,e)->2P
    m=Measurement([0.],reshape([.5,.5],2,1),obs;likelihood,response)
    spec=ParameterSpec(:unused;initial=0.,prior=Prior(Normal();coordinates=:physical))
    problem=EstimationProblem((spec,),(InferenceExperiment(:run,pb,m;dt=.1),))
    @test logposterior(problem,[0.];on_failure=:reject)==-Inf
    @test pb.u0==ones(2,1)
    obs2=MomentObservable(0)
    make(s)=EstimationProblem((spec,),(InferenceExperiment(:run,pb,
        Measurement([0.],[2.],obs2;likelihood=GaussianLikelihood(sigma=s));dt=.1),))
    @test_throws ArgumentError PopulationBalanceModels._check_comparable_problems((base=make(.1),changed_noise=make(1.)))
    @test PopulationBalanceModels._check_comparable_problems((base=make(.1),same=make(.1)))===nothing
end

@testset "Audit: support and zero-feed axial ledgers" begin
    g=MomentSet(0:3;coordinate=:length)
    op=compile(PBMModel(),g;method=RankAdaptiveQMOM(2),boundary=MomentBoundary(upper=1.,right=:escape))
    @test_throws ArgumentError PBMProblem(op,[1.,2.,4.,8.],(0.,1.))
    localop=compile(PBMModel(),g;method=RankAdaptiveQMOM(2))
    localpb=PBMProblem(PBMSystem((particles=localop,);accounting=(CumulativeAccounting(:particles;properties=(material=3,)),)),
        (particles=[1.,.5,.25,.125],),(0.,.1))
    domain=AxialDispersion(AxialGrid([0.,.2,1.]);velocity=.4,dispersion=.01)
    pb=axial_problem(localpb,domain)
    F=axial_fluxes(pb.operator,pb.u0,pb.p,0.)
    @test all(iszero,F[:,1])
    sol=simulate(pb;dt=.02,method=MeasureEuler());u=last(sol.u)
    flux=diagnostics(sol).reactor_flux_integrals
    for k in g.orders
        initial=sum(domain.grid.widths.*population(pb,pb.u0,:particles)[k+1,:])
        final=sum(domain.grid.widths.*population(pb,u,:particles)[k+1,:])
        @test final+flux.outlet[k+1]-flux.inlet[k+1]≈initial rtol=1e-12
    end
    @test all(iszero,cumulative(pb,u,:particles,:upper_escape,:material))
end

@testset "Audit: rejected coupled steps are transactional" begin
    g=MomentSet(0:3;coordinate=:length)
    op=compile(PBMModel(Growth((x,p,t,c)->c.bulk.c)),g;method=RankAdaptiveQMOM(2))
    sys=PBMSystem((particles=op,);nonnegative_bulk=(:c,),accounting=(CumulativeAccounting(:particles),),
        exchanges=(MomentExchange(:particles,:c,3),))
    pb=PBMProblem(sys,(particles=[1.,1.,1.,1.],),(0.,.5);bulk0=(c=.5,))
    before=copy(pb.u0);sol=simulate(pb;dt=.5,method=MeasureEuler())
    @test diagnostics(sol).rejected_steps>0
    @test pb.u0==before
    @test all(u->isapprox(bulk(pb,u,:c)+population(pb,u,:particles)[4],1.5;rtol=1e-12),sol.u)
    @test all(iszero,cumulative(pb,last(sol.u),:particles,:extinction,:material))
end

@testset "Audit: accepted RK states remain closure-admissible" begin
    source=MomentSource((p,t,c)->1e-6,(k,p,t,c)->1e9^k)
    basis=MomentSet(0:2;coordinate=:length)
    op=compile(PBMModel(source),basis;method=LogNormalClosure(quadrature_order=16))
    pb=PBMProblem(op,ones(3),(0.,1.));before=copy(pb.u0)
    # An Euler combination of two positive atomic measures is realizable, but
    # its broad lognormal representation is unresolved by the selected rule.
    candidate=pb.u0 .+ reshape([1e-6,1e3,1e12],:,1)
    @test realizability(basis,candidate).valid
    @test_throws NumericalFailure simulate(pb;dt=1.,method=ForwardEuler())
    @test pb.u0==before
    resolved=compile(PBMModel(source),basis;method=LogNormalClosure(quadrature_order=128))
    sol=simulate(PBMProblem(resolved,ones(3),(0.,1.));dt=1.,method=ForwardEuler())
    @test last(sol.u)≈candidate
    @test diagnostics(sol).policy===:ForwardEuler
end

@testset "Audit: runtime exact-moment failure propagation" begin
    calls=Ref(0)
    rate=(p,t,ctx)->(calls[]+=1;1.)
    op=compile(PBMModel(Growth(PolynomialRate((rate,)))),MomentSet(0:3))
    pb=PBMProblem(op,ones(4),(0.,1.));before=copy(pb.u0)
    invalid=reshape([1.,0.,1.,0.],:,1)
    @test_throws NumericalFailure rhs!(zero(invalid),invalid,op,pb.p,0.)
    @test calls[]==0
    measurement=Measurement([.5],[1.],MomentObservable(0);likelihood=GaussianLikelihood(sigma=.1))
    solver=(pb,ts)->(rhs!(zero(invalid),invalid,pb.operator,pb.p,first(ts));nothing)
    spec=ParameterSpec(:unused;initial=0.,prior=Prior(Normal();coordinates=:inference))
    estimation=EstimationProblem((spec,),(InferenceExperiment(:run,pb,measurement;solver),))
    @test logposterior(estimation,[0.];on_failure=:reject)==-Inf
    @test pb.u0==before
end

@testset "Audit: sampling coordinates away from initialization" begin
    pb=PBMProblem(compile(PBMModel(),Grid1D([0.,1.,2.])),ones(2),(0.,1.))
    measurement=Measurement([0.],reshape([1.2,-.4],2,1),
        FunctionObservable((u,op,p,t)->[p.a,p.b]);likelihood=GaussianLikelihood(sigma=1.))
    specs=(ParameterSpec(:a;initial=0.,prior=Prior(Normal();coordinates=:physical)),
           ParameterSpec(:b;initial=0.,prior=Prior(Normal();coordinates=:inference)))
    problem=EstimationProblem(specs,(InferenceExperiment(:run,pb,measurement;dt=.1),))
    map=(to_sampler=z->[asinh(z[1]),z[2]],from_sampler=y->[sinh(y[1]),y[2]],
        logabsdetjac=y->log(cosh(y[1])),label=:sinh_audit)
    for z in ([0.,0.],[.6,-.2],[-5.,3.],[7.,-4.])
        report=check_sampling_map(problem,map,z)
        @test report.valid
        @test report.logdensity_sampler≈logposterior(problem,z)+log(cosh(report.y[1]))
        @test report.logdensity_inference≈logposterior(problem,z)
    end
    wrong=merge(map,(logabsdetjac=y->log(cosh(y[1]))+.1y[1]^4,))
    @test check_sampling_map(problem,wrong,[0.,0.]).valid
    @test_throws ArgumentError check_sampling_map(problem,wrong,[3.,-2.])
    reverse=(to_sampler=z->[-z[1],z[2]],from_sampler=y->[-y[1],y[2]],
        logabsdetjac=y->zero(first(y)),label=:negative_determinant)
    @test check_sampling_map(problem,reverse,[-4.,3.]).valid
    # An exponential map covers only half of this Gaussian posterior support.
    restricted=(to_sampler=z->[log(z[1]),z[2]],from_sampler=y->[exp(y[1]),y[2]],
        logabsdetjac=y->y[1],label=:wrong_support)
    @test_throws DomainError check_sampling_map(problem,restricted,[-2.,1.])
end
