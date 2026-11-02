module ForwardExecutionTests
using Test, PopulationBalanceModels, ForwardDiff, LinearAlgebra
using Distributions: Uniform, Gamma, Normal, LogNormal, cdf, product_distribution, truncated

# An outside-module numerical extension, using only documented public hooks.
struct ParameterHazard
    parameter::Symbol
end
(r::ParameterHazard)(x,p,t,ctx)=getproperty(p,r.parameter)*(1+t)
struct PreparedRemoval <: AbstractTermMethod end
PopulationBalanceModels.prepare_term(term::Death{ParameterHazard},::PreparedRemoval,g::Grid1D;kwargs...)=(widths=copy(g.widths),)
PopulationBalanceModels.prepare_run(term::Death{ParameterHazard},::PreparedRemoval,cache,g::Grid1D,p)=
    (widths=cache.widths,coefficient=getproperty(p,term.rate.parameter),)
function PopulationBalanceModels.add_term!(du,u,term::Death{ParameterHazard},::PreparedRemoval,cache,op,p,t,ctx)
    du .-=cache.coefficient*(1+t).*u
    (zero(eltype(du)),zero(eltype(du)))
end

@testset "Public parameter-preparation extension and isolated working state" begin
    g=Grid1D([0.,.4,1.])
    model=PBMModel(removal=Death(ParameterHazard(:k)))
    op=compile(model,g;methods=TermMethods(by_term=(removal=PreparedRemoval(),)))
    base=PBMProblem(op,g.widths,(0.,.2);p=(k=.4,))
    solver=ForwardSolver(RK4();dt=.002)
    for k in (.4,.8,.4)
        sol=solver(remake(base;p=(k=k,)),[.2])
        @test sum(last(sol.u))≈exp(-k*(.2+.2^2/2)) atol=4e-14
    end
    @test !hasproperty(op.prepared[1].cache,:coefficient)
    f=k->sum(last(solver(remake(base;p=(k=k,)),[.2]).u))
    @test ForwardDiff.derivative(f,.4)≈-.22exp(-.4*.22) rtol=1e-11
    # Every solve starts from the base state and does not alter its data.
    @test base.u0==reshape(g.widths,:,1)
end

@testset "Configured fixed execution and prescribed interval stages" begin
    g=Grid1D([0.,.2,.6,1.])
    model=PBMModel(removal=Death((x,p,t,c)->p.k))
    op=compile(model,g)
    pb=PBMProblem(op,g.widths,(0.,1.);p=(k=.4,))
    ts=[0.,.13,.6,1.]
    for algorithm in (ForwardEuler(),SSPRK22(),SSPRK33(),SSPRK54(),RK4())
        legacy=simulate(pb;dt=.01,method=algorithm,saveat=ts)
        configured=ForwardSolver(algorithm;dt=.01)(pb,ts)
        @test configured.t==ts
        @test configured.u==legacy.u
        @test configured.problem===pb
    end
    @test_throws ArgumentError simulate(pb;solver=ForwardSolver(;dt=.1),dt=.2)
    @test_throws ArgumentError simulate(pb;solver=ForwardSolver(;dt=.1,check_nonnegative=false),check_nonnegative=true)
    @test_throws ArgumentError ForwardSolver(MeasureEuler();dt=.1,control=StepControl())
    @test_throws ArgumentError ForwardSolver(;dt=.1,callback=identity)
    @test_throws ArgumentError ForwardSolver(:algorithm;backend=:external,constraints=:nonnegative)
    stationary=StationaryPBMProblem(op,g.widths;autonomous=true,p=(k=.4,))
    @test_throws ArgumentError UQProblem(stationary,UncertainInputs((k=Uniform(.2,.6),));
        outputs=(number=MomentObservable(0),),solver=ForwardSolver(;dt=.1))
    @test_throws ArgumentError InferenceExperiment(:stationary,stationary,
        Measurement([1.],MomentObservable(0);likelihood=GaussianLikelihood(sigma=1.));solver=ForwardSolver(;dt=.1))
    fp32=PBMProblem(op,Float32.(g.widths),(0.,.2);p=(k=.4,))
    @test eltype(last(ForwardSolver(;dt=.01)(fp32,[.2]).u))===Float32
    history=OperatingHistory([.4];parameters=((k=.3,),(k=1.1,)))
    changed=remake(pb;history)
    # RK endpoint stages must see the OLD rate. A t<change callback would give
    # an O(dt) defect here even though the solution itself is continuous.
    sol=ForwardSolver(RK4();dt=.002)(changed,[0.,.4,.7,1.])
    @test sum(sol.u[2])≈exp(-.3*.4) atol=2e-13
    @test sum(sol.u[end])≈exp(-.3*.4-1.1*.6) atol=2e-13
    @test operating_inputs(changed,.4).parameters.k==1.1
    @test operating_inputs(changed,.4;side=:left).parameters.k==.3
    observable=FunctionObservable((u,op,p,t)->p.k*sum(u))
    @test observe(observable,sol;index=2)≈[1.1exp(-.12)] atol=2e-13
    @test length(diagnostics(sol).segments)==2
    @test sol.problem===changed
    @test pb.p==(k=.4,) && pb.history===nothing
    @test_throws ArgumentError OperatingHistory([.5,.3])
    @test_throws ArgumentError remake(pb;history=OperatingHistory([1.]))
    @test_throws DimensionMismatch OperatingHistory([.5];parameters=((k=1.,),))
    fields=OperatingHistory([.4];fields=((rate=.3,),(rate=1.1,)))
    fop=compile(PBMModel(removal=Death((x,p,t,c)->c.fields.rate)),g)
    fsol=simulate(PBMProblem(fop,g.widths,(0.,1.);history=fields);method=RK4(),dt=.002)
    @test sum(last(fsol.u))≈sum(last(sol.u)) atol=2e-14
    @test_throws NumericalFailure simulate(changed;solver=ForwardSolver(;dt=.001,maxsteps=5))
    # A fixed, prescribed schedule remains differentiable.
    f=k->sum(last(simulate(remake(changed;p=(k=k,),history=OperatingHistory([.4];
        parameters=(p->(k=p.k,),p->(k=2p.k,))));dt=.005,method=RK4()).u))
    @test ForwardDiff.derivative(f,.3)≈-1.6exp(-.48) rtol=2e-10
    tensor=Grid2D(0.:.25:1.,0.:.25:1.)
    periodic=SizeBC(Periodic(),Periodic())
    transport=compile(PBMModel(Growth2D((1.,0.))),tensor;bc=SizeBC2D(x=periodic,y=periodic))
    shift=ForwardSolver(ExactShift();dt=.25)
    initial=reshape(collect(1.:16.),4,4)
    aligned=PBMProblem(transport,initial,(0.,1.);history=OperatingHistory([.5]))
    @test last(shift(aligned,[0.,.5,1.]).u)==initial
    @test_throws ArgumentError shift(remake(aligned;history=OperatingHistory([.3])),[1.])
end

@testset "Controlled stages, rollback and budgets" begin
    g=Grid1D([0.,.3,1.])
    op=compile(PBMModel(source=Birth((x,p,t,c)->10c.bulk.available)),g)
    system=PBMSystem((particles=op,);bulk_rhs=(ctx,p,t)->(available=-10ctx.bulk.available,),
        nonnegative_bulk=(:available,))
    pb=PBMProblem(system,(particles=zeros(2),),(0.,.3);bulk0=(available=1.,))
    solver=ForwardSolver(ForwardEuler();dt=.2,control=StepControl(cfl=nothing,bound=(pb,u,t)->.2))
    sol=solver(pb,[0.,.3])
    @test diagnostics(sol).rejected_steps==1
    @test all(u->minimum(u)>=0,sol.u)
    @test all(u->sum(u)≈1.,sol.u) # accepted source + bulk inventory
    @test pb.u0==[0.,0.,1.]
    @test_throws NumericalFailure simulate(pb;solver=ForwardSolver(ForwardEuler();dt=.3,
        control=StepControl(cfl=nothing,bound=(pb,u,t)->.3,max_rejections=0)))
    @test_throws NumericalFailure simulate(pb;solver=ForwardSolver(ForwardEuler();dt=.3,maxsteps=1,
        control=StepControl(cfl=nothing,bound=(pb,u,t)->.3)))
    @test_throws ArgumentError simulate(pb;solver=ForwardSolver(;dt=.1,
        control=StepControl(cfl=nothing,guard=(pb,u,t)->1)))
    failure=compile(PBMModel(source=Birth((x,p,t,c)->error("programming error"))),g)
    @test_throws ErrorException simulate(PBMProblem(failure,zeros(2),(0.,1.));
        solver=ForwardSolver(;dt=.1,control=StepControl()))
    loss=compile(PBMModel(removal=Death((x,p,t,c)->p.k)),g)
    controlled=ForwardSolver(;dt=.2,control=StepControl())
    decimal=PBMProblem(loss,g.widths,(0.,1.);p=(k=1.,))
    decimal_sol=ForwardSolver(;dt=.1,control=StepControl())(decimal,[1.])
    @test sum(last(decimal_sol.u))≈exp(-1.) atol=2e-5
    dualsolve=k->sum(last(controlled(PBMProblem(loss,g.widths,(0.,.2);p=(k=k,)),[.2]).u))
    @test_throws ArgumentError ForwardDiff.derivative(dualsolve,.2)
    # A rejected trial must also undo historical escape increments.
    escaping=compile(PBMModel(growth=Growth((x,p,t,c)->-1.)),g;
        bc=SizeBC(OpenBoundary(),NoFlux()))
    ledger_system=PBMSystem((A=escaping,);bulk_rhs=(ctx,p,t)->(available=-10ctx.bulk.available,),
        nonnegative_bulk=(:available,),accounting=(CumulativeAccounting(:A;properties=(v=1,)),))
    ledger_pb=PBMProblem(ledger_system,(A=[1.,0.],),(0.,.3);bulk0=(available=1.,))
    ledger_sol=solver(ledger_pb,[0.,.3])
    @test diagnostics(ledger_sol).rejected_steps==1
    for u in ledger_sol.u
        escaped=cumulative(ledger_pb,u,:A,:lower_escape,:number)
        @test sum(population(ledger_pb,u,:A))+escaped≈1. atol=2e-16
    end
end

@testset "Physical initialization: integrals, support, moments and remake" begin
    g=Grid1D([0.,.1,.7,1.])
    op=compile(PBMModel(),g)
    desc=InitialDistribution(Uniform(0.,2.);total=3.)
    pb=PBMProblem(op,desc,(0.,1.))
    @test vec(pb.u0)≈1.5g.widths atol=2e-16
    @test initialization_report(pb).excluded_number≈1.5
    renormalized=InitialDistribution(Uniform(0.,2.);total=3.,truncation=:renormalize)
    @test sum(initialize(op,renormalized))≈3.
    density=InitialDensity((x,p,t)->p.A*(x+t);support=(0.,2.))
    a=PBMProblem(op,density,(.2,1.);p=(A=2.,))
    @test vec(a.u0)≈diff(g.edges.^2).+.4g.widths atol=2e-15
    @test initialization_report(a).excluded_number≈3.4 atol=3e-15
    b=remake(a;p=(A=3.,))
    @test b.u0≈1.5a.u0
    @test a.p.A==2. && a.u0!=b.u0
    @test remake(a;u0=ones(3),p=(A=3.,)).initial_description===nothing
    @test remake(pb;u0=pb.u0,p=(A=100.,)).u0==pb.u0
    f=A->sum(remake(a;p=(A=A,)).u0)
    @test ForwardDiff.derivative(f,2.)≈.7 atol=2e-15
    gaussian=InitialDistribution(Normal(.3,.1);total=2.)
    @test vec(initialize(op,gaussian))≈2diff(cdf.(Normal(.3,.1),g.edges)) atol=3e-16
    tensor=Grid2D([0.,.2,1.],[0.,.3,2.])
    joint=InitialDensity((xy,p,t)->p.A*(1+xy[1]*xy[2]);support=((0.,1.),(0.,2.)))
    tstate=initialize(compile(PBMModel(),tensor),joint;p=(A=2.,))
    exact=2 .* (tensor.x.widths*tensor.y.widths' + diff(tensor.x.edges.^2)*diff(tensor.y.edges.^2)'/4)
    @test tstate≈exact atol=3e-15
    independent=InitialDistribution(product_distribution([Uniform(0.,2.),Uniform(0.,3.)]);total=6.)
    joint_pb=PBMProblem(compile(PBMModel(),tensor),independent,(0.,1.))
    @test joint_pb.u0≈tensor.x.widths*tensor.y.widths' atol=3e-15
    @test initialization_report(joint_pb).excluded_number≈4. atol=3e-15
    for family in (ExactClosure(),QMOM(),RankAdaptiveQMOM(),DQMOM())
        mop=compile(PBMModel(),MomentSet(0:3);moment_method=family)
        mp=PBMProblem(mop,InitialDistribution(Gamma(1.,1.);total=2.),(0.,1.))
        @test [only(observe(MomentObservable(k),mp,mp.u0;time=0.)) for k in 0:3]≈[2.,2.,4.,12.] atol=2e-13
    end
    for family in (LogNormalClosure(),EQMOM())
        mop=compile(PBMModel(),MomentSet(0:2);moment_method=family)
        @test size(initialize(mop,InitialDistribution(LogNormal(0.,.4);total=1.)))==(3,1)
    end
    normalop=compile(PBMModel(),MomentSet(0:3);moment_method=ExactClosure())
    halfnormal=initialize(normalop,InitialDistribution(truncated(Normal(),0.,Inf);total=2.))
    @test vec(halfnormal)≈2 .* [1.,sqrt(2/pi),1.,2sqrt(2/pi)] atol=2e-15
    @test_throws ArgumentError initialize(op,InitialDensity((x,p,t)->-1.))
    @test_throws ArgumentError initialize(op,InitialDistribution(Uniform(2.,3.);total=1.,truncation=:renormalize))
    momentop=compile(PBMModel(),MomentSet(0:3);moment_method=DQMOM())
    @test_throws NumericalFailure initialize(momentop,InitialDistribution(Gamma(1.,1.);total=0.))
    @test initialize(op,InitialDensity((x,p,t)->0.))==zeros(3,1)
    sys=PBMSystem((A=op,B=compile(PBMModel(),tensor)))
    sp=PBMProblem(sys,(A=density,B=joint),(.2,1.);bulk0=(c=2.,),p=(A=2.,))
    rebuilt=remake(sp;p=(A=3.,))
    @test population(rebuilt,rebuilt.u0,:A)≈1.5population(sp,sp.u0,:A)
    @test population(rebuilt,rebuilt.u0,:B)≈1.5population(sp,sp.u0,:B)
    @test bulk(rebuilt,rebuilt.u0,:c)==2.
    @test initialization_report(sp).A.policy===:physical
end

@testset "Continuous daughters: independent count, volume, tails and parameter binding" begin
    g=Grid1D([0.,.2,.5,1.,2.])
    # b(r)=2 is a binary uniform volume split: integral r*b dr = 1.
    law=ContinuousDaughters((r,parent,p,t,c)->2.;support=(0.,1.),count=2.)
    op=compile(PBMModel(fragmentation=Breakage((x,p,t,c)->1.,law;overflow=:discard)),g)
    run=prepare_run(op,(;));u=reshape([0.,0.,0.,1.],:,1);du=zero(u)
    rhs!(du,u,run,(;),0.)
    # Unresolved count=2*x_min/parent and property=x_min^2/parent.
    x=last(g.centers);xmin=first(g.centers)
    @test sum(du)≈1-2xmin/x atol=2e-14
    @test only(moment(g,du,1))≈-xmin^2/x atol=2e-14
    sys=PBMSystem((A=op,);accounting=(CumulativeAccounting(:A;properties=(volume=1,)),))
    pb=PBMProblem(sys,(A=vec(u),),(0.,.2))
    sol=simulate(pb;dt=.001,method=RK4())
    @test minimum(last(sol.u))>=0
    rates=accounting_rates(pb,pb.u0)
    @test rates.A.breakage_underflow.number≈2xmin/x atol=2e-14
    @test rates.A.breakage_underflow.volume≈xmin^2/x atol=2e-14
    for state in sol.u
        resolved=only(moment(g,population(pb,state,:A),1))
        escaped=cumulative(pb,state,:A,:breakage_underflow,:volume)
        @test resolved+escaped≈x atol=3e-14
    end
    # Independently integrate the uniform daughter's linear pivot hats and
    # compare the transient against a matrix exponential (not the planner).
    Q=zeros(length(g),length(g))
    for j in 1:length(g)
        parent=g.centers[j];Q[j,j]-=1
        for i in 1:j-1
            a,b=g.centers[i:i+1]
            Q[i,j]+=(b-a)/parent
            Q[i+1,j]+=(b-a)/parent
        end
    end
    @test vec(population(pb,last(sol.u),:A))≈exp(.2Q)*vec(u) atol=3e-14
    # Nonadditive length: b(r)=6r² preserves length³, total count=2.
    lengthlaw=ContinuousDaughters((r,y,p,t,c)->6r^2;support=(0.,1.),count=2.,conserved_order=3)
    lengthop=compile(PBMModel(fragmentation=Breakage((x,p,t,c)->1.,lengthlaw;
        overflow=:discard,coordinate_map=AdditiveCoordinate(x->x^3,cbrt))),g)
    dlength=zero(u);rhs!(dlength,u,prepare_run(lengthop,(;)),(;),0.)
    @test only(moment(g,dlength,3))≈-xmin^6/x^3 atol=3e-13
    # Existing non-power additive maps also remain usable. In log-volume x,
    # binary uniform daughter volume on [1,exp(parent)-1] conserves exp(x).
    loggrid=Grid1D([.8,1.,1.5,2.];coordinate=:log_volume)
    logarithmic=ContinuousDaughters((x,parent,p,t,c)->2exp(x)/(exp(parent)-2);
        coordinates=:daughter,support=(parent,p)->(0.,log(exp(parent)-1)),
        count=2.,conserved_order=nothing)
    logop=compile(PBMModel(Breakage((x,p,t,c)->1.,logarithmic;
        overflow=:discard,coordinate_map=AdditiveCoordinate(exp,log))),loggrid)
    logu=reshape([0.,0.,1.],:,1);logdu=zero(logu)
    rhs!(logdu,logu,prepare_run(logop,(;)),(;),0.)
    expected_fines=(exp(2first(loggrid.centers))-1)/(exp(last(loggrid.centers))-2)
    @test sum(exp.(loggrid.centers).*vec(logdu))≈-expected_fines atol=2e-13
    @test daughter_moment(lengthlaw,2.,2,(;),0.,nothing)≈6*2.0^2/5 atol=2e-13
    # Parameterized asymmetric density; convex mixtures of volume-conserving
    # laws remain conservative and must reprepare when p changes.
    mixed=ContinuousDaughters((r,y,p,t,c)->p.a*2+(1-p.a)*6*(1-r);
        support=(0.,1.),count=(y,p)->2p.a+3*(1-p.a))
    mop=compile(PBMModel(Breakage((x,p,t,c)->1.,mixed;overflow=:discard)),g)
    base=PBMProblem(mop,vec(u),(0.,.1);p=(a=.4,))
    changed=remake(base;p=(a=.7,))
    fresh=PBMProblem(mop,vec(u),(0.,.1);p=(a=.7,))
    @test simulate(changed;dt=.001).u==simulate(fresh;dt=.001).u
    @test base.p.a==.4
    f=a->sum(last(simulate(remake(base;p=(a=a,));dt=.01).u))
    @test ForwardDiff.derivative(f,.4)≈(f(.4+1e-5)-f(.4-1e-5))/2e-5 rtol=3e-8
    invalid=ContinuousDaughters((r,y,p,t,c)->1.;support=(0.,1.),count=1.)
    bad=compile(PBMModel(Breakage((x,p,t,c)->1.,invalid;overflow=:discard)),g)
    @test_throws ArgumentError prepare_run(bad,(;))
    @test_throws ArgumentError compile(PBMModel(Breakage(PolynomialRate((1.,)),law)),MomentSet(0:3);moment_method=ExactClosure())
    momentop=compile(PBMModel(Breakage((x,p,t,c)->1.,law)),MomentSet(0:3);moment_method=QMOM())
    mu=reshape([1.,1.,2.,6.],:,1);dmu=zero(mu)
    rhs!(dmu,mu,momentop,(;),0.)
    @test vec(dmu)≈[1.,0.,-2/3,-3.] atol=2e-13
    @test dmu[2]==0. # declared conservation contribution is analytical
    historyop=compile(PBMModel(Breakage((x,p,t,c)->p.k,law;overflow=:discard)),g)
    historysystem=PBMSystem((A=historyop,);accounting=(CumulativeAccounting(:A;properties=(volume=1,)),))
    historypb=PBMProblem(historysystem,(A=vec(u),),(0.,.2);p=(k=1.,),
        history=OperatingHistory([.1];parameters=((k=1.,),(k=2.,))))
    @test accounting_rates(historypb,historypb.u0;time=.1).A.breakage_underflow.number≈4xmin/x atol=3e-14
    shaped=ContinuousDaughters((r,y,p,t,c)->p.a/(1-exp(-p.a))*(exp(-p.a*r)+exp(-p.a*(1-r)));
        support=(0.,1.),count=2.)
    shaped_model=PBMModel(Breakage((x,p,t,c)->1.,shaped;overflow=:discard))
    coarse=compile(shaped_model,g;methods=TermMethods(breakage=FixedPivot(
        daughter_quadrature=GaussLegendre(1),daughter_maxorder=4,daughter_rtol=1e-12)))
    @test_throws NumericalFailure prepare_run(coarse,(a=20.,))
    resolved=prepare_run(compile(shaped_model,g),(a=20.,))
    rhs!(du,u,resolved,(a=20.,),0.)
    @test sum(du)>0
    dynamic=ContinuousDaughters((r,y,p,t,c)->2.;support=(0.,1.),count=2.,dependencies=(:time,))
    @test_throws ArgumentError prepare_run(compile(PBMModel(Breakage((x,p,t,c)->1.,dynamic;overflow=:discard)),g),(;))
end
end
