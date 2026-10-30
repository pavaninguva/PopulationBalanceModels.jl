using Test, PopulationBalanceModels

# A raw-moment family outside the implementation, with instrumented preparation
# and missing-moment evaluation. Its two-point distribution is independent of
# the package's inversion algorithm.
struct InterfaceCountingFamily <: AbstractMomentClosure
    preparations::Base.RefValue{Int}
    missing_calls::Base.RefValue{Int}
end
PopulationBalanceModels.validate_moment_method(::InterfaceCountingFamily,model,g)=nothing
PopulationBalanceModels.validate_moment_state(::InterfaceCountingFamily,g,u)=nothing
PopulationBalanceModels.check_moment_state(::InterfaceCountingFamily,g,u,tol)=nothing
function PopulationBalanceModels.prepare_moment_state(m::InterfaceCountingFamily,g,u,p,t,ctx)
    m.preparations[]+=1
    (nodes=[1.,3.],weights=fill(u[1]/2,2))
end
function PopulationBalanceModels.reconstructed_moment(m::InterfaceCountingFamily,g,u,q,k)
    m.missing_calls[]+=1
    sum(q.weights.*q.nodes.^k)
end

# A declared analytical user term, extending the capability and evaluator hooks.
struct InterfaceMomentSink <: AbstractPBMTerm end
PopulationBalanceModels.moment_dependencies(::InterfaceMomentSink,k::Integer)=(k,)
PopulationBalanceModels.analytical_moment(::InterfaceMomentSink,k,getmoment,p,t,ctx)=-p.loss*getmoment(k)

struct InterfaceIntegralSink <: AbstractPBMTerm
    preparations::Base.RefValue{Int}
end
PopulationBalanceModels.supports_moment_integral(::InterfaceIntegralSink,::MomentSet)=true
function PopulationBalanceModels.prepare_term(term::InterfaceIntegralSink,::QuadratureMomentRule,g::MomentSet;kwargs...)
    term.preparations[]+=1
    copy(g.orders)
end
function PopulationBalanceModels.moment_term!(du,u,term::InterfaceIntegralSink,::QuadratureMomentRule,orders,
                                             method,q,op,p,t,ctx)
    for k in orders
        du[k+1]-=p.loss*sum(q.weights.*q.nodes.^k)
    end
end

@testset "Automatic moment compilation" begin
    rhs(op,u;p=(;),t=0.)=(d=zero(reshape(u,:,1));rhs!(d,reshape(u,:,1),op,p,t);vec(d))
    basis=MomentSet(0:3);mu=[1.,1.,2.,6.]

    @testset "Closed supported equations and explicit family identity" begin
        model=PBMModel(grow=Growth(PolynomialRate((.2,.1))),
            birth=Nucleation((p,t,ctx)->.3,.4),removal=Death(PolynomialRate((.05,))))
        expected=[.3-.05,
            .2+.1+.3*.4-.05,
            2*(.2+.1*2)+.3*.4^2-.05*2,
            3*(.2*2+.1*6)+.3*.4^3-.05*6]
        default=compile(model,basis)
        strict=compile(model,basis;moment_method=ExactClosure())
        @test default.closure isa ExactClosure
        @test rhs(default,mu)≈expected
        @test rhs(strict,mu)≈expected
        @test !moment_plan(default).requires_closure
        for family in (QMOM(),RankAdaptiveQMOM())
            op=compile(model,basis;moment_method=family)
            @test typeof(op.closure)==typeof(family)
            @test !moment_plan(op).requires_closure
            @test rhs(op,mu)≈expected
        end
        # Explicit QMOM still rejects a nonempty deficient state even though the
        # evolution formula needs no reconstruction. The default exact path does not.
        explicit=compile(model,basis;moment_method=QMOM())
        @test_throws NumericalFailure rhs(explicit,ones(4))
        @test all(isfinite,rhs(default,ones(4)))
    end

    @testset "Quadratic growth: only missing M4 comes from closure" begin
        model=PBMModel(grow=Growth(PolynomialRate((0.,0.,(p,t,ctx)->p.a))))
        op=compile(model,basis;moment_method=QMOM())
        # The two-node Gaussian rule has nodes 2±sqrt(2). Its node polynomial
        # gives M4=4M3-2M2=20; the unit exponential instead has true M4=24.
        @test rhs(op,mu;p=(a=.7,))≈.7 .*[0.,2.,12.,60.]
        @test rhs(op,mu;p=(a=1.,))[4]!=3factorial(4)
        plan=moment_plan(op)
        @test plan.family.nodes==2
        @test plan.state_layout==:raw_moments
        @test plan.orders==(0,1,2,3)
        @test plan.requires_closure
        @test [e.treatment for e in plan.terms.grow]==[:analytical,:analytical,:analytical,:closure]
        @test plan.terms.grow[end].missing==(4,)
        @test plan.terms.grow[2].dependencies==(2,)
        mixed=compile(PBMModel(grow=Growth(PolynomialRate((1.,2.,.7)))),basis;moment_method=QMOM())
        @test rhs(mixed,mu)[4]≈3*2+6*6+3*.7*20
        @test moment_plan(mixed).terms.grow[end].missing==(4,)
        # A numerical zero in a parameter callback must not erase a dependency.
        @test_throws ArgumentError compile(model,basis;moment_method=ExactClosure())
        @test rhs(op,mu;p=(a=0.,))==zeros(4)
        for size_scale in (1e-6,1e3),number_scale in (1e-9,1e9)
            scales=number_scale .* size_scale .^ (0:3)
            @test rhs(op,mu.*scales;p=(a=.7/size_scale,))./scales≈.7 .* [0.,2.,12.,60.]
        end
        zero_kernel=compile(PBMModel(Aggregation(ConstantKernel(0.);
            coordinate_map=AdditiveCoordinate(x->x^3,cbrt))),MomentSet(0:2);moment_method=ExactClosure())
        @test rhs(zero_kernel,mu[1:3])==zeros(3)
        @test !moment_plan(zero_kernel).requires_closure
        identity_split=compile(PBMModel(Breakage(PolynomialRate((0.,0.,1.)),
            SelfSimilarDaughters((1.,),(1.,)))),basis;moment_method=ExactClosure())
        @test rhs(identity_split,mu)==zeros(4)
    end

    @testset "Ordinary callbacks reuse the physical model" begin
        model=PBMModel(growth=Growth((x,p,t,ctx)->p.a*x^2),
            death=Death((x,p,t,ctx)->.2),primary=Nucleation((p,t,ctx)->.3,.4))
        op=compile(model,basis;moment_method=QMOM())
        expected=.1 .*[0.,2.,12.,60.]-.2 .*mu+.3 .*[.4^k for k in 0:3]
        @test rhs(op,mu;p=(a=.1,))≈expected
        @test moment_plan(compile(model,basis)).family isa RankAdaptiveQMOM
        @test moment_plan(op).terms.growth[1].treatment==:analytical
        @test moment_plan(op).terms.growth[2].treatment==:quadrature
        # Existing direct-discretization API, unchanged, with identical callbacks.
        grid=Grid1D(0.:.1:4.)
        direct=compile(model,grid;methods=TermMethods(growth=MUSCL()))
        initial=grid.widths./sum(grid.widths)
        @test sum(rhs(direct,initial;p=(a=.1,)))≈.3-.2 atol=1e-13
        @test term_methods(direct).growth isa MUSCL
    end

    @testset "Family-specific missing moments and conservation identities" begin
        model=PBMModel(loss=Death(PolynomialRate((0.,1.))))
        g=MomentSet(0:2);initial=[1.,1.,2.]
        gamma=compile(model,g;moment_method=EQMOM(;secondary_nodes=2))
        lognormal=compile(model,g;moment_method=LogNormalClosure())
        @test rhs(gamma,initial)≈[-1.,-2.,-6.]
        @test rhs(lognormal,initial)≈[-1.,-2.,-8.]
        # A two-point secondary rule resolves the tracked Gamma moments but
        # not M4/M5. Forced quadrature deliberately preserves that approximation.
        high=PBMModel(grow=Growth(PolynomialRate((0.,0.,0.,0.,1.))))
        auto=compile(high,g;moment_method=EQMOM(1;secondary_nodes=2))
        forced=compile(high,g;moment_method=EQMOM(1;secondary_nodes=2),
            methods=TermMethods(growth=QuadratureMomentRule()))
        refined=compile(high,g;moment_method=EQMOM(1;secondary_nodes=4),
            methods=TermMethods(growth=QuadratureMomentRule()))
        @test rhs(auto,initial)≈[0.,24.,240.]
        @test rhs(forced,initial)≈[0.,20.,136.]
        @test rhs(refined,initial)≈rhs(auto,initial)
        # Even size-dependent fragmentation conserves the declared daughter power.
        term=Breakage((x,p,t,c)->x^6,UniformVolumeDaughters();coordinate_map=AdditiveCoordinate(x->x^3,cbrt))
        op=compile(PBMModel(split=term),basis;moment_method=QMOM())
        @test moment_plan(op).terms.split[4].treatment==:analytical
        @test rhs(op,mu)[4]==0
        # Exact known daughter moments and polynomial hazards use only missing
        # outer moments, including with a length-based additive-property map.
        gamma=compile(PBMModel(split=Breakage(PolynomialRate((0.,1.)),UniformVolumeDaughters();
            coordinate_map=AdditiveCoordinate(x->x^3,cbrt))),g;moment_method=EQMOM(1))
        @test rhs(gamma,initial)≈[1.,1.,1.2]
    end

    @testset "Exact-only errors and independent empty sources" begin
        @test_throws ArgumentError compile(PBMModel(),basis;moment_method=ExactClosure(),boundary=MomentBoundary(left=:absorb))
        legacy=compile(PBMModel(Growth(PolynomialRate((-1.,)))),basis;
            closure=ExactClosure(),boundary=MomentBoundary(left=:absorb))
        @test moment_plan(legacy).requires_closure
        @test !moment_plan(legacy).boundary.residual_supported
        legacy_solution=simulate(PBMProblem(legacy,ones(4),(0.,1.2));dt=.1,
            method=MeasureEuler(reconstruction=RankAdaptiveQMOM(2)))
        @test vec(last(legacy_solution.u))≈zeros(4) atol=1e-9
        message(f)=try f(); "no error" catch e; @test e isa ArgumentError; sprint(showerror,e) end
        err=message(()->compile(PBMModel(grow=Growth(PolynomialRate((0.,0.,1.)))),basis;moment_method=ExactClosure()))
        @test occursin(":grow",err) && occursin("equation 3",err) && occursin("untracked moment 4",err)
        err=message(()->compile(PBMModel(grow=Growth((x,p,t,c)->x)),basis;moment_method=ExactClosure()))
        @test occursin("no analytical rule",err) && !occursin("untracked",err)
        err=message(()->compile(PBMModel(feed=Birth((x,p,t,c)->exp(-x))),basis;moment_method=QMOM()))
        @test occursin(":feed",err) && occursin("source",err)
        # Unknown callback structure does not prevent a known structural zero.
        onlynumber=compile(PBMModel(grow=Growth((x,p,t,c)->error("must not evaluate"))),MomentSet(0:0);moment_method=ExactClosure())
        @test rhs(onlynumber,[1.])==[0.]
        sources=PBMModel(point=Nucleation((p,t,c)->2.,0.),
            distributed=MomentSource((p,t,c)->3.,(k,p,t,c)->factorial(k)))
        for family in (QMOM(),RankAdaptiveQMOM(),ExactClosure())
            op=compile(sources,basis;moment_method=family)
            @test rhs(op,zeros(4))≈[5.,3.,6.,18.]
            @test rhs(op,mu)≈rhs(op,zeros(4))
        end
    end

    @testset "Resolution, aliases and deliberate overrides" begin
        model=PBMModel(grow=Growth(PolynomialRate((.1,.2))))
        for (family,g,explicit) in ((QMOM(),basis,QMOM(2)),
                (RankAdaptiveQMOM(rtol=1e-9),basis,RankAdaptiveQMOM(2;rtol=1e-9)),
                (EQMOM(secondary_nodes=16),MomentSet(0:4),EQMOM(2;secondary_nodes=16)),
                (DQMOM(condition_limit=1e10),basis,DQMOM(2;condition_limit=1e10)))
            resolved=compile(model,g;moment_method=family).closure
            @test all(getfield(resolved,f)==getfield(explicit,f) for f in fieldnames(typeof(explicit)))
        end
        @test_throws ArgumentError compile(model,MomentSet(0:2);moment_method=QMOM())
        @test_throws ArgumentError compile(model,basis;moment_method=EQMOM())
        @test_throws ArgumentError compile(model,MomentSet(0:4);moment_method=RankAdaptiveQMOM())
        @test_throws ArgumentError compile(model,MomentSet(0:4);moment_method=LogNormalClosure())
        @test_throws ArgumentError compile(PBMModel(Growth((x,p,t,c)->x)),MomentSet(0:2))
        @test_throws ArgumentError compile(model,basis;moment_method=QMOM(),method=QMOM(2))
        @test_throws ArgumentError compile(model,basis;method=QMOM(2),closure=ExactClosure())
        for op in (compile(model,basis;method=QMOM(2)),compile(model,basis;closure=QMOM(2)))
            @test rhs(op,mu)≈[0.,.3,1.,4.2]
        end
        forced=compile(model,basis;moment_method=QMOM(),methods=TermMethods(growth=QuadratureMomentRule()))
        @test moment_plan(forced).terms.grow[2].treatment==:quadrature
        @test rhs(forced,mu)≈[0.,.3,1.,4.2]
        @test_throws ArgumentError compile(model,basis;moment_method=ExactClosure(),methods=TermMethods(growth=QuadratureMomentRule()))
        @test_throws ArgumentError compile(PBMModel(grow=Growth(PolynomialRate((0.,0.,1.)))),basis;
            moment_method=QMOM(),methods=TermMethods(growth=ExactMomentRule()))
    end

    @testset "Current stage and shared preparation/missing-moment cache" begin
        family=InterfaceCountingFamily(Ref(0),Ref(0))
        model=PBMModel(growth=Growth(PolynomialRate((0.,0.,1.))),
            removal=Death(PolynomialRate((0.,1.))))
        raw=[.5*(1+3^k) for k in 0:3]
        op=compile(model,basis;moment_method=family)
        rhs(op,raw)
        @test family.preparations[]==1
        @test family.missing_calls[]==1 # M4 shared by both physical terms
        rhs(op,2raw)
        @test family.preparations[]==2 && family.missing_calls[]==2
        coefficient=(p,t,c)->p.a+t+c.fields.level
        op=compile(PBMModel(Growth(PolynomialRate((coefficient,)))),basis;
            fields=(cell,t)->(level=2t,),moment_method=QMOM())
        @test rhs(op,mu;p=(a=.2,),t=.5)≈[0.,1.7,3.4,10.2]
        @test rhs(op,mu;p=(a=.4,),t=1.)≈[0.,3.4,6.8,20.4]
        extended=compile(PBMModel(sink=InterfaceMomentSink()),basis;moment_method=ExactClosure())
        @test rhs(extended,mu;p=(loss=.2,))≈-.2mu
        term=InterfaceIntegralSink(Ref(0))
        extended=compile(PBMModel(sink=term),basis;moment_method=QMOM())
        @test rhs(extended,mu;p=(loss=.2,))≈-.2mu
        @test rhs(extended,mu;p=(loss=.4,))≈-.4mu
        @test term.preparations[]==1
    end

    @testset "DQMOM state chain rule and coupled material" begin
        nodes=[2-sqrt(2),2+sqrt(2)];weights=[(2+sqrt(2))/4,(2-sqrt(2))/4]
        m=DQMOM();state=quadrature_state(m,nodes,weights)
        op=compile(PBMModel(Growth(PolynomialRate((0.,0.,.1)))),basis;moment_method=m)
        du=rhs(op,state)
        @test moment_plan(op).state_layout==:weights_weighted_abscissas
        @test [moment_rate(op.closure,basis,state,du,k) for k in 0:3]≈.1 .*[0.,2.,12.,60.]
        op=compile(PBMModel(Growth(PolynomialRate(((p,t,c)->p.k*c.bulk.c,)))),basis;moment_method=QMOM())
        system=PBMSystem((crystals=op,);exchanges=(MomentExchange(:crystals,:c,3),),nonnegative_bulk=(:c,))
        pb=PBMProblem(system,(crystals=mu,),(0.,.02);bulk0=(c=2.,),p=(k=.1,))
        sol=simulate(pb;method=SSPRK33(),dt=.001)
        @test all(u->isapprox(bulk(pb,u,:c)+moment(basis,population(pb,u,:crystals),3),8.;atol=2e-12),sol.u)
        @test bulk(pb,last(sol.u),:c)<2.
        @test !moment_plan(op).requires_closure
    end
end
