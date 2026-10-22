# Analytical tests of the opt-in positive-measure policy and physical ledgers.
using LinearAlgebra

@testset "Positive measures, extinction and accounting" begin
    raw(x,w,K)=[sum(w.*x.^k) for k in 0:K]
    uniform(k,t;a=1.,b=2.,N=1.,g=1.)=N/((b-a)*(k+1))*(max(b-g*t,0.)^(k+1)-max(a-g*t,0.)^(k+1))
    function system_problem(op,u;tf=2.1,c=1.,factor=.7,mechanisms=(:growth,:nucleation),account=true)
        exchange=op isa PopulationBalanceModels.CompiledMomentPBM ? MomentExchange(:p,:c,3;factor,mechanisms) :
            ConservativeExchange(:p,:c,(x,p,t,ctx)->factor*x^3;mechanisms)
        sys=PBMSystem((p=op,);exchanges=(exchange,),accounting=account ? (CumulativeAccounting(:p),) : (),nonnegative_bulk=(:c,))
        PBMProblem(sys,(p=u,),(0.,tf);bulk0=(c=c,))
    end
    @testset "Empty startup and independent temporal refinement" begin
        for n in (1,2,3),family in (:adaptive,:fixed,:exact)
            g=MomentSet(0:2n-1;coordinate=:length)
            closure=family===:adaptive ? RankAdaptiveQMOM(n) : family===:fixed ? QMOM(n) : ExactClosure()
            op=compile(PBMModel(Growth(PolynomialRate((.8,)))+Nucleation((p,t,ctx)->1.7,0.)),g;closure)
            method=family===:exact ? MeasureEuler(reconstruction=RankAdaptiveQMOM(n)) : MeasureEuler()
            for dt in (.2,.1)
                sol=simulate(PBMProblem(op,zeros(2n),(0.,1.));dt,saveat=[0.,.2,.6,1.],method)
                @test all(isapprox(vec(u),[1.7*.8^k*t^(k+1)/(k+1) for k in g.orders];rtol=2e-8,atol=1e-13) for (u,t) in zip(sol.u,sol.t))
                @test diagnostics(sol).admissibility_failures==0
                @test all(realizability(g,u).valid for u in sol.u)
            end
        end
        # A rate callback verifies that every physical evaluation sees admissible
        # moments, including the exactly empty start (no invalid Euler predictor).
        calls=Ref(0);g=MomentSet(0:3)
        checked=(p,t,ctx)->begin
            quadrature(ctx.grid,ctx.N,RankAdaptiveQMOM(2));calls[]+=1;1.
        end
        op=compile(PBMModel(Growth((x,p,t,ctx)->.5)+Nucleation(checked,0.)),g;closure=RankAdaptiveQMOM(2))
        sol=simulate(PBMProblem(op,zeros(4),(0.,.4));dt=.1,method=MeasureEuler())
        @test calls[]>0
        @test diagnostics(sol).accepted_steps>0
        # Positive pushforward is first order for variable growth, independently
        # of spatial/quadrature refinement: one atom follows dx/dt=x.
        op=compile(PBMModel(Growth((x,p,t,ctx)->x)),g;closure=RankAdaptiveQMOM(2))
        errors=[abs(simulate(PBMProblem(op,ones(4),(0.,.5));dt,method=MeasureEuler()).u[end][2]-exp(.5)) for dt in (.1,.05,.025)]
        @test all(1.7 .< errors[1:2]./errors[2:3] .< 2.1)
        # Coupled birth, growth and aggregation, beyond transport-only examples.
        op=compile(PBMModel(Growth((x,p,t,ctx)->.1)+Nucleation((p,t,ctx)->.2,.2)+Aggregation(ConstantKernel(.1))),g;closure=RankAdaptiveQMOM(2))
        sol=simulate(PBMProblem(op,zeros(4),(0.,1.));dt=.02,method=MeasureEuler())
        @test all(realizability(g,u).valid for u in sol.u)
        @test all(isfinite,sol.u[end]) && sol.u[end][1]>.15
        budget=diagnostics(sol).number_changes
        @test sum(values(budget))≈sol.u[end][1] atol=1e-10
        @test budget[(:particles,:aggregation)]<0
        # Sectional CFL evaluation must preserve the shared bulk context too.
        grid=Grid1D([0.,.5,1.,2.];coordinate=:length)
        sop=compile(PBMModel(Growth((L,p,t,ctx)->.1*ctx.bulk.c)),grid)
        spb=system_problem(sop,[0.,1.,0.];tf=.1)
        ss=simulate(spb;dt=.01,method=MeasureEuler())
        @test bulk(spb,ss.u[end],:c)+.7sum(population(spb,ss.u[end],:p).*grid.centers.^3)≈1+.7*.75^3 atol=1e-12
    end
    @testset "Atomic hitting events and solute exchange" begin
        g=MomentSet(0:3;coordinate=:length)
        op=compile(PBMModel(Growth((x,p,t,ctx)->-1.)),g;closure=RankAdaptiveQMOM(2),boundary=MomentBoundary(left=:absorb))
        pb=system_problem(op,raw([1.,2.],[1.,2.],3);factor=.7)
        ts=[0.,.5,1.,1.5,2.,2.1]
        sol=simulate(pb;dt=.08,saveat=ts,method=MeasureEuler())
        for (t,u) in zip(ts,sol.u)
            alive=[x>t+1e-9 for x in (1.,2.)]
            exact=[sum((w*max(x-t,0.)^k for (x,w,a) in zip((1.,2.),(1.,2.),alive) if a);init=0.0) for k in 0:3]
            @test population(pb,u,:p)≈exact rtol=2e-9 atol=1e-10
            @test cumulative(pb,u,:p,:extinction)≈3-exact[1] atol=1e-9
            @test cumulative(pb,u,:p,:extinction,:material)==0
            @test bulk(pb,u,:c)+.7*population(pb,u,:p)[4]≈1+.7*17 atol=2e-9
        end
        @test all(iszero,population(pb,sol.u[end],:p))
        @test diagnostics(sol).active_rank[end][:p]==0
        @test maximum(abs,diagnostics(sol).reconstruction_defect[:p])<1e-8
        # Empty absolute observables remain defined; no fabricated normalized PSD.
        @test PopulationBalanceModels._observe(PopulationMoment(:p,0),sol.u[end],pb.operator,(;),2.1)==[0.]
        @test_throws ArgumentError simulate(pb;dt=.1,check_nonnegative=false,method=MeasureEuler())
    end
    @testset "Continuum dissolution: quadrature versus grid error" begin
        moment_errors=Float64[]
        full_moment_errors=Vector{Float64}[]
        for n in (2,4,6)
            g=MomentSet(0:2n-1;coordinate=:length)
            op=compile(PBMModel(Growth((x,p,t,c)->-1.)),g;closure=RankAdaptiveQMOM(n;rtol=1e-10),boundary=MomentBoundary(left=:absorb))
            pb=system_problem(op,[uniform(k,0.) for k in g.orders])
            ts=collect(0.:.05:2.1)
            sol=simulate(pb;dt=.05,saveat=ts,method=MeasureEuler(reconstruction_rtol=1e-9))
            push!(moment_errors,maximum(abs(population(pb,u,:p)[1]-uniform(0,t)) for (u,t) in zip(sol.u,ts)))
            push!(full_moment_errors,[maximum(abs(population(pb,u,:p)[k+1]-uniform(k,t)) for (u,t) in zip(sol.u,ts)) for k in 1:3])
            @test all(iszero,population(pb,sol.u[end],:p))
            @test all(population(pb,u,:p)[1]≈1. for (u,t) in zip(sol.u,ts) if t<1.)
            @test all(abs(bulk(pb,u,:c)-1-.7*(uniform(3,0.)-population(pb,u,:p)[4]))<2e-7 for u in sol.u)
        end
        @test moment_errors[3]<moment_errors[2]<moment_errors[1]
        @test all(full_moment_errors[3].<full_moment_errors[2].<full_moment_errors[1])
        @test moment_errors[3]<.12 # finite quadrature approximates a continuum trace
        sectional_errors=Float64[]
        for bins in (40,80,160)
            grid=Grid1D(range(0.,3.;length=bins+1);coordinate=:length)
            N=[max(0.,min(r,2.)-max(l,1.)) for (l,r) in zip(grid.edges[1:end-1],grid.edges[2:end])]
            op=compile(PBMModel(Growth((x,p,t,c)->-1.)),grid;bc=SizeBC(AbsorbingZero(),NoFlux()))
            pb=system_problem(op,N)
            sol=simulate(pb;dt=.3*minimum(grid.widths),saveat=[0.,.5,1.5,2.1],method=MeasureEuler())
            push!(sectional_errors,abs(sum(population(pb,sol.u[3],:p))-.5))
            initial=1+.7*sum(N.*grid.centers.^3)
            @test all(abs(bulk(pb,u,:c)+.7*sum(population(pb,u,:p).*grid.centers.^3)-initial)<2e-12*initial for u in sol.u)
            @test cumulative(pb,sol.u[end],:p,:extinction,:material)==0
            # Historical particle removal exactly complements resolved number.
            @test sum(population(pb,sol.u[end],:p))+cumulative(pb,sol.u[end],:p,:extinction)≈sum(N) atol=2e-12
            empty=zeros(length(sol.u[end]))
            obs=PopulationHistogram(:p,grid.edges;normalize=true)
            @test_throws NumericalFailure PopulationBalanceModels._observe(obs,empty,pb.operator,(;),2.1)
        end
        @test sectional_errors[end]<sectional_errors[1]
        @test_throws ArgumentError compile(PBMModel(),Grid1D([1.,2.,3.]);bc=SizeBC(AbsorbingZero(),NoFlux()))
    end
    @testset "Finite computational boundary and fixed-pivot losses" begin
        g=MomentSet(0:3;coordinate=:length)
        op=compile(PBMModel(Growth((x,p,t,ctx)->1.)),g;closure=RankAdaptiveQMOM(2),boundary=MomentBoundary(upper=2.,right=:escape))
        pb=system_problem(op,raw([.5,1.],[1.,2.],3);tf=1.6,c=20.)
        sol=simulate(pb;dt=.1,saveat=[0.,.5,1.,1.5,1.6],method=MeasureEuler())
        for (t,u) in zip(sol.t,sol.u)
            esc=sum((w for (x,w) in zip((.5,1.),(1.,2.)) if x+t>=2.0-1e-9);init=0.0)
            @test cumulative(pb,u,:p,:upper_escape)≈esc atol=1e-9
            @test cumulative(pb,u,:p,:upper_escape,:material)≈8esc atol=1e-8
            @test bulk(pb,u,:c)+.7*(population(pb,u,:p)[4]+cumulative(pb,u,:p,:upper_escape,:material))≈20+.7*(.125+2) atol=2e-8
        end
        @test all(iszero,population(pb,sol.u[end],:p))
        @test cumulative(pb,sol.u[end],:p,:extinction)==0
        @test sum(values(diagnostics(sol).number_changes))-cumulative(pb,sol.u[end],:p,:upper_escape)≈-3. atol=1e-9
        cutoff=compile(PBMModel(Growth((x,p,t,c)->-1.)),g;closure=RankAdaptiveQMOM(2),
            boundary=MomentBoundary(lower=.5,left=:escape))
        cpb=system_problem(cutoff,ones(4);tf=.6)
        cs=simulate(cpb;dt=.05,method=MeasureEuler())
        @test cumulative(cpb,cs.u[end],:p,:lower_escape)≈1. atol=1e-9
        @test cumulative(cpb,cs.u[end],:p,:lower_escape,:material)≈.5^3 atol=1e-10
        @test bulk(cpb,cs.u[end],:c)≈1+.7*(1-.5^3) atol=1e-10
        @test cumulative(cpb,cs.u[end],:p,:extinction)==0
        upper_errors=Float64[]
        for n in (2,4,6)
            basis=MomentSet(0:2n-1;coordinate=:length)
            init=[(.8^(k+1)-.4^(k+1))/(.4*(k+1)) for k in basis.orders]
            op=compile(PBMModel(Growth((x,p,t,c)->1.)),basis;closure=RankAdaptiveQMOM(n;rtol=1e-10),
                boundary=MomentBoundary(upper=1.,right=:escape))
            upb=system_problem(op,init;tf=.7,c=20.)
            us=simulate(upb;dt=.02,saveat=0.:.02:.7,method=MeasureEuler(reconstruction_rtol=1e-9))
            push!(upper_errors,maximum(abs(population(upb,u,:p)[1]-clamp((1-t-.4)/.4,0.,1.)) for (t,u) in zip(us.t,us.u)))
            @test all(abs(cumulative(upb,u,:p,:upper_escape,:material)-cumulative(upb,u,:p,:upper_escape))<1e-8 for u in us.u)
            @test all(abs(bulk(upb,u,:c)+.7*(population(upb,u,:p)[4]+cumulative(upb,u,:p,:upper_escape,:material))-20-.7*init[4])<1e-7 for u in us.u)
        end
        @test upper_errors[3]<upper_errors[2]<upper_errors[1]
        # Sectional face uses its resolved endpoint pivot property, consistently
        # with its center-weighted discrete material, rather than edge^3.
        grid=Grid1D([0.,1.,2.,3.];coordinate=:length)
        sop=compile(PBMModel(Growth((x,p,t,c)->1.)),grid;bc=SizeBC(NoFlux(),OpenBoundary()))
        spb=system_problem(sop,[0.,0.,1.];tf=.5,c=30.)
        ssol=simulate(spb;dt=.02,method=SSPRK33())
        for u in ssol.u
            @test bulk(spb,u,:c)+.7*(sum(population(spb,u,:p).*grid.centers.^3)+cumulative(spb,u,:p,:upper_escape,:material))≈30+.7*2.5^3 atol=1e-11
        end
        @test bulk(spb,ssol.u[end],:c)≈30 atol=1e-12 # escaped solid is not dissolved
        @test cumulative(spb,ssol.u[end],:p,:upper_escape,:material)≈2.5^3*cumulative(spb,ssol.u[end],:p,:upper_escape) atol=1e-12
        for which in (:aggregation,:breakage)
            grid=Grid1D([.5,1.5,2.5,3.5];coordinate=:volume)
            term=which===:aggregation ? Aggregation(ConstantKernel(2.);overflow=:discard) : Breakage((x,p,t,c)->2.,(x,p,t,c)->((x/2,2.),);overflow=:discard)
            initial=which===:aggregation ? [0.,0.,2.] : [2.,0.,0.]
            op=compile(PBMModel(term),grid)
            sys=PBMSystem((p=op,);accounting=(CumulativeAccounting(:p;properties=(material=1,)),),
                exchanges=(ConservativeExchange(:p,:c,(x,p,t,c)->x;mechanisms=(which,)),))
            pb=PBMProblem(sys,(p=initial,),(0.,.1);bulk0=(c=10.,))
            mech=which===:aggregation ? :aggregation_overflow : :breakage_underflow
            rates=accounting_rates(pb,pb.u0)
            @test getproperty(getproperty(rates.p,mech),:number)≈(which===:aggregation ? 4. : 8.)
            @test getproperty(getproperty(rates.p,mech),:material)≈(which===:aggregation ? 24. : 4.)
            sol=simulate(pb;dt=.005,method=SSPRK33())
            for u in sol.u
                @test sum(grid.centers.*population(pb,u,:p))+cumulative(pb,u,:p,mech,:material)≈sum(grid.centers.*initial) atol=1e-12
                @test bulk(pb,u,:c)≈10 atol=1e-12
                removed=cumulative(pb,u,:p,mech)
                @test sum(population(pb,u,:p))≈(which===:aggregation ? 2-2removed : 2-removed/2) atol=1e-11
            end
            bad=which===:aggregation ? Aggregation(ConstantKernel(2.)) : Breakage((x,p,t,c)->2.,(x,p,t,c)->((x/2,2.),))
            @test_throws NumericalFailure simulate(PBMProblem(compile(PBMModel(bad),grid),initial,(0.,.1));dt=.005)
        end
    end
    @testset "Conservative axial mixtures on nonuniform cells" begin
        g=MomentSet(0:3;coordinate=:length)
        op=compile(PBMModel(),g;closure=RankAdaptiveQMOM(2))
        localpb=PBMProblem(PBMSystem((p=op,);nonnegative_bulk=(:c,)),(p=ones(4),),(0.,.1);bulk0=(c=3.,))
        z=AxialGrid([0.,.1,.35,.7,1.])
        feed=(p,t)->(populations=(p=raw([.4],[2.],3),),bulk=(c=4.,))
        d=AxialDispersion(z;velocity=.7,dispersion=.015,inlet=feed)
        ap=axial_problem(localpb,d)
        u=copy(ap.u0)
        for i in axes(u,2)
            population(ap,u,:p)[:,i].=raw([.5+.2i],[1.0+.1i],3)
        end
        ap=remake(ap;u0=u)
        h=.001;F=axial_fluxes(ap.operator,u,(;),0.)
        step=simulate(remake(ap;tspan=(0.,h));dt=h,method=MeasureEuler())
        delta=(step.u[end]-u)*z.widths
        @test delta≈h*(F[:,1]-F[:,end]) atol=2e-12
        integ=diagnostics(step).reactor_flux_integrals
        @test delta≈integ.inlet-integ.outlet atol=2e-12
        sol=simulate(remake(ap;tspan=(0.,.3));dt=.3,method=MeasureEuler()) # automatic CFL rejections
        @test diagnostics(sol).rejected_steps>0
        @test all(realizability(g,view(population(ap,state,:p),:,i)).valid for state in sol.u for i in axes(u,2))
        @test all(>=(0),bulk(ap,sol.u[end],:c))
        integ=diagnostics(sol).reactor_flux_integrals
        @test (sol.u[end]-u)*z.widths≈integ.inlet-integ.outlet atol=2e-11
        bad=copy(u);bad[1:4,1].=[1.,0.,1.,0.]
        @test_throws Exception remake(ap;u0=bad)
        @test_throws ArgumentError simulate(axial_problem(localpb,d;axial_scheme=MUSCL());dt=.001,method=MeasureEuler())
        ledgerlocal=PBMProblem(PBMSystem((p=op,);accounting=(CumulativeAccounting(:p),)),(p=ones(4),),(0.,.02))
        periodic=AxialDispersion(z;velocity=.5,dispersion=.01,boundary=:periodic)
        ledgerpb=axial_problem(ledgerlocal,periodic)
        initial=copy(ledgerpb.u0)
        li=ledgerpb.operator.local_operators[1].ledger_indices.p.upper_escape.number
        initial[li,:].=[1.,2.,3.,4.]
        ledgerpb=remake(ledgerpb;u0=initial)
        ledger_sol=simulate(ledgerpb;dt=.001,method=MeasureEuler())
        @test cumulative(ledgerpb,ledger_sol.u[end],:p,:upper_escape)==[1.,2.,3.,4.]
        @test all(iszero,axial_fluxes(ledgerpb.operator,initial,(;),0.)[li,:])
        # Periodic smooth transport: independently refine axial mesh and CFL.
        errors=Float64[]
        for n in (20,40,80)
            zg=AxialGrid([(i/n)^1.15 for i in 0:n])
            domain=AxialDispersion(zg;velocity=1.,dispersion=0.,boundary=:periodic)
            pb=axial_problem(localpb,domain)
            initial=copy(pb.u0)
            for i in 1:n
                amp=1+.2sin(2pi*zg.centers[i])
                initial[1:4,i].=amp.*[1.,.8,.8^2,.8^3]
            end
            pb=remake(pb;u0=initial)
            sol=simulate(pb;dt=.3minimum(zg.widths),method=MeasureEuler())
            exact=1 .+.2sin.(2pi.*(zg.centers.-.1))
            push!(errors,sum(abs.(vec(population(pb,sol.u[end],:p)[1,:]).-exact).*zg.widths))
            @test sol.u[end]*zg.widths≈initial*zg.widths atol=2e-11
        end
        @test errors[3]<errors[2]<errors[1]
    end
    @testset "Unsupported closures, singular states and depletion diagnostics" begin
        for (closure,g) in ((EQMOM(1),MomentSet(0:2)),(LogNormalClosure(),MomentSet(0:2)))
            op=compile(PBMModel(),g;closure)
            @test_throws ArgumentError simulate(PBMProblem(op,zeros(length(g)),(0.,1.));dt=.1,method=MeasureEuler())
        end
        op=compile(PBMModel(),MomentSet(0:3);closure=ExactClosure())
        @test_throws ArgumentError simulate(PBMProblem(op,ones(4),(0.,1.));dt=.1,method=MeasureEuler())
        direct=DQMOM(1);op=compile(PBMModel(),MomentSet(0:1);method=direct)
        @test_throws ArgumentError simulate(PBMProblem(op,quadrature_state(direct,[1.],[1.]),(0.,1.));dt=.1,method=MeasureEuler())
        fixed=compile(PBMModel(Growth((x,p,t,c)->-1.)),MomentSet(0:3);closure=QMOM(2),boundary=MomentBoundary(left=:absorb))
        @test_throws ArgumentError simulate(PBMProblem(fixed,raw([1.,2.],[1.,1.],3),(0.,1.));dt=.1,method=MeasureEuler())
        @test_throws ArgumentError compile(PBMModel(),Grid1D([0.,1.,2.]);bc=SizeBC(NoFlux(),AbsorbingZero()))
        # Necessary PSD tests alone cannot certify a singular truncated sequence.
        g=MomentSet(0:3);u=[1.,0.,0.,1.]
        @test realizability(g,u).valid # PSD is necessary, not sufficient here
        @test_throws Exception quadrature(g,u,RankAdaptiveQMOM(2))
        op=compile(PBMModel(Death((x,p,t,ctx)->20.)),g;closure=RankAdaptiveQMOM(2))
        sol=simulate(PBMProblem(op,ones(4),(0.,.1));dt=.1,method=MeasureEuler())
        @test diagnostics(sol).rejected_steps>0
        @test diagnostics(sol).admissibility_failures==0
    end
end

using ForwardDiff
@testset "Coupled order, material correction and derivative limits" begin
    g=MomentSet(0:3;coordinate=:length)
    op=compile(PBMModel(Growth((x,p,t,ctx)->ctx.bulk.c)),g;closure=RankAdaptiveQMOM(2))
    sys=PBMSystem((p=op,);exchanges=(MomentExchange(:p,:c,3),),nonnegative_bulk=(:c,))
    pb=PBMProblem(sys,(p=ones(4),),(0.,.4);bulk0=(c=1.,))
    L=1.;h=1e-4;f(x)=2-x^3
    for _ in 1:4000
        a=f(L);b=f(L+h*a/2);c=f(L+h*b/2);d=f(L+h*c)
        L+=h*(a+2b+2c+d)/6
    end
    errors=Float64[]
    for dt in (.04,.02,.01)
        sol=simulate(pb;dt,method=MeasureEuler())
        push!(errors,abs(population(pb,sol.u[end],:p)[2]-L))
        @test bulk(pb,sol.u[end],:c)+population(pb,sol.u[end],:p)[4]≈2. atol=1e-12
    end
    @test all(1.8 .< errors[1:2]./errors[2:3] .< 2.2)
    @test_throws ArgumentError ForwardDiff.derivative(x->simulate(remake(pb;p=(k=x,));dt=.01,method=MeasureEuler()).u[end][1],1.)
    grid=Grid1D([.5,1.5,2.5,3.5];coordinate=:volume)
    op=compile(PBMModel(Aggregation(ConstantKernel(2.);overflow=:discard)),grid)
    sys=PBMSystem((p=op,);exchanges=(ConservativeExchange(:p,:c,(x,p,t,c)->x;mechanisms=(:aggregation,)),))
    pb=PBMProblem(sys,(p=[0.,0.,2.],),(0.,.1);bulk0=(c=10.,))
    sol=simulate(pb;dt=.005)
    @test bulk(pb,sol.u[end],:c)≈10. atol=1e-12
end

@testset "Moment product accounting and scaled material balances" begin
    for which in (:aggregation,:breakage)
        basis=MomentSet(0:3;coordinate=:volume)
        term=which===:aggregation ? Aggregation(ConstantKernel(2.);overflow=:discard) :
            Breakage((x,p,t,c)->2.,(x,p,t,c)->((x/2,2.),);overflow=:discard)
        boundary=which===:aggregation ? MomentBoundary(upper=3.5) : MomentBoundary(lower=.75)
        x=which===:aggregation ? 3. : 1.
        op=compile(PBMModel(term),basis;closure=RankAdaptiveQMOM(2),boundary)
        sys=PBMSystem((p=op,);exchanges=(MomentExchange(:p,:c,1;mechanisms=(which,)),),
            accounting=(CumulativeAccounting(:p;properties=(volume=1,)),))
        initial=[2x^k for k in 0:3]
        pb=PBMProblem(sys,(p=initial,),(0.,.01);bulk0=(c=1.,))
        sol=simulate(pb;dt=.01,method=MeasureEuler())
        mech=which===:aggregation ? :aggregation_overflow : :breakage_underflow
        @test cumulative(pb,sol.u[end],:p,mech)≈(which===:aggregation ? .04 : .08) atol=1e-12
        @test population(pb,sol.u[end],:p)[2]+cumulative(pb,sol.u[end],:p,mech,:volume)≈initial[2] atol=1e-12
        @test bulk(pb,sol.u[end],:c)≈1. atol=1e-12
        @test population(pb,sol.u[end],:p)[1]+cumulative(pb,sol.u[end],:p,mech)≈
            initial[1]+sum(values(diagnostics(sol).number_changes)) atol=1e-12
    end
    # Concentration scales span many orders without an absolute clipping floor.
    for (N,unit) in ((1e-8,1e-6),(1e8,1e-6),(1e-8,1e3))
        basis=MomentSet(0:3;coordinate=:length)
        op=compile(PBMModel(Growth((L,p,t,c)->-unit)),basis;closure=RankAdaptiveQMOM(2),
            boundary=MomentBoundary(left=:absorb))
        initial=[N*(.75unit)^k for k in 0:3]
        c0=.7initial[4]
        sys=PBMSystem((p=op,);exchanges=(MomentExchange(:p,:c,3;factor=.7),),
            accounting=(CumulativeAccounting(:p),),nonnegative_bulk=(:c,))
        pb=PBMProblem(sys,(p=initial,),(0.,.8);bulk0=(c=c0,))
        sol=simulate(pb;dt=.02,saveat=[0.,.4,.8],method=MeasureEuler())
        total=c0+.7initial[4]
        @test all(abs(bulk(pb,u,:c)+.7population(pb,u,:p)[4]-total)/total<1e-10 for u in sol.u)
        @test all(iszero,population(pb,sol.u[end],:p))
        @test cumulative(pb,sol.u[end],:p,:extinction)≈N rtol=1e-10
    end
end

using ForwardDiff
@testset "Final boundary and source contracts" begin
    grid=Grid1D([0.,1.,2.];coordinate=:length)
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1.,0.))),grid;evolution=Inguva2022(),bc=SizeBC(AbsorbingZero(),NoFlux()))
    @test_throws ArgumentError compile(PBMModel(),Grid2D(grid,grid);bc=SizeBC2D(x=SizeBC(AbsorbingZero(),NoFlux())))
    basis=MomentSet(0:7;coordinate=:length)
    for n in (3,12)
        term=Breakage((x,p,t,c)->2.,UniformVolumeDaughters(quadrature_order=n);coordinate_map=AdditiveCoordinate(x->x^3,cbrt))
        op=compile(PBMModel(term),basis;closure=RankAdaptiveQMOM(4))
        pb=PBMProblem(op,ones(8),(0.,.01))
        if n==3
            @test_throws ArgumentError simulate(pb;dt=.01,method=MeasureEuler())
        else
            sol=simulate(pb;dt=.01,method=MeasureEuler())
            @test vec(sol.u[end])≈[1+.02*(6/(k+3)-1) for k in 0:7] rtol=1e-8 atol=1e-12
            @test sol.u[end][4]≈1. atol=1e-12
        end
    end
    grid=Grid1D([0.,1.,2.,3.];coordinate=:length)
    op=compile(PBMModel(Growth((x,p,t,c)->p.v)),grid;bc=SizeBC(NoFlux(),OpenBoundary()))
    sys=PBMSystem((p=op,);accounting=(CumulativeAccounting(:p),),exchanges=(ConservativeExchange(:p,:c,(x,p,t,c)->x^3),))
    pb=PBMProblem(sys,(p=[0.,0.,1.],),(0.,.2);bulk0=(c=1.,),p=(v=1.,))
    function escaped(v)
        current=remake(pb;p=(v=v,),u0=pb.u0.+zero(v))
        sol=simulate(current;dt=.01,method=RK4())
        cumulative(current,sol.u[end],:p,:upper_escape)
    end
    @test ForwardDiff.derivative(escaped,1.)≈.2exp(-.2) rtol=1e-8
end

@testset "No artificial hits at stagnant growth boundaries" begin
    g=MomentSet(0:3;coordinate=:length)
    op=compile(PBMModel(Growth((x,p,t,c)->-x)),g;closure=RankAdaptiveQMOM(2),boundary=MomentBoundary(left=:absorb))
    sol=simulate(PBMProblem(op,ones(4),(0.,1.));dt=1.,method=MeasureEuler())
    @test sol.u[end][1]≈1. atol=1e-12
    @test sol.u[end][2]>0
    @test diagnostics(sol).rejected_steps>0
end
