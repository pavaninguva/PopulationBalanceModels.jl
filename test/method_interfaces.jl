# These extensions are defined by a package user, outside module PopulationBalanceModels.
struct ExactPolynomialBirth <: AbstractTermMethod
    primitive::Function
end
PopulationBalanceModels.prepare_term(::Birth,::ExactPolynomialBirth,::Grid1D;kwargs...)=nothing
function PopulationBalanceModels.add_term!(dN,N,term::Birth,m::ExactPolynomialBirth,cache,op,p,t,ctx)
    for i in eachindex(N)
        dN[i]+=m.primitive(op.grid.edges[i+1],p,t,ctx)-m.primitive(op.grid.edges[i],p,t,ctx)
    end
    (0.0,0.0)
end

# Contract test for a family with a different state layout. This one-node
# growth example is deliberately not advertised as a general DQMOM solver.
struct OneNodeGrowth <: AbstractDirectMomentMethod end
function PopulationBalanceModels.validate_moment_method(::OneNodeGrowth,model,g)
    all(t->t isa Growth,model.terms) || throw(ArgumentError("test method handles growth only"))
end
PopulationBalanceModels.moment_state_length(::OneNodeGrowth,g)=3 # weight, node, auxiliary marker
PopulationBalanceModels.moment_value(::OneNodeGrowth,g,u,k)=u[1]*u[2]^k
PopulationBalanceModels.moment_rate(::OneNodeGrowth,g,u,du,k)=du[1]*u[2]^k+(k==0 ? 0.0 : k*u[1]*u[2]^(k-1)*du[2])
function PopulationBalanceModels.validate_moment_state(::OneNodeGrowth,g,u)
    length(u)==3 && all(isfinite,u) && all(u.>=0) || throw(ArgumentError("invalid direct state"))
end
function PopulationBalanceModels.check_moment_state(::OneNodeGrowth,g,u,tolerance)
    all(isfinite,u) && minimum(u)>=-tolerance || throw(NumericalFailure("invalid direct stage"))
end
PopulationBalanceModels.prepare_moment_state(::OneNodeGrowth,g,u,p,t,ctx)=nothing
function PopulationBalanceModels.moment_term!(du,u,term::Growth,::QuadratureMomentRule,cache,::OneNodeGrowth,q,op,p,t,ctx)
    du[2]+=term.rate(u[2],p,t,ctx)
end

@testset "Per-term methods and family extension interfaces" begin
    @testset "Named policies, quadrature and independent source choices" begin
        g=Grid1D([0.0,0.5,1.0]); N=zeros(2,1); dN=similar(N)
        model=PBMModel(a=Birth((x,p,t,c)->x^2),b=Birth((x,p,t,c)->x^4))
        op=compile(model,g;methods=TermMethods(birth=MidpointRule(),by_term=(b=GaussLegendre(3),)))
        rhs!(dN,N,op,(;),0.0)
        @test term_methods(op).a isa MidpointRule
        @test term_methods(op).b.order==3
        @test sum(dN)≈0.3125+0.2
        @test_throws ArgumentError compile(model,g;methods=TermMethods(by_term=(typo=FixedPivot(),)))
        @test_throws ArgumentError TermMethods(birth=:gauss)
        @test_throws ArgumentError compile(model,g;methods=TermMethods(birth=FixedPivot()))
        @test_throws ArgumentError compile(model,g;methods=TermMethods(typo=FixedPivot()))
        birth=PBMModel(Birth((x,p,t,c)->p.k*x^4))
        extension=ExactPolynomialBirth((x,p,t,c)->p.k*x^5/5)
        exop=compile(birth,g;methods=TermMethods(birth=extension))
        rhs!(dN,N,exop,(k=2.0,),0.0)
        @test sum(dN)≈0.4
        loss=compile(PBMModel(Death((x,p,t,c)->x^2)),g;methods=TermMethods(death=CellAverageLoss()))
        rhs!(dN,ones(2,1),loss,(;),0.0)
        @test vec(dN)≈[-1/12,-7/12]
        transport=Growth((x,p,t,c)->0.2); source=Nucleation((p,t,c)->2.0,0.0)
        bcop=compile(PBMModel(transport+source),g;methods=TermMethods(nucleation=BoundaryInjection()))
        legacy=compile(PBMModel(transport),g;bc=SizeBC(InflowFlux((p,t,c)->2.0),NoFlux()))
        a=similar(N); b=similar(N); rhs!(a,N,bcop,(;),0.0); rhs!(b,N,legacy,(;),0.0)
        @test a≈b
        gaussian=compile(PBMModel(transport+source),g;methods=TermMethods(nucleation=GaussianInjection(width=0.1,truncation=:physical)))
        rhs!(a,N,gaussian,(;),0.0)
        @test sum(a)≈1.0
        @test_throws ArgumentError compile(PBMModel(source),g;methods=TermMethods(nucleation=BoundaryInjection()))
        @test_throws ArgumentError compile(PBMModel(transport+source),g;bc=SizeBC(InflowFlux((p,t,c)->2.0),NoFlux()),methods=TermMethods(nucleation=BoundaryInjection()))
        sys=PBMSystem((a=gaussian,);exchanges=(ConservativeExchange(:a,:c,(x,p,t,c)->x),))
        prob=PBMProblem(sys,(a=zeros(2),),(0.0,0.1);bulk0=(c=2.0,))
        sol=simulate(prob;dt=0.005)
        @test bulk(prob,sol.u[end],:c)+moment(g,population(prob,sol.u[end],:a),1)≈2.0
        agg=compile(PBMModel(Aggregation(ConstantKernel(1.0))),g;methods=TermMethods(aggregation=FixedPivot(overflow=:discard)))
        @test diagnostics(agg,ones(2,1),(;),0.0).unresolved_birth_number_rate[1]>0
    end
    @testset "2D method selection" begin
        g=Grid2D(0:0.5:1,0:0.5:1); N=zeros(2,2); dN=similar(N)
        op=compile(PBMModel(Birth((xy,p,t,c)->xy[1]^2*xy[2]^2)),g;methods=TermMethods(birth=GaussLegendre(3)))
        rhs!(dN,N,op,(;),0.0)
        @test sum(dN)≈1/9
        op=compile(PBMModel(Nucleation2D((p,t,c)->4.0,(0.0,0.0))),g;methods=TermMethods(nucleation=GaussianInjection(width=(0.1,0.1),truncation=:physical)))
        rhs!(dN,N,op,(;),0.0)
        @test sum(dN)≈1.0
        @test_throws ArgumentError compile(PBMModel(Nucleation2D((p,t,c)->1.0,(0.0,0.0))),g;methods=TermMethods(nucleation=BoundaryInjection()))
    end
    @testset "Mixed weak-form rules and a non-QMOM closure" begin
        g=MomentSet(0:3); μ=reshape([1.0,1.0,2.0,6.0],:,1); a=similar(μ); b=similar(μ)
        model=PBMModel(growth=Growth(PolynomialRate((0.1,0.02))),agg=Aggregation(ConstantKernel(0.2)))
        mixed=compile(model,g;method=QMOM(2),methods=TermMethods(by_term=(growth=ExactMomentRule(),)))
        exact=compile(model,g;method=ExactClosure())
        rhs!(a,μ,mixed,(;),0.0); rhs!(b,μ,exact,(;),0.0)
        @test a≈b rtol=1e-12
        @test term_methods(mixed).growth isa ExactMomentRule
        @test all(e->e.treatment==:analytical,moment_plan(mixed).terms.agg)
        @test_throws ArgumentError compile(model,g;method=ExactClosure(),methods=TermMethods(aggregation=QuadratureMomentRule()))
        logg=MomentSet(0:2); initial=reshape([1.0,exp(0.125),exp(0.5)],:,1)
        lm=compile(PBMModel(Growth((x,p,t,c)->sqrt(x))),logg;method=LogNormalClosure())
        derivative=similar(initial); rhs!(derivative,initial,lm,(;),0.0)
        @test vec(derivative)≈[0,exp(0.25*0.5^2/2),2exp(0.25*1.5^2/2)] rtol=1e-10
        @test_throws ArgumentError compile(model,g;method=LogNormalClosure())
        @test quadrature(logg,zeros(3),LogNormalClosure()).weights==zeros(16)
        @test sum(quadrature(logg,[2.0,0.0,0.0],LogNormalClosure()).weights)≈2.0
    end
    @testset "Direct-state family with bulk and observation adapters" begin
        basis=MomentSet(0:1)
        op=compile(PBMModel(Growth((x,p,t,c)->p.g*c.bulk.c)),basis;method=OneNodeGrowth())
        sys=PBMSystem((a=op,);exchanges=(MomentExchange(:a,:c,1),),nonnegative_bulk=(:c,))
        prob=PBMProblem(sys,(a=[2.0,1.0,0.0],),(0.0,0.2);bulk0=(c=3.0,),p=(g=0.1,))
        sol=simulate(prob;dt=0.002)
        @test length(population(prob,sol.u[end],:a))==3
        final=population(prob,sol.u[end],:a)
        @test bulk(prob,sol.u[end],:c)+2final[2]≈5.0 atol=1e-12
        @test PopulationBalanceModels._observe(PopulationMoment(:a,1),sol.u[end],prob.operator,prob.p,0.2)==[2final[2]]
        localprob=PBMProblem(op,[2.0,1.0,0.0],(0.0,0.2);p=(g=0.1,))
        @test PopulationBalanceModels._observe(MomentObservable(1),localprob.u0,op,localprob.p,0.0)==[2.0]
    end
end
