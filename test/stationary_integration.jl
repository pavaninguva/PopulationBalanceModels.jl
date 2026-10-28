using Test, PopulationBalanceModels, NonlinearSolve, LinearAlgebra
@testset "Data-free stationary forward UQ" begin
    g=Grid1D([0.,.4,1.])
    op=compile(PBMModel(Birth((x,p,t,c)->2.)+Death((x,p,t,c)->p.k)),g)
    base=StationaryPBMProblem(op,ones(2);p=(k=1.,),autonomous=true)
    inputs=UncertainInputs([(k=.5,),(k=1.,),(k=2.,)])
    uq=UQProblem(base,inputs;outputs=(number=MomentObservable(0),))
    result=propagate(uq,MonteCarlo(3);max_solves=3)
    @test result.values≈[4. 2. 1.] atol=1e-8
    @test result.layout.times===nothing && diagnostics(result).solves==3
end

@testset "Stationary nonlinear collision moments and closure Jacobians" begin
    p=(J=1.,K=.4,d=.3)
    N=2p.J/(p.d+sqrt(p.d^2+2p.K*p.J))
    M1=p.J/p.d; M2=(p.J+p.K*M1^2)/p.d
    for (closure,raw) in ((QMOM(1),[N,M1]),(RankAdaptiveQMOM(1),[N,M1]),
            (EQMOM(1),[N,M1,M2]),(LogNormalClosure(),[N,M1,M2]))
        model=PBMModel(Nucleation((p,t,c)->p.J,1.)+
            Aggregation(ConstantKernel((p,t,c)->p.K))+
            Death((x,p,t,c)->p.d))
        op=compile(model,MomentSet(0:length(raw)-1);method=closure)
        prob=StationaryPBMProblem(op,.8raw;p,autonomous=true)
        sol=solve_stationary(prob;abstol=1e-8)
        @test sol.converged
        @test vec(sol.u)≈raw rtol=1e-7
        @test p.J-p.d*sol.u[1]-p.K*sol.u[1]^2/2≈0 atol=1e-8
    end
    method=DQMOM(1)
    direct=compile(PBMModel(Nucleation((p,t,c)->p.J,2.)+Death((x,p,t,c)->p.d)),
        MomentSet(0:1);method)
    guess=quadrature_state(method,[2.],[.8p.J/p.d])
    sol=solve_stationary(StationaryPBMProblem(direct,guess;p,autonomous=true))
    @test sol.converged
    @test moment_value(method,direct.grid,sol.u,0)≈p.J/p.d atol=1e-8
    @test moment_value(method,direct.grid,sol.u,1)≈2p.J/p.d atol=1e-8
end

@testset "Analytical MSMPR: solve accuracy versus spatial error" begin
    p=(G=.4,J=1.2,tau=2.)
    errors=Float64[]
    for n in (40,80,160)
        g=Grid1D(range(0,12;length=n+1);coordinate=:length)
        model=PBMModel(growth=Growth((L,p,t,ctx)->p.G),
            births=Nucleation((p,t,ctx)->p.J,0.),
            withdrawal=Dilution((L,p,t,ctx)->1/p.tau))
        op=compile(model,g;methods=TermMethods(nucleation=BoundaryInjection()),
            bc=SizeBC(NoFlux(),OpenBoundary()))
        prob=StationaryPBMProblem(op,zeros(n);p,autonomous=true)
        sol=solve_stationary(prob;abstol=1e-10)
        @test sol.converged
        @test sol.u isa Matrix
        @test diagnostics(sol).scaled_balance_norm<1e-10
        # Independent bidiagonal finite-volume balance, not the assembled rhs!.
        A=Bidiagonal(fill(p.G/g.widths[1]+1/p.tau,n),fill(-p.G/g.widths[1],n-1),:L)
        source=zeros(n); source[1]=p.J
        @test vec(sol.u)≈A\source atol=1e-11
        a=p.G*p.tau
        exact=p.J*p.tau.*(exp.(-g.edges[1:end-1]/a).-exp.(-g.edges[2:end]/a))
        push!(errors,sum(abs.(vec(sol.u).-exact)))
        out=p.G*sol.u[end]/g.widths[end]
        @test p.J≈sum(sol.u)/p.tau+out atol=1e-10
        @test prob.guess==zeros(n,1)
        # Explicit backend and flattened nonlinear-problem interoperability.
        if n==40
            raw=NonlinearSolve.solve(nonlinearproblem(prob),NewtonRaphson();abstol=1e-11,reltol=0)
            @test raw.u≈vec(sol.u) atol=1e-10
            failed=solve_stationary(prob,NewtonRaphson();maxiters=1,abstol=1e-30)
            @test !failed.converged
        end
    end
    @test all(errors[1:2]./errors[2:3].>1.7)

    # Unbounded exact moment equations have no size-domain or grid error.
    model=PBMModel(Growth(PolynomialRate(((p,t,c)->p.G,)))+
        Nucleation((p,t,c)->p.J,0.)+Dilution(PolynomialRate(((p,t,c)->1/p.tau,))))
    op=compile(model,MomentSet(0:3;coordinate=:length))
    exact=[factorial(k)*p.J*p.G^k*p.tau^(k+1) for k in 0:3]
    sol=solve_stationary(StationaryPBMProblem(op,.7exact;p,autonomous=true))
    @test sol.converged
    @test vec(sol.u)≈exact rtol=1e-10
end

@testset "Stationary coupled MSMPR, same-stage exchange and continuation" begin
    p=(J=.5,kg=.2,tau=2.,cs=1.,feed=5.,alpha=.3)
    model=PBMModel(Growth(PolynomialRate(((p,t,c)->p.kg*(c.bulk.c-p.cs),)))+
        Nucleation((p,t,c)->p.J,0.)+Dilution(PolynomialRate(((p,t,c)->1/p.tau,))))
    op=compile(model,MomentSet(0:3;coordinate=:length))
    sys=PBMSystem((crystals=op,);bulk_rhs=(ctx,p,t)->(c=(p.feed-ctx.bulk.c)/p.tau,),
        exchanges=(MomentExchange(:crystals,:c,3;factor=(p,t,ctx)->p.alpha),),nonnegative_bulk=(:c,))
    guess=[factorial(k)*p.J*(p.kg*2)^k*p.tau^(k+1) for k in 0:3]
    prob=StationaryPBMProblem(sys,(crystals=guess,);bulk=(c=3.,),p,autonomous=true)
    sol=solve_stationary(prob;abstol=1e-10)
    @test sol.converged
    # Independent monotone scalar material equation, solved by bisection.
    lo=p.cs; hi=p.feed
    for _ in 1:60
        c=(lo+hi)/2
        f=c+p.alpha*6p.J*(p.kg*(c-p.cs))^3*p.tau^4-p.feed
        if f>0; hi=c; else; lo=c; end
    end
    c=(lo+hi)/2; G=p.kg*(c-p.cs)
    @test bulk(sol,:c)≈c atol=1e-9
    @test population(sol,:crystals)≈[factorial(k)*p.J*G^k*p.tau^(k+1) for k in 0:3] rtol=1e-9
    @test bulk(sol,:c)+p.alpha*population(sol,:crystals)[4]≈p.feed atol=1e-9
    @test stateview(sol).bulk.c==bulk(sol,:c)
    branch=continue_stationary(prob,λ->merge(p,(feed=3.5+1.5λ,)),(0.,1.);abstol=1e-9)
    @test branch.converged
    @test branch.parameters[end]==1
    @test bulk(branch.solutions[end],:c)≈bulk(sol,:c) atol=1e-8
    @test all(s->s.converged,branch.solutions)
    @test length(prob.guess)==5
end

@testset "Stationary ledgers are rates; constraints preserve original balances" begin
    g=Grid1D(range(0,1;length=11))
    op=compile(PBMModel(Growth((x,p,t,c)->.4)+Nucleation((p,t,c)->1.,0.)+
        Dilution((x,p,t,c)->.5)),g;bc=SizeBC(NoFlux(),OpenBoundary()),
        methods=TermMethods(nucleation=BoundaryInjection()))
    sys=PBMSystem((A=op,);accounting=(CumulativeAccounting(:A;properties=(material=3,)),))
    prob=StationaryPBMProblem(sys,(A=zeros(10),);autonomous=true)
    @test length(prob.guess)==prob.operator.physical_length==10
    @test prob.operator.length>10
    sol=solve_stationary(prob)
    @test sol.converged
    rates=accounting_rates(sol)
    @test rates.A.upper_escape.number>0
    @test rates.A.upper_escape.number≈.4population(sol,:A)[end]/g.widths[end]
    @test 1≈.5sum(population(sol,:A))+rates.A.upper_escape.number atol=1e-9
    @test rates.A.upper_escape.material≈rates.A.upper_escape.number*g.centers[end]^3

    # Singular conservative transport needs an inventory equation, not extra rows.
    periodic=compile(PBMModel(Growth((x,p,t,c)->1.)),g;bc=SizeBC(Periodic(),Periodic()))
    constrained=StationaryPBMProblem(periodic,2g.widths;autonomous=true,
        constraints=(1=>(u,p,t)->sum(u)-3,))
    steady=solve_stationary(constrained)
    @test steady.converged
    @test vec(steady.u)≈3g.widths atol=1e-9
    @test maximum(abs,steady.balance_residual)<1e-9
    # Replacing a nonredundant row must not conceal an unsatisfied original rate.
    loss=compile(PBMModel(Death((x,p,t,c)->1.)),g)
    bad=StationaryPBMProblem(loss,ones(10);autonomous=true,
        constraints=(1=>(u,p,t)->u[1]-1,))
    @test !solve_stationary(bad).converged
end

@testset "Stationary axial feed, diffusion and partial continuation failures" begin
    g=Grid1D([0.,.5,1.]); z=AxialGrid([0.,.05,.2,.5,1.])
    d=.3; v=.7; feed=[1.,2.]
    domain=AxialDispersion(z;velocity=v,dispersion=0.,inlet=(p,t)->feed)
    op=compile(PBMModel(Death((x,p,t,c)->d)),g;domain)
    prob=StationaryPBMProblem(op,ones(2,4);autonomous=true)
    sol=solve_stationary(prob)
    @test sol.converged
    expected=hcat((feed.*prod(v./(v.+d.*z.widths[1:c])) for c in 1:4)...)
    @test sol.u≈expected atol=1e-9
    F=axial_fluxes(op,sol.u,prob.p,prob.time)
    @test vec(F[:,1]-F[:,end])≈d.*vec(sum(sol.u.*z.widths';dims=2)) atol=1e-9
    # Size diffusion retains the native Ito equilibrium and wrapped conservation.
    s=range(0,1;length=21); graded=Grid1D(s.^1.3)
    diff=compile(PBMModel(SizeDiffusion((x,p,t,c)->1+x)),graded)
    initial=graded.widths./(1 .+graded.centers)
    target=sum(initial)
    eq=StationaryPBMProblem(diff,1.2initial;autonomous=true,
        constraints=(1=>(u,p,t)->sum(u)-target,))
    @test vec(solve_stationary(eq).u)≈initial atol=1e-9

    birthloss=compile(PBMModel(Birth((x,p,t,c)->1.)+Death((x,p,t,c)->p.d)),g)
    bp=StationaryPBMProblem(birthloss,ones(2);p=(d=1.,),autonomous=true)
    path=continue_stationary(bp,λ->(d=λ<=.6 ? 1. : -1.,),(0.,1.);min_step=.01)
    @test !path.converged
    @test maximum(path.parameters)<=.6
    @test any(a->!a.accepted,path.attempts)
    @test all(s->s.converged,path.solutions)
    @test bp.p==(d=1.,)
end

@testset "Stationary tensor and coupled axial layouts" begin
    grid=Grid2D(Grid1D([0.,.3,1.]),Grid1D([0.,.5,1.5,2.]))
    op=compile(PBMModel(Birth((xy,p,t,c)->2.)+Death((xy,p,t,c)->.5)),grid)
    prob=StationaryPBMProblem(op,zeros(size(grid));autonomous=true)
    sol=solve_stationary(prob)
    @test sol.converged
    @test sol.u≈4 .* grid.x.widths*grid.y.widths' atol=1e-9

    g=Grid1D([0.,.5,1.]); z=AxialGrid([0.,.2,.5,1.])
    localop=compile(PBMModel(Death((x,p,t,c)->.3)),g)
    sys=PBMSystem((A=localop,);bulk_rhs=(ctx,p,t)->(c=-.3ctx.bulk.c,),nonnegative_bulk=(:c,))
    localproblem=StationaryPBMProblem(sys,(A=[1.,2.],);bulk=(c=3.,),autonomous=true)
    domain=AxialDispersion(z;velocity=.7,dispersion=0.,
        inlet=(p,t)->(populations=(A=[1.,2.],),bulk=(c=3.,)))
    steady=axial_problem(localproblem,domain)
    sol=solve_stationary(steady)
    @test sol.converged
    expected=hcat(([1.0,2.0,3.0] .* prod(0.7 ./ (0.7 .+ 0.3 .* z.widths[1:c])) for c in 1:3)...)
    @test sol.u≈expected atol=1e-9
    @test axial_stateview(sol,3).bulk.c≈expected[3,3] atol=1e-9
end
