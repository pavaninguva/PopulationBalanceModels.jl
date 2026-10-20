@testset "Coupled populations and paper requirements" begin
    @testset "Boundary/distributed creation and nonuniform MUSCL" begin
        g=Grid1D([0.0,0.2,0.6,1.2,2.0,3.0,5.0,6.2])
        model=PBMModel(Growth((x,p,t,c)->0.7))
        op=compile(model,g;growth_scheme=MUSCL())
        N=reshape((3 .+2 .*g.centers).*g.widths,:,1); dN=similar(N)
        rhs!(dN,N,op,NamedTuple(),0.0)
        @test sum(dN)≈0 atol=1e-13
        @test vec(dN)[3:5]./g.widths[3:5]≈fill(-1.4,3) atol=1e-13
        backwards=compile(PBMModel(Growth((x,p,t,c)->-0.7)),g;growth_scheme=MUSCL())
        rhs!(dN,N,backwards,NamedTuple(),0.0)
        @test vec(dN)[3:5]./g.widths[3:5]≈fill(1.4,3) atol=1e-13
        injected=compile(model,g;bc=SizeBC(InflowFlux((p,t,c)->2.0),NoFlux()))
        rhs!(dN,zero(N),injected,NamedTuple(),0.0)
        @test sum(dN)≈2.0
        distributed=DistributedNucleation((p,t,c)->3.0,(x,p,t,c)->exp(-((x-1.1)/0.3)^2))
        source=compile(PBMModel(distributed),g)
        rhs!(dN,zero(N),source,NamedTuple(),0.0)
        @test sum(dN)≈3.0
        unresolved=compile(PBMModel(DistributedNucleation((p,t,c)->1.0,(x,p,t,c)->0.0)),g)
        @test_throws NumericalFailure rhs!(dN,N,unresolved,NamedTuple(),0.0)
        off=compile(PBMModel(DistributedNucleation((p,t,c)->0.0,(x,p,t,c)->error("not evaluated"))),g)
        rhs!(dN,N,off,NamedTuple(),0.0)
        @test iszero(sum(dN))
    end

    @testset "Volume-conserving collisions on diameter bins" begin
        g=Grid1D(0.5:1.0:8.5;coordinate=:diameter)
        geometry=AdditiveCoordinate(x->x^3,cbrt)
        agg=compile(PBMModel(Aggregation((x,y,p,t,c)->0.5;coordinate_map=geometry)),g)
        N=zeros(8,1); N[1]=2; dN=similar(N)
        rhs!(dN,N,agg,NamedTuple(),0.0)
        @test sum(dN)≈-1
        @test only(moment(g,dN,3))≈0 atol=1e-13
        @test dN[2]≈1/7
        br=compile(PBMModel(Breakage((x,p,t,c)->x>=2 ? 1.0 : 0.0,
            (x,p,t,c)->((x/cbrt(2),2.0),);coordinate_map=geometry)),g)
        N.=0;N[2]=1
        rhs!(dN,N,br,NamedTuple(),0.0)
        @test sum(dN)≈1
        @test only(moment(g,dN,3))≈0 atol=1e-12
        @test_throws ArgumentError compile(PBMModel(Aggregation((x,y,p,t,c)->1;
            coordinate_map=AdditiveCoordinate(x->x^3,identity))),g)
    end

    @testset "Transfers across different grids" begin
        a=Grid1D(0.5:1:5.5)
        b=Grid1D([0.5,1.2,2.8,4.2,5.8])
        sys=PBMSystem((A=compile(PBMModel(),a),B=compile(PBMModel(),b));
            interactions=(PopulationTransfer(:A,:B,(x,p,t,c)->p.k),))
        n=zeros(5);n[2]=2
        prob=PBMProblem(sys,(A=n,B=zeros(4)),(0.0,1.0);p=(k=0.3,))
        du=similar(prob.u0);rhs!(du,prob.u0,prob.operator,prob.p,0.0)
        @test sum(population(prob,du,:A))≈-0.6
        @test sum(population(prob,du,:B))≈0.6
        @test moment(a,population(prob,du,:A),1)+moment(b,population(prob,du,:B),1)≈0 atol=1e-14
        sol=simulate(prob;dt=0.01)
        @test sum(population(prob,sol.u[end],:A))≈2*exp(-0.3) atol=1e-9
        @test sum(population(prob,sol.u[end],:B))≈2*(1-exp(-0.3)) atol=1e-9
        @test_throws ArgumentError PopulationTransfer(:A,:A,(x,p,t,c)->1)
        reordered=PBMProblem(sys,(B=zeros(4),A=n),(0,1);p=(k=0.3,))
        @test population(reordered,reordered.u0,:A)==n
        @test_throws ArgumentError PBMProblem(sys,(wrong=n,B=zeros(4)),(0,1))
    end

    @testset "Cross collisions have no identical-pair half factor" begin
        g=Grid1D(0.5:1:8.5)
        op=compile(PBMModel(),g)
        sys=PBMSystem((A=op,B=op,C=op);
            interactions=(CrossAggregation(:A,:B,:C,(x,y,p,t,c)->0.2),))
        a=zeros(8);b=zeros(8);a[1]=2;b[2]=3
        prob=PBMProblem(sys,(A=a,B=b,C=zeros(8)),(0.0,0.1))
        du=similar(prob.u0);rhs!(du,prob.u0,prob.operator,prob.p,0.0)
        @test population(prob,du,:A)[1]≈-1.2
        @test population(prob,du,:B)[2]≈-1.2
        @test population(prob,du,:C)[3]≈1.2
        @test sum(du)≈-1.2
        @test sum(moment(g,population(prob,du,name),1) for name in (:A,:B,:C))≈0 atol=1e-13
        @test_throws ArgumentError CrossAggregation(:A,:A,:C,(x,y,p,t,c)->1)
    end

    @testset "Shared bulk depletion closes a discrete balance" begin
        g=Grid1D(0.5:1:5.5)
        A=compile(PBMModel(Nucleation((p,t,c)->p.kA*c.bulk.c,1.0)),g)
        B=compile(PBMModel(Nucleation((p,t,c)->p.kB*c.bulk.c,2.0)),g)
        exchanges=(ConservativeExchange(:A,:c,(x,p,t,c)->x),ConservativeExchange(:B,:c,(x,p,t,c)->x))
        sys=PBMSystem((A=A,B=B);exchanges,nonnegative_bulk=(:c,))
        prob=PBMProblem(sys,(A=zeros(5),B=zeros(5)),(0,1);bulk0=(c=5.0,T=-2.0,),p=(kA=0.1,kB=0.05))
        du=similar(prob.u0);rhs!(du,prob.u0,prob.operator,prob.p,0.0)
        balance=bulk(prob,du,:c)+sum(moment(g,population(prob,du,name),1) for name in (:A,:B))
        @test balance≈0 atol=1e-14
        sol=simulate(prob;dt=0.01)
        @test bulk(prob,sol.u[end],:c)≈5*exp(-0.2) atol=1e-9
        @test bulk(prob,sol.u[end],:T)≈-2
        total=bulk(prob,sol.u[end],:c)+sum(moment(g,population(prob,sol.u[end],name),1) for name in (:A,:B))
        @test total≈5 atol=1e-12
        @test stateview(prob,sol.u[end]).populations.A===population(prob,sol.u[end],:A)
        @test_throws ArgumentError PBMProblem(sys,(A=zeros(5),B=zeros(5)),(0,1);bulk0=(wrong=1.0,))
        @test_throws ArgumentError ConservativeExchange(:A,:c,(x,p,t,c)->x;mechanisms=(:typo,))
    end

    @testset "Bulk state is evaluated at every RK stage" begin
        g=Grid1D(0.5:1:3.5)
        op=compile(PBMModel(Death((x,p,t,c)->c.bulk.c)),g)
        sys=PBMSystem((A=op,);bulk_rhs=(ctx,p,t)->(c=-ctx.bulk.c,),nonnegative_bulk=(:c,))
        prob=PBMProblem(sys,(A=ones(3),),(0,1);bulk0=(c=1.0,))
        sol=simulate(prob;dt=0.005)
        @test bulk(prob,sol.u[end],:c)≈exp(-1) atol=1e-8
        @test population(prob,sol.u[end],:A)≈fill(exp(-(1-exp(-1))),3) atol=1e-8
    end

    @testset "Dissolution returns particle material to the bulk" begin
        g=Grid1D(0.5:1:8.5;coordinate=:diameter)
        op=compile(PBMModel(Growth((x,p,t,c)->-0.2)),g;growth_scheme=MUSCL())
        sys=PBMSystem((A=op,);exchanges=(ConservativeExchange(:A,:c,(x,p,t,c)->x^3),),nonnegative_bulk=(:c,))
        initial=zeros(8);initial[4]=1
        prob=PBMProblem(sys,(A=initial,),(0.0,0.5);bulk0=(c=0.0,))
        du=similar(prob.u0);rhs!(du,prob.u0,prob.operator,prob.p,0.0)
        @test bulk(prob,du,:c)>0
        @test bulk(prob,du,:c)+moment(g,population(prob,du,:A),3)≈0 atol=1e-13
        sol=simulate(prob;dt=0.01)
        @test bulk(prob,sol.u[end],:c)+moment(g,population(prob,sol.u[end],:A),3)≈64 atol=1e-11
    end

    @testset "Joint distribution/bulk observations and fitting residuals" begin
        g=Grid1D(0.5:1:4.5)
        op=compile(PBMModel(Nucleation((p,t,c)->p.k*c.bulk.c,1.0)),g)
        sys=PBMSystem((A=op,);exchanges=(ConservativeExchange(:A,:c,(x,p,t,c)->x),),nonnegative_bulk=(:c,))
        prob=PBMProblem(sys,(A=zeros(4),),(0,1);bulk0=(c=2.0,),p=(k=0.3,))
        ts=[0.0,0.5,1.0]
        number=Observation(ts,2 .* (1 .-exp.(-0.3.*ts)),PopulationMoment(:A,0);sigma=0.01)
        concentration=Observation(ts,2 .*exp.(-0.3.*ts),BulkObservable(:c);sigma=0.01)
        cal=CalibrationProblem([Experiment(prob,number,concentration)];parameters=(z,e)->(k=exp(z[1]),),dt=0.0025)
        @test maximum(abs.(residuals(cal,[log(0.3)])))<1e-8
        @test sensitivity_report(cal,[log(0.3)]).rank==1
        hist=PopulationHistogram(:A,g.edges;weight=x->x^6,normalize=true)
        obs=Observation([0.5],reshape([1.0,0,0,0],:,1),hist;sigma=0.1)
        ex=Experiment(prob,obs) # No particles at t=0: future PSD is still valid.
        hcal=CalibrationProblem([ex];parameters=(z,e)->(k=exp(z[1]),),dt=0.005)
        @test maximum(abs.(residuals(hcal,[log(0.3)])))<1e-12
    end
end
