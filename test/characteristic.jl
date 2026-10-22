using Distributions: Normal, cdf, pdf

@testset "Inguva 2022: physical meshes, continuous densities and exact bin shifts" begin
    open=SizeBC(OpenBoundary(),OpenBoundary())
    p=(g0=0.434,g1=0.2604);law=AffineGrowth(:g0,:g1)
    grid=characteristic_grid(law,(0.0,1.4);dt=0.01,p)
    op=compile(PBMModel(growth=Growth(law)),grid;evolution=Inguva2022(),p,bc=open)
    @test last(grid.edges)>=1.4
    @test maximum(grid.widths)>minimum(grid.widths)
    @test term_methods(op).growth isa Inguva2022
    normal=Normal(0.2,sqrt(0.0005/2))
    integral(lo,hi)=hi<=lo ? 0.0 : cdf(normal,hi)-cdf(normal,lo)
    counts(t)=[integral(max(0.,(a-p.g0*expm1(p.g1*t)/p.g1)*exp(-p.g1*t)),
        max(0.,(b-p.g0*expm1(p.g1*t)/p.g1)*exp(-p.g1*t))) for (a,b) in zip(grid.edges[1:end-1],grid.edges[2:end])]
    pb=PBMProblem(op,counts(0.),(0.0,1.0);p)
    sol=simulate(pb;dt=0.01,saveat=[0.1,0.4,1.0])
    @test sol.steps==100
    @test maximum(abs,vec(sol.u[end])-counts(1.0))<2e-14
    @test sum(sol.u[end])≈sum(pb.u0) atol=1e-13
    @test minimum(sol.u[end])>=0
    histogram=DistributionObservable(grid.edges;coordinate=:length)
    @test PopulationBalanceModels._observe(histogram,sol.u[end],op,p,1.0)≈vec(sol.u[end]) rtol=1e-12
    @test PopulationBalanceModels._observe(MomentObservable(0),sol.u[end],op,p,1.0)==[sum(sol.u[end])]
    for a in (0.2,0.6,0.9),t in (0.0,0.43)
        foot=(a-p.g0*expm1(p.g1*t)/p.g1)*exp(-p.g1*t)
        reference=foot<0 ? 0.0 : pdf(normal,foot)*exp(-p.g1*t)
        @test characteristic_density(op,a->pdf(normal,a),a,t;p)≈reference atol=1e-12
    end
    @test_throws ArgumentError simulate(pb;dt=0.005,saveat=[0.1])
    @test_throws ArgumentError simulate(pb;dt=0.01,saveat=[0.105])
    @test_throws NumericalFailure simulate(pb;dt=0.01,maxsteps=5,saveat=[0.1])
    @test_throws ArgumentError simulate(remake(pb;p=(g0=0.5,g1=0.3));dt=0.01)
    @test_throws ArgumentError rhs!(similar(pb.u0),pb.u0,op,p,0.0)
    @test_throws ArgumentError compile(PBMModel(Growth((a,p,t,c)->0.4)),grid;evolution=Inguva2022(),bc=open)
    @test_throws ArgumentError compile(PBMModel(Growth(law)+Death((a,p,t,c)->1)),grid;evolution=Inguva2022(),p,bc=open)
    @test_throws ArgumentError compile(PBMModel(Growth(law)),Grid1D(range(0,1.4;length=30));evolution=Inguva2022(),p,bc=open)
    @test_throws ArgumentError compile(PBMModel(Growth(law)),grid;evolution=Inguva2022(),p)
    @test_throws ArgumentError compile(PBMModel(Growth(law)),grid;evolution=Inguva2022(),p,bc=open,methods=TermMethods(growth=Upwind()))
    @test_throws ArgumentError characteristic_grid(AffineGrowth(0.,1.),(0.,1.);dt=0.01)
    @test_throws ArgumentError characteristic_grid(AffineGrowth(1.,-1.),(0.,2.);dt=0.01)
    @test_throws ArgumentError characteristic_grid(law,(0.,1.4);dt=0.01,p,endpoint=:exact)
    # Constant growth and a nonzero lower endpoint have the same public path.
    constant=AffineGrowth(0.5,0.0)
    g=characteristic_grid(constant,(0.1,1.1);dt=0.1,endpoint=:exact)
    periodic=SizeBC(Periodic(),Periodic())
    q=compile(PBMModel(Growth(constant)),g;evolution=Inguva2022(),bc=periodic)
    init=collect(1.0:length(g));s=simulate(PBMProblem(q,init,(0.,0.3));dt=0.1,saveat=[0.3])
    @test vec(last(s.u))==circshift(init,3)
    @test_throws ArgumentError compile(PBMModel(Growth(law)),grid;evolution=Inguva2022(),p,bc=periodic)
end

@testset "Inguva–Braatz 2023: certified commuting growth" begin
    law=SeparableGrowth2D(AffineGrowth(0.1,0.05),AffineGrowth(0.5,0.25))
    grid=characteristic_grid(law,((0.,2.),(0.,2.));dt=0.05,endpoint=:truncate)
    b=SizeBC(OpenBoundary(),OpenBoundary());bc=SizeBC2D(x=b,y=b)
    op=compile(PBMModel(Growth2D(law)),grid;evolution=InguvaBraatz2023(),bc)
    n=Normal(0.4,0.05)
    back(a,t,b,s)=max(0.,a*exp(-s*t)+b*expm1(-s*t)/s)
    axis(g,t,b,s)=[cdf(n,back(hi,t,b,s))-cdf(n,back(lo,t,b,s)) for (lo,hi) in zip(g.edges[1:end-1],g.edges[2:end])]
    reference(t)=axis(grid.x,t,0.1,0.05)*axis(grid.y,t,0.5,0.25)'
    sol=simulate(PBMProblem(op,reference(0.),(0.,1.));dt=0.05,saveat=[1.])
    @test maximum(abs,sol.u[end]-reference(1.))<1e-14
    @test sum(sol.u[end])≈sum(reference(0.)) atol=1e-13
    @test PopulationBalanceModels._observe(MomentObservable((0,0)),sol.u[end],op,(;),1.0)==[sum(sol.u[end])]
    @test_throws ArgumentError PopulationBalanceModels._observe(HistogramObservable([0.,1.,2.]),sol.u[end],op,(;),1.0)
    x=2.4exp(0.05)-2;y=2.4exp(0.25)-2
    @test characteristic_density(op,(a,b)->pdf(n,a)*pdf(n,b),(x,y),1.)≈pdf(n,0.4)^2*exp(-0.3) rtol=1e-13
    @test_throws ArgumentError compile(PBMModel(Growth2D((x,y,p,t,c)->(x+y,1.))),grid;evolution=InguvaBraatz2023(),bc)
    wrong=Grid2D(grid.x,characteristic_grid(law.y,(0.,2.);dt=0.1))
    @test_throws ArgumentError compile(PBMModel(Growth2D(law)),wrong;evolution=InguvaBraatz2023(),bc)
    @test_throws ArgumentError compile(PBMModel(Growth2D(law)),grid;evolution=Inguva2022(),bc)
end
