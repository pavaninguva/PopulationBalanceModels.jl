include("fixtures/example_modules.jl")
@testset "Physical term declarations and integrated source balance" begin
    g=Grid1D(0.0:0.1:2.0)
    G=(x,p,t,ctx)->0.2
    J=(p,t,ctx)->1.0
    positional=PBMModel(Growth(G)+Nucleation(J,0.4))
    named=PBMModel(growth=Growth(rate=G),primary=Nucleation(rate=J,size=0.4))
    a=compile(positional,g);b=compile(named,g)
    N=reshape(copy(g.widths),:,1);da=similar(N);db=similar(N)
    rhs!(da,N,a,(;),0.0);rhs!(db,N,b,(;),0.0)
    @test da==db
    a=compile(PBMModel(GaussianNucleation(J;center=0.4,width=0.01)),g)
    b=compile(PBMModel(GaussianNucleation(rate=J,center=0.4,width=0.01)),g)
    rhs!(da,N,a,(;),0.0);rhs!(db,N,b,(;),0.0)
    @test da==db
    # Source centre lies inside a bin but far from its pivot in units
    # of sigma. Midpoint sampling underflows; bin integration must still create J.
    grid=Grid1D([1e-9,3e-9,10e-9,30e-9];coordinate=:diameter)
    p=(rate=3.,center=2.1e-9,width=1e-13,rho=1000.)
    rate=(p,t,ctx)->p.rate
    for normalization in (:total_rate,:density_amplitude)
        gaussian=GaussianNucleation(rate;center=p.center,width=p.width)
        physical=GaussianNucleation(rate;center=p.center,width=p.width,truncation=:physical)
        source=normalization===:total_rate ? gaussian : IntegratedNucleation(rate,
            (lo,hi,p,t,ctx)->sqrt(2π)*p.width*physical.fraction(lo,hi,p,t,ctx);
            normalization=:density_amplitude)
        op=compile(PBMModel(nucleation=source),grid)
        exchange=ConservativeExchange(:particles,:solute,(L,p,t,ctx)->p.rho*π/6*L^3;
            mechanisms=(:nucleation,))
        system=PBMSystem((particles=op,);exchanges=(exchange,),nonnegative_bulk=(:solute,))
        prob=PBMProblem(system,(particles=zeros(length(grid)),),(0.,0.001);bulk0=(solute=1.,),p)
        du=similar(prob.u0);rhs!(du,prob.u0,prob.operator,prob.p,0.0)
        expected=normalization===:total_rate ? p.rate : p.rate*sqrt(2π)*p.width
        @test sum(population(prob,du,:particles))≈expected rtol=1e-12
        @test bulk(prob,du,:solute)+p.rho*π/6*moment(grid,population(prob,du,:particles),3)≈0 atol=1e-30
    end
end

@testset "Compact ADI: independent dense update and norm stability" begin
    using LinearAlgebra
    n=12; h=1/n; grid=Grid2D(range(0,1;length=n+1),range(0,1;length=n+1))
    periodic=SizeBC(Periodic(),Periodic()); speeds=(0.7,-0.3)
    op=compile(PBMModel(Growth2D(speeds)),grid;bc=SizeBC2D(x=periodic,y=periodic))
    f=[1+0.3cos(2π*(i+2j)/n)+0.1sin(2π*(3i-j)/n) for i in 1:n,j in 1:n]
    area=grid.x.widths*grid.y.widths';prob=PBMProblem(op,f.*area,(0.0,0.1))
    M=Matrix{Float64}(I,n,n);D=zeros(n,n)
    for i in 1:n
        M[i,mod1(i-1,n)]=0.25;M[i,mod1(i+1,n)]=0.25
        D[i,mod1(i-1,n)]=-3/(4h);D[i,mod1(i+1,n)]=3/(4h)
    end
    gamma=1/(2-2^(1/3))
    for order in (2,4)
        truth=copy(f)
        for c in (order==2 ? (1.0,) : (gamma,1-2gamma,gamma))
            qx=(M+0.05c*speeds[1]*D)\(M-0.05c*speeds[1]*D)
            qy=(M+0.05c*speeds[2]*D)\(M-0.05c*speeds[2]*D)
            truth=qx*truth*qy'
        end
        sol=simulate(prob;method=HOCADI(time_order=order),dt=0.1,check_nonnegative=false)
        @test density(grid,last(sol.u))≈truth rtol=1e-13 atol=1e-13
        @test norm(truth)≈norm(f) rtol=1e-13
    end
    # Pure Fourier modes, rather than positive offset states alone, expose any
    # amplification error in the highest represented frequencies.
    for k in (1,5),cfl in (0.25,4.0)
        initial=[cos(2π*k*i/n)*sin(2π*j/n) for i in 1:n,j in 1:n]
        # PBMProblem correctly rejects signed initial numbers; add a constant
        # and remove the conserved mean when measuring fluctuation norm.
        p=PBMProblem(op,(2 .+ initial).*area,(0.0,2.0))
        sol=simulate(p;method=HOCADI(),dt=cfl*h,check_nonnegative=false)
        final=density(grid,last(sol.u));final .-= sum(final)/length(final)
        @test norm(final)≈norm(initial) rtol=2e-12
    end
end

include("fixtures/constant_growth_gaussian.jl")
@testset "Inguva–Braatz published Gaussian case" begin
    prob=InguvaBraatzCase1.problem(20;boundary=:open)
    @test sum(prob.u0)≈50*π*0.005 rtol=1e-12
    sol=simulate(prob;method=ExactShift(),dt=0.1,saveat=[1.0])
    @test maximum(abs,last(sol.u).-InguvaBraatzCase1.reference(prob.operator.grid,1.0))<1e-14
    @test prob.operator.grid.x.edges[end]==2.0
end

_load_example(:InguvaBraatzCase2, "inguva_braatz_case2.jl")
@testset "Inguva–Braatz Case 2: affine growth and mapped bin numbers" begin
    case=InguvaBraatzCase2
    prob=case.problem(20)
    @test sum(prob.u0)≈50π*0.005 rtol=1e-13
    @test case.analytical_density(0.4,0.4,0.0)==50.0
    x=2.4exp(0.05)-2;y=2.4exp(0.25)-2
    @test case.analytical_density(x,y,1.0)≈50exp(-0.3) rtol=1e-13
    @test sum(case.reference(prob.operator.grid,1.0))≈sum(prob.u0) rtol=1e-13
    mapped=case.mapped_problem(20)
    sol=simulate(mapped.problem;dt=mapped.dt,saveat=[1.0])
    @test sum(abs,last(sol.u).-case.reference(mapped.physical_grid,1.0))<2e-13
    @test sum(last(sol.u))≈sum(mapped.problem.u0) rtol=1e-13
    @test_throws ArgumentError simulate(prob;method=ExactShift(),dt=0.05)
end

@testset "Shared growth parameters and physical error measures" begin
    study=InguvaBraatzCase2
    p=study.parameters(offset=(0.15,0.45),slope=(0.1,0.2))
    prob=study.problem(20;p)
    @test collect(only(prob.operator.model.terms).rate(0.4,0.4,prob.p,0.0,nothing))≈[0.19,0.53]
    x=(p.center[1]+p.offset[1]/p.slope[1])*exp(p.slope[1])-p.offset[1]/p.slope[1]
    y=(p.center[2]+p.offset[2]/p.slope[2])*exp(p.slope[2])-p.offset[2]/p.slope[2]
    @test study.analytical_density(x,y,1.0,p)≈p.amplitude*exp(-sum(p.slope)) rtol=1e-13
    # A parameter change must affect both the mapped transport and reference.
    for slopes in ((0.1,0.2),(0.0,0.0))
        q=merge(p,(slope=slopes,));mapped=study.mapped_problem(20;p=q)
        sol=simulate(mapped.problem;dt=mapped.dt,saveat=[1.0])
        @test sum(abs,last(sol.u).-study.reference(mapped.physical_grid,1.0,q))<3e-13
    end
    # Subdivide only the low-error region of the same piecewise constant field.
    # An area norm must stay fixed even though the unweighted bin mean changes.
    norms=map(([0.0,0.5,1.0],[0.0,0.125,0.25,0.5,1.0])) do edges
        g=Grid2D(edges,[0.0,0.5,1.0]);area=g.x.widths*g.y.widths'
        f=[x<0.5 ? 1.0 : 3.0 for x in g.x.centers,y in g.y.centers]
        study.metrics(f.*area,zeros(size(g)),g)
    end
    @test norms[1].RMS_area≈sqrt(5)
    @test norms[2].RMS_area≈norms[1].RMS_area
    @test norms[1].RMSE!=norms[2].RMSE

    recipe=Gunawan2004Case2
    p=recipe.parameters(kg=20.0,size_factor=0.2)
    mu=[1.0,200.0,40000.0,8e6];c0=1.05*recipe.solubility(32.0)
    prob=recipe.build_moments(;initial_moments=mu,c0,p)
    du=similar(prob.u0);rhs!(du,prob.u0,prob.operator,p,0.0)
    reference=similar(prob.u0);recipe.moment_rhs!(reference,vcat(mu,c0),p,0.0)
    @test du≈reference rtol=1e-13
end

@testset "Evolution input contracts and two-coordinate rate errors" begin
    g=Grid2D(0.0:0.1:1.0,0.0:0.1:1.0)
    for bad in ((1.0,), (1.0,2.0,3.0), (NaN,1.0), ("a",1.0))
        @test_throws ArgumentError compile(PBMModel(Growth2D(rate=bad)),g)
        op=compile(PBMModel(Growth2D(rate=(x,y,p,t,ctx)->bad)),g)
        @test_throws ArgumentError rhs!(zeros(size(g)),ones(size(g)),op,(;),0.0)
    end
    periodic=SizeBC(Periodic(),Periodic())
    op=compile(PBMModel(Growth2D(rate=(1.0,1.0))),g;bc=SizeBC2D(x=periodic,y=periodic))
    prob=PBMProblem(op,ones(size(g)),(0.0,1.0))
    state_dependent=compile(PBMModel(Growth2D(rate=(x,y,p,t,ctx)->(moment(ctx,(0,0)),0.0))),g;
        bc=SizeBC2D(x=periodic,y=periodic))
    @test suggest_dt(state_dependent,ones(size(g)),(;),0.0)≈0.0005
    for method in (SSPRK33(),ExactShift(),HOCADI())
        @test_throws ArgumentError simulate(prob;method,dt=0.1,saveat=[0.0],maxsteps=0)
        @test_throws ArgumentError simulate(prob;method,dt=0.1,maxsteps=1.5)
        @test_throws ArgumentError simulate(prob;method,dt=0.1,saveat=[0.2,0.1])
    end
end

_load_example(:CoolingCrystallizationBenchmark, "cooling_crystallization_benchmark.jl")
@testset "Cooling initial conditions and numerical reference" begin
    @test CoolingCrystallizationBenchmark.Gunawan2004Case2.reported_seed_density(195.5)≈-0.012647 atol=1e-14
    @test_throws ArgumentError CoolingCrystallizationBenchmark.run_benchmark(;
        initial_density=CoolingCrystallizationBenchmark.Gunawan2004Case2.reported_initial_density,
        c0=0.1166704,provenance="Literal printed initial condition")
    recipe=CoolingCrystallizationBenchmark.Gunawan2004Case2
    @test recipe.reported_initial_density(195.5)==recipe.reported_seed_density(195.5)
    @test recipe.benchmark_initial_density(195.5)≈0.08
    @test minimum(recipe.benchmark_initial_density.(range(180.5,210.5;length=101)))>=0
    mu=CoolingCrystallizationBenchmark.continuum_moments(recipe.benchmark_initial_density)
    @test mu[1]≈1.6 atol=1e-13
    @test mu[2]/mu[1]≈195.5 atol=1e-12
    fine=Grid1D([0.0,0.2,0.4,0.6,0.8,1.0]);coarse=Grid1D([0.0,0.3,0.7,1.0])
    projected=CoolingCrystallizationBenchmark.rebin(fine,2fine.widths,coarse)
    @test projected≈2coarse.widths
    @test sum(projected)≈2
    mktempdir() do output
        result=CoolingCrystallizationBenchmark.run_benchmark(;grids=(20,),reference_bins=40,
            check_bins=80,dt=0.1,tf=1.0,output)
        @test result.spatial_reference_change>=0
        @test result.temporal_reference_change<1e-8
        @test isfile(joinpath(output,"errors.csv"))
    end
end
