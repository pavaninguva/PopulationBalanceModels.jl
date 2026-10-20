@testset "Physical term declarations and integrated recipe source" begin
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
    # Physical critical size lies inside a bin but far from its pivot in units
    # of sigma. Midpoint sampling underflows; bin integration must still create J.
    grid=Grid1D([1e-9,3e-9,10e-9,30e-9];coordinate=:diameter)
    for normalization in (:total_rate,:density_amplitude)
        prob=CEJ2025LNP.build(;grid,initial_lipid=12.0,FRR=3.0,tau=0.0,
            temperature=t->298.15,viscosity=(w,T)->1e-3,solubility=(w,T)->1.0,
            equilibrium_ratio=(L,w,T)->1.0,attachment_without_peg=(L,M,w,T)->1.0,
            molecular_volume=5e-28,particle_density=1000.0,nucleation_width=1e-13,
            nucleation_normalization=normalization,tspan=(0.0,0.001),kg=0.0,kPEG=0.0)
        du=similar(prob.u0);rhs!(du,prob.u0,prob.operator,prob.p,0.0)
        rate=CEJ2025LNP.nucleation_rate(3.0,298.15,prob.p.gamma,prob.p.Vm,prob.p.An)
        expected=normalization===:total_rate ? rate : rate*sqrt(2π)*prob.p.nucleation_width
        @test sum(population(prob,du,:lnp))≈expected rtol=1e-12
        @test bulk(prob,du,:lipid)+prob.p.rho*π/6*moment(grid,population(prob,du,:lnp),3)≈0 atol=1e-30
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

include("../examples/inguva_braatz_case1.jl")
@testset "Inguva–Braatz published Gaussian case" begin
    prob=InguvaBraatzCase1.problem(20;boundary=:open)
    @test sum(prob.u0)≈50*π*0.005 rtol=1e-12
    sol=simulate(prob;method=ExactShift(),dt=0.1,saveat=[1.0])
    @test maximum(abs,last(sol.u).-InguvaBraatzCase1.reference(prob.operator.grid,1.0))<1e-14
    @test prob.operator.grid.x.edges[end]==2.0
end

include("../examples/shin_lnp_reproduction.jl")
@testset "LNP observation export conventions" begin
    grid=Grid1D([1e-9,3e-9,10e-9,30e-9];coordinate=:diameter)
    prob=CEJ2025LNP.build(;grid,initial_distribution=[1e10,2e10,0.0],
        initial_lipid=1.0,FRR=3.0,tau=0.0,temperature=t->298.15,
        viscosity=(w,T)->1e-3,solubility=(w,T)->1.0,
        equilibrium_ratio=(L,w,T)->1.0,attachment_without_peg=(L,M,w,T)->1.0,
        molecular_volume=5e-28,particle_density=1000.0,nucleation_width=1e-9,
        nucleation_normalization=:total_rate,tspan=(0.0,0.001),kg=0.0,kPEG=0.0)
    mktempdir() do output
        @test_throws ArgumentError ShinLNPReproduction.run_case(prob;dt=0.001,
            saveat=[0.001],provenance="",output)
        sol=ShinLNPReproduction.run_case(prob;dt=0.001,saveat=[0.001],
            provenance="Synthetic stationary distribution for export verification",output)
        lines=readlines(joinpath(output,"psd.csv"))[2:end]
        intensity=[parse(Float64,split(line,',')[4]) for line in lines]
        @test sum(intensity.*grid.widths)≈1 rtol=1e-14
        @test sol.u[end]≈prob.u0 rtol=1e-14
    end
end
