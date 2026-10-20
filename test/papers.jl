include("../examples/papers/gunawan2004_case2.jl")
include("../examples/papers/cej2025_lnp.jl")

@testset "Paper model recipes" begin
    @test Gunawan2004Case2.temperature(0.0)==32.0
    @test Gunawan2004Case2.solubility(32.0)≈0.1166704
    Lmax=0.136/(2*3.48e-4)
    @test Gunawan2004Case2.reported_seed_density(Lmax)<0
    badgrid=Grid1D(180.0:2:212.0;coordinate=:length)
    @test_throws ArgumentError Gunawan2004Case2.build(;grid=badgrid,
        initial_density=Gunawan2004Case2.reported_seed_density,c0=0.2)
    seed=L->180.5<=L<=210.5 ? 0.08*(1-((L-195.5)/15)^2) : 0.0
    grid=Grid1D(0.0:2:1600.0;coordinate=:length)
    problem=Gunawan2004Case2.build(;grid,initial_density=seed,c0=Gunawan2004Case2.solubility(32.0),tspan=(0.0,10.0))
    du=similar(problem.u0)
    rhs!(du,problem.u0,problem.operator,problem.p,1.0)
    @test bulk(problem,du,:c)+problem.p.rho*moment(grid,population(problem,du,:crystals),3)≈0 atol=1e-20
    moments=[moment(grid,population(problem,problem.u0,:crystals),k) for k in 0:3]
    ref=vcat(moments,bulk(problem,problem.u0,:c)); dref=similar(ref)
    Gunawan2004Case2.moment_rhs!(dref,ref,problem.p,1.0)
    @test dref[5]+problem.p.rho*dref[4]≈0 atol=1e-20
    @test dref[1]>0
    @test CEJ2025LNP.aqueous_fraction(0.0,3.0,0.01)==0.0
    @test CEJ2025LNP.aqueous_fraction(1.0,3.0,0.001)≈0.75
    @test CEJ2025LNP.aqueous_fraction(0.0,3.0,0.0)≈0.75
    @test CEJ2025LNP.nucleation_rate(1.0,298.15,0.01,5e-28,2e22)==0
    @test isinf(CEJ2025LNP.critical_length(1.0,298.15,0.01,5e-28))
    @test CEJ2025LNP.brownian_kernel(10e-9,10e-9,298.15,1e-3)≈8*CEJ2025LNP.kB*298.15/(3e-3)
    @test CEJ2025LNP.loglinear_mole_solubility(1e-8,1e-3,0.5)≈sqrt(1e-11)
    g=Grid1D(exp.(range(log(1e-9),log(2e-6);length=41));coordinate=:diameter)
    N=zeros(length(g));N[18]=1e14
    cej=CEJ2025LNP.build(;grid=g,initial_distribution=N,initial_lipid=10.0,FRR=3.0,tau=0.01,
        temperature=t->298.15,viscosity=(w,T)->1e-3,solubility=(w,T)->10exp(-8w),
        equilibrium_ratio=(L,w,T)->1.0,attachment_without_peg=(L,M,w,T)->1.0,
        molecular_volume=5e-28,particle_density=1000.0,nucleation_width=1e-9,
        nucleation_normalization=:total_rate,tspan=(0.0,0.02),kg=0.05)
    derivative=similar(cej.u0);time=0.01
    rhs!(derivative,cej.u0,cej.operator,cej.p,time)
    fields=cej.operator.system.fields(cej.p,time)
    total=bulk(cej,cej.u0,:lipid)+cej.p.rho*π/6*moment(g,population(cej,cej.u0,:lnp),3)
    change=bulk(cej,derivative,:lipid)+cej.p.rho*π/6*moment(g,population(cej,derivative,:lnp),3)
    @test change≈-fields.dilution*total rtol=1e-12
    @test all(isfinite,derivative)
    @test CEJ2025LNP.paper_intensity_mean(g,N)≈g.centers[18]
end
