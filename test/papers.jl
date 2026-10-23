using ForwardDiff, LinearAlgebra
include("../examples/papers/gunawan2004_case2.jl")
include("../examples/papers/cej2025_lnp.jl")

@testset "Paper model recipes" begin
    @test Gunawan2004Case2.temperature(0.0)==32.0
    @test Gunawan2004Case2.solubility(32.0)≈0.1166704
    Lmax=0.136/(2*3.48e-4)
    @test Gunawan2004Case2.reported_seed_density(Lmax)<0
    badgrid=Grid1D(vcat(0.0,collect(180.0:2:212.0));coordinate=:length)
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

@testset "DLVO reconstruction and LNP mechanism accounting" begin
    kw=(temperature=298.15,ionic_strength=75.,relative_permittivity=65.)
    @test CEJ2025LNP.dlvo_attachment(25e-9,75e-9;kw...,potential=0.,hamaker=0.)≈1 atol=1e-14
    a=CEJ2025LNP.dlvo_attachment(25e-9,75e-9;kw...,order=128)
    b=CEJ2025LNP.dlvo_attachment(75e-9,25e-9;kw...,order=128)
    refined=CEJ2025LNP.dlvo_attachment(25e-9,75e-9;kw...,order=256)
    @test a≈b rtol=1e-13
    @test a≈refined rtol=0.005
    @test CEJ2025LNP.debye_wavenumber(100.,298.15,78.4)≈10CEJ2025LNP.debye_wavenumber(1.,298.15,78.4)
    @test_throws ArgumentError CEJ2025LNP.dlvo_attachment(0.,1.;kw...)
    g=Grid1D(exp.(range(log(1e-9),log(2e-6);length=21));coordinate=:diameter)
    N=zeros(length(g));N[10]=1e15;N[end]=1e10
    prob=CEJ2025LNP.build(;grid=g,initial_distribution=N,initial_lipid=10.,FRR=3.,tau=0.,
        temperature=t->298.15,viscosity=(w,T)->1e-3,solubility=(w,T)->.01,
        equilibrium_ratio=(L,w,T)->1.,attachment_without_peg=(L,M,w,T)->1.,
        molecular_volume=5e-28,particle_density=1000.,nucleation_width=1e-9,
        nucleation_normalization=:total_rate,tspan=(0.,.02),kg=.05,overflow=:discard,track_consumption=true)
    d=similar(prob.u0);rhs!(d,prob.u0,prob.operator,prob.p,0.)
    @test bulk(prob,d,:lipid)+bulk(prob,d,:nucleated_lipid)+bulk(prob,d,:grown_lipid)≈0 atol=1e-10
    particle_mass_rate=prob.p.rho*π/6*moment(g,population(prob,d,:lnp),3)
    @test particle_mass_rate+bulk(prob,d,:lipid)+bulk(prob,d,:escaped_lipid)≈0 atol=1e-10
    @test bulk(prob,d,:escaped_lipid)>0
    @test bulk(prob,d,:nucleated_lipid)>0
end

include("../examples/continuous_precipitation.jl")
include("../examples/protein_crystallization.jl")
@testset "Paper case studies" begin
    p=ContinuousPrecipitation.build(nx=8,nz=8,tf=.5)
    s=ContinuousPrecipitation.advance(p,[.5]);@test minimum(last(s.u))>=-1e-8
    @test p.operator.domain.velocity≈1.67e-7/(1.46e-4*.21)
    pb=ProteinCrystallization.build(bins=40,tf=20.)
    sol=simulate(pb;dt=.5)
    @test last(sol.u)[end]<pb.u0[end]
    @test all(iszero,population(pb,pb.u0,:crystals))
    data=ProteinCrystallization.observations()
    @test length(data.concentration.t)==28 && length(data.D50.t)==4
    # L³-weighted median of a constant number density on [0,2] is 2/2^(1/4).
    g=Grid1D([0.,1.,2.]);op=compile(PBMModel(),g)
    base=PBMProblem(PBMSystem((crystals=op,)),(crystals=[1.,1.],),(0.,1.))
    @test ProteinCrystallization.volume_median(base.u0,base.operator,(;),0.)≈2/2^(1/4)
end

@testset "Protein prior coordinates and scheduled forward accuracy" begin
    recipe=ProteinCrystallization
    for (scenario,sd,jdensity,half_density) in ((:broad,3.,1/20,1/20),(:previous,1.5,1/9,0.))
        problem=recipe.estimation_problem(bins=40,priors=scenario)
        spec=only(filter(s->s.name===:AG,problem.parameters))
        for x in (.05,.367,2.)
            # Independent physical lognormal density, not the implementation Jacobian.
            expected=exp(-.5*((log(x)-log(.367))/sd)^2)/(sd*sqrt(2pi)*x)
            @test recipe.physical_prior_density(spec,x)≈expected
        end
        spec=only(filter(s->s.name===:j,problem.parameters))
        @test recipe.physical_prior_density(spec,2.)≈jdensity
        @test recipe.physical_prior_density(spec,.5)≈half_density
    end
    @test_throws ArgumentError recipe.estimation_problem(priors=:unknown)
    mktempdir() do directory
        write(joinpath(directory,"map_parameters.csv"),
            "parameter,inference_coordinate,physical_value\nlogAJ,999,18.9\nj,999,2.14\nAG,999,.367\ng,999,2.45\n")
        for scenario in (:broad,:previous)
            problem=recipe.estimation_problem(bins=8,priors=scenario)
            z=recipe.saved_map(directory,problem)
            # The deliberately incorrect old coordinates must never be reused.
            @test all(isapprox(getproperty(physical_parameters(problem,z,:batch),name),value) for (name,value) in pairs((logAJ=18.9,j=2.14,AG=.367,g=2.45)))
        end
    end
    pb=recipe.build(bins=40,tf=80.)
    scheduled=recipe.scheduled_solver(pb,[0.,20.,50.,80.])
    reference=simulate(pb;dt=.1,saveat=scheduled.t)
    @test scheduled.t==[0.,20.,50.,80.]
    @test maximum(abs(bulk(pb,a,:c)-bulk(pb,b,:c)) for (a,b) in zip(scheduled.u,reference.u))<.01
    @test all(minimum(u)>=0 for u in scheduled.u)
end

@testset "Protein reference-rate sampling coordinates" begin
    p=ProteinCrystallization.estimation_problem(bins=40)
    map=ProteinCrystallization.reference_sampling_map(p)
    z=initial_coordinates(p);y=map.to_sampler(z)
    @test map.from_sampler(y)≈z atol=1e-12
    @test det(ForwardDiff.jacobian(map.from_sampler,y))≈1. atol=1e-12
    @test map.logabsdetjac(y)==0
    @test logposterior(p,z)≈logposterior(p,map.from_sampler(y)) atol=1e-10
end

@testset "Tubular refinement paths and RK4 reference controller" begin
    recipe=ContinuousPrecipitation
    configurations=recipe.benchmark_configurations()
    for name in ("upwind","muscl","weno5")
        rows=filter(c->c.mesh===:uniform && c.size_method==name,configurations)
        @test sort([c.nx for c in rows if c.nz==96])==[12,24,48,72]
        @test sort([c.nz for c in rows if c.nx==72])==[16,32,64,96]
    end
    # No precipitation: independent augmented matrix exponential for the linear
    # conservative axial Upwind/dispersion system with Danckwerts feed.
    p=merge(recipe.parameters(),(kp=0.,kb=0.,kg=0.))
    problem=recipe.build(nx=8,nz=4,p=p,axial_scheme=Upwind(),tf=.1)
    dz=p.L/4;v=p.Q/(p.area*p.porosity);d=p.Dax/dz^2
    A=zeros(5,5)
    for j in 1:4
        A[j,j]=-v/dz-d*((j>1)+(j<4))
        j>1 && (A[j,j-1]=v/dz+d)
        j<4 && (A[j,j+1]=d)
    end
    A[1,5]=v/dz*p.cfeed
    exact=(exp(.1A)*[0.,0.,0.,0.,1.])[1:4]
    solution=recipe.advance(problem,[0.,.1];method=RK4())
    @test bulk(problem,last(solution.u),:c)≈exact rtol=1e-8 atol=1e-12
    @test all(iszero,population(problem,last(solution.u),:crystals))
    @test_throws ArgumentError recipe.advance(problem,[.1,0.])
    @test_throws ArgumentError recipe.advance(problem,[.1];fraction=0.)
end
