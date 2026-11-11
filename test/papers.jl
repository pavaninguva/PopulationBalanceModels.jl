include("fixtures/example_modules.jl")
using ForwardDiff, LinearAlgebra
_load_example(:Gunawan2004Case2, "papers/gunawan2004_case2.jl")

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
end

_load_example(:ContinuousPrecipitation, "continuous_precipitation.jl")
_load_example(:ProteinCrystallization, "protein_crystallization.jl")
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
