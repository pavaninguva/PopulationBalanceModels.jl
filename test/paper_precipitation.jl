# Run with julia --project=paper test/paper_precipitation.jl.
# Small independent checks only; no production simulations are launched.
using Test, PopulationBalanceModels, TOML, ForwardDiff, LinearAlgebra
include(joinpath(@__DIR__, "..", "paper", "precipitation", "run.jl"))
const P = PaperPrecipitation
const C = P.ContinuousPrecipitation

@testset "Logarithmic tubular precipitation contracts" begin
    cfg = TOML.parsefile(joinpath(@__DIR__, "..", "paper", "precipitation", "config.toml"))["production"]
    jobs = P.configurations(cfg)
    @test Set((j.nx,j.nz) for j in jobs if j.axis=="size") ==
        Set((n,500) for n in (50,100,200,400))
    @test Set((j.nx,j.nz) for j in jobs if j.axis=="axial") ==
        Set((1000,n) for n in (50,100,200,400))
    @test all(j -> (j.nx,j.nz)==(1000,500), filter(j->j.axis in ("reference","time"),jobs))

    a = C.build(nx=6,nz=4,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=.01)
    b = C.build(nx=8,nz=6,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=.01)
    ga, gb = P.grid(a), P.grid(b)
    u, v = copy(a.u0), copy(b.u0)
    population(a,u,:crystals) .= 2 .* ga.widths
    population(b,v,:crystals) .= 3 .* gb.widths
    bulk(a,u,:c) .= 2.
    bulk(b,v,:c) .= 3.
    # Independent exact integrals of constant densities on unequal partitions.
    @test P.field_error(a,u,b,v) ≈ 1/3 rtol=2e-14
    @test C.outlet_error(a,u,b,v) ≈ 1/3 rtol=2e-14
    @test P.bulk_error(a,u,b,v).normalized_bulk_L1 ≈ 1/3 rtol=2e-14
    @test P.bulk_error(a,u,b,v).maximum_bulk_difference_kg_m3 == 1.
    @test P.field_error(b,v,b,v) == 0.

    # A single boundary source removes the represented first-bin material,
    # not an independently evaluated continuum nucleus mass.
    localop = first(a.operator.local_operators).system.populations.crystals
    state = zeros(length(ga),1)
    derivative = zero(state)
    p = a.p
    context = (;bulk=(c=2p.ceq,), fields=NamedTuple(), populations=NamedTuple())
    rhs!(derivative,state,localop,p,0.;context)
    expected_births = p.kp / 1e12  # s=1, no existing solids
    @test sum(derivative) ≈ expected_births rtol=2e-14
    @test derivative[1] ≈ expected_births rtol=2e-14
    @test all(iszero,derivative[2:end])
    @test localop.prepared[1].cache isa PopulationBalanceModels._NonuniformWENO5

    state = copy(a.u0)
    bulk(a,state,:c) .= 2p.ceq
    du = zero(state)
    rhs!(du,state,a.operator,p,0.)
    Ndot, cdot = population(a,du,:crystals), bulk(a,du,:c)
    faces = axial_fluxes(a.operator,state,p,0.)
    weights = vcat(1e12p.rho*p.kv .* ga.centers.^3,1.)
    expected = sum(weights .* (faces[:,1]-faces[:,end]))
    actual = sum(weights .* du .* a.operator.domain.grid.widths')
    @test actual ≈ expected rtol=2e-13 atol=1e-15
    # Axial population transport is zero for this empty population. The local
    # solute uptake follows the very same first-bin source.
    axial_c = (faces[end,1:end-1]-faces[end,2:end]) ./ a.operator.domain.grid.widths
    @test cdot-axial_c ≈ -1e12p.rho*p.kv*ga.centers[1]^3 .* vec(sum(Ndot;dims=1)) rtol=2e-12 atol=1e-15
end

@testset "Same-stage derivative exchange equals generic conservative exchange" begin
    for scheme in (Upwind(),WENO5())
        ordinary = C.build(;nx=16,nz=6,mesh=:log,scheme,axial_scheme=MUSCL(),tf=1.)
        optimized = P.build(;nx=16,nz=6,mesh=:log,scheme,axial_scheme=MUSCL(),tf=1.)
        axis = P.grid(ordinary)
        for scenario in 1:3
            state = copy(ordinary.u0)
            if scenario>1
                bulk(ordinary,state,:c) .= range(.2,9.;length=6)
            end
            if scenario==3
                density = 500exp.(-((log.(axis.centers).-log(10e-6))./.4).^2)
                population(ordinary,state,:crystals) .= axis.widths.*density .* (1 .+ .1cos.(1:6))'
            end
            expected, actual = zero(state), zero(state)
            rhs!(expected,state,ordinary.operator,ordinary.p,.3)
            rhs!(actual,state,optimized.operator,optimized.p,.3)
            @test actual ≈ expected rtol=5e-14 atol=1e-14*max(1.,maximum(abs,expected))
        end
    end
end

@testset "Conservative preconditioner versus independent dense Jacobians" begin
    # Differentiate the complete native system, independently of the colored
    # growth Jacobian and rank-one formula used by the preconditioner.
    for scheme in (Upwind(),WENO5())
    pb = P.build(;nx=8,nz=3,mesh=:log,scheme,axial_scheme=Upwind(),tf=1.)
    state = copy(pb.u0)
    population(pb,state,:crystals) .= [1e-6exp(-i/2) for i in 1:8]
    bulk(pb,state,:c) .= [6.,7.,8.]
    n, cells = size(state)
    local_jacobian = zeros(length(state),length(state))
    for j in 1:cells
        function local_rhs(x)
            result = similar(x)
            rhs!(result,x,pb.operator.local_operators[j],pb.p,0.)
            result
        end
        block = (j-1)*n+1:j*n
        local_jacobian[block,block] .= ForwardDiff.jacobian(local_rhs,state[:,j])
    end
    function axial_rhs(x)
        values = reshape(x,size(state))
        faces = axial_fluxes(pb.operator,values,pb.p,0.)
        vec((faces[:,1:end-1]-faces[:,2:end])./pb.operator.domain.grid.widths')
    end
    axial_jacobian = ForwardDiff.jacobian(axial_rhs,vec(state))
    right = reshape(sin.(1:length(state)),size(state))
    hooks = P.S.preconditioner(pb)
    for gamma in (1e-4,.01,.1)
        jcur = [0]
        hooks.setup(pb.p,0.,state,zero(state),false,jcur,gamma)
        @test jcur[1] == 1
        expected = (I-gamma*axial_jacobian) \ ((I-gamma*local_jacobian) \ vec(right))
        actual = zero(state)
        hooks.solve(actual,right,pb.p,0.,state,zero(state),gamma,0.,1)
        @test vec(actual) ≈ expected rtol=2e-10 atol=2e-11
    end
    updates = hooks.updates[]
    jcur = [1]
    hooks.setup(pb.p,0.,state,zero(state),true,jcur,.12)
    @test hooks.updates[] == updates
    @test jcur[1] == 0
    # Between setup calls the whole preconditioner uses its cached gamma,
    # rather than combining new coupling coefficients with old factors.
    actual = zero(state)
    hooks.solve(actual,right,pb.p,0.,state,zero(state),.13,0.,1)
    expected = (I-.12axial_jacobian) \ ((I-.12local_jacobian) \ vec(right))
    @test vec(actual) ≈ expected rtol=2e-10 atol=2e-11
    # Below saturation the local Jacobian is exactly zero. Large artificial
    # Krylov population directions must not erase a tiny bulk direction through
    # cancellation of two separately summed material inventories.
    bulk(pb,state,:c) .= 0.
    tiny = copy(right)
    tiny[end,:] .= 1e-12
    actual = zero(state)
    gamma = .01
    hooks.setup(pb.p,0.,state,zero(state),false,[0],gamma)
    hooks.solve(actual,tiny,pb.p,0.,state,zero(state),gamma,0.,1)
    expected = reshape((I-gamma*axial_jacobian) \ vec(tiny),size(state))
    @test bulk(pb,actual,:c) ≈ bulk(pb,expected,:c) rtol=2e-13 atol=1e-25
    end
end
