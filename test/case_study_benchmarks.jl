# Short checks for the exported comparison protocols; no production benchmark runs.
using Test, PopulationBalanceModels
include("../examples/constant_growth_methods.jl")
include("../examples/continuous_precipitation.jl")

@testset "Benchmark configurations and independent outlet integral" begin
    quick = ContinuousPrecipitation.benchmark_configurations()
    full = ContinuousPrecipitation.benchmark_configurations(full=true)
    @test length(quick) == 25
    @test length(full) == 125
    for mesh in (:log,:uniform)
        pairs = [(c.nx,c.nz) for c in full if c.mesh==mesh && c.size_method=="muscl" && c.axial_method=="muscl"]
        @test Set(pairs) == Set(Iterators.product((12,25,50,100,200),(37,75,150,300,600)))
    end
    @test all(c.mesh==:uniform for c in quick if c.size_method=="weno5")
    a = ContinuousPrecipitation.build(nx=6,nz=6,mesh=:log)
    b = ContinuousPrecipitation.build(nx=8,nz=6,mesh=:uniform)
    ga = first(a.operator.local_operators).system.populations.crystals.grid
    gb = first(b.operator.local_operators).system.populations.crystals.grid
    u,v = copy(a.u0),copy(b.u0)
    # Constant densities 2 and 3 on a common domain: exact normalized L1=1/3,
    # regardless of either mesh or the number of partitions in their union.
    population(a,u,:crystals)[:,end] .= 2 .* ga.widths
    population(b,v,:crystals)[:,end] .= 3 .* gb.widths
    @test ContinuousPrecipitation.outlet_error(a,u,b,v) ≈ 1/3 rtol=1e-14
    @test ContinuousPrecipitation.outlet_error(b,v,b,v) == 0
    @test_throws NumericalFailure ContinuousPrecipitation.outlet_error(b,v,b,b.u0)
    # Open-reactor material changes equal inlet minus outlet flux. Local crystal
    # uptake cancels against solute even with this example's 1e12 count scaling.
    state = copy(b.u0)
    bulk(b,state,:c) .= range(.5,1.;length=6)
    population(b,state,:crystals) .= gb.widths .* ones(1,6)
    derivative = similar(state)
    rhs!(derivative,state,b.operator,b.p,0.)
    faces = axial_fluxes(b.operator,state,b.p,0.)
    property = vcat(1e12*b.p.rho*b.p.kv .* gb.centers.^3,1.)
    inventory_rate = sum(property .* derivative .* b.operator.domain.grid.widths')
    boundary_rate = sum(property .* (faces[:,1] .- faces[:,end]))
    @test inventory_rate ≈ boundary_rate rtol=1e-12 atol=1e-14
    @test_throws ArgumentError ContinuousPrecipitation.build(nx=8,nz=6,mesh=:log,scheme=WENO5())
    sol = ContinuousPrecipitation.advance(b,[.01])
    @test minimum(last(sol.u)) >= -1e-8

    choice = only(filter(c->c[1]=="weno3",ConstantGrowthMethods.combinations()))
    @test choice[3] isa WENO3 && choice[4] isa SSPRK33
    grid = Grid1D(range(0.,1.;length=81))
    N = ConstantGrowthMethods.smooth_counts(grid,0.)
    truth = ConstantGrowthMethods.smooth_counts(grid,4.)
    result = ConstantGrowthMethods.measure(grid,N,truth,choice[3],choice[4],4.,.0625,SizeBC(Periodic(),Periodic()))
    @test abs(result.number_drift) < 1e-12
    @test 0 < result.rmse < .01
    hoc = only(filter(c->c[1]=="hoc",ConstantGrowthMethods.combinations()))
    @test hoc[3] === nothing && hoc[4] isa HOCFiniteVolume
    coarse = ConstantGrowthMethods.measure(grid,N,truth,hoc[3],hoc[4],4.,.0625,SizeBC(Periodic(),Periodic()))
    fine_grid = Grid1D(range(0.,1.;length=161))
    fine = ConstantGrowthMethods.measure(fine_grid,ConstantGrowthMethods.smooth_counts(fine_grid,0.),
        ConstantGrowthMethods.smooth_counts(fine_grid,4.),hoc[3],hoc[4],4.,.03125,SizeBC(Periodic(),Periodic()))
    @test abs(coarse.number_drift) < 1e-12
    @test 3.8 < coarse.rmse/fine.rmse < 4.2 # independent integrated sine/cosine reference
    @test all(c.axial_method=="muscl" for c in vcat(quick,full))
end
