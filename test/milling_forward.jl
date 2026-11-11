using Test, PopulationBalanceModels, LinearAlgebra
include("../examples/verification/milling_forward.jl")
const MF = MillingForward

@testset "Barrasso forward comparison: inputs, quadrature and observation" begin
    readings = MF.reference_data()
    @test length(readings) == 92
    @test all(count(r -> r.rpm == rpm && r.quantity == quantity, readings) == 23
        for rpm in (3350., 4923.), quantity in ("d50_um", "holdup_g"))
    pb, defect = MF.problem()
    p = pb.p
    uq = MF.MU.problem()
    # Compare the actual assembled operators, including feed and observation,
    # rather than just matching scheme labels or grid sizes.
    for key in (:pivots,:edges,:feed,:selection,:Q,:low0,:low1,:retained)
        @test getproperty(p,key) == getproperty(uq.base.p,key)
    end
    @test first(p.pivots) ≈ 6.54e-17 rtol=2e-15
    @test last(p.pivots) ≈ 1e-3 rtol=2e-14
    @test all(diff(p.edges) .> 0)
    @test defect < 2e-12
    @test p.rho*p.number_scale*dot(p.pivots, population(pb, pb.u0, :mill)) ≈ .0074 rtol=2e-15
    @test p.rho*p.number_scale*dot(p.pivots, p.feed) ≈ .0074 rtol=2e-15

    # Independent analytic interval moments, versus log-coordinate quadrature.
    exact = MF.analytic_daughters(p)
    numeric = pb.operator.system.populations.mill.model.terms[2].daughters
    @test vec(p.pivots' * exact.eta) + exact.low1 ≈ p.pivots rtol=2e-13
    expected_number = MF.MM.relative_moment(0, 1., p.sigma, p.daughter_ratio) /
        MF.MM.relative_moment(1, 1., p.sigma, p.daughter_ratio)
    # Daughter coefficients are counts, of order expected_number per parent.
    @test maximum(sum(abs.(numeric.eta-exact.eta); dims=1)) < 2e-13*expected_number
    @test vec(sum(numeric.eta; dims=1)) + numeric.underflow_number ≈
        fill(expected_number, length(p.pivots)) rtol=2e-13

    @test all(iszero,uq.base.u0)
    # The UQ rebuild path must still agree after changing every uncertain input.
    theta=(delta=.5,alpha=9e-6,gamma=.4,sigma=2.2)
    rebuilt=uq.setup(theta,uq.base)
    shared=MF.MU.forward_problem(theta;rpm=3350.,initial_feed_duration=1.,tf=180.)
    for key in (:pivots,:edges,:feed,:selection,:Q,:low0,:low1,:retained)
        @test getproperty(rebuilt.p,key) == getproperty(shared.p,key)
    end
    N=population(pb,pb.u0,:mill)
    @test MF.MM.outlet_median(p,N,1.) ==
        MF.MM.outlet_median(merge(uq.base.p,(segment_speed=3350.,)),N,1.)

    du = similar(pb.u0)
    rhs!(du, pb.u0, pb.operator, pb.p, 0.)
    uptake = p.rho*p.number_scale*dot(p.pivots, population(pb, du, :mill))
    @test uptake + bulk(pb, du, :discharged_mass) ≈ p.mass_feed rtol=2e-13
    @test du[findall(==(0), p.retained)] == zeros(count(==(0), p.retained))
    sol = MF.MS.solve(pb, [1., 5., 10., 30., 60., 100., 180.])
    @test uq.solver(pb,sol.t).u == sol.u
    check = MF.exponential_check(pb, sol)
    @test check.maximum_d50_difference_um < 1e-3
    @test check.maximum_holdup_difference_g < 1e-3
    @test minimum(minimum.(sol.u)) >= 0
end
