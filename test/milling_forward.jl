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
    @test p.pivots ≈ 6.54e-17 .* (1:100).^6 rtol=2e-15
    @test all(diff(p.edges) .> 0)
    @test defect < 2e-12
    @test p.rho*p.number_scale*dot(p.pivots, population(pb, pb.u0, :mill)) ≈ .0074 rtol=2e-15
    @test p.rho*p.number_scale*dot(p.pivots, p.feed) ≈ .0074 rtol=2e-15

    # Independent analytic interval moments, versus log-coordinate quadrature.
    exact = MF.analytic_daughters(p)
    numeric = pb.operator.system.populations.mill.model.terms[2].daughters
    @test maximum(sum(abs.(numeric.eta-exact.eta); dims=1)) < 2e-10
    @test vec(p.pivots' * exact.eta) + exact.low1 ≈ p.pivots rtol=2e-13
    expected_number = MF.MM.relative_moment(0, 1., p.sigma, p.daughter_ratio) /
        MF.MM.relative_moment(1, 1., p.sigma, p.daughter_ratio)
    @test vec(sum(numeric.eta; dims=1)) + numeric.underflow_number ≈
        fill(expected_number, length(p.pivots)) rtol=2e-13

    # Constant number density in v in [1,3]: its volume-weighted median solves
    # (v^2-1)/(9-1)=1/2, so v=sqrt(5), rather than the count median v=2.
    synthetic = (pivots=[2.], edges=[1., 3.], feed=[1.], retained=[0.],
        Q=zeros(1, 1), low0=[0.], low1=[0.], segment_speed=1.)
    @test MF.finite_bin_median(synthetic, [0.], 0.) ≈ 1e6cbrt(6sqrt(5)/π) rtol=2e-15
    @test_throws NumericalFailure MF.finite_bin_median(merge(synthetic, (feed=[0.],)), [0.], 0.)

    du = similar(pb.u0)
    rhs!(du, pb.u0, pb.operator, pb.p, 0.)
    uptake = p.rho*p.number_scale*dot(p.pivots, population(pb, du, :mill))
    @test uptake + bulk(pb, du, :discharged_mass) ≈ p.mass_feed rtol=2e-13
    @test du[findall(==(0), p.retained)] == zeros(count(==(0), p.retained))
    sol = MF.MS.solve(pb, [1., 5., 10., 30., 60., 100., 180.])
    check = MF.exponential_check(pb, sol)
    @test check.maximum_d50_difference_um < 1e-3
    @test check.maximum_holdup_difference_g < 1e-3
    @test minimum(minimum.(sol.u)) >= 0
end
