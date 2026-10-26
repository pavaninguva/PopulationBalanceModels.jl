using Test, PopulationBalanceModels, Optim, LinearAlgebra
isdefined(@__MODULE__, :bayesian_linear_fixture) || include("fixtures/bayesian_linear.jl")

@testset "MAP integrates optional Gauss–Newton covariance without changing its target" begin
    f=bayesian_linear_fixture()
    baseline=fit(f.problem)
    result=fit(f.problem; uncertainty=:gauss_newton)
    @test result.converged
    @test result.uncertainty isa GaussNewtonApproximation
    @test result.minimizer≈baseline.minimizer rtol=1e-10
    @test result.logposterior≈baseline.logposterior rtol=1e-12
    @test result.uncertainty.covariance≈f.covariance rtol=1e-10
    @test result.uncertainty.diagnostics.valid
    @test baseline.uncertainty===nothing
    @test fit(f.problem; uncertainty=:laplace).uncertainty isa LaplaceApproximation
    @test_throws ArgumentError fit(f.problem; uncertainty=:unknown)
end
