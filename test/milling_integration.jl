using Test, PopulationBalanceModels, LinearAlgebra
import SciMLBase, Sundials
include("../examples/milling_uncertainty.jl")
const MU=MillingUncertainty

@testset "Segmented milling BDF, independent exponential and UQ integration" begin
    pb=MU.MillingModel.build()
    times=[.01,5.,29.999,30.,30.001,31.,200.]
    sol=MU.MillingSolver.solve(pb,times)
    exact=MU.exponential_reference(pb,times)
    @test sol.t==times && sol.steps>0
    @test maximum(maximum(abs,sol.u[i]-exact.u[i]) for i in eachindex(times))<5e-8
    @test MU.state_diagnostics(pb,sol).maximum_mass_error_kg<1e-11
    @test MU.state_diagnostics(pb,sol).minimum_inventory>=-1e-12
    @test maximum(abs(only(observe(MU.MillingModel.median_observable(),sol;index=i))-
        only(observe(MU.MillingModel.median_observable(),exact;index=i))) for i in eachindex(times))<1e-3
    @test last(MU.inputs().names)===:sigma
    problem=MU.problem(;times)
    sample=propagate(problem,MonteCarlo(8;seed=29);max_solves=8)
    @test all(sample.success) && sample.metadata.solves==8
    @test uq_summary(sample).mean.d50[1,6]>0
    # The state is continuous at 30 s, but the instantaneous algebraic outlet
    # changes with speed. Explicitly verify the stated right-hand convention.
    u=sol.u[4];N=population(pb,u,:mill)
    before=MU.MillingModel.outlet(merge(pb.p,(segment_speed=4923.,)),N,30.)
    after=MU.MillingModel.outlet(pb.p,N,30.)
    @test after.resolved≈(1500/4923).*before.resolved.+(1-1500/4923).*(1 .-pb.p.retained).*pb.p.feed rtol=3e-15
    @test MU.MillingModel.speed(pb.p,prevfloat(30.))==4923
    @test MU.MillingModel.speed(pb.p,30.)==1500
end
