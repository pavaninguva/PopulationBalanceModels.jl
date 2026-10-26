"""Solver-control diagnostic; not a manuscript timing or production dataset.

From the repository root: julia --project=paper paper/verification/precipitation/pilot_order.jl 3
"""
using PopulationBalanceModels
include(joinpath(@__DIR__,"../../../examples/continuous_precipitation.jl"))
include(joinpath(@__DIR__,"../../precipitation_solver.jl"))
C,S = ContinuousPrecipitation,PaperPrecipitationSolver
order = isempty(ARGS) ? 3 : parse(Int,only(ARGS))
warm = C.build(nx=8,nz=6,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=.01)
S.advance(warm,[.01];max_order=order,progress=false)
pb = C.build(nx=400,nz=200,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=10.)
GC.gc()
measured = @timed S.advance(pb,[1.,5.,10.];max_order=order,reltol=1e-7,
    population_abstol=1e-14,bulk_abstol=1e-11,dtmax=.5)
println("Maximum BDF order=",order,"; seconds=",measured.time,
    "; steps=",measured.value.steps,"; diagnostics=",diagnostics(measured.value))
pb = C.build(nx=12,nz=12,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=5.)
implicit = S.advance(pb,[1.,5.];max_order=order,reltol=1e-8,
    population_abstol=1e-16,bulk_abstol=1e-12,dtmax=.1,progress=false)
native = C.advance(pb,[1.,5.];fraction=.075)
for i in eachindex(implicit.t)
    relative = sum(abs.(population(pb,implicit.u[i],:crystals).-population(pb,native.u[i],:crystals)))/
        sum(abs,population(pb,native.u[i],:crystals))
    println("t=",implicit.t[i],"; population difference from refined RK4=",relative,
        "; maximum bulk difference=",maximum(abs.(bulk(pb,implicit.u[i],:c).-bulk(pb,native.u[i],:c))))
end
