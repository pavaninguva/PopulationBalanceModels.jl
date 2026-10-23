"""Compare constant maximum-order-three BDF with a state-preserving BDF2
continuation. This expensive diagnostic is not part of the default workflow.
"""
using PopulationBalanceModels
include(joinpath(@__DIR__,"../../../examples/continuous_precipitation.jl"))
include(joinpath(@__DIR__,"../../precipitation_solver.jl"))
C,S = ContinuousPrecipitation,PaperPrecipitationSolver
warm = C.build(nx=8,nz=6,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=.02)
bulk(warm,warm.u0,:c) .= 2warm.p.ceq
S.advance(warm,[.01,.02];late_max_order=2,restart_time=.01,progress=false)
solutions = []
for late_order in (0,2)
    pb = C.build(nx=400,nz=200,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=200.)
    GC.gc()
    measured = @timed S.advance(pb,[5.,40.,80.,200.];max_order=3,late_max_order=late_order,
        restart_time=5.,reltol=1e-7,population_abstol=1e-14,bulk_abstol=1e-11,dtmax=.5)
    push!(solutions,measured.value)
    println("Late maximum order=",late_order," (zero means no restart); seconds=",measured.time,
        "; steps=",measured.value.steps)
end
pb = solutions[1].problem
for i in eachindex(solutions[1].t)
    a,b = solutions[2].u[i],solutions[1].u[i]
    difference = sum(abs.(population(pb,a,:crystals).-population(pb,b,:crystals)))/sum(abs,population(pb,b,:crystals))
    println("t=",solutions[1].t[i],"; relative population difference=",difference,
        "; maximum bulk difference=",maximum(abs.(bulk(pb,a,:c).-bulk(pb,b,:c))))
end
