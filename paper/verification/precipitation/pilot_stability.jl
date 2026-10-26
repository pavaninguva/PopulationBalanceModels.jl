using PopulationBalanceModels
include(joinpath(@__DIR__,"../../../examples/continuous_precipitation.jl"))
include(joinpath(@__DIR__,"../../precipitation_solver.jl"))
C,S=ContinuousPrecipitation,PaperPrecipitationSolver
solutions=[]
for flag in (false,true)
    warm=C.build(nx=8,nz=6,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=.01)
    S.advance(warm,[.01];stability_limit_detect=flag,max_order=5,progress=false)
    pb=C.build(nx=100,nz=100,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=5.)
    GC.gc()
    measured=@timed S.advance(pb,[1.,5.];reltol=1e-7,population_abstol=1e-14,bulk_abstol=1e-11,dtmax=.5,stability_limit_detect=flag,max_order=5)
    push!(solutions,measured.value)
    println("stability_limit_detect=",flag," seconds=",measured.time," steps=",measured.value.steps);flush(stdout)
end
for i in 1:2
    a,b=solutions[1].u[i],solutions[2].u[i];pb=solutions[1].problem
    println("time=",solutions[1].t[i]," relative_population_difference=",sum(abs.(population(pb,a,:crystals).-population(pb,b,:crystals)))/sum(abs,population(pb,b,:crystals)))
end
