using PopulationBalanceModels
include("papers/gunawan2004_case2.jl")
using .Gunawan2004Case2

# Demonstration initialization, NOT a correction claimed for the publication.
# Replace with the author's unrounded seed and confirmed c0 for reproduction.
seed=L->180.5<=L<=210.5 ? 0.08*(1-((L-195.5)/15)^2) : 0.0
grid=Grid1D(range(0.0,1600.0;length=801);coordinate=:length)
problem=Gunawan2004Case2.build(;grid,initial_density=seed,
    c0=solubility(temperature(0.0)),tspan=(0.0,50.0))
solution=simulate(problem;dt=0.02,saveat=[0.0,10.0,50.0])
initial_total=bulk(problem,problem.u0,:c)+problem.p.rho*moment(grid,population(problem,problem.u0,:crystals),3)
final_total=bulk(problem,solution.u[end],:c)+problem.p.rho*moment(grid,population(problem,solution.u[end],:crystals),3)
println("Dissolved concentration at final time: ",bulk(problem,solution.u[end],:c))
println("Discrete total-material drift: ",final_total-initial_total)
