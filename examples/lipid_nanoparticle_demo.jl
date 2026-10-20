using PopulationBalanceModels
include("papers/cej2025_lnp.jl")
using .CEJ2025LNP

# Wiring demonstration with EXPLICIT SYNTHETIC CLOSURES. These values do not
# establish a reproduction of the CEJ paper. All mechanisms and coupling paths
# are exercised; replace closures/inputs with the author's confirmed model.
grid=Grid1D(exp.(range(log(1e-9),log(2e-6);length=61));coordinate=:diameter)
problem=CEJ2025LNP.build(;grid,initial_lipid=10.0,FRR=3.0,tau=0.01,
    temperature=t->298.15,viscosity=(w,T)->1e-3,
    solubility=(w,T)->10.0*exp(-8w),
    equilibrium_ratio=(L,w,T)->1.0,
    attachment_without_peg=(L,M,w,T)->1.0,
    molecular_volume=5e-28,particle_density=1000.0,nucleation_width=1e-9,
    nucleation_normalization=:total_rate,tspan=(0.0,0.02),kg=0.05,
    overflow=:discard)
solution=simulate(problem;dt=2e-5,saveat=[0.0,0.005,0.02])
N=population(problem,solution.u[end],:lnp)
println("Synthetic demonstration particle concentration: ",sum(N))
println("Paper-defined intensity mean [nm]: ",1e9*paper_intensity_mean(grid,N))
println("Free lipid [kg/m^3]: ",bulk(problem,solution.u[end],:lipid))
