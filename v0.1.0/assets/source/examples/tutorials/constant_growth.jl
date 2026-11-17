# One translating population, in the benchmark's coordinate and time units.
using PopulationBalanceModels

p = (speed=0.1, height=100.0, center=0.2, width=0.03)
growth(x,p,t,ctx) = p.speed
initial_density(x,p,t) = p.height*exp(-0.5*((x-p.center)/p.width)^2)

model = PBMModel(growth=Growth(growth))
grid = Grid1D(range(0.0,1.0;length=201);coordinate=:length)
boundary = SizeBC(OpenBoundary(),OpenBoundary()) # Zero inflow; free outflow.
operator = compile(model,grid;methods=TermMethods(growth=MUSCL()),bc=boundary)

# Physical number density on [0,1], without probability normalization.
initial = InitialDensity(initial_density;support=(0.0,1.0))
problem = PBMProblem(operator,initial,(0.0,4.0);p)
dt = 0.5*minimum(grid.widths)/p.speed # CFL=0.5 for this constant velocity.
solution = simulate(problem;method=SSPRK33(),dt,saveat=[0.0,2.0,4.0])
@show observe(MomentObservable(0),solution)

# Optional, after the calculation (requires the plotting environment):
# import PythonPlot
# picture = plot_distribution(solution;xlabel="Size coordinate",ylabel="Number density")
# save_figure(picture,"growth.png")
# Exact bin-integral errors, method sweeps and timings: constant_growth_methods.jl.
