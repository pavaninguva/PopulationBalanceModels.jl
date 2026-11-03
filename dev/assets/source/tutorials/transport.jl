# Joint number density in two internal coordinates on [0,2]^2.
using PopulationBalanceModels

p = (offset=(0.1,0.5),slope=(0.05,0.25),amplitude=50.0,
     center=(0.4,0.4),width=(0.05,0.05))
growth(x,y,p,t,ctx) = (p.offset[1]+p.slope[1]*x,p.offset[2]+p.slope[2]*y)
initial_density(xy,p,t) = p.amplitude*exp(-0.5*((xy[1]-p.center[1])/p.width[1])^2
                                            -0.5*((xy[2]-p.center[2])/p.width[2])^2)
model = PBMModel(growth=Growth2D(growth))
grid = Grid2D(range(0.0,2.0;length=51),range(0.0,2.0;length=51))
axis_boundary = SizeBC(OpenBoundary(),NoFlux())
operator = compile(model,grid;methods=TermMethods(growth=WENO5(epsilon=1e-12)),
    bc=SizeBC2D(x=axis_boundary,y=axis_boundary))
initial = InitialDensity(initial_density;support=((0.0,2.0),(0.0,2.0)))
problem = PBMProblem(operator,initial,(0.0,1.0);p)
dt = suggest_dt(operator,problem.u0,p,0.0;cfl=0.4)

if abspath(PROGRAM_FILE) == @__FILE__
    # Diagnostic signed WENO undershoots are retained; no clipping is performed.
    solution = simulate(problem;method=SSPRK33(),dt,saveat=[0.0,1.0],check_nonnegative=false)
    @show observe(MomentObservable((0,0)),solution)
    # Optional: import PythonPlot
    # plot_field(solution;orientation=:internal_x,xlabel="First coordinate",ylabel="Second coordinate")
end
# Analytical bin integrals/travel-time methods: inguva_braatz_case2.jl.
