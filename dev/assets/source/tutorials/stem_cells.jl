# Brown's synthetic model: cargo in arbitrary units and time in model units.
using PopulationBalanceModels
using Distributions: Normal, Beta, pdf

p = (differentiation=true,)
cargo_rate(x,p,t,ctx) = (50+x^2)/(100+x^2)-0.005x
division_rate(x,p,t,ctx) = (x/100)^5
differentiation_rate(x,p,t,ctx) = p.differentiation ? 0.5/(1+0.005exp(0.25x)) : zero(x)
daughters = ContinuousDaughters((r,parent,p,t,ctx)->2pdf(Beta(40.0,40.0),r);
    coordinates=:ratio,support=(0.0,1.0),count=2.0,conserved_order=1)
model = PBMModel(
    expression=Growth(cargo_rate),
    division=Breakage(division_rate,daughters;overflow=:discard,
        coordinate_map=AdditiveCoordinate(identity,identity)),
    differentiation=Death(differentiation_rate),
)
grid = Grid1D(range(0.0,120.0;length=601);coordinate=:oct4)
operator = compile(model,grid;bc=SizeBC(),
    methods=TermMethods(growth=MUSCL(),breakage=FixedPivot(daughter_quadrature=GaussLegendre(8))))
initial = InitialDistribution(Normal(50.0,0.5);total=1.0,
    support=(0.0,100.0),truncation=:renormalize)
problem = PBMProblem(operator,initial,(0.0,20.0);p)
# Autonomous rates: this combined growth/loss bound applies throughout this run.
dt = min(0.05,suggest_dt(operator,problem.u0,p,0.0;cfl=0.35))

if abspath(PROGRAM_FILE) == @__FILE__
    solution = simulate(problem;method=SSPRK33(),dt,negative_tolerance=0.0,saveat=[0.0,10.0,20.0])
    @show observe(MomentObservable(0),solution)
    # Optional: import PythonPlot
    # plot_distribution(solution;xlabel="Cargo (arbitrary units)",ylabel="Cell number density")
end
# Diagnostic division/exit counters and cumulative daughter losses can be added
# afterward; they do not feed back. See the stem_cell_accounting.jl companion.
