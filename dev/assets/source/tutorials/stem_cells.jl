# Minimal stem cells tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using Distributions: Normal, Beta, pdf

# Physical definitions from examples/stem_cell_population.jl.
cargo_rate(x, p, t, ctx) = (50 + x^2) / (100 + x^2) - 0.005x

division_rate(x, p, t, ctx) = (x / 100)^5

differentiation_rate(x, p, t, ctx) = p.differentiation ? 0.5 / (1 + 0.005exp(0.25x)) : zero(x)

# From examples/stem_cell_population.jl (inputs).
"""Continuous binary-division number measure: two Beta(40,40) cargo fractions."""
division_daughters() = ContinuousDaughters(
    (r,parent,p,t,ctx)->2pdf(Beta(40.0,40.0),r);
    coordinates=:ratio,support=(0.0,1.0),count=2.0,conserved_order=1)

"""Unit-normalized physical initial law on [0,100], with explicit cell concentration."""
function initial_distribution(;concentration=1.0)
    isfinite(concentration) && concentration>0 || throw(ArgumentError("initial concentration must be positive"))
    InitialDistribution(Normal(50.0,0.5);total=concentration,
        support=(0.0,100.0),truncation=:renormalize)
end

# Physical definitions from examples/stem_cell_population.jl.
function assemble_problem(grid,daughters,initial;tf,differentiation,
                          methods=TermMethods(growth=MUSCL()))
    model = PBMModel(
        expression=Growth(cargo_rate),
        division=Breakage(division_rate, daughters; overflow=:discard,
                          coordinate_map=AdditiveCoordinate(identity, identity)),
        differentiation=Death(differentiation_rate),
    )
    operator = compile(model, grid; methods, bc=SizeBC())
    function counters(ctx, p, t)
        cells = ctx.populations.stem
        divisions = sum(division_rate(x, p, t, ctx) * cells[i] for (i, x) in enumerate(grid.centers))
        exits = sum(differentiation_rate(x, p, t, ctx) * cells[i] for (i, x) in enumerate(grid.centers))
        exit_cargo = sum(x * differentiation_rate(x, p, t, ctx) * cells[i] for (i, x) in enumerate(grid.centers))
        (divisions=divisions, differentiated_cells=exits, differentiated_cargo=exit_cargo)
    end
    system = PBMSystem((stem=operator,); bulk_rhs=counters,
        exchanges=(ConservativeExchange(:stem, :synthesis_balance, (x,p,t,ctx)->x;
                                         mechanisms=(:growth,)),),
        accounting=(CumulativeAccounting(:stem; properties=(cargo=1,)),),
        nonnegative_bulk=(:divisions, :differentiated_cells, :differentiated_cargo))
    PBMProblem(system, (stem=initial,), (0.0, float(tf));
        p=(differentiation=differentiation,),
        bulk0=(divisions=0.0, differentiated_cells=0.0,
               differentiated_cargo=0.0, synthesis_balance=0.0))
end

function continuous_problem(;bins=600,xmax=120.0,tf=200.0,differentiation=false,
                            concentration=1.0,daughter_quadrature=GaussLegendre(8))
    bins isa Integer && bins>=20 || throw(ArgumentError("use at least 20 cargo bins"))
    100<=xmax<=180 || throw(ArgumentError("use 100 ≤ xmax ≤ 180; this case keeps R positive"))
    isfinite(tf) && tf>0 || throw(ArgumentError("tf must be positive"))
    differentiation isa Bool || throw(ArgumentError("differentiation must be true or false"))
    grid=Grid1D(range(0.0,xmax;length=bins+1);coordinate=:oct4)
    daughters=division_daughters()
    initial=initial_distribution(;concentration)
    methods=TermMethods(growth=MUSCL(),breakage=FixedPivot(;daughter_quadrature))
    assemble_problem(grid,daughters,initial;tf,differentiation,methods)
end

function tutorial_problem(; differentiation=false, bins=600)
    # A short transient in synthetic model time, not the balanced-growth study.
    continuous_problem(; bins, tf=20.0, differentiation)
end

function main(; bins=600, differentiation=true, times=[0.0,10.0,20.0])
    problem = tutorial_problem(; bins, differentiation)
    # Autonomous, state-independent rates: this bound applies throughout the run.
    dt = min(0.05, suggest_dt(problem.operator, problem.u0, problem.p, 0.0; cfl=0.35))
    solver = ForwardSolver(SSPRK33(); dt, negative_tolerance=0.0)
    solution = simulate(problem; solver, saveat=times)
    @show observe(PopulationMoment(:stem, 0), solution)
    @show bulk(problem, last(solution.u), :differentiated_cells)
    solution
end

# Optional after import PythonPlot; synthetic model time is not hours.
function plot_results(solution)
    distribution = plot_distribution(solution; population=:stem,
        xlabel="Cargo (arbitrary units)", ylabel="Cell number density")
    number = plot_observable(solution, PopulationMoment(:stem,0);
        xlabel="Model time", ylabel="Stem-cell concentration")
    (; distribution, number)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
