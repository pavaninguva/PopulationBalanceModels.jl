# Minimal stem cells tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
# Prepared Beta(40,40) two-daughter rule and exact initial bin integrals.
include(joinpath(pkgdir(PopulationBalanceModels), "examples", "stem_cell_population.jl"))
using .StemCellPopulation: beta_daughters, initial_population

# Physical definitions from examples/stem_cell_population.jl.
cargo_rate(x, p, t, ctx) = (50 + x^2) / (100 + x^2) - 0.005x

division_rate(x, p, t, ctx) = (x / 100)^5

differentiation_rate(x, p, t, ctx) = p.differentiation ? 0.5 / (1 + 0.005exp(0.25x)) : zero(x)

function build_problem(; bins=600, xmax=120.0, tf=200.0, differentiation=false,
                       daughter_order=12, concentration=1.0)
    bins isa Integer && bins >= 20 || throw(ArgumentError("use at least 20 cargo bins"))
    100 <= xmax <= 180 || throw(ArgumentError("use 100 ≤ xmax ≤ 180; this case keeps R positive"))
    isfinite(tf) && tf > 0 || throw(ArgumentError("tf must be positive"))
    differentiation isa Bool || throw(ArgumentError("differentiation must be true or false"))
    grid = Grid1D(range(0.0, xmax; length=bins+1); coordinate=:oct4)
    daughters = beta_daughters(order=daughter_order)
    model = PBMModel(
        expression=Growth(cargo_rate),
        division=Breakage(division_rate, daughters; overflow=:discard,
                          coordinate_map=AdditiveCoordinate(identity, identity)),
        differentiation=Death(differentiation_rate),
    )
    operator = compile(model, grid; methods=TermMethods(growth=MUSCL()), bc=SizeBC())
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
    PBMProblem(system, (stem=initial_population(grid; concentration),), (0.0, float(tf));
        p=(differentiation=differentiation,),
        bulk0=(divisions=0.0, differentiated_cells=0.0,
               differentiated_cargo=0.0, synthesis_balance=0.0))
end

function tutorial_problem(; differentiation=false, bins=600)
    # A short transient in synthetic model time, not the balanced-growth study.
    build_problem(; bins, tf=20.0, differentiation)
end

function main()
    problem = tutorial_problem(differentiation=true)
    # Autonomous, state-independent rates: this bound applies throughout the run.
    dt = min(0.05, suggest_dt(problem.operator, problem.u0, problem.p, 0.0; cfl=0.35))
    solution = simulate(problem; method=SSPRK33(), dt,
        saveat=[0.0,10.0,20.0], negative_tolerance=0.0)
    @show observe(PopulationMoment(:stem, 0), solution)
    @show bulk(problem, last(solution.u), :differentiated_cells)
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
