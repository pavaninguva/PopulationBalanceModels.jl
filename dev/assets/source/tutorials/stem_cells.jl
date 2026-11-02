# Minimal stem cells tutorial.
# Generated from maintained scientific source sections and a short run block.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using LinearAlgebra
using Distributions: Normal, cdf, ccdf

# From examples/stem_cell_population.jl (model).
"""Brown et al. (2025), Eq. 7: net intracellular cargo change, in a.u./model time."""
cargo_rate(x, p, t, ctx) = (50 + x^2) / (100 + x^2) - 0.005x

"""Division hazard (Eq. 6), not the daughter birth rate."""
division_rate(x, p, t, ctx) = (x / 100)^5

"""Eq. 9: exit from the pluripotent cohort, not cell death or diffusivity."""
differentiation_rate(x, p, t, ctx) = p.differentiation ? 0.5 / (1 + 0.005exp(0.25x)) : zero(x)

"""
Positive Gauss–Jacobi quadrature of TWO expected daughters for r ~ Beta(q,q).
A mother y gives daughters r*y and (1-r)*y. The symmetric marginal alone,
with weights summing to two, supplies the deterministic binary-division PBM.
The rule integrates ratio polynomials through degree 2*order-1 exactly in
exact arithmetic. It conserves the additive cargo y, not y^3.
"""
function beta_daughters(; q=40.0, order=12)
    isfinite(q) && q >= 1 || throw(ArgumentError("require finite q ≥ 1"))
    order isa Integer && order >= 2 || throw(ArgumentError("at least two daughter nodes required"))
    alpha = float(q) - 1
    offdiagonal = [sqrt(k * (k + 2alpha) / ((2k + 2alpha)^2 - 1)) for k in 1:order-1]
    rule = eigen(SymTridiagonal(zeros(order), offdiagonal))
    ratios = (rule.values .+ 1) ./ 2
    counts = 2 .* rule.vectors[1, :].^2
    SelfSimilarDaughters(ratios, counts; measure_order=1)
end

"""Exact bin probabilities of Normal(50,0.5), truncated to [0,100]."""
function initial_population(grid; concentration=1.0)
    isfinite(concentration) && concentration > 0 || throw(ArgumentError("initial concentration must be positive"))
    normal = Normal(50.0, 0.5)
    normalization = cdf(normal, 100.0) - cdf(normal, 0.0)
    map(grid.edges[1:end-1], grid.edges[2:end]) do left, right
        a, b = max(0.0, left), min(100.0, right)
        b <= a && return 0.0
        # Use the survival function in the right tail to avoid subtracting two ones.
        probability = a >= 50 ? ccdf(normal, a) - ccdf(normal, b) : cdf(normal, b) - cdf(normal, a)
        concentration * probability / normalization
    end
end

"""
Build the native sectional PBM with coupled historical exit/count balances.
The differentiated compartment records counts and cargo AT EXIT; it has no
subsequent proliferation, death or protein dynamics. CumulativeAccounting
separately records daughter underflow. This avoids the currently unsupported
combination of population-transfer interactions and size-loss ledgers.

The scalar synthesis_balance is a negative diagnostic counter supplied by
ConservativeExchange from the selected discrete growth residual. It is not an
extracellular nutrient concentration. No biological cargo reservoir is invented.
"""
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

"""Solve with MUSCL / fixed pivot / SSPRK33 and a combined static loss bound.
All coefficients are autonomous and independent of the population and counters,
so the initial suggest_dt bound applies for the entire trajectory. It is still
a recommendation: native stage checks remain enabled, with zero negative tolerance.
"""
function solve_case(; maximum_dt=0.05, cfl=0.35, saveat=nothing, kwargs...)
    isfinite(maximum_dt) && maximum_dt > 0 || throw(ArgumentError("maximum_dt must be positive"))
    0 < cfl <= 0.35 || throw(ArgumentError("use 0 < cfl ≤ 0.35 for this case"))
    problem = build_problem(; kwargs...)
    dt = min(maximum_dt, suggest_dt(problem.operator, problem.u0, problem.p, 0.0; cfl))
    times = isnothing(saveat) ? range(0.0, last(problem.tspan); length=101) : saveat
    simulate(problem; dt, saveat=times, method=SSPRK33(), negative_tolerance=0.0)
end

function state_summary(problem, state, time)
    grid = problem.operator.system.populations.stem.grid
    cells = population(problem, state, :stem)
    initial = population(problem, problem.u0, :stem)
    number = sum(cells)
    cargo = moment(grid, cells, 1)
    lost_number = cumulative(problem, state, :stem, :breakage_underflow)
    lost_cargo = cumulative(problem, state, :stem, :breakage_underflow, :cargo)
    upper_loss = cumulative(problem, state, :stem, :breakage_overflow)
    divisions = bulk(problem, state, :divisions)
    exits = bulk(problem, state, :differentiated_cells)
    exit_cargo = bulk(problem, state, :differentiated_cargo)
    synthesis = -bulk(problem, state, :synthesis_balance)
    probabilities = conditional_cdf(grid, cells, grid.edges; window=(0.0, last(grid.edges)))
    quantiles = cdf_quantiles(grid.edges, probabilities, [0.1, 0.5, 0.9])
    (time=time, stem=number, differentiated=exits, mean_cargo=cargo/number,
     q10=quantiles[1], q50=quantiles[2], q90=quantiles[3],
     cargo=cargo, exit_cargo=exit_cargo, synthesized=synthesis, divisions=divisions,
     lost_number=lost_number, lost_cargo=lost_cargo, upper_loss=upper_loss,
     number_balance=number+exits+lost_number-sum(initial)-divisions,
     cargo_balance=cargo+exit_cargo+lost_cargo-synthesis-moment(grid, initial, 1),
     minimum_count=minimum(cells), upper_bin_fraction=last(cells)/number)
end

function tutorial_problem(; differentiation=false, bins=600)
    # A short transient in synthetic model time, not the balanced-growth study.
    build_problem(; bins, tf=20.0, differentiation)
end

function main()
    for differentiation in (false, true)
        problem = tutorial_problem(; differentiation)
        dt = min(0.05, suggest_dt(problem.operator, problem.u0, problem.p, 0.0; cfl=0.35))
        solution = simulate(problem; method=SSPRK33(), dt,
            saveat=[0.0,10.0,20.0], negative_tolerance=0.0)
        summary = state_summary(problem, last(solution.u), last(solution.t))
        @show differentiation summary.stem summary.differentiated
        @show summary.number_balance summary.cargo_balance
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
