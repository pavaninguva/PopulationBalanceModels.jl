# Minimal constant growth tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using Distributions: Normal

# From examples/constant_growth_methods.jl (velocity).
const growth = 0.1

# Physical definitions from examples/constant_growth_methods.jl.
const gaussian_height = 100.0

const gaussian_center = 0.2

const gaussian_width = 0.03 # standard deviation, independent of bin width

gaussian_initial(t=0.0) = InitialDistribution(
    Normal(gaussian_center+growth*t,gaussian_width);
    total=gaussian_height*gaussian_width*sqrt(2pi),
    support=(growth*t,1.0+growth*t),truncation=:physical)

function tutorial_problem(; bins=200)
    grid = Grid1D(range(0.0, 1.0; length=bins+1); coordinate=:length)
    model = PBMModel(growth=Growth(AffineGrowth(growth, 0.0)))
    boundary = SizeBC(OpenBoundary(), OpenBoundary())
    operator = compile(model, grid;
        methods=TermMethods(growth=MUSCL()), bc=boundary)
    initial = gaussian_initial() # Number concentration, with physical [0,1] support.
    problem = PBMProblem(operator, initial, (0.0, 4.0))
    reference = vec(initialize(operator, gaussian_initial(4.0))) # Exact translation.
    (; grid, model, boundary, initial, problem, reference)
end

function main(; bins=200)
    case = tutorial_problem(; bins)
    dt = 0.5 * first(case.grid.widths) / growth
    solver = ForwardSolver(SSPRK33(); dt)
    solution = simulate(case.problem; solver, saveat=[4.0])
    error = (vec(last(solution.u)) .- case.reference) ./ case.grid.widths
    @show sum(case.grid.widths .* abs.(error)) # Integrated L1 density error.
    solution
end

# Optional: import PythonPlot; plot_results(solution). No solve/export occurs here.
plot_results(solution) = plot_distribution(solution; times=[last(solution.t)],
    xlabel="Size coordinate", ylabel="Number density")

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
