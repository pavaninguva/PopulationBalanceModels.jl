# Minimal constant growth tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using Distributions: Normal, cdf, ccdf

# From examples/constant_growth_methods.jl (velocity).
const growth = 0.1

# From examples/constant_growth_methods.jl (gaussian).
const gaussian_height = 100.0
const gaussian_center = 0.2
const gaussian_width = 0.03 # standard deviation, independent of bin width
const gaussian_profile = Normal(gaussian_center,gaussian_width)
gaussian_density(a) = 0 <= a <= 1 ?
    gaussian_height*exp(-0.5*((a-gaussian_center)/gaussian_width)^2) : 0.0

"""Exact bin numbers for a Gaussian initially on [0,1], with zero inflow.
Use survival probabilities above the centre to retain small Gaussian tails.
"""
function gaussian_counts(grid,t)
    shift = growth*t
    scale = gaussian_height*gaussian_width*sqrt(2pi)
    map(grid.edges[1:end-1],grid.edges[2:end]) do a,b
        lo,hi = max(a-shift,0.0),min(b-shift,1.0)
        hi <= lo && return 0.0
        fraction = lo >= gaussian_center ?
            ccdf(gaussian_profile,lo)-ccdf(gaussian_profile,hi) :
            cdf(gaussian_profile,hi)-cdf(gaussian_profile,lo)
        scale*fraction
    end
end

function tutorial_problem(; bins=200)
    grid = Grid1D(range(0.0, 1.0; length=bins+1); coordinate=:length)
    model = PBMModel(growth=Growth(AffineGrowth(growth, 0.0)))
    boundary = SizeBC(OpenBoundary(), OpenBoundary())
    (; grid, model, boundary, initial=gaussian_counts(grid, 0.0),
       reference=gaussian_counts(grid, 4.0))
end

function main()
    case = tutorial_problem()
    operator = compile(case.model, case.grid;
        methods=TermMethods(growth=MUSCL()), bc=case.boundary)
    problem = PBMProblem(operator, case.initial, (0.0, 4.0))
    dt = 0.5 * first(case.grid.widths) / growth
    solution = simulate(problem; method=SSPRK33(), dt, saveat=[4.0])
    error = (vec(last(solution.u)) .- case.reference) ./ case.grid.widths
    @show sum(case.grid.widths .* abs.(error)) # Integrated L1 density error.
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
