# Minimal constant growth tutorial.
# Generated from maintained scientific source sections and a short run block.
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

# From examples/constant_growth_methods.jl (pairings).
# Each tuple names a size reconstruction and time method. A complete space–time
# update has no separate reconstruction. Inguva2022 requires C=1.
combinations() = [
    ("upwind", "euler", Upwind(), ForwardEuler()),
    ("muscl", "ssprk33", MUSCL(), SSPRK33()),
    ("weno3", "ssprk33", WENO3(), SSPRK33()),
    ("weno5", "ssprk33", WENO5(), SSPRK33()),
    ("weno5", "rk4", WENO5(), RK4()),
    ("hoc", "midpoint", nothing, HOCFiniteVolume()),
    ("inguva2022", "coupled", nothing, Inguva2022()),
]

function tutorial_problem(; bins=200)
    grid = Grid1D(range(0.0, 1.0; length=bins+1); coordinate=:length)
    model = PBMModel(growth=Growth(AffineGrowth(growth, 0.0)))
    boundary = SizeBC(OpenBoundary(), OpenBoundary())
    (; grid, model, boundary, initial=gaussian_counts(grid, 0.0),
       reference=gaussian_counts(grid, 4.0))
end

function main()
    case = tutorial_problem()
    for (name, time_name, scheme, time_method) in combinations()
        operator = scheme === nothing ?
            compile(case.model, case.grid; evolution=time_method, bc=case.boundary) :
            compile(case.model, case.grid;
                methods=TermMethods(growth=scheme), bc=case.boundary)
        problem = PBMProblem(operator, case.initial, (0.0, 4.0))
        CFL = time_method isa Inguva2022 ? 1.0 : 0.5
        dt = CFL * first(case.grid.widths) / growth
        # Signed diagnostic comparison: expose undershoots rather than clipping.
        solution = simulate(problem; method=time_method, dt,
            saveat=[4.0], check_nonnegative=false)
        error = (vec(last(solution.u)) .- case.reference) ./ case.grid.widths
        L1 = sum(case.grid.widths .* abs.(error))
        RMSE = sqrt(sum(case.grid.widths .* error.^2) / sum(case.grid.widths))
        @show name time_name CFL L1 RMSE minimum(density(case.grid, last(solution.u)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
