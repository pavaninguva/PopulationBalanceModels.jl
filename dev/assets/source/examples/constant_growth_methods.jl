module ConstantGrowthMethods
using PopulationBalanceModels

const growth = 0.1
initial_density(a) = a >= 0 ? 100exp(-a/0.01) : 0.0
primitive(a) = a <= 0 ? 0.0 : -expm1(-a/0.01)

"""Analytical bin numbers for the translating exponential with zero inflow."""
function exact_counts(grid,t)
    [b <= growth*t ? 0.0 : exp(-max(a-growth*t,0)/0.01) *
        (-expm1(-(b-max(a,growth*t))/0.01))
        for (a,b) in zip(grid.edges[1:end-1],grid.edges[2:end])]
end

"""Bin integrals of a smooth periodic density; shift the integration endpoints."""
function smooth_counts(grid,t)
    a = grid.edges[1:end-1] .- growth*t
    b = grid.edges[2:end] .- growth*t
    grid.widths .+ 0.2 .* (cos.(2pi.*a) .- cos.(2pi.*b)) ./ (2pi) .+
        0.1 .* (sin.(4pi.*b) .- sin.(4pi.*a)) ./ (4pi)
end

# Each tuple names a size reconstruction and time method. A complete space–time
# update has no separate reconstruction. Inguva2022 requires C=1.
combinations() = [
    ("upwind", "euler", Upwind(), ForwardEuler()),
    ("muscl", "ssprk33", MUSCL(), SSPRK33()),
    ("weno3", "ssprk33", WENO3(), SSPRK33()),
    ("weno5", "ssprk33", WENO5(), SSPRK33()),
    ("weno5", "rk4", WENO5(), RK4()),
    ("hoc", "midpoint", nothing, HOCFiniteVolume()),
    ("lax_wendroff", "coupled", nothing, LaxWendroff()),
    ("inguva2022", "coupled", nothing, Inguva2022()),
]

function make_problem(grid,initial,scheme,time_method,tf,bc)
    model = PBMModel(growth=Growth(AffineGrowth(growth,0.0)))
    operator = if scheme === nothing
        compile(model,grid; evolution=time_method,bc)
    else
        compile(model,grid; growth_scheme=scheme,bc)
    end
    PBMProblem(operator,initial,(0.0,tf))
end

function measure(grid,initial,truth,scheme,time_method,tf,dt,bc)
    problem = make_problem(grid,initial,scheme,time_method,tf,bc)
    # ONE integration, after the separate small warmups in the driver.
    # Keep signed undershoots: these are diagnostic transport comparisons.
    GC.gc() # outside timing: do not charge earlier configurations' garbage to this solve
    timed = @timed simulate(problem; method=time_method,dt,check_nonnegative=false)
    solution = timed.value
    N = vec(last(solution.u))
    f = density(grid,N)
    error = density(grid,N .- truth)
    (; N,f,rmse=sqrt(sum(error.^2 .* grid.widths)/sum(grid.widths)),
        l1=sum(abs.(error) .* grid.widths),maximum_error=maximum(abs.(error)),
        minimum_density=minimum(f),negative_number=-sum(min.(N,0)),
        number_drift=sum(N)-sum(initial),seconds=timed.time,bytes=timed.bytes,gc_seconds=timed.gctime,steps=solution.steps)
end

include("support/constant_growth_benchmark.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
end
