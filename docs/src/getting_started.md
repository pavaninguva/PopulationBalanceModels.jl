# Getting started

## Installation

Extract the source archive and enter its inner `PopulationBalanceModels.jl` directory.
Julia 1.10 or newer is required. From that directory run:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); using PopulationBalanceModels; Pkg.test()'
julia --project=. examples/batch_population_balance.jl
```

For installation into another working environment, use `Pkg.develop(path="/absolute/path/to/PopulationBalanceModels.jl")`.
Repository URL installation is described in [Publishing](publishing.md).

## A complete population balance

This example executes when the documentation builds:

```@example quickstart
using PopulationBalanceModels

grid = Grid1D(range(0.0, 10.0; length=101); coordinate=:volume)
model = PBMModel(
    growth = Growth(rate=(v, p, t, ctx) -> p.G),
    primary = Nucleation(rate=(p, t, ctx) -> p.J, size=0.0),
    loss = Death(rate=(v, p, t, ctx) -> p.k),
)
methods = TermMethods(
    growth = MUSCL(),
    by_term = (primary = BoundaryInjection(:left),),
)
operator = compile(model, grid; methods)
initial = exp.(-((grid.centers .- 2.0) ./ 0.4).^2) .* grid.widths
problem = PBMProblem(operator, initial, (0.0, 1.0);
    p=(G=0.3, J=0.05, k=0.02))
solution = simulate(problem; dt=0.01, saveat=0.0:0.25:1.0)

@assert all(isfinite, solution.u[end])
@assert minimum(solution.u[end]) >= 0.0
(number=only(moment(grid, solution.u[end], 0)),
 first_moment=only(moment(grid, solution.u[end], 1)))
```

`initial` contains bin-integrated number concentration. Multiplying a density
profile by `grid.widths` converts it to that representation. `density` performs
the reverse conversion. A single-population solution stores `N[bin, 1]`.

`BoundaryInjection` interprets nucleation as an inward number flux; it needs a
compatible growth term and a source located at the selected domain boundary.
Choose a Gaussian or pivot source method when those assumptions do not apply.

## Case studies

The [three case studies](case_studies/index.md) build on this workflow:
analytical two-coordinate transport, cooling crystallization with a dissolved
concentration balance, and lipid nanoparticle formation. Each separates its
physical rates, term discretization, time integration, and observations.

Run `examples/inguva_braatz_case1.jl` for the analytical comparison and
`examples/cooling_crystallization_benchmark.jl` for the 1000-second coupled
numerical study. The LNP page distinguishes the explicit synthetic demonstration
from the author inputs still needed for the paper comparison.

Optional Optim and SciML dependencies belong in a separate working environment;
see [Observations and estimation](inference.md).
