# Coupled populations

The two populations below share a scalar solute state. Nucleation consumes
solute and a transfer interaction moves particles between populations.

```@example coupled
using PopulationBalanceModels

grid = Grid1D(0.5:1.0:10.5)
precursors = compile(PBMModel(
    Nucleation((p, t, ctx) -> p.k * ctx.bulk.c, 2.0)), grid)
products = compile(PBMModel(), grid)

system = PBMSystem((precursors=precursors, products=products);
    interactions=(
        PopulationTransfer(:precursors, :products,
            (size, p, t, ctx) -> p.transfer),
    ),
    exchanges=(
        ConservativeExchange(:precursors, :c,
            (size, p, t, ctx) -> size),
    ),
    nonnegative_bulk=(:c,))

problem = PBMProblem(system,
    (precursors=zeros(length(grid)), products=zeros(length(grid))),
    (0.0, 3.0); bulk0=(c=5.0,), p=(k=0.1, transfer=0.3))
solution = simulate(problem; dt=0.01, saveat=0.0:0.25:3.0)
final = stateview(problem, solution.u[end])

@assert bulk(problem, solution.u[end], :c) >= 0.0
(precursor_number=sum(population(problem, solution.u[end], :precursors)),
 product_number=sum(population(problem, solution.u[end], :products)),
 solute=bulk(problem, solution.u[end], :c))
```

The system packs all states into one solver vector and evaluates their current
values together at every time-integration stage. Laws can access `ctx.bulk`,
`ctx.populations`, `ctx.fields` and `ctx.system`.

`ConservativeExchange` derives the bulk exchange from the selected discrete
population contribution with a fixed particle weight. Declare transfers and
other material contributions consistently with the physical balance you intend.

The current coupled-system backend supports named one-coordinate sectional and
moment populations. The standalone two-coordinate transport backend is not yet
a general coupled two-coordinate system solver.


## Shared state and independent representations

A `PBMSystem((precursors=A,products=B); bulk_rhs=(ctx,p,t)->(;), fields=(p,t)->(;))`
combines compiled operators, each with its own grid or moment basis. Population
initial values are a named tuple and bulk initial values enter `PBMProblem`
through `bulk0`. Names determine identity even if initial tuples are reordered.
`population`, `bulk`, and `stateview` recover views from the packed solution.

Local laws access `ctx.bulk` and `ctx.system`; system callbacks access named
`ctx.populations`, `ctx.grids`, `ctx.fields`, `ctx.derivatives`, and `ctx.operators`.
`bulk_rhs` returns a named tuple of bulk derivatives. Prescribed fields are
operating histories; dynamical bulk quantities belong in the state vector.
Mark constrained bulk concentrations with `nonnegative_bulk=(:c,)`.

## Transfers and unlike-population events

`PopulationTransfer(:A,:B,rate; destination_size=..., multiplicity=1.0,
conserved_measure=identity, overflow=:error)` removes one source particle per
hazard event and deposits expected products on the target's pivots. Declare a
size map and multiplicity consistent with the chosen conserved measure. Target
and source grids may differ; out-of-range products require an explicit policy.

`CrossAggregation(:A,:B,:C,kernel; coordinate_map=nothing, overflow=:error)`
models ``A+B\to C``. Each unlike-type pair occurs once, without the identical-pair
half factor. Common additive coordinates or an explicit property map are
required. Both transfer and cross aggregation currently require 1D sectional
populations; moment interactions need an explicit `CustomCoupling`.

`CustomCoupling(f!)` adds to population derivative views and returns bulk
contributions. The callback must not reset contributions from other mechanisms.
It is responsible for its own physical and numerical conservation checks.

## Derive exchange from the same discrete residual

`ConservativeExchange(:particles,:c,weight; mechanisms=(:growth,:birth,:nucleation))`
subtracts the selected resolved particle-property rate from the bulk derivative.
It uses the selected term methods, including external policies. For a fixed
particle weight ``w(x)``, this aligns the discretization of ``c+\sum_i w(x_i)N_i``.
It does not establish continuum moment accuracy at a finite grid size.
A time- or state-dependent weight needs the additional derivative of that weight.

Open size boundaries are rejected for this exchange because an escaping-particle
reservoir must be accounted for separately. Collision truncation also needs an
explicit ledger; instantaneous diagnostics alone do not close a cumulative balance.
`MomentExchange(:particles,:c,k; factor=...)` uses the selected moment rate and
its state-to-moment chain rule. Its invariant assumes a fixed factor.

All of these rates use the same current stage state. Freezing solute while
advancing the distribution would define a different integration algorithm.
The standalone `Grid2D` backend is not currently a block in `PBMSystem`.
