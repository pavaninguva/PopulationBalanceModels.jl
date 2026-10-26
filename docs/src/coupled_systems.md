# Coupled populations

A process can contain several particle populations that share the same liquid
phase. Each distribution has its own balance, but its rates may depend on a
common solute concentration. Transfers between populations must remove the
source particles and create the corresponding destination particles in the
same calculation.

Consider precursors created from solute at size ``v_c=2``. They become products
with hazard ``h``, retaining their size. For an additive volume coordinate,

```math
\frac{\partial n_A(v,t)}{\partial t}
=k c(t)\delta(v-v_c)-h n_A(v,t),
```

```math
\frac{\partial n_B(v,t)}{\partial t}=h n_A(v,t),\qquad
\frac{dc(t)}{dt}=-v_c k c(t).
```

This is a simple transfer model: each newly created precursor consumes ``v_c``
units of solute, while conversion to product moves existing material. The total
``c+\int_0^\infty v[n_A(v,t)+n_B(v,t)]\,dv`` should stay fixed.

## Compile each population

```@example coupled
using PopulationBalanceModels

grid = Grid1D(0.5:1.0:10.5;coordinate=:volume)
precursors = compile(PBMModel(
    primary=Nucleation(rate=(p,t,ctx)->p.k*ctx.bulk.c,size=2.0)),grid)
products = compile(PBMModel(),grid)
nothing # hide
```

The precursor callback reads the **current** shared solute concentration. The
product operator is empty because this population changes only through transfer.
The populations could use different grids; here matching grids make the material
balance especially easy to inspect.

## Add transfer and solute exchange

```@example coupled
system = PBMSystem((precursors=precursors,products=products);
    interactions=(PopulationTransfer(:precursors,:products,
        (size,p,t,ctx)->p.transfer),),
    exchanges=(ConservativeExchange(:precursors,:c,
        (size,p,t,ctx)->size),),
    nonnegative_bulk=(:c,))
nothing # hide
```

PopulationTransfer removes precursors and deposits products. ConservativeExchange
subtracts the material created by nucleation from solute. Transfer does not
consume solute again: it conserves material between the two particle populations.

## Initialize and advance the shared state

```@example coupled
problem = PBMProblem(system,
    (precursors=zeros(length(grid)),products=zeros(length(grid))),
    (0.0,3.0);bulk0=(c=5.0,),p=(k=0.1,transfer=0.3))
solution = simulate(problem;dt=0.01,saveat=0.0:0.25:3.0)

final = solution.u[end]
A = population(problem,final,:precursors)
B = population(problem,final,:products)
c = bulk(problem,final,:c)
@assert abs(c+moment(grid,A,1)+moment(grid,B,1)-5.0) < 1e-12
(precursor_number=sum(A),product_number=sum(B),solute=c)
```

PBMSystem packs the populations and concentration into one solver vector. At
every integration stage it evaluates their derivatives from the same state.
`population`, `bulk`, and `stateview` recover named views from that vector.
Freezing concentration during a particle update would define a different method.

The coupled backend accepts named 1D sectional and moment populations. The
standalone Grid2D transport operator is not yet a coupled-system population block.

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

`ConservativeExchange(:particles,:c,weight; mechanisms=(:growth,:size_diffusion,:birth,:nucleation))`
subtracts the selected resolved particle-property rate from the bulk derivative.
It uses the selected term methods, including external policies. For a fixed
particle weight ``w(x)``, this aligns the discretization of ``c+\sum_i w(x_i)N_i``.
It does not establish continuum moment accuracy at a finite grid size.
A time- or state-dependent weight needs the additional derivative of that weight.

Size diffusion preserves number but generally changes material moments. Its
contribution is included by default; an explicitly supplied `mechanisms` tuple
is preserved. For example, with length-based bin counts and
`weight=(L,p,t,ctx)->rho_kv*L^3`, the closed-system invariant is
``c+\rho k_v\sum_i L_i^3N_i`` per unit process volume. Use this exchange **or**
the corresponding contribution in `bulk_rhs`, never both. Diffusion-only
evolution may take material from or return it to the bulk. This is exchange,
not particle escape. Reflecting/periodic diffusion has no external removal.
For constant G and D with negligible boundaries, the separate continuum check
is ``dM_3/dt=3GM_2+6DM_1``; it need only converge to the discrete rate.

For example, `SizeDiffusion((L,p,t,ctx)->p.D*ctx.bulk.c/p.c_ref)` uses the current
stage's dissolved concentration. `suggest_dt(problem.operator,u,p,t)` accepts
packed sectional system states and uses that same context. It bounds local
population growth/diffusion/loss rates, not bulk depletion or interactions;
systems with moments or inter-population interactions need a separate bound.

Open size boundaries are rejected for this exchange because an escaping-particle
reservoir must be accounted for separately. Collision truncation also needs an
explicit ledger; instantaneous diagnostics alone do not close a cumulative balance.
`MomentExchange(:particles,:c,k; factor=...)` uses the selected moment rate and
its state-to-moment chain rule. Its invariant assumes a fixed factor.

All of these rates use the same current stage state. Freezing solute while
advancing the distribution would define a different integration algorithm.
The standalone `Grid2D` backend is not currently a block in `PBMSystem`.

## Account for removed solids separately from dissolved solute

`CumulativeAccounting(:crystals;properties=(volume=3,))` appends historical
number and volume losses to a `PBMSystem`. With a compatible exchange, growth
returns shrinkage material to solute and sends computational-boundary material
to its escape ledger. It never assigns that same escaped material to both.
Use `cumulative(problem,u,:crystals,:upper_escape,:volume)` to read a ledger and
`accounting_rates(problem,u,p,t)` for instantaneous sectional loss rates.
The optional final `property` argument defaults to `:number`; other names must
be declared in `properties`. Unknown population, mechanism and property names
raise `ArgumentError` listing the available names. An axial accessor returns a
view across reactor cells; those historical entries remain stationary.
Accounting adds `9 * (1 + length(properties))` states per declared population,
including `:nucleation_underflow` and `:nucleation_overflow`. Use named accessors
instead of assumptions about the packed offsets or total state length.
Old packed states require repacking against the new operator. Restore named
populations, bulk values and matching histories; the added birth-loss histories
cannot be inferred from an old saved state that never recorded them.
[The accounting contract](methods/realizable_evolution.md) defines units,
endpoint-pivot weighting, collision restrictions and the discrete invariant.
The executable `examples/realizable_dissolution.jl` verifies both representations
with a shared solute, separately from numerical reconstruction defects.
