# Coupled populations

A process can contain several populations that share concentrations or other
bulk states. Each distribution has its own balance, but its rates may depend on a
common environment. Transfers between populations must remove the
source entities and create the corresponding destination entities in the
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

The coupled backend accepts named Grid1D, Grid2D and supported moment
populations. One- and two-coordinate populations may share the same bulk states.

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
is ``dM_3/dt=3GM_2+6DM_1``. Under suitable mesh/domain refinement, the discrete
moment rate should approach this continuum expression. Exact discrete inventory
conservation is a separate check.

For example, `SizeDiffusion((L,p,t,ctx)->p.D*ctx.bulk.c/p.c_ref)` uses the current
stage's dissolved concentration. `suggest_dt(problem.operator,u,p,t)` accepts
packed sectional system states and uses that same context. It bounds local
population growth/diffusion/loss rates, not bulk depletion or interactions;
systems with moments or inter-population interactions need a separate bound.

For eligible **Grid1D** populations, open-boundary exchange requires compatible
`CumulativeAccounting`, so escaped solid material is kept separate from dissolved
solute. It rejects without that accounting. Property weights must satisfy the
[accounting contract](methods/realizable_evolution.md); inter-population
interactions cannot be combined with these ledgers. **Grid2D** exchange requires
closed/periodic boundaries because tensor escape accounting is unavailable.
Collision truncation likewise needs an explicit ledger; instantaneous rates
alone do not close a cumulative balance.
`MomentExchange(:particles,:c,k; factor=...)` uses the selected moment rate and
its state-to-moment chain rule. Its invariant assumes a fixed factor.

## Two internal coordinates and growth dispersion

For a rectangular Grid2D, each state entry is an integrated count over an
internal-coordinate cell. `population(problem,u,:crystals)` and
`ctx.derivatives.crystals` are matrices with `size(grid)`. Other populations
retain their vector layouts; the full solver state remains a packed vector.
Use `PopulationMoment(:crystals,(kx,ky))` for a joint moment. A marginal
measurement requires an explicit `SystemObservable`; the one-coordinate
histogram/CDF constructors do not choose a marginal for you.

Here the two coordinate velocities and diagonal diffusivities depend on the
current solute concentration. The exchanged property is proportional to
``x y^2``, a possible fixed-shape volume convention:

```@example coupled_tensor
using PopulationBalanceModels
grid = Grid2D(0.0:0.2:4.0, 0.0:0.2:3.0)
model = PBMModel(
    growth=Growth2D((x,y,p,t,ctx)->(p.gx*ctx.bulk.c,p.gy*ctx.bulk.c)),
    dispersion=SizeDiffusion((xy,p,t,ctx)->(p.Dx*ctx.bulk.c,p.Dy*ctx.bulk.c)),
)
operator = compile(model,grid) # Upwind growth and finite-volume diffusion
system = PBMSystem((crystals=operator,);
    exchanges=(ConservativeExchange(:crystals,:c,(xy,p,t,ctx)->p.alpha*xy[1]*xy[2]^2),),
    nonnegative_bulk=(:c,))
N0 = zeros(size(grid)); N0[8,6] = 1.0 # a bin-integrated initial population
p = (gx=0.1,gy=0.05,Dx=0.02,Dy=0.01,alpha=0.001)
problem = PBMProblem(system,(crystals=N0,),(0.0,0.05);bulk0=(c=1.0,),p)
inventory(u) = bulk(problem,u,:c)+p.alpha*moment(grid,population(problem,u,:crystals),(1,2))
total = inventory(problem.u0)
upper = copy(problem.u0); upper[end] = total # c <= total for nonnegative solids
dt = suggest_dt(problem.operator,upper,p,0.0;cfl=0.3)
solution = simulate(problem;dt)
@assert maximum(abs(inventory(u)-total) for u in solution.u) < 1e-12
(shape=size(population(problem,last(solution.u),:crystals)),
 material_error=inventory(last(solution.u))-total)
```

The closed-domain discrete invariant is
``c+\alpha\sum_{i,j}x_i y_j^2N_{ij}``, including diffusion's material change.
With spatially constant coefficients and negligible boundary traces, its
continuum particle-property rate is
``G_x M_{02}+2G_y M_{11}+2D_y M_{10}``. Upwind also introduces mesh-dependent
smearing; agreement with this continuum rate needs separate refinement.
The coefficient bounds used above follow from the nonnegative inventory
and the declared monotone concentration dependence; `suggest_dt` itself does
not predict future coefficients or bound bulk depletion.

To observe a joint moment or an x marginal from the same solution:

```@example coupled_tensor
joint = observe(PopulationMoment(:crystals,(1,2)),solution)
x_bins = SystemObservable((state,op,p,t)->
    vec(sum(state.populations.crystals;dims=2)))
Nx = observe(x_bins,solution) # x-bin integrals, summed over y
nx = Nx ./ grid.x.widths     # density per unit x
@assert sum(Nx) ≈ sum(population(problem,last(solution.u),:crystals))
(joint_moment=only(joint),number=sum(Nx))
```

The tensor counts already include both cell widths. Summing over y therefore
needs **no additional** ``\Delta y_j``. A standalone tensor uses
`FunctionObservable((N,op,p,t)->vec(sum(N;dims=2)))` instead. Conditional CDFs
can then be defined explicitly on the marginal's Grid1D; no constructor guesses
a marginal or a measurement weighting.

Tensor `ConservativeExchange` requires closed/periodic size boundaries.
Tensor escape ledgers, zero-size extinction and tensor collision/transfer
algorithms are unavailable. Tensor populations without SizeDiffusion can also
use `axial_problem`; intrinsic SizeDiffusion still rejects axial adapters.
The existing [GRD example](assets/source/examples/random_fluctuation_grd.jl)
has a smooth two-coordinate version:

```sh
julia --project=. examples/random_fluctuation_grd.jl --two-coordinate
```

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
view across reactor cells. Historical entries are **not advected or dispersed**;
they still accumulate the local removal in time. Stationary root problems
exclude these historical unknowns and expose instantaneous accounting rates.
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
