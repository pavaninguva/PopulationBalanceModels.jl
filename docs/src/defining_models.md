# Defining models

A population balance starts with an internal coordinate and a statement
of what changes the entities in the population. The coordinate might be crystal
size, cell-marker content, age or another modeled property. The numerical grid comes later.
This page explains how those choices enter PBMModel and how parameters and
boundary conditions reach the rate callbacks.

## Choose a coordinate and units

For a density ``n(x,t)``, ``n(x,t)\,dx`` is number concentration in a size
interval. If ``x`` has units X, the density has units number/(process volume X).
A process basis other than volume is possible, but it must be used consistently
in the population and bulk balances.

`Grid1D(edges;coordinate=:length)` labels a length coordinate. Other common
labels are `:volume`, `:mass`, and `:diameter`. A label performs no unit conversion.
Changing a model from diameter to volume requires transforming the density,
growth law, and collision products, not just relabelling an axis.

A direct PDE discretization on `Grid1D` stores integrated bin populations,
also called the *sectional* representation:

```math
N_i(t)=\int_{x_{i-1/2}}^{x_{i+1/2}}n(x,t)\,dx.
```

For a smooth initial density, `n0.(grid.centers) .* grid.widths` approximates
these integrals by midpoint quadrature. Integrate discontinuities and narrow
peaks more carefully. `density(grid,N)` divides by widths, while
`moment(grid,N,k)` computes ``\sum_i x_i^kN_i`` using centre pivots. Nonlinear
moments can therefore have quadrature error even when bin numbers are exact.

A [MomentSet](moment_methods.md) declares the tracked moment basis. Raw-moment
methods evolve those moments; DQMOM evolves weights and weighted abscissas and
recovers the moments through `moment_value`. Select the family with
`moment_method=QMOM()` during moment compilation; omitted node counts are inferred
from the requested basis. The basis alone does not prescribe
the solver-state layout. [Grid2D](case_studies/transport.md) instead stores
integrated numbers over rectangular cells in two internal coordinates.

## Translate a physical term into a callback

Consider growth with a size-dependent velocity and removal with hazard ``k``:

```math
\frac{\partial n(L,t)}{\partial t}
+\frac{\partial[G(L,t)n(L,t)]}{\partial L}=-k n(L,t),
\qquad G(L,t)=g_0(1+\alpha L).
```

The code supplies the velocity and hazard separately:

```@example model_definition
using PopulationBalanceModels

function growth_rate(L,p,t,ctx)
    p.g0 * (1 + p.alpha*L)
end
removal_rate(L,p,t,ctx) = p.k

model = PBMModel(
    growth = Growth(rate=growth_rate),
    loss = Death(rate=removal_rate),
)
nothing # hide
```

The callback returns a physical rate; it does not assemble a face flux or
modify a derivative vector. Growth may be negative for dissolution. Death is
a nonnegative hazard. Representative callback contracts are:

| Declaration | Callback result |
|:--|:--|
| `Growth(rate=G)` | `G(x,p,t,ctx)`: coordinate/time |
| `Growth2D(rate=G)` | `G(x,y,p,t,ctx)`: pair of coordinate velocities |
| `SizeDiffusion(D)` | `D(x,p,t,ctx)` in 1D, or `D(xy,p,t,ctx)` returning `(Dx,Dy)` in 2D: coordinate²/time |
| `Nucleation(rate=J,size=xc)` | `J(p,t,ctx)`: number/(process volume time) at a fixed size |
| `Birth(source=B)` | `B(x,p,t,ctx)`: number/(process volume coordinate time) |
| `Death(rate=d)` or `Dilution(rate=d)` | `d(x,p,t,ctx)`: inverse-time hazard |
| `Aggregation(K)` | `K(x,y,p,t,ctx)`: process volume/time for number concentrations |
| `Breakage(a,daughters)` | parent hazard and expected daughter counts, including quadrature weights |

Distributed/integrated source constructors and moment sources have their own
contracts in the [term-specific numerical pages](numerical_methods.md).
Keywords such as `rate`, `source` and `size` belong to those constructors;
PBMModel's outer keyword names are user-selected labels, not a fixed catalogue.

## Keep parameters and changing states distinct

Parameters are attached to the problem:

```@example model_definition
grid = Grid1D(0.0:0.1:10.0;coordinate=:length)
operator = compile(model,grid;methods=TermMethods(growth=MUSCL()))
p = (g0=0.2,alpha=0.1,k=0.02)
problem = PBMProblem(operator,zeros(length(grid)),(0.0,1.0);p)
changed = remake(problem;p=merge(problem.p,(g0=0.3,)))
nothing # hide
```

`remake` keeps the operator and initial state while changing the supplied
keywords. It does not recompute an initial condition that depends on a parameter.
Build that state again if it should change too.

A rate may also depend on values that evolve during integration. For example,
the [crystallization calculation](case_studies/crystallization.md) uses
`ctx.bulk.c` for dissolved concentration and `ctx.fields.T` for a prescribed
cooling temperature. In a single-population compile, fields have signature
`fields(cell,t)`; a PBMSystem instead supplies shared fields through
`fields(p,t)`. Bulk states belong in PBMProblem's `bulk0` and are solved as ODEs.

For a 1D population, `ctx.N` is the local state and `ctx.grid` its representation.
Use `moment(ctx,k)` to read a moment without assuming a bin or moment-state
layout. In a coupled system, `moment(ctx.system,:crystals,3)` accesses a named
population. The two-coordinate context supports `moment(ctx,(kx,ky))`.
Callbacks read the current Runge–Kutta stage state and must not mutate it.

## Name terms when choosing different treatments

Names on the left of PBMModel's keywords identify individual terms. Types on
the right identify the physical mechanism:

```julia
model = PBMModel(
    primary = Nucleation(rate=primary_rate,size=0.0),
    secondary = Nucleation(rate=secondary_rate,size=0.8),
    growth = Growth(rate=growth_rate),
)
methods = TermMethods(
    growth=MUSCL(),
    by_term=(primary=BoundaryInjection(:left),secondary=FixedPivot()),
)
```

Here both named sources are nucleation, but one is a boundary influx and the
other is an interior point source. `PBMModel(term1 + term2)` also works; it assigns
names automatically. Explicit names make individual method choices easier to read.

## Decide what crosses a size boundary

For one coordinate, `SizeBC(left,right)` specifies the two exterior transport faces. Its default is
NoFlux at both ends.

| Boundary | Meaning |
|:--|:--|
| `NoFlux()` | Set transport flux to zero; particles cannot leave through this face |
| `OpenBoundary()` | Allow outward transport; assume zero incoming density |
| `AbsorbingZero()` | Left edge at physical zero; disappearing particles carry zero positive-power material |
| `InflowFlux(J)` | Prescribe inward number flux `J(p,t,ctx)`, independently of velocity |
| `Periodic()` | Wrap transport between paired endpoints; use at both ends |

These are advection boundary meanings. At positive-growth outflow,
`OpenBoundary()` uses the outgoing numerical trace; it imposes no additional
zero-gradient condition on the density or on `G*n`. For growth plus diffusion,
the [GRD contract](methods/dispersion.md) defines the **total** exterior flux.
For positive growth, a closed upper edge retains particles in the last bin.
An open edge lets them leave the represented population. Set the range and
material accounting accordingly. Periodic boundaries are useful for numerical
tests but are generally inappropriate for physical particle size and cannot be
combined with sectional collisions.

A bare `InflowFlux` boundary needs Growth in the model. A zero-size nucleation
term can instead select BoundaryInjection at the matching edge; without size
diffusion, active injection requires inward or zero growth velocity.
With [`SizeDiffusion`](methods/dispersion.md),
BoundaryInjection specifies the total incoming number flux even without Growth.
SizeDiffusion currently requires closed or paired periodic base boundaries.
Choose one declaration for the same source. These flux forms
avoid dividing a prescribed flux by a possibly vanishing growth velocity.
The first positive pivot still assigns incoming particles a finite discrete size.

For two coordinates, `SizeBC2D(x=...,y=...)` specifies each axis. Advection's
`InflowFlux` is per tangential coordinate, integrated over the corresponding
face width. This is distinct from point nucleation: tensor boundary-point
injection is unavailable. Tensor diffusion accepts closed or paired-periodic
conditions independently per axis, with no prescribed inflow diffusion closure.
`BatchDomain()` describes a well-mixed population. A separate
[AxialDispersion domain](axial_reactors.md) adds a physical axial mesh, inlet,
and outlet to a one- or two-coordinate sectional population. Intrinsic size
boundaries and reactor boundaries are specified independently.

For raw moments, `MomentBoundary(left=:absorb)` declares atomic node removal
with native `MeasureEuler`; `:escape` instead removes particles at a finite
computational cutoff and retains their remaining material in a selected ledger.
A finite moment vector does not determine a continuum boundary trace. See the
[explicit boundary and support contract](methods/realizable_evolution.md).
