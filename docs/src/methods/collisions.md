# Aggregation and breakage

## Aggregation events

For an additive coordinate ``x``, a symmetric binary kernel gives

```math
\frac{\partial n(x,t)}{\partial t}
=\frac{1}{2}\int_0^x K(x-y,y,t)n(x-y,t)n(y,t)\,dy
-n(x,t)\int_0^\infty K(x,y,t)n(y,t)\,dy.
```

`Aggregation(kernel; coordinate_map=nothing, overflow=:error)` receives
`kernel(x,y,p,t,ctx)`. The kernel must be symmetric and nonnegative. For number
concentrations its units are process volume/time. Same-bin event rates contain
``K_{ii}N_i^2/2`` and remove two particles. Distinct-bin events occur at
``K_{ij}N_iN_j`` and remove one particle from each parent bin.

FixedPivot is the implemented sectional algorithm. Products are interpolated
between pivots in an additive property ``q(x)``. The weights satisfy

```math
w_l+w_r=1,\qquad w_lq(x_l)+w_rq(x_r)=q(x_i)+q(x_j).
```

This preserves one product per event and its additive property. On a diameter
grid, `AdditiveCoordinate(L->L^3,cbrt)` produces
``L_*=(L_i^3+L_j^3)^{1/3}`` and conserves volume. Adding diameters would conserve
the wrong quantity. The compiler checks an increasing positive property map
and its inverse at the grid pivots.

## Breakage events

```math
\frac{\partial n(x,t)}{\partial t}
=\int_x^\infty a(y,t)b(x\mid y,t)n(y,t)\,dy-a(x,t)n(x,t).
```

``a`` is a parent hazard in inverse time; ``b(x\mid y,t)`` is an expected daughter
count density, not a probability density alone. Conservation requires
``\int q(x)b(x\mid y,t)\,dx=q(y)``.

`Breakage(rate,daughters; coordinate_map=...,overflow=...)` receives
`rate(x,p,t,ctx)` and `daughters(x,p,t,ctx)`, the latter returning
`(daughter_size,expected_number)` pairs. Equal binary splitting in an additive
coordinate is `((x/2,2.0),)`. A continuous daughter approximation must include
quadrature weights and multiplicity in the expected counts. FixedPivot deposits
those daughters and checks their additive-property sum against the parent.

## Boundaries, overflow, and conservation

Aggregation lowers total particle number; binary breakage raises it. The conserved
quantity is the additive property, provided products remain represented.
Products outside the pivot range raise an error by default.
`FixedPivot(overflow=:discard)` makes unresolved loss explicit; `:inherit` uses
the physical term's legacy overflow setting. `diagnostics` reports instantaneous
unresolved birth number and property rates, not cumulative reservoirs.

Periodic size boundaries are incompatible with collisions. Use an adequate size
range or integrate an explicit escaped-product reservoir before claiming a closed
physical balance. The current quadratic pair calculation is transparent but can
be costly on large grids. Alternative sectional collision algorithms need their
own `prepare_term` and `add_term!` implementations; the interface is not evidence
that those algorithms are provided.

The implementation is in `src/numerics/sectional/collisions.jl` and
`deposition.jl`. See [Coupled systems](../coupled_systems.md) for unlike-population
collisions, which do not use the identical-pair half factor, and
[Moment methods](../moment_methods.md) for weak-form collision integration.
