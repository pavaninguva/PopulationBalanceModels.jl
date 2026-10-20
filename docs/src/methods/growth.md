# Growth and dissolution

## Physical law and units

```math
\frac{\partial n(x,t)}{\partial t}
+\frac{\partial [G(x,t)n(x,t)]}{\partial x}=0.
```

``G`` is the rate of change of the internal coordinate of one particle. On a
length grid its units are length/time; on a volume grid they are volume/time.
It can be negative for dissolution. A conservative derivative is required:
``G\,\partial n/\partial x`` alone omits ``n\,\partial G/\partial x`` when growth
depends on size.

```julia
function growth_rate(L,p,t,ctx)
    s = max(ctx.bulk.c / ctx.fields.csat - 1, 0.0)
    p.kg * s^p.exponent * (1 + p.size_factor * L)
end
model = PBMModel(growth=Growth(rate=growth_rate))
```

`rate(x,p,t,ctx)` is evaluated at face coordinates using current stage states.
`p` holds model parameters; `ctx.bulk` holds shared dynamical concentrations;
`ctx.fields` holds prescribed operating conditions. The function returns velocity,
not a derivative vector or particle number flux. It should not modify the state.

## Sectional face flux

For bin numbers, growth contributes

```math
\frac{dN_i}{dt}=F_{i-1/2}-F_{i+1/2},\qquad
F_{i+1/2}=G(x_{i+1/2},t)\,n^{\mathrm{up}}_{i+1/2}.
```

The sign of the face velocity selects the upwind density. Interior fluxes cancel
when summing bin equations, so total number changes only through boundaries.
A size-weighted moment also changes because particles grow.

`Upwind()` reconstructs a constant density from each bin average and supports
nonuniform grids. It is first order in space and can diffuse narrow peaks.
`MUSCL()` reconstructs limited linear profiles with nonuniform-grid geometry in
1D. The implementation rescales slopes so reconstructed endpoints of a
nonnegative cell average remain nonnegative. A suitable time step is still needed.

`WENO5(epsilon=...)` reconstructs a face state from five cell averages using
nonlinear smoothness weights. It requires uniform grids with at least six bins.
The fifth-order statement applies to the smooth interior reconstruction. Near
nonperiodic boundaries the implementation falls back to upwind; temporal error
and boundaries can reduce the order of a full solve. There is no positivity
limiter for WENO here.

Choose these with `TermMethods(growth=MUSCL())`. The implementation is in
`src/numerics/sectional/transport.jl`; `prepare_term` checks eligibility and
`add_term!` adds the conservative flux differences.

## Size boundaries

`NoFlux()` sets the face flux to zero. `OpenBoundary()` allows outward flux and
assumes zero incoming density. `InflowFlux(rate)` prescribes inward number flux
independently of velocity; its units differ from those of `Growth.rate`.
`Periodic()` must be used at both ends and is mainly a transport diagnostic.

A positive velocity and a no-flux upper edge retain material in the final bin;
this is not the same as letting particles leave the size domain. Put the edge
far enough away or account for escaped material explicitly. A negative velocity
at zero size similarly needs a physical extinction/reservoir model if particles
dissolve away; a moment closure does not infer that boundary process.

## Two internal coordinates

```math
\frac{\partial n(x,y,t)}{\partial t}
+\frac{\partial [G_x(x,y,t)n(x,y,t)]}{\partial x}
+\frac{\partial [G_y(x,y,t)n(x,y,t)]}{\partial y}=0.
```

`Growth2D((Gx,Gy))` declares constant velocities; `Growth2D((x,y,p,t,ctx)->(Gx,Gy))`
uses a callback. `Grid2D` stores bin numbers over areas. Flux differences in each
direction are multiplied by the tangential bin width. Upwind and uniform-grid
WENO5 are supported; 2D MUSCL is not implemented.

The [Inguva–Braatz case](../case_studies/transport.md) compares variable-growth residuals
with an analytical solution and a case-specific transformed-mesh adapter. Direct methods
have additional boundary and constant-growth restrictions, described under
[Time integration](time.md).
