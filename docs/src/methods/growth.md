# Growth and dissolution

Growth changes the size of an existing particle. Without boundary crossings,
it should change the distribution's shape and size moments while keeping its
total number fixed. This is why the implementation computes a common flux at
each face shared by neighbouring bins.

## Specify a particle velocity

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

## Convert velocity to a face flux

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

`VanLeer()` is a limited linear alternative. It uses the harmonic mean of the
left and right physical slopes when they have the same sign, and zero slope
otherwise. It also caps reconstructed endpoints at zero for nonnegative bin
averages. Both linear limiters use actual center distances on nonuniform grids;
neither makes an arbitrary time integrator positivity preserving.

`WENO3(epsilon=...)` combines two linear reconstructions using squared-slope
smoothness indicators. Its ideal weights are calculated from local cell widths,
so it supports uniform and nonuniform 1D grids. It is a third-order interior
finite-volume reconstruction, with first-order upwind fallback where a boundary
removes a required neighbor. Fixed epsilon, smooth extrema, mesh stretching,
and time integration affect observed convergence. It has no positivity limiter.
For both WENO choices, epsilon regularizes squared-density smoothness indicators;
rescaling a density changes its relative size. It is distinct from bin width
and from a physical nucleation spread.
This two-stencil option follows the WENO23 construction discussed by
[Zhang et al. (2024)](https://doi.org/10.1016/j.compchemeng.2024.108612);
it is distinct from the five-cell fifth-order option below.

`WENO5(epsilon=...)` reconstructs a face state from five cell averages using
nonlinear smoothness weights. It requires uniform grids with at least six bins.
The fifth-order statement applies to the smooth interior reconstruction. Near
nonperiodic boundaries the implementation falls back to upwind; temporal error
and boundaries can reduce the order of a full solve. There is no positivity
limiter for WENO here.

Choose these with `TermMethods(growth=MUSCL())`. The implementation is in
`src/numerics/sectional/transport.jl`; `prepare_term` checks eligibility and
`add_term!` adds the conservative flux differences.

Two unlimited alternatives are available on uniform **1D** grids:

| Choice | Density at the face, for positive growth |
|:--|:--|
| `Centered()` | ``(\bar n_i+\bar n_{i+1})/2`` |
| `LinearUpwind()` | ``(3\bar n_i-\bar n_{i-1})/2`` |

Here ``\bar n_i=N_i/\Delta x``. Negative growth reverses the upwind stencil.
LinearUpwind uses first-order upwind at an interior face where its second
upstream cell is unavailable. Both schemes conserve number through common
face fluxes, but neither limits oscillations or guarantees positivity.
Centered advection paired with forward Euler is unstable. LinearUpwind also
has a stricter step limit than first-order upwind; a Courant number of one is
unstable with the supplied SSPRK33 and RK4 integrators. `suggest_dt` is an
upwind/source heuristic and does not certify these combinations.

Select the reconstruction and integrator independently:

```julia
operator = compile(model,grid;methods=TermMethods(growth=LinearUpwind()),bc)
solution = simulate(PBMProblem(operator,N0,tspan);method=RK4(),dt=dt)
```

Lax–Wendroff and Beam–Warming instead include their own time update. They are
selected with `evolution=LaxWendroff()` or `evolution=BeamWarming()` during
compilation, and currently require homogeneous, constant, explicitly declared
1D growth. See [Time integration](time.md) for their formulas and eligibility.
The [constant-growth comparison](../case_studies/constant_growth.md) shows why
a higher time order alone does not remove spatial diffusion.

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

`Growth2D(rate=(Gx,Gy))` declares constant velocities;
`Growth2D(rate=(x,y,p,t,ctx)->(Gx,Gy))` uses a callback returning two finite real
velocities. `Grid2D` stores bin numbers over areas. Flux differences in each
direction are multiplied by the tangential bin width. Upwind and uniform-grid
WENO5 are supported; 2D MUSCL is not implemented.

The [two-dimensional growth calculation](../case_studies/transport.md) compares
variable-growth residuals with an analytical solution and InguvaBraatz2023.
The [trajectory methods](characteristics.md) explain the required growth-law
declarations, physical meshes and CFL=1 time steps for Inguva2022 and
InguvaBraatz2023.
