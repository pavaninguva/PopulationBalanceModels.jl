# Defining models

## Choose a coordinate and concentration basis

For a one-coordinate density ``n(x,t)``, ``n\,dx`` is particle number per reference
volume (or another explicitly chosen process basis). If ``x`` has units X and
time has units T, ``n`` has units number/(reference volume X). The local equation is

```math
\partial_t n+\partial_x(Gn)=B-dn+J\delta(x-x_c)+\mathcal A[n]+\mathcal F[n].
```

`Grid1D(edges; coordinate=:volume)` stores edges, widths, and center pivots.
Other coordinate labels include mass, length, and diameter. Labels do not perform
unit conversion. Supply all rates, boundaries, grids, and bulk states in a
consistent unit system. Changing from diameter to volume requires the density
Jacobian as well as transformed growth and collision laws.

The sectional state is ``N_i=\int_{e_i}^{e_{i+1}}n(x)\,dx``. For a sampled smooth
initial density, `n0.(grid.centers) .* grid.widths` is a midpoint approximation.
Use actual bin integrals when available. `density(grid,N)` divides by widths;
`moment(grid,N,k)` uses pivot weights ``\sum_i x_i^k N_i``. Nonlinear moments
therefore have quadrature error even when bin numbers are exact.

`MomentSet(0:K; coordinate=:volume)` stores raw moments instead of bins.
`Grid2D` stores bin numbers over two internal coordinates. These are distinct
representations; see [Moment methods](moment_methods.md) and
[Two-coordinate transport](case_studies/transport.md).

## Declare named physical mechanisms

```julia
model = PBMModel(
    growth = Growth((x,p,t,ctx) -> p.G),
    primary = Nucleation((p,t,ctx) -> p.J, 0.0),
    loss = Death((x,p,t,ctx) -> p.k),
)
```

Growth callbacks return ``dx/dt`` (X/T). Birth callbacks return density per time;
death and dilution return nonnegative hazards (1/T). Nucleation callbacks return
total number creation per reference volume per time. Aggregation kernels have
reference-volume/time units when particle number is dimensionless. Breakage
callbacks return a parent hazard and expected daughter counts, including any
quadrature weights. [Numerical methods](numerical_methods.md) explains their
conservative residuals and selectable policies.

`PBMModel(term1 + term2)` is a concise alternative that assigns mechanism-based
names. Explicit names are useful when several mechanisms of the same class need
different methods. Callbacks receive parameters `p`, time `t`, and the current
stage context `ctx`. A sectional local context exposes `grid`, `N`, `cell`, and
prescribed `fields`. In coupled models it also exposes `bulk` and `system`.
For laws intended to work across representations, use `moment(ctx,k)` rather
than indexing the stored state directly.

## Compile boundaries and methods

`compile(model,grid; methods=TermMethods(), bc=SizeBC())` checks compatibility
and prepares each term. The result is an operator; `PBMProblem(operator,u0,tspan; p)`
adds initial data and parameters. `simulate(problem; dt=...)` selects time evolution.
Changing a reconstruction does not change the physical law.

`NoFlux()` suppresses the face flux. `OpenBoundary()` allows outward growth
flux and assumes zero incoming distribution. `Periodic()` must be used at both
ends; it is useful for transport diagnostics and usually inappropriate for
physical particle size. Periodic size boundaries cannot be combined with collisions.
`SizeBC(left,right)` defaults to no flux at both ends.

`InflowFlux((p,t,ctx)->J)` prescribes inward number flux and requires a Growth
term. It avoids evaluating ``n=J/G`` when growth vanishes. A point source can
instead select `BoundaryInjection(:left)` when its location equals that edge;
active injection requires inward or zero velocity. Do not specify the same
source as both a boundary flux and a nucleation term. The finite first pivot
gives incoming particles a small positive discrete size even at a zero-size edge.

Only `BatchDomain` is compiled. `AbstractPopulationDomain` leaves room for future
spatial adapters without placing spatial file formats in physical laws.
