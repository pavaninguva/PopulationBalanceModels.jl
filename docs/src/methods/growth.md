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
nonlinear smoothness weights. It supports uniform and fixed nonuniform grids,
including logarithmic meshes, with at least six bins.
The fifth-order statement applies to the smooth interior reconstruction. Near
nonperiodic boundaries the implementation falls back to upwind; temporal error
and boundaries can reduce the order of a full solve. There is no positivity
limiter for WENO here.

### WENO on a nonuniform mesh

Use the same growth callback and `WENO5()` choice; only the bin edges change:

```julia
grid = Grid1D(exp.(range(log(1e-6),log(1e-3);length=161));coordinate=:length)
model = PBMModel(growth=Growth((L,p,t,ctx)->p.k*L))
operator = compile(model,grid;methods=TermMethods(growth=WENO5()),
    bc=SizeBC(OpenBoundary(),OpenBoundary()))
# Supply N0 as integrals over these bins, not samples at their centres.
problem = PBMProblem(operator,N0,(0.0,1.0);p=(k=0.1,))
solution = simulate(problem;method=RK4(),dt=0.001)
```

The reconstruction fits three quadratic polynomials to the **cell averages**
``N_i/\Delta x_i`` on overlapping three-bin stencils. The optimal linear
combination agrees with the five-bin quartic reconstruction at the face.
Both those weights and the polynomial coefficients use the actual edges.
The Jiang–Shu smoothness indicator is

```math
\beta_r=\sum_{\ell=1}^{2}(\Delta x_i)^{2\ell-1}
\int_{x_{i-1/2}}^{x_{i+1/2}}
\left[\frac{d^\ell p_r(x)}{dx^\ell}\right]^2\,dx,
\qquad
\omega_r=\frac{d_r/(\epsilon+\beta_r)^2}
{\sum_s d_s/(\epsilon+\beta_s)^2}.
```

Here `i` is the donor bin, selected by the sign of growth. Compilation prepares
the geometry for both signs and, for periodic boundaries, the wrapped cell
widths. Coefficients are calculated in a local scaled coordinate to avoid
conditioning problems caused solely by choosing metres rather than micrometres.
An unresolved polynomial solve or nonpositive optimal weight raises an error;
no other reconstruction is substituted. These are fixed-mesh coefficients:
recompile after changing edges. Changing kinetic parameters does not change
the geometry. Uniform grids retain the existing explicit formulas.
This follows the finite-volume construction of
[Wang, Feng and Spiteri (2008)](https://www.cs.usask.ca/~spiteri/nuweno.pdf).

Tests verify quadratic reconstruction, quartic optimal weights, the uniform
limit, length-unit rescaling, signed fluxes and fifth-order interior residuals
on smoothly graded meshes. A log-mesh characteristic example is available as
[a complete script](../assets/source/examples/nonuniform_weno.jl):

```sh
julia --project=. examples/nonuniform_weno.jl
```

For that smooth ``G(x)=x`` calculation, the measured integrated L1 errors at
64, 128 and 256 bins are ``1.11\times10^{-3}``, ``4.49\times10^{-5}`` and
``1.21\times10^{-6}`` (successive slopes 4.63 and 5.21). RK4 uses a local
Courant bound of 0.15. Number changes stay below ``1.5\times10^{-13}``;
the script reports small signed undershoots rather than hiding them. These
are full-solve measurements, separate from the residual-order tests.

Mesh support does not imply fifth-order convergence for every full calculation.
Strong grading, critical points, epsilon and lower-order boundary closures can
limit it. Small bins also restrict explicit time steps: use the actual local
widths and growth rates, rather than a mean spacing. `suggest_dt` remains a
heuristic, not a stability or positivity certificate. The conservative flux
difference and cumulative escape accounting are unchanged. There is no clipping
or added positivity limiter. In particular, sharp or almost-empty distributions
may fail the native stage checks; the diagnostic example explicitly reports
signed undershoots with those checks disabled.

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
direction are multiplied by the tangential bin width. Upwind and WENO5 on fixed
uniform or nonuniform rectangular axes are supported; 2D MUSCL is not implemented.
Directional fifth-order residual convergence is tested for constant
velocities. General cross-coordinate velocities are evaluated at transverse
cell centres: this face integration can limit overall accuracy to second order.
Nonuniform WENO does not add higher-order transverse quadrature.

The [two-dimensional growth calculation](../case_studies/transport.md) compares
variable-growth residuals with an analytical solution and InguvaBraatz2023.
The [trajectory methods](characteristics.md) explain the required growth-law
declarations, physical meshes and CFL=1 time steps for Inguva2022 and
InguvaBraatz2023.

`compile(...; evolution=HOCFiniteVolume(),bc,p)` selects a complete conservative
compact update instead of a residual reconstruction. Its positive-growth,
separability, source and boundary contracts, together with fourth-order spatial
and second-order temporal evidence, are specified in
[Compact-method verification](compact_verification.md). It rejects evolving
bulk feedback and signed dissolution; retain residual methods for those cases.

## Random-fluctuation growth-rate dispersion

`SizeDiffusion` describes independent, rapidly varying growth fluctuations.
The deterministic population equation uses the Ito/Fokker–Planck convention:

```math
\frac{\partial n(L,t)}{\partial t}
=-\frac{\partial [G(L,t)n(L,t)]}{\partial L}
+\frac{\partial^2[D(L,t)n(L,t)]}{\partial L^2},\qquad
F=G n-\frac{\partial(D n)}{\partial L}.
```

Its particle interpretation is ``dL_i=G\,dt+\sqrt{2D}\,dW_i``, with independent
fluctuations for different particles. This Brownian description is a
coarse-grained closure; the package solves the deterministic PDE, not sampled
trajectories. It differs from a permanent crystal-specific growth multiplier.
Boundaries enforce the allowed coordinate support. Continuous nucleation mixes
different particle ages, so diffusion does not imply a Gaussian full population.

Supply a constant or `D(L,p,t,ctx)` in coordinate²/time. Callbacks must be pure;
they receive the current stage's time, parameters, prescribed fields and shared
`ctx.bulk`. Negative or nonfinite coefficients raise `DomainError`; zero adds
no diffusion. No constitutive relation is imposed. One optional relation is
``D=\theta G^2``, where ``\theta`` has time units and combines relative fluctuation
variance with correlation time. Physical diffusion is separate from numerical
smearing introduced by a growth reconstruction.

```@example grd
using PopulationBalanceModels
grid = Grid1D(range(0,8;length=161);coordinate=:length)
model = PBMModel(
    growth=Growth((L,p,t,ctx)->p.G),
    dispersion=SizeDiffusion((L,p,t,ctx)->p.D),
)
operator = compile(model,grid;methods=TermMethods(growth=MUSCL()))
# Integral of a smooth, localized initial density, rather than centre samples.
using SpecialFunctions: erf
cdf(L) = (1+erf((L-3)/sqrt(2*0.25)))/2
N0 = diff(cdf.(grid.edges))
p = (G=0.2,D=0.08)
problem = PBMProblem(operator,N0,(0.0,0.5);p)
dt = suggest_dt(operator,problem.u0,p,0.0;cfl=0.4)
solution = simulate(problem;dt,method=SSPRK33())
sum(solution.u[end])-sum(N0) # closed-boundary number balance
```

The default `FiniteVolumeDiffusion()` evaluates ``D_i`` at bin centres and adds
equal and opposite contributions from every face:

```math
q_{i+1/2}=-\frac{D_{i+1}N_{i+1}/w_{i+1}-D_iN_i/w_i}{x_{i+1}-x_i},\qquad
\frac{dN_i}{dt}\mathrel{+}=q_{i-1/2}-q_{i+1/2}.
```

Select it explicitly with `TermMethods(size_diffusion=FiniteVolumeDiffusion())`
or a named `by_term` override. Geometry is prepared once; coefficients are
reevaluated at every residual call. The flux differences ``D n``, **not** ``n``
times an averaged D. For size-dependent D it therefore differs from Fickian
``\partial_L(D\,\partial_L n)``. In particular, a reflecting diffusion-only
stationary state has ``n_i\propto1/D_i``, rather than uniform density.

| Configuration | Support |
|:--|:--|
| Standalone or named `PBMSystem` population | Residual-based `Grid1D` bin integrals; growth, sources and supported local terms compose normally |
| Uniform / nonuniform meshes | Conservative fluxes on either; second-order spatial consistency verified on uniform and smoothly graded meshes, not arbitrary grading |
| `SizeBC()` | Zero external **total** flux; for diffusion alone, ``\partial_L(Dn)=0`` rather than ``\partial_L n=0`` |
| Paired `Periodic()` | Conservative wrapped face using ``(w_1+w_m)/2`` as centre distance; a mathematical test boundary, not physical size recycling |
| `BoundaryInjection(:left)` with `Nucleation(J,first(grid.edges))` | Adds ``F_{left}=J`` once, with or without Growth; right injection analogously sets ``F_{right}=-J`` |
| Native Euler / SSPRK22 / SSPRK33 / SSPRK54 / RK4 | Same residual and stage checks; user supplies dt. RK4 has no SSP positivity guarantee |
| Optional `odeproblem` | Same residual; no native positivity checks or automatic diffusion step bound |
| MomentSet, Grid2D, axial adapters, MeasureEuler and direct characteristic/compact updates | Rejected; no diffusion rule is implemented for these combinations |

For closed boundaries the growth and diffusion base exterior fluxes are zero;
an explicitly selected boundary source replaces the prescribed total exterior
flux by J. It does not also impose ``n=J/G`` or a second boundary density.
`OpenBoundary`, `AbsorbingZero` and `InflowFlux` are rejected with SizeDiffusion:
their advection meaning alone does not define this second-order boundary problem.
There is no diffusion escape ledger under the supported closed/periodic policies.

The explicit recommendation adds outgoing diffusion, growth and loss rates.
For diffusion, ``r_i=(D_i/w_i)\sum_{faces}1/h_{face}``, omitting closed exterior
faces. Thus a uniform interior cell with constant positive growth contributes
``G/\Delta L+2D/\Delta L^2``. Choosing the minimum of separate growth and diffusion
steps misses this sum. `suggest_dt` returns a local recommendation, not an
adaptive controller: later coefficient changes, bulk depletion and high-order
growth reconstructions still need checking. RK4 intermediate stages can fail
even below a forward-Euler bound; no checks are disabled or populations clipped.

For constant G and D away from boundaries, ``dM_3/dt=3GM_2+6DM_1``.
Number-conserving diffusion can therefore change particle material. The default
[`ConservativeExchange`](../coupled_systems.md) includes `:size_diffusion` and
uses the **actual discrete** weighted residual. Do not add an analytical
``6DM_1`` sink on top of this exchange. Continuum moment accuracy needs mesh
refinement separately from the exact discrete material balance.

The [synthetic broadening example](../assets/source/examples/random_fluctuation_grd.jl)
uses exact Gaussian bin integrals and reports number, mean, variance and L1 error:

```sh
julia --project=. examples/random_fluctuation_grd.jl
julia --project=. -e 'using Test, PopulationBalanceModels; include("test/size_diffusion.jl")'
```

The focused tests also cover a current-stage bulk-dependent diffusivity,
independent temporal refinement, nonuniform periodic geometry, boundary births,
discrete material exchange and ForwardDiff derivatives. As elsewhere in the
residual API, promote `u0` to the parameter's scalar type when differentiating
a forward solve; no event sensitivities or automatic solver choice are added.
In the Gaussian example, integrated L1 error falls from ``8.37\times10^{-4}``
at 120 bins to ``2.29\times10^{-4}`` at 240 and ``6.22\times10^{-5}`` at 480;
the variance approaches ``0.25+2(0.08)(0.5)=0.33``. The separate smoothly graded
heat-equation test gives slopes 1.98 and 2.00, while time refinement against a
fixed semidiscrete eigenmode gives SSPRK33 slopes 3.05 and 3.03. These tests
separate physical broadening, spatial error and time error.

## Disappearance at physical zero

Choose `SizeBC(AbsorbingZero(),NoFlux())` for a sectional zero-size outflow, or
`MomentBoundary(left=:absorb)` and native `MeasureEuler()` for supported atomic
moment characteristics. A positive computational cutoff uses escape, carrying
remaining solid material, rather than zero-size extinction. See the
[boundary and material contract](realizable_evolution.md).

A size-independent rate may be declared as
`Growth(PolynomialRate(((p,t,ctx) -> rate(p,t,ctx),)))`.
The sectional growth-flux assembly evaluates this degree-zero coefficient once,
using the current stage's bulk state and time. This avoids repeating a costly
kinetic law at every face. Ordinary callbacks remain supported and are evaluated
at each face; no spatial independence is inferred by sampling. Rate callbacks
must be pure.
