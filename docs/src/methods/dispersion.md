# Internal-coordinate diffusion and GRD

`SizeDiffusion` adds rapidly varying growth fluctuations to a population model.
It also works without a mean-growth term.
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

| Representation / choice | Implemented contract |
|:--|:--|
| Batch Grid1D | Bin-integrated counts on a fixed uniform or nonuniform grid; standalone or named in PBMSystem |
| Grid1D boundaries | `SizeBC(NoFlux(),NoFlux())` or paired `Periodic()`; NoFlux prescribes zero external **total** flux |
| Grid1D boundary births | `Nucleation(J,first(grid.edges))` with `BoundaryInjection(:left)` adds total influx J once, with or without Growth; right injection is analogous |
| Batch Grid2D | Diagonal diffusion on fixed rectangular axes, each uniform or nonuniform; standalone or with shared bulk states |
| Grid2D boundaries | `SizeBC2D`, closed or paired periodic independently per axis; no boundary-point injection, diffusion inflow/outflow, escape ledger or extinction |
| Sources and other terms | Eligible interior sources and supported residual terms compose normally; supplied tensor collisions/transfers are unavailable |
| Native time integration | Euler, SSPRK22/33/54 and RK4 use the same residual and stage checks; user supplies dt |
| Optional ODE and stationary solvers | Eligible residuals remain available; external ODE solves do not acquire native positivity/CFL checks. Stationary derivatives require a smooth isolated compatible root |
| Excluded | Moment diffusion, mixed derivative terms, intrinsic diffusion in either axial adapter, MeasureEuler and complete characteristic/compact updates |

Second-order diffusion consistency is verified on uniform and smoothly graded
meshes, not arbitrary grading. A method name does not supply a positivity
limiter: WENO can fail native stage checks.
For closed boundaries the growth and diffusion base exterior fluxes are zero;
an explicitly selected **1D** boundary source replaces the prescribed total exterior
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

### Two internal coordinates

For two fluctuating coordinates, the implemented diagonal Ito model is

```math
\frac{\partial n(x,y,t)}{\partial t}
+\frac{\partial [G_x(x,y,t)n(x,y,t)]}{\partial x}
+\frac{\partial [G_y(x,y,t)n(x,y,t)]}{\partial y}
=\frac{\partial^2[D_x(x,y,t)n(x,y,t)]}{\partial x^2}
+\frac{\partial^2[D_y(x,y,t)n(x,y,t)]}{\partial y^2}.
```

Use `SizeDiffusion((Dx,Dy))` for constants, or
`SizeDiffusion((xy,p,t,ctx)->(Dx,Dy))`, where `xy=(x,y)`. Each coefficient may
depend on both coordinates, time and current `ctx.bulk`. Their units are
x²/time and y²/time. A scalar shorthand applies the same numerical coefficient
to both axes and assumes compatible coordinate units. The coordinate Wiener
increments are independent and the diffusion tensor is diagonal; a matrix
coefficient is rejected. This does not imply independence of the full joint
distribution. Initial correlations, cross-coordinate rate dependence and coupled
population/bulk dynamics must still be considered.

Unlike `SizeDiffusion`, the velocity callback is
`Growth2D((x,y,p,t,ctx)->(Gx,Gy))`: it receives two separate coordinates.

Here ``N_{ij}`` integrates over a rectangular cell, so its density is
``n_{ij}=N_{ij}/(\Delta x_i\Delta y_j)``. The x-face flux is

```math
q^x_{i+1/2,j}=-\frac{D^x_{i+1,j}n_{i+1,j}-D^x_{ij}n_{ij}}{x_{i+1}-x_i}.
```

The same flux times ``\Delta y_j`` leaves one cell and enters its neighbour;
y faces analogously use ``\Delta x_i``. This preserves total number on uniform
or nonuniform rectangular grids. Closed and paired-periodic conditions can be
chosen independently for each axis. Existing interior `Nucleation2D` fixed-pivot
or Gaussian sources compose with the residual; corner/boundary point injection
is not implemented for tensors.

The explicit loss coefficient adds **both** directional contributions,
``D^x_{ij}\sum 1/h_x\,/\Delta x_i+D^y_{ij}\sum 1/h_y\,/\Delta y_j``, and growth/removal.
On a uniform interior cell it is
``|G_x|/\Delta x+|G_y|/\Delta y+2D_x/\Delta x^2+2D_y/\Delta y^2``.
Upwind and a compatible SSP step preserve nonnegativity under the tested bound;
WENO has no positivity limiter and can fail native checks on a narrow coarse
profile. No population is clipped. Periodic/graded heat-equation refinements
verify the diffusion order independently from the time integrator. The
translating-Gaussian test with Upwind explicitly separates its numerical variance
``|G|\Delta x\,t`` from the physical increment ``2Dt``.

[Bulk coupling](../coupled_systems.md#Two-internal-coordinates-and-growth-dispersion)
uses the same-stage weighted **complete** residual. Run the synthetic example with
`julia --project=. examples/random_fluctuation_grd.jl --two-coordinate` or the
focused core check below. Optional tensor solver/estimation tests use the
[verification environment](../verification.md#Optional-integrations):

```sh
julia --project=. -e 'using Test, PopulationBalanceModels; include("test/tensor_diffusion.jl")'
julia --project=/tmp/pbm-optional test/tensor_integration.jl
```

