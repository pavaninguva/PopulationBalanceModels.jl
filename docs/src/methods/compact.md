# Compact finite-volume growth

`HOCFiniteVolume()` advances **bin-integrated populations** in
one or two size coordinates. It is a conservative adaptation, with fourth-order
interior spatial reconstruction and second-order time integration. It supports
declared positive size-dependent or prescribed time-dependent growth and
physical inflow/outflow. Earlier HOCADI and nodal paper transcriptions are kept in the
[verification notes](compact_verification.md); they are different updates.

## Start with a complete calculation

```@example compact_first
using PopulationBalanceModels
grid = Grid1D(range(0, 2; length=81); coordinate=:length)
growth = AffineGrowth(0.4, 0.15)
model = PBMModel(growth=Growth(growth))
bc = SizeBC(InflowFlux(1.0), OpenBoundary())
operator = compile(model, grid; evolution=HOCFiniteVolume(), bc)
# Integral of the positive stationary background 1/G(x).
initial = diff(log.(0.4 .+ 0.15 .* grid.edges)) ./ 0.15
problem = PBMProblem(operator, initial, (0.0, 0.4))
solution = simulate(problem; dt=0.0002)
diagnostics(solution).number_accounting[end]
```

For two coordinates, use
`Growth2D(SeparableGrowth2D(AffineGrowth(0.4,0.1), ExponentialGrowth(0.3,0.12)))`
and `SizeBC2D(x=bc_x,y=bc_y)`. Axis lengths, bin counts and spacings may differ.
[Runnable examples](../assets/source/examples/compact_growth.jl) include smooth
translations, incoming particles, variable growth and boundary nucleation.
The [benchmark driver](../assets/source/examples/verification/compact_growth.jl)
keeps reporting and timing outside those short model definitions.
Use a checkout, or retain the paths `examples/compact_growth.jl` and
`examples/verification/compact_growth.jl`. The latter includes `../compact_growth.jl`;
the two identically named files must not overwrite each other.

## Equation, states and compact faces

The supported physical equation is

```math
\frac{\partial f(x,t)}{\partial t}
+\frac{\partial [G(x,t)f(x,t)]}{\partial x}=S(x,t),
```

or its separable two-coordinate counterpart,

```math
\frac{\partial f(x,y,t)}{\partial t}
+\frac{\partial [G_1(x,t)f(x,y,t)]}{\partial x}
+\frac{\partial [G_2(y,t)f(x,y,t)]}{\partial y}=S(x,y,t).
```

States remain ``N_i=\int_{x_{i-1/2}}^{x_{i+1/2}} f\,dx`` and
``N_{ij}=\int\!\int_{C_{ij}} f\,dx\,dy`` per process volume. Thus
``U_i=N_i/h`` is a cell average. The independently derived face relation is

```math
\frac14\widehat f_{j-1}+\widehat f_j+\frac14\widehat f_{j+1}
=\frac34(U_j+U_{j+1}).
```

Here face ``j`` lies between cells ``j`` and ``j+1``. Expanding the **cell
integrals** about that face gives the same terms through ``h^2`` on both
sides; their difference is ``h^4 f^{(4)}/120+O(h^6)``. Consequently the smooth
face error is ``-h^4 f^{(4)}/180+O(h^6)`` and its interior flux difference is
fourth order. This is not an assumption that cell averages are nodal values.
For constant periodic growth, eliminating these faces gives
``M\dot{\mathbf N}=-gD\mathbf N``, with diagonal/off-diagonal mass weights
``1,1/4`` and ``D=3(E_+-E_-)/(4h)``. This is exactly the normalized
``\Delta t\to0`` spatial limit of Zhang2025 Eq. 15. Uniform bin integration
commutes with that constant-coefficient stencil. The new midpoint update does
not retain the paper's ``(g\Delta t/h)^2`` corrections or claim its fourth-order
time formula.
For variable growth, multiply the reconstructed density by the **face** rate:
``F_j=G(x_{j+1/2},t)\widehat f_j``. Applying a nodal stencil to ``G_iU_i``
would define a different discretization.

Periodic faces wrap. A physical left boundary prescribes number flux directly:
`OpenBoundary` and `NoFlux` give zero incoming flux; `InflowFlux` gives the
specified flux. Positive growth requires right `OpenBoundary`. Its trace is
the degree-four polynomial reconstruction from the last five cell averages:

```math
\widehat f_n=\frac{137}{60}U_n-\frac{163}{60}U_{n-1}
+\frac{137}{60}U_{n-2}-\frac{21}{20}U_{n-3}+\frac15U_{n-4}.
```

This outflow trace has fifth-order truncation error. Smooth whole-solve
physical-boundary convergence is measured separately below; periodic Fourier
stability alone is not a proof for this boundary closure.
In 2D each line contains populations integrated over its transverse bin.
Incoming `InflowFlux` values are uniform face-flux densities per transverse
size, multiplied by that bin's width. Nonuniform face profiles are not supported.

## Complete time update and number balance

At each directional step, freeze prescribed coefficients at the full step's
midpoint. Solve for a midpoint number flux using

```math
N_i^{\mathrm{new}}=N_i^{\mathrm{old}}+\Delta t(F_{i-1}-F_i),\qquad
\mathbf N^{\mathrm{mid}}=\mathbf N^{\mathrm{old}}
-\frac{\Delta t}{2}D\mathbf F^{\mathrm{mid}}.
```

Writing the compact relation as ``P\widehat{\mathbf f}=Q\mathbf N/h`` gives

```math
\left[P\,\operatorname{diag}(1/G^{\mathrm{mid}})
+\frac{\Delta t}{2h}QD\right]\mathbf F^{\mathrm{mid}}
=Q\mathbf N^{\mathrm{old}}/h,
```

with known inflow terms moved to the right. ``D\mathbf F`` means right minus
left flux. The interior system is tridiagonal. Periodic corners and the
one-sided outflow row use bounded-rank corrections to a tridiagonal LU:
setup, storage and each line solve cost ``O(n)``. Factors are reused only when
the directional step and **all** face rates are unchanged. Caches are local
to a simulation; `remake` rechecks parameters and builds fresh factors.

In 2D the composition is half x, full y, half x. Homogeneous separable
directional operators commute; the symmetric composition also handles the
supported inhomogeneous boundary data to second order. Interior point sources
use source/transport/source splitting, with midpoint quadrature at the quarter
and three-quarter times. The **complete method is second order in time**,
including prescribed time dependence and this source splitting. No negative
substep is used. No transformation ``u=Gf`` is used, so time dependence does
not lose a ``G_t`` contribution.
The order statement requires smooth solutions and sufficiently smooth
coefficient, source and incoming-flux histories. A discontinuity or singular
birth front does not satisfy the smooth spatial-error assumptions.

`Nucleation(J,lower_edge)` on a physical 1D domain supplies one incoming flux
``J(t)`` at the step midpoint. Interior point births contribute their cell
integral. A `Nucleation2D` corner contributes **once** to the containing cell,
not independently through two faces. That singular-source deposition preserves
injected number; it does not have smooth-density fourth-order accuracy or an
exact subcell size moment. Positive-growth outflow-face births are rejected.
Use `PrescribedRate((p,t,ctx)->...)` for prescribed source/inflow callbacks.
Do not specify the same boundary birth again as `InflowFlux`.

Every actual substep accumulates incoming number, births and the computed
signed outflow. At saved times, `diagnostics(solution).number_accounting` holds
`inflow` and `outflow` tuples by axis, `births`, `inventory`, and

```math
\mathrm{defect}=\sum N(t)-\sum N(0)
-\mathrm{inflow}-\mathrm{births}+\mathrm{outflow}.
```

These are integrated during the solves, not reconstructed from output samples.
They are historical number accounting, not an escaped-particle reservoir with
ongoing growth. Sectional state lengths are unchanged. Material moments
generally **change** under growth. Existing `moment` observations retain their
center-weighted bin convention; their quadrature error is separate from the
transport order. No population–bulk exchange is implemented for this policy.

## Compilation contracts and supported combinations

| Feature | `HOCFiniteVolume` |
|:--|:--|
| Representation | Bin integrals, 1D/2D; uniform rectangular axes, at least six cells each |
| Growth | Strictly positive `AffineGrowth`, `ExponentialGrowth`, `TimeScaledGrowth`; declared `HOCGrowthLaw` |
| Time dependence | Prescribed, evaluated at the midpoint; order 2 |
| Boundaries | Paired periodic, or left zero/prescribed inflow and right open outflow |
| Source | Prescribed point nucleation as described above; order 2 in time |
| Temporal order | 2 |
| Bulk/axial/collisions/removal | Rejected; use residual methods |
| Positivity | Not guaranteed; default checks reject negative bin populations |
| External ODE/AD sensitivities | Not supported |

Compilation checks the **assembled model**, not just the growth term. Ordinary
unclassified callbacks are rejected, even if diagnostic samples match an affine
law. Existing callbacks continue to work for residual methods. For a general
law, explicitly declare its assumptions:

```julia
law = HOCGrowthLaw((x,p,t,ctx) -> 0.4 + 0.1sin(2pi*x);
    lower_bound=0.3, dependencies=(:coordinate,), periodic=true)
```

This declares a positive C⁶ law on the entire compiled axis and, here, a smooth
periodic extension. It is a user contract, not a mathematical proof obtained
from probes. Allowed dependencies are the own `:coordinate`, `:time`,
`:parameters`, and prescribed `:fields`. `:bulk`, population feedback and the
other direction's coordinate are rejected. The lower-bound callback has
signature `(p,t,ctx)` and must bound the full spatial domain. All numerical
face values and the bound are checked when constructing/remaking a problem,
at coefficient evaluations and at accepted endpoints. The contract must also
hold between sampled times; arbitrary callbacks cannot be certified there.
Simulation entry repeats the numerical checks, including requests that save
only the initial time, so changed parameter containers or fields cannot reuse
an earlier problem-construction check.
Nonconstant affine/exponential laws require physical boundaries. Unknown
periodic laws need `periodic=true`, including matching derivatives.
For this policy, `TimeScaledGrowth` uses an affine or exponential autonomous
spatial law; other prescribed smooth laws use `HOCGrowthLaw` directly.

For physical boundaries or spatially variable laws, the verified envelope
requires ``\max(G^{\mathrm{mid}})\Delta t/h\le1`` on each axis, using the numerical
face rates. The solver rejects violations
and suggests a smaller step. This restriction is **not** a positivity theorem.
Spatially constant periodic growth has unit-modulus midpoint factors for any
Courant number, but accuracy still requires time refinement. Resolve narrow
source pulses independently of this transport restriction. Dispersive negative
lobes remain possible; `check_nonnegative=false` explicitly retains them for
diagnostics without clipping. Even a positive cell state can have a signed
reconstructed outflow trace, so outflow accounting uses the actual signed flux.

## What the verified orders mean

For smooth solutions, the interior face reconstruction is fourth order in
space. The complete midpoint/symmetric update is second order in time.
At fixed Courant number the time error therefore commonly limits a grid
refinement curve to second order. A stable large step is not an accurate step.
For physical boundaries or variable laws, the implemented envelope requires
CFL ≤ 1; constant periodic midpoint transport permits larger CFL without a
Fourier growth mode. Neither statement guarantees positivity.

The retained affine/exponential small-step spatial studies give successive
L¹ slopes 4.13/4.04 and 4.11/4.04. Independent time tests hold the spatial
operator fixed. Their [full evidence and work–precision figures](compact_verification.md#Executed-refinement-and-work–precision-evidence)
separate spatial, time, boundary and source tests. Sharp-front undershoots are
reported as dispersive error, not as smooth-order evidence.

## Relation to published methods

The cell-average face reconstruction, physical outflow closure and midpoint
flux solve are a **package adaptation**. Its constant-periodic spatial limit
matches Zhang et al.'s Eq. 15, but it does not reproduce that paper's complete
time formula or Li et al.'s multilevel ADI construction. Fu et al.'s
characteristic finite-volume construction motivates an integral conservation
comparison, not a hidden characteristic remapping step here.

See [Zhang et al. (2025)](https://doi.org/10.1021/acs.iecr.4c02816),
[Li et al. (2026)](https://doi.org/10.1016/j.partic.2026.01.024) and
[Fu et al. (2025)](https://doi.org/10.1016/j.camwa.2025.08.004).
The [equation-by-equation verification notes](compact_verification.md#Relation-to-the-three-papers)
retain the literal and independently corrected nodal transcriptions and the
unresolved published-stage interpretation. They are not selectable production
implementations of those full paper algorithms.
