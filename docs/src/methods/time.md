# Time integration

## Advancing the assembled equations

After spatial/internal-coordinate discretization the state satisfies

```math
\frac{d\mathbf u(t)}{dt}=\mathbf R(\mathbf u(t),t;\mathbf p).
```

``\mathbf u`` contains all bin numbers or moment states and any bulk ODE states.
Each term method contributes to ``\mathbf R``. Time integration is selected by
`simulate(problem; method=...,dt=...)`; it is not a property of Growth or Nucleation.

The native residual-based choices are:

| Integrator | Stages | Time order | SSP property |
|:--|--:|--:|:--|
| `ForwardEuler()` | 1 | 1 | Baseline forward-Euler step restriction |
| `SSPRK22()` | 2 | 2 | Inherits a forward-Euler stability property under the same bound |
| `SSPRK33()` (default) | 3 | 3 | Inherits that property under the same bound |
| `SSPRK54()` | 5 | 4 | SSP coefficient about 1.508 relative to forward Euler |
| `RK4()` | 4 | 4 | No general SSP guarantee |

These methods advance the same compiled residual for sectional, tensor, moment,
coupled-bulk, and axial calculations. Their time order describes smooth ODE
solutions, not the complete error of a PBM with a nonsmooth distribution or
limited spatial reconstruction. In particular, changing Upwind from Euler to
RK4 leaves its first-order spatial diffusion.

SSPRK22 is the explicit trapezoidal (Heun) update:

```math
\mathbf u^{(1)}=\mathbf u^m+\Delta t\,\mathbf R(\mathbf u^m,t_m),\qquad
\mathbf u^{m+1}=\tfrac12\mathbf u^m+
\tfrac12[\mathbf u^{(1)}+\Delta t\,\mathbf R(\mathbf u^{(1)},t_m+\Delta t)].
```

RK4 evaluates the residual at ``t_m``, twice at ``t_m+\Delta t/2``, and at
``t_m+\Delta t``, with weights ``1/6,1/3,1/3,1/6``. Intermediate stage states
receive the same state checks as SSPRK33.

`SSPRK33()` uses the following three-stage update:

```math
\begin{aligned}
\mathbf u^{(1)}&=\mathbf u^m+\Delta t\,\mathbf R(\mathbf u^m,t_m),\\
\mathbf u^{(2)}&=\frac34\mathbf u^m+\frac14\left[
\mathbf u^{(1)}+\Delta t\,\mathbf R(\mathbf u^{(1)},t_m+\Delta t)\right],\\
\mathbf u^{m+1}&=\frac13\mathbf u^m+\frac23\left[
\mathbf u^{(2)}+\Delta t\,\mathbf R(\mathbf u^{(2)},t_m+\Delta t/2)\right].
\end{aligned}
```

All mechanisms, shared concentrations, and prescribed fields use the current
stage state/time. Freezing the concentration while advancing the PSD would be
a different method. SSP describes inheritance of a forward-Euler stability
property under a compatible step restriction; it does not guarantee positivity
for an arbitrary spatial reconstruction.

`dt` is a maximum fixed step, shortened to land on output times. Default state
checks reject nonfinite values and concentrations below `-negative_tolerance`
(default ``10^{-12}``), at intermediate stages and completed steps. Values are not clipped.
`suggest_dt` estimates an explicit transport/source bound, including outgoing
face velocities, size-diffusion losses and removal/collision hazards. The rates
are added within each bin: for constant G and D, an interior uniform bin has
the bound ``\Delta t\le C/(|G|/\Delta L+2D/\Delta L^2)``. It is not a proven positivity
bound for WENO or custom residuals. For 2D unsplit transport the sum of both
coordinate Courant numbers matters.
For a packed sectional PBMSystem, `suggest_dt(problem.operator,u,p,t)` uses
the current bulk context but does not bound bulk depletion or interactions.
Neither interface automatically recomputes or enforces a step during `simulate`.
See [size diffusion](growth.md#Random-fluctuation-growth-rate-dispersion) for
boundary, nonuniform geometry and RK-stage restrictions.

Load SciMLBase to use `odeproblem(problem)` with an external ODE solver. This
exports the same residual, including bulk equations. An adaptive or stiff solver
still needs a suitable error scale and treatment of nonnegative states. It does
not export direct compact or exact-shift updates.

## Constant-growth space–time schemes

For constant growth,

```math
\frac{\partial f(a,t)}{\partial t}
+g\frac{\partial f(a,t)}{\partial a}=0,
```

Lax–Wendroff uses a time-centered flux. For positive ``g``, with
``C=g\Delta t/\Delta a`` and ``\bar f_i=N_i/\Delta a``, it is

```math
F_{i+1/2}=\frac g2(\bar f_i+\bar f_{i+1})
-\frac{g^2\Delta t}{2\Delta a}(\bar f_{i+1}-\bar f_i),\qquad
N_i^{m+1}=N_i^m+\Delta t(F_{i-1/2}-F_{i+1/2}).
```

Beam–Warming uses an upwind-biased correction:

```math
F_{i+1/2}=g\left[\bar f_i+\frac{1-C}{2}(\bar f_i-\bar f_{i-1})\right].
```

The negative-velocity version reverses the upwind stencil. These are complete
second-order space–time schemes, not a face reconstruction to pass through
another RK method. At ``|C|=1`` both reduce to a cell shift for this homogeneous
constant-growth problem. Away from that limit, both can create oscillations
and negative densities. There is no limiter or clipping.

```@example constant_time
using PopulationBalanceModels
grid = Grid1D(0.0:0.01:1.0;coordinate=:length)
model = PBMModel(growth=Growth(AffineGrowth(0.1,0.0)))
bc = SizeBC(OpenBoundary(),OpenBoundary())
operator = compile(model,grid;evolution=LaxWendroff(),bc)
N0 = exp.(-((grid.centers.-0.2)./0.06).^2).*grid.widths
solution = simulate(PBMProblem(operator,N0,(0.0,0.2));
    dt=0.1,check_nonnegative=false) # allow and inspect dispersive undershoots
minimum(solution.u[end])
```

Compilation requires a uniform 1D grid with at least three bins, one Growth
term with an `AffineGrowth(g,0)` declaration, no residual-method overrides or
prescribed fields, and Periodic, OpenBoundary, or NoFlux size boundaries.
Sources, nucleation, bulk feedback, variable growth, and axial transport are
rejected. Parameters named in the growth declaration are rechecked when
simulating. This implementation restricts **both** schemes to ``|g|dt/\Delta a\le1``;
it does not use Beam–Warming's larger possible stability interval.

At an open boundary the prescribed incoming density is zero and outgoing flux
uses the adjacent bin. NoFlux sets the boundary face flux to zero; it can retain
particles in the last bin. Beam–Warming falls back to upwind at the first
interior face missing an upstream cell. These boundary treatments can reduce
the full solution's order. `dt` may shorten at output times; a shortened step
loses the exact-shift property unless it also travels a whole cell.

The [growth-method comparison](../case_studies/constant_growth.md) records
errors, negative number, and elapsed time rather than treating a finished run
as evidence of stability or positivity.

## Compact/ADI variant

`HOCFiniteVolume()` adds a distinct conservative compact policy for 1D/2D
bin integrals, positive separable growth, prescribed time dependence, point
births and physical inflow/outflow. Its complete midpoint/symmetric update is
**second order in time**, with fourth-order interior space. Select it during
`compile(...; evolution=HOCFiniteVolume(),bc,p)`. The
[compact specification](compact_verification.md) derives its cell-average face
reconstruction, source treatment, number accounting and compatibility checks.
The older periodic `HOCADI` construction described next remains unchanged.

`HOCADI(time_order=2)` or `HOCADI(time_order=4)` advances homogeneous constant
growth on uniform periodic 2D grids. It rejects sources, bulk coupling, and
physical nonperiodic compact boundaries. Its spatial derivative is defined by

```math
\frac14d_{i-1}+d_i+\frac14d_{i+1}
=\frac{3}{4h}(f_{i+1}-f_{i-1}),\qquad M\mathbf d=D\mathbf f.
```

Each coordinate uses a Cayley solve,

```math
\left(M+\frac{g\Delta t}{2}D\right)\mathbf f^{\mathrm{new}}
=\left(M-\frac{g\Delta t}{2}D\right)\mathbf f^{\mathrm{old}}.
```

One x solve and one y solve define ``S_2``. The fourth-order time composition is

```math
S_4(\Delta t)=S_2(\gamma\Delta t)S_2((1-2\gamma)\Delta t)S_2(\gamma\Delta t),
\qquad \gamma=(2-2^{1/3})^{-1}.
```

The negative middle step is restricted here to reversible autonomous transport.
It must not be applied to irreversible nucleation, collision, or bulk processes.
Cyclic tridiagonal factorization uses a rank-two correction; each directional
sweep costs linear work in the number of tensor cells.

This is a specified variant, not a verified transcription of Li et al.,
[doi:10.1016/j.partic.2026.01.024](https://doi.org/10.1016/j.partic.2026.01.024).
The supplement is now available. Its one-dimensional coefficients can be checked
independently, as described in [Compact-method verification](compact_verification.md).
Those coefficients define a different update from this periodic variant. The
main paper's multilevel update and physical boundary closure still need reconciliation.

## Stability is different from positivity

For a periodic Fourier mode with wave angle ``\theta``, the compact matrices
have symbols ``m=1+\frac12\cos\theta`` and
``d=\frac{3\mathrm i}{2h}\sin\theta``. A directional Cayley step has amplification

```math
A(\theta)=\frac{m-\mathrm i z}{m+\mathrm i z},\qquad
z=\frac{3g\Delta t}{4h}\sin\theta,\qquad |A(\theta)|=1.
```

The fourth-order composition is also unit modulus, including its negative step.
Thus this periodic constant-growth discretization has no growing Fourier mode
in exact arithmetic. It is dispersive and does not preserve positivity.
Narrow Gaussians and fronts can develop substantial negative undershoots even
while the discrete L2 norm remains constant.

`examples/verification/compact_adi.jl` checks every resolved Fourier mode over signed
Courant numbers 0.1–10 and evolves smooth/rectangular profiles for ten periods.
The recorded norm changes are below ``5\times10^{-14}``; negative densities
remain. Independent dense-matrix tests verify the cyclic solves for both time
orders. These results support stability of the implemented periodic variant,
not a positivity claim or physical-boundary/paper-fidelity claim.
The figure below retains the smooth-profile and sharp-front evidence for this
periodic constant-growth diagnostic. In particular, the front cut shows the
negative lobes despite bounded norm. The
[two-coordinate growth case](../case_studies/transport.md) is a separate,
coordinate-dependent physical problem.

![Periodic transport: smooth accuracy, sharp fronts, and negative undershoots](../assets/benchmark_2d.png)

## Exact integer shifts

`ExactShift()` implements the eligible constant-growth class of Inguva and Braatz
(2023). Each direction must travel an integer number of cells at every step and
output time. Periodic or paired open zero-inflow boundaries are supported.
Fractional remapping is rejected. An integer shift moves bin numbers exactly;
no interpolation or flux reconstruction is required.

ExactShift remains the older constant-velocity interface. For autonomous affine
growth, select Inguva2022 or InguvaBraatz2023 during compilation. These
[trajectory methods](characteristics.md) construct and check the corresponding
physical mesh and time-step restrictions. They do not expose a semi-discrete
residual for SSPRK33 or an external ODE solver.

## A first-order admissible measure policy

`MeasureEuler()` is a complete native positive-measure update for the supported
raw-moment/sectional systems. It is not another Runge–Kutta tableau. It constructs
admissible source/growth/collision substeps, includes optional cumulative losses
in those substeps, and couples bulk exchange through their material changes.
Its formal order is one, including splitting and bulk coupling. The
[method contract](realizable_evolution.md) gives the compatibility matrix,
depletion/CFL restrictions and diagnostics. External adaptive ODE solvers do
not inherit this guarantee or its atomic boundary events.
