# Comparing growth and time methods

A population can broaden for two different reasons: its particles may have
different growth rates, or its numerical approximation may smear the size
distribution. A constant-growth calculation separates these effects. Every
particle moves at the same rate, so the true distribution translates without
changing shape. Any broadening, oscillation, or loss of number comes from the
calculation or its boundaries.

This model follows Case 1 of
[Inguva, Schickel, and Braatz (2022)](https://web.mit.edu/braatzgroup/Inguva_ComputChemEng_2022.pdf).
It also demonstrates a special result from that study: at a Courant number of
one, some simple schemes reduce to an exact cell shift. A more accurate ODE
integrator does not necessarily preserve that cancellation.

## Model and analytical solution

For intrinsic coordinate ``a\ge0``, solve

```math
\frac{\partial f(a,t)}{\partial t}
+0.1\frac{\partial f(a,t)}{\partial a}=0,
\qquad f(a,0)=100\exp(-a/0.01).
```

There are no births, deaths, or collisions. The left boundary supplies no new
particles. The analytical solution before the upper boundary matters is

```math
f(a,t)=
\begin{cases}
100\exp[-(a-0.1t)/0.01],&a\ge0.1t,\\
0,&a<0.1t.
\end{cases}
```

The runner uses ``0\le a\le1`` and stops at ``t=4``. It applies a zero-flux
upper boundary as in the source comparison. The initial tail near that boundary
is exponentially small; the dominant feature travels to ``a=0.4``. The sharp
leading edge is a difficult test for unlimited high-order schemes.

## Specify the law, grid, and boundaries

```@example growth_comparison
using PopulationBalanceModels
include(joinpath(pkgdir(PopulationBalanceModels),"examples","constant_growth_methods.jl"))
import .ConstantGrowthMethods
const comparison = ConstantGrowthMethods

model = PBMModel(growth=Growth(AffineGrowth(0.1,0.0)))
grid = Grid1D(range(0.0,1.0;length=201);coordinate=:length)
bc = SizeBC(OpenBoundary(),NoFlux())
N0 = comparison.exact_counts(grid,0.0)
nothing # hide
```

`AffineGrowth(0.1,0.0)` declares a constant velocity in coordinate units per
time. `N0` contains bin integrals of the initial density. Sampling the steep
initial exponential only at bin centers would introduce an avoidable initial
quadrature error.

## Pair a face reconstruction with an integrator

```@example growth_comparison
operator = compile(model,grid;methods=TermMethods(growth=MUSCL()),bc)
problem = PBMProblem(operator,N0,(0.0,4.0))
dt = 0.5*first(grid.widths)/0.1
solution = simulate(problem;method=SSPRK33(),dt)
nothing # hide
```

Changing `MUSCL()` to `Upwind()`, `WENO5()`, `Centered()`, or `LinearUpwind()`
changes the growth face flux. Changing `SSPRK33()` to `ForwardEuler()`,
`SSPRK22()`, or `RK4()` changes how the assembled residual advances in time.
The runner compares selected combinations, rather than assuming every pairing
is stable at the same step size. The
[growth](../methods/growth.md) and [time](../methods/time.md) pages describe
their stencils, boundaries, and restrictions.

Lax–Wendroff and Beam–Warming include their own time updates:

```@example growth_comparison
lw = compile(model,grid;evolution=LaxWendroff(),bc)
lw_problem = PBMProblem(lw,N0,(0.0,4.0))
lw_solution = simulate(lw_problem;dt,check_nonnegative=false)
minimum(lw_solution.u[end])
```

`evolution=BeamWarming()` selects the other complete update. These compiled
operators reject variable growth and additional mechanisms. They do not supply
an ODE residual for a second integrator. Negative values are deliberately
retained in this benchmark and exported; the normal default state checks
would reject a run with a substantial undershoot.

## Compare the solutions

![Constant-growth solutions with two Courant numbers](../assets/constant_growth/solutions.png)

At ``C=0.5``, upwind broadens the peak. Increasing its time order does not remove
the spatial diffusion. MUSCL and WENO reduce that broadening, while the unlimited
schemes can create appreciable negative lobes. Their negative number and minimum
density are reported in `errors.csv`.

At ``C=1``, Upwind with ForwardEuler, Lax–Wendroff, and Beam–Warming shift the
bin numbers one cell per step, within floating-point error. Upwind with RK4
still solves the semi-discrete upwind equations and remains diffusive. The
exact cancellation is a property of the complete update, not of the upwind
spatial stencil alone. The existing `Inguva2022()` method provides a direct
shift on a matched travel-time grid; its variable-growth extension is explained
in [Transport along growth trajectories](../methods/characteristics.md).

The unlimited LinearUpwind pairing is a useful counterexample: ``C=1`` is
outside the stability region of both SSPRK33 and RK4 for that residual. Those
runs grow enormously and are retained as failed choices in the comparison.
A transport CFL of one is not a general guarantee for a higher-order stencil.

## Measure error and work

For each run, compare numerical bin averages with **analytical bin averages**:

```math
\bar f_i^{\mathrm{exact}}(t)=\frac{1}{\Delta a_i}
\int_{a_{i-1/2}}^{a_{i+1/2}}f(a,t)\,da,
```

```math
E_{\mathrm{RMS}}=
\left[\frac{1}{a_{\max}-a_{\min}}
\sum_i\Delta a_i(\bar f_i-\bar f_i^{\mathrm{exact}})^2\right]^{1/2}.
```

The runner also records mean and maximum absolute errors, total-number drift,
negative number, step count, and elapsed time. The initial condition and its
zero-inflow front are nonsmooth; the smooth-solution formal orders should not
be read off this refinement curve. The reference is integrated analytically,
not obtained by refining one of the numerical schemes.

![Errors, refinement, and elapsed work for growth and time-method combinations](../assets/constant_growth/performance.png)

The heatmap uses a display floor of ``10^{-13}``; annotations identify errors
above its color range. Timings are individual warmed runs on the generating
machine, and include residual assembly and allocations. They are suitable for
understanding this calculation's tradeoffs, not for ranking Julia solvers on
other hardware. These finite-volume errors also differ from a nodal
finite-difference RMSE; this is a comparison of the same physical benchmark,
not a digitization of the paper's figure.

Regenerate the data and figures from the repository root:

```sh
julia --project=. examples/constant_growth_methods.jl
python docs/benchmarks/plot_transport_comparisons.py constant --usetex
julia --project=docs docs/make.jl
```
