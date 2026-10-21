# Comparing growth and time methods

If every particle grows at the same rate, the size distribution should move
without broadening. This makes constant growth a useful test: the exact answer
is known, and numerical diffusion, oscillations and computational cost can be
compared without uncertainty about the physical kinetics.

We use Case 1 of [Inguva, Schickel and Braatz (2022)](https://web.mit.edu/braatzgroup/Inguva_ComputChemEng_2022.pdf),
then repeat the comparison with a smooth periodic distribution to measure
convergence. The two inputs serve different purposes. A sharp front tests
resolution and positivity; a smooth input tests the accuracy of the schemes.

## Model, initial condition and boundaries

For ``0\le a\le1``,

```math
\frac{\partial f(a,t)}{\partial t}
+0.1\frac{\partial f(a,t)}{\partial a}=0,
\qquad f(a,0)=100\exp(-a/0.01).
```

The incoming number flux is zero. The far boundary lets particles leave; the
initial tail that can reach it by ``t=4`` is negligible. The analytical density is

```math
f(a,t)=\begin{cases}
100\exp[-(a-0.1t)/0.01],&a\ge0.1t,\\
0,&a<0.1t.
\end{cases}
```

All numerical states are bin integrals, initialized and assessed against exact
bin integrals of this function. Sampling the steep exponential only at bin
centres would add a separate initialization error.

```@example growth_comparison
using PopulationBalanceModels
include(joinpath(pkgdir(PopulationBalanceModels),"examples","constant_growth_methods.jl"))
import .ConstantGrowthMethods
model = PBMModel(growth=Growth(AffineGrowth(0.1,0.0)))
grid = Grid1D(range(0.,1.;length=201);coordinate=:length)
bc = SizeBC(OpenBoundary(),OpenBoundary())
initial = ConstantGrowthMethods.exact_counts(grid,0.)
operator = compile(model,grid;methods=TermMethods(growth=MUSCL()),bc)
problem = PBMProblem(operator,initial,(0.,4.))
solution = simulate(problem;method=SSPRK33(),dt=0.5*first(grid.widths)/0.1)
nothing # hide
```

## A representative set of methods

The comparison uses six pairings. This keeps the plots readable while showing
first-order transport, limited reconstruction, high-order reconstruction,
temporal error and a characteristic shift.

| Growth treatment | Time method | Courant number in the smooth/timing comparison |
|:--|:--|:--|
| Upwind | Forward Euler | 0.5 |
| MUSCL | SSPRK33 | 0.5 |
| WENO5 | SSPRK33 | 0.5 |
| WENO5 | RK4 | 0.5 |
| Lax–Wendroff complete update | Built into the scheme | 0.5 |
| Inguva2022 characteristic update | Matched shift | 1 |

The Courant number is ``C=G\Delta t/\Delta a``. Inguva2022 requires a grid
uniform in travel time and a step that moves one whole cell. For constant
``G=0.1``, an ordinary uniform mesh meets that requirement:

```@example growth_comparison
shift_operator = compile(model,grid;evolution=Inguva2022(),bc)
shift_problem = PBMProblem(shift_operator,initial,(0.,4.))
shift_solution = simulate(shift_problem;dt=first(grid.widths)/0.1)
maximum(abs.(last(shift_solution.u).-ConstantGrowthMethods.exact_counts(grid,4.)))
```

That ``C=1`` is labelled in every comparison. It is not a valid ``C=0.5`` run
of the characteristic method. Its lower cost includes taking fewer steps and
using a direct copy rather than reconstructing face fluxes. Other available
schemes remain documented in [Growth](../methods/growth.md) and
[Time integration](../methods/time.md); they have not been removed from the package.

## What happens with Lax–Wendroff?

The constant-growth update is

```math
\bar f_i^{m+1}=\bar f_i^m
-\frac{C}{2}(\bar f_{i+1}^m-\bar f_{i-1}^m)
+\frac{C^2}{2}(\bar f_{i+1}^m-2\bar f_i^m+\bar f_{i-1}^m).
```

It is selected as a complete space–time method:

```julia
operator = compile(model,grid;evolution=LaxWendroff(),bc)
solution = simulate(PBMProblem(operator,initial,(0.,4.));
    dt=0.5*first(grid.widths)/0.1,check_nonnegative=false)
```

The audit compares its interior coefficients with
[`model_1.m` in PBM_Schemes](https://github.com/pavaninguva/PBM_Schemes/blob/02d8016c4aa69e58c0c8e06777bbe579adef6970/Matlab/model_1.m).
The updates agree to roundoff when supplied the same numerical values. Repeating
the reference stencil on 201 nodes at ``C=0.5`` gives a minimum density near
``-16.09``; it also produces negative lobes. The paper's Figure 1 uses ``C=1``,
where the update reduces to a one-cell shift and this
particular error disappears.

![The moving front at CFL 0.5 and CFL 1](../assets/constant_growth/solutions.png)

At ``C=0.5`` these are bounded dispersive oscillations. For periodic advection,
the Fourier amplification satisfies

```math
|A(\theta)|^2=1-4C^2(1-C^2)\sin^4(\theta/2)\le1,
\qquad |C|\le1.
```

Stability does not imply a nonnegative density. The runner retains negative
values so their size is visible; a checked physical solve should use a suitable
positivity-preserving method when these undershoots are unacceptable.

There are also two differences to account for when comparing entire runs:

- The MATLAB solver evolves nodal values, uses a zero left ghost and a reflected
  right ghost. The package evolves bin integrals with a prescribed zero total
  inlet flux and an advective outlet. Those boundary updates are distinct.
- The MATLAB loop keeps its nominal step and can finish beyond the requested
  final time. For 200 nodes at nominal ``C=1`` it ends near 4.0201. The package
  lands on 4 by shortening the last step, which then has ``C<1`` and loses the
  exact-shift cancellation. Using aligned meshes and output times avoids that
  mismatch. The grids in this comparison are aligned.

These findings do not support changing the interior Lax–Wendroff coefficients.
They explain why a ``C=0.5`` or unaligned-output result should not be compared
with the paper's whole-cell ``C=1`` result as if they were the same calculation.
The [audit CSV](../assets/constant_growth/lax_wendroff_audit.csv) records both
boundary treatments, requested/actual times, minima and coefficient differences.

## Compare both error norms

For analytical bin averages ``\bar f_i^{\mathrm{exact}}``, we report

```math
E_{L^1}=\sum_i\Delta a_i\left|\bar f_i-\bar f_i^{\mathrm{exact}}\right|,
\qquad
E_{\mathrm{RMS}}=\left[
\frac{\sum_i\Delta a_i(\bar f_i-\bar f_i^{\mathrm{exact}})^2}
{a_{\max}-a_{\min}}\right]^{1/2}.
```

![L1 and RMSE for the moving exponential front](../assets/constant_growth/performance.png)

The front is discontinuous, so these curves do not establish formal
smooth-solution order. The companion calculation uses periodic boundaries and

```math
f(a,0)=1+0.2\sin(2\pi a)+0.1\cos(4\pi a),
\qquad f(a,t)=f(a-0.1t,0)
```

with periodic continuation. Both initialization and comparison again use exact
bin integrals.

![L1 and RMSE for smooth periodic transport](../assets/constant_growth/smooth_refinement.png)

Rather than placing arbitrarily anchored ``h^2`` and ``h^5`` lines on these
plots, we calculate the observed order from each pair of grids:

```math
p_{\mathrm{obs}}=
\frac{\log[E(N)/E(2N)]}{\log2}.
```

![Measured orders in both error norms](../assets/constant_growth/observed_orders.png)

Upwind approaches first order and Lax–Wendroff second order. MUSCL approaches
second order in ``L^1`` but about 1.8 in RMSE here, because limiting near smooth
extrema affects the norms differently. WENO5 with SSPRK33 approaches third
order as time error dominates. RK4 gives much smaller errors on this mesh range;
its observed slopes above five are finite-range behaviour, not proof of a new
order or of fifth-order space–time accuracy. At sufficiently fine resolution,
fourth-order temporal error can dominate that pairing too. The characteristic
shift is at roundoff, where an inferred convergence order is meaningless.

## Measure computation time once

![Single-run elapsed time and error versus time, including Inguva2022](../assets/constant_growth/timings.png)

Each configuration has **one measured integration**. Small, separate warmups
compile the method before measuring; the benchmark-sized problem is not run
three times. Timings include stepping, residual evaluation and allocations,
and exclude compilation, file output and plotting. They have no averaging or
repeatability bands and will vary between runs and machines.

Raw results are in [errors.csv](../assets/constant_growth/errors.csv) and
[smooth_errors.csv](../assets/constant_growth/smooth_errors.csv). They also retain
negative number, total-number drift and actual step counts.

```sh
julia --project=. examples/constant_growth_methods.jl
julia --project=. examples/audit_lax_wendroff.jl
python docs/benchmarks/plot_transport_comparisons.py constant --usetex
julia --project=docs docs/make.jl
```

## Complete runnable scripts

The code below is read from the runnable sources during the documentation build.

### Growth and timing comparison

```@eval
using Markdown, PopulationBalanceModels
Markdown.MD([Markdown.Code("julia",read(joinpath(pkgdir(PopulationBalanceModels),"examples","constant_growth_methods.jl"),String))])
```

### Lax–Wendroff audit

```@eval
using Markdown, PopulationBalanceModels
Markdown.MD([Markdown.Code("julia",read(joinpath(pkgdir(PopulationBalanceModels),"examples","audit_lax_wendroff.jl"),String))])
```
