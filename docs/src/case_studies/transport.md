# 3. A two-coordinate population balance

## Case 2: size-dependent growth

Case 2 in [Inguva and Braatz (2023)](https://web.mit.edu/braatzgroup/Inguva_ComputChemEng_2023.pdf)
uses two internal particle coordinates, ``a_1`` and ``a_2``. The joint number
density ``f(a_1,a_2,t)`` satisfies

```math
\frac{\partial f(a_1,a_2,t)}{\partial t}
+\frac{\partial\!\left[G_1(a_1)f(a_1,a_2,t)\right]}{\partial a_1}
+\frac{\partial\!\left[G_2(a_2)f(a_1,a_2,t)\right]}{\partial a_2}=0,
```

```math
G_1(a_1)=0.1+0.05a_1,\qquad G_2(a_2)=0.5+0.25a_2,
\qquad 0\le a_1,a_2\le2.
```

There are no nucleation or collision terms in this case. The velocities increase
with coordinate, so each divergence must include the product of velocity and
density. Replacing it by ``G_i\,\partial f/\partial a_i`` would solve a different
equation and fail to account for stretching of the distribution.

The initial condition is

```math
f(a_1,a_2,0)=50\exp\!\left[-\frac{(a_1-0.4)^2}{0.005}
-\frac{(a_2-0.4)^2}{0.005}\right].
```

The finite-volume calculations use zero incoming flux at the lower edges and
no flux at the upper edges. At the comparison time ``t=1``, the narrow density
is well inside the domain; boundary tails are negligible. The Gaussian's
standard deviation is 0.05 in each coordinate, not one grid cell.

## Analytical solution

Define the starting points of the characteristics that arrive at ``(a_1,a_2)``:

```math
\xi_1=(a_1+2)e^{-0.05t}-2,\qquad
\xi_2=(a_2+2)e^{-0.25t}-2.
```

Conservation along the expanding characteristics gives

```math
f(a_1,a_2,t)=f(\xi_1,\xi_2,0)e^{-0.30t}.
```

This is equivalent to Eq. 60 in the paper. The factor ``e^{-0.30t}`` is essential:
it accounts for the area expansion in coordinate space. The numerical peak
moves near ``(0.5231,1.0817)`` at time 1, rather than the ``(1.4,1.4)`` location
of the constant-growth Case 1.

Comparisons use the integral of this solution over each bin, not its value at
a bin centre. The exact bin number follows by mapping the bin's edges back to
``\xi_1,\xi_2`` and integrating the initial Gaussian there.

## Define the model and initial state in Julia

```julia
using PopulationBalanceModels
include("examples/inguva_braatz_case2.jl")
import .InguvaBraatzCase2

function growth_rate(a1,a2,p,t,ctx)
    (0.1 + 0.05a1, 0.5 + 0.25a2)
end

model = PBMModel(growth=Growth2D(growth_rate))
grid = Grid2D(range(0,2;length=101),range(0,2;length=101))
initial_bin_numbers = InguvaBraatzCase2.reference(grid,0.0)
boundary = SizeBC(OpenBoundary(),NoFlux())
operator = compile(model,grid;
    methods=TermMethods(growth=WENO5(epsilon=1e-12)),
    bc=SizeBC2D(x=boundary,y=boundary))
problem = PBMProblem(operator,initial_bin_numbers,(0.0,1.0))
solution = simulate(problem;method=SSPRK33(),dt=0.4/(0.2/0.02 + 1.0/0.02),
    saveat=[1.0],check_nonnegative=false)
```

`Growth2D` receives both coordinates and returns their two velocities. `Grid2D`
stores integrated numbers, and `density(grid,N)` divides by the physical bin
areas for plotting. The quoted time step bounds the sum of both directional
Courant numbers by 0.4. Diagnostic solves report negative lobes instead of
clipping them. Standard simulation keeps its nonnegativity check enabled.

## Exact shifts on a transformed mesh

The paper also transforms each coordinate to a particle travel time:

```math
\tau_1(a_1)=\frac{\ln(1+a_1/2)}{0.05},\qquad
\tau_2(a_2)=\frac{\ln(1+a_2/2)}{0.25},
\qquad F(\tau_1,\tau_2,t)=G_1(a_1)G_2(a_2)f(a_1,a_2,t).
```

The transformed density obeys

```math
\frac{\partial F(\tau_1,\tau_2,t)}{\partial t}
+\frac{\partial F(\tau_1,\tau_2,t)}{\partial \tau_1}
+\frac{\partial F(\tau_1,\tau_2,t)}{\partial \tau_2}=0.
```

A uniform mesh with ``\Delta\tau_1=\Delta\tau_2=\Delta t`` permits one-cell shifts
in each direction. The case-specific adapter implements the analytical maps
and initializes the transformed bin numbers from the same physical integrals:

```julia
mapped = InguvaBraatzCase2.mapped_problem(20) # Δt = 0.05; 20 steps to t = 1
solution = simulate(mapped.problem;method=ExactShift(),dt=mapped.dt,saveat=[1.0])
physical_density = density(mapped.physical_grid,solution.u[end])
```

Bin number is invariant under the coordinate transformation. Do not multiply
it again by ``G_1G_2``; that factor converts density per physical area into
density per travel-time area. Divide the output bin numbers by physical cell
areas when returning to ``f``.

The mapped grid contains only whole travel-time cells inside the physical
0–2 domain. Its final partial cells are omitted, and the resulting extents are
recorded in the CSV. It uses paired open boundaries and a different nonuniform
physical grid from the uniform FV comparison. This is a bin-average adaptation
of the paper's transformed-mesh idea, not a claim that the nodal arrays or
SharpClaw settings are identical. The core ExactShift method still accepts
only constant-speed transport; the explicit case adapter supplies that form.

HOCADI also accepts only homogeneous constant growth on periodic uniform axes.
It is not applied directly to this variable-growth physical problem. Its
[separate stability audit](../methods/time.md) remains a numerical-method test.

## Run and plot locally

```sh
julia --project=. examples/inguva_braatz_case2.jl
python docs/benchmarks/plot_case_studies.py transport
```

The runner compares upwind, WENO5, and mapped exact shifts with the analytical
bin averages. It writes 2D fields, RMSE, maximum density error, integrated L1
error, minima, and negative number to `docs/src/assets/inguva_case2/`. The
Figure 4 layout and refinement comparison should be read with the declared
nodal/bin-average and mesh differences in mind.

```@eval
using Markdown, PopulationBalanceModels
file=joinpath(pkgdir(PopulationBalanceModels),"docs","src","assets","inguva_case2","solutions.png")
Markdown.parse(isfile(file) ? "![Case 2 analytical and numerical densities](../assets/inguva_case2/solutions.png)\n\n![Case 2 signed errors](../assets/inguva_case2/errors.png)" : "Run the commands above to generate the two-dimensional solution and signed-error panels, then rebuild the manual.")
```
