# Transport along growth trajectories

Some growth equations can be solved without repeatedly reconstructing fluxes.
The key is to build a mesh on which particles take exactly one time step to
cross each cell. The package provides the double-transformation method of
[Inguva, Schickel and Braatz (2022), §2.2.1](https://web.mit.edu/braatzgroup/Inguva_ComputChemEng_2022.pdf)
as Inguva2022, and its commuting two-coordinate extension of
[Inguva and Braatz (2023), §2.2.1](https://web.mit.edu/braatzgroup/Inguva_ComputChemEng_2023.pdf)
as InguvaBraatz2023.

These are complete transport updates: the mesh and time step are part of the
method. They are selected during compilation with `evolution=...`, rather than
used as a face reconstruction inside SSPRK33.

## Change from size to travel time

Consider positive growth that depends only on size:

```math
\frac{\partial f(a,t)}{\partial t}
+\frac{\partial[G(a)f(a,t)]}{\partial a}=0.
```

Let ``\tau(a)`` be the time needed to grow from the lower domain boundary
``a_{\min}`` to size a. Define the density per unit travel time as F:

```math
\tau(a)=\int_{a_{\min}}^a\frac{1}{G(s)}\,ds,
\qquad F(\tau(a),t)=G(a)f(a,t).
```

The transformed equation has unit velocity:

```math
\frac{\partial F(\tau,t)}{\partial t}
+\frac{\partial F(\tau,t)}{\partial\tau}=0.
```

On a uniform travel-time mesh, upwind at ``\Delta t=\Delta\tau`` shifts the
solution one cell. No interpolation is needed. The implementation stores bin
numbers, so its update is ``N_i^{m+1}=N_{i-1}^{m}``. This is the cell-integral
version of the paper's nodal update: an initially exact cell integral remains
exact after each compatible shift. The initial integration error, conversion
of cell averages to point values, and floating-point error remain separate.

For ``G(a)=b+sa``, the mesh maps are

```math
\tau(a)=\frac{1}{s}\ln\!\left(1+\frac{s(a-a_{\min})}{b+sa_{\min}}\right),
\qquad
a(\tau)=a_{\min}+\frac{b+sa_{\min}}{s}(e^{s\tau}-1).
```

The implementation uses the continuous limit when s=0. Negative slopes are
allowed if growth stays strictly positive over the entire domain.

## Declare a form the compiler can check

An arbitrary Julia callback can hide dependence on time, another coordinate or
a bulk state. Sampling its values cannot prove that the transformation applies.
Use AffineGrowth to declare the supported form explicitly:

```@example characteristic_method
using PopulationBalanceModels, Distributions
p = (g0=0.434,g1=0.2604)
law = AffineGrowth(:g0,:g1)
model = PBMModel(growth=Growth(law))
dt = 0.01
grid = characteristic_grid(law,(0.0,1.4);dt,p,coordinate=:length)
boundary = SizeBC(OpenBoundary(),OpenBoundary())
operator_1d = compile(model,grid;evolution=Inguva2022(),p,bc=boundary)
term_methods(operator_1d)
```

Coefficients may be fixed numbers or parameter names. The compiler verifies
positive growth, constant travel-time spacing and compatible boundaries. It
rejects additional physical terms and ordinary growth callbacks. Selecting a
residual method in TermMethods at the same time is also an error.

The mesh above extends the upper edge to the next whole travel-time cell.
Inspect `grid.edges[end]` for its actual extent. Use `endpoint=:truncate` to
keep whole cells inside the requested domain, or `endpoint=:exact` to require
that both requested endpoints align. A shortened last cell is never inserted.

## Integrate and request compatible output times

```@example characteristic_method
initial = Normal(0.2,sqrt(0.0005/2))
number = 50sqrt(π*0.0005)
N0 = [number*(cdf(initial,b)-cdf(initial,a))
      for (a,b) in zip(grid.edges[1:end-1],grid.edges[2:end])]
problem = PBMProblem(operator_1d,N0,(0.0,1.0);p)
solution = simulate(problem;dt,saveat=[0.1,0.4,1.0])
solution.steps
```

The compiled method is the default evolution method for this operator. Every
output time must be an integer number of steps from the initial time. Unlike
SSPRK33, this method cannot shorten a step to land on an arbitrary output time.
Changing dt alone breaks CFL=1; reconstruct the mesh when changing dt.

Changing a rate parameter generally changes travel time through the cells.
Rebuild and recompile the mesh before a bin-shift run with different parameters.
The solver checks this again at run time. Number is conserved except for
particles leaving an open upper boundary. No-flux upper boundaries and
nonzero inflow are not implemented here. Periodic boundaries are supported
only for constant growth.

## Two independent particle coordinates

For independent positive laws, the equation is

```math
\frac{\partial f(a_1,a_2,t)}{\partial t}
+\frac{\partial[G_1(a_1)f(a_1,a_2,t)]}{\partial a_1}
+\frac{\partial[G_2(a_2)f(a_1,a_2,t)]}{\partial a_2}=0.
```

Set ``F=G_1G_2f`` and transform each size to travel time. Each directional
operator then shifts with unit speed. The two shifts commute, so their product
has no splitting error. Both travel-time spacings must equal the common dt:

```julia
law = SeparableGrowth2D(AffineGrowth(0.1,0.05),AffineGrowth(0.5,0.25))
grid = characteristic_grid(law,((0.0,2.0),(0.0,2.0));dt=0.05,
    endpoint=:truncate)
operator_2d = compile(PBMModel(growth=Growth2D(law)),grid;
    evolution=InguvaBraatz2023(),bc=SizeBC2D(x=boundary,y=boundary))
```

The [two-coordinate case study](../case_studies/transport.md) uses this public
interface. Cross-coordinate velocities do not satisfy this declaration. The
other paper classes involving time transformations, reactions or noncommuting
splitting are not implemented by these two methods. Use the residual-based
methods for those models unless a separate verified extension is supplied.

## Evaluate a known continuous initial density

If the initial density is available as a function, the same transformation
can evaluate the continuous solution at arbitrary positions and times:

```julia
f0(a) = 50exp(-(a-0.2)^2/0.0005)
value = characteristic_density(operator_1d,f0,0.6,0.43;p)
```

Here `operator_1d` is the 1D compiled operator. The foot of the trajectory is
``a_0=\tau^{-1}(\tau(a)-t)``, and the evaluator returns

```math
f(a,t)=f_0(a_0)\frac{G(a_0)}{G(a)}.
```

This direct function evaluation does not interpolate bin states and has no
output-time alignment restriction. It is distinct from shifting an arbitrary
supplied vector of bin numbers. The
[parameter-estimation study](../case_studies/estimating_growth.md) uses an
independent analytical expression to generate and fit continuous density data,
then checks the numerical bin solvers against that reference.
