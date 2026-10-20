# Getting started

Suppose particles enter a vessel at a constant rate, grow, and are removed with
a size-independent hazard. We want their size distribution after one unit of
time. This small model introduces the same steps used in the
[case studies](case_studies/index.md): define rates, choose how to discretize
terms, apply boundaries, and solve.

## Load the package

Use Julia 1.10 or newer. From the package checkout, run:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); using PopulationBalanceModels'
```

To work in a separate Julia environment, run
`Pkg.develop(path="/absolute/path/to/PopulationBalanceModels.jl")` there.
The remaining code blocks form one complete calculation and are executed when
this manual is built.

## Write the balance

Take particle volume ``v`` as the internal coordinate and let ``n(v,t)`` be number
density per unit particle volume and process volume. In the interior,

```math
\frac{\partial n(v,t)}{\partial t}
+\frac{\partial[G n(v,t)]}{\partial v}=-k n(v,t).
```

Particles enter at ``v=0`` with inward number flux ``J``. The upper edge is
closed. The constants below use a consistent arbitrary volume and time scale;
the package does not convert units for us.

```@example quickstart
using PopulationBalanceModels
p = (G=0.3,J=0.05,k=0.02)

growth_rate(v,p,t,ctx) = p.G
nucleation_rate(p,t,ctx) = p.J
removal_rate(v,p,t,ctx) = p.k

model = PBMModel(
    growth = Growth(rate=growth_rate),
    primary = Nucleation(rate=nucleation_rate,size=0.0),
    loss = Death(rate=removal_rate),
)
nothing # hide
```

Growth expects a coordinate velocity, not a particle flux. Death expects a
nonnegative inverse-time hazard. Nucleation expects a total number creation
rate and a location; its callback therefore has no size argument. The callbacks
receive `p`, time `t`, and the current state context `ctx`, although these
constant rates only need `p`.

## Choose bins and term methods

```@example quickstart
grid = Grid1D(range(0.0,10.0;length=101);coordinate=:volume)
methods = TermMethods(
    growth = MUSCL(),
    by_term = (primary=BoundaryInjection(:left),),
)
operator = compile(model,grid;methods,bc=SizeBC(NoFlux(),NoFlux()))
nothing # hide
```

MUSCL reconstructs a density for growth face fluxes. BoundaryInjection treats
`primary` as a prescribed inward number flux at the left edge. It adds that
source once, without dividing by the growth velocity. The NoFlux boundaries
suppress transport through the exterior faces; the named boundary source
supplies the influx separately.

The grid stores **integrated bin numbers**, not density samples:

```math
N_i(t)=\int_{v_{i-1/2}}^{v_{i+1/2}}n(v,t)\,dv.
```

For a smooth initial profile we can use midpoint integration:

```@example quickstart
initial_density = exp.(-((grid.centers .- 2.0) ./ 0.4).^2)
initial = initial_density .* grid.widths
problem = PBMProblem(operator,initial,(0.0,1.0);p)
nothing # hide
```

For discontinuities or peaks narrower than a bin, integrate the initial profile
more carefully. Changing the time integrator cannot correct an inaccurate
initial bin number.

## Advance the solution

```@example quickstart
solution = simulate(problem;method=SSPRK33(),dt=0.01,saveat=0.0:0.25:1.0)
final = solution.u[end]
final_density = density(grid,final)
(number=only(moment(grid,final,0)),first_moment=only(moment(grid,final,1)))
```

The time integrator is a separate choice from MUSCL. It sums growth, nucleation,
and removal contributions at every stage. `dt` is a maximum step; it is
shortened to reach each requested output time. A single-population solution
stores bin numbers as `N[bin,1]`. `density` divides those numbers by bin widths.

A useful check follows by integrating the PDE over the closed size domain:

```math
\frac{dM_0(t)}{dt}=J-kM_0(t),\qquad
M_0(t)=M_0(0)e^{-kt}+\frac{J}{k}(1-e^{-kt}).
```

```@example quickstart
expected = sum(initial)*exp(-p.k) + p.J/p.k*(-expm1(-p.k))
@assert abs(sum(final)-expected) < 1e-9
@assert minimum(final) >= 0
abs(sum(final)-expected)
```

This tests total number, while a grid-refinement study is still needed to assess
the shape of the distribution. Next, [Defining models](defining_models.md)
explains size coordinates and state-dependent rates. [Numerical methods](numerical_methods.md)
explains each term's treatment, and [Coupled systems](coupled_systems.md) adds
shared dissolved concentrations and interacting populations.
