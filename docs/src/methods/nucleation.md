# Nucleation

Before choosing a nucleation method, decide where particles are created in
size space. Entry at the domain boundary, creation at one interior size, and
creation over a physical size distribution are different models. A total
number rate alone does not specify which one is intended.

## Boundary creation

Zero-size nuclei can enter through the left boundary rather than an interior
source:

```math
\frac{\partial n(x,t)}{\partial t}
+\frac{\partial [G(x,t)n(x,t)]}{\partial x}=0,
\qquad G(a,t)n(a,t)=J(t).
```

``J`` has units number/(process volume time), whereas ``G`` has coordinate/time
units. Specifying number flux avoids dividing by a vanishing growth rate.
The bulk model must still explain how particles move away from the boundary.

```@example nucleation
using PopulationBalanceModels
grid = Grid1D(0.0:0.1:2.0;coordinate=:length)
p = (J=1.0,critical_size=0.8,nucleation_width=0.015)
growth_rate(x,p,t,ctx) = 0.2
nucleation_rate(p,t,ctx) = p.J

model = PBMModel(
    growth=Growth(rate=growth_rate),
    primary=Nucleation(rate=nucleation_rate, size=0.0),
)
methods = TermMethods(growth=MUSCL(), by_term=(primary=BoundaryInjection(:left),))
operator = compile(model,grid;methods)
problem = PBMProblem(operator,zeros(length(grid)),(0.0,0.1);p)
solution = simulate(problem;dt=0.01)
sum(solution.u[end])
```

The source location must equal the selected edge and the model must contain
Growth. Active injection requires inward or zero velocity. The alternative
`SizeBC(InflowFlux(nucleation_rate),NoFlux())` places the influx in the boundary
specification instead. Use one declaration for a single mechanism; specifying
both would count creation twice.

The first bin gains ``J`` particles per process volume per time. Although a
zero-size continuum source has zero higher raw moments, a finite first pivot
assigns those particles a small positive size. Bulk exchange uses that discrete
moment; refine the first bin when that contribution matters.

## Point creation inside the domain

```math
\frac{\partial n(x,t)}{\partial t}
+\frac{\partial [G(x,t)n(x,t)]}{\partial x}
=J(t)\delta(x-x_c).
```

`Nucleation(rate=J,size=xc)` uses a fixed location in the grid's coordinate.
The callback is `J(p,t,ctx)`; no size argument is passed because ``J`` is a total
rate, not a source density. FixedPivot is the default method. For
``x_l\le x_c\le x_r``, creation is shared with weights

```math
w_l=\frac{x_r-x_c}{x_r-x_l},\qquad w_r=1-w_l.
```

This preserves total number and the first raw coordinate moment within the
pivot range. It does not preserve every nonlinear particle property. A location
between the domain edge and first pivot is not inside the pivot range; use a
boundary declaration or an explicit distributed size model as appropriate.

The interior source needs no boundary injection:

```@example nucleation
point_model = PBMModel(secondary=Nucleation(rate=nucleation_rate,size=0.8))
point_operator = compile(point_model,grid;
    methods=TermMethods(nucleation=FixedPivot()))
point_problem = PBMProblem(point_operator,zeros(length(grid)),(0.0,0.1);p)
point_solution = simulate(point_problem;dt=0.01)
(number=sum(point_solution.u[end]),
 first_moment=only(moment(grid,point_solution.u[end],1)))
```

## A physical distribution of creation sizes

```math
\frac{\partial n(x,t)}{\partial t}
+\frac{\partial [G(x,t)n(x,t)]}{\partial x}=J(t)\,\phi(x,t).
```

`DistributedNucleation(rate=J,shape=phi)` samples the shape with midpoint bin
quadrature by default; `GaussLegendre(order)` integrates it numerically per bin.
The shape callback is `phi(x,p,t,ctx)`. With `normalization=:total_rate`, resolved
weights are normalized so their sum creates exactly ``J``. With
`:density_amplitude`, the supplied shape is left unnormalized, so the units of
``J`` must match that convention. These are different physical declarations.

`IntegratedNucleation(rate=J,fraction=bin_fraction)` instead receives
`bin_fraction(lo,hi,p,t,ctx)`. BinIntegral uses those integrals directly, avoiding
missed peaks between cell centres. `IntegratedBirth(integral=...)` accepts an
already dimensional bin creation rate.

## Gaussian size spread

```math
\phi(x,t)=\frac{1}{\sqrt{2\pi}\,\sigma(t)}
\exp\!\left[-\frac{(x-\mu(t))^2}{2\sigma(t)^2}\right],\qquad
\frac{dN_i}{dt}\bigg|_{\mathrm{nucleation}}=J(t)
\int_{x_{i-1/2}}^{x_{i+1/2}}\phi(x,t)\,dx.
```

```@example nucleation
source = GaussianNucleation(
    rate=nucleation_rate,
    center=(p,t,ctx)->p.critical_size,
    width=(p,t,ctx)->p.nucleation_width,
    truncation=:physical,
)
gaussian_operator = compile(PBMModel(primary=source),grid)
gaussian_problem = PBMProblem(gaussian_operator,zeros(length(grid)),(0.0,0.1);p)
gaussian_solution = simulate(gaussian_problem;dt=0.01)
sum(gaussian_solution.u[end])
```

`width` is a standard deviation in coordinate units. Hold a physical width fixed
under grid refinement. A numerical regularization width needs a separate width
convergence study; silently tying it to cell spacing changes the model.
`:physical` creates the Gaussian fraction inside the finite domain.
`:renormalize` redistributes the full declared rate into that domain.

The implementation integrates in standardized coordinates with subdivided
Gauss–Legendre quadrature over ±12 standard deviations; omitted probability
is below ``4\times10^{-33}``. It can resolve a narrow source even if no bin centre
samples it. Direct quadrature avoids subtracting nearly equal cumulative tail probabilities. These integrals preserve number according to the selected
normalization; coarse pivot moments still approximate physical mass.

`GaussianInjection(width=...)` is a numerical policy that regularizes an existing
point declaration. Unlike changing a quadrature rule, a finite width changes
higher source moments. Prefer `GaussianNucleation` when the spread is physical.

For two internal coordinates, `Nucleation2D` uses bilinear point deposition or
`GaussianInjection` with tensor-integrated weights. Boundary influx in 2D is a
number flux per unit tangential internal coordinate, multiplied by its face
width. The physical `GaussianNucleation` constructor is 1D.

The preparation and residual hooks are in `src/numerics/sectional/term_dispatch.jl`
and `integrated_sources.jl`. `examples/nucleation_sources.jl` compares the
boundary, point, and integrated Gaussian models without adding them together.
