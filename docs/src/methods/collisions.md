# Aggregation and breakage

Collisions change particle number through events. The numerical calculation
must count each event once, remove its parents, and place its products at sizes
that conserve the chosen material property. Reconstructing a growth face flux
does not perform any of these operations.

## Aggregation events

For an additive coordinate ``x``, a symmetric binary kernel gives

```math
\frac{\partial n(x,t)}{\partial t}
=\frac{1}{2}\int_0^x K(x-y,y,t)n(x-y,t)n(y,t)\,dy
-n(x,t)\int_0^\infty K(x,y,t)n(y,t)\,dy.
```

`Aggregation(kernel; coordinate_map=nothing, overflow=:error)` receives
`kernel(x,y,p,t,ctx)`. The kernel must be symmetric and nonnegative. For number
concentrations its units are process volume/time. Same-bin event rates contain
``K_{ii}N_i^2/2`` and remove two particles. Distinct-bin events occur at
``K_{ij}N_iN_j`` and remove one particle from each parent bin.

FixedPivot is the implemented sectional algorithm. Products are interpolated
between pivots in an additive property ``q(x)``. The weights satisfy

```math
w_l+w_r=1,\qquad w_lq(x_l)+w_rq(x_r)=q(x_i)+q(x_j).
```

This preserves one product per event and its additive property. On a diameter
grid, `AdditiveCoordinate(L->L^3,cbrt)` produces
``L_*=(L_i^3+L_j^3)^{1/3}`` and conserves volume. Adding diameters would conserve
the wrong quantity. The compiler checks an increasing positive property map
and its inverse at the grid pivots.

For example, the following checks the instantaneous balance for a population
with particles in two diameter bins:

```@example aggregation
using PopulationBalanceModels
grid = Grid1D([1.0,2.0,4.0,8.0,16.0];coordinate=:diameter)
kernel(L,M,p,t,ctx) = p.K
model = PBMModel(coalescence=Aggregation(kernel;
    coordinate_map=AdditiveCoordinate(L->L^3,cbrt)))
operator = compile(model,grid;methods=TermMethods(aggregation=FixedPivot()))
N = reshape([0.0,1.0,2.0,0.0],:,1)
dN = similar(N)
rhs!(dN,N,operator,(K=0.1,),0.0)
@assert abs(only(moment(grid,dN,3))) < 1e-12
(number_rate=sum(dN),volume_rate=only(moment(grid,dN,3)))
```

The negative number rate counts the reduction of two parents to one product.
The third diameter moment is unchanged. That conservation check concerns the
event residual; time-step size and domain truncation still matter in a solve.

## Breakage events

```math
\frac{\partial n(x,t)}{\partial t}
=\int_x^\infty a(y,t)b(x\mid y,t)n(y,t)\,dy-a(x,t)n(x,t).
```

``a`` is a parent hazard in inverse time; ``b(x\mid y,t)`` is an expected daughter
count density, not a probability density alone. Conservation requires
``\int q(x)b(x\mid y,t)\,dx=q(y)``.

`Breakage(rate,daughters; coordinate_map=...,overflow=...)` receives
`rate(x,p,t,ctx)` and `daughters(x,p,t,ctx)`, the latter returning
`(daughter_size,expected_number)` pairs. Equal binary splitting in an additive
coordinate is `((x/2,2.0),)`. A continuous daughter approximation must include
quadrature weights and multiplicity in the expected counts. FixedPivot deposits
those daughters and checks their additive-property sum against the parent.

`UniformVolumeDaughters()` describes a continuous binary law in a length
coordinate: ``b(x|y)=6x^2/y^3``. Its integral is two and its third moment is
``y^3``. Use `AdditiveCoordinate(x->x^3,cbrt)`. Moment weak forms use its
analytical daughter moments through `daughter_moment`; sectional rules use
positive Gauss–Legendre expected counts followed by fixed-pivot deposition.
The [moment case study](../case_studies/moment_crystallization.md) shows both.

## Boundaries, overflow, and conservation

Aggregation lowers total particle number; binary breakage raises it. The conserved
quantity is the additive property, provided products remain represented.
Products outside the pivot range raise an error by default.
`FixedPivot(overflow=:discard)` makes unresolved loss explicit; `:inherit` uses
the physical term's legacy overflow setting. `diagnostics` reports instantaneous
unresolved birth number and property rates, not cumulative reservoirs.

Periodic size boundaries are incompatible with collisions. Use an adequate size
range or integrate an explicit escaped-product reservoir before claiming a closed
physical balance. The current quadratic pair calculation is transparent but can
be costly on large grids. Alternative sectional collision algorithms need their
own `prepare_term` and `add_term!` implementations; the interface is not evidence
that those algorithms are provided.

The implementation is in `src/numerics/sectional/collisions.jl` and
`deposition.jl`. See [Coupled systems](../coupled_systems.md) for unlike-population
collisions, which do not use the identical-pair half factor, and
[Moment methods](../moment_methods.md) for weak-form collision integration.
