# Numerical methods

## Representation, residual, and evolution

A grid stores bin-integrated numbers; a moment basis stores selected integrals.
Neither determines the algorithm used for each physical mechanism. `compile`
selects and prepares term methods, and `simulate` chooses how the resulting
operator advances in time. In 1D a Growth residual is the face-flux difference
``\dot N_i=F_i-F_{i+1}``; all source and collision contributions add to this residual.

## Growth fluxes and time integration

`Upwind()` uses the upwind piecewise-constant density and supports nonuniform
grids and either sign of growth. `MUSCL()` reconstructs limited slopes with
nonuniform-grid geometry in 1D. `WENO5()` uses nonlinear fifth-order interior
reconstruction on a uniform grid with at least six bins; the order of a complete
solve also depends on boundaries, smoothness, and time integration. WENO5 has
no proven positivity limiter here. Size boundaries act on growth fluxes and
must be chosen independently of the reconstruction.

`simulate(problem; dt, method=SSPRK33(), saveat=...)` evaluates the complete
residual at three stages of a third-order SSP Runge–Kutta step. Population and
bulk states advance simultaneously. The maximum step is shortened at requested
output times. The solver checks finite values and, by default, negative states
at stages; `negative_tolerance=1e-12` allows roundoff without clipping. A failed
check raises `NumericalFailure`.

`suggest_dt(op,N,p,t; cfl=0.5)` bounds explicit upwind/source updates heuristically.
It is not a positivity guarantee for WENO or arbitrary custom terms. Refine both
grid and time step; stiff collision/source systems may require a suitable external
solver through `odeproblem`. That adapter exports the residual and does not add
a positivity treatment. `HOCADI` and `ExactShift` are direct evolution methods
for eligible 2D transport, discussed in the [transport case study](case_studies/transport.md).

## Source and collision discretization

WENO reconstructs **growth fluxes**. It does not discretize every term in the
PBM. Model declarations select physical laws; source integration and particle
event deposition determine how those laws enter the discrete residual.

| Mechanism | Current numerical treatment | Main contract |
|---|---|---|
| Boundary nucleation | `InflowFlux(J)` enters the conservative growth flux at the first/last face | J is inward number flux; do not divide by G |
| Point nucleation | `Nucleation(J,x0)` deposits between adjacent pivots | Preserves number and the first raw coordinate moment within the pivot range |
| Resolved distributed nucleation | `DistributedNucleation(J,shape)` uses midpoint bin quadrature | Choose total-rate normalization or density amplitude explicitly |
| Integrated distributed nucleation | `IntegratedNucleation(J,fraction)` uses supplied bin integrals | No midpoint sampling of a potentially unresolved narrow peak |
| Gaussian nucleation | `GaussianNucleation(J;center,width,truncation)` integrates each bin in standardized coordinates | Width is a physical standard deviation, independent of mesh size |
| Distributed birth | `Birth(source)` midpoint quadrature, or `IntegratedBirth(integral)` | Source is a number density rate, or an actual bin-integrated rate |
| Death / dilution | Local nonnegative hazard times bin number | Advanced simultaneously with all other residual contributions |
| Aggregation | Symmetric pair-event quadrature and fixed-pivot product deposition | Half factor for identical pairs; additive-property map; explicit overflow policy |
| Breakage | Parent hazard plus weighted expected daughter counts | Daughter quadrature weights must include multiplicity and conserve the declared property |

The reference solver evaluates all these mechanisms and all bulk states at
each SSPRK(3,3) stage. It does not silently split a stiff source from growth.
An implicit/IMEX policy and alternative cell-average/flux collision schemes
are later extensions. The current collision implementation is O(n²) and
prioritizes transparent bookkeeping over cached performance.

For growth into the domain at size a, boundary nucleation is

```math
(Gn)(a,t)=J(t).
```

At zero size this contributes J to the continuum number moment and zero to
higher raw moments. In a finite-volume distribution the first bin represents
that influx at its finite pivot, so higher discrete source moments have a
mesh-dependent error. Direct inward flux avoids a singular `n=J/G` evaluation;
physical transport away from a zero-growth boundary still needs a consistent
model. An `InflowFlux` boundary requires a Growth term.

Example alternatives for **one** creation mechanism:

```julia
boundary = compile(PBMModel(Growth(G)), grid;
    bc=SizeBC(InflowFlux(J),NoFlux()))
point = compile(PBMModel(Growth(G)+Nucleation(J,xc)), grid)
gaussian = compile(PBMModel(Growth(G)+GaussianNucleation(J;
    center=xc,width=sigma,truncation=:renormalize)), grid)
```

Do not activate these alternatives together for the same physical nucleation
event. A finite Gaussian width changes the model's higher source moments; it
is not just a harmless way to draw a Dirac delta. Hold a physical width fixed
while refining the grid. If width is numerical regularization, study its limit
separately from mesh refinement.

`:renormalize` creates exactly J particles over the finite resolved domain.
`:physical` retains J times the Gaussian probability inside that domain.
The Gaussian integrator uses subdivided eight-point Gauss–Legendre quadrature
over ±12 standard deviations; omitted normal probability is below 4e-33.
It integrates narrow peaks even when no bin center samples the peak.
Neither policy guarantees an exact nonlinear particle-mass moment on coarse
pivots. Bulk exchange uses the actual discrete source moment, and continuum
moment accuracy must be checked by refinement.

The 2D FV path currently supports growth, midpoint birth/removal,
bilinear point deposition and midpoint distributed creation. Its boundary
InflowFlux is a number flux **per unit tangential internal coordinate**;
the face integral multiplies by the tangential bin width. It is distinct from
the total 1D boundary number flux. The GaussianNucleation constructor is 1D; the 2D point-source
GaussianInjection policy separately integrates tensor Gaussian fractions. General 2D aggregation/breakage product-shape rules are deferred.


## Fixed-pivot event bookkeeping

For an identical-population aggregation event ``i+j\to k``, an unordered pair
is counted once. Distinct-bin events have rate ``K_{ij}N_iN_j``; same-bin events
have rate ``K_{ii}N_i^2/2`` and remove two particles. The kernel must be symmetric
and nonnegative. The implementation evaluates pair deposition on the fly with
quadratic cost in bin count. The supported sectional collision algorithm is
fixed pivot; the extension interface does not imply other algorithms exist.

A product of additive size ``q_*`` between pivots ``q_l,q_r`` receives weights
``w_l=(q_r-q_*)/(q_r-q_l)`` and ``w_r=1-w_l``. These preserve one product birth
and its additive property. On a diameter grid supply
`AdditiveCoordinate(L->L^3,cbrt)` so the product is
``(L_1^3+L_2^3)^{1/3}``; adding diameters would conserve the wrong quantity.
The compiler checks positivity, monotonicity, and map/inverse consistency.

For breakage, `daughters(x,p,t,ctx)` returns `(daughter_size, expected_number)`
pairs. A binary equal split in an additive coordinate is `((x/2,2.0),)`.
The weighted daughter property must equal the parent's. A normalized daughter
probability without multiplicity is insufficient. Continuous daughter quadrature
must include both integration weights and expected daughter counts.

Out-of-range deposition raises an error by default. `FixedPivot(overflow=:discard)`
makes loss explicit; `:inherit` uses the physical term's legacy overflow policy.
`diagnostics` reports instantaneous unresolved birth number and additive-size
rates, not cumulative overflow reservoirs. Closed balances require an adequate
domain or an explicitly integrated reservoir. Number decreases under aggregation
and can increase under breakage; additive-property conservation is the invariant.

## Named methods and hooks


Declare the physical law once. Select an algorithm by mechanism, then override
individual named terms where needed.

```@example methods
using PopulationBalanceModels

grid = Grid1D(0.0:0.25:10.0; coordinate=:volume)
model = PBMModel(
    growth = Growth((v, p, t, ctx) -> p.G),
    primary = Nucleation((p, t, ctx) -> p.J, 0.0),
    secondary = Nucleation((p, t, ctx) -> p.J / 2, 0.8),
    loss = Death((v, p, t, ctx) -> 0.001 * v^2),
)

methods = TermMethods(
    growth = MUSCL(),
    death = CellAverageLoss(quadrature=GaussLegendre(4)),
    by_term = (
        primary = BoundaryInjection(:left),
        secondary = GaussianInjection(width=0.05, truncation=:renormalize),
    ),
)

operator = compile(model, grid; methods)
problem = PBMProblem(operator, zeros(length(grid)), (0.0, 0.2);
    p=(G=0.2, J=1.0))
solution = simulate(problem; dt=0.002)
@assert sum(solution.u[end]) > 0.0
term_methods(operator)
```

The choice order is a named term override, a mechanism policy, then the default.
Gaussian regularization preserves the declared rate when renormalized over the
domain, but a finite width changes higher moments relative to a point source.

Collision terms use fixed pivot. Other algorithms can extend the public
`prepare_term` and `add_term!` contract in [Development](development.md).


The implemented 1D combinations are:

| Physical term | Default | Other selectable methods |
|:--|:--|:--|
| Growth | Upwind | MUSCL, uniform-grid WENO5 |
| Birth, DistributedNucleation | MidpointRule | GaussLegendre(order) |
| IntegratedBirth, IntegratedNucleation | BinIntegral | User-supplied bin integral |
| Death, Dilution | PointwiseLoss | CellAverageLoss(quadrature=GaussLegendre(4)) |
| Nucleation | FixedPivot | BoundaryInjection, GaussianInjection |
| Aggregation, Breakage | FixedPivot | External methods through public hooks |
| CustomTerm | NativeResidual | Explicit external methods |

`PointwiseLoss` samples the hazard at the pivot. `CellAverageLoss` integrates
the hazard over the bin against its piecewise-constant density. Neither choice
changes the time integrator into an exponential or implicit update. Death
removes particles physically; dilution changes a concentration's process-volume
basis and needs a consistent bulk dilution balance.

`growth_scheme=` remains a shorthand; an explicit `TermMethods` growth choice
takes precedence. Named overrides precede class choices and defaults. Unknown
names and unsupported combinations raise errors at preparation. See
[Development](development.md) for `prepare_term` and `add_term!` contracts.
