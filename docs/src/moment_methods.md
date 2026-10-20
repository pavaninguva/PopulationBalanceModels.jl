# Moment methods

Sometimes the quantities of interest are total number, mean size, or total
particle material rather than every bin of a distribution. A moment model
advances integrals directly:

```math
\mu_k(t)=\int_0^\infty x^kn(x,t)\,dx.
```

This differs from calling `moment(grid,N,k)` on a sectional solution. In a
moment model the moments are the dynamical variables. We need equations for
their rates, and often an assumption to express those rates using finitely many
tracked moments.

## Integrate the balance against a particle property

For vanishing unmodelled boundary flux, the growth, source, removal, and collision
contributions are

```math
\frac{d\mu_k(t)}{dt}
=k\int_0^\infty x^{k-1}G(x,t)n(x,t)\,dx+J(t)x_c^k+S_k(t)
-\int_0^\infty x^kd(x,t)n(x,t)\,dx+\mathcal A_k(t)+\mathcal F_k(t),
```

with additive-coordinate aggregation

```math
\mathcal A_k(t)=\frac12\int_0^\infty\int_0^\infty
[(x+y)^k-x^k-y^k]K(x,y,t)n(x,t)n(y,t)\,dx\,dy
```

and breakage

```math
\mathcal F_k(t)=\int_0^\infty a(x,t)n(x,t)
\left[\sum_j\nu_j(x,t)x_j(x,t)^k-x^k\right]dx.
```

For a nonadditive coordinate, the aggregate size is obtained through the declared
AdditiveCoordinate map. Boundary losses at zero or infinity are not inferred.
In particular, dissolution reaching zero needs an extinction treatment beyond
the current raw-moment rules.

## Use an exact closure when the equations close

Affine growth and constant-kernel aggregation in an additive coordinate produce
a closed hierarchy through any chosen integer order. Declare the polynomial
structure so the compiler can find its moment dependencies:

```@example moments
using PopulationBalanceModels
basis = MomentSet(0:3;coordinate=:volume)
model = PBMModel(
    growth=Growth(rate=PolynomialRate(((p,t,ctx)->p.g,0.1))),
    primary=Nucleation(rate=(p,t,ctx)->p.J,size=0.0),
    aggregation=Aggregation(ConstantKernel((p,t,ctx)->p.K)),
)
operator = compile(model,basis;closure=ExactClosure())
problem = PBMProblem(operator,[1.0,1.0,2.0,6.0],(0.0,1.0);
    p=(g=0.2,J=0.1,K=0.3))
solution = simulate(problem;dt=0.001)
vec(solution.u[end])
```

ExactClosure also supports polynomial removal, self-similar polynomial-rate
breakage, point creation, and MomentSource. Removal and breakage polynomial
coefficients must be nonnegative; growth coefficients may be signed. Compilation
rejects equations that need untracked moments. A distributed source must provide
its raw moments through `MomentSource(J,(k,p,t,ctx)->E_xk)`.

The [cooling-crystallization model](case_studies/crystallization.md) has an exact
four-moment closure because its growth is affine in length. Its
`build_moments` routine uses the same kinetic parameters and nucleation callback
as the sectional builder, and couples the third moment to dissolved concentration.
This supplies a useful check independent of the sectional discretization.

## Approximate an unclosed hierarchy

General size-dependent rates can require integrals that the tracked moments do
not determine. The implemented approximation families make different assumptions:

- `QMOM(n)` (also `QuadratureClosure(n)`) reconstructs positive quadrature nodes
  and weights from moments 0:2n−1. The rate callbacks are evaluated through their
  weak forms at those nodes. Scaled Hankel inversion is checked by reconstructing
  the input moments. Empty states are supported; rank-deficient nonempty states
  fail rather than silently changing node count or repairing moments.
- `LogNormalClosure(quadrature_order=16)` fits a lognormal density to moments
  0:2 and integrates its weak forms with cached normal quadrature. Empty and
  point-mass limits are supported. A three-moment assumed density cannot recover
  multimodality, even if its weak-form quadrature is accurate.

`realizability` reports necessary Hankel positivity checks. These do not prove
realizability at singular truncated-moment boundaries. A closure needs its own
accuracy and convergence assessment; selecting QMOM does not establish agreement
with a full size distribution for nonlinear nanoparticle kinetics.

## Select weak-form rules and couple populations

The reconstruction family and each term's integration rule can be chosen
separately. For example:

```julia
operator = compile(model,basis;method=QMOM(2),
    methods=TermMethods(growth=ExactMomentRule(),
                        aggregation=QuadratureMomentRule()))
```

Here polynomial growth is integrated analytically while aggregation uses the
reconstructed quadrature. `closure=` remains available for raw-moment families.
The reconstruction is shared within a residual evaluation; `term_methods(operator)`
records the rules used. `examples/moment_closures.jl` exercises all three families.

Moment populations can share a PBMSystem with 1D sectional populations.
`MomentExchange(:particles,:c,k;factor=...)` derives bulk exchange from the actual
moment rate. Its invariant ``c+\mathrm{factor}\,\mu_k`` assumes a fixed conversion
factor; a changing density or composition needs its additional balance.
PopulationTransfer and CrossAggregation currently require sectional populations;
moment interactions can be written explicitly through CustomCoupling.

Use MomentObservable, PopulationMoment, or a custom observation to measure the
result. Histogram predictions require a declared reconstruction or measurement
model. Optim and SciML adapters can use moment problems, subject to the same
closure and boundary restrictions.

The [extension interface](development.md) also supports solver states other than
raw moments, with explicit maps for moment values and their rates. Its tested
direct-state example is an extension contract, not a general DQMOM solver.
EQMOM, CQMOM, multivariate moment closures, and maximum-entropy reconstruction
remain future implementations.
