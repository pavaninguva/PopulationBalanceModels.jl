# Moment methods

`moment(grid,N,k)` still measures a distribution. `MomentSet(0:K)` instead
stores and advances the raw moments themselves:

```math
\mu_k=\int_0^\infty x^k n(x)\,dx.
```

For zero unmodelled boundary flux, its weak form includes

```math
\frac{d\mu_k}{dt}=k\int x^{k-1}Gn\,dx+Jx_c^k+S_k
-\int x^k d(x)n\,dx+\mathcal A_k+\mathcal F_k,
```

```math
\mathcal A_k=\tfrac12\iint[(x+y)^k-x^k-y^k]K(x,y)n(x)n(y)\,dx\,dy,
```

```math
\mathcal F_k=\int a(x)n(x)
\left[\sum_j\nu_j(x)x_j(x)^k-x^k\right]dx.
```

For nonadditive coordinates the aggregate coordinate is obtained through
the declared AdditiveCoordinate map. Boundary losses at zero/infinity are
not inferred; supply a model with the assumed vanishing flux or explicitly
extend the equations. In particular, dissolution reaching zero needs a
particle-extinction treatment beyond these current raw-moment rules.

```julia
moments = MomentSet(0:3;coordinate=:volume)
model = PBMModel(
    Growth(PolynomialRate(((p,t,ctx)->p.g, 0.1))) +
    Nucleation((p,t,ctx)->p.J,0.0) +
    Aggregation(ConstantKernel((p,t,ctx)->p.K)))
op = compile(model,moments;closure=ExactClosure())
prob = PBMProblem(op,[1.0,1.0,2.0,6.0],(0.0,1.0);
    p=(g=0.2,J=0.1,K=0.3))
sol = simulate(prob;dt=0.001)
```

ExactClosure supports polynomial growth/removal, point nucleation,
MomentSource, constant-kernel aggregation in an additive coordinate and
self-similar polynomial-rate breakage. Removal/breakage polynomial
coefficients must be nonnegative; growth coefficients may be signed.
If an equation requires an untracked moment, compilation rejects it.
It does not silently set missing moments to zero. General distributed birth
must supply source integrals via `MomentSource(J,(k,p,t,ctx)->E_xk)`.

`QuadratureClosure(n)` uses moments 0:2n-1 to reconstruct positive quadrature
nodes/weights from scaled Hankel matrices. The same kinetic callbacks then
evaluate growth, removal, aggregation and daughter weak forms. This is a QMOM
approximation for general laws, not an exact closure. Each inversion verifies
the reconstructed moments. Empty all-zero states are supported; rank-deficient
nonempty moment vectors raise a numerical failure rather than silently reduce
node count or repair the state. Rank-adaptive inversion is future work.
`realizability` reports necessary Hankel positivity checks; these alone do not
prove realizability at singular truncated-moment boundaries.

The implemented moment families can participate in PBMSystem alongside 1D
sectional populations. Their laws share stage-level bulk/field context.
`MomentExchange(:particles,:c,k;factor=...)` conserves a selected moment-based
particle/bulk quantity using the actual moment residual.
The invariant c+factor*mu_k assumes factor is fixed in time/state; a changing
density or composition requires its additional balance contribution.
Sectional PopulationTransfer/CrossAggregation are deliberately restricted to
sectional populations; moment interactions can be written explicitly through
CustomCoupling until typed moment transfer rules are added.

MomentObservable, PopulationMoment, BulkObservable, custom observations,
Experiment, CalibrationProblem and the optional Optim/SciML adapters work with
moment models. Histograms require a declared distribution reconstruction or
measurement model; finite moments do not determine a unique PSD.
For standalone 2D distributions use `MomentObservable((kx,ky))` or a
FunctionObservable for marginals/bins.

Gunawan case 2 has an exact finite moment closure because growth is affine
in length. `Gunawan2004Case2.build_moments` expresses it through the native
model API and MomentExchange. The optional tests compare that implementation
to the separately coded continuum equations. It retains the explicit seed/c0
requirements and does not resolve the publication's invalid printed seed.
The CEJ nonlinear size laws and aggregation would require a chosen closure
and convergence study; equivalence to its full PSD is not established merely
by choosing QMOM. Multivariate QMOM/CQMOM, DQMOM and maximum-entropy PSD
reconstruction are not implemented.

## Choosing a family and weak-form rule

`compile(model,basis; method=QMOM(2), methods=TermMethods(growth=ExactMomentRule()))`
selects a reconstruction family separately from each term's integration rule.
`closure=` remains accepted for the raw-moment families. `term_methods(op)`
records the resulting rules. The family shares its reconstruction within a
residual evaluation.

`LogNormalClosure(quadrature_order=16)` requires moments 0:2, fits a lognormal,
and uses cached normal quadrature for weak forms. Empty and point-mass limits
are supported. Quadrature adequacy is checked, but a three-moment assumed
distribution cannot recover multimodality and needs its own closure study.
See `examples/moment_closures.jl` for all three implemented families.

The direct-state contract separates solver state from moment values and rates.
Its external test is an interface check, not a general DQMOM implementation.
EQMOM, CQMOM, and other families require concrete algorithms and verification.
