# Numerical methods

Writing a kinetic law does not determine how it should be discretized. Growth
moves particles through size-bin faces, nucleation creates particles, and a
collision removes parents and deposits products. Their rates have different
units and their numerical contributions must preserve different quantities.
Time integration then advances the sum of those contributions.

The manual treats these operations separately:

| Part of the calculation | Numerical treatment |
|:--|:--|
| [Growth and dissolution](methods/growth.md) | Signed face fluxes with Upwind, MUSCL, van Leer, WENO3, WENO5, Centered, or LinearUpwind, subject to grid restrictions |
| [Transport along growth trajectories](methods/characteristics.md) | Inguva2022 and InguvaBraatz2023 for eligible autonomous growth, with matched meshes and time steps |
| [Nucleation](methods/nucleation.md) | Boundary influx, point deposition, or an integrated size distribution |
| [Birth, death, and dilution](methods/sources.md) | Bin source integration and nonnegative removal hazards |
| [Aggregation and breakage](methods/collisions.md) | Fixed-pivot event counting and deposition in an additive property |
| [Time evolution](methods/time.md) | Euler, SSPRK22, SSPRK33, SSPRK54, RK4, optional external ODE solvers, or restricted complete transport updates |
| [Axial flow and dispersion](axial_reactors.md) | Upwind axial advection and centered dispersive fluxes, assembled with intrinsic mechanisms |

Each linked page gives the physical equation, callback units, discrete treatment,
boundary assumptions, and restrictions. Alternative collision methods require
an implementation through the extension interface; fixed pivot is the supplied
sectional algorithm.

## Available choices by term

For a one-coordinate sectional model, these are the implemented choices.
The default is listed first. Select a mechanism in TermMethods or override a
named term with `by_term`; this does not select a time integrator.

| Physical term | Supplied methods | Conditions |
|:--|:--|:--|
| `Growth` | `Upwind`, `MUSCL`, `VanLeer`, `WENO3`, `WENO5`, `Centered`, `LinearUpwind` | The first four support nonuniform 1D meshes. WENO5 needs a uniform mesh and at least six bins; Centered and LinearUpwind need uniform meshes. WENO and unlimited schemes have no positivity guarantee. |
| `Nucleation` at one size | `FixedPivot`, `BoundaryInjection`, `GaussianInjection` | Fixed pivot deposits between pivots. Boundary injection needs growth and an inward/zero velocity at that edge. Gaussian width is specified in coordinate units, independently of mesh spacing. |
| `Birth`, `DistributedNucleation` | `MidpointRule`, `GaussLegendre(n)` | Callbacks return source density per coordinate; quadrature integrates over each bin. |
| `IntegratedBirth`, `IntegratedNucleation` | `BinIntegral` | Callback supplies the bin integral directly, including its physical normalization. |
| `Death`, `Dilution` | `PointwiseLoss`, `CellAverageLoss(quadrature=GaussLegendre(n))` | Nonnegative removal hazard, evaluated at pivots or averaged in each cell. |
| `Aggregation`, `Breakage` | `FixedPivot` | Event counting and additive-coordinate product deposition; error/discard overflow policy. No other sectional collision algorithm is supplied. |
| `CustomTerm` | `NativeResidual` | User residual and its conservation/validation contract. |

Two-coordinate models have a smaller set: Growth2D supports Upwind and WENO5;
Birth and DistributedNucleation support midpoint or tensor Gauss quadrature;
Death and Dilution support pointwise or cell-average loss; Nucleation2D supports
fixed-pivot and bin-integrated Gaussian deposition. WENO5 needs uniform axes.
Other 1D reconstruction policies, sectional collisions and boundary point
injection are not general 2D implementations.

MomentSet selects a closure separately from each term's weak-form rule.
ExactClosure uses ExactMomentRule for supported polynomial/closed terms.
QMOM (QuadratureClosure), RankAdaptiveQMOM, Gamma EQMOM, DQMOM and LogNormalClosure use QuadratureMomentRule by
default, with exact rules available for eligible named terms. A closure must
provide any missing higher moments; neither a requested moment order nor a
quadrature rule creates an unavailable closure. DQMOM evolves positive weights
and weighted abscissas; the other families evolve raw moments. See
[Moment methods](moment_methods.md) for support and realizability restrictions.

For axial reactors, intrinsic terms retain their representation-specific
choices. The single-population adapter uses axial upwind fluxes. The coupled
`axial_problem` adapter also accepts `axial_scheme=MUSCL()`, `WENO3()` and other
eligible residual reconstructions. Dispersion uses centered gradient fluxes.

| Evolution choice | Applied to | Eligibility |
|:--|:--|:--|
| `ForwardEuler`, `SSPRK22`, `SSPRK33` (default), `SSPRK54`, `RK4` | The assembled ODE residual, including bulk balances | User chooses a stable step; SSP does not repair a nonmonotone spatial operator. |
| Optional SciML algorithms through `odeproblem` | The same residual | External solver controls time stepping; native stage checks/CFL control are not added. |
| `LaxWendroff`, `BeamWarming` | Complete 1D constant-growth update | Uniform mesh, declared constant AffineGrowth, growth alone; not paired with a second time integrator. |
| `Inguva2022`, `InguvaBraatz2023` | Complete 1D/2D transformed update | Eligible positive autonomous affine/separable growth, matched travel-time meshes, CFL=1 steps and aligned outputs. |
| `ExactShift` | Complete constant-growth 2D update | Integer shifts on compatible meshes, with eligible open/periodic boundaries. |
| `HOCADI` | Complete compact 2D update | Periodic constant growth only. Specified variant with dispersive undershoots; not verified as the Li paper's method. |

The Li supplementary-coefficient audit is a diagnostic implementation, not
another public selectable variable-growth HOCADI method. The method pages
explain these restrictions beside their equations and boundary treatments.

## Select a representation first

Grid1D stores ``N_i(t)=\int_{x_{i-1/2}}^{x_{i+1/2}}n(x,t)\,dx``. Grid2D stores
numbers over rectangular internal-coordinate bins. Neither stores a pointwise
density. MomentSet instead stores integrals weighted by powers of the coordinate;
its closure and weak-form choices are described in [Moment methods](moment_methods.md).
The compiler checks that a requested treatment exists for the chosen representation.

## Select a method for each term

Suppose a model has boundary nucleation and a second creation mechanism at an
interior size. Give the sources names, then select their treatments explicitly:

```@example methods
using PopulationBalanceModels

growth_rate(x,p,t,ctx) = p.G
primary_rate(p,t,ctx) = p.J
secondary_rate(p,t,ctx) = p.J/2

model = PBMModel(
    growth = Growth(rate=growth_rate),
    primary = Nucleation(rate=primary_rate,size=0.0),
    secondary = Nucleation(rate=secondary_rate,size=0.8),
)
grid = Grid1D(0.0:0.25:10.0;coordinate=:volume)
methods = TermMethods(
    growth = MUSCL(),
    by_term = (primary=BoundaryInjection(:left),secondary=FixedPivot()),
)
operator = compile(model,grid;methods)
term_methods(operator)
```

`growth=MUSCL()` applies to the growth mechanism. Entries in `by_term` apply to
the names declared in this particular model. Selection uses a named override
first, a mechanism choice second, and the default method otherwise. Unknown
names and unsupported combinations raise errors. The older `growth_scheme=`
keyword remains available; an explicit TermMethods growth choice takes precedence.

Compilation prepares the selected methods. Adding parameters and an initial
state produces a problem that can be integrated:

```@example methods
problem = PBMProblem(operator,zeros(length(grid)),(0.0,0.2);p=(G=0.2,J=1.0))
solution = simulate(problem;method=SSPRK33(),dt=0.002)
sum(solution.u[end])
```

The temporal derivative is supplied by the integrator; it is not declared as a
physical PBMModel term. SSPRK33 evaluates and sums all selected residuals at
its stages, including any [bulk ODEs](coupled_systems.md). It does not apply a
separate time update to growth and nucleation.

For an external algorithm, `prepare_term` validates the term/method combination
and prepares its cache; `add_term!` adds its residual. The
[extension guide](development.md) gives the signatures and conservation contract.

## Joint moment admissibility

Componentwise positivity does not guarantee realizability of raw moments. The
opt-in [MeasureEuler policy](methods/realizable_evolution.md) evolves positive
measures and checks closure admissibility before physical rates and at accepted
states. Its compatibility matrix distinguishes raw moment families, sectional
updates and the conservative low-order axial guarantee.
