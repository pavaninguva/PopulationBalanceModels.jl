# Numerical methods

Writing a kinetic law does not determine how it should be discretized. Growth
moves particles through size-bin faces, nucleation creates particles, and a
collision removes parents and deposits products. Their rates have different
units and their numerical contributions must preserve different quantities.
Time integration then advances the sum of those contributions.

The manual treats these operations separately:

| Part of the calculation | Numerical treatment |
|:--|:--|
| [Growth and dissolution](methods/growth.md) | Signed face fluxes with Upwind, MUSCL, WENO5, Centered, or LinearUpwind, subject to grid restrictions |
| [Transport along growth trajectories](methods/characteristics.md) | Inguva2022 and InguvaBraatz2023 for eligible autonomous growth, with matched meshes and time steps |
| [Nucleation](methods/nucleation.md) | Boundary influx, point deposition, or an integrated size distribution |
| [Birth, death, and dilution](methods/sources.md) | Bin source integration and nonnegative removal hazards |
| [Aggregation and breakage](methods/collisions.md) | Fixed-pivot event counting and deposition in an additive property |
| [Time evolution](methods/time.md) | Euler, SSPRK22, SSPRK33, RK4, optional external ODE solvers, or restricted complete transport updates |
| [Axial flow and dispersion](axial_reactors.md) | Upwind axial advection and centered dispersive fluxes, assembled with intrinsic mechanisms |

Each linked page gives the physical equation, callback units, discrete treatment,
boundary assumptions, and restrictions. Alternative collision methods require
an implementation through the extension interface; fixed pivot is the supplied
sectional algorithm.

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
