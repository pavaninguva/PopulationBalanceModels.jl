# Numerical methods

Consider a number density ``n(x,t)`` on a single internal coordinate ``x``:

```math
\frac{\partial n(x,t)}{\partial t}
+\frac{\partial [G(x,t)n(x,t)]}{\partial x}
=B(x,t)+J(t)\delta(x-x_c)-[d(x,t)+\lambda(t)]n(x,t)
+\mathcal A[n](x,t)+\mathcal F[n](x,t).
```

The terms have different numerical treatments. Growth transports existing
particles through cell faces. Nucleation creates particles at a boundary, point,
or distribution of sizes. Aggregation and breakage count events and deposit
products. Time integration advances the sum of those contributions and any
coupled bulk balances.

A sectional state stores ``N_i(t)=\int_{x_{i-1/2}}^{x_{i+1/2}}n(x,t)\,dx``.
`Grid1D` defines those cells; `density(grid,N)` divides bin numbers by widths.
A `MomentSet` instead stores integrals weighted by powers of the coordinate.
Choose the representation before specifying term methods.

Read the treatments separately:

- [Growth](methods/growth.md): face fluxes, reconstruction, boundaries, and two-coordinate transport.
- [Nucleation](methods/nucleation.md): boundary influx, point deposition, distributed sources, and normalization.
- [Birth, death, and dilution](methods/sources.md): bin integration and removal hazards.
- [Aggregation and breakage](methods/collisions.md): event counting and additive-property deposition.
- [Time integration](methods/time.md): simultaneous stages, time-step restrictions, compact/ADI, and exact shifts.
- [Moment methods](moment_methods.md): weak forms and closure assumptions.

## Declare physics, then choose methods

```@example methods
using PopulationBalanceModels

growth_rate(x,p,t,ctx) = p.G
primary_rate(p,t,ctx) = p.J
secondary_rate(p,t,ctx) = p.J / 2

model = PBMModel(
    growth = Growth(rate=growth_rate),
    primary = Nucleation(rate=primary_rate, size=0.0),
    secondary = Nucleation(rate=secondary_rate, size=0.8),
)
grid = Grid1D(0.0:0.25:10.0; coordinate=:volume)
methods = TermMethods(
    growth=MUSCL(),
    by_term=(primary=BoundaryInjection(:left), secondary=FixedPivot()),
)
operator = compile(model,grid; methods)
problem = PBMProblem(operator,zeros(length(grid)),(0.0,0.2);p=(G=0.2,J=1.0))
solution = simulate(problem;method=SSPRK33(),dt=0.002)
@assert sum(solution.u[end]) > 0
term_methods(operator)
```

`Growth(rate=...)` supplies a coordinate velocity; `Nucleation(rate=...,size=...)`
supplies a total creation rate and location. The named keyword constructors and
the original positional forms are equivalent. Descriptive callback functions
make the arguments and units visible without embedding long anonymous expressions.

`TermMethods` chooses a method by mechanism class, with `by_term` overrides for
individual model names. Precedence is named override, class choice, then default.
Unknown names and unsupported combinations raise errors. `growth_scheme=` remains
a compatible shorthand; an explicit growth policy takes precedence.

`compile` resolves policies and caches; `rhs!` adds their contributions to a
common derivative. `simulate` selects time integration independently. The
[developer guide](development.md) documents `prepare_term`, `add_term!`, and the
moment-family hooks for extending those steps.
