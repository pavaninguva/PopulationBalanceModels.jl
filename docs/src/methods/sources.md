# Birth, death, and dilution

A local source adds entities to an internal-coordinate interval; a removal
hazard acts on those already present. `Death` and `Dilution` both supply a
nonnegative sink. Their physical interpretation and associated bulk balances
are part of the model declaration.

## Distributed birth

```math
\frac{\partial n(x,t)}{\partial t}=B(x,t),\qquad
\frac{dN_i}{dt}=\int_{x_{i-1/2}}^{x_{i+1/2}}B(x,t)\,dx.
```

`Birth(source=B)` receives `B(x,p,t,ctx)` in number/(process volume coordinate
time). The source must be nonnegative. MidpointRule multiplies its centre value
by the bin width; GaussLegendre integrates within the bin. An unresolved narrow
source needs an integral, not a higher time-integration order.

`IntegratedBirth(integral=F)` receives `F(lo,hi,p,t,ctx)` in number/(process volume
time). Its default BinIntegral policy adds that value directly. No extra bin
width is applied. Use this form when analytical source integrals are known.

## Removal hazards

```math
\frac{\partial n(x,t)}{\partial t}=-d(x,t)n(x,t).
```

`Death(rate=d)` supplies a nonnegative hazard `d(x,p,t,ctx)` in inverse time.
PointwiseLoss samples the hazard at a pivot and contributes ``-d_iN_i``.
`CellAverageLoss(quadrature=GaussLegendre(4))` averages the hazard over the bin
against the piecewise-constant density. Both act on the current stage state.
Neither exponentiates the hazard or chooses an implicit time update.

Particle removal may carry material out of the modeled system. If it should
return material to a dissolved state, specify that exchange; the term name
alone does not supply a mass balance.

## Dilution and the concentration basis

`Dilution(rate=lambda)` implements the concentration sink

```math
\frac{\partial n(x,t)}{\partial t}=-\lambda(t)n(x,t),\qquad
\lambda\geq0.
```

The constructor does not infer vessel flows or volume changes. Two common
choices have different inventory meanings:

- **Expansion without particle loss:** ``\lambda=\dot V/V`` for increasing
  volume and a particle-free feed. With no other mechanisms,
  ``V\int n\,dx`` is constant even though the concentration decreases.
- **Well-mixed withdrawal at fixed volume:** ``\lambda=Q_{\mathrm{out}}/V=1/\tau``.
  Particles leave with the outlet stream, so the total inventory decreases
  unless a separate inlet or birth term supplies them. This is the sink in
  the [stationary crystallizer](../steady_state.md).

For a varying-volume, well-mixed vessel with ``\dot V=Q_{\mathrm{in}}-Q_{\mathrm{out}}``,
the flow-only inventory equation is

```math
\frac{d[V(t)n(x,t)]}{dt}
=Q_{\mathrm{in}}n_{\mathrm{in}}(x,t)-Q_{\mathrm{out}}n(x,t).
```

Dividing by V gives the inlet source and the combined concentration loss
``-(Q_{\mathrm{out}}/V+\dot V/V)n``. Use that consistent balance rather than
adding the fixed-volume withdrawal and expansion interpretations twice.
Supply feed populations and dissolved-species balances separately, for example
through `Birth` and `PBMSystem.bulk_rhs`. Neither `Dilution` nor its keyword
label creates a feed or a solute equation. The hazard callback may depend on
size, but homogeneous volume change and well-mixed withdrawal normally do not.

The [LNP case](../case_studies/nanoparticles.md) combines prescribed solvent
history with a volume-consistent particle/free-lipid balance. Death and dilution
need not have the same bulk exchange policy. Their implementation hooks are in
`src/numerics/sectional/local_sources.jl` and `term_dispatch.jl`.
