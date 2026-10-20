# Birth, death, and dilution

A local source adds particles directly to a size interval. A removal hazard
acts on the particles already there. Dilution has the same mathematical loss
form as a removal hazard, but represents a changing process-volume basis and
must also be applied to other concentrations.

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

For a material element of process volume ``V(t)``, concentration dilution is

```math
\frac{\partial n(x,t)}{\partial t}=-\lambda(t)n(x,t),\qquad
\lambda(t)=\frac{1}{V(t)}\frac{dV(t)}{dt}.
```

`Dilution(rate=lambda)` has the same discrete hazard form as Death but a different
physical meaning. Apply the consistent dilution rate to dissolved concentrations
as well. Without physical production/loss, ``V(t)\int n(x,t)\,dx`` is conserved,
not the number concentration alone. A size-dependent dilution callback is
accepted, but uniform process-volume dilution normally has no size dependence.

The [LNP case](../case_studies/nanoparticles.md) combines prescribed solvent
history with a volume-consistent particle/free-lipid balance. Death and dilution
need not have the same bulk exchange policy. Their implementation hooks are in
`src/numerics/sectional/local_sources.jl` and `term_dispatch.jl`.
