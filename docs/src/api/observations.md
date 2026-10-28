# Observations and discrepancies

Observation operators map eligible forward solutions to measurements. Explicit
conditioning windows, weighting and reconstruction assumptions remain part of
that map. The same operators serve transient and stationary calculations.

Conditional CDFs and quantiles integrate partial cells of bin populations and
reject an empty window. Deterministic residual/Wasserstein comparison blocks
have different statistical meanings from a measurement likelihood. See
[Parameter estimation](../inference.md) for both workflows, and
[Likelihoods and Bayesian estimation](estimation.md) for `Measurement` and
`DistributionObservable`.

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:observations]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
