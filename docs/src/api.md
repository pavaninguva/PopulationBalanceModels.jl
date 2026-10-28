# API reference

Choose a topic below, or use the symbol index. Reference pages
contain constructor and operation docstrings; the manual chapters explain how
to assemble a complete calculation. Start with the
[first model](getting_started.md), [first MAP fit](inference.md#A-complete-first-fit)
or [first UQ calculation](uncertainty_quantification.md#A-first-calculation).

## Model and solve

- [Models and representations](api/models.md): physical terms, rate laws,
  grids, moment bases, boundary conditions and model declarations.
- [Numerical methods and compilation](api/numerics.md): term discretization,
  reconstructions, moment methods, compatibility checks and prepared operators.
- [Problems, solvers and integrations](api/problems.md): transient and stationary
  problems, time integration, root solving, continuation and solver entry points.

## Couple and observe

- [Coupling and accounting](api/coupling.md): named populations, bulk balances,
  interactions, state accessors and cumulative accounting.
- [Observations and discrepancies](api/observations.md): measurement operators,
  conditional CDFs/quantiles and deterministic comparison blocks.

## Parameter estimation

[Estimation interfaces](api/estimation.md) cover parameters, priors, likelihoods,
MAP diagnostics, uncertainty, derivatives, profiles and posterior sampling.
Bayesian and legacy frequentist/discrepancy interfaces retain their distinct
statistical meanings; see the [estimation guide](inference.md).

## Forward uncertainty quantification

[UQ interfaces](api/uncertainty.md) cover uncertain inputs, forward scenarios,
MC propagation, PCE construction/evaluation and summaries. The
[UQ guide](uncertainty_quantification.md) develops a complete data-free calculation.

## Numerical extensions

Public extension hooks are documented with their
[representations](api/models.md), [numerical methods](api/numerics.md) and
[coupling interfaces](api/coupling.md). The
[developer guide](development.md) explains their dispatch and validation contracts.

## Exported docstrings

This index links to each canonical reference entry, including documented
signatures of operations shared by several subsystems.

```@index
Pages = Main.APIReference.pages()
Order = [:constant, :type, :function, :macro]
```
