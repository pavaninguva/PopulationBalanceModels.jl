# Numerical methods and compilation

`TermMethods` selects per-mechanism defaults and per-name overrides. `compile`
checks compatibility and prepares the selected implementation for a grid or
moment representation. Spatial residuals, moment closures and complete
space–time updates have different contracts; a constructor alone does not
establish that a combination is supported.

The [method guide](../numerical_methods.md), [moment guide](../moment_methods.md)
and [compact-method guide](../methods/compact.md) explain those restrictions.
Public extension hooks are described in [Contributing and internals](../development.md).

For moment formulations, `compile(model,basis; moment_method=QMOM())` infers
the node count from the basis and automatically mixes analytical contributions
with reconstruction where needed. [`moment_plan`](@ref) reports per-equation
dependencies and missing moments. Legacy family aliases and explicit weak-form
overrides remain available; see [the moment workflow](../moment_methods.md).

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:numerics]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
