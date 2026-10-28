# Numerical methods and compilation

`TermMethods` selects per-mechanism defaults and per-name overrides. `compile`
checks compatibility and prepares the selected implementation for a grid or
moment representation. Spatial residuals, moment closures and complete
space–time updates have different contracts; a constructor alone does not
establish that a combination is supported.

The [method guide](../numerical_methods.md), [moment guide](../moment_methods.md)
and [compact-method guide](../methods/compact.md) explain those restrictions.
Public extension hooks are described in [Contributing and internals](../development.md).

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:numerics]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
