# Models and representations

Terms describe physical mechanisms; their rate functions describe dependence on
coordinates, parameters, time and the current context. Grids store bin integrals.
`MomentSet` specifies a moment basis, while the chosen moment method determines
the solver state. Boundary conditions and declared growth-law structure also
belong to the model specification.

Start with [Defining models](../defining_models.md). For a complete program, see
[Getting started](../getting_started.md); for numerical policy choices, see
[Numerical methods](../numerical_methods.md).

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:models]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
