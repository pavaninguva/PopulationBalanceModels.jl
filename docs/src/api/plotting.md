# Plotting

The optional PythonPlot extension shares one package-owned Matplotlib renderer
with repository Python plotters. See the [workflow guide](../plotting.md) for
setup, explicit statistical coordinates, composition and scientific limitations.
Rendering never runs inference or forward propagation.

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:plotting]
Public = true
Private = false
Order = [:function]
```
