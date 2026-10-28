# Problems, solvers and integrations

Transient problems provide initial conditions and a time span; stationary
problems provide a numerical guess. Native integration, optional external ODE
solvers and stationary root solvers retain their own compatibility and state
checks. `suggest_dt` recommends a local explicit step; it is not an adaptive
controller.

See [Time integration](../methods/time.md),
[Stationary population balances](../steady_state.md) and the
[optional-dependency setup](../verification.md#Optional-integrations).
The shared optional `fit` entry point is used by the
[parameter-estimation interfaces](estimation.md).

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:problems]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
