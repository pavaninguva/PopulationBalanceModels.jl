# Parameter estimation

Bayesian estimation shares one statistical model across MAP, uncertainty and
sampling. Direct distributions always declare physical priors; every parameter
requires an explicit transform. MAP, local covariance and sampling use the same
declared-coordinate density, including physical-prior Jacobians exactly once.
Full-Hessian Laplace and sensitivity-based
Gauss–Newton approximations have separate assumptions and diagnostics.
The older Gaussian-calibration and deterministic-discrepancy interfaces retain
their distinct interpretations.

Begin with the [complete first MAP fit](../inference.md#A-complete-first-fit).
The [estimation guide](../inference.md) covers likelihoods, derivatives, profiles,
posterior sampling and prediction budgets. Eligible stationary models use the
same interfaces through [StationarySolver](../steady_state.md#Observe-and-estimate-parameters).

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:estimation]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
