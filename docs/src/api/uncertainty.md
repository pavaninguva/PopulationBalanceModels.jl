# [Forward uncertainty quantification](@id Forward-UQ-reference)

These constructors and operations propagate a declared input law through
eligible forward problems without observations or fitting. Existing posterior
draws are one possible input source. MC and non-intrusive PCE use the same
scenario and observation definitions, with explicit solve budgets.
PCE supports SVD regression and probability-quadrature projection, an extensible
orthonormal-basis contract, and validation/refinement before selecting a surrogate.

The [UQ guide](../uncertainty_quantification.md) provides a complete first run,
input-coordinate conventions, failure handling and the distinction between
input uncertainty, sampling error and surrogate error. The
[milling case](../case_studies/milling_uncertainty.md) illustrates a larger workflow.

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:uq]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
