# Numerical checks

Use [Verification and reproduction](verification.md) for environment setup,
package tests, documentation checks and retained-result verification.

The relevant numerical assumptions and limitations are documented with each method:

- [Growth discretization and boundaries](methods/growth.md).
- [Compact transport and its independent verification](methods/compact_verification.md).
- [Moment reconstruction, closure and state validity](moment_methods.md).
- [Realizable evolution, extinction and material accounting](methods/realizable_evolution.md).
- [Parameter estimation and uncertainty](inference.md).

Case-study pages explain the references and numerical checks behind their
figures. In particular, the [moment comparison](case_studies/moment_crystallization.md)
distinguishes analytical references, reconstruction error and unresolved closure
evolution; the [protein comparison](case_studies/protein_crystallization.md)
separates posterior sampling, local uncertainty approximations and model adequacy.
