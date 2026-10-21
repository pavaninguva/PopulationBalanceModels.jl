# Changelog

## Unreleased

- Added uniform-grid 1D Centered and LinearUpwind growth reconstructions, and restricted conservative LaxWendroff and BeamWarming space–time updates with compile-time eligibility and runtime CFL checks.
- Added ForwardEuler, SSPRK22, and classical RK4 alongside SSPRK33 for the assembled residual, preserving stage checks and simultaneous bulk integration.
- Added a constant-growth analytical benchmark comparing spatial/time-method combinations, including exact CFL=1 shifts, negative undershoots, unstable combinations, refinement, and measured elapsed work.
- Added AxialGrid and AxialDispersion for prescribed constant axial flow and dispersion with one or two intrinsic coordinates, conservative nonuniform-mesh fluxes, Danckwerts inlet/zero-gradient outlet, and periodic or closed alternatives.
- Reworked cooling crystallization around stated nonnegative initial conditions, a finer sectional distribution reference, spatial/time refinement, independent moment verification, and regenerated figures. Retained the printed initial polynomial literally under reported_initial_density, with the old recipe name available for compatibility.
- Added public Inguva2022 and InguvaBraatz2023 transformed transport methods, structured affine/separable growth laws, physical travel-time meshes, and compilation/runtime eligibility checks. Retained ExactShift for its existing constant-velocity interface.
- Reworked the growth-fitting study around analytical point densities with additive Gaussian noise, two fitted coefficients, continuous plots, lognormal versus timescale-bounded uniform priors, and chain-level sampling evidence.
- Exposed per-chain BFMI, acceptance rates, step sizes, energy/transition statistics, warmup statistics and Monte Carlo mean errors through diagnostics(samples).
- Renamed the estimation chapter and navigation to Parameter estimation.
- Added a shared Bayesian EstimationProblem for MAP, optional full-posterior Laplace curvature, and optional AdvancedHMC NUTS sampling, with explicit prior coordinates and transform Jacobians.
- Added multinomial and exposed Poisson count likelihoods, normalized Gaussian likelihoods with inferred noise/covariance, and contrast handling for normalized fractions.
- Added shared/local/fixed parameter declarations, uncertain initial-condition setup, observation responses, joint nuisance covariance, likelihood/posterior profiles, prior comparisons, and numerical refits.
- Replaced the manual's introductory trivial kinetic fit with a synthetic size-dependent growth study based on the 2022 affine-growth benchmark, including parameter correlations, sampling diagnostics and numerical-refinement evidence.
- Preserved the existing Gaussian CalibrationProblem interface. Reused the Distributions/StatsAPI fit and loglikelihood function bindings to avoid ambiguous imports.
- Added Random as a standard-library dependency and AdvancedHMC/MCMCDiagnosticTools as optional extension dependencies.

- Added fixed correlated Gaussian observation blocks, forward-solver adapters, predictions, and SVD-based local covariance calculations.
- Added fit diagnostics, checked joint confidence ellipsoids, multiple-start fitting, and nuisance-reoptimized likelihood profiles, with explicit known-error/estimated-scale assumptions.
- Added a synthetic two-rate uncertainty example with residuals, mapped confidence regions, profiles, and solver-refinement checks.
- Added a native-moment crystallization study using reported paracetamol kinetics and explicitly reconstructed monotonic cooling inputs; documented the additional extinction and distribution information needed for the full cycling figure.
- Added visible simulation and accuracy figures, including clearly labelled retained demonstration results. Applied LaTeX typography and removed plot grid lines.
- Documented the DLVO-to-collision interface and added unit-specified inverse-Debye-length and unequal-sphere van der Waals helpers to the LNP recipe; retained the unresolved full attachment convention.


- Renamed the package and optional extensions to PopulationBalanceModels, preserving its UUID and version.
- Made `docs/src/` the authoritative manual and consolidated standalone architecture, numerical, validation, and publishing reports.
- Rewrote the model tutorials around equations, parameters, rate callbacks, boundaries, numerical treatment, and observations, with executable code blocks.
- Organized the case studies around three physical problems: batch cooling crystallization, lipid nanoparticle formation, and growth in two particle dimensions.
- Separated the mathematical and numerical treatment of growth, nucleation, local sources and removal, collisions, and time integration.
- Added keyword term declarations while retaining positional constructors, including keyword forms for two-coordinate growth and nucleation.
- Shared crystallization rate callbacks and parameters between sectional and exactly closed moment builders.
- Made the two-coordinate benchmark and its analytical and mapped references use the same explicit parameters. Added area-weighted density errors and actual physical cell edges to exported fields and plots.
- Replaced the published manual's constant-growth comparison with the coordinate-dependent growth problem and its analytical solution. Retained earlier diagnostics outside the case studies.
- Added zero-particle initialization, the full 4320-minute nanoparticle output schedule, adaptive-solver export support, and peak-scaled intensity outputs.
- Kept the printed crystallization initial polynomial literal; the current benchmark uses a separately labelled nonnegative initial condition and does not claim to repair the published coefficients.
- Retained compact/ADI stability, convergence, and sharp-front evidence, including negative undershoots and the restriction to periodic constant growth.
- Updated navigation, links, versioned documentation deployment, and local contribution instructions. Simplified the README and repository root.

### Fixes

- Used SciML's parameterless sentinel for empty parameter tuples in the ODE adapter, preventing symbolic-remake recursion while preserving the original empty NamedTuple inside PBM callbacks.
- Kept independent Gaussian measurement covariance diagonal during factorization, avoiding dense storage and cubic work for large density datasets while retaining likelihood normalization.
- Integrated narrow moving Gaussian nucleation sources over bins in the nanoparticle recipe, preserving the explicit source-normalization convention.
- Included boundary nucleation in the crystallization recipe's discrete material exchange.
- Supplied the operator in two-coordinate callback contexts so `moment(ctx,(kx,ky))` works consistently with other representations.
- Validated two-coordinate velocity callback results as two finite real components, with a direct error for invalid output.
- Used common evolution-argument validation for SSPRK33 and direct transport methods; invalid step limits are now rejected even for initial-state-only output.

The inherited implementation includes sectional and moment models, shared bulk
balances, population interactions, inference adapters, and restricted two-coordinate
transport. Earlier internal milestones are not treated as published releases.
