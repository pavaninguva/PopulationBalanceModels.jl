# Changelog

## Unreleased

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
- Kept the printed crystallization seed literal and checked its nonnegativity before running; removed substitute initialization from the benchmark default.
- Retained compact/ADI stability, convergence, and sharp-front evidence, including negative undershoots and the restriction to periodic constant growth.
- Updated navigation, links, versioned documentation deployment, and local contribution instructions. Simplified the README and repository root.

### Fixes

- Integrated narrow moving Gaussian nucleation sources over bins in the nanoparticle recipe, preserving the explicit source-normalization convention.
- Included boundary nucleation in the crystallization recipe's discrete material exchange.
- Supplied the operator in two-coordinate callback contexts so `moment(ctx,(kx,ky))` works consistently with other representations.
- Validated two-coordinate velocity callback results as two finite real components, with a direct error for invalid output.
- Used common evolution-argument validation for SSPRK33 and direct transport methods; invalid step limits are now rejected even for initial-state-only output.

The inherited implementation includes sectional and moment models, shared bulk
balances, population interactions, inference adapters, and restricted two-coordinate
transport. Earlier internal milestones are not treated as published releases.
