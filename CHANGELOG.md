# Changelog

## Unreleased

- Separated growth, nucleation, source/removal, collision, and time-integration documentation; wrote derivatives explicitly throughout the manual.
- Added named keyword term declarations while retaining positional constructors.
- Added the Inguva–Braatz constant-growth Gaussian case, analytical bin-average comparisons, and two-dimensional solution/error plots.
- Audited periodic compact/ADI amplification, long-time norm stability, and agreement with independent dense matrix updates; retained measured negative lobes.
- Added a 1000-second Gunawan-law coupling/refinement comparison with independent moment equations and explicitly labelled demonstration initialization.
- Added a Shin-model result export interface requiring input provenance; kept synthetic LNP figures distinct from the pending paper reproduction.
- Fixed the LNP recipe's moving narrow nucleation source by integrating bin fractions instead of sampling only bin centres; preserved the explicit unnormalized-amplitude convention.
- Made growth and boundary nucleation separate named terms in the crystallization recipe and included both in discrete material exchange.

- Renamed the package and optional extensions to PopulationBalanceModels; preserved the UUID and version.
- Reorganized the manual around model definition, numerical treatment, coupling, inference, and case studies.
- Consolidated standalone development reports into the authoritative documentation under `docs/src/`.
- Reworked examples into case studies tied to descriptive runnable scripts.
- Documented supported combinations, numerical assumptions, conservation, and extension interfaces.
- Updated manual navigation, links, assets, and versioned documentation deployment configuration.
- Simplified the README and repository root.
- Preserved scientific limitations and distinguished demonstration recipes from validated paper reproductions.

The inherited development work includes sectional and moment models, shared bulk
balances, population interactions, inference adapters, and restricted two-coordinate
transport. Earlier internal milestones are not treated as published releases.
