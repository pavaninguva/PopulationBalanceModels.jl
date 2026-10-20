# PopulationBalanceModels.jl

A population balance describes how a distribution of particles changes through
transport in particle properties and through discrete creation, removal, and
collision events. PopulationBalanceModels turns those mechanisms into an
executable model while keeping physical laws separate from numerical choices.

Begin with [Getting started](getting_started.md), then define coordinates,
units, boundaries, and kinetic laws in [Defining models](defining_models.md).
[Coupled systems](coupled_systems.md) connect distributions to shared process
states. [Numerical methods](numerical_methods.md) explains how the equations
become residuals and time updates; [Moment methods](moment_methods.md) covers
reduced representations. [Observations and estimation](inference.md) maps the
same forward model to data.

The [case studies](case_studies/index.md) focus on batch cooling crystallization,
lipid nanoparticle formation, and a two-coordinate balance with size-dependent
growth and an analytical solution. Read the
[API reference](api.md) alongside these explanations. The [developer guide](development.md)
covers internals and extensions, and [publishing](publishing.md) describes
versioned documentation hosting.

The active scope is nonspatial population balances. A two-coordinate distribution
uses two internal particle properties, not two physical-space coordinates.
CFD transport is deferred. Numerical benchmarks test specific equations and
configurations; they do not establish general positivity, closure accuracy, or
reproduction of the paper examples.
