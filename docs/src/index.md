# PopulationBalanceModels.jl

PopulationBalanceModels solves population balances for particle distributions.
A model can describe growth in size, creation by nucleation, removal, aggregation,
and breakage. Distributions can share concentrations or other process states,
which are integrated with the population equations.

The package separates the physical model from its numerical treatment. For
example, a growth term specifies a particle velocity; a reconstruction determines
its numerical face flux; a time integrator advances the resulting equations.
This lets the same kinetic law be studied with different numerical methods.

Start with [a complete first model](getting_started.md). Then read
[Defining models](defining_models.md) for coordinates, units, rates, and
boundaries, or [Coupled systems](coupled_systems.md) to include bulk balances.
The [numerical guide](numerical_methods.md) explains the treatment of each term,
including restrictions on grids, boundaries, and positivity.

The [case studies](case_studies/index.md) develop cooling crystallization,
lipid nanoparticle formation, and growth in two particle dimensions. For fitting
models to data, see [Parameter estimation](inference.md). Signatures and
extension hooks are collected in the [API reference](api.md) and
[developer guide](development.md).

A two-dimensional particle distribution resolves two intrinsic coordinates,
such as length and width. The separate [axial reactor domain](axial_reactors.md)
adds physical position with prescribed constant advection and dispersion to
one- or two-coordinate sectional populations. General CFD coupling remains
outside the current scope.
