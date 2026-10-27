# PopulationBalanceModels.jl

PopulationBalanceModels solves population balances for particle and cell distributions.
A model can describe growth in size, creation by nucleation, removal, aggregation,
and breakage. Distributions can share concentrations or other process states,
which are integrated with the population equations.

The package separates the physical model from its numerical treatment. For
example, a growth term specifies velocity in an internal coordinate, such as
crystal length or cell-marker content; a reconstruction determines
its numerical face flux; a time integrator advances the resulting equations.
This lets the same kinetic law be studied with different numerical methods.

Start with [a complete first model](getting_started.md). Then read
[Defining models](defining_models.md) for coordinates, units, rates, and
boundaries, or [Coupled systems](coupled_systems.md) to include bulk balances.
The [numerical guide](numerical_methods.md) explains the treatment of each term,
including restrictions on grids, boundaries, and positivity.
For a direct solution of time-independent balances, see
[Stationary population balances](steady_state.md), including a well-mixed
continuous crystallizer and parameter continuation.

The [case studies](case_studies/index.md) develop cooling crystallization,
lipid nanoparticle formation, growth in two particle dimensions, and
[stem-cell marker dynamics](case_studies/stem_cells.md). For fitting
models to data, see [Parameter estimation](inference.md). Signatures and
extension hooks are collected in the [API reference](api.md) and
[developer guide](development.md).

A two-dimensional population resolves two internal coordinates,
such as length and width. On a grid, the package discretizes the population PDE
as integrated numbers in coordinate cells; the manual calls this a *sectional*
representation. Moment methods use a finite moment basis instead.
The separate [axial reactor domain](axial_reactors.md)
adds physical position with prescribed constant advection and dispersion to
one- or two-coordinate sectional populations. General CFD coupling remains
outside the current scope.
