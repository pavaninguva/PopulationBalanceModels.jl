# Case studies

Run each script from the package root with `julia --project=. examples/SCRIPT.jl`.
The estimation script needs Optim in a separate working environment. The sequence
builds from local kinetics to process coupling and then to observation models.

1. [Batch population balance](batch.md): growth, creation, removal, and measured moments.
2. [Interacting populations](interacting.md): independent grids and a common solute balance.
3. [Aggregation and breakage](collisions.md): event counting and truncation.
4. [Cooling crystallization](crystallization.md): Gunawan case 2 with explicit demonstration initialization.
5. [Two-coordinate transport](transport.md): WENO, compact/ADI, and eligible exact shifts.
6. [Estimating kinetic parameters](estimation.md): analytical synthetic observations and local rank.
7. [Lipid nanoparticle formation](nanoparticles.md): CEJ model recipe with explicit demonstration closures.

Each case links its executable implementation. Numerical consistency checks and
synthetic fits are distinct from experimental validation. Supporting scripts
`nucleation_sources.jl`, `named_term_methods.jl`, `coupled_moment_balance.jl`,
`moment_closures.jl`, and `periodic_weno_transport.jl` isolate numerical choices
introduced in the manual.
