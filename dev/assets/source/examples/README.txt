# Examples and reproduction drivers

Run commands from the repository root with this checkout active. Include a
case-study module to reuse its builders; execute its file to run its driver.
The manual links short modelling walkthroughs and complete downloadable sources.

## Start here

```sh
julia --project=. examples/batch_population_balance.jl
julia --project=. examples/named_term_methods.jl
julia --project=. examples/interacting_populations.jl
```

These small examples introduce physical terms, per-term methods and shared bulk
balances. Other capability examples are `nucleation_sources.jl`,
`aggregation_and_breakage.jl`, `moment_closures.jl`, `coupled_moment_balance.jl`,
`nonuniform_weno.jl`, `random_fluctuation_grd.jl` and `axially_dispersed_growth.jl`.
`random_fluctuation_grd.jl --two-coordinate` demonstrates a coupled tensor
population with bulk-dependent growth/dispersion and a closed material balance.
`pce_refinement.jl` compares regression and Gaussian-quadrature PCE for a beta-
distributed removal rate, with validation, refinement and an independent final
test. It needs only the ordinary package dependencies.
`steady_state_crystallizer.jl` needs the optional NonlinearSolve integration.
`estimate_stationary_crystallizer.jl` adds synthetic stationary Bayesian
calibration with Optim, implicit sensitivities and local uncertainty; optional
sampling uses AdvancedHMC and MCMCDiagnosticTools.

## Manual case studies

| Calculation | Driver | Walkthrough |
|:--|:--|:--|
| MC and PCE for continuous milling | [milling_uncertainty.jl](milling_uncertainty.jl) | [Milling uncertainty](../docs/src/case_studies/milling_uncertainty.md) |
| Gaussian growth and time methods | [constant_growth_methods.jl](constant_growth_methods.jl) | [Growth comparison](../docs/src/case_studies/constant_growth.md) |
| Batch cooling | [cooling_crystallization_benchmark.jl](cooling_crystallization_benchmark.jl) | [Cooling crystallization](../docs/src/case_studies/crystallization.md) |
| Extinction and solute release | [realizable_dissolution.jl](realizable_dissolution.jl) | [Boundary and material accounting](../docs/src/methods/realizable_evolution.md) |
| Aggregation/fragmentation and source pulse | [qamar_moment_benchmarks.jl](qamar_moment_benchmarks.jl) | [Moment methods](../docs/src/case_studies/moment_crystallization.md) |
| Tubular precipitation | [continuous_precipitation.jl](continuous_precipitation.jl) | [Continuous precipitation](../docs/src/case_studies/continuous_precipitation.md) |
| Isothermal CNT versus CNT + GRD | [isothermal_protein_crystallization.jl](isothermal_protein_crystallization.jl) | [Protein crystallization](../docs/src/case_studies/protein_crystallization.md) |
| Stem-cell marker dynamics | [stem_cell_population.jl](stem_cell_population.jl) | [Stem cells](../docs/src/case_studies/stem_cells.md) |
| Lipid nanoparticles | [shin_lnp_reproduction.jl](shin_lnp_reproduction.jl), [shin_lnp_long_time.jl](shin_lnp_long_time.jl) | [Nanoparticles](../docs/src/case_studies/nanoparticles.md) |
| Two-coordinate variable growth | [inguva_braatz_case2.jl](inguva_braatz_case2.jl) | [Two-dimensional growth](../docs/src/case_studies/transport.md) |
| Synthetic Bayesian growth fit | [estimate_size_dependent_growth.jl](estimate_size_dependent_growth.jl) | [Growth estimation](../docs/src/case_studies/estimating_growth.md) |

[Documentation figure commands](../docs/benchmarks/README.md) identify the data
and plotter for each case. The independent [paper workflow](../paper/README.md)
reuses these builders, with its own configurations, results and figures.
`support/` contains reporting, sampling and benchmark helpers, not alternative
physical solvers. `papers/` contains the shared cooling and nanoparticle recipes.

The active protein entry point is
`paper/protein/run.jl`. Its retained CNT/GRD draws
are under `paper/protein/data/`; the earlier
`protein_crystallization.jl` and the historical fixture archive belong
to a different, historical inference problem. Keep their observations, priors,
draws and metadata together; do not mix them with the active comparison.

## Numerical verification

These are numerical investigations, separate from the physical walkthroughs:

| Driver | Purpose |
|:--|:--|
| `verification/milling_forward.jl` | Barrasso Figure 7 startup comparison, fixed-endpoint refinement and independent time/quadrature checks; uses the milling environment |
| `verification/compact_growth.jl` | Conservative HOC spatial/time refinement and work comparison |
| `verification/compact_adi.jl` | Legacy compact/ADI Fourier and long-period diagnostics |
| `verification/li2026_supplement.jl` | Literal supplement coefficients and independently corrected variant |
| `verification/lax_wendroff.jl` | Independent nodal stencil and conservative transport comparison |
| `realizable_startup.jl`, `realizable_axial_transport.jl` | Positive-measure startup and axial balance |
| `quadrature_moment_methods.jl` | Reconstruction and tracked/reconstructed moment checks |

The compact/stencil files under `verification/` were previously named `audit_*` or
`benchmark_compact_growth.jl` at this directory's top level. Their numerical
equations and default output locations are unchanged. The separate
collision-EQMOM investigation remains parked; this organization does not
resolve or remove any recorded failure.

Smaller historical teaching calculations remain runnable: the original
`estimate_kinetic_parameters.jl`, `kinetic_parameter_uncertainty.jl`,
`temperature_programmed_crystallization.jl`, `cooling_crystallization_demo.jl`
and `lipid_nanoparticle_demo.jl`. Tests still exercise their distinct contracts.
`two_coordinate_transport.jl` and `periodic_weno_transport.jl` provide analytical
transport checks. They are not additional active case-study pages.
