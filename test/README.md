# Verification

Run from the repository root. The core suite includes Aqua and independent
analytical, conservation, refinement, admissibility and failure checks:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Files are grouped by behaviour, rather than by development milestone:

| Area | Main files |
|:--|:--|
| Physical terms and coupling | `sources_moments.jl`, `systems.jl`, `papers.jl`, `size_diffusion.jl` |
| Transport and time methods | `transport_integrators.jl`, `nonuniform_weno.jl`, `compact_finite_volume.jl`, `tensor.jl` |
| Moment representations and accounting | `extended_moments.jl`, `measure_evolution.jl`, `qamar_moments.jl` |
| Axial and stationary balances | `axial.jl`, `axial_system.jl`, `stationary.jl` |
| Calibration, Bayesian targets and derivatives | `uncertainty.jl`, `bayesian.jl`, `differentiation.jl`, `isothermal_protein.jl` |
| Scientific regression cases | `audit_regressions.jl`, `numerical_contracts.jl`, `stem_cells.jl`, `li_supplement.jl` |

`fixtures/` holds independent references and reusable setup.
`fixtures/example_modules.jl` loads each example module once per enclosing test
module, so the same builders can be used in several files without replacing
modules and repeating compilation. It also supports standalone test-file runs.
The cleanup retains the numerical assertions and their independent references.

## Optional integrations

Use an isolated working environment containing this checkout and the dependencies
listed in [CI](../.github/workflows/CI.yml). Then run the corresponding entry points:

```sh
julia --project=/path/to/optional-env test/integration.jl
julia --project=/path/to/optional-env test/bayesian_integration.jl
julia --project=/path/to/optional-env test/protein_parallel.jl
julia --project=/path/to/optional-env test/stationary_integration.jl
julia --project=/path/to/sensitivity-env test/sensitivity_integration.jl
```

The inference tests use small analytical targets; they do not regenerate the
long protein posterior. Minimum/current Julia jobs, normal compiled-module
sensitivity checks, and selective JET analysis remain in CI.

## Retained results and utilities

```sh
python paper/check_protein.py --retained
python paper/check_data.py --case 1 --retained
python paper/check_data.py --case 2 --retained
python test/plotting_summaries.py
python test/stem_cell_results.py
python test/docs_links.py docs/build
julia --project=. test/tools/run_examples.jl
julia --project=. test/tools/environment.jl
julia --project=. test/tools/profile_paths.jl working_tree
```

Use the plotting environment in `paper/requirements.txt` for the Python checks.
`--retained` checks stored data, normalizations, balances and posterior diagnostics;
it explicitly does **not** assert that today's checkout is their producer.
Omitting it retains the strict source-match check. A source mismatch is not a
reason to overwrite posterior draws or their provenance. The longer standalone
`audit_protein.jl` and `audit_qamar_diagnosis.jl` retain their historical numerical
questions and are not run as part of a directory-cleanup smoke check.

`tools/` contains executable reporting utilities, not extra core test suites.
The tubular paper dataset's missing time-refinement run is an existing incomplete
study; its separate `--case 3 --retained` check must continue to fail until that
run is completed. No documentation figure is substituted for it.
Performance probes report warmed allocations/work separately from the paper's
single-solve timing protocol. Build documentation before checking local links.
