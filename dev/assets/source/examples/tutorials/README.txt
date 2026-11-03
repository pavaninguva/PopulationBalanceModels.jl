# Case-study walkthroughs

These authored files are the source of the manual's complete scripts and their
downloads. `docs/case_study_scripts.jl` reads them verbatim; it does not assemble
programs from benchmark modules. Run from a checkout with its environment active:

```sh
julia --project=. examples/tutorials/constant_growth.jl
julia --project=. examples/tutorials/crystallization.jl
julia --project=. examples/tutorials/moment_crystallization.jl
julia --project=. examples/tutorials/continuous_precipitation.jl
julia --project=. examples/tutorials/stem_cells.jl
julia --project=. examples/tutorials/transport.jl
julia --project=. examples/tutorials/nanoparticles.jl
julia --project=docs examples/tutorials/estimating_growth.jl
julia --project=docs examples/tutorials/protein_crystallization.jl
```

The last two commands fit parameters. Merely including those files constructs
their estimation problems without optimization, sampling or local uncertainty.
The constant-growth file is an inexpensive ordinary script and also solves on
include. The other population scripts prepare the model on include and execute
their solve when run as a program. There is no required `main`/builder/plotting
hierarchy. No tutorial writes numerical exports or modifies retained assets.

Milling has two explicit stages in its optional environment:

```julia
include("examples/tutorials/milling_uncertainty.jl") # Prepare, no ensemble.
nominal = solver(problem,times)
include("examples/tutorials/milling_propagation.jl") # 16 MC + 12 PCE solves.
```

The second stage uses the existing MC realizations as independent PCE validation
inputs; training points are separate. Polynomial sampling is an optional further
step after validation. These small demonstrations are not production campaigns.

The stem-cell accounting extension is also explicit:

```julia
include("examples/tutorials/stem_cells.jl")
include("examples/tutorials/stem_cell_accounting.jl")
accounted = simulate(accounted_problem;method=SSPRK33(),dt,
    negative_tolerance=0.0,saveat=[0.0,10.0,20.0])
```

It retains counters and cumulative daughter losses without feeding back into the
population. Counts/cargo at differentiation exit are not another population.

## Companions and downloads

| Walkthrough | Companion and purpose |
|:--|:--|
| Growth, cooling, moments, two-coordinate growth | None for the first calculation; analytical comparisons stay in the maintained verification drivers |
| Tubular precipitation | `../support/precipitation_numerics.jl`: the unchanged transport/depletion step bound |
| Protein | `../support/protein_data.jl`: original CSV reading and fixed split; `../support/protein_execution.jl`: checked interval solver and prescribed calibration starts |
| Stem cells | `stem_cell_accounting.jl`: optional complete diagnostic balances |
| Growth estimation | `../support/affine_growth_reference.jl`: exact densities/bin integrals and seeded synthetic point data |
| Nanoparticles | `../support/lnp_material.jl`, `lnp_attachment.jl`: material formulas and parameter-specific DLVO preparation; optional `lnp_stiff_execution.jl` for the existing specialized long-time path |
| Milling | `../support/milling_model.jl`, `milling_solver.jl`: daughter projection, classifier, median and constrained/material-coordinate BDF; `milling_propagation.jl` for MC/PCE stages |

When downloading individual files, preserve this layout (or use a checkout):

```text
examples/
  tutorials/                 # Authored walkthrough and any next-stage script
  support/                   # Its linked companion(s)
  data/protein_crystallization/tang2018_isothermal/
    concentration.csv        # Protein only
    psd_cdf.csv
    psd_summary.csv
```

Only the installed package, declared optional solver/inference dependencies and
these linked files are needed; no whole reproduction driver is imported.
The old `assets/source/tutorials/` download paths remain available after a docs
build, alongside the canonical `assets/source/examples/tutorials/` copies.

## Plot existing results

Numerical calculations do not load Python. Each file ends with short commented
recipes using `plot_distribution`, `plot_observable`, `plot_fit`, `plot_field` or
`plot_uq`. Enable the optional PythonPlot extension, then execute the applicable
calls on the prepared solution/predictions. Save explicitly with `save_figure`.
Plotting never performs a new solve, fit, propagation or polynomial training.
See the manual's plotting guide for environment setup and coordinate semantics.

## Verification

```sh
julia --project=docs test/case_study_scripts.jl
julia --project=docs test/case_study_migration.jl
```

Run the same tests in an environment also containing Sundials and SciMLBase for
the small milling stages. PythonPlot rendering is tested separately by
`test/plotting_integration.jl`. Short trajectories and fixed-parameter posterior
checks do not certify historical production results under a changed checkout.
