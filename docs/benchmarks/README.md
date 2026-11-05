# Documentation figures

## Milling uncertainty

Numerical production and plotting are separate:

```sh
julia --project=examples/environments/milling -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=examples/environments/milling examples/verification/milling_forward.jl
python test/milling_results.py --forward
python docs/benchmarks/plot_milling_uncertainty.py --forward --usetex
julia --project=examples/environments/milling examples/milling_uncertainty.jl --smoke
python docs/benchmarks/plot_milling_uncertainty.py --smoke
julia --project=examples/environments/milling examples/milling_uncertainty.jl --full
python test/milling_results.py
python docs/benchmarks/plot_milling_uncertainty.py --usetex
```

Final PNG/PDF/SVG figures read only completed production exports under
`docs/src/assets/milling_uncertainty/data/`. Smoke data and previews stay under
`.cache/milling_uncertainty/`. The plotter verifies producer-file hashes and
requires real LaTeX for final output; it never runs a forward solve. See the
[case study](../src/case_studies/milling_uncertainty.md) for source assumptions,
interval definitions and numerical/validation checks.

`--forward` selects a separate constant-speed comparison with Barrasso (2013),
Figure 7. Its graph readings and extraction record are under
`examples/data/milling_uncertainty/barrasso_figure7.*`; its newly calculated
trajectories, errors and producer record are under the assets' `forward/`
directory. This is a partial reproduction, not a validated replacement for the
retained geometric-grid, empty-start UQ ensemble. Plotting it neither solves nor
changes either numerical dataset.

## Existing figure workflows

These plotters consume the documentation examples' exported data. They are
separate from the [manuscript workflow](../../paper/README.md). The active
protein and logarithmic tubular pages share their paper figure producers;
the commands below otherwise use the documentation exports. Documenter copies
current downloadable sources and builds pages; it does not rerun expensive fits
or numerical studies.

Run commands from the repository root. Plotting requires NumPy and Matplotlib;
`--usetex` additionally requires LaTeX, dvipng, amsmath and Computer Modern fonts.
Use the [paper Python requirements](../../paper/requirements.txt) for a pinned
environment. Plotters retain actual bin edges, signed numerical values and the
case's stated density normalization; they do not create missing numerical data.
The [case pages](../src/case_studies/index.md) supply model assumptions,
interpretation, parameters and numerical restrictions.

Before redrawing retained manual exports, run
`python tools/retained_data.py fetch --profile docs`. The explicit fetch verifies
the original bytes without running Julia. Use `--profile paper` when also
redrawing publication figures, and see the [data instructions](../../tools/README.md)
for individual groups and offline snapshots. Missing exports are not a request
to launch an expensive production calculation.

The moment, cooling and historical compact-comparison plotters export 300-dpi
PNG plus vector PDF/SVG using the shared style. Their presentation updates
leave the original CSVs, timings and producer records unchanged. Moment
relative-error panels use percent; their ``1e-16`` fractional display floor is
``1e-14`` percent, and signed differences remain in the numerical exports.
An absolute-error minimum can mark a signed zero crossing, not a more accurate
closure. The historical compact plot retains its recorded minimum-of-three
warmed timing protocol; other cases keep their own stated timing protocols.

## Calculation and plotting entry points

| Calculation | Numerical driver under `examples/` | Plot command under `docs/benchmarks/` |
|:--|:--|:--|
| Gaussian growth/time comparison | `constant_growth_methods.jl` | `plot_transport_comparisons.py constant --usetex` |
| Batch cooling | `cooling_crystallization_benchmark.jl` | `plot_transport_comparisons.py cooling --usetex` |
| Qamar moment comparisons | `qamar_moment_benchmarks.jl` | `plot_case_studies.py moments --usetex`, then `plot_moment_comparison.py --usetex` |
| Historical uniform-size tubular precipitation | `continuous_precipitation.jl` | `plot_population_studies.py precipitation --usetex` |
| Two-coordinate variable growth | `inguva_braatz_case2.jl` | `plot_case_studies.py transport --usetex` |
| Stem-cell marker dynamics | `stem_cell_population.jl` | `plot_stem_cells.py --usetex` |
| Synthetic Bayesian growth fit | `estimate_size_dependent_growth.jl` | `plot_bayesian_growth.py --usetex` |
| Axial advection/dispersion | `axially_dispersed_growth.jl` | `plot_transport_comparisons.py axial --usetex` |
| Gamma density reconstruction | `quadrature_moment_methods.jl` | `plot_population_studies.py moments --usetex` |

For example, redraw the bundled Gaussian comparison without a numerical run:

```sh
python docs/benchmarks/plot_transport_comparisons.py constant --usetex
julia --project=docs docs/make.jl
python test/docs_links.py docs/build
```

Optional fitting/sampling/stiff-solver integrations need the working environment
specified on the corresponding case page. The LNP reconstruction takes an
explicit input file; use the [nanoparticle page](../src/case_studies/nanoparticles.md)
for its driver, unresolved closures and retained long-time limitations.

The cooling calculation uses a declared nonnegative initial distribution and a
refined numerical reference, not a repaired version of the printed polynomial.
The manual and paper growth figures read the same compact Gaussian selection
under `docs/src/assets/constant_growth/`: 40--640-bin refinements, 200-bin
profiles, zero inflow/open outflow and final time 4. Its checksums and original
Julia 1.10.12 timing environment are retained in `gaussian_selection.json`.
No current figure or quoted order uses the older periodic experiment.
For a deliberate new calculation, run
`ConstantGrowthMethods.run(output="/tmp/gaussian-growth")` after including
`examples/constant_growth_methods.jl`, and pass `--input /tmp/gaussian-growth`
to either plotter. Both accept `--output` for separate figures. The numerical
driver otherwise writes to `.cache/benchmarks/constant_growth`; it does not
overwrite the bundled selection or its producer records.
Qamar `--methods-only` reuses the retained sectional reference; it still records
failed closures and incomplete trajectories. Collision EQMOM remains unresolved.

The active tubular study uses `julia --project=paper paper/precipitation/run.jl`
and `python paper/precipitation/plot.py`. It uses logarithmic-size WENO5, axial
MUSCL, separate 50–400-cell refinement paths and a common 1000 × 500 reference
with a same-mesh temporal check. The plotter publishes the same PNGs to paper
and manual assets. `examples/continuous_precipitation.jl` supplies the common
physical builder; its older `--size-resolution` and `--full` modes retain their
historical uniform-size studies. They do not regenerate the active figures.
Neither calculation is a CADET backend or an analytical coupled PSD solution.

The stem-cell `--smoke` calculation writes to the system temporary directory;
the full synthetic study exports the separate grid, time, daughter-quadrature
and domain checks. It does not fit experimental measurements.

## Active protein comparison

The isothermal comparison has three scenarios: CNT, constant-D0 GRD and
growth-dependent GRD (`D_G = k_D G`). They share the model/inference implementation
and paper/documentation figure producer:

```sh
julia --project=paper -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
python tools/retained_data.py fetch --profile paper
python paper/protein/plot.py --data-only
python paper/protein/plot.py
# Expensive: calculate new results only when intentionally reproducing them.
julia --project=paper paper/protein/run.jl --full --model=all --output=/tmp/protein-reproduction
```

The [case driver guide](../../paper/README.md)
records its settings. The sixteen selected paper PNGs are copied to
`docs/src/assets/protein_isothermal/` by that producer. Preserve the retained
posterior draws, priors, chain diagnostics, coordinate settings, predictions and
provenance under `paper/protein/data/`.
`python test/protein_results.py --retained` checks those stored artifacts without
claiming equivalence to a later checkout. Default production checks remain strict.
Plot-only work does not require new sampling, but its source/provenance checks
must still be respected; do not relabel historical draws as a revised target.

## Verification and historical calculations

Numerical audit/benchmark drivers now live under `examples/verification/`:

```sh
julia --project=. examples/verification/compact_growth.jl
python docs/benchmarks/plot_compact_growth.py --usetex
julia --project=. examples/verification/li2026_supplement.jl
julia --project=. examples/verification/compact_adi.jl
julia --project=. examples/verification/lax_wendroff.jl
```

The supplement's literal coefficient transcription remains separate from its
independently corrected interpretation. Moving the files does not strengthen
any order, stability, positivity or paper-reproduction claim.

`diagnostics/` retains earlier cooling/LNP evidence used by historical plotters.
Unused constant-growth tensor exports from `diagnostics/inguva_case1/`, the
superseded uniform-size precipitation PNG/SVG figures, and dated development
audit tables were moved to the external recovery archive; see the
[archive catalogue](../../test/fixtures/archives/README.md#relocated-legacy-outputs).
The tensor benchmark's producing fixture remains in
`test/fixtures/constant_growth_gaussian.jl`. All uniform-size precipitation CSVs
remain for the historical plot command above and retained-data tests; that
command can recreate its old images without affecting the current log-grid figures.

The earlier protein calculation
is `examples/protein_crystallization.jl`; its samples and figures are preserved in
`test/fixtures/archives/legacy_docs_assets.zip`, with different observations and
priors from the active isothermal comparison. Its plotters remain available as
`plot_population_studies.py protein --usetex` and
`plot_protein_posterior.py --usetex`; use the
[historical-fixture extraction recipe](../../test/fixtures/archives/README.md)
first. Earlier teaching calculations are listed
in the [example catalogue](../../examples/README.md).

`plot_population_studies.py` replaces the vague `plot_new_case_studies.py` name;
its numerical inputs and plotting routines are unchanged.
`python test/plotting_summaries.py` checks probability contours, bounded densities,
Pareto selection and numerical export conventions. The case-study pages describe
numerical references and the limits of each comparison.
