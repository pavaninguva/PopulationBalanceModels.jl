# Paper figures

Each case is one directory. `run.jl` calculates the results; `plot.py` draws the
figures from saved results. Settings are in `config.toml`, numerical exports in
`data/`, and finished figures in `figures/`.

All figure exporters save **300-dpi PNG only**, including the architecture
schematic. No PDF/SVG derivatives are retained. Numerical tables and provenance
receipts remain alongside the figures and data.

| Directory | Case | Saved results |
|:--|:--|:--|
| [growth/](growth/) | Growth and time schemes | Available |
| [moments/](moments/) | Aggregation–fragmentation | Available |
| [precipitation/](precipitation/) | Tubular precipitation | Logarithmic WENO5 / axial MUSCL refinement study |
| [protein/](protein/) | CNT versus CNT + GRD | MAPs, MCMC draws and latent predictions available |

Run from the repository root. Set up the Julia checkout and Python 3.12 plotting
environment once; final figures require LaTeX:

```sh
julia --project=paper -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
python -m pip install -r paper/requirements.txt
```

For example:

```sh
julia --project=paper paper/growth/run.jl
python paper/growth/plot.py
```

Use `moments` or `precipitation` in the same commands. Plotting validates the
saved data and preserves its original producer identity; it never reruns a
simulation or treats old timings as newly measured results. Incomplete
precipitation studies are rejected unless their reference, independent
refinements and same-mesh time check are complete; documentation data are not
substituted.

Tubular precipitation varies 50, 100, 200 and 400 logarithmic size bins at 500
fixed axial cells, then 50, 100, 200 and 400 axial cells at 1000 fixed size bins.
Both paths use one 1000 × 500 reference. The checked BDF adapter is used in time;
the spatial residual remains geometry-aware WENO5 / axial MUSCL. Its outlet
figure includes the retained digitized Zhang Figure 9(c) points with extraction
uncertainty. Full runs are expensive and resume matching completed configurations.
`--only=size_Nx50` (or another recorded configuration name) executes one job;
the no-argument command collects the complete study and calculates its errors.
The closed growth/nucleation model uses the normal `PBMSystem.bulk_rhs` callback
and its current population derivative for material exchange. Tests compare this
with `ConservativeExchange`; it avoids repeated transport evaluation and does
not add a separate analytical uptake sink.

For protein, redraw the retained results without fitting or sampling:

```sh
python paper/protein/plot.py
```

A deliberate full recalculation uses `julia --project=paper paper/protein/run.jl --full`;
it includes both expensive MCMC runs. `--map` fits only the MAPs. The same plot
script accepts `--data-only`, `--gauss-newton` or `--laplace` for those figures;
the corresponding Julia `--gauss-newton`/`--laplace` modes calculate uncertainty
separately. The [manual](../docs/src/case_studies/protein_crystallization.md)
explains these analyses and their limits.

After a reviewed package change, `julia --project=paper paper/protein/run.jl
--compatibility-only` checks the retained posterior values and gradients without
refitting or sampling. Original manifests and analysis receipts are preserved;
reuse requires matching observations, settings, priors and resolved dependencies.

[architecture/](architecture/README.md) is the independent schematic generator.
`_support/` contains implementation helpers, not additional entry points.
Tests and historical runs live under `test/`, not here. Current protein samples
are in `protein/data/{cnt,cnt_grd}/`; original manifests remain alongside them.

The growth figures follow the [manual case study](../docs/src/case_studies/constant_growth.md):
a Gaussian translated to 4 s, two Courant-number panels on 200 bins, separate
Gaussian and periodic error plots, observed periodic orders, and runtime/work–precision.
The seven pairings include both WENO5 + SSPRK3 and WENO5 + RK4. The smooth
refinement uses 40–640 bins. The five outputs are `growth_profiles.png`,
`growth_gaussian_errors.png`, `growth_smooth_refinement.png`,
`growth_observed_orders.png` and `growth_work_precision.png`.
All figure data are calculated by this paper driver,
not copied from documentation assets. A smoke check writes only ignored cache outputs:

```sh
julia --project=paper paper/growth/run.jl --smoke
python paper/growth/plot.py --smoke --preview
python test/paper_data.py --case 1
```

The protein plotting commands default to figures only, preserving stored numerical
tables, posterior draws and uncertainty-analysis receipts. Add `--tables` explicitly
when regenerating derived tables, including with `--gauss-newton` or `--laplace`.
