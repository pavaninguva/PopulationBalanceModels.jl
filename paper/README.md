# Paper figures

Each case is one directory. `run.jl` calculates results or explicitly stages
retained production data; `plot.py` draws from saved results. Settings are in `config.toml`, numerical exports in
`data/`, and finished figures in `figures/`.

All figure exporters save **300-dpi PNG only**, including the architecture
schematic. No PDF/SVG derivatives are retained. Numerical tables and provenance
receipts remain alongside the figures and data.

| Directory | Case | Saved results |
|:--|:--|:--|
| [growth/](growth/) | Growth and time schemes | Available |
| [milling/](milling/) | Cone-mill forward comparison and MC/PCE uncertainty | Retained production curves and original producers |
| [precipitation/](precipitation/) | Tubular precipitation | Logarithmic WENO5 / axial MUSCL refinement study |
| [protein/](protein/) | Fixed-γ CNT versus CNT + constant D0 | MAPs, MCMC draws and latent predictions available |

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

Use `precipitation` in the same commands. Plotting validates the
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

The milling case also has exactly two outputs: `milling_forward.png` and
`milling_uq.png`. Stage the retained data once, then redraw independently:

```sh
julia --project=paper paper/milling/run.jl --retained
python paper/milling/plot.py
python test/paper_data.py --case 2 --retained
```

The forward figure shows outlet d50 and hold-up for 3350/4923 RPM using one
100-pivot fixed-pivot/CVODE-BDF calculation and the UQ example's pivot-CDF
median convention. Reference points are published simulation readings, not
experiments. Its power grid, initial feed inventory and point-normalized feed
remain distinct from the geometric-grid, empty-start UQ calculation. The UQ
figure reuses the manual's three-panel MC/PCE comparison, including the full
12,000-draw MC ensemble. It does not rerun that ensemble or reinterpret the
stored output bands. Compact curve tables and unchanged producer manifests
live in `milling/data/`; full samples, coefficients and verification exports
remain in the original documented reproduction dataset.

A deliberate recalculation reuses the shared example builders/drivers and
requires a fresh directory:

```sh
julia --project=paper paper/milling/run.jl --full --output=/tmp/milling-paper
python paper/milling/plot.py --input=/tmp/milling-paper
```

This executes the original forward verification (including its supporting grid
checks) and full MC/PCE study; it can be expensive. Only the selected 100-pivot
forward curves and the three UQ summaries enter the two paper figures.
Staging and plotting validate retained numerical hashes without asserting
that historical producers equal today's checkout. Figure-only commands never
rewrite numerical tables or producer records.

The retired moments paper case and obsolete growth plots are preserved outside
`paper/` in the historical archive under `test/fixtures/archives/paper_history/`.
The documentation's aggregation/moment case and its verification remain active.

For protein, redraw the retained results without fitting or sampling:

```sh
python paper/protein/plot.py
```

A deliberate full recalculation uses a fresh output directory:

```sh
julia --project=paper paper/protein/run.jl --full --output=/tmp/protein-controlled
python paper/protein/plot.py --input=/tmp/protein-controlled --tables
```

This fits both MAPs, continues four chains per model until joint diagnostics pass
or a finite budget is reached, then propagates verified draws. The controller
keeps one warmup per chain; `--min-samples=500 --batch-samples=250
--max-samples=10000 --consecutive-passes=2` are the defaults. Its chains currently
advance serially. `--stopping=fixed` instead uses the configured fixed draw
budgets and isolated workers; an explicit `--samples=N` also selects fixed mode.
`--map` fits only the MAPs. Existing posterior outputs cannot be replaced
implicitly: use `--output` for a new run or deliberately select `--force`.
The current three-/four-parameter models fix γ = 0.64 mJ/m², use g ~ Uniform(1,5),
and use constant D0 in the GRD model. Rates are expressed in L, μm and min units.
The former four-/five-parameter posterior results and uncertainty receipts are
preserved in `test/fixtures/archives/paper_history/protein_variable_gamma_theta_20261028.zip`;
they are not samples from the revised target. The same plot
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

The growth case has exactly two figure outputs: `growth_profiles.png` and
`growth_work_precision.png`. The profiles translate the Gaussian to 4 s on
200 bins, with separate CFL=0.5 and CFL=1 panels. The work–precision figure
uses the smooth periodic problem on 40–640 bins. The seven pairings include
both WENO5 + SSPRK3 and WENO5 + RK4. Error/order tables remain as supporting
numerical verification; their extra paper plots have been removed.
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
