# Paper figures

Each case is one directory. `run.jl` calculates results or explicitly stages
retained production data; `plot.py` draws from saved results. Settings are in `config.toml`, numerical exports in
`data/`, and finished figures in `figures/`.

All figure exporters save **300-dpi PNG only**, including the architecture
schematic. No PDF/SVG derivatives are retained. Numerical tables and provenance
receipts remain alongside the figures and data.

Paper panels use lowercase Roman labels `(i)`, `(ii)`, `(iii)`, `(iv)`.
Documentation copies keep alphabetic labels; the package's shared plotting
defaults are unchanged. Caption references follow the destination's convention.
The current paper PNGs stay with the source; their numerical inputs are optional
downloads.

| Directory | Case | Saved results |
|:--|:--|:--|
| [growth/](growth/) | One translating Gaussian: growth and time schemes | Compact retained figure selection bundled with the source |
| [milling/](milling/) | Cone-mill forward comparison and MC/PCE uncertainty | Bundled production curves from one numerical model |
| [precipitation/](precipitation/) | Tubular precipitation | Logarithmic WENO5 / axial MUSCL refinement study |
| [protein/](protein/) | Fixed-γ CNT, constant D0 and growth-dependent GRD | MAPs, MCMC draws and latent predictions available |

Run from the repository root. Set up the Julia checkout and Python 3.12 plotting
environment once; final figures require LaTeX:

```sh
python tools/retained_data.py fetch --profile paper
julia --project=paper -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
python -m pip install -r paper/requirements.txt
```

For example:

```sh
python paper/growth/plot.py
```

The growth plotter reads the bundled Gaussian metrics/profiles under
`docs/src/assets/constant_growth/`; neither growth figure needs the production
folders restored. The profile, error and timing views use the same Gaussian,
zero inflow/open outflow, $G=0.1$ and final time 4, with common refinements
40, 80, 160, 320 and 640. The old periodic experiment is not selected.
`--input` accepts an explicit fresh Gaussian calculation or another derived
selection; `--output` keeps its figures separate. The growth production driver
also accepts `--output=...` and otherwise writes under `.cache/`.

Other cases redraw retained results after fetching their inputs.
A deliberate new calculation uses `run.jl` with
the documented case settings; fetching never runs a simulation. The payload
inventory and offline snapshot instructions are in [tools/README.md](../tools/README.md).
Original producer metadata stays alongside the optional numerical exports.

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
experiments. Both calculations call one forward builder with 100 geometric volume
pivots from 6.54×10⁻¹⁷ to 10⁻³ m³, integrated lognormal feed, eight-point daughter
quadrature, analytic fines, the same outlet-median rule and checked CVODE BDF
tolerances. The initial feed inventory and constant speeds describe the Barrasso
experiment; the UQ experiment starts empty and changes speed at 30 s. The UQ
figure reuses the manual's three-panel MC/PCE comparison, including the full
12,000-draw MC ensemble. It does not rerun that ensemble or reinterpret the
stored output bands. Compact curve tables and their production receipts
live in `milling/data/`; full samples, coefficients and verification exports
are bundled under `docs/src/assets/milling_uncertainty/data/`. Both datasets were
recalculated with the shared builder. Staging checks their numerical settings
and shared-source hashes before selecting curves.

A deliberate recalculation reuses the shared example builders/drivers and
requires a fresh directory:

```sh
julia --project=paper paper/milling/run.jl --full --output=/tmp/milling-paper
python paper/milling/plot.py --input=/tmp/milling-paper
```

This executes the shared-model forward verification (including its supporting grid
checks) and full MC/PCE study; it can be expensive. Only the selected 100-pivot
forward curves and the three UQ summaries enter the two paper figures.
Staging and plotting validate retained numerical hashes without asserting
that historical producers equal today's checkout. Figure-only commands never
rewrite numerical tables or producer records.

The retired moments paper case and obsolete growth plots are preserved in the
external recovery archive listed in the
[historical-asset catalogue](../test/fixtures/archives/README.md#relocated-legacy-outputs).
The documentation's aggregation/moment case and its verification remain active.

For protein, redraw the retained results without fitting or sampling:

```sh
python paper/protein/plot.py
python paper/protein/plot.py --gauss-newton
```

Recalculate the separate Gauss–Newton analyses only when needed (new forward and
derivative evaluations, and a recorded local MAP refinement for unversioned old
records; no sampling or latent-prediction propagation):

```sh
julia --project=paper paper/protein/run.jl --gauss-newton --model=all
python paper/protein/plot.py --gauss-newton --tables
```

Plot commands default to figures only. `--tables` additionally writes derived
summaries under `data/analysis/`; it does not rewrite producer result tables.

A deliberate full recalculation uses a fresh output directory:

```sh
julia --project=paper paper/protein/run.jl --full --output=/tmp/protein-controlled
python paper/protein/plot.py --input=/tmp/protein-controlled --tables
```

This fits all three MAPs, continues four chains per model until joint diagnostics pass
or a finite budget is reached, then propagates verified draws. The controller
keeps one warmup per chain; `--min-samples=500 --batch-samples=250
--max-samples=10000 --consecutive-passes=2` are the defaults. Its chains currently
advance serially. `--stopping=fixed` instead uses the configured fixed draw
budgets and isolated workers; an explicit `--samples=N` also selects fixed mode.
`--map` fits only the MAPs. Existing posterior outputs cannot be replaced
implicitly: use `--output` for a new run or deliberately select `--force`.
The current three-/four-parameter models fix γ = 0.64 mJ/m², use g ~ Uniform(1,5),
and compare zero dispersion, constant D0 and growth-dependent D_G = k_D G.
Rates are expressed in L, μm and min units; k_D has units of length.
The former four-/five-parameter posterior results and uncertainty receipts are
preserved in `test/fixtures/archives/paper_history/protein_variable_gamma_theta_20261028.zip`;
they are not samples from the revised target. The plot script accepts `--data-only` and `--gauss-newton` for the current
figures; `--model=both` selects the original two scenarios. The Julia
`--gauss-newton` mode calculates local uncertainty separately. An explicitly
requested `--laplace` calculation/plot remains available; archived full-Hessian
rejections are not claims of accepted Laplace results for the new models. The [manual](../docs/src/case_studies/protein_crystallization.md)
explains these analyses and their limits.

After a reviewed package change, `julia --project=paper paper/protein/run.jl
--compatibility-only` checks the retained posterior values and gradients without
refitting or sampling. Original manifests and analysis receipts are preserved;
reuse requires matching observations, settings, priors and resolved dependencies.

[architecture/](architecture/README.md) is the independent schematic generator.
`_support/` contains implementation helpers, not additional entry points.
Tests and historical runs live under `test/`, not here. Current protein samples
are in `protein/data/{cnt,cnt_grd,cnt_growth_grd}/`; original manifests remain alongside them.

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

Protein uses the Pusey solubility, 2.16 g/L, for all three scenarios. The third
scenario has size-independent mean growth and `D_G = k_D * G`, with `k_D` in
micrometres and an Exponential prior of mean 0.1 micrometre. This is a distinct
coefficient from constant `D0` (mean-prior 0.01 squared micrometres/min). To
calculate only the new scenario, use
`julia --project=paper paper/protein/run.jl --full --model=cnt_growth_grd`.
This does not enter the production paths for the original two scenarios. Their
draws, latent bands and producer manifests are retained; new compatibility and
Gauss–Newton records live in `protein/data/analysis/`. The source review in
`_support/protein_case_review.json` is tied to the checked source inventory and
is invalidated by further numerical edits. A redraw checks recorded producer
integrity rather than asserting equivalence to a newer checkout. Digitized PSDs
and quantiles are scatter only; numerical density lines connect saved bin
averages without smoothing or remeshing.
