# Manuscript calculations

These drivers are independent of Documenter and never read documentation assets.
They include the existing physical-model/example modules without running those
modules' standalone benchmarks. The fourth case is the
[Tang isothermal protein batch](case_studies/04_protein_crystallization/README.md),
comparing CNT kinetics with CNT plus random-fluctuation growth dispersion.
It runs independently of the first three. Its two models share fixed observation
uncertainties; four- and five-dimensional NUTS posteriors supply separate corner
plots and latent prediction bands.

Run commands from the repository root. Julia must load **this checkout**; the
drivers check `pathof(PopulationBalanceModels)` before doing any calculation.
The workflow was tested on WSL/Linux with Julia 1.10.3 and Python 3.12.3.
Use Python 3.12 for the pinned plotting environment (`tomllib` needs at least 3.11).

## Directory map and retained results

- `case_studies/`: one entry point and configuration for each of the four cases.
- `results/<case>/production/`: indexed numerical data and provenance.
- `figures/<case>/production/`: retained manuscript figures.
- `verification/`: optional solver pilots and the caption compilation check.
- `literature/` and `provenance/`: source-data attribution and producer records.

Smoke results and figures are regenerable and ignored by Git. Only the protein
smoke/pilot records that were already retained are kept with its other evidence.
Superseded growth runs are in one [historical archive](results/01_growth_time_methods/historical/README.md).
The active growth index and production figures are unchanged.

Protein samples are intentional reproducibility assets. Keep `posterior_draws.csv`,
sampler statistics, manifests, coordinate maps, priors, prediction exports and
historical sample archives together. Binary worker checkpoints remain ignored
local caches and are preserved by this cleanup. No posterior is regenerated.

To inspect retained outputs after unrelated package development:

```sh
python paper/check_protein.py --retained
python paper/check_data.py --case 1 --retained
python paper/check_data.py --case 2 --retained
```

These verify stored file hashes and scientific/data contracts without asserting
that the current source tree is the original producer. The default checks and
production plot generation remain strict. The retained datasets predate the
stationary extension, so their source-match gates already reject this checkout;
do not rewrite their provenance to make that gate pass. Preserve the recorded
producer state when reproducing or comparing historical timings.

The tubular paper dataset is incomplete: its active index has no
`time_refinement` run, and production precipitation figures have not been
generated. `python paper/check_data.py --case 3 --retained` correctly rejects
that incomplete dataset. The documentation's separate tubular figures are not
substitutes. Existing partial data and solver-pilot evidence are preserved;
completing that numerical study is separate from repository organization.

## Environment and execution

```sh
julia --project=paper paper/setup.jl
python -m pip install -r paper/requirements.txt
```

Final plotting also needs a working TeX installation with `latex`, `dvipng`,
`amsmath` and Computer Modern fonts. Figures use external LaTeX, not mathtext.
Each PDF is checked for embedded fonts; each PNG carries 300-dpi metadata.
The Julia manifest is retained; package solver dependencies are unchanged.
Its development path records this checkout; `setup.jl` rewrites that path if the
repository is moved while retaining resolved dependency versions.
For the first three datasets, `provenance/numerical_checkout.patch` and
`numerical_sources.json` capture their uncommitted numerical changes, including
nonuniform WENO5. The later protein runs record their own numerical sources and
producer hashes in each model's result folder. Git HEAD alone does not identify
either implementation. Apply a snapshot only to its matching clean baseline;
the drivers always load the normal root package.

First exercise the small configurations and the numerical contracts:

```sh
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=paper paper/run_all.jl --smoke
julia --project=paper paper/checks.jl
python paper/plot_all.py --smoke
python paper/check_data.py --smoke
```

The production calculations, sequentially (`run_all.jl` fits the protein MAPs;
the explicit second command completes both protein posteriors):

```sh
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=paper paper/run_all.jl
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --full
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --verify
python paper/plot_all.py
python paper/check_data.py
python paper/check_protein.py
```

Or run a case independently:

```sh
julia --project=paper paper/case_studies/01_growth_time_methods/run.jl
julia --project=paper paper/case_studies/02_aggregation_fragmentation/run.jl
julia --project=paper paper/case_studies/03_tubular_precipitation/run.jl
python paper/plot_all.py --case 1
python paper/plot_all.py --case 2
python paper/plot_all.py --case 3
```

The last three commands are **plot-only** regeneration. Numerical files, current
model/driver hashes and mode are validated first. Smoke data live under
`results/<case>/smoke/`; final data under `production/`. The plotter cannot use
smoke data for a production figure. `--smoke --preview` permits a clearly named
fallback preview on a machine without TeX; it cannot generate final figures.

Each case has one TOML file with separate smoke and production configurations.
Matching completed runs are reused. A cache key includes model and extension
sources, example builders, the case driver/configuration, shared workflow code
and the resolved Julia environment. Every dataset records the actual Git HEAD,
full dirty working-tree state, source hashes, parameters, grids, methods,
tolerances, CPU, Julia/package versions, threads, step count and measured cost.
Changing plotting style does not rerun a solve. A changed numerical input does.
A reused timing still belongs to the CPU and environment in its provenance;
it is not a new measurement on the current machine. The plotter rejects mixed
CPU/Julia/thread/dependency environments within a case. Archive the old mode's
results before measuring on a different machine.

Only one measured solve is used per configuration after a separate small
compilation warmup. Model/grid assembly, explicit garbage collection, export and
plotting are excluded from integration time. Integrator initialization, checks
and saved states are included.
Tubular timings also include sparse console progress messages; numerical data
exports occur after the timer stops.
Production timings must be run without concurrent substantial workloads. They
are single measurements, with no repeatability error bars.

## Drivers and source problems

- [Growth driver](case_studies/01_growth_time_methods/run.jl) and
  [configuration](case_studies/01_growth_time_methods/config.toml): translating
  exponential from [Inguva, Schickel and Braatz (2022), Case 1](https://web.mit.edu/braatzgroup/Inguva_ComputChemEng_2022.pdf),
  plus the explicitly declared smooth periodic test.
- [Moment driver](case_studies/02_aggregation_fragmentation/run.jl) and
  [configuration](case_studies/02_aggregation_fragmentation/config.toml):
  [Qamar et al., problem 4](https://doi.org/10.1002/aic.12264), published online
  in 2010 and in AIChE Journal in 2011.
- [Tubular driver](case_studies/03_tubular_precipitation/run.jl) and
  [configuration](case_studies/03_tubular_precipitation/config.toml):
  [Zhang et al. (2024), Case 7A](https://doi.org/10.1016/j.compchemeng.2024.108612).

The reused builders are [constant growth](../examples/constant_growth_methods.jl),
[aggregation/fragmentation](../examples/qamar_moment_benchmarks.jl) and
[precipitation](../examples/continuous_precipitation.jl). The paper drivers handle
configuration, controlled execution, numerical comparisons and export.
[precipitation_solver.jl](precipitation_solver.jl) wraps the existing ODE adapter;
[plot_all.py](plot_all.py) and [plotting_style.py](plotting_style.py) handle figures.

## What the figures measure

**Growth.** Exactly six pairings are used. `Inguva2022` has its native `C=1`
update; HOC is `HOCFiniteVolume` with its native midpoint update. Other pairings
use `C=0.5`. Profiles at 4 s have freshly solved **100-bin** data. Both initial
and reference populations are analytical bin integrals. Smooth-periodic RMSE
is width weighted; L1 is the integrated absolute density error. Observed orders
are undefined if either error is within `1000 eps(Float64)` of the order-one
reference scale, including the characteristic method's nearly exact shifts.
The initial 40--640 series showed WENO slopes exceeding nominal orders; the
manuscript series therefore extends to 1,280 and 2,560 bins to expose finite-range
transitions and roundoff. Superseded exploratory timings are kept separately;
they are neither averaged nor selected to improve a work--precision curve.
Sharp-front undershoots are retained with nonnegativity checks explicitly
disabled for this diagnostic comparison.
The unchanged defaults are `MUSCL(theta=1.5)` and WENO smoothness parameter
`epsilon=1e-6`; the complete HOC update has time order two.

**Moments.** Only Qamar problem 4 is used. No pulse or problem-5 collision run is
started. Histories use the representation-aware moment interface, including
DQMOM's weights/weighted abscissas. The lognormal third moment is reconstructed,
not an evolved conserved quantity. The main figure uses each method's base
timestep; halved steps and QMOM four-node runs are separate supporting data.
Errors in the first and second moments are percent absolute relative errors;
signed differences are also exported. The logarithmic error panels omit values
at or below `1000 eps(Float64)` relative (2.22e-11 percent), including initial
roundoff artifacts. Values are never floored or changed in the numerical data.
The `tracked` column identifies membership in the raw moment-equation basis;
DQMOM evolves weights and weighted abscissas rather than raw moment entries.
The derived summary labels that state layout and each material evaluation.
Plot-derived tables are written to each result mode's `analysis/` subdirectory;
solver exports and their checksums are immutable during plot-only regeneration.
The two Gamma-EQMOM first-moment error dips and two QMOM second-moment dips
correspond to signed zero crossings. Accurate lower moments do not make the
lognormal reconstructed third moment a conserved quantity.
Error maxima and reported ranks refer to exported times; the integrators still
apply their state/closure checks internally.

**Tubular precipitation.** Case 7A uses actual logarithmic edges from 1 to
900 micrometres and a uniform axial grid. Size reconstruction is the checkout's
geometry-aware WENO5; axial reconstruction is MUSCL. It is not CADET WENO35 in
both coordinates. The state is bin-integrated number scaled by `10^12`; exports
restore physical number concentrations before division by actual bin widths.
Zero initial solute/population, primary and secondary nucleation, the closed
size domain, Danckwerts inflow and zero-gradient outlet are unchanged.

The native explicit pilot required 11,387 steps to 5 s on 100 x 100 cells.
Its exploratory wall time (about 158 s) included possible background setup work
and is not a comparative production timing. Production uses the existing SciML residual adapter with
CVODE-BDF with stability-limit detection and factored-transport preconditioned GMRES. Bidiagonal Upwind size
transport, tridiagonal axial transport and diagonal solute depletion approximate
the Newton matrix for the linear solve; they do not change the WENO5/MUSCL residual or
project physical states. Every accepted internal step is checked with the same
native state checker/tolerance. Newton iterates are auxiliary states. This is
not a positivity-preserving method and inherits no native RK guarantee.
Production caps BDF order at three through 5 s, then restarts at maximum order
two from an exact copy of the accepted population and bulk state. This discards
numerical history, without splitting the physical equations. The initial step
is 1e-7 s in each phase; phase step counts and observed orders are retained.
The order cap is motivated by [CVODE's discussion of stability restrictions
for weakly damped advective modes](https://sundials.readthedocs.io/en/v6.6.1/cvode/Mathematics_link.html#bdf-stability-limit-detection);
the pilot costs alone do not diagnose the spatial operator's spectrum.
An untimed positive-growth toy exercises both phases before fresh measured
problems are built with the declared empty initial state.

Constant maximum-order-three/five fine pilots developed costly small steps;
order two was costly during startup. A BDF2 continuation from 5 s differed from
the completed 400 x 200 order-three calculation by at most 3.2e-7 in relative
population L1 at 40, 80 and 200 s. Independent native-RK4 tests cover 1, 5 and
10 s, including the restart. Same-mesh tighter-tolerance runs check production
accuracy. The [solver pilots](verification/precipitation/) reproduce these diagnostics.
`provenance/solver_pilots.json` preserves the
interrupted and superseded evidence. Those costs and incomplete trajectories
do not enter final figures; interruptions were for cost, not failed state checks.

The reference uses 400 logarithmic size bins and 400 uniform axial cells.
The temporal check uses that same mesh and all four
requested times. Size and axial checks each change one coordinate while holding
the other at reference resolution. All normalized L1 comparisons integrate
piecewise-constant densities on a **common union of bin edges** (and axial
partitions for fields), against the same numerical reference. This differs from
the paper's spline interpolation. Case 7 has no analytical coupled PSD solution.
For an outlet comparison,

\[
E_{L^1}=\frac{\int |n_h(x)-n_{\mathrm{ref}}(x)|\,dx}
                   {\int n_{\mathrm{ref}}(x)\,dx}.
\]

Field comparisons also integrate over axial position. These errors are
dimensionless fractions: 0.0038 means 0.38 percent.
They include differences between piecewise-constant representations on different
meshes; they are not measurements of fifth-order convergence. The separate
inventory, outlet-solute and material-moment comparisons help interpret that
representation error.
`analysis/checked_refinement.csv` also flags early outlet signals below the
nominal sum of component absolute tolerances. The 5 s outlet is effectively
empty, so its relative error is not an accuracy measure. Raw ratios remain
available when their denominator is positive; nonpositive totals are undefined.
Interpretation uses the field error until the outlet signal rises
above that scale. This scale is not a rigorous global error bound.

`accounting.csv` distinguishes dissolved and solid inventories, inlet/outlet
material fluxes, the instantaneous discrete inventory derivative, negative
population contribution and upper-bin occupancy. Area-normalized inventories
use the interstitial area; column totals multiply by `area * porosity`.
The rate defect is an instantaneous residual/face-flux identity, not a
postprocessed cumulative flux ledger. At birth, bin-integrated sectional material
uses the first bin centre; the discrete-to-critical-nucleus material ratio is
recorded for each mesh. No initial mass is silently assigned to another size.
The run summary's extrema cover all accepted CVODE steps; `accounting.csv`
reports the four saved fields.

Volume curves are normalized **size densities** per micrometre, integrating to
one with the actual widths; their discrete material weights use arithmetic bin
centres cubed. They are not density per log-size and are not the
paper's percentage of material in each logarithmic bin. Field colors share one
normalization. True zeros and signed nonpositive values cannot appear on a log
color scale; these are masked and the negative contribution is reported. There
is no spatial interpolation, spline smoothing or clipping.

## Literature inputs

`literature/provenance.json` records the source PDFs' hashes and extraction
calibration; no PDF is copied here. To regenerate source data:

```sh
python paper/fetch_literature.py
python paper/extract_literature.py --pdf local_papers/Zhang2024.pdf
```

The pinned public CADET numerical file uses size Upwind and axial WENO23 on
200 x 200 cells, CADET 5.0.4. It is labelled accordingly. Figure 9(c)'s blue
WENO35/WENO35 curve is digitized separately over the unambiguous 1.5--40 micrometre
range, with conservative two-pixel uncertainty. Ambiguous curve intersections
and low-density tails are omitted. Extracted points are literature data,
not package predictions or experimental observations. No full-domain error
is inferred from the partial digitization.

Physical parameters follow SI Table S8. The package retains its rounded
`Q=1.67e-7 m^3/s`; the public CADET input uses `10 mL/min = 1.6666666667e-7 m^3/s`.
Both use `v_ax=Q/(area*porosity)`. The Figure 9(c) legend has axial counts
1, 2, 3, 13, 50, 200; this workflow uses the supported matching subset
2, 3, 13, 50, 200 plus the refined package reference. A one-cell axial mesh
is outside the package adapter's support and is not fabricated.

`captions.tex` supplies six standalone caption macros. Compact figure data,
per-run provenance and derived verification tables are retained with the figures.
The downloaded HDF5 and upstream settings script are regenerable ignored inputs;
their hashes, pinned URLs and upstream licence are retained.

## Reading the exports

| Production figure | Vector PDF | 300-dpi PNG |
| --- | --- | --- |
| 100-bin growth profiles | [PDF](figures/01_growth_time_methods/production/growth_profiles_N100.pdf) | [PNG](figures/01_growth_time_methods/production/growth_profiles_N100.png) |
| Growth convergence | [PDF](figures/01_growth_time_methods/production/growth_convergence.pdf) | [PNG](figures/01_growth_time_methods/production/growth_convergence.png) |
| Growth work--precision | [PDF](figures/01_growth_time_methods/production/growth_work_precision.pdf) | [PNG](figures/01_growth_time_methods/production/growth_work_precision.png) |
| Moment methods | [PDF](figures/02_aggregation_fragmentation/production/moment_methods.pdf) | [PNG](figures/02_aggregation_fragmentation/production/moment_methods.png) |
| Precipitation fields | Incomplete dataset; not generated | Not generated |
| Precipitation outlet | Incomplete dataset; not generated | Not generated |

Each `results/<case>/<mode>/index.toml` identifies the active run directories.
Within a run, `provenance.toml` lists checksummed numerical files and settings.
Growth files contain bin edges, integrated populations, densities and exact
values. Moment files contain raw signed differences as well as percent errors.
For precipitation, `population_1.csv.gz` through `population_4.csv.gz` hold
physical bin populations in Julia column-major order (size index fastest), at
5, 40, 80 and 200 s. `size_edges.csv`, `axial_edges.csv` and `bulk_*.csv` supply
their geometry and solute values. `outlet.csv` contains both linear-size density
conventions, with actual edges.

The `analysis/` tables collect timing, timestep/node comparisons, signed
crossings, reference differences and literature comparisons. Digitized-point
differences are pointwise over the extracted range, not a full-domain published
solution error. Public-CADET L1 differences compare two distinct discretizations;
they are not errors against an analytical reference.

## Architecture and retained protein analysis

```sh
# No Julia simulations:
python paper/architecture/generate_pbm_architecture.py --usetex
# Separate compatibility/Laplace evidence; no production cache invalidation:
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --laplace
# Read-only production inputs, matching paper/manual figures:
python paper/plot_all.py --case 4 --figures-only
python paper/plot_all.py --case 4 --laplace-comparison
python paper/check_protein.py
```

The [architecture generator](architecture/README.md) records its reviewed source
revision. The [protein workflow](case_studies/04_protein_crystallization/README.md)
explains the compatibility receipts and rejected Laplace calculations. Retained
MCMC results and original producer manifests are preserved, not relabelled as
new production calculations. The figures-only command requires exact source
matching or the separately verified, source-bound protein compatibility receipt.
It never rewrites numerical tables. Comparison outputs and marginal/correlation
tables live in the separate protein `analysis/` directory.
