# Isothermal CNT and CNT + GRD comparison

The physical models, same-stage solute balance, fixed-error likelihood and
four/five-parameter priors live in `examples/isothermal_protein_crystallization.jl`.
`run.jl` is the entry point; `report.jl` handles exports and checkpointed isolated
NUTS chains. It contains no independent PBM or statistical target.

```sh
julia --project=paper paper/setup.jl
python -m pip install -r paper/requirements.txt
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --data-only
python paper/plot_all.py --case 4 --data-only
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --smoke
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --map
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --full
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --verify
python paper/plot_all.py --case 4
python paper/check_protein.py
```

To restyle or regenerate the figures from the retained production exports:

```sh
python paper/plot_all.py --case 4 --figures-only
```

This command checks the saved-result manifests, then updates the five figures,
captions and plot provenance. It reads the retained predictions and posterior
draws without running Julia, refitting parameters or rewriting numerical result
tables. PNGs are rendered from the vector PDFs at 300 dpi; PDF and SVG exports
retain actual LaTeX typography. Density curves connect the saved bin averages at
their bin centers, without a smoothing filter.

`--model=cnt`, `--model=cnt_grd` or `--model=both` (default) select models.
The no-flag command is MAP-only; `--sample` aliases `--full` and produces both
posterior runs unless one model is selected. `--bins=N` overrides both default
meshes (CNT 400, GRD 100). CNT retains 500 samples per chain and GRD requests
2,000 per chain, each in four independent worker processes. CNT uses 300 warmup iterations and target
acceptance 0.97 after a retained pilot had one divergence. The GRD configuration uses
1,000 warmup iterations, target acceptance 0.999 and maximum tree depth 12.
Its positive-barrier rate coordinates are followed by a frozen, invertible
calibration-pilot preconditioner; half the starts use each calibration basin.
`sampling_transport.json` records parameter-space CDF, shift and conditional
Cholesky coefficients and the initial tuning matrix. These are not extra
physical populations, statistical priors, observation-noise parameters or
posterior covariance estimates. The full inverse-map Jacobian is included.
The auxiliary growth-rate
coordinate uses reference supersaturation 6, selected from calibration-pilot
geometry rather than held-out agreement. Earlier tuning attempts, including
the completed 0.90 run with 49 retained divergences, are archived separately. The joint
screen, rather than the requested sample count, determines completion.

The retained production runs pass that screen with zero divergences or depth
hits: CNT has maximum Rhat 1.00585 and minimum bulk/tail ESS 782/681; GRD has
1.00185 and 2,542/2,200. Their four-worker posterior wall times were 58.5 min
and 7.99 h, respectively, on the recorded i9-13900H/Julia 1.10.3 environment.
These include warmup and depend on the sampler configuration; they are not
forward-solve benchmarks. Matching completed checkpoints are reused.

The GRD production mesh was reduced only after calibration-only 100/200-bin
checks: CDF differences were at most 0.0032 at the tested parameter points
versus the assumed 0.03 CDF standard deviation; concentration changes were
below 0.007 g/L. Its kinetics and posterior are fitted afresh on 100 bins.
The active reports retain 100/200 and 200/400 comparisons, a finer-grid refit,
and representative posterior refinement checks. The interrupted 200-bin
high-acceptance run supplies no posterior intervals.
`--warmup=N --samples=N --chains=N --seed=N --predictive-draws=N --target-acceptance=P` changes explicit
budgets. Checkpoints match numerical source, parameters and sampler settings;
`--force` refits the MAP. Do not reuse the archived six-dimensional draws.
`--verify` re-evaluates retained posterior densities, parameter transformations
and AD/mapped gradients at the first and last draw of every chain; it never fits.

Results in `production/shared` are data-only. Each model has its own folder,
source/configuration manifest, normalized prior metadata, MAP, physical/z draws,
sampler statistics, latent predictions, residuals, quantiles, material and
numerical checks. Plots validate these manifests and generate five figures and
the common parameter/prior tables. The test observations do not enter fitting.
Noise scales are fixed assumptions (5% of each observed concentration and 0.03
absolute correlated-CDF fraction), not nuisance parameters or measured errors.

Earlier reproduction verification on Julia 1.10.3 passed the 2,131-check core suite, 500 focused
case/sampler checks, and 64 optional integration checks in the isolated CI-style
environment. The independent Python result checker passed all six contracts.
Documenter built successfully; its local check found zero broken links/assets
among 2,388 references. All five PDFs have embedded fonts and matching 300-dpi
PNGs; the generated LaTeX tables were compiled and visually checked.

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=paper -e 'include("test/isothermal_protein.jl"); include("test/protein_parallel.jl")'
python3 test/docs_links.py docs/build
git diff --check
```

No source PDFs are tracked. Old outputs and producer sources are preserved in
`paper/results/04_protein_crystallization/historical/cnt_inferred_noise_8abcd0f.zip`.
That archive has a different likelihood and six inferred coordinates. Current
results must be generated afresh. Interrupted sequential pilots are recorded
separately and are not posterior samples for the displayed comparison.
The fixed-error sampling pilot archive retains the failed CNT screen and
interrupted direct-rate GRD attempt. The positive-barrier map and tests live in
`examples/support/isothermal_protein_sampling.jl`; they change sampling
coordinates, not the model or priors. The numerical prior-wall probe is retained
in `production/shared`, alongside the initial mesh study.

## Retained-target compatibility and Laplace analysis

These commands never call the reproduction/cache-invalidation path. In
particular, they do not delete posterior files, fit new chains, propagate latent
predictions or rewrite production tables:

```sh
# Re-evaluate unchanged targets at the saved MAP and retained chain/tail points.
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --compatibility-only
# The same compatibility check, followed by public full-posterior Laplace checks.
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --laplace
python paper/plot_all.py --case 4 --figures-only
python paper/plot_all.py --case 4 --laplace-comparison
python paper/check_protein.py
python test/protein_publication.py
```

`--model=cnt` and `--model=cnt_grd` also work for the Julia analysis modes.
New evidence and tables live under `results/04_protein_crystallization/analysis/`
relative to `paper/`, separate from the original producer manifests. The review
reconstructs their source blobs at 80a7d76; their original producer record still
reports its actual dirty 8abcd0f checkout. Four subsequent changes are the
reviewed stationary exports and unchanged packing/accounting/axial refactors.
Required and resolved dependency versions are unchanged. Compatibility receipts
record the exact current source set, settings, priors, transformations, data and
original file hashes. Plotting accepts them only while those checks remain
current; arbitrary future changes require a new explicit review. The original
manifests are not relabelled as current production runs.

Both saved optimizer results retain verified stationarity. Neither current
protein target passes the public Laplace safeguards at any of three derivative
verification step sizes: CNT's Hessian agreement is 0.070–0.236; CNT+GRD's
agreement is much smaller (0.00018–0.00057), but its smallest eigenvalue (0.971)
is below the absolute derivative-verification resolution (about 1,973–6,224).
Positive eigenvalues alone do not resolve weak curvature. No covariance, Gaussian
corner or Laplace interval is reported. The comparison figures show these
rejection diagnostics beside the retained MCMC corners, with individual panels
for manuscript layouts. Their marginal tables explicitly mark Laplace as
rejected. The MCMC fit bands, posterior draws and kinetic estimates are reused
unchanged; no local MAP refinement was needed.

`laplace.json` retains every attempt, full-posterior coordinates, eigenvalues,
condition estimates, gradient checks and evaluation counts. A successful future
analysis would export its full Hessian/covariance, exact transformed marginals,
seeded Gaussian parameter draws and any prior-support violations without
filtering. None is substituted for this rejected result. The tuning matrices in
the sampler remain tuning, not posterior covariance estimates.

The publication plotting handoff was integrated as source changes. The five
main plots share corresponding axes and outside panel letters, use normalized
analytical prior overlays and connect stored density bin averages at their
centers for display only. All figures have LaTeX PDF/SVG and matching 300-dpi
PDF-rendered PNGs; matching files are copied into the manual. Corner axes use
unit-reference logarithms; broad priors are not renormalized to a cropped view.

This update reran 51 Bayesian/Laplace API checks, the 500 focused protein and
synthetic sampler checks, four read-only analysis-mode checks, and 20 Python
result/transformation/plotting contracts. No runtime numerical source changed,
so the full core suite was not repeated. Documenter (Julia 1.10.12) built with
the existing API/search-size warnings only; local link/asset and embedded-font,
300-dpi, paper/manual-copy checks also passed. Analysis used Julia 1.10.3,
Optim 2.3.2 and ForwardDiff 1.4.6, matching the producer environment. Plot receipts
record the actual Python/NumPy/Matplotlib versions. All 140 retained production
files, including draws, predictions, numerical tables and original manifests,
remained byte identical.

## Sensitivity-based posterior covariance

The package now exposes `gauss_newton_approximation(problem,map)` and
`fit(problem; uncertainty=:gauss_newton)`. The direct calculation uses the
joint whitened Gaussian prediction Jacobian plus the actual prior curvature in
inference coordinates, including transformation Jacobians. It omits
residual-weighted second prediction derivatives and is deliberately distinct
from the full-Hessian Laplace method and from sampler tuning.

```sh
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --gauss-newton
python paper/plot_all.py --case 4 --gauss-newton-comparison
```

The analysis is read-only with respect to `production/`. It reconstructs the
same genuine retained MAP results and checks current posterior values/gradients
against the unchanged target before calculating uncertainty. The new inference
API is reviewed in `uncertainty_compatibility.json`, bound to exact source hashes;
changes to that review or its sources invalidate reuse. Older compatibility
receipts are kept under each model's `analysis/.../history/`. Rechecking an
identical target preserves its compatibility receipt, so running the two
uncertainty modes in either order does not invalidate the other calculation.
Neither command samples, refits or propagates new PBM prediction bands.

Each model writes `analysis/<model>/gauss_newton.json`, including the Jacobian,
likelihood information, prior Hessian, joint precision/covariance, parameter
order and numerical checks. The plotting command produces separate
`protein_gauss_newton_mcmc_<model>` comparisons and individual corner panels,
plus `gauss_newton_marginal_comparison.{csv,md,tex}` and a correlation table in
each model's analysis folder. Analytical transformed Gaussian marginals and
probability ellipses use the same displayed coordinates, priors and MAP as
MCMC. Cheap seeded Gaussian draws are untruncated; support violations are
reported. The main physical-result figures retain their existing MCMC bands.

The numerical checks accept both current protein calculations. Covariance-scaled
information discrepancies are approximately 0.024% and 0.11%, with joint
precision condition numbers 2.21e6 and 3.32e7. The GRD check required a smaller verification step to resolve its near-mode gradient; the rejected first attempt is retained. This checks local derivatives of
the existing discrete models, not the approximation's global statistical
accuracy. CNT's growth-exponent interval is substantially wider than MCMC's;
GRD's local approximation misses its low-dispersion tail and compensation basin.
The corresponding CSVs retain the disagreement rather than presenting either
approximation as interchangeable with the sampled posterior. Full-Hessian
Laplace still fails its original safeguards; those checks were not relaxed.

The initial package support is fixed scalar/array Gaussian errors and covariance
matrices, including independent contrasts. Inferred/noise callbacks, count and
truncated-Gaussian likelihoods explicitly reject. Posterior and prior coordinates
remain those declared by `ParameterSpec`; data-null directions supported by the
prior are reported as such. No residual variance inflation, diagonal jitter or
pseudoinverse is used.

Final verification of this capability passed 2,304 core checks on Julia 1.10.12,
181 optional integration checks, 65 focused analytical/Optim checks on each of
Julia 1.10.3 and 1.13.2, six read-only mode checks and 21 Python contracts.
Documenter built with only its existing API/search-size warnings; 2,622 local
links/assets had zero failures. New figure PDFs have embedded fonts, PNGs are
300 dpi, and paper/manual copies match. The full command/environment record is
`paper/results/04_protein_crystallization/analysis/gauss_newton_verification.json`.
