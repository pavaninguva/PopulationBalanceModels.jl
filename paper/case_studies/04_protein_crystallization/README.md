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

Verification on Julia 1.10.3 passed the 2,131-check core suite, 500 focused
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
