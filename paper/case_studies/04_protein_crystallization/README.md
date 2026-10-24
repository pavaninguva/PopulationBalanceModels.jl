# Isothermal CNT batch: data → model → Bayesian fit → prediction

The active calculation uses the unchanged original Tang 100 mL concentration
and conditional **number** distributions. Nucleation and growth follow Mitchell
Case 3 (CNT plus empirical growth). Table 3 informs explicitly declared priors;
it did not publish the Bayesian distributions used here. Final same-data
estimates are not reused as independent priors.

From the repository root:

```sh
julia --project=paper paper/setup.jl
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --smoke
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --sample
python paper/plot_all.py --case 4
python paper/check_protein.py
```

Smoke runs an uncalibrated forward solve. Production estimates the shared
posterior mode from three starts, verifies any Laplace approximation, freezes
the fit, predicts held-out measurements and checks conservation and refinement.
`--sample` additionally runs four NUTS chains (300 warmup, 500 retained draws
per chain). Set `PBM_PROTEIN_WARMUP` and `PBM_PROTEIN_SAMPLES` to explicitly
change that budget. No posterior summary or predictive interval is generated
unless the package's diagnostic screen passes.

All physical and inference definitions live in
`examples/isothermal_protein_crystallization.jl`. This directory has one thin
reproduction entry point, not separate fitting, onset-selection and reporting
frameworks. The original power-law/onset outputs were superseded; their history
is retained in Git at e101043. Historical Mitchell audit assets remain separate.

Production CSVs/JSONs record fitted parameters, individual concentration/CDF
predictions, signed residuals, quantiles (diagnostic only), material balance,
refinement, execution times and source hashes. Plot-only regeneration verifies
those hashes. PDF/300-dpi PNG/SVG figures use actual LaTeX. The model, likelihood
and prior assumptions are explained in the manual's protein case study.

`solve_seconds` measures the complete forward call, including model setup,
after a compilation warmup; it is not a comparative solver benchmark.
The three-start optimizer and sampler have separate wall-time reports.
The initial diagonal-tuning sampling attempt was interrupted before producing
retained results; `sampling_pilot.json` records it. The production run uses
expected-information tuning, without interpreting it as posterior covariance.

Distribution distances are reporting diagnostics. They integrate squared
quantile differences exactly for the tabulated CDFs and resolved bin densities;
2048/4096/8192 midpoint comparisons are retained as a quadrature check.
The original sampling producer and reporting-only corrections are recorded in
`provenance.json`; posterior target values are rechecked before reuse.
