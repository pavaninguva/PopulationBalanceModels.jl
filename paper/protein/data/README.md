# Fixed-gamma CNT and constant-diffusion GRD

Fresh calibration-only fits and posterior samples for the revised Tang isothermal
comparison: gamma = 0.64 mJ/m², g ~ Uniform(1,5), three CNT or four CNT+GRD
coordinates. Population states use particles/L; lengths μm and time min.
The D0 prior is Exponential with mean 0.01 μm²/min, a declared heuristic.

Each model folder retains its original producer settings, source/data hashes,
MAP, raw NUTS draws, checkpoint diagnostics, latent forward predictions and
refinement checks. `shared/` holds the unchanged observations and fixed-error
assumptions. `analysis/tables/` contains derived summaries, not new fit inputs.
`producer_recipe.zip` preserves the exact example, drivers and configuration
used by these runs. Subsequent reporting-only fixes put refreshed verification
receipts in `analysis/` and avoid re-exporting shared inputs in verification
mode, preserving the original producer records and posterior definition.

The earlier variable-gamma, D = theta G² targets, draws, figures and uncertainty
receipts are preserved in
`test/fixtures/archives/paper_history/protein_variable_gamma_theta_20261028.zip`
with an internal file manifest and adjacent SHA256. They are historical results,
not draws for the active model. The producing working tree was dirty; source
hashes and captured settings, rather than the commit name alone, identify it.
