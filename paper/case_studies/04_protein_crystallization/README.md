# Isothermal protein crystallization

This coupled population–solute model uses the original Tang 100 mL isothermal
concentration and conditional **number** distributions.
It does not rerun the older Mitchell-based Bayesian example.

From the repository root:

```sh
julia --project=paper paper/setup.jl
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl --smoke
julia --project=paper paper/case_studies/04_protein_crystallization/run.jl
python paper/plot_all.py --case 4
```

Smoke mode runs an uncalibrated forward calculation only. Production runs fit
the baseline and one predeclared onset candidate, select using calibration,
check/refit numerical resolution, freeze, predict the held-out data, and refit
the predeclared sensitivities. No posterior sampler runs. Each optimization
start is checkpointed under `paper/results/04_protein_crystallization/production/fits/`.
The cache includes the numerical definitions, data, settings and seed. Delete
an individual checkpoint to repeat that optimization; the retained timing is
the original measurement, not a timing of loading the checkpoint.

The active observation CSVs, immutable input configuration and numerical
digitization provenance are under
`examples/data/protein_crystallization/tang2018_isothermal/`.
`frozen_model.json` records the choices before the validation evaluation.
`manifest.json` checksums the completed output and its defining source files;
plot-only regeneration rejects changed or incomplete inputs. Final figures use
actual LaTeX and are written to `paper/figures/04_protein_crystallization/production/`.

Loss weights are comparison units, not measured sensor uncertainties. Profile
losses and start dispersion must not be presented as confidence intervals.

Verify the retained numerical exports with `python paper/check_protein.py`.
The result register records original optimizer timings; the final forward solve
is measured separately after compilation warmup, excluding model setup and I/O.
`cache_metadata_reuse.json` records verified metadata-only cache migrations
(resolver fingerprint and test-only dependencies); numerical sources, data and
resolved runtime package versions were unchanged.
