# Superseded growth runs

`superseded_runs.zip` retains 36 run directories (108 numerical/provenance files)
that were not selected by `../production/index.toml` at the cleanup baseline
`ba90fd09049ea227bc7658d949f66aec66b9524f`. Archive entries retain their original
run-directory names and bytes. Extract into a separate directory to inspect
their inputs or timings; do not merge them into the active index.

The archive includes earlier refinement configurations and timing measurements.
They are historical measurements, not repeated samples of the current timing
experiment. Active production data, figures, indexes and checksums are unchanged.

`exponential_production.zip` preserves the former paper growth dataset and its
producer scripts before the Gaussian/manual alignment. Original numerical files
and manifests retain their bytes; the active Gaussian workflow is `paper/growth/`.
