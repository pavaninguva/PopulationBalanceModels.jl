# Retained historical fixtures

`legacy_docs_assets.zip` preserves the superseded protein posterior and older
calibration/compact/illustration outputs that are no longer part of the manual.
The adjacent manifest records every original file hash. Current Tang CNT/GRD
samples remain uncompressed under `paper/protein/data/`.

To run the historical protein audit or plot its old target, extract once:

```sh
python tools/retained_data.py fetch --group historical-tests
python -m zipfile -e test/fixtures/archives/legacy_docs_assets.zip .cache/legacy_docs_assets
julia --project=paper test/audit_protein.jl
```

These are historical targets, not current protein estimates. Extraction never
writes into `docs/src` or changes current production outputs.

`paper_history/` retains the historical protein fixture required by regression
tests, the pilot behind a frozen compatibility preconditioner, and original
producer source snapshots. Current samples and all recorded
observation/likelihood inputs remain together in `paper/protein/data/`.

`paper_history/protein_variable_gamma_theta_20261028.zip` preserves the previous
protein target with inferred gamma and D = theta G²: the complete model/shared
outputs, original manifests, posterior draws, uncertainty receipts/history, source recipe
and figures. Its internal manifest and adjacent SHA256 record the original
bytes. The active protein case fixes gamma and compares CNT, constant-D0 GRD
and growth-dependent GRD with three/four inferred coordinates. Historical
covariance/coordinate tests replay this archive explicitly; they do not claim
equivalence to the active targets.

## Relocated legacy outputs

On 2026-11-03, 50 unused historical exports (39.7 MiB) were moved outside the
checkout. The [relocation manifest](relocated_assets_20261103.json) records
each original path, byte count, SHA-256 and reason, together with the recovery
archive's SHA-256. It is separate from the unchanged producer manifests.

The recovery archive is `legacy-assets-f5a9a86-20261103.zip`, in the sibling
directory `PopulationBalanceModels-legacy-20261103/`. It contains the retired
paper studies and entry points, old constant-growth tensor exports, superseded
uniform-size precipitation images, and dated development audit reports.
To inspect them from the repository root:

```sh
python -m zipfile -e ../PopulationBalanceModels-legacy-20261103/legacy-assets-f5a9a86-20261103.zip /tmp/pbm-legacy-review
```

Entries retain their original checkout-relative paths; nested archives retain
their original bytes. Extract into a separate directory before selecting any
files to restore. These historical targets must not be merged into active
numerical indexes or used as current posterior uncertainty.

The uniform-size precipitation CSVs remain because a documented historical
plot command and retained-data tests still consume them. The two Qamar
diagnostic CSVs remain linked from the moment case. Current figures, numerical
datasets and required historical fixtures were not relocated.
