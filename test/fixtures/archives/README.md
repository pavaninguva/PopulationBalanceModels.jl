# Retained historical fixtures

`legacy_docs_assets.zip` preserves the superseded protein posterior and older
calibration/compact/illustration outputs that are no longer part of the manual.
The adjacent manifest records every original file hash. Current Tang CNT/GRD
samples remain uncompressed under `paper/protein/data/`.

To run the historical protein audit or plot its old target, extract once:

```sh
python -m zipfile -e test/fixtures/archives/legacy_docs_assets.zip .cache/legacy_docs_assets
julia --project=paper test/audit_protein.jl
```

These are historical targets, not current protein estimates. Extraction never
writes into `docs/src` or changes current production outputs.

`paper_history/` holds superseded paper targets, pilot archives and original
producer records. They are preserved separately from the four current case
folders. Current samples and all recorded observation/likelihood inputs remain
together in `paper/protein/data/`.
