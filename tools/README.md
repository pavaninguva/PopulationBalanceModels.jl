# Retained scientific results

Numerical package use and small tutorials do not require production datasets.
Large numerical exports, manual figures and historical test archives are fetched
explicitly with this standard-library-only Python tool (Python 3.9+):

```sh
python tools/retained_data.py list
python tools/retained_data.py fetch --profile docs
python tools/retained_data.py fetch --profile paper
python tools/retained_data.py fetch --profile plotting-tests
python tools/retained_data.py verify --profile all
```

`docs` prepares the manual. `paper` also prepares all four publication cases.
`plotting-tests` adds the protein and historical fixtures used by optional
plotting tests. `all` fetches every payload. Use repeatable `--group` arguments
for individual cases; `list` reports their sizes.

The [inventory](retained_data.json) pins an immutable Git revision and records
the byte count and SHA-256 of each of 646 payloads (242.5 MiB before compression).
The revision identifies their **storage location**, not a new scientific producer.
Original producer manifests, settings, project snapshots, source hashes and
scientific qualifications remain unchanged in the repository and archived files.

## Fetching and preserving results

The default source is the public GitHub snapshot of the pinned revision.
It is downloaded once to `.cache/retained-downloads/` (override with
`PBM_RETAINED_CACHE` or `--cache`). Only explicitly selected, regular-file entries
are copied to their original checkout-relative locations, after checksum and
size verification. Unrequested archive contents are not extracted.

Existing files must match the inventory; differing files cause an error and are
never overwritten. This protects independent new calculations and presentation
changes. Fetching does not run Julia, fit parameters, sample chains, recompute
summaries or rewrite scientific producer metadata. Neither package loading nor
plot rendering fetches results automatically.

Materialized payloads are ignored by Git. Original paths are preserved, so the
existing plot-only and retained-data check commands work after preparation.
The current paper PNGs and four current Gaussian manual figures stay with the
source and are excluded from this download inventory, so a data fetch cannot
overwrite them. The small Gaussian figure-selection tables are also bundled;
they preserve the original numerical producer identities and timings.
For a new numerical production run, use its documented fresh output directory
and preserve its own provenance. A new snapshot needs a deliberately reviewed
inventory; do not update historical producer hashes to fit current code.

## Portable snapshots and mirrors

An offline snapshot can be made from verified existing results:

```sh
python tools/retained_data.py archive --output /path/to/retained-results.zip
python tools/retained_data.py fetch --archive /path/to/retained-results.zip
```

The local recovery copy made during this migration is in the sibling directory
`PopulationBalanceModels-reproduction-data/retained-results-f5a9a86.zip`, with its
SHA-256 alongside it. New users can use the pinned public source instead.
`--destination /path/to/checkout` restores original relative paths in a separate
directory for independent inspection or checksum tests. ZIP snapshots and the
pinned GitHub source tarball use the same per-file checks.

The previous legacy-only relocation is separate; see the
[historical-asset catalogue](../test/fixtures/archives/README.md). No Git history
is rewritten, and a source checkout can recover earlier scientific records.
