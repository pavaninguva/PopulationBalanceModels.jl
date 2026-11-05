# Documentation builds and publishing

## Local verification

From the package root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
python tools/retained_data.py fetch --profile docs
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. Documenter is a documentation dependency and the
HTML build is excluded from source archives. Ordinary local builds do not deploy.
Source and edit links are disabled when `GITHUB_REPOSITORY` is unset; CI obtains
the owner and repository from that environment variable.

## Retained case-study outputs

Retained figures and numerical tables are optional reproduction payloads. The
explicit fetch above reads an immutable source snapshot and verifies every
selected file's SHA-256 and size. It runs no numerical calculations. Existing
differing results are not overwritten. `--profile paper` prepares the publication
cases; `--profile all` also prepares historical uncertainty fixtures. Numerical
package use and small authored tutorials need none of these downloads.

The storage inventory is separate from the original producer manifests. A
successful download establishes byte integrity, not equivalence of a historical
producer to the current source. Preserve the verification qualifications in the
[reproduction guide](verification.md#Retained-results-and-present-source-equivalence).

## Repository address


The repository is `pavaninguva/PopulationBalanceModels.jl`. Its source and
versioned manual addresses are:

| Purpose | Address |
|:--|:--|
| Repository | `https://github.com/pavaninguva/PopulationBalanceModels.jl` |
| Development manual | `https://pavaninguva.github.io/PopulationBalanceModels.jl/dev/` |
| Release manual | `https://pavaninguva.github.io/PopulationBalanceModels.jl/stable/` |

URL installation uses `Pkg.add(url=repository_url)` followed by
`using PopulationBalanceModels`.

## Versioned GitHub Pages deployment

The retained Documentation workflow builds main, ordinary version tags, and pull
requests. Deployment is gated by `POPULATIONBALANCEMODELS_DOCS_DEPLOY=true` and
uses `GITHUB_REPOSITORY`; pull requests build without deploying. A dedicated
`DOCUMENTER_KEY` repository secret authenticates Documenter's SSH push to
`gh-pages`. Generate a matching key pair with DocumenterTools in a separate
environment, supplying the actual user and `repo="PopulationBalanceModels.jl"`.
Install the public half as a repository deploy key with write permission and
the private half as the Actions repository secret. Never commit either private
credentials or local environment files.

In repository Settings → Pages choose deployment from branch `gh-pages` at
`/ (root)`. Main builds publish `/dev/`. After an ordinary `vX.Y.Z` release tag
with a matching package version, Documenter supplies versioned pages and the
`/stable/` alias. The prepared package version is `0.4.0`; registration,
tagging and deployment are separate release actions. Historical producer
manifests retain their original development-version metadata.

If deployment fails, check the Documentation job, matching SSH deploy key and
secret, actual repository slug, and Pages branch settings. A missing `/stable/`
before a release is expected. Keep the docs compatibility range aligned with
future package versions. CI tests Julia 1.10 and current Julia on Linux and
Windows; the optional job exercises Optim and SciML adapters independently.
