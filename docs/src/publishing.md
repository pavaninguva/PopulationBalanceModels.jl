# Documentation builds and publishing

## Local verification

From the package root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. Documenter is a documentation dependency and the
HTML build is excluded from source archives. Ordinary local builds do not deploy.
Source and edit links are disabled when `GITHUB_REPOSITORY` is unset; CI obtains
the owner and repository from that environment variable.

## Repository address

The owner has not been supplied. This is the single owner placeholder for these
instructions: `OWNER`. Use the actual account in the following templates:

| Purpose | Address template |
|:--|:--|
| Repository | `https://github.com/OWNER/PopulationBalanceModels.jl` |
| Development manual | `https://OWNER.github.io/PopulationBalanceModels.jl/dev/` |
| Release manual | `https://OWNER.github.io/PopulationBalanceModels.jl/stable/` |

URL installation uses `Pkg.add(url=repository_url)` followed by
`using PopulationBalanceModels`. Registry availability is not assumed.

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
`/stable/` alias. Stable is a version alias, not a scientific validation claim.
The development version remains `0.4.0-DEV`; this archive does not create a
release, register the package, push changes, or publish documentation.

If deployment fails, check the Documentation job, matching SSH deploy key and
secret, actual repository slug, and Pages branch settings. A missing `/stable/`
before a release is expected. Keep the docs compatibility range aligned with
future package versions. CI tests Julia 1.10 and current Julia on Linux and
Windows; the optional job exercises Optim and SciML adapters independently.
