# Verification and reproduction

Run commands from a checkout's root so examples, data and supporting files keep
their relative paths. A successful build checks the manual's examples and links;
it does not establish solver accuracy or the equivalence of retained results to
new source code. Method-specific accuracy and applicability limits are described
alongside the algorithms and case-study results.

## Core and documentation

Use Julia 1.10 or later. The package in these commands is the checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
python test/docs_links.py docs/build
git diff --check
```

The core suite includes Aqua, analytical balances, independent references and
separate refinement checks. Statistics and Aqua are test dependencies, not
simulation-core dependencies. JET is a separate, selected-residual check in CI.
See the [test catalogue](https://github.com/pavaninguva/PopulationBalanceModels.jl/blob/main/test/README.md)
for entry points and the [current Actions results](https://github.com/pavaninguva/PopulationBalanceModels.jl/actions).

## Optional integrations

Create one working environment without adding optional packages to the package's
own Project.toml:

```sh
julia -e 'using Pkg; Pkg.activate("/tmp/pbm-optional"); Pkg.develop(path=pwd()); Pkg.add(["Optim", "NonlinearSolve", "SciMLBase", "OrdinaryDiffEqTsit5", "Distributions", "ForwardDiff", "SpecialFunctions", "AdvancedHMC", "MCMCDiagnosticTools", "Statistics"])'
julia --project=/tmp/pbm-optional test/tensor_diffusion.jl
julia --project=/tmp/pbm-optional test/tensor_integration.jl
julia --project=/tmp/pbm-optional examples/estimate_stationary_crystallizer.jl
```

Optim enables fitting; NonlinearSolve enables stationary roots; SciMLBase plus
a user-selected ODE algorithm enables external integration. AdvancedHMC and
MCMCDiagnosticTools enable optional NUTS. The fifth extension provides continuous
ODE sensitivities through SciMLBase, SciMLSensitivity and ReverseDiff. Its
[separate preparation/test recipe](https://github.com/pavaninguva/PopulationBalanceModels.jl/blob/main/test/README.md)
retains normal precompilation and resolved dependency versions. Older pinned
Enzyme stacks have recorded loading failures; they are not numerical test passes.

## Examples and figures

Each case page gives a short calculation and full-source links. Including an
example module defines its builders; executing the script invokes its driver.
Download supporting files with their directories intact, or use a checkout.
For example, `examples/verification/compact_growth.jl` includes
`../compact_growth.jl`; these identically named files occupy different directories.

For figures, use the pinned Python requirements and a working LaTeX/dvipng
installation. Plot-only commands read retained numerical exports:

```sh
python -m pip install -r paper/requirements.txt
python docs/benchmarks/plot_case_studies.py moments --usetex
python docs/benchmarks/plot_moment_comparison.py --usetex
```

The same scripts without `--usetex` produce preview typography. Replotting does
not recompute trajectories, timings, priors or posterior samples. Preserve
original producer manifests and numerical exports; record visual-only changes
separately. A new forward solve or refit needs its own configuration and provenance.

## Retained results and present-source equivalence

These are distinct levels of evidence:

1. **Integrity:** the stored files match their producer's hashes.
2. **Replay:** the producing environment reproduces its calculation.
3. **Current equivalence:** a reviewed comparison establishes that later code,
   dependencies and numerical settings leave the relevant target/results valid.

The inexpensive checks are:

```sh
python test/protein_results.py
python test/protein_publication.py
python test/plotting_summaries.py
python test/paper_data.py --case 1 --retained
python test/paper_data.py --case 2 --retained
python test/paper_data.py --case 3 --retained
python test/stem_cell_results.py
```

`--retained` checks numerical data contracts without asserting current-source
equivalence. Protein separately validates reviewed compatibility receipts while
preserving original producers. The paper precipitation check requires all
independent logarithmic-size/axial runs and the same-reference-mesh
`time_refinement` dataset; a partial campaign is rejected. The stem-cell check rejects changed
Project weak-dependency/extension declarations before numerical assertions, although its eight
numerical-file hashes match. Preserve failure reports: editing prose or
redrawing figures cannot complete numerical or source-equivalence verification.

The [moment comparison](case_studies/moment_crystallization.md) explains the
remaining collision-EQMOM limitation and links the supporting numerical checks.
