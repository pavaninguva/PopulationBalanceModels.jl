# Paper figures

Each case is one directory. `run.jl` calculates the results; `plot.py` draws the
figures from saved results. Settings are in `config.toml`, numerical exports in
`data/`, and finished figures in `figures/`.

| Directory | Case | Saved results |
|:--|:--|:--|
| [growth/](growth/) | Growth and time schemes | Available |
| [moments/](moments/) | Aggregation–fragmentation | Available |
| [precipitation/](precipitation/) | Tubular precipitation | Partial: reference time refinement is missing |
| [protein/](protein/) | CNT versus CNT + GRD | MAPs, MCMC draws and latent predictions available |

Run from the repository root. Set up the Julia checkout and Python 3.12 plotting
environment once; final figures require LaTeX:

```sh
julia --project=paper -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
python -m pip install -r paper/requirements.txt
```

For example:

```sh
julia --project=paper paper/growth/run.jl
python paper/growth/plot.py
```

Use `moments` or `precipitation` in the same commands. Plotting validates the
saved data and preserves its original producer identity; it never reruns a
simulation or treats old timings as newly measured results. The incomplete
precipitation dataset is rejected, rather than replaced with documentation data.

For protein, redraw the retained results without fitting or sampling:

```sh
python paper/protein/plot.py
```

A deliberate full recalculation uses `julia --project=paper paper/protein/run.jl --full`;
it includes both expensive MCMC runs. `--map` fits only the MAPs. The same plot
script accepts `--data-only`, `--gauss-newton` or `--laplace` for those figures;
the corresponding Julia `--gauss-newton`/`--laplace` modes calculate uncertainty
separately. The [manual](../docs/src/case_studies/protein_crystallization.md)
explains these analyses and their limits.

After a reviewed package change, `julia --project=paper paper/protein/run.jl
--compatibility-only` checks the retained posterior values and gradients without
refitting or sampling. Original manifests and analysis receipts are preserved;
reuse requires matching observations, settings, priors and resolved dependencies.

[architecture/](architecture/README.md) is the independent schematic generator.
`_support/` contains implementation helpers, not additional entry points.
Tests and historical runs live under `test/`, not here. Current protein samples
are in `protein/data/{cnt,cnt_grd}/`; original manifests remain alongside them.
