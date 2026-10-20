# Batch population balance

## Physical question and equations

How does a seeded batch change when particles grow, new particles appear at a
specified size, and existing particles are removed? The model is

```math
\frac{\partial n}{\partial t}+\frac{\partial (Gn)}{\partial x}=J\delta(x-0.2)-kn,
\quad 0\le x\le10,
```

with ``G=0.3``, ``J=0.05``, ``k=0.02``, and an initial Gaussian-shaped density
centered at 2. These are illustrative values in a consistent chosen size/time
system, not fitted material constants. The left boundary is closed and the
right boundary permits outward flux.

## Implementation and numerical choices

Run `examples/batch_population_balance.jl`.
The script builds 100 volume bins, a `PBMModel` with Growth, Nucleation, and
Death, and an upwind operator. It converts initial density to bin numbers by
multiplying widths and solves over two time units with SSPRK33 and `dt=0.01`.
Point nucleation uses fixed-pivot deposition, not a boundary influx.

## Results and interpretation

The script prints final number, first moment, and an initial step suggestion.
Number balances creation against removal and any right-boundary escape.
The first moment also responds to growth and to nuclei entering at nonzero size.
A useful continuum comparison before boundary loss is
``\frac{dM_0}{dt}=J-kM_0`` and ``\frac{dM_1}{dt}=GM_0+0.2J-kM_1``.
Compare refinement of these quantities before interpreting a density plot.
Upwind broadens fronts; the `examples/periodic_weno_transport.jl`
uses an analytical bin-average reference to study transport accuracy separately.

## Numerical output

On Julia 1.10.3, the script returned final number 0.7792083787 and first moment
1.8198902925. These are numerical results for the stated illustrative inputs.

## Executable source

This listing is read directly from the runnable script during the documentation build.

```@eval
using Markdown, PopulationBalanceModels
script = "examples/batch_population_balance.jl"
root = pkgdir(PopulationBalanceModels)
code = read(joinpath(root, script), String)
slug = get(ENV, "GITHUB_REPOSITORY", "")
link = isempty(slug) ? "" : "[Source on GitHub](https://github.com/$slug/blob/main/$script)\n\n"
Markdown.parse(link * "```julia\n" * code * "\n```")
```
