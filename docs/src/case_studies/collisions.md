# Aggregation and breakage

## Physical question and event equations

Aggregation merges particles and breakage creates daughters. For additive size,
their moment rates are

```math
\frac{dM_k}{dt}\big|_A=\tfrac12\iint[(x+y)^k-x^k-y^k]K(x,y)n(x)n(y)\,dx\,dy,
```

```math
\frac{dM_k}{dt}\big|_F=\int a(x)n(x)\left[\sum_j\nu_jx_j^k-x^k\right]dx.
```

These conserve ``M_1`` when daughter properties sum to the parent's and products
remain represented. Number is not conserved: each binary aggregation event
reduces it by one, while equal binary fragmentation increases it by one.

## Implementation and numerical choices

Run `examples/aggregation_and_breakage.jl`.
A volume grid has pivots 1 through 40. Initially pivot 4 contains unit number
concentration. The constant aggregation kernel is 0.03 and particles of size at
least 2 split with hazard 0.02 into two daughters of size ``x/2``. Both mechanisms
use fixed pivot and explicitly discard unresolved products. SSPRK33 uses
`dt=0.01` over one time unit.

## Results and interpretation

The script prints initial/final resolved first moments and instantaneous
unresolved birth rates. Any first-moment loss must be interpreted with those
truncation rates; they are not cumulative loss totals. Increase the size range
or integrate a reservoir before claiming a closed material balance.
The scientific tests check identical-pair factors, daughter multiplicity, and
nonuniform-grid conservation. They do not imply a general positivity guarantee
for arbitrary kernels and time steps. The [numerical manual](../numerical_methods.md)
derives deposition weights and diameter-space volume conservation.

## Numerical output

On Julia 1.10.3, the resolved first moment changed from 4 to
3.9999999999999787. The final unresolved property birth rate was about
2.18e-16, negligible for this particular initial state and horizon.

## Executable source

This listing is read directly from the runnable script during the documentation build.

```@eval
using Markdown, PopulationBalanceModels
script = "examples/aggregation_and_breakage.jl"
root = pkgdir(PopulationBalanceModels)
code = read(joinpath(root, script), String)
slug = get(ENV, "GITHUB_REPOSITORY", "")
link = isempty(slug) ? "" : "[Source on GitHub](https://github.com/$slug/blob/main/$script)\n\n"
Markdown.parse(link * "```julia\n" * code * "\n```")
```
