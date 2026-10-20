# Interacting populations and shared solute

## Physical question and balances

A precursor population nucleates from a shared solute pool and transfers to a
product population. Can their numerical representations differ while preserving
material? With source size 2, ``J=kc``, and transfer hazard ``r``, the continuum
bookkeeping is

```math
\frac{\partial n_A}{\partial t}=kc\,\delta(x-2)-rn_A,\qquad
\frac{\partial n_B}{\partial t}=rn_A,\qquad \frac{dc}{dt}=-2kc.
```

The conserved quantity is ``c+M_{1,A}+M_{1,B}``, assuming common additive size
units and no truncation. Solute concentration is expressed in that same
particle-property concentration basis.

## Implementation and numerical choices

Run `examples/interacting_populations.jl`.
It compiles a uniform precursor grid and a nonuniform product grid, then adds
`PopulationTransfer` and `ConservativeExchange` to `PBMSystem`. Initially both
populations are empty and ``c=5``; ``k=0.1`` and ``r=0.3``. SSPRK33 advances all
states with `dt=0.01` to time 3. Pivot deposition transfers particles across the
two grids; the exchange uses the actual selected nucleation residual.

## Results and interpretation

The script reports free solute, both population numbers, and total material.
The total should remain close to 5 within floating-point/integration error.
Transfer changes population identity without consuming solute a second time.
Check number and property conservation separately if adding fragmentation,
a destination-size map, or a nonunit transfer multiplicity. The
[coupling manual](../coupled_systems.md) supplies a complete executed model and
explains unlike-population collisions and moment-based bulk exchange.

## Numerical output

On Julia 1.10.3, final solute was 2.7440581799, precursor number 0.7112098833,
and product number 0.4167610268; total material was 4.999999999999909.

## Executable source

This listing is read directly from the runnable script during the documentation build.

```@eval
using Markdown, PopulationBalanceModels
script = "examples/interacting_populations.jl"
root = pkgdir(PopulationBalanceModels)
code = read(joinpath(root, script), String)
slug = get(ENV, "GITHUB_REPOSITORY", "")
link = isempty(slug) ? "" : "[Source on GitHub](https://github.com/$slug/blob/main/$script)\n\n"
Markdown.parse(link * "```julia\n" * code * "\n```")
```
