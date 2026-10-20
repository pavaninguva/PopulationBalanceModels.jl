# Estimating kinetic parameters

## Physical question and analytical data

Can a common first-order removal hazard be recovered from two experiments with
different initial concentrations? For ``\partial_t n=-kn``, total number obeys
``M_0(t)=M_0(0)e^{-kt}``. The example generates data analytically with ``k=0.4``
at times 0, 0.2, 0.6, 1.2, and 2, for initial scales 1 and 2. This independent
formula avoids generating synthetic data with the same numerical solver used
in the fit. Standard deviation 0.02 defines the illustrative likelihood; the
observations contain no random noise.

## Implementation and numerical choices

Run `examples/estimate_kinetic_parameters.jl`
in a working environment containing this package and Optim (see the
[developer guide](../development.md)). The script compiles a Death-only model,
uses `MomentObservable(0)`, constructs two Experiments, and fits the common
parameter through `k=exp(z[1])`. The initial guess is 0.15. The forward reference
solver uses `dt=0.01` and Optim is limited to 100 iterations.

## Results and interpretation

The script prints the fitted physical hazard, optimizer convergence, local
residual-Jacobian rank, and covariance. A fit near 0.4 checks this synthetic
inverse problem and the optional extension. Full local rank distinguishes its
single parameter locally; it does not prove that a multimechanism PBM can
identify all its rates. The covariance is in log-rate coordinates and assumes
the declared Gaussian observation errors. Repeat with refined integration and
realistic noise before interpreting uncertainty. Joint bulk, number, and PSD
measurements can constrain different mechanisms, as explained in
[Observations and estimation](../inference.md).

## Numerical output

On Julia 1.10.3 with Optim 2.3.1, the fitted hazard was 0.3999999989, the
optimizer converged, and the local Jacobian rank was one. The reported
log-rate covariance was approximately 0.000310686.

## Executable source

This listing is read directly from the runnable script during the documentation build.

```@eval
using Markdown, PopulationBalanceModels
script = "examples/estimate_kinetic_parameters.jl"
root = pkgdir(PopulationBalanceModels)
code = read(joinpath(root, script), String)
slug = get(ENV, "GITHUB_REPOSITORY", "")
link = isempty(slug) ? "" : "[Source on GitHub](https://github.com/$slug/blob/main/$script)\n\n"
Markdown.parse(link * "```julia\n" * code * "\n```")
```
