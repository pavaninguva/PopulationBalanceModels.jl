# PopulationBalanceModels.jl

Population balance models in Julia: define particle mechanisms, compile sectional
or moment operators, couple populations to bulk balances, and map solutions to observations.

From this source directory (Julia 1.10 or newer):

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); using PopulationBalanceModels; Pkg.test()'
julia --project=. examples/batch_population_balance.jl
```

The [manual](docs/src/index.md) explains model definition, numerical methods,
coupling, inference, and runnable case studies. Start with the
[complete first model](docs/src/getting_started.md). Build the HTML manual with:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Sectional collisions use fixed pivot. Moment closures and high-order transport
have method-specific restrictions; see the [numerical manual](docs/src/numerical_methods.md).
The Gunawan and CEJ examples explicitly use demonstration inputs and do not
establish published-figure reproduction. CFD coupling is deferred.

See [contributing](CONTRIBUTING.md), [changes](CHANGELOG.md), and the [MIT license](LICENSE).
No registry availability is asserted.
