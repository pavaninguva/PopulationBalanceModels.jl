# PopulationBalanceModels.jl

Population balance models in Julia: define particle mechanisms, compile sectional
or moment operators, couple populations to bulk balances, and map solutions to observations.

[Documentation](https://pavaninguva.github.io/PopulationBalanceModels.jl/dev/)

From this source directory (Julia 1.10 or newer):

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); using PopulationBalanceModels; Pkg.test()'
julia --project=. examples/batch_population_balance.jl
```

The [manual source](docs/src/index.md) explains model definition, numerical methods,
coupling, inference, and runnable case studies. Start with the
[complete first model](docs/src/getting_started.md). Build the HTML manual with:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The [example catalogue](examples/README.md) separates short usage examples,
case-study drivers and numerical verification. See [verification commands](test/README.md)
and the separate [paper workflow](paper/README.md) for tests and manuscript assets.

[Parameter estimation](docs/src/inference.md) uses one model for MAP, optional
Laplace uncertainty and optional MCMC. The [growth-kinetics example](docs/src/case_studies/estimating_growth.md)
fits analytical density curves with Gaussian noise and compares two prior choices.
Optim enables fitting; AdvancedHMC and MCMCDiagnosticTools enable sampling.

[Trajectory transport methods](docs/src/methods/characteristics.md) implement
eligible 1D and separable 2D affine growth with matched travel-time meshes and
CFL=1 steps, checked during compilation and simulation.

Sectional collisions use fixed pivot. Moment closures and high-order transport
have method-specific restrictions; see the [numerical manual](docs/src/numerical_methods.md).
The [growth-scheme comparison](docs/src/case_studies/constant_growth.md) separates
spatial and temporal errors. [Axial reactor transport](docs/src/axial_reactors.md)
adds prescribed advection and dispersion to one- or two-coordinate populations.
The cooling benchmark uses explicitly chosen initial conditions and a refined
sectional reference; the nanoparticle model requires explicit material closures.
Their case-study pages describe these limits. CFD coupling is deferred.

See [contributing](CONTRIBUTING.md), [changes](CHANGELOG.md), and the [MIT license](LICENSE).
