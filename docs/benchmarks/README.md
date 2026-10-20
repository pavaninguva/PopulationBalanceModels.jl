# Reproducing the numerical figures

Run the Julia calculations from the package root before plotting:

```sh
julia --project=. examples/inguva_braatz_case1.jl
julia --project=. examples/audit_compact_adi.jl
julia --project=. examples/cooling_crystallization_benchmark.jl
julia --project=. examples/lipid_nanoparticle_demo.jl
python docs/benchmarks/plot_case_studies.py
```

The plot script requires NumPy and Matplotlib. Neither is a Julia package or
Documenter dependency. Pass `transport`, `cooling`, or `lnp` to render an
individual case. CSV arrays and input/environment metadata accompany figures
in `docs/src/assets/`; the documentation build uses these checked-in results
and does not launch expensive benchmarks.

`inguva_case1/` follows Section 3.1, equations 55–56, and the t=1 Figure 2
problem in Inguva and Braatz (2023). It uses finite-volume bin averages for all
methods. The published nodal Upwind/Exact results and SharpClaw WENO settings
are not identical implementations. Each method's boundary treatment is recorded.

`compact_audit/` checks the implemented periodic variant, rather than a literal
reproduction of the HOC–ADI paper. It records ten-period L2/number changes and
negative lobes for smooth and rectangular initial states.

`cooling_benchmark/` follows the equations and 1000-second horizon of Gunawan
Example 2. The seed and initial concentration are explicit demonstration
assumptions. The comparison is against independently integrated continuum
moment equations, not against digitized published CSDs.

`lnp_demo/` uses synthetic inputs. It demonstrates integrated nucleation and
coupling/observation exports; it does not reproduce a Shin paper figure.
`examples/shin_lnp_reproduction.jl` accepts explicit input configurations for
future comparison to Figure 4, without filling missing paper inputs silently.

Primary papers:

- https://web.mit.edu/braatzgroup/Inguva_ComputChemEng_2023.pdf
- https://web.mit.edu/braatzgroup/80_High_resolution_algorithms_for_multidimensional_population_balance_equations.pdf
- https://web.mit.edu/braatzgroup/Shin_ChemEngJ_2025.pdf

The earlier `results_2d.csv`, cut arrays, and `benchmark_2d` figure at the asset
root are a separate historical periodic diagnostic. `plot_2d.py` renders those
records; they are not the Inguva paper benchmark.
