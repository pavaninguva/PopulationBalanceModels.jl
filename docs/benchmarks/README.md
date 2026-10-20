# Running the three case studies

From the package root, use these calculation/plot pairs:

```sh
julia --project=. examples/cooling_crystallization_benchmark.jl /path/to/gunawan_inputs.jl
python docs/benchmarks/plot_case_studies.py cooling

julia --project=/path/to/working_environment examples/shin_lnp_reproduction.jl /path/to/shin_inputs.jl
python docs/benchmarks/plot_case_studies.py lnp

julia --project=. examples/inguva_braatz_case2.jl
python docs/benchmarks/plot_case_studies.py transport
```

NumPy and Matplotlib are plotting dependencies only. `--data-dir PATH` selects
an alternative exported-data directory. The plotter requires the appropriate
CSV files; it does not fabricate missing curves or substitute demonstration data.

The Gunawan input file returns `initial_density`, `c0`, and `provenance`; it can
also set `grids`, `dt`, and `initial_moments`. The literal Eq.34 polynomial is
retained but rejected as negative on its support. Resolve that discrepancy before
claiming published-figure reproduction. The moderate published spacing is 0.5 µm
(3200 bins) and the fine reference spacing is 0.05 µm (32000 bins), both with a
0.01-second time step. Current comparisons use the package's Upwind/MUSCL,
not the paper's HR1/Lax–Wendroff algorithms.

The Shin input file returns `inputs`, `solver`, and `provenance`, optionally
`saveat`. `inputs` includes the grid and numerical material closures; the Figure 4
adapter supplies zero initial particles, the 10 kg/m³ organic feed, FRR 3,
instantaneous mixing, and the 4320-minute horizon by default. Use an adaptive stiff
solver for that long horizon. The output schedule includes 1, 40, 100, 170, and
4320 minutes. Experimental data are not needed to render the model-only panels.

Inguva–Braatz Case 2 is self-contained: it supplies the separable affine growth
laws, Gaussian initial condition, analytical solution, and a case-specific
travel-time transform for exact bin shifts. Uniform FV and nonuniform mapped
mesh differences are recorded in `errors.csv`.

New study results are written under `docs/src/assets/gunawan_example2/`,
`shin_case/`, and `inguva_case2/`. Rebuild the manual after rendering figures.
The docs build does not run these expensive simulations. When results have not
been generated, the page gives the commands rather than displaying substitute
figures.

`diagnostics/` retains earlier constant-growth and demonstration-initialization
results outside the published manual. The constant-growth Gaussian is now a
regression fixture in `test/fixtures/`. The original periodic rectangle/smooth
benchmark and compact/ADI stability audit remain numerical-method diagnostics,
not additional case-study pages.
