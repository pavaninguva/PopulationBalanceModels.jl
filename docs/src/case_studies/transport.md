# Two-coordinate transport

## Physical question and implementation

A joint distribution over two internal properties can advect at different speeds
in each direction. This case isolates transport from nucleation and process
coupling so analytical bin averages can distinguish numerical error from model
uncertainty. Run `examples/two_coordinate_transport.jl`.
It compares a smooth profile and a rectangle on 32, 64, and 128 bins per axis.

The general `Grid2D` residual supports growth, midpoint birth/removal,
bilinear point deposition, and midpoint distributed creation. Boundary influx
has units of number flux per tangential internal coordinate and is integrated
over the face width. The per-term 2D GaussianInjection regularizes a point
using tensor bin fractions; the `GaussianNucleation` physical constructor is 1D.
2D collisions, general integrated-source terms, and coupled 2D system blocks
are unsupported. These restrictions are separate from the narrower eligibility
of the direct methods below.

## Compact/ADI equations and paper fidelity

Li et al., Particuology 110 (2026), 310–319,
[DOI 10.1016/j.partic.2026.01.024](https://doi.org/10.1016/j.partic.2026.01.024),
is the HOC–ADI reference paper. The inherited paper assessment records unresolved issues that prevent claiming a faithful reproduction:

- Eq. 1 uses separate g1 and g2, but Eqs. 6–15 contain g1*g2 multiplying both
  directional derivatives. Used literally as the leading transport coefficient,
  that does not recover Eq. 1 for independently chosen velocities.
- The compact coefficients a1–a5 and important derivation details are left to
  supplementary data. Those supplementary coefficients are not available to this implementation. The printed time levels/intermediate updates are also ambiguous.
- In the stated Case 1, g1=g2=0.2, the domain is [0,2]² and t=100. Under
  zero incoming flux the Gaussian center would move from (0.4,0.4) to
  (20.4,20.4), outside the domain. Units, horizon or boundary conventions need
  clarification before reproducing its reported nonzero final distributions.

These are reproducibility observations about the supplied equations, not a
claim that no corrected author implementation exists. A literal paper backend
remains pending supplementary equations or author code.

`HOCADI(time_order=4)` instead implements the following fully specified,
independently tested **compact/ADI variant** for homogeneous constant growth:

```math
\tfrac14 d_{i-1}+d_i+\tfrac14 d_{i+1}
=\tfrac{3}{4h}(f_{i+1}-f_{i-1}),\qquad Md=Df.
```

For each direction with speed g, the Cayley update is

```math
(M+\tfrac{g\Delta t}{2}D)f^{new}
=(M-\tfrac{g\Delta t}{2}D)f^{old}.
```

Taking one x solve and one y solve defines S₂(Δt). The tensor operators
commute in this restricted problem. `time_order=2` uses S₂ directly.
`time_order=4` uses

```math
S_4(\Delta t)=S_2(\gamma\Delta t)
S_2((1-2\gamma)\Delta t)S_2(\gamma\Delta t),\quad
\gamma=(2-2^{1/3})^{-1}.
```

This fourth-order composition has a negative middle substep. It is restricted
to reversible, homogeneous autonomous transport here. No birth, death,
aggregation, breakage or bulk coupling is silently run backward. Periodic
uniform axes are required; one-sided physical compact closures remain work.
The spatial derivative is fourth order, and independent tests isolate second-
and fourth-order temporal convergence against the exact semidiscrete Fourier
solution. Cyclic tridiagonal LU plus rank-two correction gives O(nx*ny) work
per directional sweep. Linear L2 stability does not establish positivity.

## The 2D comparison

Run `julia --project=. examples/two_coordinate_transport.jl`. Results and environment
metadata are under `docs/src/assets/`; the script has no plotting dependency.

The shared PDE is

```math
\partial_t n+0.5\partial_xn+0.25\partial_yn=0,
\quad (x,y)\in[0,1]\times[0,0.5],\quad t_f=0.25,
```

with periodic diagnostic boundaries. Smooth separable Fourier densities have
analytical **bin averages**. A rectangular pulse separately tests discontinuous
fronts. All methods use identical bin-integrated initial numbers, grids,
output times and analytical references. Uniform averages themselves satisfy
the constant-advection PDE, allowing the compact derivative and exact shifts
to evolve the same representation without a nodal/average comparison error.

| Method | Spatial/evolution treatment | Eligibility |
|---|---|---|
| WENO5 + SSPRK(3,3) | Unsplit conservative FV residual, summed CFL 0.4 | Current general 2D growth/source residual |
| HOCADI4_variant | Compact derivative plus composed directional implicit solves; directional CFL 1 | Constant homogeneous growth, periodic uniform axes |
| InguvaBraatz_constant_shift | Commuting directional cell shifts at CFL 1 | Constant homogeneous growth and integer cell travel at every step/output |

The last row implements the constant-growth class of Inguva and Braatz (2023),
Section 2.1, Eqs. 9–14. For this class it is also exact for bin averages.
The special transformed meshes, separable variable growth and noncommuting
split cases from later sections are not implemented yet. Fractional steps or
incompatible output times raise an error rather than quietly interpolate.
An integer multi-cell shift is equivalent to repeated exact one-cell substeps.

The methods cannot share a stable explicit time step and CFL=1 exact shifts
on this fixed grid. The benchmark therefore declares method-appropriate steps
and reports them. Accuracy-versus-cost curves are more informative than a
fixed-grid speed ranking. Timing uses the minimum of three warmed forward
solves, includes numerical factorization/setup and output copies, excludes
Julia compilation and common model/initial construction. It measures this
reference implementation on the recorded host, not a universal library ranking.

Measured smooth-case results at 128 × 128 bins (Julia 1.10.10, one thread,
AMD EPYC 9V74; minimum of three warmed solves):

| Method | L1 density error | Time | Allocated bytes |
|---|---:|---:|---:|
| WENO5 + SSPRK3 | 6.24e-08 | 172.7 ms | 132,062,896 |
| Compact/ADI fourth-order variant | 4.51e-07 | 35.03 ms | 3,604,880 |
| Inguva–Braatz constant shift | 1.88e-17 | 1.314 ms | 396,208 |

The compact variant is faster at this grid and chosen step, with a larger
error. This is not an accuracy-matched speed claim. Its measured L1 refinement
order from 64 to 128 bins is 3.967. All three conserve total number to roughly
5e-15 relative or better. Exact-shift precision applies to this eligible
homogeneous class, not a general nucleating or coupled PBM.

For diagnostic comparison, positivity checks are disabled and both minimum
density and negative particle number are recorded. Production simulation
defaults to rejecting negative concentration. The compact scheme shows
substantial undershoots on the rectangle; a claim of nonoscillatory behavior
is unsupported. WENO has small undershoots too and still needs a proven
positivity limiter. No state clipping is used.

```julia
solution_weno = simulate(problem;dt=0.4/n)
solution_compact = simulate(problem;method=HOCADI(time_order=4),dt=2/n)
solution_shift = simulate(problem;method=ExactShift(),dt=2/n)
```

Here `problem` is the eligible periodic constant-growth benchmark and n is its
bins per axis. HOCADI and ExactShift are direct evolution methods selected by
simulate. `odeproblem(problem)` exports the compiled residual (FV or moments),
not those direct evolution methods.



## Historical evidence and reproducibility

The figure and raw records below are the inherited benchmark evidence, recorded
with Julia 1.10.10 on the host described in the environment file. Rerunning the
script replaces the working benchmark CSVs and metadata. Times are host-dependent;
compare errors, allocations, chosen steps, and output schedules together.

![Smooth and sharp-front transport comparisons](../assets/benchmark_2d.png)

[Full results CSV](../assets/results_2d.csv),
[environment](../assets/environment.txt),
[smooth cut](../assets/smooth_cut.csv), and
[rectangle cut](../assets/rectangle_cut.csv) preserve the recorded experiment.

At 128² bins the rectangle's minimum density was about ``-0.30990`` for compact/ADI,
``-1.6452\times10^{-7}`` for WENO, and zero for the eligible exact shift. The
compact oscillations and negative number are part of the evidence. Diagnostic
runs disable rejection to measure them; the default solver rejects negative
states beyond tolerance and never clips them.

## Recorded benchmark table

| Profile | Bins per axis | Method | L1 error | Time (ms) | Minimum density | Negative number |
|---|---:|---|---:|---:|---:|---:|
| smooth | 32 | WENO5 + SSPRK3 | 3.6645e-05 | 2.629 | 0.16039 | 0 |
| smooth | 32 | Compact/ADI fourth-order variant | 0.00010301 | 0.54437 | 0.16043 | 0 |
| smooth | 32 | Inguva–Braatz constant shift | 1.3634e-17 | 0.02693 | 0.16043 | 0 |
| smooth | 64 | WENO5 + SSPRK3 | 1.2974e-06 | 20.653 | 0.1602 | 0 |
| smooth | 64 | Compact/ADI fourth-order variant | 7.0518e-06 | 3.7823 | 0.1602 | 0 |
| smooth | 64 | Inguva–Braatz constant shift | 2.6942e-17 | 0.17459 | 0.1602 | 0 |
| smooth | 128 | WENO5 + SSPRK3 | 6.2351e-08 | 172.71 | 0.16003 | 0 |
| smooth | 128 | Compact/ADI fourth-order variant | 4.5106e-07 | 35.028 | 0.16003 | 0 |
| smooth | 128 | Inguva–Braatz constant shift | 1.8819e-17 | 1.3142 | 0.16003 | 0 |
| rectangle | 32 | WENO5 + SSPRK3 | 0.011303 | 2.6433 | -5.8337e-08 | 6.6009e-10 |
| rectangle | 32 | Compact/ADI fourth-order variant | 0.026398 | 0.52662 | -0.24681 | 0.0083505 |
| rectangle | 32 | Inguva–Braatz constant shift | 0 | 0.029313 | 0 | 0 |
| rectangle | 64 | WENO5 + SSPRK3 | 0.0069534 | 21.164 | -1.0993e-07 | 9.7062e-10 |
| rectangle | 64 | Compact/ADI fourth-order variant | 0.018792 | 3.7964 | -0.42451 | 0.0056097 |
| rectangle | 64 | Inguva–Braatz constant shift | 0 | 0.19299 | 0 | 0 |
| rectangle | 128 | WENO5 + SSPRK3 | 0.0041601 | 168.15 | -1.6452e-07 | 6.1039e-10 |
| rectangle | 128 | Compact/ADI fourth-order variant | 0.013659 | 34.928 | -0.3099 | 0.0041315 |
| rectangle | 128 | Inguva–Braatz constant shift | 0 | 1.4564 | 0 | 0 |


## Executable source

This listing is read directly from the runnable script during the documentation build.

```@eval
using Markdown, PopulationBalanceModels
script = "examples/two_coordinate_transport.jl"
root = pkgdir(PopulationBalanceModels)
code = read(joinpath(root, script), String)
slug = get(ENV, "GITHUB_REPOSITORY", "")
link = isempty(slug) ? "" : "[Source on GitHub](https://github.com/$slug/blob/main/$script)\n\n"
Markdown.parse(link * "```julia\n" * code * "\n```")
```
