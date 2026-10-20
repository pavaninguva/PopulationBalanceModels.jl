# Cooling crystallization: Gunawan case 2

For a length-based number distribution f(L,t),

```math
\partial_t f+\partial_L(Gf)=B_0\delta(L),\qquad
G=k_g s^g(1+0.1L),\qquad
B_0=k_b V s^b,
```

where

```math
s=\frac{c-c_{\rm sat}(T)}{c_{\rm sat}(T)},\qquad
V=\int_0^\infty L^3 f(L,t)\,dL,
\qquad
\dot c=-3\rho_c\int_0^\infty G L^2 f\,dL.
```

The recipe implements the supersaturated branch with zero growth/nucleation
when s<=0. A signed undersaturated/dissolution law is not specified by these
fractional-power laws and must be an explicit extension.

The prescribed functions and stated parameters are

```math
c_{\rm sat}(T)=1.721\,10^{-4}T^2-5.88\,10^{-3}T+0.1286,
\quad
T(t)=32-4(1-e^{-t/18600}),
```

| Parameter | Stated value |
|---|---|
| k_b | 4.64e-7 |
| b | 1.78 |
| k_g | 116 |
| g | 1.32 |
| rho_c | 2.11e-12 g/micrometer^3 |

Use the paper's length in micrometers, time in seconds and g/g-water
concentration basis consistently. Geometric shape or reference-basis factors
must agree with its volume/mass definitions; no extra factor pi/6 is inserted
into this crystal model.

### Initialization issue

The inherited paper-recipe notes identify the reported seed law is

```math
f(L,0)=-3.48\,10^{-4}L^2+0.136L-13.3
\quad (180.5\le L\le210.5).
```

At the printed precision, even its maximum is negative:
approximately -0.01264368 at L=195.4023 micrometers. Thus directly using it
gives a negative initial population. Clipping it produces an empty seed and
changes the secondary-nucleation model. The recipe exposes this expression
for checking but requires a separately supplied initial_density.

The case-study description also does not state c(0) explicitly. Starting at
c_sat(T(0)) is a plausible initialization choice, not a verified paper input.
The runnable demonstration labels both its nonnegative compact parabolic seed
and its saturated initial concentration as demonstration assumptions.

### Numerical representation

Implement the delta source as an inward NUMBER FLUX at L=0:

```math
(Gf)(0,t)=B_0(t).
```

This avoids dividing by G(0,t), which may vanish at saturation. Do not also add
the same nuclei as a distributed source.

ConservativeExchange computes solute depletion from the exact selected
discrete distribution update and the fixed crystal quantity rho_c L^3. This
closes the discrete material balance. It converges toward the continuum
growth integral as the grid is refined; it is not an identical quadrature
formula at finite grid size. A zero-size birth represented in the first
finite-volume cell has a small positive pivot-volume contribution.

The current numerical scheme is selectable Upwind/MUSCL/uniform WENO.
The 2004 HR1 algorithm is not claimed to be implemented.

### Independent moment reference

Because G=h(t)(1+0.1L), its continuum moment equations close:

```math
\dot\mu_0=k_b\mu_3s^b,\qquad
\dot\mu_k=k\,h(t)(\mu_{k-1}+0.1\mu_k),\ k=1,2,3,\qquad
\dot c=-\rho_c\dot\mu_3,
\quad h=k_gs^g.
```

The recipe supplies moment_rhs! for [mu0,mu1,mu2,mu3,c].
It also supplies build_moments(initial_moments=...,c0=...), using native
MomentSet/ExactClosure, PolynomialRate, zero-size Nucleation and MomentExchange.
The optional integration suite compares the native solver state against the
independently coded moment_rhs! under the same initial data and cooling.
Compare the distribution's moments, concentration, material balance and
refinement behavior against this reference.

The included 50 s demonstration comparison is a numerical consistency test.
It does not reproduce the 1000 s published benchmark or use a verified paper seed.



## Run and interpret the demonstration

Run `examples/cooling_crystallization_demo.jl`.
The reusable `examples/papers/gunawan2004_case2.jl`
contains the laws, native moment model, and independent moment reference.
The script uses 800 length bins over 0–1600 micrometers and SSPRK33 with
`dt=0.02` for 50 seconds. It prints final dissolved concentration and discrete
material drift. A small drift checks the discrete exchange, not the printed
seed polynomial or published trajectories. The demonstration's compact
parabolic seed and saturated initial concentration are explicit assumptions.

Historical refinement against an independent moment ODE gave relative final
``M_3`` errors of ``1.6211\times10^{-4}`` for 400 bins and
``5.9056\times10^{-5}`` for 800 bins. These inherited results used the illustrative
initialization and a 50-second horizon, not the 1000-second publication case.
Do not treat them as measurements from a new execution.

Reference: Gunawan, Fusman, and Braatz (2004), *AIChE Journal* 50, 2738–2749,
[doi:10.1002/aic.10228](https://doi.org/10.1002/aic.10228).

## Numerical output

The Julia 1.10.3 demonstration returned final dissolved concentration
0.1166695816 and discrete material drift -1.8541e-14. The optional integration
suite also reran the 400/800-bin comparisons, agreeing with the historical
relative errors above to their reported precision.

## Executable source

This listing is read directly from the runnable script during the documentation build.

```@eval
using Markdown, PopulationBalanceModels
script = "examples/cooling_crystallization_demo.jl"
root = pkgdir(PopulationBalanceModels)
code = read(joinpath(root, script), String)
slug = get(ENV, "GITHUB_REPOSITORY", "")
link = isempty(slug) ? "" : "[Source on GitHub](https://github.com/$slug/blob/main/$script)\n\n"
Markdown.parse(link * "```julia\n" * code * "\n```")
```
