# Lipid nanoparticle formation: CEJ recipe

With D_c defined as a positive loss magnitude, the physical bookkeeping is

```math
\partial_t n+\partial_L(Gn)
=B_n+B_c-D_c.
```

The recipe implements the following components in diameter/length coordinates:

| Mechanism/closure | Implementation |
|---|---|
| Solvent history | Prescribed aqueous fraction w(t); no CFD |
| Solubility | User-supplied C_infinity(w,T), with a separate log-linear mole-solubility helper |
| CNT creation | Critical diameter and nucleation rate from Eqs. 7-8 |
| Creation-size spread | Gaussian centered at the current critical diameter; width and normalization explicit |
| Growth/dissolution | Signed G=k_g L C_infinity (S-S_star) |
| Coalescence | Brownian kernel times a supplied attachment law and fitted PEG factor |
| Material balance | Shared free-lipid state coupled to the actual particle-volume update |
| Dilution | Same volumetric dilution rate applied to particles and dissolved lipid |
| Measurements | L^6-weighted normalized histogram and final-paper Eq. 3, M7/M6 |

The implemented CNT formulas are

```math
L_c=\frac{4\gamma V_m}{k_BT\ln S},\qquad
B_0=A_n\exp\left[-\frac{16\pi\gamma^3V_m^2}
{3(k_BT)^3(\ln S)^2}\right],
```

with B0=0 for S<=1. V_m is PER-MOLECULE volume, consistent with k_B.
A molar volume would require the matching thermodynamic convention.

The collision kernel is

```math
\beta(L,\lambda)=
k_{\rm PEG}\alpha_{\rm base}(L,\lambda,w,T)
\frac{2k_BT(L+\lambda)^2}{3\mu L\lambda}.
```

The attachment_without_peg callback returns alpha_base; it must not already
include the PEG factor. Parameters k_g, A_n and k_PEG remain available for
native fitting through the problem parameter tuple.

The final paper reports A_n=2.0e22 m^-3 s^-1,
k_g=11.88 m^3 kg^-1 s^-1 and k_PEG=6.5e-6, and uses an interfacial energy
near 0.010 N/m. Reusing those values under a changed source normalization
or concentration basis requires verification, not a mechanical substitution.

### Diameter-space coalescence

An event produces a particle with

```math
L_{\rm product}=(L_1^3+L_2^3)^{1/3}.
```

Use AdditiveCoordinate(L->L^3,cbrt) for fixed-pivot interpolation.
This keeps L as the transport/measurement coordinate while conserving M3.
Depositing a product at L1+L2 would conserve the wrong physical quantity.

The equivalent number-density transformation to q=L^3 is

```math
n_q(q)=n_L(L)/(3L^2).
```

Changing coordinates without the Jacobian changes the physical distribution.
For a physical volume v=(pi/6)L^3, the corresponding Jacobian is different.

### Mixing and free-lipid balance

For the delayed-mixing case,

```math
w(t)=\frac{\mathrm{FRR}}{\mathrm{FRR}+1}(1-e^{-t/\tau}).
```

An explicit material-element interpretation gives V/Vorganic=1/(1-w).
The dilution hazard is lambda=w_dot/(1-w). With particle concentrations and
free lipid both defined per current mixture volume, the recipe applies

```math
\dot c=-\lambda c
-\rho_p\frac{\pi}{6}
\left(\dot M_3\big|_{\rm growth}+\dot M_3\big|_{\rm nucleation}\right).
```

There is also a -lambda*n term in the PBE. Coalescence does not consume
dissolved lipid. Absent truncation/other losses, the conserved material is

```math
V(t)\left[c(t)+\rho_p\frac{\pi}{6}M_3(t)\right].
```

This balance is an explicitly declared completion of the model. The main
paper discusses lipid consumption but does not print a complete free-lipid ODE.

For tau=0, use the already mixed initial concentration
Corganic/(FRR+1). For tau>0, the model starts with Corganic before dilution.
The delayed recipe currently begins at t=0; later-start initialization needs
a custom operating history/state.

### Required inputs and conventions

| Input | Why it stays explicit |
|---|---|
| Initial distribution and diameter limits | The paper mentions 200 logarithmic bins but does not give a complete grid/initial-state specification |
| Initial dissolved concentration/basis | Must agree with instantaneous versus delayed mixing |
| Temperature and viscosity law | Required by CNT/Brownian kernels; process conditions must be confirmed |
| Molecular volume and particle density | Needed to translate thermodynamics and particle volume into mass |
| C_infinity(w,T) | Log-linear mole solubility must be converted to the concentration units used for growth |
| S_star(L,w,T) | A size-dependent equilibrium closure is referenced but not fully specified |
| Nucleation width | The size-spread width must be distinct from interfacial energy |
| Gaussian normalization | Total number rate and unnormalized density amplitude differ dimensionally |
| Attachment law | Resolve the energy/separation/stability-ratio convention against author implementation |

A conventional Kelvin-ratio helper is provided as an OPTIONAL closure. It is
not automatically asserted to be the author's exact S_star implementation.

The inherited recipe notes describe the two-page supplement specifies DLVO energy expressions,
A_H=1.7e-22 J and a selected surface potential of 10 mV. Its screening formula
gives an inverse length although kappa is described as a Debye length.
Ion concentration units, separation coordinate and electrostatic prefactors
need a consistent interpretation. The recipe accepts a confirmed attachment
function or table rather than silently repairing the printed implementation.

Additional printed conventions to reconcile:

- Eq. 1 adds D_c while Eq. 11 defines a positive integral. The recipe uses
  parent removal, B_c-D_c.
- Eq. 6 prints a Gaussian without its density-normalization prefactor, and
  reuses sigma notation later used for interfacial energy.
- Eq. 14 prints an energy-barrier integral where attachment versus stability
  ratio, reciprocal and distance conventions need confirmation.
- Eq. 10's kernel arguments need to agree with the length arguments in Eq. 12.

These points are reproduction questions, not evidence about what the author's
actual code computes.

### Observations and calibration

For the final-paper definition,

```math
n_{\rm intensity}(L)=L^6 n(L),\qquad
d_{\rm paper}=M_7/M_6.
```

paper_intensity_mean uses that exact ratio under an explicit name. The
normalized histogram is a separate observable, with its measurement window
and weighting specified. Do not equate different instrument averages without
checking their definitions.

Joint PSD, free-lipid and number/moment measurements can share fitted kinetic
parameters through CalibrationProblem. The article's late-time observations
provide limited information about the absolute millisecond kinetic scale.
Testing multistart, profile likelihood and numerical refinement is part of
the planned inference validation.



## Run the model recipe

Run `examples/lipid_nanoparticle_demo.jl`
using the reusable `examples/papers/cej2025_lnp.jl`.
The demonstration starts with free lipid 10 kg/m³, FRR 3, mixing time 0.01 s,
and 60 logarithmic diameter bins from 1 nm to 2 micrometers. SSPRK33 uses a
20-microsecond step to 0.02 s. Temperature, viscosity, solubility, equilibrium
ratio, attachment, molecular volume, density, and nucleation width are explicitly
supplied demonstration choices. The growth constant also differs from the
paper's reported value. Overflow is discarded explicitly.

The script reports particle concentration, the defined intensity mean in nm,
and free lipid. These outputs demonstrate executable coupling and observation
paths. They are not validated predictions or reproductions of published figures.
A derivative-level test checks the declared material-element dilution balance;
full experimental comparison needs confirmed inputs, conventions, and refinement.
Late-time normalized PSDs alone may poorly identify an absolute fast kinetic scale.

Reference: Shin et al. (2025), *Chemical Engineering Journal* 523, 167786,
[doi:10.1016/j.cej.2025.167786](https://doi.org/10.1016/j.cej.2025.167786).
The original supplementary source is identified in the recipe comments;
its unresolved conventions remain explicit above.

## Numerical output

With the stated synthetic closures on Julia 1.10.3, the script returned
particle concentration 2.80014e18 m⁻³, intensity mean 3.24097 nm, and free lipid
3.51500 kg/m³. These magnitudes belong to the demonstration recipe only.

## Executable source

This listing is read directly from the runnable script during the documentation build.

```@eval
using Markdown, PopulationBalanceModels
script = "examples/lipid_nanoparticle_demo.jl"
root = pkgdir(PopulationBalanceModels)
code = read(joinpath(root, script), String)
slug = get(ENV, "GITHUB_REPOSITORY", "")
link = isempty(slug) ? "" : "[Source on GitHub](https://github.com/$slug/blob/main/$script)\n\n"
Markdown.parse(link * "```julia\n" * code * "\n```")
```
