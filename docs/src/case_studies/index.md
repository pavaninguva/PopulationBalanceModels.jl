# Case studies

These calculations show how a physical population balance becomes a working
model. Each develops the equations alongside the rate callbacks, initial state,
boundaries, and numerical choices used in its implementation.

- [Batch cooling crystallization](crystallization.md): seeds grow and new crystals
  nucleate as cooling changes solubility. The population consumes dissolved
  solute, so the concentration balance is solved with the distribution.
- [Crystal growth under a temperature program](moment_crystallization.md): native
  moment equations and a sectional calculation predict number, material, and mean
  sizes; a review of temperature cycling explains where extinction needs more information.
- [Lipid nanoparticle formation](nanoparticles.md): an initially particle-free
  solution nucleates, grows, and coalesces after solvent mixing. Scattering
  observations weight the resulting distribution strongly toward larger particles.
- [Growth in two particle dimensions](transport.md): coordinate-dependent growth
  moves and stretches a joint distribution. An analytical solution exposes
  numerical broadening, oscillations, and errors under grid refinement.
- [Estimating size-dependent growth](estimating_growth.md): analytical density curves
  with Gaussian noise illustrate fitting two growth rates, comparing prior choices
  and checking MCMC chains.

The two-dimensional calculation is self-contained. The crystallization seed has
an unresolved discrepancy in the printed coefficients; the nanoparticle
calculation requires numerical material closures. Those input issues are
described where the initial state and rates are introduced. Their runners and
plotting commands are provided without substituting invented inputs.
