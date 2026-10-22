# Case studies

These calculations show how a physical population balance becomes a working
model. Each develops the equations alongside the rate callbacks, initial state,
boundaries, and numerical choices used in its implementation.

- [Batch cooling crystallization](crystallization.md): seeds grow and new crystals
  nucleate as cooling changes solubility. The population consumes dissolved
  solute, so the concentration balance is solved with the distribution.
- [Comparing growth and time methods](constant_growth.md): a translating
  distribution separates spatial diffusion, temporal error, and dispersive
  undershoots; the analytical solution also tests the CFL=1 cell shift.
- [Aggregation and nucleation with moments](moment_crystallization.md): an
  analytical aggregation–fragmentation balance and a short nucleation pulse test
  QMOM, rank-adaptive QMOM, Gamma EQMOM, DQMOM and lognormal closure, separating
  shape, reconstruction and time-integration errors.
- [Lipid nanoparticle formation](nanoparticles.md): an initially particle-free
  solution nucleates, grows, and coalesces after solvent mixing. Scattering
  observations weight the resulting distribution strongly toward larger particles.
- [Growth in two particle dimensions](transport.md): coordinate-dependent growth
  moves and stretches a joint distribution. An analytical solution exposes
  numerical broadening, oscillations, and errors under grid refinement.
- [Estimating size-dependent growth](estimating_growth.md): analytical density curves
  with Gaussian noise illustrate fitting two growth rates, comparing prior choices
  and checking MCMC chains.

- [Continuous tubular precipitation](continuous_precipitation.md): axial transport
  and dispersion carry both solute and crystals; startup and outlet distributions
  are compared under refinement.
- [Inferring protein crystallization kinetics](protein_crystallization.md):
  graph-read concentration and diameter measurements constrain nucleation and
  growth in an unseeded batch. Joint posterior, prior, chain and predictive plots
  assess the fit under explicit unit, observation and prior assumptions.

The constant-growth and two-dimensional calculations have analytical solutions.
The cooling calculation uses an explicitly chosen nonnegative initial condition
and a finer sectional reference, with an independent moment check. The
nanoparticle reconstruction includes full-horizon runs and mechanism profiles,
with material assumptions and explicit failed domain/convergence checks. These assumptions
are stated beside the relevant inputs. The [axial reactor tutorial](../axial_reactors.md)
adds a physical position coordinate to an intrinsic growth model.

[Dissolving a finite population](dissolution.md) compares sectional and
rank-adaptive moment predictions with an analytical disappearance solution,
including their solute and historical removal balances.

Each page ends with a minimal calculation and links to its full reproduction
driver. Downloads preserve the source layout; a repository checkout is the
easiest way to keep recipes, data and plotting helpers together. Figure-specific
assumptions stay on the case page; detailed correction history is in the
[audit register](../audit.md).
