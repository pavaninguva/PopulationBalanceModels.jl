# Case studies

These calculations show how a physical population balance becomes a working
model. Each develops the equations alongside the rate callbacks, initial state,
boundaries, and numerical choices used in its implementation.

- [Uncertainty in continuous milling](milling_uncertainty.md): compare the
  deterministic startup with Barrasso's Figure 7, then propagate four uncertain
  breakage/screen parameters by MC and PCE. The partial forward reproduction
  and UQ use the same numerical configuration, with their respective limits.

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
- [Growth in two particle dimensions](transport.md): coordinate-dependent growth
  moves and stretches a joint distribution. An analytical solution exposes
  numerical broadening, oscillations, and errors under grid refinement.
- [Estimating size-dependent growth](estimating_growth.md): analytical density curves
  with Gaussian noise illustrate fitting two growth rates, comparing prior choices
  and checking MCMC chains.

- [Continuous tubular precipitation](continuous_precipitation.md): axial transport
  and dispersion carry both solute and crystals; startup and outlet distributions
  are compared under refinement.
- [Calibrating an isothermal protein batch](protein_crystallization.md): original
  concentration and number distributions calibrate CNT, constant growth
  dispersion and growth-dependent dispersion, conditional on fixed interfacial
  energy and Pusey solubility. Frozen predictions test later observations;
  matched Gauss–Newton/MCMC corners compare local and sampled uncertainty.
- [Stem-cell self-renewal and differentiation](stem_cells.md): intracellular
  marker production, binary division and exit from pluripotency reshape a cell
  population. Synthetic examples distinguish cohort dynamics from distributions
  sampled at division and differentiation events.

The constant-growth and two-dimensional calculations have analytical solutions.
The cooling calculation uses an explicitly chosen nonnegative initial condition
and a finer sectional reference, with an independent moment check. Numerical
qualifications stay beside the relevant inputs and figures.
The [axial reactor tutorial](../axial_reactors.md)
adds a physical position coordinate to an intrinsic growth model.

Each page ends with a complete downloadable tutorial: one configuration,
physical rates, initial/boundary conditions, assembly and a solve. Estimation
cases also show observations, priors and likelihoods; the milling example adds
an explicitly invoked MC/PCE stage. Comparisons and reporting/export loops stay in the full
reproduction drivers.

Ordinary initialization and execution use the public
[preparation/execution interfaces](../forward_execution.md). Configured solvers,
physical distributions and consequential support/normalization choices are
visible in the scripts. Prescribed moment states, empty populations and axial
initial arrays stay explicit. Model-specific bounds can be declared without
implementing a time loop. Automatic preparation needs no reader-managed cache.

The remaining helper imports have narrower roles: CSV selection, the protein
bucketed derivative schedule, the milling constrained algebraic reduction,
and the analytical growth reference. Their meaning and
limits stay beside the relevant code. The only piecewise operating change in
these nine walkthroughs is milling's 30-s speed reduction; its specialized
reduced-state/constraint backend currently prevents a direct OperatingHistory
migration. Smooth cooling remains a field callback; autonomous cases introduce
no artificial operating transitions.

The authored scripts live in `examples/tutorials/`. The displayed complete code
and download are read from the same file. Their short optional plotting recipes
call the [shared plotting interface](../plotting.md) on existing solutions,
predictions and UQ summaries. Python is not required for numerical execution.
Milling propagation and stem-cell diagnostic accounting are separate explicit
stages; including an estimation tutorial never starts a fit or posterior chain.
Paper layouts remain plot-only compositions over retained values.

Use the [common setup](../verification.md#Examples-and-figures) for dependencies
and the original data. Retained figures and production campaigns are separate
from the tutorials; their numerical qualifications remain beside them.
