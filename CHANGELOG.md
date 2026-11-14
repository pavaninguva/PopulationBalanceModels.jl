# Changelog

## 0.1.0 — unreleased

First public release, consolidating the internal development milestones.

### Interface changes and migration

- Bayesian parameters use a physical `initial`, a physical distribution as `prior`,
  and an explicit `transform`, including `IdentityTransform()` when appropriate.
  MAP, gradients, MCMC and local uncertainty share the declared inference-coordinate
  density, including physical-prior Jacobians (`inference_map_v2`). This supersedes
  the interim physical-MAP/automatic-transform interface. Tagged `Prior` wrappers
  retain their probability laws for compatibility. Older MAP/covariance objects
  require a convention check and, where incompatible, recomputation.
- `compile(model, basis; moment_method=...)` is the canonical moment interface.
  Node counts may be inferred from compatible moment orders. Legacy `method`,
  `closure` and explicit per-term rules remain compatibility interfaces.
  `MomentSet` specifies a basis; DQMOM still advances weights and weighted nodes.
- See the [migration guide](docs/src/migration.md) for executable declarations,
  retained-result conventions and the distinction between numerical and physical
  coordinates. No retained scientific result is relabelled as a new calculation.

### Forward modelling and numerical preparation

- Added `InitialDensity`, `InitialDistribution`, `initialize` and initialization
  reports for eligible grid and moment representations. Probability tails are
  excluded without redistribution by default; conditioning is explicit.
  Parameter-dependent descriptions are rebuilt by `remake`, while numerical
  arrays retain fixed-state semantics.
- Added automatic `prepare_run` and reusable continuous expected daughter-number
  laws through `ContinuousDaughters`. Preparation preserves the declared support,
  count/property balances, additive-coordinate map and unresolved-tail policy.
- Added reusable `ForwardSolver`, native `StepControl`, checked optional external
  integration and integration-work diagnostics. Controlled native RK retries
  preserve accepted states and accounting. Stability/admissibility recommendations
  are distinct from integration-error estimates; unsupported derivative/control
  combinations reject explicitly.
- Added `OperatingHistory` for fixed prescribed transitions with continuous state
  transfer, held endpoint-stage conditions, external-solver restarts and right-hand
  observations. State-triggered events, resets and fitted transition times remain
  outside this interface.
- Standalone axial `Grid1D`/`Grid2D` compilation accepts `axial_scheme` independently
  of intrinsic `TermMethods`. `DensityInlet` integrates current density values
  over intrinsic bins without implicit renormalization. Existing discretized
  inlet callbacks and coupled-system formats remain supported.
- Added diagonal Itô/Fokker–Planck diffusion on rectangular `Grid2D` meshes,
  including eligible shared bulk balances and stationary inference. Independent
  noise increments do not imply an independent joint population. Mixed diffusion,
  tensor escape ledgers, tensor boundary injection and tensor collision/transfer
  algorithms remain unavailable; intrinsic diffusion rejects axial adapters.
- Moment compilation now plans analytical and reconstruction-dependent contributions
  per equation. Tracked moments are used directly; only missing moments/integrals
  use the selected family. `moment_plan` exposes this treatment. `ExactClosure`
  is an optional strict request, preserving boundary and family-admissibility checks.
- Retained the implemented growth, collision, characteristic, compact and time
  methods with their mesh, boundary, positivity and realizability restrictions.
  The periodic `HOCADI` variant remains distinct from the Li et al. algorithm.

### Estimation and uncertainty

- Unified transient/stationary observations and estimation through `observe`,
  timeless measurements, `CellObservable` and `StationarySolver`. Eligible smooth,
  isolated stationary roots use implicit derivatives; incompatible closures need
  an explicitly selected finite-difference path.
- Added normalized likelihood families, correlated Gaussian blocks, shared/local
  and nuisance parameters, configurable derivatives, profiles and refinement
  reports. Full-Hessian Laplace and eligible fixed-error Gaussian Gauss–Newton
  uncertainty retain independent derivative, stationarity and curvature checks.
- Added continued NUTS sampling with default diagnostic stopping, explicit budgets,
  convergence history and optional MCSE requirements. Explicit `samples=N` retains
  fixed-length sampling. Local curvature used for tuning is not posterior evidence.
- Added data-free forward Monte Carlo and non-intrusive PCE using the same scenario,
  solver and named observables. Joint/supplied/posterior draws, solve budgets,
  failures and coordinate metadata remain explicit.
- PCE supports orthonormal Hermite, Legendre and beta/Jacobi bases, a recurrence
  extension interface, SVD regression, Gaussian-quadrature projection and independently
  checked validation/refinement. Training/validation outputs cannot be supplied by
  another surrogate. Analytical moments and surrogate-derived quantiles are distinct.

### Plotting, documentation and scientific examples

- Added optional `PythonPlot` integration backed by one package-owned Matplotlib
  renderer for distributions, observables, prepared fits, parameters, UQ and fields.
  Plotting performs no fitting, sampling or forward solves. Publication typography
  is opt-in; ordinary numerical use requires neither Python nor LaTeX.
- Replaced generated tutorial programs with authored, downloadable walkthroughs
  and named companions. Split the API reference into source-routed topics with
  strict coverage, routing and local-link checks.
- Protein now compares fixed-gamma CNT, constant-D0 GRD and growth-dependent
  GRD (`D_G = k_D G`) using the retained Pusey solubility choice. Observations are
  scatter points. Joint Gauss–Newton/MCMC figures preserve producer identities,
  latent-band meanings and full-Laplace rejection diagnostics.
- Tubular precipitation uses logarithmic-size WENO5, axial MUSCL, separate size
  and axial refinement and a common 1000 × 500 numerical reference with temporal
  verification. Digitized literature readings retain their extraction provenance.
- Milling includes a qualified Barrasso forward comparison and the full 12,000-draw
  MC/PCE reconstruction. Source ambiguities and surrogate-validation limits remain
  visible. Stem-cell and LNP pages retain their synthetic/reconstructed-input and
  current-source-verification qualifications; collision EQMOM remains unresolved.
- Large retained datasets, publication figures and historical test archives are
  optional checksum-verified reproduction downloads. Small model/data declarations
  and original producer metadata remain in the source tree. Fetching is explicit
  and never launches numerical production.

### Correctness and compatibility

- Optional integration CI uses portable CPU code generation to avoid the
  Julia 1.13.3 / LLVM 20 vectorizer crash during mixed stationary/transient
  gradients. Normal optimization and all derivative assertions remain enabled.
- Fixed current-Julia tutorial checks to inspect freshly included bindings in
  their latest world. Optional CI now separates environment preparation from
  test execution and uses one Python interpreter for plotting setup and tests.
- Preserved discrete inventory accounting for boundary nucleation, unresolved births,
  collisions and coupled bulk exchange without double-counting discarded material.
- Preserved zero-count likelihood derivatives, normalized covariance handling,
  physical-state checks during AD and explicit failures of incompatible declarations.
- Retained analytical/dense-reference tests for transport, closure, covariance and
  sampling-coordinate identities; historical numerical files and producer records
  are unchanged by documentation, plotting and packaging updates.

Earlier development entries describe internal milestones rather than published
releases; their original records remain in Git history and retained producer files.
