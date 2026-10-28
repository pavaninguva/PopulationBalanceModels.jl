# Changelog

## Unreleased

- Added data-free forward MC propagation and non-intrusive total-degree Hermite/Legendre PCE. Joint/supplied/posterior inputs, scenario rebuilding, existing transient/stationary observables, explicit solve budgets, retained failures, SVD regression and independent validation share one forward definition. No runtime dependencies added.
- Added a reproducible milling-UQ reconstruction with conservative daughter projection, the published algebraic screen law, exact operating segmentation, MC/PCE outputs and plot-only figures. Printed fixed-pivot/design/band ambiguities and missing initialization/domain/statistic details remain explicit reconstruction qualifications.

- Reworked tubular precipitation around logarithmic-size WENO5 and axial MUSCL, independent 50–400-cell refinement paths and a common 1000 × 500 reference with a matched temporal check. Outlet plots retain the digitized Zhang Figure 9(c) points and their extraction uncertainty.
- Added independently checked native-growth banded Jacobian preconditioning and an equivalent current-stage bulk balance to the tubular driver, avoiding duplicate transport evaluation while preserving the physical residual and state checks.

- Aligned the paper growth workflow with the manual's Gaussian benchmark, seven scheme/time pairings and separate error, observed-order and timing figures. Paper exporters now retain only 300-dpi PNG images; posterior samples and original numerical/uncertainty records are preserved.

- Removed development-session reports, findings/disposition tables and internal review tracking from the public manual. Verification instructions and case-specific scientific qualifications remain beside the relevant methods/results; the existing audit URL is retained as a short entry point outside the navigation.

- Clarified the distinct HOCADI and HOCFiniteVolume time updates, bin-integral interpretation, periodic CFL stability and dispersive undershoots. Removed unrelated literature planning notes from the compact verification notes; added independent Fourier-factor checks on unequal axes, including large Courant numbers and signed/zero velocities. Numerical algorithms are unchanged.

- Revised the manual's diffusion, tensor coupling, stationary inference, dilution, moment-state and boundary contracts. Added a complete first Bayesian MAP example and focused GRD/compact guides, preserving legacy verification anchors. Clarified retained case-study limitations and measurement conventions; restyled seven figure sets from unchanged numerical exports and recorded D01–D26/G01–G03 resolutions.

- Narrowed optional stationary-test imports to avoid the Distributions/PopulationBalanceModels `moment` name collision. Sensitivity CI now separates dependency preparation from test execution, rebuilds native package images and retains resolved environments for crash diagnosis; normal compiled modules and all derivative checks remain enabled.
- Declared Statistics as a test dependency and in the optional CI test environment, fixing isolated core-test failures; updated the example loader for current Julia world-age rules.
- Added diagonal Ito size diffusion on Grid2D, including nonuniform rectangular meshes, per-axis closed/periodic boundaries, current-stage bulk coefficients and combined explicit step recommendations. Tensor populations now participate in PBMSystem with shaped population/derivative views, conservative bulk exchange, observations and stationary estimation. Tensor axial coupling is available without intrinsic diffusion; mixed diffusion, tensor escape ledgers and tensor collision/transfer methods remain unsupported.

- Unified observation and parameter-estimation workflows for transient and stationary forward models. Added timeless measurements, `observe`, axial `CellObservable`, and `StationarySolver` with checked roots and implicit first/second sensitivities. Shared MAP, priors, likelihoods, uncertainty, profiles, prediction and optional NUTS remain one statistical target; AD-incompatible closures require an explicit finite-difference path. Added analytical and representation/solver verification plus a synthetic stationary crystallizer example.
- Stationary closure-admissibility failures now report `NumericalFailure` during numerical state validation, allowing estimation to reject invalid trials while preserving programming/model-definition errors.
- Simplified the paper workspace to four case folders, each with a run script, plot script, settings, data and figures. Removed dispatchers and duplicate guides; moved tests/history out of the active workflow. Retained protein samples and original producer manifests byte-for-byte, with separate derived analysis outputs.
- Removed the dissolution case study from the manual while retaining extinction functionality and verification. Archived superseded manual assets, including older posterior samples, with checksums instead of keeping unused versions in `docs/src/assets`.

- Added public `gauss_newton_approximation` and `fit(...; uncertainty=:gauss_newton)` for fixed-error Gaussian Bayesian estimation. Joint whitened sensitivities and declared prior curvature include nuisance parameters and coordinate Jacobians; numerical checks resolve covariance-scaled directions without jitter or pseudoinverses. Shared marginal/draw/prediction interfaces retain a distinct approximation label. Unsupported likelihoods and noise callbacks reject explicitly.
- Applied sensitivity-based uncertainty to both retained protein MAPs without refitting or resampling. Added separate MCMC comparisons and source-bound compatibility evidence; full-Hessian Laplace remains a separate, unchanged verification path.

- Integrated the protein plotting handoff with common axes, external panel labels, analytical prior overlays and vector-first LaTeX exports. Added read-only retained-target compatibility and public-API Laplace analysis, separate from production cache invalidation. Both current protein Laplace calculations fail curvature verification; comparison figures retain those failures beside the unchanged MCMC posterior rather than reporting Gaussian uncertainty.
- Added a source-checked architecture schematic distinguishing transient/stationary inputs, eligible inference paths and required/optional/development dependencies.

- Removed tracked regenerable paper smoke outputs, archived superseded unindexed growth runs, and grouped numerical verification/reporting utilities. Added example/test catalogues and reused example modules across tests without removing numerical assertions. Retained protein posterior samples, data, figures, diagnostics and historical archives unchanged.
- Separated explicit retained-result validation from current-producer verification; strict source matching remains the default. Documentation downloads now remove obsolete generated paths before copying current sources.

- Added direct stationary PBM problems and an optional NonlinearSolve adapter using RobustMultiNewton by default. Reused physical residuals, same-state bulk exchange and named accessors; excluded historical ledgers from stationary unknowns and retained their instantaneous rates.
- Added scaled balance acceptance, explicit inventory constraints, frozen-condition declarations and natural-parameter continuation with failure records. Added analytical well-mixed continuous-crystallizer verification and a runnable coupled example. Direct transport updates and moment boundary events remain unsupported for stationary solves.

- Added a documentation case study of stem-cell self-renewal and differentiation, using Brown et al.'s synthetic OCT4-cargo model. Native growth, binary breakage, removal and coupled exit balances retain cell-event and cargo accounting; runnable examples, analytical division tests and independent numerical-refinement checks accompany the figures. No experimental comparison or paper workflow is included.

- Reorganized the Tang isothermal protein case around CNT versus CNT with native random-fluctuation GRD. Both use unchanged training/test readings, fixed concentration/CDF uncertainties, common kinetic priors and separate four-/five-parameter Bayesian targets. Added data-only, fit and prior-overlaid posterior figures, generated comparison tables, native combined timestep control and numerical/target verification; archived the superseded inferred-noise results.
- Reuse a declared degree-zero size-diffusion coefficient once per residual while retaining current-stage parameter, time and bulk dependence. Verified flux and derivative agreement with the general callback path.

- Replaced the documentation's constant-growth exponential profile with a Gaussian using exact bin integrals; retained its method pairings, Courant numbers, periodic convergence companion and separate paper benchmark.

- Added native Ito/Fokker–Planck `SizeDiffusion` with conservative finite-volume fluxes on uniform/nonuniform batch Grid1D meshes, closed/periodic boundaries, current-stage bulk coefficients, boundary number injection and ForwardDiff-compatible residuals. Included diffusion in default conservative exchange and summed explicit growth/diffusion/loss recommendations, including sectional PBMSystem states. Added analytical refinement/conservation tests and a synthetic broadening example; moment, tensor, axial and direct-update diffusion remain unsupported.
- Corrected periodic sectional growth accounting: the wrapped internal face is no longer counted as particle escape or an additional bulk-material sink.

- Integrated the protein case's diagnostic Wasserstein distance analytically over common affine-quantile intervals. Independent quadrature exposed the previous finite-midpoint error; fitting still uses its unchanged correlated-CDF likelihood.

- Corrected coupled initial-state validation for forward AD: derivative components of zero-valued bins, bulk states and cumulative states no longer cause false negative-population errors during `remake`. Conditional CDF/quantile support checks likewise inspect physical mass; a positive derivative cannot make an empty window observable. Physical negativity is still rejected.

- Reuse a declared degree-zero polynomial growth rate within each sectional residual. Current-stage bulk/time dependence is retained; ordinary callbacks and higher-degree laws keep their face evaluations. Verified identical fluxes and parameter derivatives across reconstructions and boundary policies.

- Reworked the Tang isothermal protein tutorial around Mitchell Case 3's CNT nucleation and empirical growth, Bayesian joint concentration/number-CDF calibration, explicit literature-informed priors and frozen held-out prediction. The original graph readings are unchanged. Consolidated the fitting/export workflow and removed the superseded power-law/onset drivers and outputs; historical Mitchell audit assets remain separate.
- Added partial-cell conditional number CDF/quantile observables and `ResidualObservation`/`WassersteinObservation` comparison blocks. These use the existing calibration/Optim interfaces without interpreting comparison scales as noise variances or reporting confidence intervals.
- Declared the example loader's `DelimitedFiles` dependency in the isolated test target; runtime package dependencies are unchanged.

- Extended `WENO5()` to fixed nonuniform and logarithmic size meshes, rectangular tensor axes and coupled axial meshes. Geometry-dependent cell-average polynomials, optimal weights and smoothness indicators are prepared at compilation; uniform-grid formulas, conservative accounting, boundary fallback and nonnegativity checks are retained. Added analytical reconstruction, refinement, balance and derivative tests plus a runnable characteristic example. Existing case-study runs and assets are unchanged.

- Made HOC explicit in every constant-growth comparison figure and removed Lax–Wendroff from that case study. Added a dedicated tubular size-resolution workflow, finer field/outlet plots and reference-dependent error updates that preserve measured solve timings and the baseline of historical checks.

- Added an optional isolated-process runner for the protein example, joint chain diagnostics and tests. Retained the initial failed sampling budget and regenerated results with 6,000 draws. IPC imports remain outside the physical example loading path.

- Added HOC finite-volume/midpoint to the constant-growth case study. Corrected time-method selection during compilation warmup and exported allocation/GC statistics. Reworked tubular precipitation into independent size/axial refinements, a common WENO5-size/MUSCL-axial/RK4 reference and four startup fields; retained axial-WENO5 startup failures.
- Broadened the protein example's coefficient/exponent priors, retained an explicit previous-prior scenario, and documented fitted versus fixed quantities. Saved example starts now restore physical values before applying changed parameter transforms.

- Added `HOCFiniteVolume`: conservative compact transport of 1D/2D bin integrals with positive declared growth, prescribed time dependence, physical inflow/outflow and point nucleation; fourth-order interior space and second-order complete time integration. Added cumulative number-flux diagnostics, structural compilation/remake checks, independent refinement tests and runnable comparisons.
- Added `ExponentialGrowth`, `TimeScaledGrowth`, `PrescribedRate` and `HOCGrowthLaw` declarations. Kept ordinary callbacks for residual methods and the existing constant-periodic `HOCADI` behavior; published nodal formulas remain separate audit implementations. Direct transport operators now reject external ODE conversion immediately.

- Replaced full embedded reproduction drivers with minimal case-study runs and downloadable sources; separated transport benchmark reporting from model assembly.
- Added WENO3/SSPRK33 to constant-growth front, smooth convergence and single-run timing comparisons; exposed small moment errors and clarified references, normalization and incomplete trajectories.
- Added independent size/axial precipitation refinements, uniform-size WENO5 with axial MUSCL, numerical-reference checks and nondominated runtime/error fronts.
- Smoothed retained protein NUTS summaries with integrated-probability contours and added reference-rate correlations and kinetic-rate bands. Sampling targets and retained draws are unchanged.
- Marked historical LNP long-time outputs as unverified with the current strict collision checks; supplied a checked native startup instead of relaxing those checks.

- Independently verified audit corrections with number/size rescaling, Gaussian tail integrals, unequal collision events, discarded births, analytic mapped targets and upstream Hamiltonian/metric checks.
- Added a 256-bit arithmetic retry for ill-conditioned Gamma EQMOM primary inversion, retaining the selected family and all-moment acceptance tolerance. Quadrature results report `inversion_precision`; explicit rejection remains when reconstruction fails.
- Diagnosed the moment benchmark with independent characteristic moments, full allowed-width root checks, analytic high-order breakage rates and separately controlled integration. Retained failure times and unresolved limits rather than extending failed curves.
- Retained explicit MCMC predictive draw identities across Julia versions; `parameter_draws` now includes flattened MCMC `indices`. No posterior curves or sampling targets changed.
- Clarified accounting state lengths, named-accessor errors, private versus public moment hooks and representation-specific integration remedies. No inference target or protein observations/priors/noise were changed.

- Audited the clean `9e2c6fa` checkout and recorded baseline, findings, supported combinations, dependency versions and executed verification in the manual's audit register.
- Corrected unit-dependent lognormal reconstruction checks and high-order normal quadrature weights; regenerated the affected moment-family comparisons while retaining failed trajectories.
- Unified selected overflow policies with cumulative accounting, included explicitly discarded point births, and corrected zero-feed axial accounting dimensions. Accounting states gain two nucleation-loss mechanisms; use named accessors rather than packed offsets.
- Added accepted-state closure checks, finite-support initial-state checks, normalized-fraction validation and stricter statistical-model/refinement comparisons. Bayesian profile optimizations now verify stationarity; invalid ExactClosure evaluations propagate as numerical failures.
- Added `check_sampling_map` and mapped-evaluation inverse/Jacobian checks, revalidated retained protein targets/predictions, and documented sampler coordinate/metric conventions. Removed duplicate moment inversion work and added a prepared-term dispatch boundary for compiler type inference.
- Added test-only Aqua, targeted JET and minimum/current Julia CI checks; optional tests now declare ForwardDiff and sensitivity CI uses normal compiled modules.

- Added protein crystallization posterior corner, physical prior/marginal, chain/rank and predictive figures, with reproducible NUTS draws, diagnostic screens, residuals and temporal-refinement exports.
- Added optional sampler tuning matrices and checked coordinate maps for correlated parameters; retained the declared posterior, prior coordinates and returned inference-coordinate draws.
- Compared QMOM, rank-adaptive QMOM, Gamma EQMOM, DQMOM and lognormal closures in the aggregation/nucleation case study, including actual failures, reconstruction/time-step checks, conditioning, material errors and single-run timings.

- Added native first-order MeasureEuler evolution through positive quadrature measures, including exact-zero growth/nucleation startup, bounded collision/source updates, checked rank adaptation and consistent split bulk exchange.
- Added explicit zero-size absorption and finite-cutoff moment boundary policies, with atomic hitting/removal and a sectional zero-boundary trace; retained existing default size-boundary and collision-overflow behavior.
- Added optional population/mechanism cumulative number and tracked-property ledgers, integrated with the selected physical stages, and excluded escaped solids from dissolved-material exchange.
- Added conservative joint-realizability axial upwind/dispersion evolution with nonuniform-mesh depletion bounds, separate reactor face-flux integrals and stationary historical ledgers.
- Documented the compatibility matrix, order/CFL conditions, boundary reconstruction, numerical reconstruction defects and derivative/external-solver limits; added analytical startup, dissolution, escape, collision and axial checks plus three runnable examples and dissolution figures.

- Replaced the active temperature-programmed moment illustration with Qamar problems 4 and 5: analytical aggregation/breakage, a narrow nucleation pulse, moment/time refinement and a resolved sectional comparison.
- Added UniformVolumeDaughters and an analytical daughter_moment hook so moment breakage rules retain the paper's continuous daughter integrals.
- Reduced the growth/time benchmark to six representative pairings, included Inguva2022 in timing/error figures, and replaced timing medians with one measured solve per configuration.
- Plotted L1 and RMSE together in paired panels and replaced arbitrary order guides with measured refinement orders.
- Audited Lax–Wendroff against PBM_Schemes: retained the matching interior stencil and documented CFL, boundary, nodal/bin-integral and final-time differences. Added regression checks for the reference's CFL=0.5 oscillations and CFL=1 shift.

- Added one-coordinate rank-adaptive QMOM, Gamma EQMOM with explicit density/bin reconstruction, and local DQMOM with scaled source solves and moment-consistent bulk exchange.
- Added axial transport of named populations and shared scalar bulk states, structured feeds and selectable axial reconstructions while preserving conservative face fluxes.
- Added a continuous-precipitation Case 7A workflow with transient fields, outlet distributions and measured numerical error/runtime; retained explicit reference and scheme differences from Zhang2024.
- Added a protein crystallization fit to reproducibly digitized experimental concentration and diameter markers, explicit rate-unit conversions, observation assumptions and separately declared priors.
- Initially embedded complete example drivers; the current manual replaces these with short runs and source downloads.

- Made ForwardDiff the default for inference gradients, prediction sensitivities and posterior Hessians, with independent finite-difference checks and an explicit finite-difference option for incompatible calculations.
- Added an optional SciMLSensitivity adapter for continuous forward and interpolating-adjoint gradients, including initial-state, observation, noise and shared/local parameter dependence. Added ForwardDiff as a dependency and SciMLSensitivity/ReverseDiff as optional dependencies.
- Corrected the synthetic growth-estimation study to generate nonnegative truncated-Gaussian measurements and use the matching normalized likelihood, predictive noise and quantile residual checks.
- Included all supplied 1D growth reconstructions in the comparison figures, added matched Inguva shifts, and documented term-by-term scheme availability and restrictions.
- Fixed MAP multistart selection to check each candidate's stationarity before choosing among converged runs; an objective-rounding stop cannot displace a stationary fit.
- Made zero-count Poisson and multinomial contributions differentiable at zero predicted mass, preserving their normalization; added SpecialFunctions for count-likelihood log-factorials.

- Added van Leer limited-linear and geometry-aware third-order WENO reconstructions on uniform or nonuniform 1D meshes, and native five-stage fourth-order SSPRK54 time integration.
- Replaced the growth/time heatmap with sharp-front and smooth periodic convergence curves, L1/RMSE comparisons, balanced space/time refinement, and single-run computational-time plots. Checked signed Lax–Wendroff steps against its independent difference formula and bounded periodic norm.
- Added a literal nodal audit of Li2026 SI S11, including fine-grid variable-growth order loss, singular/unstable large-step boundary cases, and a separately derived/tested fourth-order coefficient variant. Kept the public periodic HOCADI variant distinct from both and from the paper's multilevel ADI algorithm.
- Added full-horizon reconstructed LNP calculations for FRR 1, 3, and 5, scaled stiff integration, explicit DLVO quadrature, mechanism/consumption histories, and escaped-volume accounting. Documented missing material inputs, assumptions, disagreement with the target diameter history, and unresolved numerical convergence.


- Added uniform-grid 1D Centered and LinearUpwind growth reconstructions, and restricted conservative LaxWendroff and BeamWarming space–time updates with compile-time eligibility and runtime CFL checks.
- Added ForwardEuler, SSPRK22, and classical RK4 alongside SSPRK33 for the assembled residual, preserving stage checks and simultaneous bulk integration.
- Added a constant-growth analytical benchmark comparing spatial/time-method combinations, including exact CFL=1 shifts, negative undershoots, unstable combinations, refinement, and measured elapsed work.
- Added AxialGrid and AxialDispersion for prescribed constant axial flow and dispersion with one or two intrinsic coordinates, conservative nonuniform-mesh fluxes, Danckwerts inlet/zero-gradient outlet, and periodic or closed alternatives.
- Reworked cooling crystallization around stated nonnegative initial conditions, a finer sectional distribution reference, spatial/time refinement, independent moment verification, and regenerated figures. Retained the printed initial polynomial literally under reported_initial_density, with the old recipe name available for compatibility.
- Added public Inguva2022 and InguvaBraatz2023 transformed transport methods, structured affine/separable growth laws, physical travel-time meshes, and compilation/runtime eligibility checks. Retained ExactShift for its existing constant-velocity interface.
- Reworked the growth-fitting study around analytical point densities with nonnegative truncated-Gaussian measurements, two fitted coefficients, continuous plots, lognormal versus timescale-bounded uniform priors, and chain-level sampling evidence.
- Exposed per-chain BFMI, acceptance rates, step sizes, energy/transition statistics, warmup statistics and Monte Carlo mean errors through diagnostics(samples).
- Renamed the estimation chapter and navigation to Parameter estimation.
- Added a shared Bayesian EstimationProblem for MAP, optional full-posterior Laplace curvature, and optional AdvancedHMC NUTS sampling, with explicit prior coordinates and transform Jacobians.
- Added multinomial and exposed Poisson count likelihoods, normalized Gaussian likelihoods with inferred noise/covariance, and contrast handling for normalized fractions.
- Added shared/local/fixed parameter declarations, uncertain initial-condition setup, observation responses, joint nuisance covariance, likelihood/posterior profiles, prior comparisons, and numerical refits.
- Replaced the manual's introductory trivial kinetic fit with a synthetic size-dependent growth study based on the 2022 affine-growth benchmark, including parameter correlations, sampling diagnostics and numerical-refinement evidence.
- Preserved the existing Gaussian CalibrationProblem interface. Reused the Distributions/StatsAPI fit and loglikelihood function bindings to avoid ambiguous imports.
- Added Random as a standard-library dependency and AdvancedHMC/MCMCDiagnosticTools as optional extension dependencies.

- Added fixed correlated Gaussian observation blocks, forward-solver adapters, predictions, and SVD-based local covariance calculations.
- Added fit diagnostics, checked joint confidence ellipsoids, multiple-start fitting, and nuisance-reoptimized likelihood profiles, with explicit known-error/estimated-scale assumptions.
- Added a synthetic two-rate uncertainty example with residuals, mapped confidence regions, profiles, and solver-refinement checks.
- Added a native-moment crystallization study using reported paracetamol kinetics and explicitly reconstructed monotonic cooling inputs; documented the additional extinction and distribution information needed for the full cycling figure.
- Added visible simulation and accuracy figures, including clearly labelled retained demonstration results. Applied LaTeX typography and removed plot grid lines.
- Documented the DLVO-to-collision interface and added unit-specified inverse-Debye-length and unequal-sphere van der Waals helpers to the LNP recipe; retained the unresolved full attachment convention.


- Renamed the package and optional extensions to PopulationBalanceModels, preserving its UUID and version.
- Made `docs/src/` the authoritative manual and consolidated standalone architecture, numerical, validation, and publishing reports.
- Rewrote the model tutorials around equations, parameters, rate callbacks, boundaries, numerical treatment, and observations, with executable code blocks.
- Organized the case studies around three physical problems: batch cooling crystallization, lipid nanoparticle formation, and growth in two particle dimensions.
- Separated the mathematical and numerical treatment of growth, nucleation, local sources and removal, collisions, and time integration.
- Added keyword term declarations while retaining positional constructors, including keyword forms for two-coordinate growth and nucleation.
- Shared crystallization rate callbacks and parameters between sectional and exactly closed moment builders.
- Made the two-coordinate benchmark and its analytical and mapped references use the same explicit parameters. Added area-weighted density errors and actual physical cell edges to exported fields and plots.
- Replaced the published manual's constant-growth comparison with the coordinate-dependent growth problem and its analytical solution. Retained earlier diagnostics outside the case studies.
- Added zero-particle initialization, the full 4320-minute nanoparticle output schedule, adaptive-solver export support, and peak-scaled intensity outputs.
- Kept the printed crystallization initial polynomial literal; the current benchmark uses a separately labelled nonnegative initial condition and does not claim to repair the published coefficients.
- Retained compact/ADI stability, convergence, and sharp-front evidence, including negative undershoots and the restriction to periodic constant growth.
- Updated navigation, links, versioned documentation deployment, and local contribution instructions. Simplified the README and repository root.

### Fixes

- Bounded native moment flights approaching stagnant or inward growth boundaries to prevent artificial finite-time removal under asymptotic shrinkage such as G=-L.

- Prevented generic sectional bulk exchange from treating discarded aggregation/breakage products as dissolved material, including when cumulative accounting is not requested.

- Declared the Printf standard library in the test target, so Pkg.test can load the existing compact-scheme audit in its isolated environment.

- Evaluated the constant-growth exponential reference with a factored exponential and expm1, avoiding cancellation of tiny tails when subtracting nearly saturated cumulative integrals.

- Used SciML's parameterless sentinel for empty parameter tuples in the ODE adapter, preventing symbolic-remake recursion while preserving the original empty NamedTuple inside PBM callbacks.
- Kept independent Gaussian measurement covariance diagonal during factorization, avoiding dense storage and cubic work for large density datasets while retaining likelihood normalization.
- Integrated narrow moving Gaussian nucleation sources over bins in the nanoparticle recipe, preserving the explicit source-normalization convention.
- Included boundary nucleation in the crystallization recipe's discrete material exchange.
- Supplied the operator in two-coordinate callback contexts so `moment(ctx,(kx,ky))` works consistently with other representations.
- Validated two-coordinate velocity callback results as two finite real components, with a direct error for invalid output.
- Used common evolution-argument validation for SSPRK33 and direct transport methods; invalid step limits are now rejected even for initial-state-only output.

The inherited implementation includes sectional and moment models, shared bulk
balances, population interactions, inference adapters, and restricted two-coordinate
transport. Earlier internal milestones are not treated as published releases.
