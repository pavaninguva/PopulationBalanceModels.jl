module PopulationBalanceModels

using LinearAlgebra
using Random
import Distributions
import ForwardDiff
using SpecialFunctions: loggamma
import Distributions: loglikelihood, fit

export Grid1D, Growth, Birth, Death, Nucleation, Aggregation, Breakage,
       AbstractPBMTerm, AbstractPopulationRepresentation, AbstractMomentBasis,
       AbstractTermMethod, AbstractReconstruction, AbstractMomentMethod,
       AbstractMomentClosure, AbstractDirectMomentMethod,
       TermMethods, term_methods, mechanism, default_method, prepare_term, add_term!,
       MidpointRule, GaussLegendre, BinIntegral, FixedPivot, PointwiseLoss,
       CellAverageLoss, BoundaryInjection, GaussianInjection, NativeResidual,
       ExactMomentRule, QuadratureMomentRule, LogNormalClosure, QMOM,
       RankAdaptiveQMOM, EQMOM, DQMOM, quadrature_state,
       reconstructed_density, reconstructed_bins,
       validate_moment_method, prepare_moment_state, moment_term!,
       moment_state_length, moment_value, moment_rate, validate_moment_state, check_moment_state,
       DistributedNucleation, Dilution, AdditiveCoordinate,
       IntegratedBirth, IntegratedNucleation, GaussianNucleation,
       Grid2D, Growth2D, Nucleation2D, SizeBC2D, ForwardEuler, SSPRK22, SSPRK33, RK4, SSPRK54, HOCADI, ExactShift,
       Centered, LinearUpwind, VanLeer, WENO3, LaxWendroff, BeamWarming,
       AxialGrid, AxialDispersion, axial_problem, axial_stateview, axial_population, axial_fluxes,
       AffineGrowth, ExponentialGrowth, SeparableGrowth2D, TimeScaledGrowth,
       PrescribedRate, HOCGrowthLaw, HOCFiniteVolume, Inguva2022, InguvaBraatz2023,
       characteristic_grid, characteristic_density,
       MomentSet, ExactClosure, QuadratureClosure, PolynomialRate,
       ConstantKernel, SelfSimilarDaughters, UniformVolumeDaughters, daughter_moment, MomentSource, MomentExchange,
       quadrature, realizability,
       CustomTerm, PBMModel, Upwind, MUSCL, WENO5, NoFlux, OpenBoundary, Periodic,
       InflowFlux, AbsorbingZero, MomentBoundary, CumulativeAccounting, cumulative, accounting_rates, MeasureEuler, SizeBC, AbstractPopulationDomain, BatchDomain, compile, PBMProblem, remake,
       rhs!, simulate, PBMSolution, density, moment, number_concentration,
       diagnostics, suggest_dt, NumericalFailure, MomentObservable,
       HistogramObservable, FunctionObservable, Observation, Experiment,
       CalibrationProblem, predictions, residuals, objective, sensitivity_report,
       fit_diagnostics, confidence_ellipsoid, ConfidenceEllipsoid, ellipse_points,
       fit_multistart, profile_likelihood,
       ParameterSpec, Prior, IdentityTransform, LogTransform, LogitTransform,
       InferenceExperiment, Measurement, EstimationProblem, DistributionObservable,
       GaussianLikelihood, TruncatedGaussianLikelihood, PoissonLikelihood, MultinomialLikelihood,
       initial_coordinates, physical_parameters, loglikelihood, logprior, logposterior,
       posterior_components, posterior_gradient, likelihood_gradient, check_gradient, MAPResult,
       SciMLSensitivitySolver, DifferentiationFailure,
       LaplaceApproximation, LaplaceFailure, laplace_approximation, marginal_covariance,
       parameter_draws, posterior_predictive, identifiability_report, estimation_profile,
       replace_priors, prior_sensitivity, refinement_study, sample_posterior, check_sampling_map, PosteriorSamples,
       PBMSystem, SystemContext, PopulationTransfer, CrossAggregation,
       ConservativeExchange, CustomCoupling, population, bulk, stateview,
       PopulationMoment, BulkObservable, PopulationHistogram, SystemObservable,
       fit, FitResult, odeproblem

include("core/interfaces.jl")
include("representations/grids.jl")
include("modeling/terms.jl")
include("numerics/sectional/reconstructions.jl")
include("modeling/boundaries.jl")
include("core/domains.jl")
include("modeling/sources.jl")
include("modeling/laws.jl")
include("modeling/tensor_terms.jl")
include("representations/moment_sets.jl")
include("numerics/policies.jl")
include("numerics/quadrature.jl")
include("core/contexts.jl")
include("compilation/sectional.jl")
include("numerics/sectional/transport.jl")
include("numerics/sectional/deposition.jl")
include("numerics/sectional/local_sources.jl")
include("numerics/sectional/collisions.jl")
include("numerics/sectional/integrated_sources.jl")
include("numerics/sectional/term_dispatch.jl")
include("time_integration/simulation.jl")
include("coupling/accounting_types.jl")
include("coupling/systems.jl")
include("numerics/moments/methods.jl")
include("compilation/moments.jl")
include("numerics/moments/weak_forms.jl")
include("numerics/moments/inversion.jl")
include("numerics/moments/adaptive_extended.jl")
include("numerics/moments/direct.jl")
include("coupling/moment_exchange.jl")
include("compilation/tensor.jl")
include("numerics/tensor/transport_sources.jl")
include("numerics/tensor/term_dispatch.jl")
include("time_integration/direct_common.jl")
include("time_integration/compact_adi.jl")
include("time_integration/exact_shift.jl")
include("time_integration/characteristic_transport.jl")
include("time_integration/constant_transport.jl")
include("compilation/compact.jl")
include("time_integration/compact_finite_volume.jl")
include("compilation/axial.jl")
include("compilation/axial_system.jl")
include("coupling/accounting.jl")
include("time_integration/measure_euler.jl")
include("inference/observations.jl")
include("inference/differentiation.jl")
include("inference/calibration.jl")
include("inference/uncertainty.jl")
include("inference/parameters.jl")
include("inference/statistical_model.jl")
include("inference/bayesian_diagnostics.jl")
include("inference/sampling_coordinates.jl")

@doc "Load Optim to enable fit(calibration, initial_parameters; ...) or MAP fit(estimation_problem; ...)." fit

"""Load `SciMLBase` to enable `odeproblem(problem; ...)`."""
function odeproblem end

end
