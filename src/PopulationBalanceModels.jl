module PopulationBalanceModels

using LinearAlgebra

export Grid1D, Growth, Birth, Death, Nucleation, Aggregation, Breakage,
       AbstractPBMTerm, AbstractPopulationRepresentation, AbstractMomentBasis,
       AbstractTermMethod, AbstractReconstruction, AbstractMomentMethod,
       AbstractMomentClosure, AbstractDirectMomentMethod,
       TermMethods, term_methods, mechanism, default_method, prepare_term, add_term!,
       MidpointRule, GaussLegendre, BinIntegral, FixedPivot, PointwiseLoss,
       CellAverageLoss, BoundaryInjection, GaussianInjection, NativeResidual,
       ExactMomentRule, QuadratureMomentRule, LogNormalClosure, QMOM,
       validate_moment_method, prepare_moment_state, moment_term!,
       moment_state_length, moment_value, moment_rate, validate_moment_state, check_moment_state,
       DistributedNucleation, Dilution, AdditiveCoordinate,
       IntegratedBirth, IntegratedNucleation, GaussianNucleation,
       Grid2D, Growth2D, Nucleation2D, SizeBC2D, SSPRK33, HOCADI, ExactShift,
       MomentSet, ExactClosure, QuadratureClosure, PolynomialRate,
       ConstantKernel, SelfSimilarDaughters, MomentSource, MomentExchange,
       quadrature, realizability,
       CustomTerm, PBMModel, Upwind, MUSCL, WENO5, NoFlux, OpenBoundary, Periodic,
       InflowFlux, SizeBC, AbstractPopulationDomain, BatchDomain, compile, PBMProblem, remake,
       rhs!, simulate, PBMSolution, density, moment, number_concentration,
       diagnostics, suggest_dt, NumericalFailure, MomentObservable,
       HistogramObservable, FunctionObservable, Observation, Experiment,
       CalibrationProblem, residuals, objective, sensitivity_report,
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
include("coupling/systems.jl")
include("numerics/moments/methods.jl")
include("compilation/moments.jl")
include("numerics/moments/weak_forms.jl")
include("numerics/moments/inversion.jl")
include("coupling/moment_exchange.jl")
include("compilation/tensor.jl")
include("numerics/tensor/transport_sources.jl")
include("numerics/tensor/term_dispatch.jl")
include("time_integration/direct_common.jl")
include("time_integration/compact_adi.jl")
include("time_integration/exact_shift.jl")
include("inference/observations.jl")
include("inference/calibration.jl")

"""Load `Optim` to enable `fit(calibration, initial_parameters; ...)`."""
function fit end

"""Load `SciMLBase` to enable `odeproblem(problem; ...)`."""
function odeproblem end

end
