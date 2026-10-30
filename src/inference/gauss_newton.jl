"""A rejected sensitivity-based approximation. `report` retains the failed
checks; no ridge, pseudoinverse or replacement uncertainty is inserted."""
struct GaussNewtonFailure{R} <: Exception
    report::R
end
Base.showerror(io::IO, e::GaussNewtonFailure) =
    print(io, "Gauss–Newton approximation rejected: ", e.report.reason)

"""Local Gaussian in physical parameter coordinates, with precision
`jacobian' * jacobian + prior_hessian`. The Jacobian is of the joint whitened
Gaussian residuals. This omits residual-weighted second prediction derivatives;
it is distinct from full-Hessian `LaplaceApproximation` and sampler tuning."""
struct GaussNewtonApproximation{F,D,P}
    center::Vector{Float64}
    physical_center::P
    coordinates::Vector{ParameterSpec}
    jacobian::Matrix{Float64}
    likelihood_information::Matrix{Float64}
    prior_hessian::Matrix{Float64}
    precision::Matrix{Float64}
    factor::F
    covariance::Matrix{Float64}
    diagnostics::D
    convention::Symbol
end

function _fixed_gaussian_blocks(problem, z)
    map(problem.experiments) do ex
        theta = physical_parameters(problem, z, ex.name)
        map(enumerate(ex.measurements)) do (index, measurement)
            likelihood = measurement.likelihood
            label = "experiment $(ex.name), measurement $index"
            likelihood isa GaussianLikelihood || throw(ArgumentError(
                "Gauss–Newton covariance requires fixed-error GaussianLikelihood; $label uses $(typeof(likelihood)). Use full-posterior Laplace or MCMC for this likelihood."))
            noise = likelihood.sigma === nothing ? likelihood.covariance : likelihood.sigma
            noise isa Union{Number,AbstractArray} || throw(ArgumentError(
                "Gauss–Newton covariance requires fixed noise arrays/scalars; $label declares a noise callback. Inferred covariance/noise information is not included by this approximation."))
            block = _gaussian(likelihood, measurement.data, zero(measurement.data), theta, ex)
            (; factor=block.factor, data=block.data, contrasts=likelihood.contrasts)
        end
    end
end

function _fixed_gaussian_residuals(problem, z, blocks, counts)
    values = _inference_predictions(problem, z, counts)
    residuals = map(eachindex(values)) do i
        map(eachindex(values[i])) do j
            block = blocks[i][j]
            prediction = block.contrasts === nothing ? values[i][j] : block.contrasts * values[i][j]
            block.factor.L \ vec(prediction - block.data)
        end
    end
    vcat((r for experiment in residuals for r in experiment)...)
end

# Relative precision perturbation in the computed posterior metric. This tests
# weak directions directly instead of comparing a global matrix error with the
# smallest eigenvalue. If ||E||_2 < e < 1, the candidate precision lies between
# (1-e)P and (1+e)P, and its covariance between P^-1/(1+e) and P^-1/(1-e).
function _information_perturbation(J, Q, reference_J, reference_Q, U)
    A = reference_J / U
    difference = (J - reference_J) / U
    prior_difference = U' \ ((Q - reference_Q) / U)
    error = A' * difference + difference' * A + difference' * difference + prior_difference
    opnorm(Symmetric(error), 2)
end

"""
    gauss_newton_approximation(problem, map; step=cbrt(eps(Float64)), ...)

Construct sensitivity-based posterior covariance at a converged MAP point:
`P = J_theta'J_theta + ∇²_theta[-log pi_theta]`, `C = P⁻¹`. `J` differentiates the joint
whitened residuals after observation responses and any independent contrasts.
Every experiment, time and fitted parameter (including nuisance parameters)
is included. Both the residual Jacobian and prior Hessian are with respect to
physical parameters. No numerical-transform Jacobian enters this covariance.

Initially supports fixed scalar/array Gaussian standard deviations or covariance
matrices. Parameter-dependent noise callbacks, count likelihoods and truncated
Gaussian likelihoods are rejected, rather than omitting their information.
ForwardDiff is the default Jacobian backend; `derivatives=:finite_difference`
is explicit. An external ODE adapter must support that prediction-Jacobian path;
this function does not silently select an adjoint or a different solver.

Checks MAP stationarity, residual/target gradient consistency, independent
step-halved prediction sensitivities and prior curvature, positive joint
precision and conditioning. `information_tolerance` bounds the precision
perturbation in its own covariance metric, including weak directions. No jitter
or pseudoinverse is used. Data rank deficiency is reported and may be supported
by prior curvature; it is not interpreted as identification by the data.

The approximation omits residual-weighted second prediction derivatives.
Successful checks establish neither a full-posterior minimum nor Gaussian
tails, unimodality or model adequacy. It is not full-Hessian Laplace. Draws are physical Gaussians and report prior/constraint support violations
without truncation or attempting invalid inverse transforms. No PBM solves are needed to draw parameters; predictive
propagation still costs solves. All covariance entries are in physical parameter units.
"""
function gauss_newton_approximation(problem::EstimationProblem, result::MAPResult;
        step=cbrt(eps(Float64)), derivative_tolerance=1e-3,
        information_tolerance=1e-2, gradient_tolerance=1e-3,
        rank_tolerance=sqrt(eps(Float64)), condition_limit=1e10,
        derivatives=problem.derivatives)
    all(x -> isfinite(x) && x > 0,
        (step, derivative_tolerance, gradient_tolerance)) &&
        0 < information_tolerance < 1 && 0 < rank_tolerance < 1 &&
        isfinite(condition_limit) && condition_limit > 1 ||
        throw(ArgumentError("positive finite steps/tolerances, relative tolerances in (0,1), and condition_limit > 1 required"))
    _derivative_choice(derivatives)
    counts = _InferenceCounts()
    _check_map_result(problem,result;_counts=counts)
    result.converged || throw(GaussNewtonFailure((valid=false, reason="MAP optimization did not converge",)))
    length(result.minimizer) == length(problem.parameters) || throw(DimensionMismatch("MAP parameter dimension differs from problem"))
    [(p.name,p.experiment) for p in result.coordinates] ==
        [(p.name,p.experiment) for p in problem.parameters] ||
        throw(ArgumentError("MAP and problem parameter order differ"))
    started = time_ns()
    internal = copy(result.minimizer)
    center = Float64.(_fitted_values(problem,internal))
    blocks = _fixed_gaussian_blocks(problem, internal)
    stationarity = _map_stationarity(problem,internal,
        posterior_gradient(problem,internal;derivatives,_counts=counts);tolerance=gradient_tolerance)
    stationarity.valid || throw(GaussNewtonFailure((valid=false,reason=stationarity.reason,stationarity)))
    scales = _local_physical_scales(problem,center)
    all(x->isfinite(x) && x>0,scales) || throw(GaussNewtonFailure((valid=false,reason="nonregular physical boundary or unresolved verification scale",)))
    z = zeros(length(center))
    physical = v -> center + scales.*v
    residual = v -> _fixed_gaussian_residuals(problem,
        _inference_coordinates(problem,physical(v)),blocks,counts)
    r = residual(z)
    J = _differentiate_jacobian(residual, z; derivatives, step=step/2)
    prior = v -> -logprior(problem, physical(v);coordinates=:physical)
    prior_gradient = v -> _differentiate_gradient(prior, v; derivatives, step=step/2)
    Q = _differentiate_hessian(prior, z; derivatives, step=step/2)
    Q = Matrix(Symmetric((Q+Q')/2))
    information = J' * J
    precision = Matrix(Symmetric(information + Q))
    eigenpair = eigen(Symmetric(precision))
    values = eigenpair.values
    condition = minimum(values) > 0 ? maximum(values)/minimum(values) : Inf
    decomposition = svd(J; full=true)
    singular = decomposition.S
    data_rank = count(x -> x > rank_tolerance * maximum(singular), singular)
    gradient = -posterior_gradient(problem, center;coordinates=:physical, derivatives, step=step/2, _counts=counts).*scales
    expected_gradient = J' * r + prior_gradient(z)
    gradient_consistency = norm(gradient-expected_gradient)/max(1,norm(gradient),norm(expected_gradient))
    base_report = (; coordinates=:physical, convention=_MAP_CONVENTION, approximation=:gauss_newton,
        verification_coordinates=:scaled_physical,verification_scales=scales,stationarity,
        observations=length(r), parameters=length(z), data_rank,
        data_singular_values=singular, data_weak_direction=decomposition.V[:,end],
        prior_curvature_eigenvalues=eigvals(Symmetric(Q)), eigenvalues=values,
        weak_direction=eigenpair.vectors[:,1], condition_number=condition,
        gradient, gradient_norm=norm(gradient,Inf), gradient_consistency,
        derivative_backend=derivatives === :finite_difference ? :finite_difference : :forwarddiff,
        step, derivative_tolerance, information_tolerance, gradient_tolerance,
        likelihood="fixed-error Gaussian; actual observation response, whitening and contrasts",
        omitted="residual-weighted second derivatives of predictions",
        prior_regularized=data_rank < length(z))
    preliminary = minimum(values) <= eps(Float64)*opnorm(precision) ? "joint information is not numerically positive definite" :
        condition > condition_limit ? "joint information is ill-conditioned" :
        norm(gradient,Inf) > gradient_tolerance ? "MAP point is not stationary" :
        gradient_consistency > derivative_tolerance ? "residual sensitivities disagree with the declared posterior gradient" : ""
    if !isempty(preliminary)
        throw(GaussNewtonFailure(merge(base_report, (valid=false, reason=preliminary,
            forward_solves=counts.solves, elapsed_seconds=(time_ns()-started)/1e9))))
    end
    factor = cholesky(Symmetric(precision))
    J1 = _fd_jacobian(residual, z; step)
    J2 = _fd_jacobian(residual, z; step=step/2)
    Q1 = _fd_jacobian(prior_gradient, z; step)
    Q2 = _fd_jacobian(prior_gradient, z; step=step/2)
    Q1 = (Q1+Q1')/2
    Q2 = (Q2+Q2')/2
    jacobian_agreement = max(norm(J-J2),norm(J1-J2))/max(norm(J),eps(Float64))
    information_errors = (
        coarse=_information_perturbation(J1,Q1,J,Q,factor.U),
        fine=_information_perturbation(J2,Q2,J,Q,factor.U),
        halving=_information_perturbation(J1,Q1,J2,Q2,factor.U))
    information_error = maximum(information_errors)
    # Near a mode, a small relative Jacobian error can still matter through
    # cancellation in J'r. Check independent residual/prior gradients in the
    # covariance metric too; do not hide that effect by normalizing by RSS.
    prior_g1 = _fd_gradient(prior,z;step)
    prior_g2 = _fd_gradient(prior,z;step=step/2)
    g1 = J1' * r + prior_g1
    g2 = J2' * r + prior_g2
    gradient_verification = max(norm(factor.U' \ (g1-gradient)),
        norm(factor.U' \ (g2-gradient)),norm(factor.U' \ (g1-g2)))
    reason = jacobian_agreement > derivative_tolerance ? "prediction sensitivities have not converged under step halving" :
        information_error > information_tolerance ? "joint information is unresolved in covariance-scaled directions" :
        gradient_verification > derivative_tolerance ? "posterior gradient is unresolved in covariance-scaled directions" : ""
    report = merge(base_report, (; valid=isempty(reason), reason, jacobian_agreement,
        information_errors, information_error, gradient_verification,
        covariance_relative_error_bound=information_error < 1 ? information_error/(1-information_error) : Inf,
        forward_solves=counts.solves, posterior_evaluations=counts.evaluations,
        elapsed_seconds=(time_ns()-started)/1e9))
    isempty(reason) || throw(GaussNewtonFailure(report))
    Jphysical = J ./ scales'
    Qphysical = (Q ./ scales) ./ scales'
    information_physical = Jphysical' * Jphysical
    precision_physical = Matrix(Symmetric(information_physical + Qphysical))
    physical_factor = cholesky(Symmetric(precision_physical))
    covariance = Matrix(Symmetric(physical_factor \ Matrix{Float64}(I,length(z),length(z))))
    GaussNewtonApproximation(center, physical_parameters(problem,internal), ParameterSpec[problem.parameters...],
        Matrix(Jphysical), Matrix(information_physical), Qphysical, precision_physical,
        physical_factor, covariance, report, _MAP_CONVENTION)
end
