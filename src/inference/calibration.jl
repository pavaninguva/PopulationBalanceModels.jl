struct Experiment{P,O}
    problem::P
    observations::O
end
function Experiment(problem::PBMProblem,observations::AbstractObservation...)
    isempty(observations) && throw(ArgumentError("an experiment needs observations"))
    for obs in observations
        problem.tspan[1]<=first(obs.times)<=last(obs.times)<=problem.tspan[2] || throw(ArgumentError("observation time outside experiment"))
        expected=_observation_size(obs.observable)
        expected===nothing || expected==size(obs.data,1) || throw(DimensionMismatch("observable output dimension differs from data"))
    end
    Experiment(problem,observations)
end

_observation_size(::Union{MomentObservable,PopulationMoment,BulkObservable})=1
_observation_size(obs::HistogramObservable)=length(obs.edges)-1
_observation_size(obs::PopulationHistogram)=length(obs.histogram.edges)-1
_observation_size(::Union{FunctionObservable,SystemObservable})=nothing

"""
    CalibrationProblem(experiments; parameters=(z,experiment)->z, dt=nothing, solver=nothing)

parameters maps unconstrained optimization coordinates into physical model
parameters separately for each experiment, e.g. positive rate = exp(z[1]).
Initial conditions/fields differ by experiment. The same parameter vector can
therefore be fitted jointly to several experiments and measurement types.
Supply `dt` for SSPRK33, or `solver=(problem,times)->solution` for an external
integrator. A solver must return precisely those output times and must check
its own success code. Initial data that depend on fitted parameters require a
separate model construction; the parameter map changes p, not u0.
"""
struct CalibrationProblem{E,F,T,S}
    experiments::E
    parameters::F
    dt::T
    solver::S
    derivatives::Symbol
end
function CalibrationProblem(experiments;parameters=(z,experiment)->z,dt=nothing,solver=nothing,derivatives=:auto)
    ex=Tuple(experiments)
    !isempty(ex) && all(e->e isa Experiment,ex) || throw(ArgumentError("provide at least one Experiment"))
    kinds=[o isa ResidualObservation for e in ex for o in e.observations]
    all(kinds) || !any(kinds) || throw(ArgumentError("do not mix discrepancy blocks and probabilistic observations in one calibration"))
    (dt===nothing ? solver!==nothing : isfinite(dt) && dt>0) ||
        throw(ArgumentError("supply a positive dt or solver=(problem,times)->solution"))
    CalibrationProblem(ex,parameters,dt,solver,_derivative_choice(derivatives))
end

"""Predicted output matrices, grouped by experiment and observation. Uses the calibration's solver and parameter map."""
function predictions(cal::CalibrationProblem,z)
    all(isfinite,z) || throw(ArgumentError("parameters must be finite"))
    map(cal.experiments) do ex
        p=cal.parameters(z,ex)
        prob=_inference_problem(remake(ex.problem;p),z)
        ts=sort!(unique(vcat((o.times for o in ex.observations)...)))
        sol=cal.solver===nothing ? simulate(prob;dt=cal.dt,saveat=ts) : cal.solver(prob,ts)
        length(sol.t)==length(ts)==length(sol.u) &&
            all(isapprox.(sol.t,ts;rtol=1e-12,atol=1e-12)) ||
            throw(ArgumentError("calibration solver must return exactly the requested output times"))
        all(u->all(isfinite,u),sol.u) || throw(NumericalFailure("calibration solver returned nonfinite states"))
        map(ex.observations) do obs
            values=map(eachindex(obs.times)) do j
                index=searchsortedfirst(ts,obs.times[j])
                pred=_observe(obs.observable,sol.u[index],prob.operator,p,obs.times[j])
                length(pred)==size(obs.data,1) || throw(DimensionMismatch("observable changed output dimension"))
                all(isfinite,pred) || throw(NumericalFailure("nonfinite observation prediction"))
                pred
            end
            hcat(values...)
        end
    end
end
function residuals(cal::CalibrationProblem,z)
    result=Vector{promote_type(eltype(z),Float64)}()
    for (ex,outputs) in zip(cal.experiments,predictions(cal,z)),(obs,pred) in zip(ex.observations,outputs)
        append!(result,_whiten(obs,vec(pred.-obs.data)))
    end
    result
end
"""
Known-covariance Gaussian negative log likelihood, including normalization, for
`Observation` blocks. For `ResidualObservation` blocks, returns the sum of their
weighted group-mean squared discrepancies, without likelihood normalization.
The two interpretations cannot be mixed. NumericalFailure
alone can optionally be mapped to Inf. Programming/model-definition errors are
never silently turned into optimizer penalties.
"""
function objective(cal::CalibrationProblem,z;on_failure=:throw)
    on_failure in (:throw,:infinity) || throw(ArgumentError("invalid failure policy"))
    r=try
        residuals(cal,z)
    catch e
        if e isa NumericalFailure && on_failure===:infinity
            return Inf
        end
        rethrow()
    end
    _discrepancy(cal) && return sum(abs2,r)
    0.5*sum(abs2,r)+sum(_noise_logdet(obs)/2+length(obs.data)*log(2π)/2
        for ex in cal.experiments for obs in ex.observations)
end

"""
Residual Jacobian (ForwardDiff by default) in optimization coordinates, its singular
values and numerical rank. Local covariance is (J'J)^(-1) ONLY at full column
rank. This local covariance alone specifies neither a confidence level nor
structural identifiability. Use confidence_ellipsoid with its assumptions and
checks, and profile_likelihood to investigate local asymmetry. Select
`derivatives=:finite_difference` explicitly for an incompatible forward model.
"""
function sensitivity_report(cal::CalibrationProblem,z;relative_step=cbrt(eps(Float64)),rank_tolerance=sqrt(eps(Float64)),derivatives=cal.derivatives)
    isfinite(relative_step) && relative_step>0 && isfinite(rank_tolerance) && 0<rank_tolerance<1 ||
        throw(ArgumentError("step must be positive and finite; rank tolerance must lie in (0,1)"))
    z=float.(copy(z))
    isempty(z) && throw(ArgumentError("provide at least one parameter"))
    J=_differentiate_jacobian(v->residuals(cal,v),z;derivatives,step=relative_step)
    decomposition=svd(J); s=decomposition.S; maxs=isempty(s) ? 0.0 : maximum(s)
    rank=count(x->x>rank_tolerance*maxs,s)
    # Avoid forming normal equations, which square the Jacobian condition number.
    covariance=!_discrepancy(cal) && rank==length(z) ? Matrix(Symmetric(decomposition.V*Diagonal(inv.(s.^2))*decomposition.V')) : nothing
    (;jacobian=J,singular_values=s,rank,covariance,derivatives=derivatives===:auto ? :forwarddiff : derivatives,
      condition_number=rank==length(z) ? first(s)/last(s) : Inf)
end

struct FitResult{Z,T,R}
    minimizer::Z
    minimum::T
    converged::Bool
    raw::R
    failed_evaluations::Int
    failure_messages::Vector{String}
end
