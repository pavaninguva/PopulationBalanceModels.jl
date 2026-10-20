struct Experiment{P,O}
    problem::P
    observations::O
end
function Experiment(problem::PBMProblem,observations::Observation...)
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
    CalibrationProblem(experiments; parameters=(z,experiment)->z, dt)

parameters maps unconstrained optimization coordinates into physical model
parameters separately for each experiment, e.g. positive rate = exp(z[1]).
Initial conditions/fields differ by experiment. The same parameter vector can
therefore be fitted jointly to several experiments and measurement types.
"""
struct CalibrationProblem{E,F,T}
    experiments::E
    parameters::F
    dt::T
end
function CalibrationProblem(experiments;parameters=(z,experiment)->z,dt)
    ex=Tuple(experiments)
    !isempty(ex) && all(e->e isa Experiment,ex) || throw(ArgumentError("provide at least one Experiment"))
    isfinite(dt) && dt>0 || throw(ArgumentError("calibration dt must be positive"))
    CalibrationProblem(ex,parameters,dt)
end
function residuals(cal::CalibrationProblem,z)
    result=Vector{promote_type(eltype(z),Float64)}()
    for ex in cal.experiments
        p=cal.parameters(z,ex)
        # The reference solver is a Float64-oriented finite-difference fitting
        # backend. Generic AD/sensitivity backends belong in dedicated extensions.
        prob=remake(ex.problem;p)
        ts=sort!(unique(vcat((o.times for o in ex.observations)...)))
        sol=simulate(prob;dt=cal.dt,saveat=ts)
        for obs in ex.observations, j in eachindex(obs.times)
            index=searchsortedfirst(sol.t,obs.times[j])
            pred=_observe(obs.observable,sol.u[index],prob.operator,p,obs.times[j])
            length(pred)==size(obs.data,1) || throw(DimensionMismatch("observable changed output dimension"))
            append!(result,(pred .- obs.data[:,j]) ./ obs.sigma[:,j])
        end
    end
    result
end
"""
Known-variance independent Gaussian negative log likelihood. NumericalFailure
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
    0.5*sum(abs2,r)+sum(sum(log,obs.sigma)+length(obs.data)*log(2π)/2
        for ex in cal.experiments for obs in ex.observations)
end

"""
Finite-difference residual Jacobian in optimization coordinates, its singular
values and numerical rank. Local covariance is (J'J)^(-1) ONLY at full column
rank. This is a local Gaussian approximation, not a confidence interval or
structural-identifiability proof; use profiles/bootstrap/posteriors in a release.
"""
function sensitivity_report(cal::CalibrationProblem,z;relative_step=cbrt(eps(Float64)),rank_tolerance=sqrt(eps(Float64)))
    relative_step>0 && rank_tolerance>0 || throw(ArgumentError("steps and rank tolerance must be positive"))
    z=float.(copy(z)); r=residuals(cal,z); J=zeros(length(r),length(z))
    isempty(z) && throw(ArgumentError("provide at least one parameter"))
    for j in eachindex(z)
        h=relative_step*max(1,abs(z[j])); zp=copy(z); zm=copy(z)
        zp[j]+=h; zm[j]-=h
        J[:,j]=(residuals(cal,zp)-residuals(cal,zm))/(2h)
    end
    s=svdvals(J); maxs=isempty(s) ? 0.0 : maximum(s)
    rank=count(x->x>rank_tolerance*maxs,s)
    covariance=rank==length(z) ? inv(Symmetric(J'*J)) : nothing
    (;jacobian=J,singular_values=s,rank,covariance,
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
