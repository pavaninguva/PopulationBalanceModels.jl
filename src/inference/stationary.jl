"""
    StationarySolver([algorithm]; condition_limit=1e12, abstol=1e-11, kwargs...)

Checked stationary forward solver for calibration and Bayesian estimation.
Requires the optional NonlinearSolve integration. Remaining keywords pass to
`solve_stationary`. Each evaluation starts from the problem's declared guess;
there is no hidden continuation or warm-start history.

ForwardDiff predictions differentiate the equilibrium equations implicitly,
not the nonlinear iterations. The equilibrated state Jacobian must be
nonsingular with condition number below `condition_limit`. This local result
requires a smooth, isolated root on a consistent branch. Quadrature closures
that cannot differentiate their residual must explicitly use
`derivatives=:finite_difference` in the estimation/calibration problem.
The current implicit implementation uses dense Jacobians and factorizations.
"""
struct StationarySolver{A,K}
    algorithm::A
    options::K
    condition_limit::Float64
end
function StationarySolver(algorithm=nothing;condition_limit=1e12,abstol=1e-11,kwargs...)
    isfinite(condition_limit) && condition_limit>1 || throw(ArgumentError("stationary sensitivity condition_limit must exceed one"))
    StationarySolver(algorithm,(;abstol,kwargs...),Float64(condition_limit))
end
(solver::StationarySolver)(problem::StationaryPBMProblem)=
    solve_stationary(problem,solver.algorithm;solver.options...)

_is_stationary(::StationaryPBMProblem)=true
_inference_problem(problem::StationaryPBMProblem,z)=problem
_validate_experiment_problem(problem,base::StationaryPBMProblem)=
    problem isa StationaryPBMProblem ? nothing : throw(ArgumentError("stationary experiment setup must return a StationaryPBMProblem"))

function _stationary_blocks(observations)
    !isempty(observations) || throw(ArgumentError("stationary experiment needs measurements"))
    all(o->o.times===nothing,observations) || throw(ArgumentError(
        "stationary measurements have no observation times; use Measurement(data, observable) or Observation(data, observable). Use separate experiments for distinct operating points."))
    observations
end

"""
    InferenceExperiment(name, problem::StationaryPBMProblem, measurements...;
                        solver=StationarySolver(), setup, fixed=NamedTuple())

One stationary operating point, sharing the transient framework's parameters,
priors, likelihoods, responses and source tags. Declare timeless measurements
with `Measurement(data, observable; likelihood=...)`. Matrix columns are
replicates of the same state. Distinct operating points need distinct experiments.
`setup(theta, base)` must return a stationary problem. A custom one-argument
solver must return a converged, checked `StationaryPBMSolution` for that problem.
No timestep or artificial time series is used.
"""
function InferenceExperiment(name::Symbol,problem::StationaryPBMProblem,measurements...;
        solver=StationarySolver(),setup=_default_setup,fixed=NamedTuple(),dt=nothing)
    dt===nothing || throw(ArgumentError("stationary estimation has no timestep; omit dt"))
    solver===nothing && (solver=StationarySolver())
    solver isa SciMLSensitivitySolver && throw(ArgumentError("SciMLSensitivitySolver is an ODE adapter; use StationarySolver for stationary experiments"))
    blocks=map(o->o isa Observation ? Measurement(o) : o,measurements)
    all(o->o isa Measurement,blocks) || throw(ArgumentError("provide Measurement or Observation blocks"))
    _stationary_blocks(blocks)
    sources=[o.source for o in blocks if o.source!==nothing]
    length(unique(sources))==length(sources) || throw(ArgumentError("dependent measurement blocks share a source; provide one joint likelihood"))
    InferenceExperiment{typeof(problem),typeof(blocks),typeof(setup),Nothing,typeof(solver),typeof(fixed)}(
        name,problem,blocks,setup,nothing,solver,fixed)
end

function Experiment(problem::StationaryPBMProblem,observations::AbstractObservation...)
    _stationary_blocks(observations)
    for observation in observations
        n=_observation_size(observation.observable)
        n===nothing || n==size(observation.data,1) || throw(DimensionMismatch("observable output dimension differs from data"))
    end
    Experiment(problem,observations)
end

function _checked_observe(observable,state,operator,p,time)
    values=_observe(observable,state,operator,p,time)
    n=_observation_size(observable)
    n===nothing || length(values)==n || throw(DimensionMismatch("observable changed output dimension"))
    all(x->x isa Real && isfinite(x),values) || throw(NumericalFailure("observation predictions must be finite and real"))
    values
end

"""
    observe(observable, solution::PBMSolution; index=length(solution.u))
    observe(observable, solution::StationaryPBMSolution)
    observe(observable, problem, state; time) # time required for transient states

Evaluate one state using the existing representation-aware observation rules.
Returns a vector, including for scalar observables. Stationary solutions must
have passed convergence checks; their declared autonomous/frozen evaluation
time is used. `CellObservable` selects a local axial state. Histograms/CDFs
from moment states still require an explicit reconstruction; DQMOM entries are
not interpreted as raw moments or PSD bins.
"""
observe(observable::AbstractObservable,problem::PBMProblem,state;time)=
    _checked_observe(observable,state,problem.operator,problem.p,time)
function observe(observable::AbstractObservable,solution::PBMSolution;index=length(solution.u))
    checkbounds(solution.u,index)
    observe(observable,solution.problem,solution.u[index];time=solution.t[index])
end
function observe(observable::AbstractObservable,problem::StationaryPBMProblem,state)
    _stationary_check(problem.operator,state,problem.negative_tolerance)
    _checked_observe(observable,state,problem.operator,problem.p,problem.time)
end
function observe(observable::AbstractObservable,solution::StationaryPBMSolution)
    solution.converged || throw(NumericalFailure("cannot observe an unconverged stationary solution"))
    observe(observable,solution.problem,solution.u)
end

function _stationary_inference_state(builder,z,solver,counts)
    problem=builder(z)
    counts===nothing || (counts.solves+=1)
    solution=solver(problem)
    solution isa StationaryPBMSolution || throw(ArgumentError("stationary inference solver must return StationaryPBMSolution, not a fabricated time series"))
    solution.problem===problem || throw(ArgumentError("stationary inference solver returned a solution for a different problem"))
    solution.converged || throw(NumericalFailure("stationary inference solve did not converge: $(solution.retcode)"))
    size(solution.u)==size(problem.guess) || throw(DimensionMismatch("stationary inference state layout changed"))
    _stationary_check(problem.operator,solution.u,problem.negative_tolerance)
    # Independently check both original balances and explicitly constrained rows.
    residual=similar(solution.u)
    stationary_residual!(residual,solution.u,problem)
    physical=_stationary_physical(problem.operator,_stationary_rates(problem,solution.u,problem.p))
    tolerance=diagnostics(solution).acceptance_tolerance
    max(maximum(abs,residual),maximum(abs,physical./problem.residual_scale))<=tolerance ||
        throw(NumericalFailure("stationary inference solution failed its physical balance check"))
    copy(solution.u)
end

function _stationary_inference_state(builder,z::AbstractVector{ForwardDiff.Dual{Tag,V,N}},solver,counts) where {Tag,V,N}
    # Recursion strips one AD layer. With nested duals it also differentiates
    # A, B and the linear solve, retaining the actual posterior Hessian.
    value=ForwardDiff.value.(z)
    state=_stationary_inference_state(builder,value,solver,counts)
    shape=size(state)
    residual=function(u,q)
        problem=builder(q)
        size(problem.guess)==shape || throw(ArgumentError("stationary sensitivities require a fixed state layout"))
        T=promote_type(eltype(u),eltype(q))
        typed_state=T.(u)
        r=similar(typed_state)
        stationary_residual!(reshape(r,shape),reshape(typed_state,shape),problem)
        r
    end
    A=ForwardDiff.jacobian(u->residual(u,value),vec(state))
    B=ForwardDiff.jacobian(q->residual(vec(state),q),value)
    # Equilibrate using primal scales, so changes of state/rate units do not
    # create false rank failures. Scaling cancels algebraically in A*S=-B.
    primal=_primal.(A)
    rows=vec(maximum(abs,primal;dims=2))
    all(x->isfinite(x) && x>0,rows) || throw(DifferentiationFailure(
        ArgumentError("stationary state Jacobian is singular/nonfinite; add a justified inventory constraint or select a well-posed branch")))
    columns=vec(maximum(abs,primal./rows;dims=1))
    all(x->isfinite(x) && x>0,columns) || throw(DifferentiationFailure(ArgumentError("stationary state Jacobian has an unconstrained direction")))
    scaled=A./rows./columns'
    singular=svdvals(_primal.(scaled))
    condition=minimum(singular)>0 ? maximum(singular)/minimum(singular) : Inf
    limit=solver isa StationarySolver ? solver.condition_limit : 1e12
    condition<=limit || throw(DifferentiationFailure(ArgumentError(
        "stationary state Jacobian is singular or ill-conditioned after equilibration (condition=$condition, limit=$limit); local implicit sensitivities are unresolved")))
    sensitivities=(lu(scaled)\(-B./rows))./columns
    seed=hcat((collect(ForwardDiff.partials(x)) for x in z)...)'
    partials=sensitivities*seed
    lifted=map(eachindex(state)) do i
        ForwardDiff.Dual{Tag}(convert(V,state[i]),ForwardDiff.Partials(ntuple(k->convert(V,partials[i,k]),N)))
    end
    reshape(lifted,shape)
end

function _experiment_solution(problem,z,experiment,counts,::StationaryPBMProblem)
    builder=q->_experiment_problem(problem,q,experiment)
    state=_stationary_inference_state(builder,z,experiment.solver,counts)
    builder(z),nothing,(u=state,)
end

function _experiment_predictions(problem,z,experiment,pb::StationaryPBMProblem,::Nothing,state)
    theta=physical_parameters(problem,z,experiment.name)
    map(experiment.measurements) do measurement
        values=observe(measurement.observable,pb,state)
        raw=repeat(reshape(values,:,1),1,size(measurement.data,2))
        output=measurement.response(raw,theta,experiment)
        size(output)==size(measurement.data) || throw(DimensionMismatch("measurement response changed output dimension"))
        all(x->x isa Real && isfinite(x),output) || throw(NumericalFailure("measurement predictions must be finite and real"))
        _normalized_fractions(measurement.observable,measurement.likelihood) && !_valid_fractions(output) &&
            throw(NumericalFailure("normalized fraction predictions must lie in [0,1] and sum to one after the measurement response"))
        output
    end
end

function _stationary_calibration_predictions(cal,z,experiment)
    builder=q->remake(experiment.problem;p=cal.parameters(q,experiment))
    solver=cal.solver===nothing ? StationarySolver() : cal.solver
    state=_stationary_inference_state(builder,z,solver,nothing)
    problem=builder(z)
    map(experiment.observations) do observation
        values=observe(observation.observable,problem,state)
        repeat(reshape(values,:,1),1,size(observation.data,2))
    end
end
