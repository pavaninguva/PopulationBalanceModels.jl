"""
    StepControl(; cfl=0.5, bound=nothing, guard=nothing, min_dt=1e-14,
                max_rejections=20, shrink=0.5)

Native residual-RK step control. Combine the current `suggest_dt` recommendation
with the solver's maximum step and an optional `bound(problem,state,t)` returning
a positive maximum step. Set `cfl=nothing` when only a separately justified bound
is available. `guard(problem,state,t)` must return Bool; false rejects a trial.
It is checked at every RK stage and the final combination, alongside the native
population/bulk/admissibility checks. Callbacks must be pure.

This is stability/admissibility control, **not** an integration-error estimate or
a bound on future coefficients. Declare model-specific bounds/guards for bulk
depletion or rapidly varying inputs. There is no clipping or projection. Rejected
trials restart from the accepted state, including all accounting components.
Parameter/state-dependent step selection is not supported by ForwardDiff or
continuous sensitivity adapters; select finite differences explicitly.
"""
struct StepControl{C,B,G,T}
    cfl::C
    bound::B
    guard::G
    min_dt::T
    max_rejections::Int
    shrink::T
end
function StepControl(;cfl=0.5,bound=nothing,guard=nothing,min_dt=1e-14,
        max_rejections=20,shrink=0.5)
    cfl===nothing || (!(cfl isa ForwardDiff.Dual) && isfinite(cfl) && 0<cfl<=1) || throw(ArgumentError("cfl must be a fixed real in (0,1] or nothing"))
    !(min_dt isa ForwardDiff.Dual || shrink isa ForwardDiff.Dual) &&
        isfinite(min_dt) && min_dt>0 && isfinite(shrink) && 0<shrink<1 ||
        throw(ArgumentError("min_dt must be positive and shrink in (0,1)"))
    max_rejections isa Integer && max_rejections>=0 || throw(ArgumentError("max_rejections must be nonnegative"))
    cfl===nothing && bound===nothing && guard===nothing && throw(ArgumentError("StepControl needs a recommendation, bound or guard"))
    a,b=promote(float(min_dt),float(shrink))
    StepControl(cfl,bound,guard,a,Int(max_rejections),b)
end

"""
    ForwardSolver(algorithm=SSPRK33(); dt=nothing, backend=:native,
                  control=nothing, check_nonnegative=true,
                  negative_tolerance=1e-12, maxsteps=10^7, kwargs...)

Reusable transient execution configuration. Use `simulate(pb;solver,saveat)` or
`solver(pb,times)` directly in InferenceExperiment and UQProblem. Native `dt` is
a fixed **maximum** step; output times shorten steps. `control=StepControl(...)`
is available for eligible residual RK methods. Complete updates and MeasureEuler
keep their existing fixed-step contracts and reject controls.

For an external ODE algorithm use `backend=:external`, load SciMLBase and the
chosen solver, and supply backend accuracy options (e.g. reltol/abstol). The
adapter checks status, completion, output coverage/shape, finiteness and saved
state admissibility; it does not validate internal ODE stages. External callbacks,
state resets and custom save options are outside this interface.
"""
struct ForwardSolver{A,D,C,K}
    algorithm::A
    dt::D
    backend::Symbol
    control::C
    check_nonnegative::Bool
    negative_tolerance::Float64
    maxsteps::Int
    options::K
end
function ForwardSolver(algorithm=SSPRK33();dt=nothing,backend=:native,control=nothing,
        check_nonnegative=true,negative_tolerance=1e-12,maxsteps=10^7,kwargs...)
    backend in (:native,:external,:sensitivity) || throw(ArgumentError("backend must be :native, :external or :sensitivity"))
    check_nonnegative isa Bool || throw(ArgumentError("check_nonnegative must be Bool"))
    dt===nothing || (dt isa Real && !(dt isa ForwardDiff.Dual) && isfinite(dt) && dt>0) ||
        throw(ArgumentError("dt must be a fixed positive finite real maximum step, not a differentiated control"))
    backend===:native && dt===nothing && throw(ArgumentError("native ForwardSolver requires dt (a maximum step)"))
    control===nothing || control isa StepControl || throw(ArgumentError("control must be StepControl or nothing"))
    control!==nothing && (backend!==:native || !(algorithm isa AbstractExplicitRK)) &&
        throw(ArgumentError("StepControl requires a native explicit residual RK algorithm"))
    backend===:native && !isempty(kwargs) && throw(ArgumentError("native solver options are dt, control, check_nonnegative, negative_tolerance and maxsteps"))
    any(k->k in (:callback,:saveat,:save_everystep,:save_start,:save_end,:dense,:tstops,:maxiters),keys(kwargs)) &&
        throw(ArgumentError("callbacks and saving/stopping options are owned by ForwardSolver"))
    any(k->k in (:isoutofdomain,:unstable_check,:constraints,:positive),keys(kwargs)) &&
        throw(ArgumentError("backend constraint callbacks are not implemented by ForwardSolver; the external contract validates saved states, not internal stages"))
    isfinite(negative_tolerance) && negative_tolerance>=0 || throw(ArgumentError("invalid negative_tolerance"))
    maxsteps isa Integer && maxsteps>0 || throw(ArgumentError("maxsteps must be a positive integer"))
    ForwardSolver(algorithm,dt,backend,control,check_nonnegative,Float64(negative_tolerance),Int(maxsteps),(;kwargs...))
end
(s::ForwardSolver)(pb::PBMProblem,times)=simulate(pb;solver=s,saveat=times)

"""
    simulate(problem; solver=ForwardSolver(...), saveat=...)
    simulate(problem; method=SSPRK33(), dt, saveat=..., ...)

Execute a transient problem and return a checked PBMSolution. The legacy
method/dt form retains fixed-maximum-step behavior. Solver configurations are
also callable as `solver(problem,times)` for estimation and forward UQ.
"""
function simulate(pb::PBMProblem;solver=nothing,dt=nothing,method=nothing,
        saveat=collect(pb.tspan),check_nonnegative=nothing,negative_tolerance=nothing,maxsteps=nothing)
    if solver===nothing
        dt===nothing && throw(ArgumentError("simulate requires dt or a configured solver"))
        check_nonnegative=check_nonnegative===nothing ? true : check_nonnegative
        negative_tolerance=negative_tolerance===nothing ? 1e-12 : negative_tolerance
        maxsteps=maxsteps===nothing ? 10^7 : maxsteps
        solver=ForwardSolver(method===nothing ? _default_evolution(pb.operator) : method;
            dt,check_nonnegative,negative_tolerance,maxsteps)
    else
        solver isa ForwardSolver || throw(ArgumentError("solver must be ForwardSolver; custom callable solvers remain supported by InferenceExperiment/UQProblem"))
        (dt===nothing && method===nothing && check_nonnegative===nothing && negative_tolerance===nothing && maxsteps===nothing) ||
            throw(ArgumentError("put method, dt and validation settings in ForwardSolver when solver is supplied"))
    end
    _execute_forward(pb,solver,saveat)
end

function _execute_forward(pb,s,times)
    ts=_evolution_args(pb,s.dt===nothing ? 1.0 : s.dt,times,s.negative_tolerance,s.maxsteps)
    pb.history===nothing || return _segmented_forward(pb,s,ts)
    op=prepare_run(pb.operator,pb.p)
    localpb=PBMProblem(op,_execution_initial(pb),pb.tspan,pb.p)
    result=_execute_backend(localpb,s,ts)
    _checked_solution(pb,result.t,result.u,result.steps,result.integration_diagnostics,s,ts)
end
_run_numeric_type(x::Real)=typeof(x)
_run_numeric_type(x::Union{Tuple,NamedTuple,AbstractArray})=foldl(promote_type,(_run_numeric_type(v) for v in x);init=Union{})
_run_numeric_type(x)=Union{}
function _execution_initial(pb)
    # Preserve legacy floating-point storage for numerical initial arrays.
    # Only supported dual bindings require automatic tangent storage promotion.
    T=eltype(pb.u0)
    _contains_dual(pb.p) && (T=promote_type(T,_run_numeric_type(pb.p)))
    if pb.history!==nothing
        for i in eachindex(pb.history.parameters)
            parameters=_interval_parameters(pb.history,pb.p,i)
            _contains_dual(parameters) && (T=promote_type(T,_run_numeric_type(parameters)))
        end
    end
    T.(pb.u0)
end
function _execute_backend(pb,s,ts)
    if s.backend===:external
        (_contains_dual(pb.u0) || _contains_dual(pb.p)) && throw(ArgumentError(
            "ForwardSolver backend=:external does not differentiate through ODE step selection; use backend=:sensitivity with SciMLSensitivitySolver or explicitly select finite differences"))
        return _external_forward(pb,s,ts)
    end
    s.backend===:sensitivity && return _configured_sensitivity_forward(pb,s,ts)
    s.control===nothing && return _simulate_fixed(pb;dt=s.dt,saveat=ts,
        check_nonnegative=s.check_nonnegative,negative_tolerance=s.negative_tolerance,
        maxsteps=s.maxsteps,method=s.algorithm)
    _controlled_forward(pb,s,ts)
end
_external_forward(pb,s,ts)=throw(ArgumentError("load SciMLBase and a compatible ODE solver for backend=:external"))
_configured_sensitivity_forward(pb,s,ts)=throw(ArgumentError("backend=:sensitivity requires SciMLSensitivitySolver"))

# Shared native/external completion contract; deliberately independent of any
# backend return-code implementation. External extensions check that as well.
function _checked_solution(pb,ts,states,steps,info,s,requested)
    length(ts)==length(states)==length(requested) && all(ts.==requested) ||
        throw(NumericalFailure("forward solver omitted or changed requested output times"))
    for state in states
        size(state)==size(pb.u0) || throw(DimensionMismatch("solver output state shape differs from the supplied problem"))
        all(isfinite,state) || throw(NumericalFailure("nonfinite saved forward state"))
        _check_problem_state(pb,state,s.check_nonnegative ? s.negative_tolerance : Inf)
        _check_admissible_state(pb.operator,state)
    end
    settings=(backend=s.backend,algorithm=string(typeof(s.algorithm)),maximum_dt=s.dt,
        maxsteps=s.maxsteps,check_nonnegative=s.check_nonnegative,
        negative_tolerance=s.negative_tolerance,control=s.control,backend_options=s.options)
    PBMSolution(ts,states,pb,steps,merge((stopping_reason=:requested_outputs_complete,),info,(settings=settings,)))
end

struct _TrialRejected <: Exception
    reason::String
end
function _trial_check(pb,u,t,s)
    try
        _check_problem_state(pb,u,s.check_nonnegative ? s.negative_tolerance : Inf)
        _check_admissible_state(pb.operator,u)
    catch e
        e isa NumericalFailure || rethrow()
        throw(_TrialRejected(sprint(showerror,e)))
    end
    s.control.guard===nothing || begin
        accepted=s.control.guard(pb,u,t)
        accepted isa Bool || throw(ArgumentError("StepControl guard must return Bool"))
        accepted || throw(_TrialRejected("declared stage guard"))
    end
    nothing
end
_contains_dual(x::ForwardDiff.Dual)=true
_contains_dual(x::Union{Tuple,NamedTuple,AbstractArray})=any(_contains_dual,x)
_contains_dual(x)=false
function _controlled_forward(pb,s,ts)
    _control_eligible(pb.operator) || throw(ArgumentError("StepControl requires an eligible residual operator without moment boundary events; complete updates and MeasureEuler retain their fixed-step contracts"))
    _contains_dual(pb.u0) || _contains_dual(pb.p) ? throw(ArgumentError(
        "StepControl has a state/parameter-dependent schedule; use fixed-step ForwardDiff or explicitly selected finite differences")) : nothing
    c=s.control
    u=copy(pb.u0); trial=similar(u);cache=ntuple(_->similar(u),6)
    template=deepcopy(pb.operator)
    # Invalid accepted starting states are not retryable trials.
    _check_problem_state(pb,u,s.negative_tolerance);_check_admissible_state(pb.operator,u)
    if c.guard!==nothing
        valid=c.guard(pb,u,first(pb.tspan))
        valid isa Bool || throw(ArgumentError("StepControl guard must return Bool"))
        valid || throw(ArgumentError("initial state violates the declared guard"))
    end
    t=first(pb.tspan);accepted=0;rejected=0;reasons=String[];states=typeof(u)[]
    for target in ts
        while t<target
            accepted+rejected<s.maxsteps || throw(NumericalFailure("forward trial budget exhausted (accepted=$accepted, rejected=$rejected)"))
            h=min(s.dt,target-t)
            if c.cfl!==nothing
                recommendation=suggest_dt(pb.operator,u,pb.p,t;cfl=c.cfl)
                recommendation isa Real && !isnan(recommendation) && recommendation>0 ||
                    throw(NumericalFailure("current coefficients did not produce a positive explicit step recommendation"))
                h=min(h,recommendation)
            end
            if c.bound!==nothing
                b=c.bound(pb,u,t)
                b isa Real && !isnan(b) && b>0 || throw(ArgumentError("declared step bound must be positive (Inf is allowed)"))
                h=min(h,b)
            end
            h isa ForwardDiff.Dual && throw(ArgumentError("differentiating a controlled step schedule is unsupported; select fixed steps or finite differences"))
            # Avoid leaving an unresolvable final sliver when dt and an output
            # differ by roundoff. Split the remainder; never exceed the bound
            # or snap the accepted state to a different integration time.
            0<target-t-h<c.min_dt && (h=min(h,(target-t)/2))
            retries=0
            while true
                h>=c.min_dt && t+h>t || throw(NumericalFailure("controlled step below min_dt; last rejection: $(isempty(reasons) ? "none" : last(reasons))"))
                accepted+rejected<s.maxsteps || throw(NumericalFailure("forward trial budget exhausted"))
                copyto!(trial,u)
                try
                    _rk_step!(trial,cache,pb,t,h,s.algorithm,s.check_nonnegative,s.negative_tolerance,
                        (v,time)->_trial_check(pb,v,time,s))
                    _trial_check(pb,trial,t+h,s)
                    copyto!(u,trial);accepted+=1
                    t=h==target-t ? target : t+h
                    break
                catch e
                    e isa _TrialRejected || rethrow()
                    rejected+=1;retries+=1;push!(reasons,e.reason)
                    retries<=c.max_rejections || throw(NumericalFailure("controlled step exhausted max_rejections: $(e.reason)"))
                    h*=c.shrink
                    # A term may use mutable scratch; discard trial working
                    # storage as well as trial state before attempting again.
                    pb=PBMProblem(prepare_run(template,pb.p),u,pb.tspan,pb.p)
                end
            end
        end
        push!(states,copy(u))
    end
    PBMSolution(ts,states,pb,accepted,(policy=Symbol(nameof(typeof(s.algorithm))),
        control=:stability_and_admissibility,accepted_steps=accepted,rejected_steps=rejected,
        rejection_reasons=reasons,stopping_reason=:requested_outputs_complete))
end
_control_eligible(op)=false
_control_eligible(op::Union{CompiledPBM,CompiledPBM2D,CompiledMomentPBM,CompiledSystem,CompiledAxial,CompiledAxialSystem})=!_moment_boundary_events(op)

function _integration_counts(sol::PBMSolution)
    info=diagnostics(sol)
    (accepted_steps=get(info,:accepted_steps,sol.steps),
        rejected_steps=get(info,:rejected_steps,missing),
        segments=haskey(info,:segments) ? length(info.segments) : 1,
        policy=get(info,:policy,:unspecified))
end
