"""
    OperatingHistory(transitions; parameters=..., fields=...)

Prescribed, parameter-independent transition times with continuous state transfer.
Supply one entry per interval (length(transitions)+1). `parameters` entries are
NamedTuple overrides or `p -> overrides`; `fields` entries are NamedTuples or
`(p,t) -> fields`. Omitted entries are empty NamedTuples. Times must be strictly
increasing and interior to the problem span. They are not fitted events.

Every integration evaluation in an interval, including its endpoint RK stages,
uses that interval's conditions. At an exact transition, observations and
`operating_inputs` use the right interval by default. External solvers restart
at each transition. No state resets or state-triggered events are supported.
"""
struct OperatingHistory{T,P,F}
    transitions::T
    parameters::P
    fields::F
end
struct _HeldOperatingInputs{H}
    history::H
    interval::Int
end
_unsegmented_history(pb)=pb.history isa OperatingHistory
function OperatingHistory(transitions;parameters=nothing,fields=nothing)
    ts=collect(transitions)
    all(x->x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x),ts) && all(diff(ts).>0) ||
        throw(ArgumentError("operating transitions must be finite, fixed, strictly increasing times"))
    n=length(ts)+1
    ps=parameters===nothing ? ntuple(_->(;),n) : Tuple(parameters)
    fs=fields===nothing ? ntuple(_->(;),n) : Tuple(fields)
    length(ps)==length(fs)==n || throw(DimensionMismatch("one operating-parameter and field entry per interval required"))
    OperatingHistory(ts,ps,fs)
end
_validate_history(::Nothing,ts,p)=nothing
function _validate_history(h::OperatingHistory,ts,p)
    all(t->first(ts)<t<last(ts),h.transitions) || throw(ArgumentError("operating transitions must lie strictly inside tspan"))
    for i in eachindex(h.parameters)
        _interval_parameters(h,p,i)
    end
    nothing
end
_validate_history(h,ts,p)=throw(ArgumentError("history must be OperatingHistory or nothing"))
function _interval_parameters(h,p,i)
    entry=h.parameters[i]
    overrides=entry isa NamedTuple ? entry : entry(p)
    overrides isa NamedTuple || throw(ArgumentError("operating parameter declaration must return NamedTuple overrides"))
    isempty(overrides) && return p
    p isa NamedTuple || throw(ArgumentError("operating parameter overrides require a NamedTuple parameter container; use fields otherwise"))
    all(k->k in keys(p),keys(overrides)) || throw(ArgumentError("operating overrides must refer to declared problem parameters"))
    merge(p,overrides)
end
function _interval_fields(h,p,t,i)
    entry=h.fields[i]
    fields=entry isa NamedTuple ? entry : entry(p,t)
    fields isa NamedTuple || throw(ArgumentError("operating fields must return a NamedTuple"))
    fields
end

"""
    operating_inputs(problem, t; side=:right)

Return `(parameters, fields, interval)` for the prescribed history at time t.
The default chooses new conditions at a transition; `side=:left` inspects the
previous interval. Rate callbacks receive these parameters and fields during
integration; use this accessor for matching condition-dependent postprocessing.
"""
function operating_inputs(pb::PBMProblem,t;side=:right)
    side in (:left,:right) || throw(ArgumentError("side must be :left or :right"))
    first(pb.tspan)<=t<=last(pb.tspan) || throw(ArgumentError("operating-input time outside problem span"))
    h=pb.history
    h===nothing && return (parameters=pb.p,fields=(;),interval=1)
    h isa _HeldOperatingInputs && return (parameters=pb.p,
        fields=_interval_fields(h.history,pb.p,t,h.interval),interval=h.interval)
    i=side===:right ? searchsortedlast(h.transitions,t)+1 : searchsortedfirst(h.transitions,t)
    p=_interval_parameters(h,pb.p,i)
    (parameters=p,fields=_interval_fields(h,p,t,i),interval=i)
end
function _operating_operator(op::CompiledPBM,p,h,i)
    f=(cell,t)->merge(op.fields(cell,t),_interval_fields(h,p,t,i))
    CompiledPBM(op.model,op.grid,op.scheme,op.bc,op.domain,f,op.prepared)
end
function _operating_operator(op::CompiledPBM2D,p,h,i)
    f=(cell,t)->merge(op.fields(cell,t),_interval_fields(h,p,t,i))
    CompiledPBM2D(op.model,op.grid,op.scheme,op.bc,f,op.prepared)
end
function _operating_operator(op::CompiledMomentPBM,p,h,i)
    f=(cell,t)->merge(op.fields(cell,t),_interval_fields(h,p,t,i))
    CompiledMomentPBM(op.model,op.grid,op.closure,f,op.prepared,op.boundary)
end
function _operating_operator(op::CompiledHOC,p,h,i)
    f=(cell,t)->merge(op.fields(cell,t),_interval_fields(h,p,t,i))
    CompiledHOC(op.model,op.grid,op.bc,f,op.rates,op.evolution,op.prepared)
end
function _operating_operator(op::Union{CompiledCharacteristic,CompiledConstantTransport},p,h,i)
    h.fields[i] isa NamedTuple && isempty(h.fields[i]) || throw(ArgumentError("this complete update has no operating-field interface"))
    op
end
function _operating_operator(op::CompiledSystem,p,h,i)
    s=op.system
    populations=map(pop->_operating_operator(pop,p,h,i),s.populations)
    system=PBMSystem(populations;bulk_rhs=s.bulk_rhs,interactions=s.interactions,
        exchanges=s.exchanges,fields=(p,t)->merge(s.fields(p,t),_interval_fields(h,p,t,i)),
        nonnegative_bulk=s.nonnegative_bulk,accounting=s.accounting)
    CompiledSystem(system,op.ranges,op.bulk_indices,op.length,op.physical_length,op.ledger_indices)
end
function _operating_operator(op::CompiledAxial,p,h,i)
    locals=map(pop->_operating_operator(pop,p,h,i),op.local_operators)
    CompiledAxial(op.grid,op.domain,locals,first(locals).prepared,op.scheme,op.reconstruction,op.prepared_inlet)
end
_operating_operator(op::CompiledAxialSystem,p,h,i)=CompiledAxialSystem(op.domain,
    map(pop->_operating_operator(pop,p,h,i),op.local_operators),op.scheme,op.reconstruction)
_operating_operator(op,p,h,i)=throw(ArgumentError("OperatingHistory is not implemented for $(typeof(op)); extend the supported operator preparation interface"))

_observation_inputs(pb,t)=(pb.operator,pb.p)
function _observation_inputs(pb::PBMProblem,t)
    pb.history===nothing && return (pb.operator,pb.p)
    pb.history isa _HeldOperatingInputs && return (pb.operator,pb.p)
    inputs=operating_inputs(pb,t)
    op=_operating_operator(pb.operator,inputs.parameters,pb.history,inputs.interval)
    (prepare_run(op,inputs.parameters),inputs.parameters)
end

function _segmented_forward(pb,s,ts)
    # Continuous sensitivity jumps require additional mathematics, not just stops.
    s.backend===:sensitivity && throw(ArgumentError("OperatingHistory is unsupported by continuous sensitivities; use native fixed-schedule AD or explicit finite differences"))
    boundaries=vcat(first(pb.tspan),filter(t->t<last(ts),pb.history.transitions),last(ts))
    last(ts)==first(pb.tspan) && return PBMSolution(ts,[copy(pb.u0)],pb,0,
        (policy=:segmented,accepted_steps=0,rejected_steps=0,segments=NamedTuple[],interval_convention=:right_observations))
    u=_execution_initial(pb);states=typeof(u)[];reports=NamedTuple[];accepted=0;rejected=0
    for k in 1:length(boundaries)-1
        a,b=boundaries[k:k+1]
        i=searchsortedlast(pb.history.transitions,a)+1
        p=_interval_parameters(pb.history,pb.p,i)
        op=prepare_run(_operating_operator(pb.operator,p,pb.history,i),p)
        # Use validated constructors: complete-update parameter/mesh restrictions
        # must be rechecked at an operating change, not bypassed by an inner call.
        segment=PBMProblem(op,u,(a,b);p)
        segment=PBMProblem(segment.operator,segment.u0,segment.tspan,segment.p,nothing,
            _HeldOperatingInputs(pb.history,i))
        requested=filter(t->(k==1 ? a<=t : a<t) && t<=b,ts)
        save=sort!(unique(vcat(requested,b)))
        remaining=s.maxsteps-accepted-rejected
        remaining>0 || throw(NumericalFailure("segmented forward trial budget exhausted"))
        local_solver=ForwardSolver(s.algorithm,s.dt,s.backend,s.control,s.check_nonnegative,
            s.negative_tolerance,remaining,s.options)
        sol=_execute_backend(segment,local_solver,save)
        for t in requested
            push!(states,copy(sol.u[searchsortedfirst(sol.t,t)]))
        end
        u=copy(last(sol.u));accepted+=sol.steps
        rejected+=get(sol.integration_diagnostics,:rejected_steps,0)
        push!(reports,(interval=i,tspan=(a,b),diagnostics=sol.integration_diagnostics))
    end
    _checked_solution(pb,ts,states,accepted,(policy=:segmented,accepted_steps=accepted,
        rejected_steps=rejected,segments=reports,interval_convention=:right_observations,
        stopping_reason=:requested_outputs_complete),s,ts)
end
