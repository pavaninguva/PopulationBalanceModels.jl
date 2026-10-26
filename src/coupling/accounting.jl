function _ledger_indices(specs,offset)
    entries=map(specs) do a
        names=(:number,keys(a.properties)...)
        NamedTuple{_LOSS_MECHANISMS}(Tuple(NamedTuple{names}(Tuple(begin
            offset+=1;offset
        end for _ in names)) for _ in _LOSS_MECHANISMS))
    end
    NamedTuple{Tuple(a.population for a in specs)}(Tuple(entries)),offset
end
function _validate_accounting(a,op)
    op isa Union{CompiledPBM,CompiledMomentPBM} || throw(ArgumentError("accounting requires a 1D sectional or raw-moment population"))
    if op isa CompiledMomentPBM
        op.closure isa AbstractMomentClosure || throw(ArgumentError("DQMOM boundary accounting is unavailable"))
        all(k->k in op.grid.orders,values(a.properties)) || throw(ArgumentError("accounted material moments must be tracked, not reconstructed"))
    end
    for term in op.model.terms
        term isa Union{Aggregation,Breakage} || continue
        for k in values(a.properties)
            # Fixed pivot conserves its additive coordinate, not arbitrary powers.
            measure=x->_measure(term,x)
            scale=measure(1.0)
            all(x->isapprox(measure(x),scale*x^k;rtol=1e-9,atol=0),(.3,.7,1.3,2.1)) ||
                throw(ArgumentError("collision accounting property x^$k must match the additive coordinate"))
        end
    end
end

"""
    cumulative(problem_or_operator, state, population, mechanism, property=:number)

Historical unresolved number or declared property, by named population and removal
mechanism. `state` may be a physical state or a derivative (instantaneous rate).
Use `accounting_rates(problem,state,p,t)` for a named tree of rate values.
For axial systems return one history per reactor cell; these histories do not flow.
"""
cumulative(prob::PBMProblem,u,population::Symbol,mechanism::Symbol,property::Symbol=:number)=
    cumulative(prob.operator,u,population,mechanism,property)
function _ledger_index(op,population,mechanism,property)
    haskey(op.ledger_indices,population) || throw(ArgumentError("no cumulative accounting for population :$population; available populations: $(keys(op.ledger_indices))"))
    entries=getproperty(op.ledger_indices,population)
    haskey(entries,mechanism) || throw(ArgumentError("unknown cumulative mechanism :$mechanism; available mechanisms: $(keys(entries))"))
    properties=getproperty(entries,mechanism)
    haskey(properties,property) || throw(ArgumentError("undeclared cumulative property :$property; available properties: $(keys(properties))"))
    getproperty(properties,property)
end
function cumulative(op::CompiledSystem,u,population::Symbol,mechanism::Symbol,property::Symbol=:number)
    i=_ledger_index(op,population,mechanism,property)
    u[i]
end
function cumulative(op::CompiledAxialSystem,u,population::Symbol,mechanism::Symbol,property::Symbol=:number)
    s=first(op.local_operators)
    i=_ledger_index(s,population,mechanism,property)
    view(u,i,:)
end
function _record_loss!(u,op,population,mechanism,x,amount)
    haskey(op.ledger_indices,population) || return
    a=only(filter(a->a.population==population,op.system.accounting))
    indices=getproperty(getproperty(op.ledger_indices,population),mechanism)
    u[indices.number]+=amount
    for property in keys(a.properties)
        u[getproperty(indices,property)]+=amount*x^getproperty(a.properties,property)
    end
end

# The same upwind flux and fixed-pivot deposition routines used by rhs! generate
# the loss ledger. Callbacks are required to be pure, as for exchange residuals.
function _sectional_losses(op,N,p,t,ctx,mechanisms=nothing)
    losses=Tuple[]
    for item in op.prepared
        term=_effective_term(item)
        mechanisms!==nothing && !(mechanism(term) in mechanisms) && continue
        if term isa Growth
            # The wrapped face transfers particles internally, not to a reservoir.
            op.bc.left isa Periodic && continue
            F=_growth_fluxes(term,N,(;grid=op.grid,bc=op.bc,scheme=_prepared_reconstruction(item.method,item.cache)),p,t,ctx)
            if F[1]<0
                absorb=op.bc.left isa AbsorbingZero
                push!(losses,(absorb ? :extinction : :lower_escape,absorb ? 0.0 : first(op.grid.centers),-F[1]))
            end
            F[end]>0 && push!(losses,(:upper_escape,last(op.grid.centers),F[end]))
        elseif term isa Union{Aggregation,Breakage}
            axis=_measureaxis(term,op.grid)
            callback=(v,r)->begin
                x=term.coordinate_map===nothing ? v : term.coordinate_map.inverse(v)
                side=v<first(axis) ? :underflow : :overflow
                push!(losses,(Symbol(mechanism(term),"_",side),x,r))
            end
            _local_rhs!(zero(N),N,term,op,p,t,ctx;on_escape=callback)
        elseif term isa Nucleation && item.method isa FixedPivot &&
               !(first(op.grid.centers)<=term.size<=last(op.grid.centers))
            rate=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
            overflow=item.method.overflow===:inherit ? :error : item.method.overflow
            callback=(x,r)->push!(losses,(x<first(op.grid.centers) ? :nucleation_underflow : :nucleation_overflow,x,r))
            _deposit!(zero(N),op.grid.centers,term.size,rate,overflow;on_escape=callback)
        end
    end
    losses
end
function _validate_exchange_property(exchange,op,popop,p,t,ctx)
    a=only(filter(a->a.population==exchange.population,op.system.accounting))
    x=popop.grid.centers
    any(values(a.properties)) do k
        factor=exchange.weight(first(x),p,t,ctx)/first(x)^k
        all(y->isapprox(exchange.weight(y,p,t,ctx),factor*y^k;rtol=1e-9,atol=0),x)
    end || throw(ArgumentError("accounted exchange must use a declared raw-power property"))
end
"""Instantaneous rates for the cumulative states, evaluated from the same residual.
Characteristic moment boundary rates are discrete events and are reported through
cumulative increments of MeasureEuler, rather than an invented density trace.
"""
function accounting_rates(prob::PBMProblem,u,p=prob.p,t=first(prob.tspan))
    prob.operator isa Union{CompiledSystem,CompiledAxialSystem} || throw(ArgumentError("accounting_rates requires a PBMSystem or coupled axial problem"))
    du=zero(u);rhs!(du,u,prob.operator,p,t)
    _accounting_rate_tree(prob.operator,du)
end
function _accounting_rate_tree(operator,du)
    op=operator isa CompiledAxialSystem ? first(operator.local_operators) : operator
    NamedTuple{keys(op.ledger_indices)}(Tuple(
        NamedTuple{_LOSS_MECHANISMS}(Tuple(NamedTuple{keys(ind)}(Tuple(
            operator isa CompiledAxialSystem ? copy(view(du,i,:)) : du[i] for i in values(ind)))
            for ind in values(getproperty(op.ledger_indices,name)))) for name in keys(op.ledger_indices)))
end

function _moment_boundary_events(op::CompiledMomentPBM)
    b=op.boundary
    b.left!==:error || b.right!==:error || b.lower!=0 || isfinite(b.upper)
end
_moment_boundary_events(op::CompiledSystem)=any(_moment_boundary_events,values(op.system.populations))
_moment_boundary_events(op::CompiledAxialSystem)=any(_moment_boundary_events,op.local_operators)
_moment_boundary_events(op)=false
