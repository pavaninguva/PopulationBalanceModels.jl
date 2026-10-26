"""Read-only views of every population, bulk state and current derivative."""
struct SystemContext{P,B,G,F,D,O}
    populations::P
    bulk::B
    grids::G
    fields::F
    derivatives::D
    operators::O
end
moment(ctx::SystemContext,name::Symbol,k)=population_moment(getproperty(ctx.operators,name),getproperty(ctx.populations,name),k)

_system_fields(p,t)=NamedTuple()
_zero_bulk(ctx,p,t)=NamedTuple()

"""
    PBMSystem(populations; bulk_rhs=(ctx,p,t)->(;), interactions=(),
              exchanges=(), fields=(p,t)->(;), nonnegative_bulk=())

A NamedTuple of separately compiled populations, shared scalar bulk states,
and simultaneous interactions. Each population may have a different Grid1D.
Bulk initial values are supplied to PBMProblem through `bulk0`.
"""
struct PBMSystem{P,B,I,E,F,N,A}
    populations::P
    bulk_rhs::B
    interactions::I
    exchanges::E
    fields::F
    nonnegative_bulk::N
    accounting::A
end
function PBMSystem(populations::NamedTuple;bulk_rhs=_zero_bulk,interactions=(),
                   exchanges=(),fields=_system_fields,nonnegative_bulk=(),accounting=())
    !isempty(populations) || throw(ArgumentError("at least one population is required"))
    any(op->op isa CompiledHOC,values(populations)) &&
        throw(ArgumentError("HOC direct transport does not support evolving bulk balances or population coupling; compile residual methods for PBMSystem"))
    all(_system_population,values(populations)) || throw(ArgumentError("compile each supported named population first"))
    for interaction in interactions
        interaction isa Union{PopulationTransfer,CrossAggregation,CustomCoupling} || throw(ArgumentError("unsupported interaction"))
        if interaction isa PopulationTransfer
            interaction.source in keys(populations) && interaction.target in keys(populations) || throw(ArgumentError("unknown transfer population"))
            all(n->getproperty(populations,n) isa CompiledPBM,(interaction.source,interaction.target)) || throw(ArgumentError("PopulationTransfer currently requires sectional populations"))
        elseif interaction isa CrossAggregation
            all(n->n in keys(populations),(interaction.left,interaction.right,interaction.product)) || throw(ArgumentError("unknown collision population"))
            all(n->getproperty(populations,n) isa CompiledPBM,(interaction.left,interaction.right,interaction.product)) || throw(ArgumentError("CrossAggregation currently requires sectional populations"))
            if interaction.coordinate_map===nothing
                all(n->getproperty(populations,n).grid.coordinate in (:volume,:mass),
                    (interaction.left,interaction.right,interaction.product)) || throw(ArgumentError("cross collisions need an additive coordinate map"))
                coords=map(n->getproperty(populations,n).grid.coordinate,(interaction.left,interaction.right,interaction.product))
                all(==(first(coords)),coords) || throw(ArgumentError("cross-collision coordinates must use a common measure"))
            end
        end
    end
    all(_system_exchange,exchanges) || throw(ArgumentError("invalid exchange"))
    !isempty(accounting) && !isempty(interactions) && throw(ArgumentError("cumulative accounting of inter-population interactions is not implemented"))
    all(a->a isa CumulativeAccounting && a.population in keys(populations),accounting) || throw(ArgumentError("invalid accounting population"))
    length(unique(a.population for a in accounting))==length(accounting) || throw(ArgumentError("duplicate population accounting"))
    for a in accounting
        _validate_accounting(a,getproperty(populations,a.population))
    end
    for exchange in exchanges
        exchange.population in keys(populations) || throw(ArgumentError("unknown exchange population"))
        op=getproperty(populations,exchange.population)
        if exchange isa ConservativeExchange && op isa CompiledPBM && (op.bc.left isa OpenBoundary || op.bc.right isa OpenBoundary)
            any(a->a.population==exchange.population,accounting) || throw(ArgumentError("open-boundary exchange requires CumulativeAccounting"))
        else
            _validate_exchange(exchange,op)
        end
    end
    PBMSystem(populations,bulk_rhs,Tuple(interactions),Tuple(exchanges),fields,Tuple(nonnegative_bulk),Tuple(accounting))
end
_system_population(op)=op isa CompiledPBM
_system_exchange(exchange)=exchange isa ConservativeExchange

_same_size(x,p,t,ctx)=x
"""A source population loses one particle; a target gains multiplicity daughters."""
struct PopulationTransfer{F,S,M,T}
    source::Symbol
    target::Symbol
    rate::F
    destination_size::S
    measure::M
    multiplicity::T
    overflow::Symbol
end
function PopulationTransfer(source::Symbol,target::Symbol,rate;
        destination_size=_same_size,multiplicity=1.0,conserved_measure=identity,overflow=:error)
    source!=target || throw(ArgumentError("transfer requires distinct populations"))
    isfinite(multiplicity) && multiplicity>0 || throw(ArgumentError("multiplicity must be positive"))
    overflow in (:error,:discard) || throw(ArgumentError("invalid transfer overflow policy"))
    PopulationTransfer(source,target,rate,destination_size,conserved_measure,multiplicity,overflow)
end

"""Unlike-population binary collisions: A+B -> C. Each ordered type pair occurs once."""
struct CrossAggregation{F,C}
    left::Symbol
    right::Symbol
    product::Symbol
    kernel::F
    coordinate_map::C
    overflow::Symbol
end
function CrossAggregation(left::Symbol,right::Symbol,product::Symbol,kernel;
                           coordinate_map=nothing,overflow=:error)
    left!=right || throw(ArgumentError("use Aggregation for identical populations and its 1/2 factor"))
    coordinate_map===nothing || coordinate_map isa AdditiveCoordinate || throw(ArgumentError("invalid additive map"))
    overflow in (:error,:discard) || throw(ArgumentError("invalid collision overflow policy"))
    CrossAggregation(left,right,product,kernel,coordinate_map,overflow)
end

"""
Balance a named bulk concentration against the discrete resolved measure change
from selected population mechanisms. weight(size,p,t,local_context) is mass (or
another exchanged quantity) per particle. This ensures discrete conservation;
it is not an independently discretized continuum moment closure.
The default mechanisms are `(:growth,:size_diffusion,:birth,:nucleation)`;
diffusion changes nonlinear material moments even when it preserves number.
Explicit selections are retained. Do not also subtract these rates in bulk_rhs.
"""
struct ConservativeExchange{F,M}
    population::Symbol
    bulk::Symbol
    weight::F
    mechanisms::M
end
function ConservativeExchange(population::Symbol,bulk::Symbol,weight;
    mechanisms=(:growth,:size_diffusion,:birth,:nucleation))
    all(x->x in (:growth,:size_diffusion,:birth,:nucleation,:aggregation,:breakage,:death,:dilution,:custom),mechanisms) ||
        throw(ArgumentError("unknown exchange mechanism"))
    ConservativeExchange(population,bulk,weight,Tuple(mechanisms))
end

function _validate_exchange(exchange::ConservativeExchange,op)
    op isa CompiledPBM || throw(ArgumentError("use MomentExchange with a moment population"))
    op.bc.left isa OpenBoundary || op.bc.right isa OpenBoundary ?
        throw(ArgumentError("resolved-moment exchange requires closed size boundaries; open-boundary reservoirs need explicit treatment")) : nothing
end

"""f!(population_derivative_views,ctx,p,t) adds interactions and returns bulk contributions."""
struct CustomCoupling{F}
    f!::F
end

struct CompiledSystem{S,R,B,A}
    system::S
    ranges::R
    bulk_indices::B
    length::Int
    physical_length::Int
    ledger_indices::A
end

"""
    suggest_dt(operator::CompiledSystem, u, p, t; cfl=0.5)

Minimum local sectional recommendation, with each population's growth,
SizeDiffusion and loss rates summed using the current shared bulk state.
Requires sectional populations without inter-population interactions. It does
not bound bulk depletion or future coefficient changes; native stage checks
still apply. Moment populations and custom interaction bounds are unsupported.
"""
function suggest_dt(op::CompiledSystem,u,p,t; cfl=0.5)
    0<cfl<=1 || throw(ArgumentError("cfl must be in (0,1]"))
    length(u)==op.length || throw(DimensionMismatch("packed system state size mismatch"))
    isempty(op.system.interactions) && all(pop->pop isa CompiledPBM,values(op.system.populations)) ||
        throw(ArgumentError("suggest_dt for PBMSystem requires sectional populations without interactions; supply a separately verified step bound otherwise"))
    ctx=_system_context(op,u,zero(u),p,t)
    minimum(suggest_dt(getproperty(op.system.populations,name),
        reshape(population(op,u,name),:,1),p,t;cfl,context=ctx) for name in keys(op.ranges))
end

function _pack_system(system::PBMSystem,u0::NamedTuple,bulk0::NamedTuple)
    Set(keys(u0))==Set(keys(system.populations)) || throw(ArgumentError("initial population names must match the system"))
    bulk0 isa NamedTuple && all(x->x isa Real && isfinite(x),values(bulk0)) || throw(ArgumentError("bulk0 must contain finite named scalar values"))
    for name in system.nonnegative_bulk
        name in keys(bulk0) || throw(ArgumentError("unknown nonnegative bulk state"))
        _primal(getproperty(bulk0,name))>=0 || throw(ArgumentError("negative initial bulk state"))
    end
    all(e->e.bulk in keys(bulk0),system.exchanges) || throw(ArgumentError("unknown bulk exchange target"))
    offset=0
    initial=Tuple(getproperty(u0,name) for name in keys(system.populations))
    ranges=map(values(system.populations),initial) do op,N
        length(N)==population_state_length(op) || throw(DimensionMismatch("population initial size mismatch"))
        all(isfinite,N) && all(x->_primal(x)>=0,N) || throw(ArgumentError("initial populations must be finite and nonnegative"))
        r=offset+1:offset+length(N); offset+=length(N); r
    end
    namedranges=NamedTuple{keys(system.populations)}(ranges)
    indices=NamedTuple{keys(bulk0)}(Tuple(offset+j for j in eachindex(values(bulk0))))
    physical=offset+length(bulk0)
    ledgers,total=_ledger_indices(system.accounting,physical)
    op=CompiledSystem(system,namedranges,indices,total,physical,ledgers)
    flat=float.(vcat((vec(N) for N in initial)...,collect(values(bulk0)),zeros(total-physical)))
    op,flat
end

function PBMProblem(system::PBMSystem,u0::NamedTuple,tspan;bulk0=NamedTuple(),p=NamedTuple())
    bulk0 isa NamedTuple || throw(ArgumentError("bulk0 must contain finite named scalar values"))
    op,flat=_pack_system(system,u0,bulk0)
    PBMProblem(op,flat,tspan;p)
end

function PBMProblem(op::CompiledSystem,u0::AbstractVector,tspan;p=NamedTuple())
    length(u0)==op.length || throw(DimensionMismatch("packed system state size mismatch"))
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] || throw(ArgumentError("invalid time span"))
    N=float.(copy(u0)); all(isfinite,N) || throw(ArgumentError("initial system state is nonfinite"))
    for r in values(op.ranges)
        all(x->_primal(x)>=0,view(N,r)) || throw(ArgumentError("initial population is negative"))
    end
    for name in keys(op.ranges)
        _validate_population_state(getproperty(op.system.populations,name),view(N,getproperty(op.ranges,name)))
    end
    all(x->_primal(x)>=0,view(N,op.physical_length+1:op.length)) || throw(ArgumentError("negative initial cumulative state"))
    for name in op.system.nonnegative_bulk
        _primal(N[getproperty(op.bulk_indices,name)])>=0 || throw(ArgumentError("initial bulk state is negative"))
    end
    PBMProblem(op,N,(float(tspan[1]),float(tspan[2])),p)
end
_validate_population_state(op,N)=nothing

population(op::CompiledSystem,u,name::Symbol)=view(u,getproperty(op.ranges,name))
population(prob::PBMProblem,u,name::Symbol)=population(prob.operator,u,name)
bulk(op::CompiledSystem,u,name::Symbol)=u[getproperty(op.bulk_indices,name)]
bulk(prob::PBMProblem,u,name::Symbol)=bulk(prob.operator,u,name)
function stateview(op::CompiledSystem,u)
    pops=NamedTuple{keys(op.ranges)}(Tuple(view(u,r) for r in values(op.ranges)))
    b=NamedTuple{keys(op.bulk_indices)}(Tuple(u[i] for i in values(op.bulk_indices)))
    (;populations=pops,bulk=b)
end
stateview(prob::PBMProblem,u)=stateview(prob.operator,u)
function _system_context(op,u,du,p,t)
    state=stateview(op,u)
    grids=map(x->x.grid,op.system.populations)
    derivatives=NamedTuple{keys(op.ranges)}(Tuple(view(du,r) for r in values(op.ranges)))
    SystemContext(state.populations,state.bulk,grids,op.system.fields(p,t),derivatives,op.system.populations)
end

function _add_bulk!(du,op,contribution)
    contribution===nothing && return
    contribution isa NamedTuple || throw(ArgumentError("bulk dynamics must return a NamedTuple"))
    for name in keys(contribution)
        name in keys(op.bulk_indices) || throw(ArgumentError("bulk dynamics returned unknown state $name"))
        value=getproperty(contribution,name)
        value isa Real && isfinite(value) || throw(NumericalFailure("nonfinite bulk derivative"))
        du[getproperty(op.bulk_indices,name)]+=value
    end
end

function _interaction!(term::PopulationTransfer,ctx,op,du,p,t)
    from=getproperty(ctx.populations,term.source); to=getproperty(ctx.derivatives,term.target)
    dfrom=getproperty(ctx.derivatives,term.source)
    sourcegrid=getproperty(ctx.grids,term.source); targetgrid=getproperty(ctx.grids,term.target)
    axis=term.measure.(targetgrid.centers)
    all(diff(axis).>0) || throw(ArgumentError("transfer measure must increase with target size"))
    for i in eachindex(from)
        iszero(from[i]) && continue
        x=sourcegrid.centers[i]; y=term.destination_size(x,p,t,ctx)
        isfinite(y) && y>0 || throw(ArgumentError("invalid destination size"))
        isapprox(term.measure(x),term.multiplicity*term.measure(y);rtol=1e-9,atol=0) ||
            throw(ArgumentError("population transfer does not conserve its declared measure"))
        rate=_nonnegative(term.rate(x,p,t,ctx),"transfer hazard")*from[i]
        dfrom[i]-=rate
        _deposit!(to,axis,term.measure(y),term.multiplicity*rate,term.overflow)
    end
end
function _interaction!(term::CrossAggregation,ctx,op,du,p,t)
    A=getproperty(ctx.populations,term.left); B=getproperty(ctx.populations,term.right)
    dA=getproperty(ctx.derivatives,term.left); dB=getproperty(ctx.derivatives,term.right)
    dC=getproperty(ctx.derivatives,term.product)
    x=getproperty(ctx.grids,term.left).centers; y=getproperty(ctx.grids,term.right).centers
    z=getproperty(ctx.grids,term.product).centers
    measure=term.coordinate_map===nothing ? identity : term.coordinate_map.measure
    axis=measure.(z)
    all(isfinite,axis) && all(diff(axis).>0) || throw(ArgumentError("invalid cross-collision additive map"))
    for j in eachindex(B), i in eachindex(A)
        (iszero(A[i]) || iszero(B[j])) && continue
        rate=_nonnegative(term.kernel(x[i],y[j],p,t,ctx),"cross-collision kernel")*A[i]*B[j]
        dA[i]-=rate; dB[j]-=rate
        _deposit!(dC,axis,measure(x[i])+measure(y[j]),rate,term.overflow)
    end
end
_interaction!(term::CustomCoupling,ctx,op,du,p,t)=_add_bulk!(du,op,term.f!(ctx.derivatives,ctx,p,t))

function _exchange!(exchange,ctx,op,du,p,t)
    popop=getproperty(op.system.populations,exchange.population)
    N=getproperty(ctx.populations,exchange.population)
    scratch=similar(N); fill!(scratch,zero(eltype(N)))
    localctx=_context(popop,reshape(N,:,1),1,t,ctx)
    for item in popop.prepared
        if mechanism(item.term) in exchange.mechanisms
            _add_prepared!(scratch,N,item,popop,p,t,localctx)
        end
    end
    change=sum(_nonnegative(exchange.weight(x,p,t,localctx),"per-particle exchanged amount")*scratch[i]
        for (i,x) in enumerate(popop.grid.centers))
    losses=_sectional_losses(popop,N,p,t,localctx,exchange.mechanisms)
    # A discarded particle remains particulate; it is not dissolved material,
    # whether or not the optional historical ledger is requested.
    change+=sum((exchange.weight(x,p,t,localctx)*r for (_,x,r) in losses);init=zero(change))
    if haskey(op.ledger_indices,exchange.population)
        _validate_exchange_property(exchange,op,popop,p,t,localctx)
    end
    du[getproperty(op.bulk_indices,exchange.bulk)]-=change
end

function rhs!(du::AbstractVector,u::AbstractVector,op::CompiledSystem,p,t)
    length(du)==length(u)==op.length || throw(DimensionMismatch("system state size mismatch"))
    fill!(du,zero(eltype(du)))
    ctx=_system_context(op,u,du,p,t)
    for name in keys(op.ranges)
        popop=getproperty(op.system.populations,name)
        rhs!(reshape(getproperty(ctx.derivatives,name),:,1),reshape(getproperty(ctx.populations,name),:,1),popop,p,t;context=ctx)
    end
    for interaction in op.system.interactions
        _interaction!(interaction,ctx,op,du,p,t)
    end
    _add_bulk!(du,op,op.system.bulk_rhs(ctx,p,t))
    for exchange in op.system.exchanges
        _exchange!(exchange,ctx,op,du,p,t)
    end
    for a in op.system.accounting
        popop=getproperty(op.system.populations,a.population)
        popop isa CompiledPBM || continue # moment ledgers are advanced by MeasureEuler
        N=getproperty(ctx.populations,a.population)
        localctx=_context(popop,reshape(N,:,1),1,t,ctx)
        for (mechanism,x,rate) in _sectional_losses(popop,N,p,t,localctx)
            _record_loss!(du,op,a.population,mechanism,x,rate)
        end
    end
    all(isfinite,du) || throw(NumericalFailure("nonfinite coupled-system derivative"))
    nothing
end

function _check_problem_state(prob::PBMProblem{O},u,tolerance) where {O<:CompiledSystem}
    all(isfinite,u) || throw(NumericalFailure("nonfinite coupled-system state"))
    for r in values(prob.operator.ranges)
        minimum(view(u,r))>=-tolerance || throw(NumericalFailure("negative population in coupled solve; reduce dt or use a suitable solver"))
    end
    for name in keys(prob.operator.ranges)
        _check_population_state(getproperty(prob.operator.system.populations,name),view(u,getproperty(prob.operator.ranges,name)),tolerance)
    end
    for name in prob.operator.system.nonnegative_bulk
        bulk(prob,u,name)>=-tolerance || throw(NumericalFailure("negative bulk concentration in coupled solve"))
    end
    nothing
end
_check_population_state(op,N,tolerance)=nothing
function _check_admissible_state(op::CompiledSystem,u)
    for name in keys(op.ranges)
        _check_admissible_state(getproperty(op.system.populations,name),population(op,u,name))
    end
    nothing
end
