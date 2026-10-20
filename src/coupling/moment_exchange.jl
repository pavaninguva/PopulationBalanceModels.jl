"""Balance a bulk scalar against `factor*dμ[order]` from selected moment mechanisms.
The invariant c+factor*μ[order] assumes a factor fixed in time and state;
varying conversion factors require their additional derivative balance.
"""
struct MomentExchange{F,M}
    population::Symbol
    bulk::Symbol
    order::Int
    factor::F
    mechanisms::M
end
function MomentExchange(population::Symbol,bulk::Symbol,order::Integer;factor=1.0,mechanisms=(:growth,:birth,:nucleation))
    all(m->m in (:growth,:birth,:nucleation,:aggregation,:breakage,:death,:dilution),mechanisms) || throw(ArgumentError("invalid moment exchange mechanism"))
    MomentExchange(population,bulk,Int(order),factor,Tuple(mechanisms))
end
_system_exchange(::MomentExchange)=true
function _validate_exchange(exchange::MomentExchange,op)
    op isa CompiledMomentPBM || throw(ArgumentError("MomentExchange requires a moment population"))
    exchange.order in op.grid.orders || throw(ArgumentError("exchange moment is not tracked"))
end
function _exchange!(exchange::MomentExchange,ctx,op,du,p,t)
    popop=getproperty(op.system.populations,exchange.population); μ=getproperty(ctx.populations,exchange.population)
    localctx=_context(popop,reshape(μ,:,1),1,t,ctx); scratch=zero(μ)
    _moment_terms!(scratch,μ,popop,p,t,localctx;mechanisms=exchange.mechanisms)
    du[getproperty(op.bulk_indices,exchange.bulk)]-=_nonnegative(_source_value(exchange.factor,p,t,localctx),"moment exchange factor")*moment_rate(popop.closure,popop.grid,μ,scratch,exchange.order)
end
