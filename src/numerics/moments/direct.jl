"""
    DQMOM(nodes; condition_limit=1e12)

One-coordinate direct quadrature method of moments, evolving weights w and
weighted abscissas q=w*x. Uses a scaled confluent Vandermonde solve to match
moments 0:2nodes-1. Sources, growth, removal and collisions use the same weak
forms as QMOM. Positive weights and distinct nonnegative nodes are required;
empty populations and node collisions are rejected rather than regularized.
The method is for batch/coupled local balances, not spatial DQMOM transport.
"""
struct DQMOM <: AbstractDirectMomentMethod
    nodes::Int
    condition_limit::Float64
    function DQMOM(n::Integer;condition_limit=1e12)
        n>=1 && isfinite(condition_limit) && condition_limit>1 || throw(ArgumentError("invalid DQMOM node count or condition limit"))
        new(n,condition_limit)
    end
end
function validate_moment_method(m::DQMOM,model,g::MomentSet)
    length(g)==2m.nodes || throw(ArgumentError("DQMOM needs a MomentSet with orders 0:2nodes-1"))
end
moment_state_length(m::DQMOM,g)=2m.nodes
"""Pack positive quadrature weights and weighted abscissas as a DQMOM initial state. The state is NOT a vector of raw moments."""
function quadrature_state(m::DQMOM,nodes,weights)
    length(nodes)==length(weights)==m.nodes || throw(DimensionMismatch("one node and weight per DQMOM component required"))
    state=vcat(collect(weights),weights.*nodes)
    validate_moment_state(m,MomentSet(0:2m.nodes-1),state)
    state
end
function quadrature(g::MomentSet,u,m::DQMOM)
    length(u)==2m.nodes || throw(DimensionMismatch("DQMOM state length mismatch"))
    w=view(vec(u),1:m.nodes)
    all(isfinite,u) && all(>(0),w) || throw(NumericalFailure("DQMOM weights must be finite and strictly positive; empty/rank-changing states need another moment family"))
    nodes=collect(view(vec(u),m.nodes+1:2m.nodes)./w)
    all(isfinite,nodes) && all(>=(0),nodes) || throw(NumericalFailure("DQMOM nodes must be finite and nonnegative"))
    (;nodes,weights=collect(w))
end
function _direct_matrix(m,g,u)
    q=quadrature(g,u,m)
    all(isfinite,u) && all(>=(0),q.nodes) || throw(NumericalFailure("invalid DQMOM state"))
    scale=sum(q.weights.*q.nodes)/sum(q.weights)
    iszero(scale) && (scale=one(scale))
    x=q.nodes./scale;n=m.nodes
    A=[i<=n ? (1-k)*x[i]^k : (k==0 ? zero(eltype(x)) : k*x[i-n]^(k-1)) for k in 0:2n-1,i in 1:2n]
    singular=svdvals(_primal.(A))
    minimum(singular)>0 && maximum(singular)/minimum(singular)<=m.condition_limit ||
        throw(NumericalFailure("DQMOM nodes collide or its scaled source system is ill-conditioned; no regularization was applied"))
    (;q.nodes,q.weights,scale,factor=lu(A))
end
function validate_moment_state(m::DQMOM,g,u)
    _direct_matrix(m,g,u)
    nothing
end
function check_moment_state(m::DQMOM,g,u,tolerance)
    _direct_matrix(m,g,u)
    nothing
end
function moment_value(m::DQMOM,g,u,k)
    k in g.orders || throw(ArgumentError("moment $k is not tracked"))
    k==0 && return sum(view(vec(u),1:m.nodes))
    k==1 && return sum(view(vec(u),m.nodes+1:2m.nodes))
    q=quadrature(g,u,m)
    sum(q.weights.*q.nodes.^k)
end
function moment_rate(m::DQMOM,g,u,du,k)
    k in g.orders || throw(ArgumentError("moment $k is not tracked"))
    q=quadrature(g,u,m);n=m.nodes
    sum((1-k)*q.nodes[i]^k*du[i]+(k==0 ? zero(eltype(du)) : k*q.nodes[i]^(k-1)*du[n+i]) for i in 1:n)
end
prepare_moment_state(m::DQMOM,g,u,p,t,ctx)=_direct_matrix(m,g,u)
_prepare_checked_moment_state(m::DQMOM,g,u,p,t,ctx)=prepare_moment_state(m,g,u,p,t,ctx)
function _direct_term!(du,u,term,rule,cache,m::DQMOM,q,op,p,t,ctx)
    raw=[moment_value(m,op.grid,u,k) for k in op.grid.orders];source=zero(raw)
    rule isa ExactMomentRule ? _exact_moment_term!(source,raw,term,op,p,t,ctx) :
        _quadrature_term!(source,raw,term,op,p,t,ctx,q)
    scaled=[source[k+1]/q.scale^k for k in op.grid.orders]
    rates=q.factor\scaled;n=m.nodes
    for i in 1:n
        du[i]+=rates[i]
        du[n+i]+=q.scale*rates[n+i]
    end
    nothing
end
moment_term!(du,u,term,rule::ExactMomentRule,cache,m::DQMOM,q,op,p,t,ctx)=_direct_term!(du,u,term,rule,cache,m,q,op,p,t,ctx)
moment_term!(du,u,term,rule::QuadratureMomentRule,cache,m::DQMOM,q,op,p,t,ctx)=_direct_term!(du,u,term,rule,cache,m,q,op,p,t,ctx)
