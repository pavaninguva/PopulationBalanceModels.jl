_daughter_zero(d::ContinuousDaughters,k)=k==d.conserved_order

# Shared READ-ONLY geometry: no parameters, states, or working arrays are stored
# here. Prewarm the selected orders at compile; parameter runs only read them.
const _DAUGHTER_RULES=Dict{Int,NamedTuple{(:nodes,:weights),Tuple{Vector{Float64},Vector{Float64}}}}()
const _DAUGHTER_RULE_LOCK=ReentrantLock()
function _daughter_quadrature(n)
    lock(_DAUGHTER_RULE_LOCK) do
        get!(_DAUGHTER_RULES,n) do
            _gauss_legendre(GaussLegendre(n))
        end
    end
end
function prepare_term(term::Breakage{F,D},m::FixedPivot,g::Grid1D;kwargs...) where {F,D<:ContinuousDaughters}
    n=m.daughter_quadrature.order
    while n<=m.daughter_maxorder
        _daughter_quadrature(n);n*=2
    end
    m.overflow===:inherit ? term : Breakage(term.rate,term.daughters;
        overflow=m.overflow,coordinate_map=term.coordinate_map)
end
function _daughter_interval(d,parent,p)
    s=d.support isa Tuple ? d.support : d.support(parent,p)
    upper=d.coordinates===:ratio ? one(parent) : parent
    s isa Tuple && length(s)==2 && all(isfinite,s) && 0<=s[1]<s[2]<=upper ||
        throw(ArgumentError("daughter support must be a finite nonempty interval within the parent"))
    d.coordinates===:ratio ? (parent*s[1],parent*s[2]) : s
end
function _daughter_density(d,x,parent,p,t,ctx)
    ratio=d.coordinates===:ratio
    value=d.density(ratio ? x/parent : x,parent,p,t,ctx)
    value isa Real && isfinite(value) && value>=0 || throw(ArgumentError("daughter density must be finite and nonnegative"))
    ratio ? value/parent : value
end
_daughter_count(d,parent,p)=d.count isa Number ? d.count : d.count(parent,p)
function _daughter_rule(d,parent,p,t,ctx,q)
    a,b=_daughter_interval(d,parent,p)
    nodes=(a+b)/2 .+(b-a)/2 .*q.nodes
    counts=[(b-a)/2*q.weights[i]*_daughter_density(d,x,parent,p,t,ctx) for (i,x) in enumerate(nodes)]
    collect(zip(nodes,counts))
end
function _check_daughter_balances(d,parent,p,pairs;rtol,atol)
    expected=_daughter_count(d,parent,p)
    expected isa Real && isfinite(expected) && expected>0 || throw(ArgumentError("expected daughter count must be finite and positive"))
    number=sum(last,pairs)
    isapprox(number,expected;rtol,atol) || throw(ArgumentError("continuous daughter count $number differs from declared $expected; no rescaling applied"))
    if d.conserved_order!==nothing
        property=sum(x^d.conserved_order*w for (x,w) in pairs)
        isapprox(property,parent^d.conserved_order;rtol,atol) || throw(ArgumentError(
            "continuous daughter law violates its conserved power (property=$property, parent=$(parent^d.conserved_order)); no rescaling applied"))
    end
    nothing
end
function _continuous_pairs(d,parent,p,t,ctx;order=16,maxorder=256,rtol=1e-9,atol=0.0)
    n=order
    old=_daughter_rule(d,parent,p,t,ctx,_daughter_quadrature(n))
    orders=d.conserved_order===nothing ? (0,) : (0,d.conserved_order)
    previous=[sum(x^k*w for (x,w) in old) for k in orders]
    while 2n<=maxorder
        n*=2
        pairs=_daughter_rule(d,parent,p,t,ctx,_daughter_quadrature(n))
        current=[sum(x^k*w for (x,w) in pairs) for k in orders]
        if all(abs.(current.-previous).<=atol.+rtol.*abs.(current))
            _check_daughter_balances(d,parent,p,pairs;rtol,atol)
            return pairs
        end
        previous=current
    end
    throw(NumericalFailure("continuous daughter quadrature unresolved at order $maxorder; refine daughter_quadrature/maxorder"))
end
(d::ContinuousDaughters)(parent,p,t,ctx)=_continuous_pairs(d,parent,p,t,ctx)
function daughter_moment(d::ContinuousDaughters,parent,k::Integer,p,t,ctx)
    d.moment===nothing || return d.moment(parent,k,p,t,ctx)
    # Check convergence of the requested moment, not only count/conservation.
    desc=(quadrature=GaussLegendre(16),maxorder=256,rtol=1e-9,atol=0.0)
    first(_checked_initial_quadrature(desc) do q
        pairs=_daughter_rule(d,parent,p,t,ctx,q)
        [sum(x^k*w for (x,w) in pairs)]
    end)[1]
end

struct _ProjectedDaughters{X,P,L,K}
    parents::X
    pairs::P
    law::L
    parameters::K
end
function (d::_ProjectedDaughters)(parent,p,t,ctx)
    isequal(p,d.parameters) || throw(ArgumentError("daughter projection parameter binding changed; use simulate or prepare_run for this binding"))
    j=searchsortedfirst(d.parents,parent)
    j<=length(d.parents) && d.parents[j]==parent || throw(ArgumentError("daughter projection requires its compiled parent pivots"))
    d.pairs[j]
end

# Integrate on each smooth hat interval. A global daughter quadrature would
# alias the pivot hats and converge poorly even when total count is resolved.
function _project_daughters(d,term,g,parent,p,q)
    lo,hi=_daughter_interval(d,parent,p)
    knots=sort!(unique(vcat(lo,filter(x->lo<x<hi,g.centers),hi)))
    axis=_measureaxis(term,g);pairs=Tuple[]
    measure=x->_measure(term,x)
    inverse=term.coordinate_map===nothing ? identity : term.coordinate_map.inverse
    for i in 1:length(knots)-1
        a,b=knots[i:i+1]
        nodes=(a+b)/2 .+(b-a)/2 .*q.nodes
        weights=[(b-a)/2*q.weights[j]*_daughter_density(d,x,parent,p,zero(parent),nothing) for (j,x) in enumerate(nodes)]
        mid=(a+b)/2
        if mid<first(g.centers) || mid>last(g.centers)
            count=sum(weights)
            if count>0
                property=sum(weights.*measure.(nodes))
                push!(pairs,(inverse(property/count),count))
            end
        else
            j=searchsortedlast(g.centers,mid)
            j<length(g.centers) || throw(ArgumentError("invalid daughter projection interval"))
            delta=axis[j+1]-axis[j]
            left=sum(weights.*[(axis[j+1]-measure(x))/delta for x in nodes])
            right=sum(weights.*[(measure(x)-axis[j])/delta for x in nodes])
            left>=0 && right>=0 || throw(NumericalFailure("negative daughter hat integral; no clipping applied"))
            left>0 && push!(pairs,(g.centers[j],left))
            right>0 && push!(pairs,(g.centers[j+1],right))
        end
    end
    pairs
end
function prepare_run(term::Breakage,m::FixedPivot,cache::Breakage,g::Grid1D,p)
    d=cache.daughters
    d isa _ProjectedDaughters && (d=d.law)
    d isa ContinuousDaughters || return deepcopy(cache)
    if :time in d.dependencies || :state in d.dependencies
        throw(ArgumentError("continuous FixedPivot projection requires parameter-only daughter laws; state/time-dependent daughter preparation is unsupported (moment quadrature evaluates current inputs)"))
    end
    n=m.daughter_quadrature.order
    project=q->[_project_daughters(d,cache,g,x,p,q) for x in g.centers]
    old=project(_daughter_quadrature(n))
    summarize=rules->vcat((begin
        bins=zeros(promote_type(eltype(g.centers),_run_numeric_type(p)),length(g))
        low0=zero(eltype(bins));low1=low0;high0=low0;high1=low0
        axis=_measureaxis(cache,g)
        for (x,w) in pairs
            property=_measure(cache,x)
            if property<first(axis)
                low0+=w;low1+=w*property
            elseif property>last(axis)
                high0+=w;high1+=w*property
            else
                _deposit!(bins,axis,property,w,:error)
            end
        end
        vcat(bins,low0,low1,high0,high1)
    end for pairs in rules)...)
    while 2n<=m.daughter_maxorder
        n*=2;new=project(_daughter_quadrature(n))
        previous=summarize(old);current=summarize(new)
        if all(abs.(previous.-current).<=m.daughter_atol.+m.daughter_rtol.*abs.(current))
            for (x,pairs) in zip(g.centers,new)
                _check_daughter_balances(d,x,p,pairs;rtol=m.daughter_rtol,atol=m.daughter_atol)
                # Also check the actual Breakage additive map independently.
                isapprox(sum(w*_measure(cache,y) for (y,w) in pairs),_measure(cache,x);
                    rtol=m.daughter_rtol,atol=m.daughter_atol) || throw(ArgumentError("daughter conserved_order and Breakage additive property disagree"))
            end
            projected=_ProjectedDaughters(g.centers,new,d,p)
            return Breakage(cache.rate,projected;overflow=cache.overflow,coordinate_map=cache.coordinate_map)
        end
        old=new
    end
    throw(NumericalFailure("continuous daughter projection unresolved at order $(m.daughter_maxorder)"))
end

function add_term!(du,u,term::Breakage,m::FixedPivot,cache::Breakage{F,D},op::CompiledPBM,p,t,ctx) where {F,D<:ContinuousDaughters}
    prepared=prepare_run(term,m,cache,op.grid,p)
    _local_rhs!(du,u,prepared,op,p,t,ctx)
end
function _effective_run_term(item::PreparedTerm{<:Breakage,<:FixedPivot},op,p)
    cache=item.cache
    cache.daughters isa ContinuousDaughters ? prepare_run(item.term,item.method,cache,op.grid,p) : cache
end
