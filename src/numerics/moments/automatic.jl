# Moment-only capability analysis. No callback is probed to infer its structure.
"""
    moment_dependencies(term, order)

Analytical moment-rule extension hook. Return the tuple of raw-moment orders
needed for this contribution, or `nothing` if no analytical expression is
implemented. An empty tuple denotes an independent source or structural zero.
Implement `analytical_moment` as well. Dependencies describe declarations, never
incidental parameter values. Generic supported rates use population quadrature.
"""
moment_dependencies(term,k::Integer)=nothing
_literal_zero(a)=a isa Number && iszero(a)
_polynomial_dependencies(c,k,shift)=Tuple(k+j-shift for (j,a) in enumerate(c) if !_literal_zero(a))
function moment_dependencies(term::Growth,k::Integer)
    k==0 && return ()
    r=term.rate
    r isa AffineGrowth && return _polynomial_dependencies((r.offset,r.slope),k,2)
    r isa PolynomialRate && return _polynomial_dependencies(r.coefficients,k,2)
    nothing
end
function moment_dependencies(term::Union{Death,Dilution},k::Integer)
    term.rate isa PolynomialRate ? _polynomial_dependencies(term.rate.coefficients,k,1) : nothing
end
moment_dependencies(::Union{Nucleation,MomentSource},k::Integer)=()
function moment_dependencies(term::Aggregation,k::Integer)
    term.kernel isa ConstantKernel && _literal_zero(term.kernel.rate) && return ()
    term.coordinate_map===nothing && k==1 && return ()
    term.kernel isa ConstantKernel || return nothing
    k==0 && return (0,)
    term.coordinate_map===nothing ? Tuple(1:k-1) : nothing
end
_daughter_zero(d::SelfSimilarDaughters,k)=k==d.measure_order || iszero(_daughter_factor(d,k))
_daughter_zero(::UniformVolumeDaughters,k)=k==3
_daughter_zero(d,k)=false
_daughter_factor(d::SelfSimilarDaughters,k)=sum(d.counts[i]*d.ratios[i]^k for i in eachindex(d.ratios))-1
_daughter_factor(::UniformVolumeDaughters,k)=6/(k+3)-1
function moment_dependencies(term::Breakage,k::Integer)
    term.rate isa PolynomialRate && all(_literal_zero,term.rate.coefficients) && return ()
    _daughter_zero(term.daughters,k) && return ()
    term.rate isa PolynomialRate && term.daughters isa Union{SelfSimilarDaughters,UniformVolumeDaughters} ?
        _polynomial_dependencies(term.rate.coefficients,k,1) : nothing
end

"""
    analytical_moment(term, order, getmoment, p, t, ctx)

Evaluate one declared analytical contribution. `getmoment(k)` supplies a
tracked moment directly or a missing moment from the selected family. Extend
this together with `moment_dependencies`; independent source moments must not
use population quadrature. Coefficients use the current time and context.
"""
function analytical_moment(term,k,getmoment,p,t,ctx)
    throw(ArgumentError("no analytical moment evaluator for $(typeof(term)), order $k"))
end
function analytical_moment(term::Growth,k,getmoment,p,t,ctx)
    k==0 && return zero(getmoment(0))
    r=term.rate
    c=r isa AffineGrowth ? (r.offset,r.slope) : r.coefficients
    value=zero(getmoment(0))
    for (j,a) in enumerate(c)
        _literal_zero(a) && continue
        coefficient=r isa AffineGrowth ? _affine_coefficient(a,p) : _source_value(a,p,t,ctx)
        isfinite(coefficient) || throw(NumericalFailure("nonfinite analytical growth coefficient"))
        value+=k*coefficient*getmoment(k+j-2)
    end
    value
end
function analytical_moment(term::Union{Death,Dilution},k,getmoment,p,t,ctx)
    value=zero(getmoment(0))
    for (j,a) in enumerate(term.rate.coefficients)
        _literal_zero(a) && continue
        value-=_nonnegative(_source_value(a,p,t,ctx),"exact removal coefficient")*getmoment(k+j-1)
    end
    value
end
analytical_moment(term::Nucleation,k,getmoment,p,t,ctx)=
    _nonnegative(term.rate(p,t,ctx),"nucleation rate")*term.size^k
analytical_moment(term::MomentSource,k,getmoment,p,t,ctx)=
    _nonnegative(term.rate(p,t,ctx),"moment source rate")*_nonnegative(term.rawmoment(k,p,t,ctx),"source raw moment")
function analytical_moment(term::Aggregation,k,getmoment,p,t,ctx)
    term.kernel isa ConstantKernel && _literal_zero(term.kernel.rate) && return zero(getmoment(0))
    term.coordinate_map===nothing && k==1 && return zero(getmoment(0))
    K=_nonnegative(_source_value(term.kernel.rate,p,t,ctx),"aggregation coefficient")
    k==0 && return -K*getmoment(0)^2/2
    K/2*sum((binomial(k,j)*getmoment(j)*getmoment(k-j) for j in 1:k-1);init=zero(getmoment(0)))
end
function analytical_moment(term::Breakage,k,getmoment,p,t,ctx)
    term.rate isa PolynomialRate && all(_literal_zero,term.rate.coefficients) && return zero(getmoment(0))
    _daughter_zero(term.daughters,k) && return zero(getmoment(0))
    value=zero(getmoment(0));factor=_daughter_factor(term.daughters,k)
    for (j,a) in enumerate(term.rate.coefficients)
        _literal_zero(a) && continue
        value+=factor*_nonnegative(_source_value(a,p,t,ctx),"exact breakage coefficient")*getmoment(k+j-1)
    end
    value
end

"""
    supports_moment_integral(term, basis)

Declare availability of a reconstruction-based population integral. External
terms also implement `prepare_term(term, QuadratureMomentRule(), basis)` and
`moment_term!`. This is an implementation capability, not an assertion that an
arbitrary callback has polynomial structure. Independent sources instead need
their own source moments (`MomentSource` or a source-specific extension).
"""
supports_moment_integral(term,::MomentSet)=term isa Union{Growth,Death,Dilution,Aggregation,Breakage,Nucleation,MomentSource}

"""
    reconstructed_moment(method, basis, state, reconstruction, order)

Evaluate a missing raw moment using the selected moment family. Tracked orders
are read directly by the compiler's moment view and never pass through this
hook. The default integrates the family's prepared positive nodes/weights;
Gamma EQMOM integrates its kernels analytically and lognormal uses its assumed
distribution. External families with another reconstruction can extend this hook.
"""
function reconstructed_moment(method::AbstractMomentMethod,g,state,q,k)
    q!==nothing && hasproperty(q,:nodes) && hasproperty(q,:weights) ||
        throw(ArgumentError("$(typeof(method)) must implement reconstructed_moment for missing order $k"))
    sum((w*x^k for (x,w) in zip(q.nodes,q.weights));init=zero(eltype(state)))
end
function reconstructed_moment(::LogNormalClosure,g,state,q,k)
    mu=vec(state)
    iszero(mu[1]) && return zero(mu[1])
    iszero(mu[2]) && return zero(mu[1])
    v=max(log(mu[3])+log(mu[1])-2log(mu[2]),zero(mu[1]))
    exp(log(mu[1])+k*(log(mu[2])-log(mu[1]))+k*(k-1)*v/2)
end
function reconstructed_moment(::EQMOM,g,state,q,k)
    sum((w*_gamma_moment(x,q.width,k) for (x,w) in zip(q.primary_nodes,q.primary_weights));init=zero(eltype(state)))
end

function _inferred_moment_nodes(g,offset,name)
    n=length(g)-offset
    n>=2 && iseven(n) || throw(ArgumentError(
        "$name cannot infer its resolution from orders 0:$(last(g.orders)); use orders 0:2N-1 for QMOM/RankAdaptiveQMOM/DQMOM, orders 0:2N for EQMOM, or explicitly select LogNormalClosure() with orders 0:2"))
    n÷2
end
_resolve_moment_method(m::AbstractMomentMethod,g)=m
_resolve_moment_method(m::QuadratureClosure,g)=m.nodes===nothing ? QMOM(_inferred_moment_nodes(g,0,"QMOM")) : m
_resolve_moment_method(m::RankAdaptiveQMOM,g)=m.max_nodes===nothing ? RankAdaptiveQMOM(_inferred_moment_nodes(g,0,"RankAdaptiveQMOM");rtol=m.rtol) : m
_resolve_moment_method(m::DQMOM,g)=m.nodes===nothing ? DQMOM(_inferred_moment_nodes(g,0,"DQMOM");condition_limit=m.condition_limit) : m
function _resolve_moment_method(m::EQMOM,g)
    m.nodes!==nothing && return m
    n=_inferred_moment_nodes(g,1,"EQMOM")
    EQMOM(n;secondary_nodes=m.secondary_nodes===nothing ? max(12,n+1) : m.secondary_nodes,rtol=m.rtol,maxiters=m.maxiters)
end

struct _AutomaticMomentRule <: AbstractTermMethod end
struct _MomentEquation
    order::Int
    treatment::Symbol
    dependencies::Tuple
    missing::Tuple
end
_needs_closure(e::_MomentEquation)=e.treatment in (:closure,:quadrature,:family_evaluator,:extension)
function _moment_equation(term,k,g;strict=false,name=:term)
    deps=moment_dependencies(term,k)
    if deps===nothing
        strict && throw(ArgumentError("term :$name, moment equation $k: no analytical rule is implemented for this rate-law declaration ($(typeof(term))); select a moment reconstruction family"))
        supports_moment_integral(term,g) || throw(ArgumentError("term :$name, moment equation $k: no supported moment treatment for $(typeof(term)); independent sources require source moments or a source-specific rule"))
        return _MomentEquation(k,:quadrature,(),())
    end
    deps isa Tuple && all(d->d isa Integer && d>=0,deps) || throw(ArgumentError("term :$name, moment equation $k: moment_dependencies must return nonnegative integer orders"))
    missing=Tuple(d for d in unique(deps) if !(d in g.orders))
    strict && !isempty(missing) && throw(ArgumentError("term :$name, moment equation $k requires untracked moment $(first(missing)); retain that order or select a reconstruction family"))
    _MomentEquation(k,isempty(missing) ? :analytical : :closure,deps,missing)
end
function _moment_override(methods,name,term)
    haskey(methods.by_term,name) && return getproperty(methods.by_term,name)
    get(methods.defaults,mechanism(term),nothing)
end
function _prepare_moment_terms(model,g,selected,methods)
    _validate_plan(methods,model)
    map(model.names,model.terms) do name,term
        rule=_moment_override(methods,name,term)
        strict=selected isa ExactClosure || rule isa ExactMomentRule
        selected isa ExactClosure && rule!==nothing && !(rule isa ExactMomentRule) &&
            throw(ArgumentError("term :$name: ExactClosure is incompatible with $(typeof(rule)); request only analytical evaluation"))
        equations=if rule===nothing || rule isa ExactMomentRule
            Tuple(_moment_equation(term,k,g;strict,name) for k in g.orders)
        else
            Tuple(_MomentEquation(k,:extension,(),()) for k in g.orders)
        end
        if rule===nothing && selected isa AbstractDirectMomentMethod && !(selected isa DQMOM)
            # Preserve existing third-party state-evolution hooks; no inverse
            # state-to-moment mapping can be invented for an arbitrary layout.
            rule=QuadratureMomentRule()
            equations=Tuple(_MomentEquation(k,:family_evaluator,(),()) for k in g.orders)
        end
        if rule===nothing
            cache=any(e->e.treatment===:quadrature,equations) ? prepare_term(term,QuadratureMomentRule(),g) : nothing
            PreparedTerm(name,term,_AutomaticMomentRule(),(evaluation=equations,prepared=cache))
        else
            if rule isa QuadratureMomentRule
                equations=Tuple(term isa Union{Nucleation,MomentSource} ?
                    _MomentEquation(k,:analytical,(),()) : _MomentEquation(k,:quadrature,(),()) for k in g.orders)
            elseif !(rule isa ExactMomentRule)
                equations=Tuple(_MomentEquation(k,:extension,(),()) for k in g.orders)
            end
            cache=prepare_term(term,rule,g)
            PreparedTerm(name,term,rule,(evaluation=equations,prepared=cache))
        end
    end
end
_moment_equations(item)=item.cache.evaluation
function _default_moment_method(model,g,methods)
    all_analytical=all(zip(model.names,model.terms)) do (name,term)
        override=_moment_override(methods,name,term)
        override!==nothing && !(override isa ExactMomentRule) && return false
        all(g.orders) do k
            deps=moment_dependencies(term,k)
            deps!==nothing && all(d->d in g.orders,deps)
        end
    end
    all_analytical ? ExactClosure() : _resolve_moment_method(RankAdaptiveQMOM(),g)
end

"""
    moment_plan(compiled)

Inspect moment compilation: the resolved `family`, `state_layout`, `state_length`,
tracked `orders`, `requires_closure`, and each named term's per-equation treatment,
dependencies and missing orders. Analytical evolution is reported independently
of the selected reconstruction family. Family validation still applies even
when every evolution contribution is analytical. Grid `term_methods` is unchanged.
"""
function moment_plan(op::CompiledMomentPBM)
    terms=NamedTuple{op.model.names}(Tuple(Tuple((order=e.order,treatment=e.treatment,
        dependencies=e.dependencies,missing=e.missing) for e in _moment_equations(item)) for item in op.prepared))
    b=op.boundary
    residual_boundary=b.left===:error && b.right===:error && b.lower==0 && isinf(b.upper)
    (family=op.closure,state_layout=op.closure isa AbstractMomentClosure ? :raw_moments :
        op.closure isa DQMOM ? :weights_weighted_abscissas : :family_defined,
     state_length=population_state_length(op),orders=Tuple(op.grid.orders),
     requires_closure=!residual_boundary || any(item->any(_needs_closure,_moment_equations(item)),op.prepared),
     boundary=(lower=b.lower,upper=b.upper,left=b.left,right=b.right,residual_supported=residual_boundary),terms)
end

mutable struct _MomentView{M,G,S,Q,R,C}
    method::M
    basis::G
    state::S
    reconstruction::Q
    raw::R
    missing::C
end
function _MomentView(method,g,state,q)
    raw=[moment_value(method,g,state,k) for k in g.orders]
    _MomentView(method,g,state,q,raw,Dict{Int,eltype(raw)}())
end
function (v::_MomentView)(k)
    k in v.basis.orders && return v.raw[k+1]
    get!(v.missing,k) do
        value=reconstructed_moment(v.method,v.basis,v.state,v.reconstruction,k)
        isfinite(value) && value>=0 || throw(NumericalFailure("invalid reconstructed moment $k from $(typeof(v.method))"))
        value
    end
end
function _automatic_raw_term!(out,term,equations,view,op,p,t,ctx,cache)
    term isa Union{Nucleation,MomentSource} && return _moment_source!(out,view.raw,term,op,p,t,ctx)
    if any(e->e.treatment===:quadrature,equations)
        # Existing evaluator hooks assemble all generic integrals once. Only
        # unresolved equations are copied; tracked analytical pieces stay exact.
        scratch=zero(view.raw)
        if view.method isa DQMOM
            _quadrature_term!(scratch,view.raw,term,op,p,t,ctx,view.reconstruction)
        else
            moment_term!(scratch,view.raw,term,QuadratureMomentRule(),cache,
                view.method,view.reconstruction,op,p,t,ctx)
        end
        for e in equations
            e.treatment===:quadrature && (out[e.order+1]+=scratch[e.order+1])
        end
    end
    for e in equations
        e.treatment===:quadrature && continue
        out[e.order+1]+=analytical_moment(term,e.order,view,p,t,ctx)
    end
end
function _automatic_term!(du,u,item,view,op,p,t,ctx)
    if op.closure isa DQMOM
        raw=zero(view.raw)
        _automatic_raw_term!(raw,item.term,item.cache.evaluation,view,op,p,t,ctx,item.cache.prepared)
        q=view.reconstruction
        rates=q.factor\[raw[k+1]/q.scale^k for k in op.grid.orders]
        n=op.closure.nodes
        for i in 1:n
            du[i]+=rates[i]
            du[n+i]+=q.scale*rates[n+i]
        end
    else
        _automatic_raw_term!(du,item.term,item.cache.evaluation,view,op,p,t,ctx,item.cache.prepared)
    end
end
