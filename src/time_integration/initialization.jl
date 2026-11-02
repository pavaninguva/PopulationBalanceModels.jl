abstract type AbstractPopulationInitial end

"""
    InitialDensity(density; support=nothing, quadrature=GaussLegendre(16),
                   rtol=1e-9, atol=1e-13, maxorder=256)

Physical **number density**, not a probability density. Supply `(x,p,t)->n` or
`((x,y),p,t)->n` for a joint tensor density. No independence is inferred and no
normalization is applied. `support=(a,b)` (or two intervals in 2D) declares where
the density exists. On grids the default is the represented domain; excluded
measure outside an undeclared support is unknown. Moment initialization requires
explicit finite support. Positive quadrature is checked by order refinement.
"""
struct InitialDensity{F,S,Q} <: AbstractPopulationInitial
    density::F
    support::S
    quadrature::Q
    rtol::Float64
    atol::Float64
    maxorder::Int
end
function InitialDensity(f;support=nothing,quadrature=GaussLegendre(16),rtol=1e-9,atol=1e-13,maxorder=256)
    _initial_accuracy(quadrature,rtol,atol,maxorder)
    InitialDensity(f,support,quadrature,Float64(rtol),Float64(atol),Int(maxorder))
end

"""
    InitialDistribution(distribution; total, support=nothing,
                        truncation=:physical, quadrature=GaussLegendre(16), ...)

Probability law with a declared total number concentration. `distribution` and
`total` may instead be `(p,t)` callbacks. Univariate grid bins use stable CDF/CCDF
differences. The default retains only probability inside the represented domain
and declared support; it does **not** redistribute an excluded tail. Explicit
`truncation=:renormalize` conditions the law on that domain/support. Tensor laws
must be genuine joint distributions; their densities use checked tensor quadrature.

Moment initialization integrates the continuous law on its support (intersected
with explicit support), not a finite grid projection. DQMOM receives weights and
weighted abscissas through its existing inversion/state machinery. Descriptions
remain attached to PBMProblem, so `remake(pb;p=...)` rebuilds parameter-dependent
initial data; numerical initial arrays instead retain fixed-state semantics.
"""
struct InitialDistribution{D,C,S,Q} <: AbstractPopulationInitial
    distribution::D
    total::C
    support::S
    truncation::Symbol
    quadrature::Q
    rtol::Float64
    atol::Float64
    maxorder::Int
end
function _initial_accuracy(q,rtol,atol,maxorder)
    q isa GaussLegendre || throw(ArgumentError("initialization requires GaussLegendre quadrature"))
    isfinite(rtol) && rtol>0 && isfinite(atol) && atol>=0 && maxorder isa Integer && maxorder>=2q.order ||
        throw(ArgumentError("invalid initialization accuracy; maxorder must allow at least one order doubling"))
end
function InitialDistribution(d;total,support=nothing,truncation=:physical,
        quadrature=GaussLegendre(16),rtol=1e-9,atol=1e-13,maxorder=256)
    truncation in (:physical,:renormalize) || throw(ArgumentError("truncation must be :physical or :renormalize"))
    _initial_accuracy(quadrature,rtol,atol,maxorder)
    InitialDistribution(d,total,support,truncation,quadrature,Float64(rtol),Float64(atol),Int(maxorder))
end
_initial_description(d::AbstractPopulationInitial)=d
_initial_value(v::Number,p,t)=v
_initial_value(f,p,t)=f(p,t)
_initial_distribution(d::Distributions.Distribution,p,t)=d
_initial_distribution(f,p,t)=f(p,t)
function _initial_interval(s)
    s isa Tuple && length(s)==2 && all(x->x isa Real && isfinite(x),s) && 0<=s[1]<s[2] ||
        throw(ArgumentError("initial support must be a finite increasing nonnegative interval"))
    s
end
function _initial_pdf(f,x,p,t)
    v=f(x,p,t)
    v isa Real && isfinite(v) && v>=0 || throw(ArgumentError("initial density must evaluate to a finite nonnegative scalar"))
    v
end
function _initial_fraction(d,a,b)
    a>=b && return zero(Distributions.cdf(d,a))
    a>Distributions.median(d) ? Distributions.ccdf(d,a)-Distributions.ccdf(d,b) :
        Distributions.cdf(d,b)-Distributions.cdf(d,a)
end
function _checked_initial_quadrature(evaluate,desc)
    n=desc.quadrature.order
    old=evaluate(_gauss_legendre(GaussLegendre(n)))
    while 2n<=desc.maxorder
        n*=2
        new=evaluate(_gauss_legendre(GaussLegendre(n)))
        all(isfinite,new) || throw(ArgumentError("nonfinite initial integral"))
        all(abs.(new.-old).<=desc.atol.+desc.rtol.*abs.(new)) && return new,n
        old=new
    end
    throw(NumericalFailure("initialization quadrature unresolved at order $(desc.maxorder); increase maxorder, refine support or supply a distribution with stable integrals"))
end
function _initial_law(desc::InitialDistribution,p,t)
    d=_initial_distribution(desc.distribution,p,t)
    d isa Distributions.ContinuousDistribution || throw(ArgumentError("initial probability law must be a continuous Distributions.jl distribution"))
    total=_initial_value(desc.total,p,t)
    total isa Real && isfinite(total) && total>=0 || throw(ArgumentError("initial total must be finite and nonnegative"))
    d,total
end
function _initial_grid(g::Grid1D,desc::InitialDistribution,p,t)
    d,total=_initial_law(desc,p,t)
    d isa Distributions.UnivariateDistribution || throw(ArgumentError("Grid1D needs a univariate initial distribution"))
    support=desc.support===nothing ? (first(g.edges),last(g.edges)) : _initial_interval(desc.support)
    lo=max(first(g.edges),support[1]);hi=min(last(g.edges),support[2])
    fractions=[_initial_fraction(d,max(a,lo),min(b,hi)) for (a,b) in zip(g.edges[1:end-1],g.edges[2:end])]
    retained=sum(fractions)
    scale=desc.truncation===:physical ? total : retained>0 ? total/retained :
        throw(ArgumentError("cannot normalize an initial law with zero represented probability"))
    N=reshape(scale.*fractions,:,1)
    N,(basis=:bin_integrals,policy=desc.truncation,retained_probability=retained,
        retained_number=sum(N),excluded_number=total*(1-retained),quadrature_order=0,
        normalization_factor=scale,support=(lo,hi))
end
function _initial_grid(g::Grid1D,desc::InitialDensity,p,t)
    support=desc.support===nothing ? (first(g.edges),last(g.edges)) : _initial_interval(desc.support)
    f=x->_initial_pdf(desc.density,x,p,t)
    evaluate=q->[b<=a ? zero(f((support[1]+support[2])/2)) : _integrate_bin(f,a,b,q)
        for (a,b) in ((max(g.edges[i],support[1]),min(g.edges[i+1],support[2])) for i in 1:length(g))]
    bins,order=_checked_initial_quadrature(evaluate,desc)
    full=desc.support===nothing ? nothing : first(_checked_initial_quadrature(q->[_integrate_bin(f,support...,q)],desc))[1]
    reshape(bins,:,1),(basis=:bin_integrals,policy=:physical,retained_number=sum(bins),
        excluded_number=full===nothing ? nothing : full-sum(bins),quadrature_order=order,support=support)
end
function _initial_grid(g::Grid2D,desc::AbstractPopulationInitial,p,t)
    if desc isa InitialDistribution
        d,total=_initial_law(desc,p,t)
        d isa Distributions.MultivariateDistribution && length(d)==2 || throw(ArgumentError("Grid2D needs an explicit two-coordinate joint distribution or InitialDensity"))
        f=(xy,p,t)->Distributions.pdf(d,collect(xy))
    else
        f=desc.density;total=nothing
    end
    domain=((first(g.x.edges),last(g.x.edges)),(first(g.y.edges),last(g.y.edges)))
    support=desc.support===nothing ? domain : Tuple(_initial_interval(s) for s in desc.support)
    length(support)==2 || throw(ArgumentError("Grid2D support needs two coordinate intervals"))
    integrate=function(q,ax,bx,ay,by)
        (bx<=ax || by<=ay) && return 0.0
        _integrate_bin(x->_integrate_bin(y->_initial_pdf(f,(x,y),p,t),ay,by,q),ax,bx,q)
    end
    bins,order=_checked_initial_quadrature(q->[integrate(q,max(g.x.edges[i],support[1][1]),
        min(g.x.edges[i+1],support[1][2]),max(g.y.edges[j],support[2][1]),
        min(g.y.edges[j+1],support[2][2])) for i in 1:length(g.x),j in 1:length(g.y)],desc)
    retained=sum(bins)
    if desc isa InitialDistribution
        scale=desc.truncation===:physical ? total : retained>0 ? total/retained : throw(ArgumentError("zero initial represented probability"))
        return bins.*scale,(basis=:bin_integrals,policy=desc.truncation,
            retained_probability=retained,retained_number=retained*scale,
            excluded_number=total*(1-retained),quadrature_order=order,support=support)
    end
    full=desc.support===nothing ? nothing : first(_checked_initial_quadrature(q->[integrate(q,support[1]...,support[2]...)],desc))[1]
    bins,(basis=:bin_integrals,policy=:physical,retained_number=retained,
        excluded_number=full===nothing ? nothing : full-retained,quadrature_order=order,support=support)
end

# Full-support analytical raw moments. These are continuous-law moments, never
# moments of a centroid projection. Unsupported laws use checked probability
# quadrature; callers may declare a finite physical truncation explicitly.
_initial_raw(d,k)=nothing
_initial_raw(d::Distributions.LogNormal,k)=exp(k*d.μ+k^2*d.σ^2/2)
_initial_raw(d::Distributions.Gamma,k)=d.θ^k*exp(loggamma(d.α+k)-loggamma(d.α))
_initial_raw(d::Distributions.Exponential,k)=d.θ^k*factorial(k)
_initial_raw(d::Distributions.Beta,k)=exp(loggamma(d.α+k)-loggamma(d.α)+loggamma(d.α+d.β)-loggamma(d.α+d.β+k))
_initial_raw(d::Distributions.Uniform,k)=(d.b^(k+1)-d.a^(k+1))/((k+1)*(d.b-d.a))
_initial_partial_raw(d,k,a,b)=a==minimum(d) && b==maximum(d) ? _initial_raw(d,k) : nothing
function _initial_partial_raw(d::Distributions.Normal,k,a,b)
    a>=b && return zero(d.μ+d.σ)
    alpha=(a-d.μ)/d.σ;beta=(b-d.μ)/d.σ
    standard=Distributions.Normal()
    mass=isinf(alpha) && alpha<0 ? (isinf(beta) ? one(d.μ+d.σ) : Distributions.cdf(standard,beta)) :
        isinf(beta) && beta>0 ? Distributions.ccdf(standard,alpha) : _initial_fraction(standard,alpha,beta)
    integrals=[mass]
    boundary(x,j)=isfinite(x) ? x^j*Distributions.pdf(standard,x) : zero(d.μ+d.σ)
    k>=1 && push!(integrals,boundary(alpha,0)-boundary(beta,0))
    for j in 2:k
        push!(integrals,boundary(alpha,j-1)-boundary(beta,j-1)+(j-1)*integrals[j-1])
    end
    sum(binomial(k,j)*d.μ^(k-j)*d.σ^j*integrals[j+1] for j in 0:k)
end
function _initial_partial_raw(d::Distributions.Truncated,k,a,b)
    lo=max(a,minimum(d));hi=min(b,maximum(d))
    value=_initial_partial_raw(d.untruncated,k,lo,hi)
    value===nothing && return nothing
    normalization=_initial_partial_raw(d.untruncated,0,minimum(d),maximum(d))
    normalization===nothing && return nothing
    value/normalization
end
function _initial_moments(op,desc::InitialDistribution,p,t)
    d,total=_initial_law(desc,p,t)
    d isa Distributions.UnivariateDistribution || throw(ArgumentError("one-coordinate moment methods need a univariate distribution"))
    lower=minimum(d);upper=maximum(d)
    if desc.support!==nothing
        a,b=_initial_interval(desc.support);lower=max(lower,a);upper=min(upper,b)
    end
    lower>=0 || throw(ArgumentError("moment initial distribution must have nonnegative support; explicitly truncate the physical law"))
    mass=_initial_fraction(d,lower,upper)
    vals=[_initial_partial_raw(d,k,lower,upper) for k in op.grid.orders]
    order=0
    if lower>=upper
        vals=zeros(length(op.grid.orders))
    elseif any(isnothing,vals)
        # Probability-space integration works on infinite physical support
        # without introducing a hidden finite numerical tail cutoff.
        a=Distributions.cdf(d,lower);b=Distributions.cdf(d,upper)
        vals,order=_checked_initial_quadrature(q->[_integrate_bin(u->Distributions.quantile(d,u)^k,a,b,q)
            for k in op.grid.orders],desc)
    else
        vals=collect(promote(vals...))
    end
    scale=desc.truncation===:physical ? total : mass>0 ? total/mass : throw(ArgumentError("zero initial probability on support"))
    raw=scale.*vals
    raw,(basis=:continuous_raw_moments,policy=desc.truncation,retained_probability=mass,
        retained_number=raw[1],excluded_number=total*(1-mass),quadrature_order=order,support=(lower,upper))
end
function _initial_moments(op,desc::InitialDensity,p,t)
    desc.support===nothing && throw(ArgumentError("InitialDensity on MomentSet requires explicit finite support"))
    a,b=_initial_interval(desc.support)
    raw,order=_checked_initial_quadrature(q->[_integrate_bin(x->x^k*_initial_pdf(desc.density,x,p,t),a,b,q)
        for k in op.grid.orders],desc)
    raw,(basis=:continuous_raw_moments,policy=:physical,retained_number=raw[1],
        excluded_number=0.0,quadrature_order=order,support=(a,b))
end
function _initialize_with_report(op,desc,p,t)
    hasproperty(op,:grid) && op.grid isa Union{Grid1D,Grid2D} || throw(ArgumentError("physical initialization unsupported for $(typeof(op))"))
    _initial_grid(op.grid,desc,p,t)
end
function _initialize_with_report(op::CompiledMomentPBM,desc,p,t)
    raw,report=_initial_moments(op,desc,p,t)
    if op.closure isa DQMOM
        q=quadrature(op.grid,raw,QMOM(op.closure.nodes))
        return reshape(quadrature_state(op.closure,q.nodes,q.weights),:,1),merge(report,(state_layout=:weights_and_weighted_abscissas,))
    end
    op.closure isa AbstractMomentClosure || throw(ArgumentError("custom direct moment family must extend initialize; its state cannot be inferred from raw moments"))
    reshape(raw,:,1),merge(report,(state_layout=:raw_moments,))
end
_initialize_with_report(op::Union{CompiledAxial,CompiledAxialSystem},desc,p,t)=
    throw(ArgumentError("axial initialization requires an explicit intrinsic-state × axial-cell array; initialize the local population first and declare its axial profile"))
PBMProblem(op::Union{CompiledAxial,CompiledAxialSystem},desc::AbstractPopulationInitial,tspan;p=(;),history=nothing)=
    throw(ArgumentError("axial PBMProblem requires an explicit intrinsic-state × axial-cell array; use initialize on the local population and declare its axial profile"))

"""
    initialize(compiled, initial; p=(;), t=0.0)

Convert InitialDensity/InitialDistribution into the selected representation's
validated solver state. Numerical arrays remain accepted and copied. Supports
Grid1D/Grid2D bin integrals, raw moment families and DQMOM; unsupported/empty/
rank-deficient states retain the selected family's existing errors. Extend this
function for a custom direct-family state layout.
"""
function initialize(op,desc::AbstractPopulationInitial;p=(;),t=0.0)
    state,_=_initialize_with_report(op,desc,p,t)
    PBMProblem(op,state,(t,t+one(t));p).u0
end
initialize(op,state::AbstractArray;p=(;),t=0.0)=PBMProblem(op,state,(t,t+one(t));p).u0

"""
    initialization_report(compiled, description; p=(;), t=0.0)
    initialization_report(problem)

Inspect initialization basis, truncation/normalization, retained and (where
known) excluded number, and quadrature order. Continuous moments differ from
moments of the finite bin projection. Reports are recalculated, not producer
receipts. Numerical arrays report `policy=:fixed_state`.
"""
initialization_report(op,desc::AbstractPopulationInitial;p=(;),t=0.0)=last(_initialize_with_report(op,desc,p,t))
initialization_report(pb::PBMProblem)=pb.initial_description===nothing ?
    (policy=:fixed_state,) : initialization_report(pb.operator,pb.initial_description;p=pb.p,t=first(pb.tspan))

struct _SystemInitial{P,B}
    populations::P
    bulk::B
end
function _remake_initial(pb,initial::_SystemInitial,p,tspan,history)
    PBMProblem(pb.operator.system,initial.populations,tspan;bulk0=initial.bulk,p,history)
end
function initialization_report(op::CompiledSystem,initial::_SystemInitial;p=(;),t=0.0)
    map(keys(initial.populations)) do name
        value=getproperty(initial.populations,name)
        value isa AbstractPopulationInitial ? initialization_report(getproperty(op.system.populations,name),value;p,t) : (policy=:fixed_state,)
    end |> reports->NamedTuple{keys(initial.populations)}(Tuple(reports))
end
