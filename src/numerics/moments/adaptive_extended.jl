"""
    RankAdaptiveQMOM(max_nodes; rtol=1e-8)

Evolve moments 0:2max_nodes-1 and use the smallest nonnegative quadrature that
reconstructs ALL tracked moments to the requested scaled tolerance. The number
of ODE states stays fixed. Singular point-mass and empty populations are allowed;
unresolved/nonrealizable inputs are rejected, not projected onto another vector.
`quadrature` reports `rank` and the scaled `moment_error`.
"""
struct RankAdaptiveQMOM <: AbstractMomentClosure
    max_nodes::Int
    rtol::Float64
    function RankAdaptiveQMOM(n::Integer;rtol=1e-8)
        n>=1 && 0<rtol<1 || throw(ArgumentError("positive node count and 0<rtol<1 required"))
        new(n,rtol)
    end
end
function validate_moment_method(m::RankAdaptiveQMOM,model,g::MomentSet)
    length(g)==2m.max_nodes || throw(ArgumentError("RankAdaptiveQMOM needs moments 0:2max_nodes-1"))
end
function _normalized_moments(g,u)
    mu=vec(u)
    length(mu)==length(g) || throw(DimensionMismatch("moment state size mismatch"))
    realizability(g,mu).valid || throw(NumericalFailure("nonrealizable moment vector"))
    iszero(mu[2]) && any(x->!iszero(x),mu[3:end]) &&
        throw(NumericalFailure("a nonnegative population with zero first moment must be supported at zero"))
    scale=mu[1]>0 && length(mu)>1 && mu[2]>0 ? mu[2]/mu[1] : 1.0
    normalized=iszero(mu[1]) ? zero(mu) : [mu[k+1]/mu[1]/scale^k for k in g.orders]
    all(isfinite,normalized) || throw(NumericalFailure("moment scaling overflow"))
    mu,scale,normalized
end
_quadrature_error(v,x,w)=maximum(abs(sum(w.*x.^k)-v[k+1])/max(1,abs(v[k+1])) for k in 0:length(v)-1)
function _adaptive_quadrature(v,max_nodes,rtol)
    for n in 1:max_nodes
        q=try
            quadrature(MomentSet(0:2n-1),v[1:2n],QMOM(n))
        catch e
            e isa NumericalFailure || rethrow()
            continue
        end
        error=_quadrature_error(v,q.nodes,q.weights)
        error<=rtol && return (;q.nodes,q.weights,rank=n,moment_error=error)
    end
    throw(NumericalFailure("no quadrature resolves every tracked moment at the requested tolerance; no moments were repaired"))
end
function quadrature(g::MomentSet,u,m::RankAdaptiveQMOM)
    mu,scale,v=_normalized_moments(g,u)
    length(mu)==2m.max_nodes || throw(DimensionMismatch("adaptive QMOM needs 2max_nodes moments"))
    iszero(mu[1]) && return (nodes=Float64[],weights=Float64[],rank=0,moment_error=0.0)
    q=_adaptive_quadrature(v,m.max_nodes,m.rtol)
    (;nodes=q.nodes.*scale,weights=q.weights.*mu[1],q.rank,q.moment_error)
end

"""
    EQMOM(nodes; kernel=:gamma, secondary_nodes=12, rtol=1e-7, maxiters=80)

One-coordinate Gamma extended QMOM using moments 0:2nodes. A mixture of `nodes`
Gamma kernels has nonnegative means and a common scale (width) parameter.
Moment deconvolution and a bracketed realizability-boundary search determine
the width; all original moments must reconstruct before a result is accepted.
Secondary Gauss–Laguerre quadrature integrates weak forms, independently of the
primary inversion. Only the Gamma kernel is implemented. Not every realizable
moment vector admits this kernel representation; failures are explicit.
"""
struct EQMOM <: AbstractMomentClosure
    nodes::Int
    secondary_nodes::Int
    rtol::Float64
    maxiters::Int
    function EQMOM(n::Integer;kernel=:gamma,secondary_nodes=max(12,n+1),rtol=1e-7,maxiters=80)
        kernel===:gamma || throw(ArgumentError("EQMOM currently implements the Gamma kernel only"))
        n>=1 && secondary_nodes>=n+1 && 0<rtol<1 && maxiters>=30 ||
            throw(ArgumentError("invalid EQMOM node count, secondary quadrature or inversion tolerance"))
        new(n,secondary_nodes,rtol,maxiters)
    end
end
function validate_moment_method(m::EQMOM,model,g::MomentSet)
    length(g)==2m.nodes+1 || throw(ArgumentError("Gamma EQMOM needs moments 0:2nodes"))
end

# E[X^k | mean=xi, scale=sigma] = product(xi+j*sigma, j=0:k-1).
# Polynomial coefficients are unsigned Stirling numbers of the first kind.
function _gamma_coefficients(k,sigma)
    coefficients=[1.0]
    for j in 0:k-1
        next=zeros(length(coefficients)+1)
        for i in eachindex(coefficients)
            next[i]+=j*sigma*coefficients[i]
            next[i+1]+=coefficients[i]
        end
        coefficients=next
    end
    coefficients
end
function _gamma_deconvolve(v,sigma)
    raw=copy(v)
    for k in 1:length(v)-1
        c=_gamma_coefficients(k,sigma)
        raw[k+1]=v[k+1]-sum(c[j+1]*raw[j+1] for j in 0:k-1)
    end
    raw
end
_gamma_moment(x,sigma,k)=prod((x+j*sigma for j in 0:k-1);init=1.0)
function _deconvolution_margin(v,n)
    H0=Symmetric([v[i+j-1] for i in 1:n+1,j in 1:n+1])
    H1=Symmetric([v[i+j] for i in 1:n,j in 1:n])
    min(eigmin(H0),eigmin(H1))
end
function _gamma_primary(g,u,m)
    mu,scale,v=_normalized_moments(g,u)
    length(mu)==2m.nodes+1 || throw(DimensionMismatch("EQMOM needs 2nodes+1 moments"))
    iszero(mu[1]) && return (nodes=Float64[],weights=Float64[],width=0.0,rank=0,moment_error=0.0)
    # The width-zero limit must match the extra moment too.
    discrete=try _adaptive_quadrature(v,m.nodes,min(m.rtol,1e-10)) catch e
        e isa NumericalFailure || rethrow(); nothing
    end
    if discrete!==nothing
        return (;nodes=discrete.nodes.*scale,weights=discrete.weights.*mu[1],width=0.0,
            discrete.rank,discrete.moment_error)
    end
    upper=max(0.0,v[3]-v[2]^2)/v[2]
    upper>0 || throw(NumericalFailure("no positive Gamma EQMOM width"))
    lower=0.0
    for _ in 1:m.maxiters
        mid=(lower+upper)/2
        mid==lower || mid==upper ? break : nothing
        margin=_deconvolution_margin(_gamma_deconvolve(v,mid),m.nodes)
        margin>=0 ? (lower=mid) : (upper=mid)
    end
    sigma=lower
    raw=_gamma_deconvolve(v,sigma)
    primary=_adaptive_quadrature(raw,m.nodes,m.rtol)
    error=maximum(abs(sum(primary.weights.*_gamma_moment.(primary.nodes,sigma,k))-v[k+1])/max(1,abs(v[k+1])) for k in g.orders)
    error<=m.rtol || throw(NumericalFailure("Gamma EQMOM inversion failed moment reconstruction; this kernel family may be ineligible"))
    (;nodes=primary.nodes.*scale,weights=primary.weights.*mu[1],width=sigma*scale,
      primary.rank,moment_error=error)
end
function quadrature(g::MomentSet,u,m::EQMOM)
    primary=_gamma_primary(g,u,m);x=Float64[];w=Float64[]
    for (xi,wi) in zip(primary.nodes,primary.weights)
        if iszero(primary.width) || iszero(xi)
            push!(x,xi);push!(w,wi);continue
        end
        alpha=xi/primary.width;n=m.secondary_nodes
        J=SymTridiagonal([alpha+2i-2 for i in 1:n],
            [sqrt(i*(i+alpha-1)) for i in 1:n-1])
        e=eigen(J)
        append!(x,primary.width.*e.values);append!(w,wi.*e.vectors[1,:].^2)
    end
    (;nodes=x,weights=w,primary_nodes=primary.nodes,primary_weights=primary.weights,
      width=primary.width,primary.rank,primary.moment_error)
end
prepare_moment_state(m::Union{RankAdaptiveQMOM,EQMOM},g,u,p,t,ctx)=quadrature(g,u,m)
function validate_moment_state(m::Union{RankAdaptiveQMOM,EQMOM},g,u)
    quadrature(g,u,m)
    nothing
end

"""Evaluate the declared Gamma EQMOM continuous density. Dirac limits have no finite density; use reconstructed_bins instead. This is a kernel reconstruction, not a uniquely determined PSD."""
function reconstructed_density(g::MomentSet,u,m::EQMOM,x::Real)
    q=_gamma_primary(g,u,m)
    q.rank==0 && return 0.0
    q.width>0 && all(>(0),q.nodes) || throw(ArgumentError("a Dirac component has no ordinary density; use reconstructed_bins"))
    sum(w*Distributions.pdf(Distributions.Gamma(xi/q.width,q.width),x) for (xi,w) in zip(q.nodes,q.weights))
end
"""
    reconstructed_bins(basis,state,method::EQMOM,edges; weight_order=0,normalize=false)

Exact Gamma-CDF bin integrals of the declared kernel mixture, optionally weighted
by x^weight_order. Includes width-zero atoms, with left-closed bins and a closed
final edge. Normalization conditions on the stated window. No particle sample
count is invented, and QMOM nodes are not treated as a reconstructed density.
"""
function reconstructed_bins(g::MomentSet,u,m::EQMOM,edges;weight_order::Integer=0,normalize=false)
    e=collect(edges)
    length(e)>=2 && first(e)>=0 && all(diff(e).>0) && weight_order>=0 || throw(ArgumentError("invalid bin edges or weight order"))
    q=_gamma_primary(g,u,m);out=zeros(length(e)-1)
    for (xi,w) in zip(q.nodes,q.weights)
        if iszero(q.width) || iszero(xi)
            i=xi==last(e) ? length(out) : searchsortedlast(e,xi)
            1<=i<=length(out) && (out[i]+=w*xi^weight_order)
        else
            alpha=xi/q.width;k=weight_order;d=Distributions.Gamma(alpha+k,q.width)
            factor=w*_gamma_moment(xi,q.width,k)
            for i in eachindex(out)
                probability=e[i]>Distributions.mean(d) ? Distributions.ccdf(d,e[i])-Distributions.ccdf(d,e[i+1]) :
                    Distributions.cdf(d,e[i+1])-Distributions.cdf(d,e[i])
                out[i]+=factor*probability
            end
        end
    end
    if normalize
        sum(out)>0 || throw(NumericalFailure("cannot normalize an empty EQMOM measurement window"))
        out./=sum(out)
    end
    out
end
