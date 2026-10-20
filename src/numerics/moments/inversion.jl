function quadrature(g::MomentSet,u,m::LogNormalClosure)
    μ=vec(u); length(μ)==length(g)==3 || throw(DimensionMismatch("lognormal closure requires moments 0:2"))
    all(isfinite,μ) && all(μ.>=0) || throw(NumericalFailure("invalid lognormal moments"))
    n=m.quadrature_order
    if iszero(μ[1])
        all(iszero,μ) || throw(NumericalFailure("zero number with nonzero moments"))
        return (;nodes=zeros(n),weights=zeros(n))
    end
    iszero(μ[2]) && iszero(μ[3]) && return (;nodes=zeros(n),weights=fill(μ[1]/n,n))
    μ[2]>0 && μ[3]>0 || throw(NumericalFailure("lognormal moments must have positive mean and second moment"))
    variance=log(μ[3])+log(μ[1])-2log(μ[2])
    variance>=-1e-12 || throw(NumericalFailure("lognormal closure has negative log variance"))
    variance=max(variance,0.0)
    meanlog=log(μ[2])-log(μ[1])-variance/2
    nodes=exp.(meanlog.+sqrt(variance).*m.nodes); weights=μ[1].*m.weights
    for k in 0:2
        isapprox(sum(weights.*nodes.^k),μ[k+1];rtol=1e-8,atol=1e-13*μ[1]) || throw(NumericalFailure("lognormal quadrature is unresolved; increase quadrature_order"))
    end
    (;nodes,weights)
end

"""
Necessary Stieltjes Hankel positivity diagnostics for raw moments. `valid`
does not prove realizability at singular boundaries; QMOM also requires an
invertible Hankel matrix and verifies its reconstructed moments.
"""
function realizability(g::MomentSet,u;rtol=1e-10)
    μ=vec(u); length(μ)==length(g) || throw(DimensionMismatch("moment size mismatch"))
    rtol>=0 && isfinite(rtol) || throw(ArgumentError("rtol must be finite and nonnegative"))
    all(isfinite,μ) || return (;valid=false,min_hankel_eigenvalue=-Inf,rank=0)
    μ[1]<0 && return (;valid=false,min_hankel_eigenvalue=-Inf,rank=0)
    iszero(μ[1]) && return (;valid=all(iszero,μ),min_hankel_eigenvalue=0.0,rank=0)
    s=length(μ)>1 && μ[2]>0 ? μ[2]/μ[1] : 1.0
    ν=[μ[k+1]/μ[1]/s^k for k in g.orders]
    all(isfinite,ν) || return (;valid=false,min_hankel_eigenvalue=-Inf,rank=0)
    n0=fld(length(g)-1,2)+1; n1=fld(length(g),2)
    H0=[ν[i+j-1] for i in 1:n0,j in 1:n0]
    e0=eigvals(Symmetric(H0)); values=e0
    n1>0 && (values=vcat(e0,eigvals(Symmetric([ν[i+j] for i in 1:n1,j in 1:n1]))))
    scale=max(1.0,maximum(abs,values)); mineig=minimum(values)
    (;valid=all(μ.>=0) && mineig>=-rtol*scale,min_hankel_eigenvalue=mineig,rank=count(>(rtol*scale),e0))
end

"""Recover n-node Gaussian quadrature from moments 0:2n-1 using scaled Hankel matrices."""
function quadrature(g::MomentSet,u,closure::QuadratureClosure=QuadratureClosure(length(g)÷2))
    n=closure.nodes; μ=vec(u)
    length(g)==length(μ)==2n || throw(DimensionMismatch("quadrature needs 2n moments"))
    if iszero(μ[1])
        all(iszero,μ) || throw(NumericalFailure("zero number moment with nonzero higher moments"))
        return (;nodes=zeros(n),weights=zeros(n))
    end
    realizability(g,μ).valid || throw(NumericalFailure("moment vector fails nonnegative-support Hankel checks"))
    s=μ[2]>0 ? μ[2]/μ[1] : 1.0
    ν=[μ[k+1]/μ[1]/s^k for k in g.orders]
    H0=Symmetric([ν[i+j-1] for i in 1:n,j in 1:n]); H1=[ν[i+j] for i in 1:n,j in 1:n]
    L=try
        cholesky(H0).L
    catch e
        e isa PosDefException || rethrow()
        throw(NumericalFailure("QMOM Hankel matrix is singular/nonpositive; reduce node count or use a rank-adaptive closure"))
    end
    J=L\H1/L'; eig=eigen(Symmetric((J+J')/2))
    minimum(eig.values)>=-1e-12 || throw(NumericalFailure("QMOM produced negative support"))
    nodes=max.(eig.values,0).*s; weights=μ[1].*(eig.vectors[1,:].^2)
    for k in g.orders
        isapprox(sum(weights.*(nodes.^k)),μ[k+1];rtol=1e-7,atol=1e-12*μ[1]*s^k) || throw(NumericalFailure("QMOM inversion failed its moment reconstruction check"))
    end
    (;nodes,weights)
end
