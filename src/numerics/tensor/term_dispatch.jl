mechanism(::Growth2D)=:growth
mechanism(::Nucleation2D)=:nucleation
default_method(::Growth2D,::Grid2D)=Upwind()
default_method(::Union{Birth,DistributedNucleation},::Grid2D)=MidpointRule()
default_method(::Union{Death,Dilution},::Grid2D)=PointwiseLoss()
default_method(::Nucleation2D,::Grid2D)=FixedPivot()
function prepare_term(::Growth2D,m::Union{Upwind,WENO5},g::Grid2D;kwargs...)
    _validate_reconstruction(m,g.x); _validate_reconstruction(m,g.y)
end
prepare_term(::Union{Birth,DistributedNucleation},::MidpointRule,::Grid2D;kwargs...)=nothing
prepare_term(::Union{Birth,DistributedNucleation},m::GaussLegendre,::Grid2D;kwargs...)=_gauss_legendre(m)
prepare_term(::Union{Death,Dilution},::PointwiseLoss,::Grid2D;kwargs...)=nothing
prepare_term(::Union{Death,Dilution},m::CellAverageLoss,::Grid2D;kwargs...)=_gauss_legendre(m.quadrature)
function prepare_term(term::Nucleation2D,m::FixedPivot,g::Grid2D;kwargs...)
    m.overflow===:discard && throw(ArgumentError("2D point-source overflow accounting is not implemented; use :inherit or :error"))
    all(a->first(a[1].centers)<=a[2]<=last(a[1].centers),zip((g.x,g.y),term.size)) || throw(ArgumentError("source must lie inside both pivot ranges"))
    nothing
end
prepare_term(::Nucleation2D,::GaussianInjection,::Grid2D;kwargs...)=nothing
function add_term!(dN,N,term::Growth2D,m::Union{Upwind,WENO5},cache,op,p,t,ctx)
    _tensor_term!(dN,N,term,(;grid=op.grid,bc=op.bc,scheme=m),p,t,ctx)
end
add_term!(dN,N,term,m::Union{MidpointRule,FixedPivot,PointwiseLoss},cache,op::CompiledPBM2D,p,t,ctx)=_tensor_term!(dN,N,term,op,p,t,ctx)
function _tensor_integral(f,g,i,j,q)
    _integrate_bin(x->_integrate_bin(y->f((x,y)),g.y.edges[j],g.y.edges[j+1],q),g.x.edges[i],g.x.edges[i+1],q)
end
function add_term!(dN,N,term::Birth,m::GaussLegendre,q,op::CompiledPBM2D,p,t,ctx)
    for j in axes(N,2),i in axes(N,1)
        dN[i,j]+=_tensor_integral(xy->_nonnegative(term.source(xy,p,t,ctx),"birth density"),op.grid,i,j,q)
    end
end
function add_term!(dN,N,term::DistributedNucleation,m::GaussLegendre,q,op::CompiledPBM2D,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"nucleation rate"); iszero(J) && return
    g=op.grid
    weights=[_tensor_integral(xy->_nonnegative(term.shape(xy,p,t,ctx),"source shape"),g,i,j,q) for i in axes(N,1),j in axes(N,2)]
    norm=term.normalization===:total_rate ? sum(weights) : one(J)
    isfinite(norm) && norm>0 || throw(NumericalFailure("unresolved 2D source"))
    dN .+= J/norm.*weights
end
function add_term!(dN,N,term::Union{Death,Dilution},m::CellAverageLoss,q,op::CompiledPBM2D,p,t,ctx)
    g=op.grid
    for j in axes(N,2),i in axes(N,1)
        dN[i,j]-=N[i,j]*_tensor_integral(xy->_nonnegative(term.rate(xy,p,t,ctx),"removal hazard"),g,i,j,q)/(g.x.widths[i]*g.y.widths[j])
    end
end
function add_term!(dN,N,term::Nucleation2D,m::GaussianInjection,cache,op::CompiledPBM2D,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"nucleation rate"); iszero(J) && return
    σ=m.width isa Tuple ? m.width : _source_value(m.width,p,t,ctx)
    σ=σ isa Real ? (σ,σ) : σ
    length(σ)==2 && all(x->isfinite(x) && x>0,σ) || throw(ArgumentError("Gaussian widths must be positive"))
    weights=map((op.grid.x,op.grid.y),term.size,σ) do axis,center,width
        [_normal_fraction((axis.edges[i]-center)/width,(axis.edges[i+1]-center)/width) for i in 1:length(axis)]
    end
    fractions=weights[1]*weights[2]'; norm=m.truncation===:renormalize ? sum(fractions) : one(J)
    norm>0 || throw(NumericalFailure("unresolved Gaussian source"))
    dN .+= J/norm.*fractions
end
