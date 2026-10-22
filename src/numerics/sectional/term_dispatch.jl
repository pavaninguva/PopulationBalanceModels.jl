default_method(::Growth,::Grid1D)=Upwind()
default_method(::Union{Birth,DistributedNucleation},::Grid1D)=MidpointRule()
default_method(::Union{IntegratedBirth,IntegratedNucleation},::Grid1D)=BinIntegral()
default_method(::Union{Death,Dilution},::Grid1D)=PointwiseLoss()
default_method(::Union{Nucleation,Aggregation,Breakage},::Grid1D)=FixedPivot()
default_method(::CustomTerm,::Grid1D)=NativeResidual()
default_method(term,representation)=throw(ArgumentError("no default numerical method; select a method explicitly for $(typeof(term))"))

prepare_term(::Growth,m::AbstractReconstruction,g::Grid1D;kwargs...)=_validate_reconstruction(m,g)
prepare_term(::Union{Birth,DistributedNucleation},::MidpointRule,::Grid1D;kwargs...)=nothing
prepare_term(::Union{Birth,DistributedNucleation},m::GaussLegendre,::Grid1D;kwargs...)=_gauss_legendre(m)
prepare_term(::Union{IntegratedBirth,IntegratedNucleation},::BinIntegral,::Grid1D;kwargs...)=nothing
prepare_term(::Union{Death,Dilution},::PointwiseLoss,::Grid1D;kwargs...)=nothing
prepare_term(::Union{Death,Dilution},m::CellAverageLoss,::Grid1D;kwargs...)=_gauss_legendre(m.quadrature)
function prepare_term(term::Aggregation,m::FixedPivot,::Grid1D;kwargs...)
    m.overflow===:inherit ? term : Aggregation(term.kernel;overflow=m.overflow,coordinate_map=term.coordinate_map)
end
function prepare_term(term::Breakage,m::FixedPivot,::Grid1D;kwargs...)
    m.overflow===:inherit ? term : Breakage(term.rate,term.daughters;overflow=m.overflow,coordinate_map=term.coordinate_map)
end
prepare_term(::CustomTerm,::NativeResidual,::Grid1D;kwargs...)=nothing
function prepare_term(term::Nucleation,m::FixedPivot,g::Grid1D;kwargs...)
    m.overflow===:discard || first(g.centers)<=term.size<=last(g.centers) || throw(ArgumentError("fixed-pivot nucleation must lie within the pivot range"))
    nothing
end
add_term!(dN,N,term::Union{Aggregation,Breakage},m::FixedPivot,prepared,op::CompiledPBM,p,t,ctx)=_local_rhs!(dN,N,prepared,op,p,t,ctx)
# The cached collision term contains the selected overflow override. All
# accounting and positivity-policy paths must use the same effective term.
_effective_term(item::PreparedTerm)=item.term
_effective_term(item::PreparedTerm{<:Union{Aggregation,Breakage},<:FixedPivot})=item.cache
function add_term!(dN,N,term::Nucleation,m::FixedPivot,cache,op::CompiledPBM,p,t,ctx)
    rate=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
    _deposit!(dN,op.grid.centers,term.size,rate,m.overflow===:inherit ? :error : m.overflow)
end
function prepare_term(term::Nucleation,m::BoundaryInjection,g::Grid1D;bc,kwargs...)
    edge=m.side===:left ? first(g.edges) : last(g.edges)
    term.size==edge || throw(ArgumentError("boundary source location must equal the selected domain edge"))
    face=m.side===:left ? bc.left : bc.right
    face isa Periodic && throw(ArgumentError("boundary injection is incompatible with periodic size space"))
    face isa InflowFlux && throw(ArgumentError("boundary influx is already specified; avoid double-counting nucleation"))
    nothing
end
function prepare_term(term::Nucleation,m::GaussianInjection,g::Grid1D;kwargs...)
    GaussianNucleation(term.rate;center=term.size,width=m.width,truncation=m.truncation)
end
function _validate_reconstruction(m,g)
    if m isa Union{VanLeer,WENO3}
        length(g)>=3 || throw(ArgumentError("VanLeer and WENO3 require at least three bins"))
    end
    if m isa Union{Centered,LinearUpwind}
        length(g)>=3 && maximum(abs.(g.widths.-first(g.widths)))<=1e-10*first(g.widths) ||
            throw(ArgumentError("Centered and LinearUpwind require a uniform grid with at least three bins"))
    end
    if m isa WENO5
        length(g)>=6 && maximum(abs.(g.widths.-first(g.widths)))<=1e-10*first(g.widths) ||
            throw(ArgumentError("WENO5 requires a uniform grid with at least six bins"))
    end
    nothing
end

function add_term!(dN,N,term::Growth,m::AbstractReconstruction,cache,op,p,t,ctx)
    fluxop=(;grid=op.grid,bc=op.bc,scheme=m)
    _local_rhs!(dN,N,term,fluxop,p,t,ctx)
end
function add_term!(dN,N,term,m::Union{MidpointRule,BinIntegral,FixedPivot,PointwiseLoss,NativeResidual},cache,op::CompiledPBM,p,t,ctx)
    _local_rhs!(dN,N,term,op,p,t,ctx)
end
function add_term!(dN,N,term::Birth,m::GaussLegendre,q,op,p,t,ctx)
    for i in eachindex(N)
        dN[i]+=_integrate_bin(x->_nonnegative(term.source(x,p,t,ctx),"birth density"),op.grid.edges[i],op.grid.edges[i+1],q)
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
function add_term!(dN,N,term::DistributedNucleation,m::GaussLegendre,q,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
    iszero(J) && return (zero(eltype(dN)),zero(eltype(dN)))
    w=[_integrate_bin(x->_nonnegative(term.shape(x,p,t,ctx),"source shape"),op.grid.edges[i],op.grid.edges[i+1],q) for i in eachindex(N)]
    norm=term.normalization===:total_rate ? sum(w) : one(J)
    isfinite(norm) && norm>0 || throw(NumericalFailure("source quadrature has no resolved weight"))
    dN .+= (J/norm).*w
    (zero(eltype(dN)),zero(eltype(dN)))
end
function add_term!(dN,N,term::Union{Death,Dilution},m::CellAverageLoss,q,op,p,t,ctx)
    for i in eachindex(N)
        hazard=_integrate_bin(x->_nonnegative(term.rate(x,p,t,ctx),"removal hazard"),op.grid.edges[i],op.grid.edges[i+1],q)/op.grid.widths[i]
        dN[i]-=hazard*N[i]
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
function add_term!(dN,N,term::Nucleation,m::BoundaryInjection,cache,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"boundary nucleation rate")
    if !iszero(J)
        growth=only(filter(term->term isa Growth,op.model.terms))
        velocity=growth.rate(term.size,p,t,ctx)
        isfinite(velocity) && (m.side===:left ? velocity>=0 : velocity<=0) || throw(ArgumentError("active boundary nucleation requires inward or zero growth velocity"))
    end
    dN[m.side===:left ? 1 : length(N)]+=J
    (zero(eltype(dN)),zero(eltype(dN)))
end
function add_term!(dN,N,term::Nucleation,m::GaussianInjection,source,op,p,t,ctx)
    _local_rhs!(dN,N,source,op,p,t,ctx)
end
