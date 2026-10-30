"""Tracked basis μ[0],…,μ[K] on one nonnegative internal coordinate. The selected moment method determines state storage: raw moments or, for DQMOM, weights and weighted abscissas. Compilation never adds orders or initial moments."""
struct MomentSet <: AbstractMomentBasis
    orders::Vector{Int}
    coordinate::Symbol
    function MomentSet(orders;coordinate=:volume)
        ks=collect(orders)
        !isempty(ks) && all(k->k isa Integer,ks) && ks==collect(0:last(ks)) ||
            throw(ArgumentError("moment orders must be contiguous integers starting at zero"))
        coordinate isa Symbol || throw(ArgumentError("coordinate must be a Symbol"))
        new(Int.(ks),coordinate)
    end
end
MomentSet(K::Integer;kwargs...)=MomentSet(0:K;kwargs...)
Base.length(g::MomentSet)=length(g.orders)
function moment(g::MomentSet,u::AbstractVector,k::Integer)
    k in g.orders || throw(ArgumentError("moment $k is not tracked"))
    length(u)==length(g) || throw(DimensionMismatch("moment state size mismatch"))
    u[k+1]
end
moment(g::MomentSet,u::AbstractMatrix,k::Integer)=[moment(g,view(u,:,j),k) for j in axes(u,2)]
number_concentration(g::MomentSet,u)=moment(g,u,0)
