"""
    Grid1D(edges; coordinate=:volume)

Internal-coordinate bins. State `N[i,c]` is the number concentration integrated
over bin i, per physical volume in cell c. `density(grid,N)` divides by bin width.
Aggregation and conservative breakage require an additive :volume or :mass axis.
"""
struct Grid1D{T<:AbstractFloat} <: AbstractPopulationRepresentation
    edges::Vector{T}
    centers::Vector{T}
    widths::Vector{T}
    coordinate::Symbol
    function Grid1D(edges; coordinate=:volume)
        e = float.(collect(edges))
        length(e) >= 3 || throw(ArgumentError("at least two bins are required"))
        all(isfinite, e) && all(diff(e) .> 0) ||
            throw(ArgumentError("edges must be finite and strictly increasing"))
        first(e) >= 0 || throw(ArgumentError("internal coordinates must be nonnegative"))
        coordinate isa Symbol || throw(ArgumentError("coordinate must be a Symbol"))
        new{eltype(e)}(e, (e[1:end-1] .+ e[2:end]) ./ 2, diff(e), coordinate)
    end
end
Base.length(g::Grid1D) = length(g.centers)
density(g::Grid1D, N::AbstractVector) = N ./ g.widths
density(g::Grid1D, N::AbstractMatrix) = N ./ g.widths
moment(g::Grid1D, N::AbstractVector, order::Real) = sum((g.centers .^ order) .* N)
moment(g::Grid1D, N::AbstractMatrix, order::Real) = vec(sum((g.centers .^ order) .* N; dims=1))
number_concentration(g::Grid1D, N) = moment(g, N, 0)


"""Two internal coordinates; N[i,j] is bin-integrated number, not a physical-space mesh."""
struct Grid2D{X,Y} <: AbstractPopulationRepresentation
    x::X
    y::Y
    Grid2D(x::Grid1D,y::Grid1D)=new{typeof(x),typeof(y)}(x,y)
end
Grid2D(xedges,yedges;coordinates=(:length,:width))=Grid2D(Grid1D(xedges;coordinate=coordinates[1]),Grid1D(yedges;coordinate=coordinates[2]))
Base.size(g::Grid2D)=(length(g.x),length(g.y))
Base.length(g::Grid2D)=length(g.x)*length(g.y)
_areas(g::Grid2D)=g.x.widths*g.y.widths'
density(g::Grid2D,N::AbstractMatrix)=N./_areas(g)
function moment(g::Grid2D,N::AbstractMatrix,orders::Tuple{Integer,Integer})
    size(N)==size(g) || throw(DimensionMismatch("tensor state size mismatch"))
    sum(N.*((g.x.centers.^orders[1])*(g.y.centers.^orders[2])'))
end
number_concentration(g::Grid2D,N)=sum(N)
