abstract type AbstractSizeBoundary end
struct NoFlux <: AbstractSizeBoundary end
"""Allow outward flux; impose zero distribution on incoming characteristics."""
struct OpenBoundary <: AbstractSizeBoundary end
struct Periodic <: AbstractSizeBoundary end
"""An inward number flux `rate(p,t,ctx)` at a size boundary, independent of G."""
struct InflowFlux{F} <: AbstractSizeBoundary
    rate::F
end
struct SizeBC{L<:AbstractSizeBoundary,R<:AbstractSizeBoundary}
    left::L
    right::R
    function SizeBC(left::AbstractSizeBoundary=NoFlux(), right::AbstractSizeBoundary=NoFlux())
        (left isa Periodic) == (right isa Periodic) ||
            throw(ArgumentError("periodicity must be set at both ends"))
        new{typeof(left),typeof(right)}(left, right)
    end
end
