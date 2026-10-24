abstract type AbstractSizeBoundary end
"""Zero external number flux. With SizeDiffusion this prescribes total
`G*n - ∂_L(D*n) = 0`; it is not a prescribed boundary density. An explicitly
selected BoundaryInjection replaces the zero total flux by its birth flux.
"""
struct NoFlux <: AbstractSizeBoundary end
"""Allow outward flux; impose zero distribution on incoming characteristics."""
struct OpenBoundary <: AbstractSizeBoundary end
"""Absorbing physical zero-size boundary. Only valid on the left edge at zero;
outgoing number disappears and carries zero material for positive powers of size.
The sectional closure uses the first-cell density for the outgoing trace.
"""
struct AbsorbingZero <: AbstractSizeBoundary end

"""
    MomentBoundary(; lower=0.0, upper=Inf, left=:error, right=:error)

Declared support and characteristic boundary treatment for MeasureEuler.
`:absorb` is allowed only at physical zero on the left; `:escape` records removal
at a finite computational edge; `:error` rejects a crossing. Atomic particles
are removed at their hitting time, without inferring a continuum density trace.
Residual-based moment solvers do not implement these boundary events.
"""
struct MomentBoundary
    lower::Float64
    upper::Float64
    left::Symbol
    right::Symbol
    function MomentBoundary(;lower=0.0,upper=Inf,left=:error,right=:error)
        isfinite(lower) && lower>=0 && upper>lower && !isnan(upper) || throw(ArgumentError("invalid moment support"))
        left in (:error,:escape,:absorb) && right in (:error,:escape) || throw(ArgumentError("invalid moment boundary policy"))
        left===:absorb && lower!=0 && throw(ArgumentError("absorption requires physical zero"))
        right===:escape && !isfinite(upper) && throw(ArgumentError("escape requires a finite edge"))
        new(lower,upper,left,right)
    end
end
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
