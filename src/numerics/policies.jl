struct MidpointRule <: AbstractTermMethod end
"""Gauss–Legendre bin quadrature with a specified point count."""
struct GaussLegendre <: AbstractTermMethod
    order::Int
    function GaussLegendre(n::Integer=4)
        n>=1 || throw(ArgumentError("quadrature order must be positive"))
        new(n)
    end
end
struct BinIntegral <: AbstractTermMethod end
struct FixedPivot <: AbstractTermMethod
    overflow::Symbol
    function FixedPivot(;overflow=:inherit)
        overflow in (:inherit,:error,:discard) || throw(ArgumentError("invalid fixed-pivot overflow policy"))
        new(overflow)
    end
end
struct PointwiseLoss <: AbstractTermMethod end
"""Cell-average a removal hazard against the piecewise-constant bin density."""
struct CellAverageLoss{Q} <: AbstractTermMethod
    quadrature::Q
end
CellAverageLoss(;quadrature=GaussLegendre(4))=CellAverageLoss(quadrature)
"""Represent a point source as a prescribed boundary number influx."""
struct BoundaryInjection <: AbstractTermMethod
    side::Symbol
    function BoundaryInjection(side::Symbol=:left)
        side in (:left,:right) || throw(ArgumentError("side must be :left or :right"))
        new(side)
    end
end
"""Finite-width numerical regularization of a point source; bin-integrated weights."""
struct GaussianInjection{W} <: AbstractTermMethod
    width::W
    truncation::Symbol
    function GaussianInjection(;width,truncation=:renormalize)
        truncation in (:renormalize,:physical) || throw(ArgumentError("invalid Gaussian truncation"))
        new{typeof(width)}(width,truncation)
    end
end
struct NativeResidual <: AbstractTermMethod end
struct ExactMomentRule <: AbstractTermMethod end
struct QuadratureMomentRule <: AbstractTermMethod end
