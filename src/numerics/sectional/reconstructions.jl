abstract type AbstractReconstruction <: AbstractTermMethod end
struct Upwind <: AbstractReconstruction end
"""Second-order centered face density on a uniform grid; no positivity limiter."""
struct Centered <: AbstractReconstruction end
"""Unlimited second-order upwind face extrapolation on a uniform grid; boundary faces fall back to Upwind."""
struct LinearUpwind <: AbstractReconstruction end
"""Geometry-aware harmonic (van Leer) limited linear reconstruction on 1D grids."""
struct VanLeer <: AbstractReconstruction end
"""Third-order two-stencil finite-volume WENO on 1D uniform or nonuniform grids; no positivity limiter."""
struct WENO3{T<:Real} <: AbstractReconstruction
    epsilon::T
    function WENO3(;epsilon=1e-6)
        isfinite(epsilon) && epsilon>0 || throw(ArgumentError("epsilon must be positive"))
        new{typeof(epsilon)}(epsilon)
    end
end
"""Limited linear FV reconstruction on nonuniform grids; check the time-step bound."""
struct MUSCL{T<:Real} <: AbstractReconstruction
    theta::T
    function MUSCL(;theta=1.5)
        1<=theta<=2 || throw(ArgumentError("theta must be in [1,2]"))
        new{typeof(theta)}(theta)
    end
end
"""Jiang–Shu finite-volume WENO5 on uniform or nonuniform fixed grids (at least six bins). Geometry is prepared during compilation; no positivity limiter."""
struct WENO5{T<:Real} <: AbstractReconstruction
    epsilon::T
    function WENO5(; epsilon=1e-6)
        isfinite(epsilon) && epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
        new{typeof(epsilon)}(epsilon)
    end
end
