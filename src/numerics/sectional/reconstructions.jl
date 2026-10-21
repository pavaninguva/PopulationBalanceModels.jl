abstract type AbstractReconstruction <: AbstractTermMethod end
struct Upwind <: AbstractReconstruction end
"""Second-order centered face density on a uniform grid; no positivity limiter."""
struct Centered <: AbstractReconstruction end
"""Unlimited second-order upwind face extrapolation on a uniform grid; boundary faces fall back to Upwind."""
struct LinearUpwind <: AbstractReconstruction end
"""Limited linear FV reconstruction on nonuniform grids; check the time-step bound."""
struct MUSCL{T<:Real} <: AbstractReconstruction
    theta::T
    function MUSCL(;theta=1.5)
        1<=theta<=2 || throw(ArgumentError("theta must be in [1,2]"))
        new{typeof(theta)}(theta)
    end
end
"""Jiang-Shu finite-volume WENO5 on a uniform internal grid; no positivity limiter."""
struct WENO5{T<:Real} <: AbstractReconstruction
    epsilon::T
    function WENO5(; epsilon=1e-6)
        isfinite(epsilon) && epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
        new{typeof(epsilon)}(epsilon)
    end
end
