"""
    Growth2D(rate)
    Growth2D(; rate)

Conservative transport in two internal coordinates. Supply either constant
velocities `(Gx,Gy)` or `rate(x,y,p,t,ctx) -> (Gx,Gy)`, in coordinate/time units.
The callback is evaluated at each face using the current stage state and time.
Choose the face reconstruction separately with `TermMethods(growth=...)`.
"""
struct Growth2D{F} <: AbstractPBMTerm
    rate::F
end
Growth2D(;rate)=Growth2D(rate)
function _validate_velocity(velocity)
    velocity isa Union{Tuple,AbstractVector} && length(velocity)==2 &&
        all(v->v isa Real && isfinite(v),velocity) ||
        throw(ArgumentError("Growth2D rate must return two finite real coordinate velocities"))
    velocity
end
_velocity(term::Growth2D,x,y,p,t,ctx)=
    _validate_velocity(term.rate isa Tuple ? term.rate : term.rate(x,y,p,t,ctx))
"""A total number source at a fixed pair of internal coordinates."""
struct Nucleation2D{F,S} <: AbstractPBMTerm
    rate::F
    size::S
    function Nucleation2D(rate,size::Tuple{Real,Real})
        all(x->isfinite(x) && x>=0,size) || throw(ArgumentError("invalid source location"))
        new{typeof(rate),typeof(size)}(rate,size)
    end
end
Nucleation2D(;rate,size)=Nucleation2D(rate,size)
struct SizeBC2D{X,Y}
    x::X
    y::Y
    function SizeBC2D(;x=SizeBC(),y=SizeBC())
        x isa SizeBC && y isa SizeBC || throw(ArgumentError("each tensor axis needs SizeBC"))
        new{typeof(x),typeof(y)}(x,y)
    end
end
