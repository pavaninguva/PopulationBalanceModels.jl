"""Two size velocities: a constant tuple or `rate(x,y,p,t,ctx)->(Gx,Gy)`."""
struct Growth2D{F} <: AbstractPBMTerm
    rate::F
end
_velocity(term::Growth2D,x,y,p,t,ctx)=term.rate isa Tuple ? term.rate : term.rate(x,y,p,t,ctx)
"""A total number source at a fixed pair of internal coordinates."""
struct Nucleation2D{F,S} <: AbstractPBMTerm
    rate::F
    size::S
    function Nucleation2D(rate,size::Tuple{Real,Real})
        all(x->isfinite(x) && x>=0,size) || throw(ArgumentError("invalid source location"))
        new{typeof(rate),typeof(size)}(rate,size)
    end
end
struct SizeBC2D{X,Y}
    x::X
    y::Y
    function SizeBC2D(;x=SizeBC(),y=SizeBC())
        x isa SizeBC && y isa SizeBC || throw(ArgumentError("each tensor axis needs SizeBC"))
        new{typeof(x),typeof(y)}(x,y)
    end
end
