"""Polynomial a₀+a₁x+…; each coefficient is numeric or a `(p,t,ctx)` callback."""
struct PolynomialRate{C}
    coefficients::C
    function PolynomialRate(coefficients)
        c=Tuple(coefficients)
        !isempty(c) || throw(ArgumentError("polynomial needs coefficients"))
        new{typeof(c)}(c)
    end
end
function (rate::PolynomialRate)(x,p,t,ctx)
    total=zero(x)
    for a in reverse(rate.coefficients)
        total=total*x+_source_value(a,p,t,ctx)
    end
    total
end
"""A size-independent aggregation coefficient; may depend on parameters/state/time."""
struct ConstantKernel{F}
    rate::F
end
(kernel::ConstantKernel)(x,y,p,t,ctx)=_source_value(kernel.rate,p,t,ctx)

"""Expected daughters `(ratio*parent,count)` with a declared conserved power."""
struct SelfSimilarDaughters{R,C}
    ratios::R
    counts::C
    measure_order::Int
    function SelfSimilarDaughters(ratios,counts;measure_order=1)
        r=Tuple(float.(ratios)); c=Tuple(float.(counts))
        length(r)==length(c)>0 || throw(ArgumentError("daughter ratios/counts must match"))
        all(x->isfinite(x) && 0<x<=1,r) && all(x->isfinite(x) && x>=0,c) || throw(ArgumentError("invalid daughter law"))
        measure_order isa Integer && measure_order>0 || throw(ArgumentError("invalid conserved power"))
        isapprox(sum(c[i]*r[i]^measure_order for i in eachindex(r)),1;rtol=1e-10) || throw(ArgumentError("daughter law does not conserve its declared power"))
        new{typeof(r),typeof(c)}(r,c,measure_order)
    end
end
(d::SelfSimilarDaughters)(x,p,t,ctx)=((x*d.ratios[i],d.counts[i]) for i in eachindex(d.ratios))

"""
Source raw moment `rate(p,t,ctx)*rawmoment(k,p,t,ctx)`. rawmoment is the kth
moment per created particle (including k=0). Use this to specify the integral
of a distributed source explicitly, rather than reconstruct it from population
quadrature nodes. A boundary birth at zero is Nucleation(rate,0.0).
"""
struct MomentSource{F,M} <: AbstractPBMTerm
    rate::F
    rawmoment::M
end
mechanism(::MomentSource)=:birth
