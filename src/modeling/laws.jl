"""
    AffineGrowth(offset, slope)

Autonomous size law `G(a)=offset+slope*a`. Each coefficient is a finite number
or a Symbol naming a parameter in `p`. This declaration rules out hidden time,
bulk-state and cross-coordinate dependence and permits compilation of the
Inguva transformation methods. Ordinary Growth/Growth2D callbacks remain valid
for the residual-based schemes.
"""
struct AffineGrowth{B,S}
    offset::B
    slope::S
    function AffineGrowth(offset,slope)
        all(v->v isa Symbol || (v isa Real && isfinite(v)),(offset,slope)) ||
            throw(ArgumentError("AffineGrowth coefficients must be finite numbers or parameter names"))
        new{typeof(offset),typeof(slope)}(offset,slope)
    end
end
function _affine_coefficient(v,p)
    value=if v isa Symbol
        hasproperty(p,v) || throw(ArgumentError("missing affine-growth parameter $v; supply p when constructing and compiling the mesh"))
        getproperty(p,v)
    else
        v
    end
    value isa Real && isfinite(value) || throw(ArgumentError("affine-growth coefficients must resolve to finite real values"))
    value
end
(g::AffineGrowth)(a,p,t,ctx)=_affine_coefficient(g.offset,p)+_affine_coefficient(g.slope,p)*a

"""
    SeparableGrowth2D(xlaw, ylaw)

Declare independent autonomous laws `G₁(a₁)` and `G₂(a₂)`, currently AffineGrowth
instances. Their transport operators commute. Cross-coordinate, time and
bulk-state dependence cannot be represented by this declaration.
"""
struct SeparableGrowth2D{X,Y}
    x::X
    y::Y
    function SeparableGrowth2D(x::AffineGrowth,y::AffineGrowth)
        new{typeof(x),typeof(y)}(x,y)
    end
end
(g::SeparableGrowth2D)(x,y,p,t,ctx)=(g.x(x,p,t,ctx),g.y(y,p,t,ctx))

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
    UniformVolumeDaughters(;quadrature_order=12)

Binary fragmentation uniform in daughter volume, expressed in a length
coordinate: b(x|y)=6x²/y³ for 0<x<y. The integral is two expected daughters;
the third length moment is conserved. Use AdditiveCoordinate(x->x^3,cbrt).
Moment rules integrate each daughter moment analytically; sectional rules use
positive Gauss–Legendre daughter counts before conservative fixed-pivot
deposition. These are fragments of one event, not probability weights.
"""
struct UniformVolumeDaughters{X,W}
    ratios::X
    counts::W
end
function UniformVolumeDaughters(;quadrature_order=12)
    quadrature_order isa Integer && quadrature_order>=3 || throw(ArgumentError("at least three daughter quadrature points required"))
    q=_gauss_legendre(GaussLegendre(quadrature_order))
    r=(q.nodes.+1)./2
    UniformVolumeDaughters(r,3 .*q.weights.*r.^2)
end
(d::UniformVolumeDaughters)(y,p,t,ctx)=((y*d.ratios[i],d.counts[i]) for i in eachindex(d.ratios))

"""
    daughter_moment(law,parent,k,p,t,ctx)

Optional analytical integral ∫x^k b(x|parent)dx per breakage event. Extend it
for a daughter law to avoid inner quadrature in moment weak forms. The law must
also supply its ordinary positive daughter quadrature for validation/sectional
use, conserving the declared additive measure. UniformVolumeDaughters supplies
6parent^k/(k+3). This is an expected-count moment, not a normalized probability.
"""
function daughter_moment end
function daughter_moment(::UniformVolumeDaughters,parent,k::Integer,p,t,ctx)
    k>=0 || throw(ArgumentError("daughter raw-moment order must be nonnegative"))
    6parent^k/(k+3)
end

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
