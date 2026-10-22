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
        hasproperty(p,v) || throw(ArgumentError("missing growth-law parameter $v; supply p when constructing and compiling the mesh"))
        getproperty(p,v)
    else
        v
    end
    value isa Real && isfinite(value) || throw(ArgumentError("growth-law coefficients must resolve to finite real values"))
    value
end
(g::AffineGrowth)(a,p,t,ctx)=_affine_coefficient(g.offset,p)+_affine_coefficient(g.slope,p)*a

"""
    SeparableGrowth2D(xlaw, ylaw)

Declare independent directional laws `G₁(a₁,p,t)` and `G₂(a₂,p,t)`. Each solver
validates the constituent laws: characteristic shifts require autonomous
AffineGrowth; HOC also accepts ExponentialGrowth, TimeScaledGrowth and declared
HOCGrowthLaw instances. Cross-coordinate and evolving bulk dependence are
excluded from those direct solvers.
"""
struct SeparableGrowth2D{X,Y}
    x::X
    y::Y
    function SeparableGrowth2D(x,y)
        new{typeof(x),typeof(y)}(x,y)
    end
end
(g::SeparableGrowth2D)(x,y,p,t,ctx)=(g.x(x,p,t,ctx),g.y(y,p,t,ctx))

"""
    ExponentialGrowth(amplitude, exponent)

Autonomous law `G(x)=amplitude*exp(exponent*x)`. Coefficients are finite numbers
or parameter names, as in `AffineGrowth`. HOC checks positivity on the domain.
"""
struct ExponentialGrowth{A,B}
    amplitude::A
    exponent::B
    function ExponentialGrowth(a,b)
        all(v->v isa Symbol || (v isa Real && isfinite(v)),(a,b)) ||
            throw(ArgumentError("ExponentialGrowth coefficients must be finite numbers or parameter names"))
        new{typeof(a),typeof(b)}(a,b)
    end
end
(g::ExponentialGrowth)(x,p,t,ctx)=_affine_coefficient(g.amplitude,p)*exp(_affine_coefficient(g.exponent,p)*x)

"""
    PrescribedRate(callback; dependencies=(:time, :parameters))

Declare a smooth prescribed `(p,t,ctx)` rate, independent of the evolving
population and bulk states. Allowed dependencies for HOC are `:time`,
`:parameters`, and prescribed `:fields`. This is a user contract, not a proof.
For a 2D HOC incoming face flux it denotes flux density per transverse size;
the solver integrates it over each transverse bin. It must be uniform along
that face. Ordinary unwrapped callbacks remain supported by residual methods.
"""
struct PrescribedRate{F,D}
    callback::F
    dependencies::D
    function PrescribedRate(f;dependencies=(:time,:parameters))
        d=Tuple(dependencies)
        all(x->x isa Symbol,d) || throw(ArgumentError("dependencies must be symbols"))
        new{typeof(f),typeof(d)}(f,d)
    end
end
(r::PrescribedRate)(p,t,ctx)=r.callback(p,t,ctx)

"""
    TimeScaledGrowth(law, factor::PrescribedRate)

Positive prescribed modulation of an autonomous spatial law. HOC evaluates
the factor at the midpoint of each full time step, without an autonomous
`G*f` transformation. No population or evolving bulk feedback is supported.
"""
struct TimeScaledGrowth{L,F}
    law::L
    factor::F
    TimeScaledGrowth(law,factor::PrescribedRate)=new{typeof(law),typeof(factor)}(law,factor)
end
(g::TimeScaledGrowth)(x,p,t,ctx)=g.law(x,p,t,ctx)*g.factor(p,t,ctx)

"""
    HOCGrowthLaw(rate; lower_bound, dependencies=(:coordinate,:parameters), periodic=false)

Declare a smooth (C⁶), strictly positive directional law `rate(x,p,t,ctx)` for HOC.
`lower_bound(p,t,ctx)` must be a positive lower bound on the entire compiled
axis at that time. `periodic=true` additionally declares a smooth periodic
extension, including derivatives. These assumptions are supplied by the user;
sampling does not certify them. HOC checks the bound and every numerical face
value on problem construction and every update. Allowed dependencies are the
own `:coordinate`, `:time`, `:parameters`, and prescribed `:fields`.
"""
struct HOCGrowthLaw{F,B,D}
    rate::F
    lower_bound::B
    dependencies::D
    periodic::Bool
    function HOCGrowthLaw(rate;lower_bound,dependencies=(:coordinate,:parameters),periodic=false)
        d=Tuple(dependencies)
        all(x->x isa Symbol,d) || throw(ArgumentError("dependencies must be symbols"))
        new{typeof(rate),typeof(lower_bound),typeof(d)}(rate,lower_bound,d,periodic)
    end
end
(g::HOCGrowthLaw)(x,p,t,ctx)=g.rate(x,p,t,ctx)

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
