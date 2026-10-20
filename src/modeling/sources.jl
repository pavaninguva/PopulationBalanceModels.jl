"""Bin-integrated number source `integral(lo,hi,p,t,ctx)` per time."""
struct IntegratedBirth{F} <: AbstractPBMTerm
    integral::F
end

"""
    IntegratedNucleation(rate, fraction; normalization=:total_rate)

`fraction(lo,hi,p,t,ctx)` integrates a nonnegative source shape over a bin.
`:total_rate` renormalizes on the resolved domain so sum(dN)=rate.
`:density_amplitude` uses the supplied bin integrals without renormalization.
For a physical probability density this preserves the fraction inside the domain.
"""
struct IntegratedNucleation{F,Q} <: AbstractPBMTerm
    rate::F
    fraction::Q
    normalization::Symbol
    function IntegratedNucleation(rate,fraction;normalization=:total_rate)
        normalization in (:total_rate,:density_amplitude) || throw(ArgumentError("invalid source normalization"))
        new{typeof(rate),typeof(fraction)}(rate,fraction,normalization)
    end
end

"""
    GaussianNucleation(rate; center, width, truncation=:renormalize)

One-dimensional Gaussian creation with bin-integrated weights. `center` and
standard-deviation `width` are numbers or `(p,t,ctx)` callbacks.
`:renormalize` creates the full rate on the finite domain;
`:physical` retains only the physical Gaussian fraction inside that domain.
Width is a physical model parameter, independent of grid spacing.
"""
function GaussianNucleation(rate;center,width,truncation=:renormalize)
    truncation in (:renormalize,:physical) || throw(ArgumentError("invalid Gaussian truncation policy"))
    fraction=function(lo,hi,p,t,ctx)
        μ=_source_value(center,p,t,ctx); σ=_source_value(width,p,t,ctx)
        isfinite(μ) && isfinite(σ) && σ>0 || throw(ArgumentError("Gaussian center must be finite and width positive"))
        _normal_fraction((lo-μ)/σ,(hi-μ)/σ)
    end
    IntegratedNucleation(rate,fraction;normalization=truncation===:renormalize ? :total_rate : :density_amplitude)
end

mechanism(::IntegratedBirth)=:birth
mechanism(::IntegratedNucleation)=:nucleation
