abstract type AbstractPBMTerm end
struct TermSum{T<:Tuple} <: AbstractPBMTerm
    terms::T
end
_terms(a::AbstractPBMTerm) = (a,)
_terms(a::TermSum) = a.terms
Base.:+(a::AbstractPBMTerm, b::AbstractPBMTerm) = TermSum((_terms(a)..., _terms(b)...))

"""
    Growth(rate)
    Growth(; rate)

Conservative transport in the internal coordinate x, with PDE contribution
``-\\frac{\\partial (Gn)}{\\partial x}``. Supply `rate(x,p,t,ctx) -> G` in coordinate units per time;
this is a velocity, not a particle birth rate or a precomputed face flux.
The callback is evaluated at grid faces and at the current integration stage.
Select Upwind/MUSCL/WENO5 separately through `TermMethods(growth=...)`.
For eligible homogeneous growth, declare AffineGrowth and select the complete
Inguva2022 transport update with `compile(...; evolution=Inguva2022(),p)`.
"""
struct Growth{F} <: AbstractPBMTerm
    rate::F
end
"""
    SizeDiffusion(D)
    SizeDiffusion(; diffusivity=D)

Random-fluctuation growth-rate dispersion in one internal coordinate L:
``∂ₜn = ∂²_L(D n)``. Together with Growth the total flux is
``F = G n - ∂_L(D n)`` (Ito/Fokker–Planck convention), **not** Fickian
``-D ∂_L n`` for size-dependent D. D is a finite nonnegative constant or a
pure callback `D(L,p,t,ctx)` in coordinate²/time, evaluated at bin centres
using the current integration-stage context, including `ctx.bulk`.

Supports residual-based Grid1D sectional populations, standalone or in PBMSystem,
on uniform/nonuniform meshes with NoFlux at both ends or paired Periodic ends.
The default FiniteVolumeDiffusion acts on bin densities Nᵢ/widthᵢ. Select it with
`TermMethods(size_diffusion=FiniteVolumeDiffusion())`. BoundaryInjection adds
one prescribed total number flux, including without Growth. Other size
boundaries, moment/tensor representations and direct evolution are unsupported.
Use suggest_dt for the combined explicit growth/diffusion/loss recommendation;
it does not automatically control simulate's timestep.
"""
struct SizeDiffusion{F} <: AbstractPBMTerm
    diffusivity::F
end
"""Distributed number-density source `source(v,p,t,ctx)` per size per time."""
struct Birth{F} <: AbstractPBMTerm
    source::F
end
"""First-order removal with nonnegative hazard `rate(v,p,t,ctx)`."""
struct Death{F} <: AbstractPBMTerm
    rate::F
end
"""Uniform concentration dilution hazard; distinct from particle destruction."""
struct Dilution{F} <: AbstractPBMTerm
    rate::F
end
"""A time/state-dependent distributed source with explicit normalization semantics."""
struct DistributedNucleation{F,D} <: AbstractPBMTerm
    rate::F
    shape::D
    normalization::Symbol
    function DistributedNucleation(rate,shape;normalization=:total_rate)
        normalization in (:total_rate,:density_amplitude) || throw(ArgumentError("invalid source normalization"))
        new{typeof(rate),typeof(shape)}(rate,shape,normalization)
    end
end
"""Fixed, increasing coordinate map to an additive particle property and its inverse."""
struct AdditiveCoordinate{F,I}
    measure::F
    inverse::I
end
"""
    Nucleation(rate, size)
    Nucleation(; rate, size)

Point creation `J*δ(x-size)`, where `rate(p,t,ctx) -> J` is total number
per process-volume per time and `size` is a fixed location in the grid's
coordinate (not necessarily an additive property). The default FixedPivot
method requires a location inside the pivot range. For an edge source select
BoundaryInjection explicitly; for a physical size spread use GaussianNucleation.
Do not add the same creation event again as an InflowFlux boundary.
"""
struct Nucleation{F,T<:Real} <: AbstractPBMTerm
    rate::F
    size::T
    function Nucleation(rate, size::Real)
        isfinite(size) && size >= 0 || throw(ArgumentError("nucleation size must be nonnegative"))
        new{typeof(rate),typeof(size)}(rate, size)
    end
end

"""
    Aggregation(kernel; overflow=:error)

Symmetric binary kernel `kernel(v,w,p,t,ctx)`. A fixed-pivot deposition preserves
number of birth events and additive size inside the pivot range. :discard must
be explicit and its instantaneous losses are exposed by `diagnostics`.
"""
struct Aggregation{F,C} <: AbstractPBMTerm
    kernel::F
    overflow::Symbol
    coordinate_map::C
    function Aggregation(kernel; overflow=:error,coordinate_map=nothing)
        overflow in (:error, :discard) || throw(ArgumentError("overflow must be :error or :discard"))
        coordinate_map===nothing || coordinate_map isa AdditiveCoordinate || throw(ArgumentError("invalid additive map"))
        new{typeof(kernel),typeof(coordinate_map)}(kernel, overflow,coordinate_map)
    end
end

"""
    Breakage(rate, daughters; overflow=:error)

`rate(v,p,t,ctx)` is the parent hazard. `daughters(v,p,t,ctx)` returns an iterable
of `(daughter_size, expected_number)` pairs, including quadrature weights if
discretizing a continuous daughter distribution. The weighted additive size
must equal parent size; a probability density alone is not a daughter count.
"""
struct Breakage{F,D,C} <: AbstractPBMTerm
    rate::F
    daughters::D
    overflow::Symbol
    coordinate_map::C
    function Breakage(rate, daughters; overflow=:error,coordinate_map=nothing)
        overflow in (:error, :discard) || throw(ArgumentError("overflow must be :error or :discard"))
        coordinate_map===nothing || coordinate_map isa AdditiveCoordinate || throw(ArgumentError("invalid additive map"))
        new{typeof(rate),typeof(daughters),typeof(coordinate_map)}(rate, daughters, overflow,coordinate_map)
    end
end

"""Add a local source using `f!(dN,N,grid,p,t,ctx)`. Add to dN; do not reset it."""
struct CustomTerm{F} <: AbstractPBMTerm
    f!::F
end

struct PBMModel{T<:Tuple,N<:Tuple}
    terms::T
    names::N
    function PBMModel(terms::Tuple,names::Tuple)
        length(terms)==length(names) && all(t->t isa AbstractPBMTerm,terms) || throw(ArgumentError("model entries must be PopulationBalanceModels terms"))
        all(n->n isa Symbol,names) && length(unique(names))==length(names) || throw(ArgumentError("term names must be unique symbols"))
        new{typeof(terms),typeof(names)}(terms,names)
    end
end
PBMModel(term::AbstractPBMTerm) = PBMModel(_terms(term))
function PBMModel(terms::Tuple)
    names=Symbol[]
    for term in terms
        base=mechanism(term); name=base; index=1
        while name in names
            index+=1; name=Symbol(base,"_",index)
        end
        push!(names,name)
    end
    PBMModel(terms,Tuple(names))
end
PBMModel(terms::NamedTuple)=PBMModel(Tuple(values(terms)),Tuple(keys(terms)))
PBMModel(;kwargs...)=PBMModel((;kwargs...))
mechanism(::AbstractPBMTerm)=:custom


mechanism(::Growth)=:growth
mechanism(::SizeDiffusion)=:size_diffusion
mechanism(::Birth)=:birth
mechanism(::Union{Nucleation,DistributedNucleation})=:nucleation
mechanism(::Aggregation)=:aggregation
mechanism(::Breakage)=:breakage
mechanism(::Death)=:death
mechanism(::Dilution)=:dilution
mechanism(::CustomTerm)=:custom

# Named keyword forms expose physical inputs without changing positional APIs.
Growth(;rate)=Growth(rate)
SizeDiffusion(;diffusivity)=SizeDiffusion(diffusivity)
Birth(;source)=Birth(source)
Death(;rate)=Death(rate)
Dilution(;rate)=Dilution(rate)
Nucleation(;rate,size)=Nucleation(rate,size)
DistributedNucleation(;rate,shape,normalization=:total_rate)=
    DistributedNucleation(rate,shape;normalization)
