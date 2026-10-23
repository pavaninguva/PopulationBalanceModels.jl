abstract type AbstractObservable end
struct MomentObservable{K} <: AbstractObservable
    order::K
    cell::Int
end
MomentObservable(order;cell=1)=MomentObservable(order,cell)
struct HistogramObservable{T} <: AbstractObservable
    edges::Vector{T}
    cell::Int
    function HistogramObservable(edges;cell=1)
        e=float.(collect(edges))
        length(e)>=2 && all(isfinite,e) && all(diff(e).>0) || throw(ArgumentError("histogram edges must be increasing"))
        new{eltype(e)}(e,cell)
    end
end
"""Observation callback `f(N,operator,p,t)` returning a scalar or vector."""
struct FunctionObservable{F} <: AbstractObservable
    f::F
end
struct PopulationMoment{K} <: AbstractObservable
    population::Symbol
    order::K
end
struct BulkObservable <: AbstractObservable
    name::Symbol
end
struct PopulationHistogram{H,W} <: AbstractObservable
    population::Symbol
    histogram::H
    weight::W
    normalize::Bool
end
_unitweight(x)=one(x)
function PopulationHistogram(population::Symbol,edges;weight=_unitweight,normalize=false)
    PopulationHistogram(population,HistogramObservable(edges),weight,normalize)
end
struct SystemObservable{F} <: AbstractObservable
    f::F
end
_vector(x::Number)=[x]
_vector(x)=vec(collect(x))
_observe(obs::MomentObservable,N,op,p,t)=[moment(op.grid,view(N,:,obs.cell),obs.order)]
function _observe(obs::MomentObservable,N,op::CompiledMomentPBM,p,t)
    obs.cell==1 || throw(ArgumentError("a batch moment model has one population block"))
    [population_moment(op,N,obs.order)]
end
function _observe(obs::MomentObservable,N,op::CompiledPBM2D,p,t)
    obs.cell==1 || throw(ArgumentError("a batch 2D distribution has no physical cell index"))
    obs.order isa Tuple{Integer,Integer} || throw(ArgumentError("use MomentObservable((kx,ky)) for a 2D distribution"))
    [moment(op.grid,N,obs.order)]
end
_observe(obs::HistogramObservable,N,op::CompiledMomentPBM,p,t)=throw(ArgumentError("histograms require an explicit distribution reconstruction/measurement model for moments"))
_observe(obs::HistogramObservable,N,op::CompiledPBM2D,p,t)=throw(ArgumentError("use FunctionObservable for a 2D bin or marginal measurement"))
function _observe(obs::MomentObservable,N,op::Union{CompiledCharacteristic,CompiledHOC},p,t)
    obs.cell==1 || throw(ArgumentError("a batch direct-transport population has one physical cell"))
    op.grid isa Grid1D && return [moment(op.grid,view(N,:,obs.cell),obs.order)]
    obs.order isa Tuple{Integer,Integer} || throw(ArgumentError("use MomentObservable((kx,ky)) for a 2D direct-transport state"))
    [moment(op.grid,N,obs.order)]
end
function _observe(obs::HistogramObservable,N,op::Union{CompiledCharacteristic,CompiledHOC},p,t)
    op.grid isa Grid1D || throw(ArgumentError("use FunctionObservable for a 2D bin or marginal measurement"))
    _observe(obs,N,(;grid=op.grid),p,t)
end
_observe(obs::FunctionObservable,N,op,p,t)=_vector(obs.f(N,op,p,t))
_observe(obs::PopulationMoment,N,op::CompiledSystem,p,t)=
    [population_moment(getproperty(op.system.populations,obs.population),population(op,N,obs.population),obs.order)]
_observe(obs::BulkObservable,N,op::CompiledSystem,p,t)=[bulk(op,N,obs.name)]
_observe(obs::SystemObservable,N,op::CompiledSystem,p,t)=_vector(obs.f(stateview(op,N),op,p,t))
function _observe(obs::PopulationHistogram,N,op::CompiledSystem,p,t)
    popop=getproperty(op.system.populations,obs.population)
    popop isa CompiledMomentPBM && throw(ArgumentError("a moment population needs an explicit distribution reconstruction/measurement model for histograms"))
    weights=obs.weight.(popop.grid.centers)
    all(isfinite,weights) && all(weights.>=0) || throw(ArgumentError("histogram observation weights must be finite and nonnegative"))
    weighted=weights.*population(op,N,obs.population)
    result=_observe(obs.histogram,reshape(weighted,:,1),popop,p,t)
    if obs.normalize
        total=sum(result)
        total>0 || throw(NumericalFailure("cannot normalize an empty measured distribution"))
        result ./= total
    end
    result
end
function _observe(obs::HistogramObservable,N,op,p,t)
    e=obs.edges; g=op.grid
    # Experimental bins need not coincide with the internal grid. Reconstruction
    # here is piecewise constant and linear in bin concentrations.
    e[1]>=g.edges[1] && e[end]<=g.edges[end] || throw(ArgumentError("observation histogram lies outside model grid"))
    y=zeros(eltype(N),length(e)-1)
    for j in eachindex(y), i in 1:length(g)
        overlap=max(zero(eltype(e)),min(e[j+1],g.edges[i+1])-max(e[j],g.edges[i]))
        y[j]+=overlap/g.widths[i]*N[i,obs.cell]
    end
    y
end

abstract type AbstractObservation end

"""
    Observation(times, data, observable; sigma=1.0, covariance=nothing)

Rows are measured outputs; columns are times. Known independent Gaussian
standard deviations can be a scalar or a matrix matching data. For a scalar
observable, a vector of data is accepted. Missing data must be explicitly
selected before construction. Alternatively supply a fixed positive-definite
`covariance` for `vec(data)` (outputs within each time, then times). It can
include correlations across outputs and times within this observation. Do not
supply sigma as well. Different Observation objects are independent blocks.
Normalized histograms have a singular covariance and need independent contrasts
or a likelihood for compositional data.
"""
struct Observation{T,Y,O,S,F} <: AbstractObservation
    times::T
    data::Y
    observable::O
    sigma::S
    factor::F
end
function Observation(times,data,observable::AbstractObservable;sigma=nothing,covariance=nothing)
    ts=float.(collect(times)); Y=data isa AbstractVector ? reshape(float.(copy(data)),1,:) : float.(copy(data))
    !isempty(ts) && all(isfinite,ts) && all(diff(ts).>0) || throw(ArgumentError("observation times must increase"))
    Y isa AbstractMatrix && size(Y,2)==length(ts) || throw(DimensionMismatch("one data column per time required"))
    all(x->x isa Real && isfinite(x),Y) || throw(ArgumentError("data must be finite and real"))
    if covariance!==nothing
        sigma===nothing || throw(ArgumentError("supply sigma or covariance, not both"))
        C=float.(Matrix(covariance))
        size(C)==(length(Y),length(Y)) || throw(DimensionMismatch("covariance must match vec(data)"))
        all(x->x isa Real && isfinite(x),C) && issymmetric(C) || throw(ArgumentError("covariance must be finite, real and symmetric"))
        factor=try
            cholesky(Symmetric(C))
        catch e
            e isa PosDefException || rethrow()
            throw(ArgumentError("covariance must be positive definite; remove dependent measurements"))
        end
        return Observation(ts,Y,observable,reshape(sqrt.(diag(C)),size(Y)),factor)
    end
    sigma===nothing && (sigma=1.0)
    S=sigma isa Number ? fill(float(sigma),size(Y)) : float.(copy(sigma))
    size(S)==size(Y) || throw(DimensionMismatch("sigma must match data"))
    all(x->x isa Real && isfinite(x) && x>0,S) || throw(ArgumentError("standard deviations must be positive, finite and real"))
    Observation(ts,Y,observable,S,nothing)
end

_whiten(obs,raw)=obs.factor===nothing ? raw./vec(obs.sigma) : obs.factor.L\raw
_noise_logdet(obs)=obs.factor===nothing ? 2sum(log,obs.sigma) : 2sum(log,diag(obs.factor.L))
