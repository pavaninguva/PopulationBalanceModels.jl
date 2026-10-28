"""
    UncertainInputs(source; map=identity, valid=x->true, names=nothing,
                    coordinates=:physical, iid=nothing, metadata=(;), ids=nothing)

Forward uncertainty without observations or a likelihood. `source` is a named
tuple of independent univariate distributions, a univariate/joint distribution,
a sampler `rng -> joint_sample`, or supplied samples (matrix columns or vector
entries). `map` deterministically maps a whole joint sample to physical inputs;
`valid` declares physical admissibility. Samples are never clipped or redrawn.

Distributions are iid sources. Supplied samples and custom samplers default to
`iid=false`; declaring `iid=true` is the caller's sampling assumption. Correlation
inside a joint distribution/sample is preserved. `coordinates` names the input
coordinates before `map`; record the transformation in `metadata`.

`UncertainInputs(parameter_draws(...))` preserves physical draws, inference
coordinates, identities and support flags. MCMC draws are not declared iid.
`UncertainInputs(estimation, posterior; n, seed)` selects through `parameter_draws`.
"""
struct UncertainInputs{S,F,V,N,M}
    source::S
    map::F
    valid::V
    names::N
    coordinates::Symbol
    iid::Bool
    metadata::M
    kind::Symbol
end

function UncertainInputs(source;map=identity,valid=x->true,names=nothing,
        coordinates=:physical,iid=nothing,metadata=(;),ids=nothing)
    coordinates isa Symbol || throw(ArgumentError("input coordinates must have a symbolic name"))
    metadata isa NamedTuple || throw(ArgumentError("input metadata must be a NamedTuple"))
    if source isa NamedTuple && haskey(source,:physical) && haskey(source,:z)
        method=get(source,:method,:imported_draws)
        identity_values=collect(get(source,:indices,1:length(source.physical)))
        meta=merge(metadata,(;source_method=method,inference_coordinates=source.coordinates,
            inference_draws=copy(source.z),support_flags=copy(source.in_support)))
        return UncertainInputs(source.physical;map,valid,coordinates=:physical,
            iid=method in (:laplace,:gauss_newton),metadata=meta,ids=identity_values)
    end
    kind=if source isa NamedTuple && !isempty(source) &&
            all(d->d isa Distributions.UnivariateDistribution,values(source))
        names===nothing || Tuple(names)==keys(source) || throw(ArgumentError("input names must follow the marginal order"))
        names=keys(source)
        :independent
    elseif source isa Distributions.Distribution
        :distribution
    elseif source isa AbstractMatrix || source isa AbstractVector
        isempty(source) && throw(ArgumentError("supplied input samples must be nonempty"))
        :samples
    elseif applicable(source,MersenneTwister(0))
        :sampler
    else
        throw(ArgumentError("use distributions, a joint rng->sample callback, or supplied samples"))
    end
    names=names===nothing ? nothing : Tuple(names)
    names===nothing || (!isempty(names) && all(x->x isa Symbol,names) && length(unique(names))==length(names)) ||
        throw(ArgumentError("input names must be distinct symbols"))
    ids===nothing || (metadata=merge(metadata,(;ids=collect(ids))))
    independent=iid===nothing ? kind in (:independent,:distribution) : Bool(iid)
    UncertainInputs(source,map,valid,names,coordinates,independent,metadata,kind)
end

function UncertainInputs(problem::EstimationProblem,posterior::PosteriorSamples;
        n=size(posterior.z,2)*size(posterior.z,3),seed=2022,kwargs...)
    draws=parameter_draws(problem,posterior;n,seed)
    UncertainInputs(draws;metadata=(;posterior_shape=size(posterior.z),selection_seed=seed),kwargs...)
end

_uq_finite(x::Real)=isfinite(x)
_uq_finite(x::Union{NamedTuple,Tuple,AbstractArray})=all(_uq_finite,x)
_uq_finite(x)=true
function _uq_named(input,x)
    input.names===nothing && return x
    x isa NamedTuple && keys(x)==input.names && return x
    values=x isa Number ? (x,) : Tuple(x)
    length(values)==length(input.names) || throw(DimensionMismatch("one joint input value per name is required"))
    NamedTuple{input.names}(values)
end

function _uq_samples(input,n,rng)
    source=input.source
    raw=if input.kind===:independent
        [NamedTuple{keys(source)}(Tuple(rand(rng,d) for d in values(source))) for _ in 1:n]
    elseif input.kind===:distribution
        [_uq_named(input,rand(rng,source)) for _ in 1:n]
    elseif input.kind===:sampler
        [_uq_named(input,source(rng)) for _ in 1:n]
    else
        available=source isa AbstractMatrix ? size(source,2) : length(source)
        n<=available || throw(ArgumentError("requested $n samples but only $available supplied samples are available"))
        [_uq_named(input,source isa AbstractMatrix ? copy(source[:,i]) : deepcopy(source[i])) for i in 1:n]
    end
    identities=get(input.metadata,:ids,collect(1:n))
    length(identities)>=n || throw(DimensionMismatch("missing supplied sample identities"))
    deepcopy(raw),collect(identities[1:n])
end

"""`MonteCarlo(n; seed=2022)` requests n serial joint realizations. The solve
budget is supplied separately to `propagate`; supplied sample sets are used in
their recorded order rather than resampled."""
struct MonteCarlo
    n::Int
    seed::Int
    function MonteCarlo(n::Integer;seed=2022)
        n>0 || throw(ArgumentError("Monte Carlo requires a positive sample count"))
        new(n,seed)
    end
end
