abstract type AbstractInferenceTransform end

"""Identity internal coordinates, θ=location+scale*z. Explicit identity never requests automatic selection."""
struct IdentityTransform <: AbstractInferenceTransform
    location::Float64
    scale::Float64
    function IdentityTransform(;location=0.0,scale=1.0)
        isfinite(location) && isfinite(scale) && scale>0 || throw(ArgumentError("invalid coordinate scale"))
        new(location,scale)
    end
end
"""Positive parameter θ=scale*exp(z). The reference scale changes internal coordinates, not the physical MAP target."""
struct LogTransform <: AbstractInferenceTransform
    scale::Float64
    function LogTransform(;scale=1.0)
        isfinite(scale) && scale>0 || throw(ArgumentError("log scale must be positive"))
        new(scale)
    end
end
"""Open interval θ=lower+(upper-lower)*logistic(z). Endpoints are not regular interior MAP points."""
struct LogitTransform <: AbstractInferenceTransform
    lower::Float64
    upper::Float64
    function LogitTransform(lower,upper)
        isfinite(lower) && isfinite(upper) && lower<upper && isfinite(upper-lower) || throw(ArgumentError("finite ordered bounds and width required"))
        new(lower,upper)
    end
end
"""
    ShiftedLogTransform(bound; upper=false, scale=1.0)

One-sided internal coordinates: θ=bound+scale*exp(z), or
θ=bound-scale*exp(z) with `upper=true`. Its absolute Jacobian is scale*exp(z).
No endpoint clipping or overflow repair is applied.
"""
struct ShiftedLogTransform <: AbstractInferenceTransform
    bound::Float64
    upper::Bool
    scale::Float64
    function ShiftedLogTransform(bound;upper=false,scale=1.0)
        isfinite(bound) && isfinite(scale) && scale>0 && upper isa Bool || throw(ArgumentError("finite one-sided bound and positive scale required"))
        new(bound,upper,scale)
    end
end
_physical(t::IdentityTransform,z)=t.location+t.scale*z
_physical(t::LogTransform,z)=exp(log(t.scale)+z)
_physical(t::ShiftedLogTransform,z)=t.bound+(t.upper ? -1 : 1)*exp(log(t.scale)+z)
_logistic(z)=z>=0 ? inv(1+exp(-z)) : exp(z)/(1+exp(z))
_physical(t::LogitTransform,z)=t.lower+(t.upper-t.lower)*_logistic(z)
_coordinate(t::IdentityTransform,x)=(x-t.location)/t.scale
_coordinate(t::LogTransform,x)=x>0 ? log(x)-log(t.scale) : throw(ArgumentError("log parameter must be positive"))
function _coordinate(t::ShiftedLogTransform,x)
    delta=t.upper ? t.bound-x : x-t.bound
    delta>0 || throw(ArgumentError("parameter must lie strictly $(t.upper ? "below" : "above") $(t.bound)"))
    log(delta)-log(t.scale)
end
_coordinate(t::LogitTransform,x)=t.lower<x<t.upper ? log(x-t.lower)-log(t.upper-x) : throw(ArgumentError("initial value must be strictly inside bounds"))
_logjac(t::IdentityTransform,z)=log(t.scale)
_logjac(t::Union{LogTransform,ShiftedLogTransform},z)=log(t.scale)+z
_softplus(x)=x>zero(x) ? x+log1p(exp(-x)) : log1p(exp(x))
_logjac(t::LogitTransform,z)=log(t.upper-t.lower)-_softplus(z)-_softplus(-z)
_transform_bounds(::IdentityTransform)=(-Inf,Inf)
_transform_bounds(::LogTransform)=(0.0,Inf)
_transform_bounds(t::LogitTransform)=(t.lower,t.upper)
_transform_bounds(t::ShiftedLogTransform)=t.upper ? (-Inf,t.bound) : (t.bound,Inf)
_transform_bounds(::AbstractInferenceTransform)=nothing
_transform_valid(t::AbstractInferenceTransform,z)=isfinite(_physical(t,z))
_transform_valid(t::LogTransform,z)=isfinite(_physical(t,z)) && _physical(t,z)>0
_transform_valid(t::LogitTransform,z)=t.lower<_physical(t,z)<t.upper
_transform_valid(t::ShiftedLogTransform,z)=isfinite(_physical(t,z)) && (t.upper ? _physical(t,z)<t.bound : _physical(t,z)>t.bound)
_transform_derivative(t::AbstractInferenceTransform,z)=ForwardDiff.derivative(v->_physical(t,v),z)

"""
    Prior(distribution; coordinates)

Advanced/compatibility density declaration. Prefer passing a continuous
univariate distribution directly to ParameterSpec: it is a physical prior.
`:physical` is equivalent. An explicit `:inference` prior requires an explicit
transform: physical MAP subtracts its forward log Jacobian; sampling restores
that Jacobian exactly once. Its probability law is never reinterpreted.
"""
struct Prior{D}
    distribution::D
    coordinates::Symbol
    function Prior(distribution;coordinates)
        distribution isa Distributions.ContinuousUnivariateDistribution || throw(ArgumentError("parameter priors must be proper continuous univariate Distributions.jl distributions"))
        coordinates in (:physical,:inference) || throw(ArgumentError("prior coordinates must be :physical or :inference"))
        new{typeof(distribution)}(distribution,coordinates)
    end
end

# These library families have a declared connected interval support. The
# generic Distributions.support fallback also returns a RealInterval for custom
# distributions with holes, so that fallback alone cannot certify auto selection.
const _IntervalPrior = Union{Distributions.Normal,Distributions.LogNormal,
    Distributions.Exponential,Distributions.Gamma,Distributions.Beta,Distributions.Uniform,
    Distributions.Cauchy,Distributions.TDist,Distributions.Laplace,Distributions.Logistic,
    Distributions.Weibull,Distributions.InverseGamma,Distributions.Chisq,Distributions.FDist}
_prior_interval(d::_IntervalPrior)=(minimum(d),maximum(d))
function _prior_interval(d::Distributions.Truncated)
    _prior_interval(d.untruncated)===nothing ? nothing : (minimum(d),maximum(d))
end
_prior_interval(d)=nothing
function _automatic_transform(d)
    bounds=_prior_interval(d)
    bounds===nothing && throw(ArgumentError("automatic coordinates cannot certify a connected support for $(typeof(d)); supply an explicit compatible transform (IdentityTransform covers the real line)"))
    lo,hi=bounds
    lo<hi && !isnan(lo) && !isnan(hi) || throw(ArgumentError("invalid prior support"))
    if isfinite(lo) && isfinite(hi)
        LogitTransform(lo,hi)
    elseif isfinite(lo)
        iszero(lo) ? LogTransform() : ShiftedLogTransform(lo)
    elseif isfinite(hi)
        ShiftedLogTransform(hi;upper=true)
    else
        IdentityTransform()
    end
end
function _validate_transform_support(transform,prior)
    prior.coordinates===:physical || return
    bounds=_prior_interval(prior.distribution)
    range=_transform_bounds(transform)
    bounds===nothing || range===nothing || (range[1]<=bounds[1] && range[2]>=bounds[2]) ||
        throw(ArgumentError("$(typeof(transform)) removes part of the physical prior support $bounds; use a covering transform, or explicitly truncate the prior to the intended physical domain"))
end

"""
    ParameterSpec(name; initial, prior, transform=:auto, role=:kinetic, experiment=nothing)

One fitted physical scalar. Supply its physical starting guess and a continuous
univariate prior, e.g. `ParameterSpec(:g0; initial=0.2,
prior=LogNormal(log(0.2),0.5))`. The guess is neither a prior statistic nor a fixed
parameter; it never mutates the base forward problem. Kinetic callbacks receive
physical candidate values.

Automatic coordinates use the prior's declared support: identity on R, log on
(0,Inf), logit on a finite interval, shifted/reflected log on other half-lines.
Explicit transforms override auto selection without removing allowed support.
Inspect `parameterization(problem)` or `spec.transform`. Custom/unclassified
supports require an explicit compatible transform. An explicit inference prior
requires an explicit transform. Invalid guesses are rejected, never clamped.

MAP maximizes the PHYSICAL posterior. MCMC includes the internal-coordinate
Jacobian. `experiment=nothing` shares the scalar; a name makes it local.
`role=:nuisance` labels initial conditions, calibration or noise parameters.
Fixed values belong in the estimation problem or experiment's `fixed` tuple.
"""
struct ParameterSpec{T,P}
    name::Symbol
    initial::Float64
    prior::P
    transform::T
    role::Symbol
    experiment::Union{Nothing,Symbol}
    automatic::Bool
    function ParameterSpec(name::Symbol;initial,prior,transform=:auto,role=:kinetic,experiment=nothing)
        role in (:kinetic,:nuisance) || throw(ArgumentError("role must be :kinetic or :nuisance"))
        experiment===nothing || experiment isa Symbol || throw(ArgumentError("experiment must be a name"))
        prior=prior isa Prior ? prior : Prior(prior;coordinates=:physical)
        automatic=transform===:auto
        automatic && prior.coordinates===:inference && throw(ArgumentError("an inference-coordinate prior requires an explicit transform; prefer a direct physical prior"))
        transform=automatic ? _automatic_transform(prior.distribution) : transform
        transform isa AbstractInferenceTransform || throw(ArgumentError("transform must be :auto or an inference transform"))
        _validate_transform_support(transform,prior)
        initial isa Real && isfinite(initial) && isfinite(Float64(initial)) || throw(ArgumentError("initial must be a finite physical starting value"))
        z=_coordinate(transform,initial)
        isfinite(z) && _transform_valid(transform,z) || throw(ArgumentError("physical initial value cannot be represented by the selected transform"))
        x=prior.coordinates===:physical ? initial : z
        Distributions.insupport(prior.distribution,x) && isfinite(Distributions.logpdf(prior.distribution,x)) ||
            throw(ArgumentError("initial is outside prior support or has nonfinite prior density; endpoint singularities are not regular interior starts"))
        new{typeof(transform),typeof(prior)}(name,initial,prior,transform,role,experiment,automatic)
    end
end

"""Initial internal vector z from physical ParameterSpec guesses. This does not change the base PBMProblem."""
initial_coordinates(problem)=[_coordinate(p.transform,p.initial) for p in problem.parameters]
_fitted_values(problem,z)=[_physical(p.transform,v) for (p,v) in zip(problem.parameters,z)]
function _inference_coordinates(problem,theta)
    length(theta)==length(problem.parameters) || throw(DimensionMismatch("one physical value per fitted parameter required"))
    [_coordinate(p.transform,x) for (p,x) in zip(problem.parameters,theta)]
end
function _physical_parameter_tuple(problem,values)
    shared_indices=findall(p->p.experiment===nothing,problem.parameters)
    shared=merge(problem.fixed,NamedTuple{Tuple(problem.parameters[i].name for i in shared_indices)}(Tuple(values[i] for i in shared_indices)))
    locals=map(problem.experiments) do ex
        indices=findall(p->p.experiment===ex.name,problem.parameters)
        merge(ex.fixed,NamedTuple{Tuple(problem.parameters[i].name for i in indices)}(Tuple(values[i] for i in indices)))
    end
    (;shared,experiments=NamedTuple{Tuple(ex.name for ex in problem.experiments)}(locals))
end

"""
    physical_parameters(problem,z[,experiment_name])

Map INTERNAL coordinates to physical model parameters, including fixed values.
Return `(shared=...,experiments=...)`, or the merged tuple for one experiment.
This conversion says nothing about the reference density. `fit` reports the
physical-density MAP; stored MCMC z values sample a Jacobian-corrected density.
"""
function physical_parameters(problem,z)
    length(z)==length(problem.parameters) || throw(DimensionMismatch("one coordinate per fitted parameter required"))
    all(isfinite,z) || throw(ArgumentError("inference coordinates must be finite"))
    _physical_parameter_tuple(problem,_fitted_values(problem,z))
end
physical_parameters(problem,z,name::Symbol)=let p=physical_parameters(problem,z)
    merge(p.shared,getproperty(p.experiments,name))
end

"""Inspect physical guesses/priors, scopes, and resolved internal transforms. `automatic` distinguishes automatic selection from an explicit identity."""
parameterization(problem)=[(name=p.name,experiment=p.experiment,role=p.role,initial=p.initial,
    prior=p.prior.distribution,prior_coordinates=p.prior.coordinates,transform=p.transform,
    automatic=p.automatic,map_density=:physical,sampling_density=:inference) for p in problem.parameters]

_log_parameter_jacobian(problem,z)=sum(_logjac(p.transform,v) for (p,v) in zip(problem.parameters,z))
function _parameter_scales(problem)
    map(problem.parameters) do p
        sd=_prior_interval(p.prior.distribution)===nothing ? NaN : Distributions.std(p.prior.distribution)
        scale=p.prior.coordinates===:physical ? sd : abs(_transform_derivative(p.transform,_coordinate(p.transform,p.initial)))*sd
        isfinite(scale) && scale>0 ? Float64(scale) : max(abs(p.initial),1.0)
    end
end
function _physical_gradient(problem,z,g;indices=collect(eachindex(z)))
    base=_parameter_scales(problem)
    theta=_fitted_values(problem,z)
    # A weak prior can have an enormous SD unrelated to local parameter units.
    # Use a fixed physical start/current scale, bounded by a finite prior scale;
    # it never vanishes with an unconstraining derivative near an endpoint.
    # An exactly zero start/center uses unit numerical scale, capped by the
    # prior SD, rather than a potentially enormous weak-prior scale.
    scales=[begin
        reference=max(abs(problem.parameters[i].initial),abs(theta[i]))
        min(base[i],reference>0 ? reference : 1.0)
    end for i in indices]
    derivatives=[_transform_derivative(problem.parameters[i].transform,z[i]) for i in indices]
    physical=g./derivatives
    (;physical,scaled=physical.*scales,scales,derivatives)
end

function _parameter_interval(p)
    p.prior.coordinates===:physical && return _prior_interval(p.prior.distribution)
    prior_bounds=_prior_interval(p.prior.distribution)
    range=_transform_bounds(p.transform)
    prior_bounds===nothing && return range
    range===nothing && return nothing
    endpoints=(_physical(p.transform,prior_bounds[1]),_physical(p.transform,prior_bounds[2]))
    (min(endpoints...),max(endpoints...))
end
function _interior_start(problem,z)
    theta=_fitted_values(problem,z)
    all(eachindex(theta)) do i
        bounds=_parameter_interval(problem.parameters[i])
        bounds===nothing || bounds[1]<theta[i]<bounds[2]
    end
end

# Numerical verification uses an affine scaling of PHYSICAL values, never a
# nonlinear posterior reparameterization. The scale does not change the density.
function _local_physical_scales(problem,theta)
    base=_parameter_scales(problem)
    [begin
        bounds=_parameter_interval(p)
        distance=bounds===nothing ? Inf : min(x-bounds[1],bounds[2]-x)
        local_scale=max(abs(x),abs(p.initial))
        min(base[i],local_scale>0 ? local_scale : 1.0,distance)
    end for (i,(p,x)) in enumerate(zip(problem.parameters,theta))]
end
function _map_stationarity(problem,z,g;indices=collect(eachindex(z)),tolerance=1e-3)
    report=_physical_gradient(problem,z,g;indices)
    theta=_fitted_values(problem,z)
    boundary=any(enumerate(indices)) do (j,i)
        bounds=_parameter_interval(problem.parameters[i])
        bounds!==nothing && begin
            distance=min(theta[i]-bounds[1],bounds[2]-theta[i])
            distance<=0 || (distance/report.scales[j]<=sqrt(tolerance) &&
                norm(g,Inf)<=tolerance && norm(report.scaled,Inf)>tolerance)
        end
    end
    valid=!boundary && all(isfinite,report.scaled) && norm(report.scaled,Inf)<=tolerance
    reason=boundary ? "boundary-limited/nonregular mode; an interior physical MAP is not established" :
        !all(isfinite,report.scaled) ? "unresolved physical gradient or singular transform derivative" :
        valid ? "" : "physical posterior is not stationary in parameter-scaled units"
    merge(report,(;valid,reason,boundary_limited=boundary,tolerance))
end
