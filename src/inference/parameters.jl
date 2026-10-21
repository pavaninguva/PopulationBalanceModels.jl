abstract type AbstractInferenceTransform end

"""Identity coordinates with optional location and positive scale: θ=location+scale*z."""
struct IdentityTransform <: AbstractInferenceTransform
    location::Float64
    scale::Float64
    function IdentityTransform(;location=0.0,scale=1.0)
        isfinite(location) && isfinite(scale) && scale>0 || throw(ArgumentError("invalid coordinate scale"))
        new(location,scale)
    end
end
"""Positive physical parameter θ=scale*exp(z); scale is a fixed physical reference."""
struct LogTransform <: AbstractInferenceTransform
    scale::Float64
    function LogTransform(;scale=1.0)
        isfinite(scale) && scale>0 || throw(ArgumentError("log scale must be positive"))
        new(scale)
    end
end
"""Open bounded physical interval, θ=lower+(upper-lower)*logistic(z)."""
struct LogitTransform <: AbstractInferenceTransform
    lower::Float64
    upper::Float64
    function LogitTransform(lower,upper)
        isfinite(lower) && isfinite(upper) && lower<upper || throw(ArgumentError("invalid bounds"))
        new(lower,upper)
    end
end
_physical(t::IdentityTransform,z)=t.location+t.scale*z
_physical(t::LogTransform,z)=t.scale*exp(z)
_logistic(z)=z>=0 ? inv(1+exp(-z)) : exp(z)/(1+exp(z))
_physical(t::LogitTransform,z)=t.lower+(t.upper-t.lower)*_logistic(z)
_coordinate(t::IdentityTransform,x)=(x-t.location)/t.scale
_coordinate(t::LogTransform,x)=x>0 ? log(x/t.scale) : throw(ArgumentError("log parameter must be positive"))
_coordinate(t::LogitTransform,x)=t.lower<x<t.upper ? log((x-t.lower)/(t.upper-x)) : throw(ArgumentError("initial value must be strictly inside bounds"))
_logjac(t::IdentityTransform,z)=log(t.scale)
_logjac(t::LogTransform,z)=log(t.scale)+z
_softplus(x)=max(x,zero(x))+log1p(exp(-abs(x)))
_logjac(t::LogitTransform,z)=log(t.upper-t.lower)-_softplus(z)-_softplus(-z)
_transform_valid(t::AbstractInferenceTransform,z)=isfinite(_physical(t,z))
_transform_valid(t::LogTransform,z)=isfinite(_physical(t,z)) && _physical(t,z)>0
_transform_valid(t::LogitTransform,z)=t.lower<_physical(t,z)<t.upper

"""
    Prior(distribution; coordinates)

Declare a univariate density in `:physical` or `:inference` coordinates.
Physical densities receive the transform Jacobian exactly once. Inference
densities receive none. There is intentionally no implicit coordinate choice.
"""
struct Prior{D}
    distribution::D
    coordinates::Symbol
    function Prior(distribution;coordinates)
        coordinates in (:physical,:inference) || throw(ArgumentError("prior coordinates must be :physical or :inference"))
        new{typeof(distribution)}(distribution,coordinates)
    end
end

"""
    ParameterSpec(name; initial, prior, transform=IdentityTransform(), role=:kinetic, experiment=nothing)

One fitted scalar. `initial` is physical; optimization and sampling use z.
`experiment=nothing` shares it across experiments; an experiment name makes it
local. Use role=:nuisance for initial conditions, gains, or noise parameters.
Fixed parameters belong in the estimation problem or experiment's `fixed` tuple.
"""
struct ParameterSpec{T,P}
    name::Symbol
    initial::Float64
    prior::P
    transform::T
    role::Symbol
    experiment::Union{Nothing,Symbol}
    function ParameterSpec(name::Symbol;initial,prior::Prior,transform=IdentityTransform(),role=:kinetic,experiment=nothing)
        role in (:kinetic,:nuisance) || throw(ArgumentError("role must be :kinetic or :nuisance"))
        experiment===nothing || experiment isa Symbol || throw(ArgumentError("experiment must be a name"))
        isfinite(initial) && isfinite(_coordinate(transform,initial)) || throw(ArgumentError("invalid initial parameter"))
        new{typeof(transform),typeof(prior)}(name,initial,prior,transform,role,experiment)
    end
end

"""Initial inference-coordinate vector, obtained from physical ParameterSpec initial values."""
initial_coordinates(problem)=[_coordinate(p.transform,p.initial) for p in problem.parameters]

"""
    physical_parameters(problem,z[,experiment_name])

Return `(shared=...,experiments=...)`, including fixed values. With an
experiment name return its merged physical parameter tuple. Shared names cannot
be shadowed locally. The returned mode is the transform of a z-density mode;
it is generally not the physical-density mode.
"""
function physical_parameters(problem,z)
    length(z)==length(problem.parameters) || throw(DimensionMismatch("one coordinate per fitted parameter required"))
    all(isfinite,z) || throw(ArgumentError("inference coordinates must be finite"))
    values=[_physical(p.transform,v) for (p,v) in zip(problem.parameters,z)]
    shared_indices=findall(p->p.experiment===nothing,problem.parameters)
    shared=merge(problem.fixed,NamedTuple{Tuple(problem.parameters[i].name for i in shared_indices)}(Tuple(values[i] for i in shared_indices)))
    locals=map(problem.experiments) do ex
        indices=findall(p->p.experiment===ex.name,problem.parameters)
        merge(ex.fixed,NamedTuple{Tuple(problem.parameters[i].name for i in indices)}(Tuple(values[i] for i in indices)))
    end
    (;shared,experiments=NamedTuple{Tuple(ex.name for ex in problem.experiments)}(locals))
end
physical_parameters(problem,z,name::Symbol)=let p=physical_parameters(problem,z)
    merge(p.shared,getproperty(p.experiments,name))
end
