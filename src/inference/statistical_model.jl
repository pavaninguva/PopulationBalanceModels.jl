abstract type AbstractMeasurementLikelihood end

"""
    GaussianLikelihood(;sigma=nothing,covariance=nothing,contrasts=nothing)

Normalized Gaussian likelihood, including its determinant. Supply a positive
standard deviation or covariance (constant or `(theta,experiment)->value`).
Covariance acts on vec(data). For normalized fractions supply a full-row-rank
contrast matrix annihilating the all-ones vector; a singular full covariance
is projected before factorization. No fictitious particle count is introduced.
"""
struct GaussianLikelihood{S,C,A} <: AbstractMeasurementLikelihood
    sigma::S
    covariance::C
    contrasts::A
end
function GaussianLikelihood(;sigma=nothing,covariance=nothing,contrasts=nothing)
    (sigma===nothing)!=(covariance===nothing) || throw(ArgumentError("supply exactly one of sigma and covariance"))
    if contrasts!==nothing
        contrasts=Matrix{Float64}(contrasts)
        all(isfinite,contrasts) && size(contrasts,1)>0 && rank(contrasts)==size(contrasts,1) || throw(ArgumentError("contrasts must have full row rank"))
    end
    GaussianLikelihood(sigma,covariance,contrasts)
end

"""
    TruncatedGaussianLikelihood(;sigma,lower=0.0)

Independent Gaussian measurements conditioned to be at least `lower`. The
normalization depends on the predicted density and sigma and is retained in
MAP, Laplace, profiles and sampling. Sigma may be scalar, data-shaped, or a
`(theta,experiment)->value` callback. Correlated truncation is not implemented.
The latent PBM density is the underlying Gaussian location, not the conditional
mean. This model neither clips measurements nor assigns particle counts.
"""
struct TruncatedGaussianLikelihood{S,T} <: AbstractMeasurementLikelihood
    sigma::S
    lower::T
end
function TruncatedGaussianLikelihood(;sigma,lower=0.0)
    isfinite(lower) || throw(ArgumentError("finite truncation boundary required"))
    TruncatedGaussianLikelihood(sigma,float(lower))
end

"""Bin counts conditional on each column's observed sampled-particle total. Predictions are nonnegative bin integrals, normalized over the declared measurement window."""
struct MultinomialLikelihood <: AbstractMeasurementLikelihood end

"""Absolute bin counts with a known sampling exposure/volume. Exposure may be a scalar or a matrix matching data; it is never inferred from histogram normalization."""
struct PoissonLikelihood{E} <: AbstractMeasurementLikelihood
    exposure::E
end
function PoissonLikelihood(;exposure)
    all(x->x isa Real && isfinite(x) && x>=0,exposure isa Number ? (exposure,) : exposure) || throw(ArgumentError("known exposure must be finite and nonnegative"))
    PoissonLikelihood(exposure)
end

"""
    DistributionObservable(edges; coordinate, population=nothing, weight=x->1, normalize=false, response=nothing)

Integrate a sectional density over measurement bins, independently of solver
bins. `weight` specifies number, volume or instrument intensity weighting.
Optional `response(lo,hi,x,p,t)` is the probability that true size x is detected
in that measured bin (including sizing error/thresholds). Weighted integrals use
eight-point Gauss quadrature within solver bins. Physical discontinuities should
coincide with integration edges or use a custom observation model. A moment
state needs an explicitly supplied reconstruction via Function/SystemObservable.
"""
struct DistributionObservable{E,W,R} <: AbstractObservable
    edges::E
    coordinate::Symbol
    population::Union{Nothing,Symbol}
    weight::W
    normalize::Bool
    response::R
end
function DistributionObservable(edges;coordinate,population=nothing,weight=_unitweight,normalize=false,response=nothing)
    e=HistogramObservable(edges).edges
    DistributionObservable(e,coordinate,population,weight,normalize,response)
end
_observation_size(o::DistributionObservable)=length(o.edges)-1
_normalized_fractions(observable,likelihood)=likelihood isa GaussianLikelihood &&
    hasproperty(observable,:normalize) && observable.normalize
function _valid_fractions(values)
    all(x->isfinite(x) && 0<=x<=1,values) &&
        all(s->isapprox(s,1;rtol=0,atol=1e-10),sum(values;dims=1))
end
function _observe(o::DistributionObservable,u,op,p,t)
    if op isa CompiledSystem
        o.population===nothing && throw(ArgumentError("name the observed population"))
        pop=getproperty(op.system.populations,o.population)
        state=population(op,u,o.population)
    else
        pop=op;state=vec(u)
    end
    (pop isa CompiledPBM || (pop isa Union{CompiledCharacteristic,CompiledHOC} && pop.grid isa Grid1D)) ||
        throw(ArgumentError("distribution observations require a 1D bin-integrated state or an explicit reconstruction"))
    g=pop.grid
    g.coordinate===o.coordinate || throw(ArgumentError("measurement and model coordinates differ; supply a coordinate-response model"))
    first(o.edges)>=first(g.edges) && last(o.edges)<=last(g.edges) || throw(ArgumentError("measurement window lies outside model domain"))
    result=zeros(promote_type(eltype(state),_numeric_type(p)),length(o.edges)-1)
    for j in eachindex(result),i in eachindex(state)
        lo=o.response===nothing ? max(o.edges[j],g.edges[i]) : g.edges[i]
        hi=o.response===nothing ? min(o.edges[j+1],g.edges[i+1]) : g.edges[i+1]
        hi>lo || continue
        mid=(lo+hi)/2; half=(hi-lo)/2
        integral=0.0
        for q in eachindex(_GL8_X),sign in (-1,1)
            x=mid+sign*half*_GL8_X[q]
            w=applicable(o.weight,x,p,t) ? o.weight(x,p,t) : o.weight(x)
            response=o.response===nothing ? 1.0 : o.response(o.edges[j],o.edges[j+1],x,p,t)
            isfinite(w) && w>=0 && isfinite(response) && 0<=response<=1 || throw(NumericalFailure("invalid measurement weight/response"))
            integral+=half*_GL8_W[q]*w*response
        end
        result[j]+=state[i]/g.widths[i]*integral
    end
    if o.normalize
        sum(result)>0 || throw(NumericalFailure("empty measurement window"))
        result ./= sum(result)
    end
    result
end

"""
    Measurement(times,data,observable; likelihood,response=(y,theta,experiment)->y,source=nothing)

Rows are outputs, columns are times. `response` can apply an inferred calibration
factor. Mark blocks derived from the same measurements with the same `source`;
an experiment rejects duplicate source tags. Combine correlated derived outputs
in one joint block instead of treating histogram and derived moments independently.
"""
struct Measurement{T,Y,O,L,R,S}
    times::T
    data::Y
    observable::O
    likelihood::L
    response::R
    source::S
end
function Measurement(times,data,observable::AbstractObservable;likelihood::AbstractMeasurementLikelihood,
        response=(y,theta,experiment)->y,source=nothing)
    ts=float.(collect(times))
    Y=data isa AbstractVector ? reshape(float.(copy(data)),1,:) : float.(copy(data))
    !isempty(ts) && all(isfinite,ts) && all(diff(ts).>0) || throw(ArgumentError("measurement times must increase"))
    Y isa AbstractMatrix && size(Y,1)>0 && size(Y,2)==length(ts) || throw(DimensionMismatch("one measurement column per time required"))
    _measurement(ts,Y,observable,likelihood,response,source)
end
"""`Measurement(data, observable; likelihood, response, source)` declares
stationary measurements without timestamps. Vectors contain outputs; matrix
columns are replicate observations of one equilibrium. Likelihood normalization,
covariance, response and source-dependence rules are unchanged."""
function Measurement(data,observable::AbstractObservable;likelihood::AbstractMeasurementLikelihood,
        response=(y,theta,experiment)->y,source=nothing)
    Y=data isa AbstractVector ? reshape(float.(copy(data)),:,1) : float.(copy(data))
    Y isa AbstractMatrix && all(>(0),size(Y)) || throw(DimensionMismatch("stationary data must be a nonempty output vector or output-by-replicate matrix"))
    _measurement(nothing,Y,observable,likelihood,response,source)
end
Measurement(::Nothing,data,observable::AbstractObservable;kwargs...)=Measurement(data,observable;kwargs...)
function _measurement(ts,Y,observable,likelihood,response,source)
    all(x->x isa Real && isfinite(x),Y) || throw(ArgumentError("measurements must be finite and real"))
    expected=_observation_size(observable)
    expected===nothing || expected==size(Y,1) || throw(DimensionMismatch("observable and data sizes differ"))
    likelihood isa TruncatedGaussianLikelihood && any(Y.<likelihood.lower) && throw(ArgumentError("measurement below the likelihood truncation boundary"))
    if likelihood isa Union{PoissonLikelihood,MultinomialLikelihood}
        all(x->x>=0 && isinteger(x),Y) || throw(ArgumentError("count likelihoods require actual nonnegative integer counts"))
        likelihood isa PoissonLikelihood && hasproperty(observable,:normalize) && observable.normalize && throw(ArgumentError("absolute Poisson counts require unnormalized predictions"))
    elseif hasproperty(observable,:normalize) && observable.normalize
        likelihood isa GaussianLikelihood || throw(ArgumentError("normalized fractions require a joint contrast likelihood"))
        A=likelihood.contrasts
        A!==nothing && size(A,2)==size(Y,1) && norm(A*ones(size(Y,1)))<1e-10 ||
            throw(ArgumentError("normalized fractions need independent contrasts and their covariance"))
        _valid_fractions(Y) || throw(ArgumentError("normalized fraction data must lie in [0,1] and sum to one at every time"))
    end
    Measurement(ts,Y,observable,likelihood,response,source)
end
Measurement(o::Observation)=Measurement(o.times,o.data,o.observable;
    likelihood=o.factor===nothing ? GaussianLikelihood(sigma=o.sigma) : GaussianLikelihood(covariance=Matrix(o.factor)))

_default_setup(theta,base)=remake(base;p=merge(base.p,theta))

"""
    InferenceExperiment(name,problem,measurements...;dt=nothing,solver=nothing,setup,fixed=NamedTuple())

One operating history. `setup(theta,base_problem)` rebuilds parameters and,
optionally, uncertain initial conditions/bulk states. The default merges theta
into a NamedTuple p. A custom solver must check success and return exactly the
requested times. Existing Observation blocks can be adapted with Measurement.
"""
struct InferenceExperiment{P,M,F,D,S,B}
    name::Symbol
    problem::P
    measurements::M
    setup::F
    dt::D
    solver::S
    fixed::B
end
function InferenceExperiment(name::Symbol,problem::PBMProblem,measurements...;
        dt=nothing,solver=nothing,setup=_default_setup,fixed=NamedTuple())
    !isempty(measurements) || throw(ArgumentError("experiment needs measurements"))
    blocks=map(o->o isa Observation ? Measurement(o) : o,measurements)
    all(o->o isa Measurement,blocks) || throw(ArgumentError("provide Measurement or Observation blocks"))
    (dt===nothing ? solver!==nothing : isfinite(dt) && dt>0) || throw(ArgumentError("supply dt or an external solver"))
    for o in blocks
        o.times===nothing && throw(ArgumentError("transient measurements require explicit times"))
        problem.tspan[1]<=first(o.times)<=last(o.times)<=problem.tspan[2] || throw(ArgumentError("measurement outside experiment"))
    end
    sources=[o.source for o in blocks if o.source!==nothing]
    length(unique(sources))==length(sources) || throw(ArgumentError("dependent measurement blocks share a source; provide one joint likelihood"))
    InferenceExperiment{typeof(problem),typeof(blocks),typeof(setup),typeof(dt),typeof(solver),typeof(fixed)}(name,problem,blocks,setup,dt,solver,fixed)
end

"""
    EstimationProblem(parameters,experiments;fixed=NamedTuple(),constraints=p->true,gradient=nothing)

Shared statistical model for MAP, Laplace and sampling. Priors are independent
scalar declarations; use inference transforms for bounds and constraints for
additional physical restrictions. A `gradient(z)` callback must declare
`gradient_target=:physical` for d ell_theta(T(z))/dz, or `:sampling` for the
legacy Jacobian-corrected density. The package adjusts the other target and
independently verifies both. Physical MAP, physical local covariance and
Jacobian-corrected sampling use the same likelihood and physical prior law.
`derivatives=:auto` uses ForwardDiff or a SciMLSensitivitySolver adapter.
Finite differences require `derivatives=:finite_difference`. Differentiability
depends on the solver, representation and callbacks; failures never silently
switch methods. All fitted nuisance parameters are included.
"""
struct EstimationProblem{P,E,F,C,G}
    parameters::P
    experiments::E
    fixed::F
    constraints::C
    gradient::G
    derivatives::Symbol
    gradient_target::Symbol
end
function EstimationProblem(parameters,experiments;fixed=NamedTuple(),constraints=p->true,gradient=nothing,gradient_target=nothing,derivatives=:auto)
    ps=Tuple(parameters);es=Tuple(experiments)
    !isempty(ps) && all(p->p isa ParameterSpec,ps) || throw(ArgumentError("declare fitted parameters"))
    !isempty(es) && all(e->e isa InferenceExperiment,es) || throw(ArgumentError("provide inference experiments"))
    names=[e.name for e in es]
    length(unique(names))==length(names) || throw(ArgumentError("experiment names must be unique"))
    keys=[(p.experiment,p.name) for p in ps]
    length(unique(keys))==length(keys) || throw(ArgumentError("duplicate parameter declaration"))
    shared=[p.name for p in ps if p.experiment===nothing]
    isempty(intersect(shared,collect(Base.keys(fixed)))) || throw(ArgumentError("a parameter cannot be both fixed and fitted"))
    for e in es
        localnames=[p.name for p in ps if p.experiment===e.name]
        isempty(intersect(vcat(shared,collect(Base.keys(fixed))),vcat(localnames,collect(Base.keys(e.fixed))))) || throw(ArgumentError("local parameters cannot shadow shared parameters"))
        isempty(intersect(localnames,collect(Base.keys(e.fixed)))) || throw(ArgumentError("local parameter is both fixed and fitted"))
    end
    all(p->p.experiment===nothing || p.experiment in names,ps) || throw(ArgumentError("parameter names an unknown experiment"))
    gradient_target===nothing || gradient_target in (:physical,:sampling) || throw(ArgumentError("gradient_target must be :physical or :sampling"))
    gradient===nothing || gradient_target in (:physical,:sampling) || throw(ArgumentError(
        "gradient callbacks require gradient_target=:physical (gradient of the physical log posterior with respect to z) or :sampling (legacy Jacobian-corrected z-density gradient)"))
    EstimationProblem(ps,es,fixed,constraints,gradient,_derivative_choice(derivatives),
        gradient_target===nothing ? :physical : gradient_target)
end

_noise_value(x,theta,ex)=applicable(x,theta,ex) ? x(theta,ex) : x
function _gaussian(l::GaussianLikelihood,Y,P,theta,ex)
    A=l.contrasts
    if A!==nothing
        size(A,2)==size(Y,1) || throw(DimensionMismatch("contrast dimension mismatch"))
    end
    mean=A===nothing ? P : A*P
    data=A===nothing ? Y : A*Y
    n=length(data); original=length(Y)
    C=if l.covariance!==nothing
        value=Matrix(_noise_value(l.covariance,theta,ex))
        if A!==nothing && size(value)==(original,original)
            B=kron(Matrix{Float64}(I,size(Y,2),size(Y,2)),A)
            value=B*value*B'
        end
        value
    else
        s=_noise_value(l.sigma,theta,ex)
        s=s isa Number ? fill(s,size(Y)) : s
        size(s)==size(Y) || throw(DimensionMismatch("Gaussian sigma must match original data"))
        all(x->x isa Real && isfinite(x) && x>0,s) || throw(NumericalFailure("invalid Gaussian noise scale"))
        C0=Diagonal(vec(s).^2)
        A===nothing ? C0 : let B=kron(Matrix{Float64}(I,size(Y,2),size(Y,2)),A);B*C0*B';end
    end
    size(C)==(n,n) || throw(DimensionMismatch("Gaussian covariance dimension mismatch"))
    valid=C isa Diagonal ? all(isfinite,C.diag) :
        all(x->x isa Real && isfinite(x),C) && isapprox(C,C';rtol=1e-12,atol=0)
    valid || throw(NumericalFailure("invalid Gaussian covariance"))
    factor=try
        cholesky(C isa Diagonal ? C : Symmetric(C))
    catch e
        e isa PosDefException || rethrow()
        throw(NumericalFailure("Gaussian covariance is not positive definite after observation projection"))
    end
    (;mean,data,factor)
end
function _measurement_loglikelihood(l::GaussianLikelihood,Y,P,theta,ex)
    g=_gaussian(l,Y,P,theta,ex)
    r=g.factor.L\vec(g.mean-g.data)
    -(sum(abs2,r)+2sum(log,diag(g.factor.L))+length(r)*log(2π))/2
end
function _truncated_scales(l,Y,theta,ex)
    s=_noise_value(l.sigma,theta,ex)
    s=s isa Number ? fill(s,size(Y)) : s
    size(s)==size(Y) || throw(DimensionMismatch("truncated Gaussian sigma must match data"))
    all(x->x isa Real && isfinite(x) && x>0,s) || throw(NumericalFailure("invalid truncated Gaussian noise scale"))
    s
end
function _measurement_loglikelihood(l::TruncatedGaussianLikelihood,Y,P,theta,ex)
    any(Y.<l.lower) && return -Inf
    s=_truncated_scales(l,Y,theta,ex)
    sum(Distributions.logpdf(Distributions.Normal(P[i],s[i]),Y[i])-
        Distributions.logcdf(Distributions.Normal(),(P[i]-l.lower)/s[i]) for i in eachindex(Y))
end

function _count_means(l::PoissonLikelihood,Y,P)
    all(P.>=0) || throw(NumericalFailure("negative predicted bin count"))
    exposure=l.exposure
    exposure isa Number || size(exposure)==size(P) || throw(DimensionMismatch("exposure must match count data"))
    means=P.*exposure
    all(isfinite,means) || throw(NumericalFailure("nonfinite Poisson mean"))
    means
end
function _count_means(::MultinomialLikelihood,Y,P)
    all(P.>=0) || throw(NumericalFailure("negative predicted bin integral"))
    result=copy(P)
    for j in axes(P,2)
        total=sum(P[:,j]); n=sum(Y[:,j])
        if total<=0
            n==0 || throw(NumericalFailure("empty predicted measurement window"))
            result[:,j].=0
            continue
        end
        result[:,j].*=n/total
    end
    result
end
function _measurement_loglikelihood(l::PoissonLikelihood,Y,P,theta,ex)
    mu=_count_means(l,Y,P)
    # For zero counts log p(0|mu)=-mu, including mu=0. Avoid the
    # indeterminate derivative of the generic 0*log(mu) expression.
    sum(iszero(Y[i]) ? -mu[i] : Y[i]*log(mu[i])-mu[i]-loggamma(Y[i]+1) for i in eachindex(Y))
end
function _measurement_loglikelihood(::MultinomialLikelihood,Y,P,theta,ex)
    all(P.>=0) || throw(NumericalFailure("negative predicted bin integral"))
    total=0.0
    for j in axes(Y,2)
        n=Int(sum(Y[:,j]));n==0 && continue
        mass=sum(P[:,j]);mass>0 || throw(NumericalFailure("empty predicted measurement window"))
        total+=loggamma(n+1)-sum(loggamma(y+1) for y in Y[:,j])
        # A zero count contributes exactly zero even when its probability is
        # zero. Keep every predicted bin in mass so its normalization derivative
        # remains present; do not evaluate an indeterminate 0*log(0).
        total+=sum(Y[i,j]*log(P[i,j]/mass) for i in axes(Y,1) if !iszero(Y[i,j]))
    end
    total
end

mutable struct _InferenceCounts
    evaluations::Int
    solves::Int
    failures::Int
    messages::Vector{String}
    gradient_evaluations::Int
    adjoint_solves::Int
end
_InferenceCounts()=_InferenceCounts(0,0,0,String[],0,0)
function _record_failure!(counts,e)
    counts===nothing && return
    counts.failures+=1
    message=sprint(showerror,e)
    length(counts.messages)<10 && !(message in counts.messages) && push!(counts.messages,message)
end

function _experiment_problem(problem,z,ex)
    physical=physical_parameters(problem,z)
    all(isfinite(_physical(p.transform,v)) for (p,v) in zip(problem.parameters,z)) || throw(NumericalFailure("nonfinite physical parameters"))
    problem.constraints(physical) || throw(NumericalFailure("physical parameter constraint failed"))
    theta=merge(physical.shared,getproperty(physical.experiments,ex.name))
    pb=ex.setup(theta,ex.problem)
    _validate_experiment_problem(pb,ex.problem)
    _inference_problem(pb,z)
end
_validate_experiment_problem(pb,base::PBMProblem)=pb isa PBMProblem ? nothing :
    throw(ArgumentError("transient experiment setup must return a PBMProblem"))
_experiment_times(ex)=sort!(unique(vcat((m.times for m in ex.measurements)...)))
_experiment_solution(problem,z,ex,counts)=_experiment_solution(problem,z,ex,counts,ex.problem)
function _experiment_solution(problem,z,ex,counts,::PBMProblem)
    pb=_experiment_problem(problem,z,ex);ts=_experiment_times(ex)
    counts===nothing || (counts.solves+=1)
    sol=ex.solver===nothing ? simulate(pb;dt=ex.dt,saveat=ts) : ex.solver(pb,ts)
    length(sol.t)==length(ts)==length(sol.u) && all(isapprox.(sol.t,ts;rtol=1e-12,atol=1e-12)) || throw(NumericalFailure("inference solver did not return requested times; check its success code"))
    all(u->all(isfinite,u),sol.u) || throw(NumericalFailure("nonfinite inference state"))
    pb,ts,sol
end
function _experiment_predictions(problem,z,ex,pb,ts,states)
    theta=physical_parameters(problem,z,ex.name)
    map(ex.measurements) do m
        raw=hcat([_observe(m.observable,states[searchsortedfirst(ts,t)],pb.operator,pb.p,t) for t in m.times]...)
        output=m.response(raw,theta,ex)
        size(output)==size(m.data) || throw(DimensionMismatch("measurement response changed output dimension"))
        all(x->x isa Real && isfinite(x),output) || throw(NumericalFailure("measurement predictions must be finite and real"))
        _normalized_fractions(m.observable,m.likelihood) && !_valid_fractions(output) &&
            throw(NumericalFailure("normalized fraction predictions must lie in [0,1] and sum to one after the measurement response"))
        output
    end
end
function _experiment_loglikelihood(problem,z,ex,pb,ts,states)
    predictions=_experiment_predictions(problem,z,ex,pb,ts,states)
    theta=physical_parameters(problem,z,ex.name)
    sum(_measurement_loglikelihood(m.likelihood,m.data,P,theta,ex) for (m,P) in zip(ex.measurements,predictions))
end
function _experiment_loglikelihood(problem,z,ex,counts)
    pb,ts,sol=_experiment_solution(problem,z,ex,counts)
    _experiment_loglikelihood(problem,z,ex,pb,ts,sol.u)
end
function _inference_predictions(problem::EstimationProblem,z,counts=nothing)
    map(problem.experiments) do ex
        pb,ts,sol=_experiment_solution(problem,z,ex,counts)
        _experiment_predictions(problem,z,ex,pb,ts,sol.u)
    end
end
predictions(problem::EstimationProblem,z)=_inference_predictions(problem,z)

"""Sum normalized measurement log likelihoods across every time, block and experiment. Priors are excluded; no bin/snapshot averaging is applied."""
function loglikelihood(problem::EstimationProblem,z;coordinates=:inference,on_failure=:throw,_counts=nothing)
    on_failure in (:throw,:reject) || throw(ArgumentError("failure policy must be :throw or :reject"))
    coordinates in (:inference,:physical) || throw(ArgumentError("input coordinates must be :inference or :physical"))
    try
        internal=coordinates===:physical ? _inference_coordinates(problem,z) : z
        sum(_experiment_loglikelihood(problem,internal,ex,_counts) for ex in problem.experiments)
    catch e
        e isa NumericalFailure || rethrow()
        _record_failure!(_counts,e)
        on_failure===:reject ? -Inf : rethrow()
    end
end

"""
    logprior(problem, z; coordinates=:inference)

Physical prior log density, evaluated at θ=T(z). `coordinates=:physical` accepts
an ordered physical vector instead. Direct distributions and legacy physical
Prior declarations add no optimization Jacobian. An advanced inference prior
is converted with the inverse Jacobian. Each scalar prior is counted once,
independently of experiments/blocks; no PBM solve is performed.
"""
function logprior(problem::EstimationProblem,v;coordinates=:inference)
    length(v)==length(problem.parameters) || throw(DimensionMismatch("parameter dimension mismatch"))
    coordinates in (:inference,:physical) || throw(ArgumentError("input coordinates must be :inference or :physical"))
    all(isfinite,v) || return -Inf
    if coordinates===:inference
        all(_transform_valid(p.transform,z) for (p,z) in zip(problem.parameters,v)) || return -Inf
    end
    values=coordinates===:physical ? v : _fitted_values(problem,v)
    total=zero(eltype(values))
    for (i,p) in enumerate(problem.parameters)
        if p.prior.coordinates===:physical
            term=Distributions.logpdf(p.prior.distribution,values[i])
        else
            bounds=_transform_bounds(p.transform)
            bounds===nothing || bounds[1]<values[i]<bounds[2] || return -Inf
            z=coordinates===:inference ? v[i] : _coordinate(p.transform,values[i])
            isfinite(z) || return -Inf
            term=Distributions.logpdf(p.prior.distribution,z)-_logjac(p.transform,z)
        end
        (isnan(term) || term == -Inf) && return -Inf
        total+=term
    end
    problem.constraints(_physical_parameter_tuple(problem,values)) || return -Inf
    total
end

"""
    logposterior(problem,z; coordinates=:inference,on_failure=:throw)

PHYSICAL log-posterior density at θ=T(z), used by MAP. The default argument is
an internal vector; `coordinates=:physical` accepts physical values in parameter
order. Numerical coordinates do not add a Jacobian to this objective. Use
`sampling_logposterior` for the density with respect to dz. NumericalFailure
can be rejected explicitly; programming errors propagate.
"""
function logposterior(problem::EstimationProblem,v;coordinates=:inference,on_failure=:throw,_counts=nothing)
    _counts===nothing || (_counts.evaluations+=1)
    lp=logprior(problem,v;coordinates)
    isfinite(lp) || return lp
    lp+loglikelihood(problem,v;coordinates,on_failure,_counts)
end

"""Jacobian-corrected prior density with respect to internal dz, without PBM
solves. `logprior` instead returns the physical density at the same point."""
function sampling_logprior(problem::EstimationProblem,z)
    value=logprior(problem,z)
    isfinite(value) ? value+_log_parameter_jacobian(problem,z) : value
end

"""
    sampling_logposterior(problem,z; on_failure=:throw)

Density with respect to INTERNAL coordinates dz:
`logposterior(problem,z) + log|det DT(z)|`. All priors retain their physical
probability law. An auxiliary sampling map adds its own determinant separately.
This function is not the physical MAP objective.
"""
function sampling_logposterior(problem::EstimationProblem,z;on_failure=:throw,_counts=nothing)
    value=logposterior(problem,z;on_failure,_counts)
    isfinite(value) ? value+_log_parameter_jacobian(problem,z) : value
end

"""Physical likelihood, prior and posterior components at an internal vector
(or an ordered physical vector with `coordinates=:physical`). The density's
reference coordinates are always physical; no transform Jacobian is added."""
function posterior_components(problem::EstimationProblem,z;coordinates=:inference,kwargs...)
    likelihood=loglikelihood(problem,z;coordinates,kwargs...);prior=logprior(problem,z;coordinates)
    (;loglikelihood=likelihood,logprior=prior,logposterior=likelihood+prior,
      density_coordinates=:physical,argument_coordinates=coordinates,convention=:physical_map_v1)
end

const _MAP_CONVENTION=:physical_map_v1

"""MAP output. `minimizer` contains internal coordinates; `parameters` and the
reported component densities use the physical MAP convention. `convention`
prevents older coordinate-density MAP/uncertainty results being silently reused."""
struct MAPResult{P,C,L}
    minimizer::Vector{Float64}
    parameters::P
    loglikelihood::Float64
    logprior::Float64
    logposterior::Float64
    converged::Bool
    diagnostics::C
    coordinates::Vector{ParameterSpec}
    uncertainty::L
    convention::Symbol
end
function MAPResult(z,p,ll,lp,post,converged,diagnostics,coordinates,uncertainty;
        convention=:unspecified)
    MAPResult(z,p,ll,lp,post,converged,diagnostics,coordinates,uncertainty,convention)
end
function _check_parameterization(problem,specs)
    length(specs)==length(problem.parameters) || throw(DimensionMismatch("parameter dimension differs from result"))
    all(a.name==b.name && a.experiment==b.experiment && a.role==b.role &&
        a.transform==b.transform && a.prior.coordinates==b.prior.coordinates &&
        a.prior.distribution==b.prior.distribution for (a,b) in zip(specs,problem.parameters)) ||
        throw(ArgumentError("result parameter order, priors or transforms differ from the estimation problem"))
end
function _check_map_result(problem,result;_counts=nothing)
    isdefined(result,:convention) && result.convention===_MAP_CONVENTION || throw(ArgumentError(
        "MAP result has an old or unspecified density convention. Refit the physical posterior; do not relabel an earlier z-density mode or its uncertainty. Analytical/manual results must declare convention=:physical_map_v1 and pass the same checks."))
    _check_parameterization(problem,result.coordinates)
    parts=posterior_components(problem,result.minimizer;_counts)
    all(isapprox(a,b;rtol=1e-10,atol=1e-8) for (a,b) in
        ((parts.loglikelihood,result.loglikelihood),(parts.logprior,result.logprior),
         (parts.logposterior,result.logposterior))) || throw(ArgumentError("MAP result does not match this physical posterior at its center"))
    nothing
end

"""
    sample_posterior(problem; stopping=:auto, samples=nothing, warmup=500,
        chains=4, min_samples=500, batch_samples=250, max_samples=10_000,
        consecutive_passes=2, min_ess=400, rhat_limit=1.01, bfmi_limit=0.3, ...)

Load AdvancedHMC and MCMCDiagnosticTools for optional NUTS. Sampling is never
automatic in `fit`. With no `samples` count, the default runs warmup once per
chain, freezes adaptation and continues the same chains in batches. At least
two chains are required. Once the minimum retained length is reached, the
rank-normalized R-hat, bulk/tail ESS, energy BFMI and finite-MCSE screen must
pass at `consecutive_passes` successive checkpoints. Draw counts are per chain.
The budget is `max_samples`; this is not a wall-time or forward-solve budget.

An explicit `samples=N` preserves fixed-length behavior. Equivalently select
`stopping=:fixed, samples=N`; single-chain runs remain available in that mode.
`stopping=:diagnostics` explicitly selects the controller and rejects `samples`;
use `min_samples`/`max_samples` instead. A fixed run still reports failed checks.

`diagnostics(result)` records `stopping_policy`, `stopping_reason`, actual
`samples`, and `check_history`. Reasons are `:diagnostics_passed`, `:max_samples`,
`:sampler_pathology` or `:fixed_length`. Diagnostic mode stops unsuccessfully
when a retained divergence/depth hit makes its zero-tolerance screen impossible
to pass by appending draws. No bad draw is discarded and no kernel is silently
retuned. A final single-check `screen_passed` does not satisfy an unfinished
consecutive-pass rule: `diagnostics_passed` is false on budget exhaustion.

Optional `diagnostic_quantities=(rate=theta -> theta.shared.rate,...)` supplies
deterministic finite scalar functions of the physical `(shared,experiments)`
parameter tuple, without additional PBM solves. These quantities also enter the
R-hat/ESS screen. Absolute precision requirements can be supplied as
`mcse_limits=(rate=(mean=0.01,quantiles=(0.025=>0.03,0.975=>0.03)),)`;
a scalar limit means mean MCSE only. Limits have the quantity's physical units.
Quantile MCSE concerns uncertainty in an estimated interval endpoint, not the
posterior interval width. Constant/unresolved diagnostics cannot pass. These
sequential checks are a computational stopping policy, not proof of global
mixing, a model-validation test or a rigorous sequential-coverage guarantee.

`initial_covariance=C` optionally initializes
the HMC inverse mass matrix in sampling coordinates (inference coordinates by default); C must be finite,
symmetric positive definite and is used only for tuning, not returned as a
posterior uncertainty estimate. A tuple/vector of one such matrix per chain
allows different initial tuning, with independent adaptation. Neither form can
be combined with `laplace` initialization. A supplied physical Laplace covariance
is locally converted through inverse parameter/map Jacobians for tuning and
initialization; it remains distinct from sampler-coordinate covariance estimates.
An optional `sampling_map=(to_sampler,from_sampler,logabsdetjac,label)` changes
only auxiliary sampler coordinates; logabsdetjac is the log absolute determinant
of the Jacobian of from_sampler.
Inverses and Jacobians are checked at every mapped target evaluation; transformed
gradients are independently checked at chain starts. Use check_sampling_map at
additional representative inference-coordinate points, including tails. Returned
z values and statistics.log_density stay in declared inference coordinates;
statistics.sampler_log_density and Hamiltonian energies retain sampler coordinates.
"""
function sample_posterior end
"""Load Optim for nuisance-reoptimized `estimation_profile(...;target=:likelihood|:posterior)`; posterior profiles have no ordinary likelihood confidence cutoff."""
function estimation_profile end
