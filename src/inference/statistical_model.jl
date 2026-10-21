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
function _observe(o::DistributionObservable,u,op,p,t)
    if op isa CompiledSystem
        o.population===nothing && throw(ArgumentError("name the observed population"))
        pop=getproperty(op.system.populations,o.population)
        state=population(op,u,o.population)
    else
        pop=op;state=vec(u)
    end
    pop isa CompiledPBM || throw(ArgumentError("distribution observations require a sectional state or an explicit reconstruction"))
    g=pop.grid
    g.coordinate===o.coordinate || throw(ArgumentError("measurement and model coordinates differ; supply a coordinate-response model"))
    first(o.edges)>=first(g.edges) && last(o.edges)<=last(g.edges) || throw(ArgumentError("measurement window lies outside model domain"))
    result=zeros(length(o.edges)-1)
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
    all(x->x isa Real && isfinite(x),Y) || throw(ArgumentError("measurements must be finite and real"))
    expected=_observation_size(observable)
    expected===nothing || expected==size(Y,1) || throw(DimensionMismatch("observable and data sizes differ"))
    if likelihood isa Union{PoissonLikelihood,MultinomialLikelihood}
        all(x->x>=0 && isinteger(x),Y) || throw(ArgumentError("count likelihoods require actual nonnegative integer counts"))
        likelihood isa PoissonLikelihood && hasproperty(observable,:normalize) && observable.normalize && throw(ArgumentError("absolute Poisson counts require unnormalized predictions"))
    elseif hasproperty(observable,:normalize) && observable.normalize
        A=likelihood.contrasts
        A!==nothing && size(A,2)==size(Y,1) && norm(A*ones(size(Y,1)))<1e-10 ||
            throw(ArgumentError("normalized fractions need independent contrasts and their covariance"))
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
additional physical restrictions. `gradient(z)` may supply a log-posterior
gradient; it is checked against finite differences before fitting/sampling.
Otherwise central finite differences are used. No automatic AD compatibility is
assumed for a solver/representation. All fitted nuisance parameters are included.
"""
struct EstimationProblem{P,E,F,C,G}
    parameters::P
    experiments::E
    fixed::F
    constraints::C
    gradient::G
end
function EstimationProblem(parameters,experiments;fixed=NamedTuple(),constraints=p->true,gradient=nothing)
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
    EstimationProblem(ps,es,fixed,constraints,gradient)
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
        C0=Matrix(Diagonal(vec(s).^2))
        A===nothing ? C0 : let B=kron(Matrix{Float64}(I,size(Y,2),size(Y,2)),A);B*C0*B';end
    end
    size(C)==(n,n) || throw(DimensionMismatch("Gaussian covariance dimension mismatch"))
    all(x->x isa Real && isfinite(x),C) && isapprox(C,C';rtol=1e-12,atol=0) || throw(NumericalFailure("invalid Gaussian covariance"))
    factor=try
        cholesky(Symmetric(C))
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
    sum(Distributions.logpdf(Distributions.Poisson(mu[i]),Y[i]) for i in eachindex(Y))
end
function _measurement_loglikelihood(::MultinomialLikelihood,Y,P,theta,ex)
    all(P.>=0) || throw(NumericalFailure("negative predicted bin integral"))
    total=0.0
    for j in axes(Y,2)
        n=Int(sum(Y[:,j]));n==0 && continue
        mass=sum(P[:,j]);mass>0 || throw(NumericalFailure("empty predicted measurement window"))
        total+=Distributions.logpdf(Distributions.Multinomial(n,P[:,j]/mass),Int.(Y[:,j]))
    end
    total
end

mutable struct _InferenceCounts
    evaluations::Int
    solves::Int
    failures::Int
    messages::Vector{String}
end
_InferenceCounts()=_InferenceCounts(0,0,0,String[])
function _record_failure!(counts,e)
    counts===nothing && return
    counts.failures+=1
    message=sprint(showerror,e)
    length(counts.messages)<10 && !(message in counts.messages) && push!(counts.messages,message)
end

function _inference_predictions(problem::EstimationProblem,z,counts=nothing)
    physical=physical_parameters(problem,z)
    all(isfinite(_physical(p.transform,v)) for (p,v) in zip(problem.parameters,z)) || throw(NumericalFailure("nonfinite physical parameters"))
    problem.constraints(physical) || throw(NumericalFailure("physical parameter constraint failed"))
    map(problem.experiments) do ex
        theta=merge(physical.shared,getproperty(physical.experiments,ex.name))
        pb=ex.setup(theta,ex.problem)
        pb isa PBMProblem || throw(ArgumentError("setup must return a PBMProblem"))
        ts=sort!(unique(vcat((m.times for m in ex.measurements)...)))
        counts===nothing || (counts.solves+=1)
        sol=ex.solver===nothing ? simulate(pb;dt=ex.dt,saveat=ts) : ex.solver(pb,ts)
        length(sol.t)==length(ts)==length(sol.u) && all(isapprox.(sol.t,ts;rtol=1e-12,atol=1e-12)) || throw(NumericalFailure("inference solver did not return requested times; check its success code"))
        all(u->all(isfinite,u),sol.u) || throw(NumericalFailure("nonfinite inference state"))
        map(ex.measurements) do m
            raw=hcat([_observe(m.observable,sol.u[searchsortedfirst(ts,t)],pb.operator,pb.p,t) for t in m.times]...)
            output=m.response(raw,theta,ex)
            size(output)==size(m.data) || throw(DimensionMismatch("measurement response changed output dimension"))
            all(x->x isa Real && isfinite(x),output) || throw(NumericalFailure("measurement predictions must be finite and real"))
            output
        end
    end
end
predictions(problem::EstimationProblem,z)=_inference_predictions(problem,z)

"""Sum normalized measurement log likelihoods across every time, block and experiment. Priors are excluded; no bin/snapshot averaging is applied."""
function loglikelihood(problem::EstimationProblem,z;on_failure=:throw,_counts=nothing)
    on_failure in (:throw,:reject) || throw(ArgumentError("failure policy must be :throw or :reject"))
    try
        outputs=_inference_predictions(problem,z,_counts)
        sum(_measurement_loglikelihood(m.likelihood,m.data,P,physical_parameters(problem,z,ex.name),ex)
            for (ex,blocks) in zip(problem.experiments,outputs) for (m,P) in zip(ex.measurements,blocks))
    catch e
        e isa NumericalFailure || rethrow()
        _record_failure!(_counts,e)
        on_failure===:reject ? -Inf : rethrow()
    end
end

"""Sum declared prior densities in inference coordinates, adding each physical prior's Jacobian exactly once. No PBM solve is performed."""
function logprior(problem::EstimationProblem,z)
    length(z)==length(problem.parameters) || throw(DimensionMismatch("parameter dimension mismatch"))
    all(isfinite,z) || return -Inf
    total=0.0
    for (p,value) in zip(problem.parameters,z)
        _transform_valid(p.transform,value) || return -Inf
        x=p.prior.coordinates===:physical ? _physical(p.transform,value) : value
        isfinite(x) || return -Inf
        term=Distributions.logpdf(p.prior.distribution,x)
        isfinite(term) || return -Inf
        total+=term+(p.prior.coordinates===:physical ? _logjac(p.transform,value) : 0.0)
    end
    physical=physical_parameters(problem,z)
    problem.constraints(physical) || return -Inf
    total
end

"""The single posterior density in declared z coordinates used by MAP, Laplace and MCMC. NumericalFailure can be rejected explicitly; programming errors propagate."""
function logposterior(problem::EstimationProblem,z;on_failure=:throw,_counts=nothing)
    _counts===nothing || (_counts.evaluations+=1)
    lp=logprior(problem,z)
    isfinite(lp) || return -Inf
    lp+loglikelihood(problem,z;on_failure,_counts)
end

"""Separate log-likelihood, transformed log-prior and their sum at the same coordinate vector."""
function posterior_components(problem::EstimationProblem,z;kwargs...)
    likelihood=loglikelihood(problem,z;kwargs...);prior=logprior(problem,z)
    (;loglikelihood=likelihood,logprior=prior,logposterior=likelihood+prior)
end

"""MAP output, including physical parameters, declared coordinate specifications, component densities, diagnostics and optional Laplace approximation."""
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
end

"""Load AdvancedHMC and MCMCDiagnosticTools to enable explicit `sample_posterior(problem;...)`. Sampling is never automatic in fit."""
function sample_posterior end
"""Load Optim for nuisance-reoptimized `estimation_profile(...;target=:likelihood|:posterior)`; posterior profiles have no ordinary likelihood confidence cutoff."""
function estimation_profile end
