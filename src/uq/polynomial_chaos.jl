"""`PolynomialChaos(degree; condition_limit=1e8)` selects a full total-degree,
non-intrusive, orthonormal Hermite/Legendre regression. Only independent normal
and uniform reference marginals are supported; a declared deterministic map may
produce correlated physical inputs. No basis is inferred from posterior samples."""
struct PolynomialChaos
    degree::Int
    condition_limit::Float64
    function PolynomialChaos(degree::Integer;condition_limit=1e8)
        degree>=0 || throw(ArgumentError("PCE degree must be nonnegative"))
        isfinite(condition_limit) && condition_limit>1 || throw(ArgumentError("PCE condition_limit must exceed one"))
        new(degree,Float64(condition_limit))
    end
end

"""A failed training/validation ensemble, retained as `evaluations` for diagnosis.
Failed points are never removed to manufacture a regression fit."""
struct PCEFailure <: Exception
    message::String
    evaluations::UQResult
end
Base.showerror(io::IO,e::PCEFailure)=print(io,e.message)

"""Fitted non-intrusive PCE. Coefficient rows follow `multi_indices`; columns
follow the training output/time layout. `training` and `validation` retain their
distinct full-model evaluations. Inspect diagnostics before using the surrogate."""
struct PCESurrogate{P,T,V,R}
    problem::P
    method::PolynomialChaos
    multi_indices::Matrix{Int}
    coefficients::Matrix{Float64}
    training::T
    validation::V
    report::R
end
diagnostics(pce::PCESurrogate)=pce.report

function _pce_marginals(inputs)
    inputs.kind===:independent && all(d->d isa Union{Distributions.Normal,Distributions.Uniform},inputs.source) ||
        throw(ArgumentError("PCE requires independent Normal/Uniform reference marginals; supply an explicit map for physical dependence, not an empirical posterior basis"))
    for d in inputs.source
        a,b=Distributions.params(d)
        (d isa Distributions.Normal ? b>0 : b>a) || throw(ArgumentError("PCE reference marginals must be nondegenerate"))
    end
    Tuple(values(inputs.source))
end
function _pce_points(inputs,points)
    d=length(_pce_marginals(inputs))
    if points isa NamedTuple
        points.coordinates===inputs.coordinates || throw(ArgumentError("PCE design coordinate declaration differs from the input source"))
        points.names==inputs.names || throw(ArgumentError("PCE design input ordering differs"))
        points=points.samples
    elseif points isa UQResult
        points=points.inputs
    end
    X=if points isa AbstractMatrix
        Matrix{Float64}(points)
    else
        hcat((p isa NamedTuple ? Float64[getproperty(p,k) for k in inputs.names] :
            p isa Real ? [Float64(p)] : Float64.(collect(p)) for p in points)...)
    end
    ndims(X)==2 && size(X,1)==d && size(X,2)>0 || throw(DimensionMismatch("PCE points must have one row per reference coordinate and columns for samples"))
    all(isfinite,X) || throw(ArgumentError("PCE reference points must be finite"))
    for (j,dist) in enumerate(_pce_marginals(inputs))
        all(x->Distributions.insupport(dist,x),X[j,:]) ||
            throw(ArgumentError("PCE design leaves the support of reference coordinate $(inputs.names[j])"))
    end
    X
end
function _pce_indices(d,degree)
    indices=Vector{Int}[]
    function visit(prefix,left,remaining)
        if remaining==1
            push!(indices,vcat(prefix,left))
        else
            for k in left:-1:0
                visit(vcat(prefix,k),left-k,remaining-1)
            end
        end
    end
    for order in 0:degree
        visit(Int[],order,d)
    end
    reduce(vcat,permutedims.(indices))
end
function _pce_basis(inputs,indices,X)
    marginals=_pce_marginals(inputs)
    degree=maximum(indices);n=size(X,2)
    polynomials=ones(length(marginals),degree+1,n)
    for (j,dist) in enumerate(marginals)
        a,b=Distributions.params(dist)
        for i in 1:n
            z=dist isa Distributions.Normal ? (X[j,i]-a)/b : (2X[j,i]-a-b)/(b-a)
            degree==0 && continue
            polynomials[j,2,i]=dist isa Distributions.Normal ? z : sqrt(3.)*z
            for k in 1:degree-1
                if dist isa Distributions.Normal
                    polynomials[j,k+2,i]=(z*polynomials[j,k+1,i]-sqrt(k)*polynomials[j,k,i])/sqrt(k+1)
                else
                    polynomials[j,k+2,i]=sqrt(2k+3)/(k+1)*((2k+1)*z*polynomials[j,k+1,i]/sqrt(2k+1)-k*polynomials[j,k,i]/sqrt(2k-1))
                end
            end
        end
    end
    basis=ones(n,size(indices,1))
    for term in axes(indices,1),j in axes(indices,2),i in 1:n
        basis[i,term]*=polynomials[j,indices[term,j]+1,i]
    end
    basis
end

"""
    pce_design(inputs; n, method=:lhs, seed=2022, bounds=nothing)

Record an n-point training design in the input source's reference coordinates.
Default LHS stratifies each marginal CDF on its entire support. Optional d-by-2
`bounds` instead defines a uniformly stratified training box; this changes the
training design, never the propagated input law. Save the returned points, not
only their seed. `method=:mc` supplies ordinary random training points.
"""
function pce_design(inputs::UncertainInputs;n,method=:lhs,seed=2022,bounds=nothing)
    marginals=_pce_marginals(inputs);d=length(marginals)
    n isa Integer && n>0 || throw(ArgumentError("positive design size required"))
    method in (:lhs,:mc) || throw(ArgumentError("training design must be :lhs or :mc"))
    if bounds!==nothing
        bounds=Matrix{Float64}(bounds)
        size(bounds)==(d,2) && all(isfinite,bounds) && all(bounds[:,2].>bounds[:,1]) ||
            throw(ArgumentError("training bounds must be a finite d-by-2 matrix of increasing intervals"))
    end
    rng=MersenneTwister(seed);X=zeros(d,n)
    for (j,dist) in enumerate(marginals)
        order=method===:lhs ? randperm(rng,n) : ones(Int,n)
        for i in 1:n
            u=method===:lhs ? (order[i]-1+rand(rng))/n : rand(rng)
            X[j,i]=bounds===nothing ? Distributions.quantile(dist,u) : bounds[j,1]+u*(bounds[j,2]-bounds[j,1])
        end
    end
    all(isfinite,X) || throw(ArgumentError("design contains an infinite marginal quantile"))
    (;samples=X,names=inputs.names,coordinates=inputs.coordinates,method,seed,
      bounds,design_measure=bounds===nothing ? :marginal_cdf : :uniform_training_box,
      changes_input_law=false)
end

function _pce_success(result)
    all(result.success) || throw(PCEFailure("PCE cannot fit or validate after failed forward evaluations; inspect evaluations.failures",result))
    result
end

"""
    fit_pce(problem, PolynomialChaos(degree); design, validation=nothing, max_solves)

Fit every output/time using one SVD least-squares factorization. Rank and condition
checks precede model solves; no normal equations, ridge or missing-point deletion
is used. Designs and validation can be recorded designs, matrices or existing
UQResults from this same UQProblem. Reusing evaluations incurs zero new solves.
Validation must not overlap the training inputs; disjointness does not itself
prove statistical independence. A missing validation set is reported as untested.
"""
function fit_pce(problem::UQProblem,method::PolynomialChaos;design,validation=nothing,max_solves)
    construction_started=time_ns()
    X=_pce_points(problem.inputs,design)
    d,n=size(X)
    terms=binomial(big(d+method.degree),method.degree)
    terms<=n || throw(ArgumentError("degree $(method.degree) needs $terms terms but design has $n samples"))
    indices=_pce_indices(d,method.degree)
    V=_pce_basis(problem.inputs,indices,X);F=svd(V)
    tolerance=maximum(size(V))*eps(Float64)*first(F.S)
    numerical_rank=count(>(tolerance),F.S)
    condition=last(F.S)>0 ? first(F.S)/last(F.S) : Inf
    numerical_rank==size(V,2) && condition<=method.condition_limit ||
        throw(ArgumentError("PCE design is rank deficient or ill-conditioned (rank=$numerical_rank/$(size(V,2)), condition=$condition); change the design or degree"))
    preparation_seconds=(time_ns()-construction_started)/1e9
    Z=validation===nothing ? nothing : _pce_points(problem.inputs,validation)
    if Z!==nothing
        training_points=Set(Tuple(column) for column in eachcol(X))
        any(column->Tuple(column) in training_points,eachcol(Z)) &&
            throw(ArgumentError("PCE validation inputs overlap the training design"))
    end
    needed=(design isa UQResult ? 0 : n)+(validation===nothing || validation isa UQResult ? 0 : size(Z,2))
    max_solves isa Integer && max_solves>=needed || throw(ArgumentError("PCE construction needs a budget of $needed new full solves"))
    training=design isa UQResult ? design : propagate(problem,MonteCarlo(n);samples=design,max_solves=n)
    training.problem===problem || throw(ArgumentError("PCE evaluations must use the same UQProblem"))
    _pce_success(training)
    tick=time_ns();coefficients=F\permutedims(training.values)
    regression_seconds=(time_ns()-tick)/1e9
    residual=V*coefficients-permutedims(training.values)
    report=(;degree=method.degree,terms=size(indices,1),rank=numerical_rank,condition,
        new_full_solves=needed,training_solves=training.metadata.solves,
        validation_new_solves=validation===nothing || validation isa UQResult ? 0 : size(Z,2),
        preparation_seconds,regression_seconds,training_rmse=vec(sqrt.(sum(abs2,residual;dims=1)/n)),
        basis=:orthonormal_product,normalization=:probability_measure,
        validation_status=validation===nothing ? :not_run : :disjoint_inputs,
        coordinates=problem.inputs.coordinates)
    surrogate=PCESurrogate(problem,method,indices,coefficients,training,nothing,report)
    checked=if validation===nothing
        nothing
    else
        reference=validation isa UQResult ? validation : propagate(problem,MonteCarlo(size(Z,2));samples=validation,max_solves=size(Z,2))
        validate_pce(surrogate,_pce_success(reference))
    end
    PCESurrogate(problem,method,indices,coefficients,training,checked,report)
end

"""Evaluate a fitted PCE at supplied reference points, or seeded MonteCarlo
reference draws, with **zero full-model solves**. Negative surrogate outputs are
retained. Returned UQResult can supply pointwise surrogate quantiles via uq_summary;
these include surrogate approximation error only when separately assessed."""
function evaluate_pce(pce::PCESurrogate,points)
    input=pce.problem.inputs
    if points isa MonteCarlo
        raw,ids=_uq_samples(input,points.n,MersenneTwister(points.seed))
        X=_pce_points(input,raw);iid=input.iid;seed=points.seed
    else
        X=_pce_points(input,points);raw=[_uq_named(input,copy(c)) for c in eachcol(X)]
        ids=collect(1:size(X,2));iid=false;seed=nothing
    end
    values=permutedims(_pce_basis(input,pce.multi_indices,X)*pce.coefficients)
    all(isfinite,values) || throw(NumericalFailure("nonfinite PCE evaluation"))
    physical=Any[nothing for _ in ids];success=trues(length(ids));failures=NamedTuple[]
    for i in eachindex(ids)
        physical[i]=input.map(deepcopy(raw[i]))
        if !(_uq_finite(physical[i]) && input.valid(physical[i]))
            success[i]=false;values[:,i].=NaN
            push!(failures,(;sample=i,id=ids[i],stage=:input,message="PCE input violates the declared physical support"))
        end
    end
    meta=(;method=:pce,solves=0,requested=length(ids),surrogate_evaluations=length(ids),
        iid,seed,coordinates=input.coordinates,input_metadata=input.metadata)
    UQResult(pce.problem,raw,physical,ids,values,success,failures,pce.training.layout,meta)
end

"""Independent-input full-model validation; errors are reported for every output
and time, separately from propagated input variance. No relative error is assigned
when a reference variance is zero. Training points must not be reused."""
function validate_pce(pce::PCESurrogate,reference::UQResult)
    reference.problem===pce.problem && reference.layout==pce.training.layout ||
        throw(ArgumentError("PCE validation requires the same forward definition and output layout"))
    _pce_success(reference)
    training=Set(Tuple(c) for c in eachcol(_pce_points(pce.problem.inputs,pce.training)))
    X=_pce_points(pce.problem.inputs,reference)
    any(c->Tuple(c) in training,eachcol(X)) && throw(ArgumentError("validation reuses training inputs"))
    predicted=evaluate_pce(pce,X)
    difference=predicted.values-reference.values;n=size(X,2)
    mu=sum(reference.values;dims=2)/n
    rms=vec(sqrt.(sum(abs2,difference;dims=2)/n))
    spread=vec(sum(abs2,reference.values.-mu;dims=2)/n)
    normalized=[spread[i]>0 ? rms[i]/sqrt(spread[i]) : NaN for i in eachindex(rms)]
    (;reference,predicted,rmse=rms,normalized_rmse=normalized,
      maximum_absolute_error=vec(maximum(abs,difference;dims=2)),
      signed_mean_error=vec(sum(difference;dims=2))/n,n,
      training_overlap=false,independence=:declared_not_certified_by_disjointness)
end

"""Analytical mean/variance/covariance of an orthonormal PCE under its declared
reference law. Quantiles require evaluate_pce followed by uq_summary, not a
Gaussian assumption. These moments do not include regression/validation error."""
function uq_summary(pce::PCESurrogate;covariance=true)
    coefficients=pce.coefficients;nonconstant=coefficients[2:end,:]
    mu=vec(coefficients[1,:]);variance=vec(sum(abs2,nonconstant;dims=1))
    (;mean=_uq_blocks(pce.training.layout,mu),variance=_uq_blocks(pce.training.layout,variance),
      covariance=covariance ? nonconstant'*nonconstant : nothing,layout=pce.training.layout,
      coordinates=pce.problem.inputs.coordinates,analytic=true,quantiles=nothing,
      approximation_error_included=false)
end
