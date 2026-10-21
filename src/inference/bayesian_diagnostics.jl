function _fd_gradient(f,z;step=cbrt(eps(Float64)))
    isfinite(step) && step>0 || throw(ArgumentError("invalid derivative step"))
    g=zeros(length(z))
    for i in eachindex(z)
        h=step*max(1,abs(z[i]));a=copy(z);b=copy(z)
        a[i]+=h;b[i]-=h
        g[i]=(f(a)-f(b))/(2h)
    end
    all(isfinite,g) || throw(NumericalFailure("gradient stencil crosses an invalid posterior/solve; check transforms, dt and starting point"))
    g
end

"""Log-posterior gradient in z, using an explicitly supplied callback or central differences. Finite differences apply to all fitted nuisance parameters too."""
function posterior_gradient(problem::EstimationProblem,z;step=cbrt(eps(Float64)),_counts=nothing)
    if problem.gradient===nothing
        _fd_gradient(v->logposterior(problem,v;on_failure=:reject,_counts),z;step)
    else
        g=collect(problem.gradient(z))
        length(g)==length(z) && all(isfinite,g) || throw(NumericalFailure("invalid supplied posterior gradient"))
        g
    end
end

"""Compare the actual posterior gradient with step-halved central differences at a specified z. This is a local check, not an automatic AD guarantee for arbitrary solvers."""
function check_gradient(problem::EstimationProblem,z;step=cbrt(eps(Float64)),tolerance=1e-3,_counts=nothing)
    f=v->logposterior(problem,v;_counts)
    coarse=_fd_gradient(f,z;step);fine=_fd_gradient(f,z;step=step/2)
    supplied=problem.gradient===nothing ? coarse : posterior_gradient(problem,z)
    error=norm(supplied-fine)/max(1,norm(fine))
    halving=norm(coarse-fine)/max(1,norm(fine))
    (;valid=error<=tolerance && halving<=tolerance,error,halving,gradient=supplied,reference=fine)
end

function _fd_hessian(f,z;step=eps(Float64)^(1/4))
    isfinite(step) && step>0 || throw(ArgumentError("invalid Hessian step"))
    n=length(z);H=zeros(n,n);f0=f(z)
    for i in 1:n
        hi=step*max(1,abs(z[i]));a=copy(z);b=copy(z);a[i]+=hi;b[i]-=hi
        H[i,i]=(f(a)-2*f0+f(b))/hi^2
        for j in 1:i-1
            hj=step*max(1,abs(z[j]));pp=copy(z);pm=copy(z);mp=copy(z);mm=copy(z)
            pp[i]+=hi;pp[j]+=hj;pm[i]+=hi;pm[j]-=hj
            mp[i]-=hi;mp[j]+=hj;mm[i]-=hi;mm[j]-=hj
            H[i,j]=H[j,i]=(f(pp)-f(pm)-f(mp)+f(mm))/(4hi*hj)
        end
    end
    all(isfinite,H) || throw(NumericalFailure("invalid Hessian stencil"))
    H
end

"""Failure of a Laplace calculation; `report` retains curvature, weak directions and the reason. No ridge/jitter is added."""
struct LaplaceFailure{R} <: Exception
    report::R
end
Base.showerror(io::IO,e::LaplaceFailure)=print(io,"Laplace approximation rejected: ",e.report.reason)

"""Joint local Gaussian approximation of the z posterior, with the full negative-log-posterior Hessian and its factorization; includes all nuisance parameters."""
struct LaplaceApproximation{F,D,P}
    center::Vector{Float64}
    physical_center::P
    coordinates::Vector{ParameterSpec}
    hessian::Matrix{Float64}
    factor::F
    covariance::Matrix{Float64}
    diagnostics::D
end

"""
    laplace_approximation(problem,map;step, hessian_tolerance=0.01,...)

Explicitly requested approximation from the actual negative log posterior, not
a likelihood-only/Gauss–Newton substitute. Checks optimizer convergence,
stationarity, gradient and Hessian step halving, positive curvature and condition
number. Invalid or weak curvature is reported through LaplaceFailure. A local
check cannot establish Gaussian shape, unimodality or accurate tail coverage.
"""
function laplace_approximation(problem::EstimationProblem,result::MAPResult;
        step=eps(Float64)^(1/4),hessian_tolerance=0.01,gradient_tolerance=1e-3,
        condition_limit=1e10)
    result.converged || throw(LaplaceFailure((reason="MAP optimization did not converge",)))
    counts=_InferenceCounts();started=time_ns()
    z=result.minimizer;f=v->-logposterior(problem,v;_counts=counts)
    H1=_fd_hessian(f,z;step);H=_fd_hessian(f,z;step=step/2)
    agreement=norm(H-H1)/max(norm(H),eps(Float64))
    eigenpair=eigen(Symmetric(H));s=eigenpair.values
    condition=minimum(s)>0 ? maximum(s)/minimum(s) : Inf
    curvature_resolution=max(norm(H-H1),eps(Float64)*max(norm(H),1.0))
    gradient=posterior_gradient(problem,z;_counts=counts)
    gradient_check=check_gradient(problem,z;_counts=counts)
    reason=agreement>hessian_tolerance ? "Hessian has not converged under step halving" :
        minimum(s)<=0 ? "negative or zero posterior curvature" :
        minimum(s)<=curvature_resolution ? "weak curvature is unresolved by the Hessian stencil" :
        condition>condition_limit ? "posterior curvature is ill-conditioned" :
        norm(gradient,Inf)>gradient_tolerance ? "posterior mode is not stationary" :
        !gradient_check.valid ? "posterior gradient check failed" : ""
    report=(;valid=isempty(reason),reason,hessian_agreement=agreement,eigenvalues=s,
        weak_direction=eigenpair.vectors[:,1],condition_number=condition,gradient,gradient_check,
        curvature_resolution,forward_solves=counts.solves,posterior_evaluations=counts.evaluations,elapsed_seconds=(time_ns()-started)/1e9)
    isempty(reason) || throw(LaplaceFailure(report))
    factor=cholesky(Symmetric(H))
    covariance=Matrix(Symmetric(factor\Matrix{Float64}(I,length(z),length(z))))
    LaplaceApproximation(copy(z),physical_parameters(problem,z),ParameterSpec[problem.parameters...],H,factor,covariance,report)
end

"""Select the kinetic (or nuisance) marginal block from the FULL Laplace covariance, in inference coordinates. This never inverts a conditional Hessian subblock."""
function marginal_covariance(problem::EstimationProblem,approximation::LaplaceApproximation;role=:kinetic,indices=nothing)
    selected=indices===nothing ? findall(p->p.role===role,problem.parameters) : collect(indices)
    !isempty(selected) && all(i->1<=i<=length(problem.parameters),selected) || throw(ArgumentError("select valid parameter indices"))
    (;indices=selected,parameters=[(problem.parameters[i].experiment,problem.parameters[i].name) for i in selected],
      covariance=approximation.covariance[selected,selected],coordinates=:inference)
end

"""MCMC draws stored as (parameters, retained draws, chains), physical values and sampler diagnostics. Inspect diagnostics before using draws for inference."""
struct PosteriorSamples{P,D,C}
    z::Array{Float64,3}
    physical::P
    diagnostics::D
    coordinates::C
end

"""Draw from an explicitly labelled Laplace approximation and transform every draw to physical coordinates. This performs no forward PBM solves."""
function parameter_draws(problem::EstimationProblem,approximation::LaplaceApproximation;n=1000,seed=2022)
    n isa Integer && n>0 || throw(ArgumentError("positive draw count required"))
    rng=MersenneTwister(seed)
    Z=approximation.center.+approximation.factor.U\randn(rng,length(approximation.center),n)
    (;z=Z,physical=[physical_parameters(problem,z) for z in eachcol(Z)],
      in_support=[isfinite(logprior(problem,z)) for z in eachcol(Z)],coordinates=approximation.coordinates,method=:laplace,seed)
end
function parameter_draws(problem::EstimationProblem,samples::PosteriorSamples;n=size(samples.z,2)*size(samples.z,3),seed=2022)
    available=reshape(samples.z,size(samples.z,1),:)
    n isa Integer && 0<n<=size(available,2) || throw(ArgumentError("select no more than the available MCMC draws"))
    indices=randperm(MersenneTwister(seed),size(available,2))[1:n];Z=available[:,indices]
    (;z=Z,physical=[physical_parameters(problem,z) for z in eachcol(Z)],
      in_support=[isfinite(logprior(problem,z)) for z in eachcol(Z)],coordinates=samples.coordinates,method=:mcmc,seed)
end

function _predictive_block(rng,m,P,theta,ex,noise)
    l=m.likelihood
    if l isa GaussianLikelihood
        g=_gaussian(l,m.data,P,theta,ex)
        latent=g.mean
        observed=noise ? reshape(vec(latent)+g.factor.L*randn(rng,length(latent)),size(latent)) : nothing
        return (;latent,observed,coordinates=l.contrasts===nothing ? :measurement : :contrasts)
    end
    latent=_count_means(l,m.data,P)
    observed=if !noise
        nothing
    elseif l isa PoissonLikelihood
        rand.(Ref(rng),Distributions.Poisson.(latent))
    else
        values=zeros(Int,size(P))
        for j in axes(P,2)
            n=Int(sum(m.data[:,j]));n==0 && continue
            values[:,j]=rand(rng,Distributions.Multinomial(n,P[:,j]/sum(P[:,j])))
        end
        values
    end
    (;latent,observed,coordinates=:measurement)
end

"""
    posterior_predictive(problem,draws;max_solves,measurement_noise=false,seed=2022)

Propagate an explicit finite set of parameter draws. Budget counts one forward
solve per experiment per draw and is checked BEFORE execution. Returns raw
predictions, latent measurement means and optionally fresh measurement noise.
Multinomial predictions condition on the observed snapshot totals. Contrast
Gaussian outputs remain in contrast coordinates; they are not fabricated PSDs.
Failures are returned by draw with their messages and counted; no clipping occurs.
"""
function posterior_predictive(problem::EstimationProblem,draws;max_solves,measurement_noise=false,seed=2022)
    Z=draws isa AbstractMatrix ? draws : draws.z
    ndims(Z)==2 && size(Z,1)==length(problem.parameters) || throw(DimensionMismatch("draw matrix must have one row per parameter"))
    required=size(Z,2)*length(problem.experiments)
    max_solves isa Integer && max_solves>=required || throw(ArgumentError("predictive propagation requires $required solves; budget is $max_solves"))
    rng=MersenneTwister(seed);results=Any[];counts=_InferenceCounts()
    for z in eachcol(Z)
        prediction=try
            isfinite(logprior(problem,z)) || throw(NumericalFailure("predictive draw is outside prior support; inspect the approximation"))
            raw=_inference_predictions(problem,z,counts)
            measurements=map(problem.experiments,raw) do ex,blocks
                theta=physical_parameters(problem,z,ex.name)
                map((m,P)->_predictive_block(rng,m,P,theta,ex,measurement_noise),ex.measurements,blocks)
            end
            (;raw,measurements)
        catch e
            e isa NumericalFailure || rethrow()
            _record_failure!(counts,e);nothing
        end
        push!(results,prediction)
    end
    (;results,solves=counts.solves,failed_draws=counts.failures,failure_messages=counts.messages,measurement_noise,seed)
end

"""Likelihood-only local prediction sensitivities/SVD and observed curvature, reported separately from prior curvature. Prediction rank alone omits information in inferred noise/covariance parameters; observed likelihood curvature includes it. Neither establishes global identifiability."""
function identifiability_report(problem::EstimationProblem,z;step=cbrt(eps(Float64)))
    flatten(v)=vcat((vec(P) for blocks in predictions(problem,v) for P in blocks)...)
    base=flatten(z);J=zeros(length(base),length(z))
    for i in eachindex(z)
        h=step*max(1,abs(z[i]));a=copy(z);b=copy(z);a[i]+=h;b[i]-=h
        J[:,i]=(flatten(a)-flatten(b))/(2h)
    end
    decomposition=svd(J);s=decomposition.S
    numerical_rank=count(x->x>sqrt(eps())*maximum(s),s)
    likelihood_hessian=_fd_hessian(v->-loglikelihood(problem,v),z)
    prior_hessian=_fd_hessian(v->-logprior(problem,v),z)
    eig=eigen(Symmetric(likelihood_hessian))
    (;prediction_jacobian=J,prediction_singular_values=s,prediction_rank=numerical_rank,
      likelihood_hessian,prior_hessian,likelihood_eigenvalues=eig.values,weak_direction=eig.vectors[:,1])
end

"""Replace all scalar prior declarations, retaining transforms, data, histories and parameter sharing. Supply one explicitly coordinated Prior per fitted parameter."""
function replace_priors(problem::EstimationProblem,priors)
    length(priors)==length(problem.parameters) || throw(DimensionMismatch("one replacement prior per parameter required"))
    ps=[ParameterSpec(p.name;initial=p.initial,prior=q,transform=p.transform,role=p.role,experiment=p.experiment) for (p,q) in zip(problem.parameters,priors)]
    EstimationProblem(ps,problem.experiments;fixed=problem.fixed,constraints=problem.constraints)
end

"""Fit explicitly supplied alternative-prior problems; compare z modes, Laplace standard deviations, and shifts in reference posterior SD units. This reports prior dependence without treating it as data identifiability."""
function prior_sensitivity(problems::NamedTuple;shift_threshold=1.0,scale_ratio_threshold=2.0,kwargs...)
    _check_comparable_problems(problems;priors_may_change=true)
    fits=map(p->fit(p;uncertainty=:laplace,kwargs...),problems)
    reference=first(fits);sd=sqrt.(diag(reference.uncertainty.covariance))
    comparisons=map(fits) do r
        shift=(r.minimizer-reference.minimizer)./sd
        posterior_sd=sqrt.(diag(r.uncertainty.covariance));ratio=posterior_sd./sd
        (;parameters=r.parameters,shift_in_reference_sd=shift,posterior_sd,converged=r.converged,
          prior_sensitive=maximum(abs,shift)>shift_threshold || maximum(ratio)>scale_ratio_threshold || minimum(ratio)<1/scale_ratio_threshold)
    end
    (;fits,comparisons,coordinates=:inference,shift_threshold,scale_ratio_threshold)
end

function _check_comparable_problems(problems;priors_may_change=false)
    isempty(problems) && throw(ArgumentError("provide named estimation problems"))
    base=first(problems)
    for p in values(problems)
        length(p.parameters)==length(base.parameters) && length(p.experiments)==length(base.experiments) || throw(ArgumentError("comparison changes parameter or experiment count"))
        for (a,b) in zip(base.parameters,p.parameters)
            a.name==b.name && a.experiment==b.experiment && a.transform==b.transform || throw(ArgumentError("comparison changes parameter coordinates"))
            priors_may_change || (a.prior.coordinates==b.prior.coordinates && a.prior.distribution==b.prior.distribution) || throw(ArgumentError("numerical refinement must retain priors"))
        end
        for (a,b) in zip(base.experiments,p.experiments)
            a.name==b.name && length(a.measurements)==length(b.measurements) || throw(ArgumentError("comparison changes experiment layout"))
            all(x.times==y.times && x.data==y.data && typeof(x.likelihood)==typeof(y.likelihood) for (x,y) in zip(a.measurements,b.measurements)) || throw(ArgumentError("comparison changes measurements or likelihood family"))
        end
    end
end

"""Refit named numerical variants, comparing mode changes to reference Laplace SDs and predictions at a common parameter vector. Label spatial, domain, integration and closure variants separately; closure comparisons measure model approximation too."""
function refinement_study(problems::NamedTuple;kwargs...)
    _check_comparable_problems(problems)
    fits=NamedTuple{keys(problems)}(Tuple(i==1 ? fit(p;uncertainty=:laplace,kwargs...) : fit(p;kwargs...) for (i,p) in enumerate(values(problems))))
    reference=first(fits);sd=sqrt.(diag(reference.uncertainty.covariance))
    basepred=vcat((vec(P) for blocks in predictions(first(problems),reference.minimizer) for P in blocks)...)
    comparisons=map(problems,fits) do p,r
        P=vcat((vec(v) for blocks in predictions(p,reference.minimizer) for v in blocks)...)
        length(P)==length(basepred) || throw(DimensionMismatch("refinement must retain measurements and observation order"))
        shift=(r.minimizer-reference.minimizer)./sd
        (;converged=r.converged,shift_in_reference_sd=r.converged ? shift : nothing,
          candidate_shift_in_reference_sd=shift,max_prediction_change=norm(P-basepred,Inf),reference_posterior_sd=sd)
    end
    (;fits,comparisons,coordinates=:inference)
end
