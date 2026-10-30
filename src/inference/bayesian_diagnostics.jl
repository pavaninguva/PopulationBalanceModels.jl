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

# Extension dispatch keeps the same experiment likelihood for direct AD,
# continuous forward sensitivities and adjoints.
function _experiment_gradient(solver,problem,z,ex,counts)
    _differentiate_gradient(v->_experiment_loglikelihood(problem,v,ex,counts),z;derivatives=:forwarddiff)
end
function _gradient_backend(problem,derivatives)
    derivatives===:finite_difference && return :finite_difference
    derivatives===:auto && problem.gradient!==nothing && return :callback
    derivatives===:auto && any(ex->ex.solver isa SciMLSensitivitySolver,problem.experiments) && return :sciml_sensitivities
    if any(ex->_is_stationary(ex.problem),problem.experiments)
        return all(ex->_is_stationary(ex.problem),problem.experiments) ? :implicit_stationary : :mixed_forwarddiff_implicit
    end
    :forwarddiff
end

"""Log-likelihood gradient in inference coordinates. Automatic mode uses a configured SciML sensitivity adapter or ForwardDiff; finite differences require an explicit request."""
function likelihood_gradient(problem::EstimationProblem,z;derivatives=problem.derivatives,step=cbrt(eps(Float64)),_counts=nothing)
    _derivative_choice(derivatives)
    if derivatives===:auto && any(ex->ex.solver isa SciMLSensitivitySolver,problem.experiments)
        _ad_evaluate(()->sum(_experiment_gradient(ex.solver,problem,z,ex,_counts) for ex in problem.experiments))
    else
        _differentiate_gradient(v->loglikelihood(problem,v;_counts),z;derivatives,step)
    end
end

"""
    posterior_gradient(problem,z; coordinates=:inference,...)

Derivative of the common declared-coordinate log posterior with respect to z. With
`coordinates=:physical`, accept an ordered physical vector and differentiate
with respect to those physical arguments; the density still refers to dz. A declared gradient callback
is adjusted from its explicit target; finite differences remain opt-in.
"""
function posterior_gradient(problem::EstimationProblem,v;coordinates=:inference,
        derivatives=problem.derivatives,step=cbrt(eps(Float64)),_counts=nothing)
    coordinates in (:inference,:physical) || throw(ArgumentError("input coordinates must be :inference or :physical"))
    if coordinates===:physical
        z=_inference_coordinates(problem,v)
        g=posterior_gradient(problem,z;derivatives,step,_counts)
        return g./[_transform_derivative(p.transform,x) for (p,x) in zip(problem.parameters,z)]
    end
    z=v
    _derivative_choice(derivatives)
    _counts===nothing || (_counts.gradient_evaluations+=1)
    backend=_gradient_backend(problem,derivatives)
    g=if backend===:callback
        supplied=collect(problem.gradient(z))
        problem.gradient_target===:physical ? supplied+_differentiate_gradient(w->_log_parameter_jacobian(problem,w),z) : supplied
    elseif backend===:sciml_sensitivities
        likelihood_gradient(problem,z;derivatives,step,_counts)+_differentiate_gradient(v->logprior(problem,v),z)
    else
        _differentiate_gradient(v->logposterior(problem,v;_counts),z;derivatives,step)
    end
    length(g)==length(z) && all(isfinite,g) || throw(NumericalFailure("invalid inference-coordinate posterior gradient"))
    g
end

"""Compatibility alias for posterior_gradient. Sampling and MAP differentiate
one declared z-density; an auxiliary map applies its own chain rule separately."""
sampling_gradient(problem::EstimationProblem,z;kwargs...)=posterior_gradient(problem,z;kwargs...)

"""
    check_gradient(problem,z; target=:map,coordinates=:inference,...)

Compare with independent step-halved finite differences. `:map` checks the
common z-density; `:sampling` is a compatibility alias. The latter
requires internal arguments. Local agreement does not prove global smoothness.
"""
function check_gradient(problem::EstimationProblem,z;target=:map,coordinates=:inference,
        derivatives=problem.derivatives,step=cbrt(eps(Float64)),tolerance=1e-3,_counts=nothing)
    target in (:map,:sampling) || throw(ArgumentError("gradient target must be :map or :sampling"))
    target===:sampling && coordinates!==:inference && throw(ArgumentError("sampling gradients use internal coordinates"))
    f=v->target===:map ? logposterior(problem,v;coordinates,_counts) : sampling_logposterior(problem,v;_counts)
    coarse=_fd_gradient(f,z;step);fine=_fd_gradient(f,z;step=step/2)
    supplied=target===:map ? posterior_gradient(problem,z;coordinates,derivatives,step,_counts) :
        sampling_gradient(problem,z;derivatives,step,_counts)
    error=norm(supplied-fine)/max(1,norm(fine))
    halving=norm(coarse-fine)/max(1,norm(fine))
    (;valid=error<=tolerance && halving<=tolerance,error,halving,gradient=supplied,reference=fine,
      backend=_gradient_backend(problem,derivatives),target,argument_coordinates=coordinates,
      density_coordinates=:inference)
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

"""Joint Gaussian in the declared inference coordinates at their MAP. Center, Hessian and covariance use z units and include every nuisance parameter. physical_center is the mapped mode, not the physical posterior mean."""
struct LaplaceApproximation{F,D,P}
    center::Vector{Float64}
    physical_center::P
    coordinates::Vector{ParameterSpec}
    hessian::Matrix{Float64}
    factor::F
    covariance::Matrix{Float64}
    diagnostics::D
    convention::Symbol
end

include("gauss_newton.jl")

"""
    laplace_approximation(problem,map;step, hessian_tolerance=0.01,...)

Explicitly requested approximation from the actual negative log posterior, not
a likelihood-only/Gauss–Newton substitute. Checks optimizer convergence,
stationarity, gradients, independent step-halved verification of the AD z Hessian,
positive curvature and condition number. Explicit finite differences use a
step-halved finite-difference Hessian. Invalid or weak curvature is reported
through LaplaceFailure. A local
check cannot establish Gaussian shape, unimodality or accurate tail coverage.
"""
function laplace_approximation(problem::EstimationProblem,result::MAPResult;
        step=eps(Float64)^(1/4),hessian_tolerance=0.01,gradient_tolerance=1e-3,
        condition_limit=1e10,derivatives=problem.derivatives)
    counts=_InferenceCounts();started=time_ns()
    _check_map_result(problem,result;_counts=counts)
    result.converged || throw(LaplaceFailure((reason="MAP optimization did not converge",)))
    z=result.minimizer
    stationarity=_map_stationarity(problem,z,posterior_gradient(problem,z;derivatives,_counts=counts);tolerance=gradient_tolerance)
    stationarity.valid || throw(LaplaceFailure((reason=stationarity.reason,stationarity)))
    f=v->-logposterior(problem,v;_counts=counts)
    if derivatives===:finite_difference
        H1=_fd_hessian(f,z;step);H=_fd_hessian(f,z;step=step/2)
        verification=norm(H-H1)
    else
        H=_differentiate_hessian(f,z;derivatives=:forwarddiff)
        gradient_function=v->_differentiate_gradient(f,v;derivatives=:forwarddiff)
        H1=_fd_jacobian(gradient_function,z;step)
        H2=_fd_jacobian(gradient_function,z;step=step/2)
        verification=max(norm(H-H2),norm(H2-H1))
    end
    agreement=verification/max(norm(H),eps(Float64))
    eigenpair=eigen(Symmetric(H));s=eigenpair.values
    condition=minimum(s)>0 ? maximum(s)/minimum(s) : Inf
    curvature_resolution=max(verification,eps(Float64)*max(norm(H),1.0))
    gradient=posterior_gradient(problem,z;derivatives,_counts=counts)
    gradient_check=check_gradient(problem,z;derivatives,_counts=counts)
    reason=agreement>hessian_tolerance ? "Hessian has not converged under step halving" :
        minimum(s)<=0 ? "negative or zero posterior curvature" :
        minimum(s)<=curvature_resolution ? "weak curvature is unresolved by derivative verification" :
        condition>condition_limit ? "posterior curvature is ill-conditioned" :
        norm(gradient,Inf)>gradient_tolerance ? "posterior mode is not stationary" :
        !gradient_check.valid ? "posterior gradient check failed" : ""
    report=(;valid=isempty(reason),reason,coordinates=:inference,convention=_MAP_CONVENTION,stationarity,hessian_agreement=agreement,hessian_backend=derivatives===:finite_difference ? :finite_difference : :forwarddiff,eigenvalues=s,
        weak_direction=eigenpair.vectors[:,1],condition_number=condition,gradient,gradient_check,
        curvature_resolution,forward_solves=counts.solves,posterior_evaluations=counts.evaluations,elapsed_seconds=(time_ns()-started)/1e9)
    isempty(reason) || throw(LaplaceFailure(report))
    factor=cholesky(Symmetric(H))
    covariance=Matrix(Symmetric(factor\Matrix{Float64}(I,length(z),length(z))))
    LaplaceApproximation(copy(z),physical_parameters(problem,z),ParameterSpec[problem.parameters...],H,factor,covariance,report,_MAP_CONVENTION)
end

function _check_approximation(problem,approximation)
    isdefined(approximation,:convention) && approximation.convention===_MAP_CONVENTION || throw(ArgumentError("old/unspecified covariance convention; recompute local uncertainty in the declared inference coordinates"))
    _check_parameterization(problem,approximation.coordinates)
    length(approximation.center)==length(problem.parameters) || throw(DimensionMismatch("approximation parameter dimension mismatch"))
end

"""Select the kinetic (or nuisance) marginal block from the FULL joint covariance, in declared inference-coordinate units. This never inverts a conditional Hessian subblock."""
function marginal_covariance(problem::EstimationProblem,approximation::Union{LaplaceApproximation,GaussNewtonApproximation};role=:kinetic,indices=nothing)
    _check_approximation(problem,approximation)
    selected=indices===nothing ? findall(p->p.role===role,problem.parameters) : collect(indices)
    !isempty(selected) && all(i->1<=i<=length(problem.parameters),selected) || throw(ArgumentError("select valid parameter indices"))
    (;indices=selected,parameters=[(problem.parameters[i].experiment,problem.parameters[i].name) for i in selected],
      covariance=approximation.covariance[selected,selected],coordinates=:inference,convention=_MAP_CONVENTION)
end

"""MCMC draws stored as (parameters, retained draws, chains), physical values and sampler diagnostics. Inspect diagnostics before using draws for inference."""
struct PosteriorSamples{P,D,C}
    z::Array{Float64,3}
    physical::P
    diagnostics::D
    coordinates::C
end

"""Return the sampler report, including rank R-hat, bulk/tail ESS, MCSE, per-chain energy/BFMI and adaptation statistics. A passed screen is evidence to assess, not a proof of convergence."""
diagnostics(samples::PosteriorSamples)=samples.diagnostics

"""
    diagnostics(z, statistics; min_ess=400, rhat_limit=1.01, bfmi_limit=0.3,
                max_depth=8, quantity_values=NamedTuple(), mcse_limits=NamedTuple())

Reassess retained MCMC draws with the same screen used by `sample_posterior`.
Requires AdvancedHMC and MCMCDiagnosticTools. `z` has dimensions
`(parameters, draws, chains)` in inference coordinates; `statistics` contains
one matching post-warmup AdvancedHMC statistics vector per chain. At least two
chains, finite rank-normalized R-hat/ESS/MCSE, the declared thresholds, adequate
energy BFMI, zero divergences and zero depth hits are required to pass.

Optional named `quantity_values` are finite `draws × chains` scalar summaries.
Each is also screened for R-hat and bulk/tail ESS. `mcse_limits` can additionally
specify absolute mean/quantile Monte Carlo errors in that quantity's units,
e.g. `(rate=(mean=0.01,quantiles=(0.025=>0.03,0.975=>0.03)),)`.
A scalar limit means a mean-MCSE limit. Constant or unresolved quantities have
undefined diagnostics and do not pass. These errors concern estimation of a
posterior summary, not posterior width or observation noise. This standalone
screen does not apply the sampler's minimum length or consecutive-pass rule.
"""
function diagnostics(z::AbstractArray{<:Real,3},statistics;kwargs...)
    throw(ArgumentError("load AdvancedHMC and MCMCDiagnosticTools for retained-chain diagnostics"))
end

"""
    parameter_draws(problem,approximation; n=1000,seed=2022)

Draw a joint Gaussian in declared inference coordinates, without PBM solves.
`z` retains every Gaussian draw; `theta`/`physical` contain the mapped values,
including nonfinite transforms or prior-support violations. `in_support` and
`transform_valid` flag failures. Nothing is clipped, filtered or reject-resampled.
The mapped center is a MAP point, not a physical posterior mean. A transformed
Gaussian is generally not a Gaussian in physical parameters.
"""
function parameter_draws(problem::EstimationProblem,approximation::Union{LaplaceApproximation,GaussNewtonApproximation};n=1000,seed=2022)
    _check_approximation(problem,approximation)
    n isa Integer && n>0 || throw(ArgumentError("positive draw count required"))
    rng=MersenneTwister(seed)
    Z=approximation.center.+approximation.factor.U\randn(rng,length(approximation.center),n)
    theta=hcat([_fitted_values(problem,z) for z in eachcol(Z)]...)
    valid=[all(isfinite,z) && all(_transform_valid(p.transform,v) for (p,v) in zip(problem.parameters,z)) for z in eachcol(Z)]
    support=[valid[i] && isfinite(logprior(problem,z)) for (i,z) in enumerate(eachcol(Z))]
    (;theta,z=Z,physical=[_physical_parameter_tuple(problem,v) for v in eachcol(theta)],
      in_support=support,transform_valid=valid,coordinates=approximation.coordinates,
      draw_coordinates=:inference,z_coordinates=:inference,convention=_MAP_CONVENTION,
      method=approximation isa LaplaceApproximation ? :laplace : :gauss_newton,seed)
end

"""Select retained MCMC draws of the physical posterior. `z` retains its internal
coordinate representation; `theta`/`physical` are physical. `indices` records
columns of reshape(samples.z, parameters, :), and must accompany predictions.
The stored chain density remains with respect to dz, including its Jacobian."""
function parameter_draws(problem::EstimationProblem,samples::PosteriorSamples;n=size(samples.z,2)*size(samples.z,3),seed=2022)
    _check_parameterization(problem,samples.coordinates)
    available=reshape(samples.z,size(samples.z,1),:)
    n isa Integer && 0<n<=size(available,2) || throw(ArgumentError("select no more than the available MCMC draws"))
    indices=randperm(MersenneTwister(seed),size(available,2))[1:n];Z=available[:,indices]
    theta=hcat([_fitted_values(problem,z) for z in eachcol(Z)]...)
    (;theta,z=Z,physical=[physical_parameters(problem,z) for z in eachcol(Z)],
      in_support=[isfinite(logprior(problem,z)) for z in eachcol(Z)],transform_valid=trues(n),
      coordinates=samples.coordinates,draw_coordinates=:inference,z_coordinates=:inference,
      posterior_law=:physical,method=:mcmc,seed,indices)
end

function _predictive_block(rng,m,P,theta,ex,noise)
    l=m.likelihood
    if l isa TruncatedGaussianLikelihood
        s=_truncated_scales(l,m.data,theta,ex)
        observed=noise ? [rand(rng,Distributions.truncated(Distributions.Normal(P[i,j],s[i,j]),l.lower,Inf)) for i in axes(P,1),j in axes(P,2)] : nothing
        return (;latent=P,observed,coordinates=:measurement)
    end
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
    support=draws isa AbstractMatrix ? nothing : get(draws,:in_support,nothing)
    valid=draws isa AbstractMatrix ? nothing : get(draws,:transform_valid,nothing)
    for (i,z) in enumerate(eachcol(Z))
        prediction=try
            if !(draws isa AbstractMatrix)
                (support===nothing || support[i]) || throw(NumericalFailure("predictive mapped Gaussian draw is outside prior/constraint support; retained without resampling"))
                (valid===nothing || valid[i]) || throw(NumericalFailure("predictive Gaussian draw has an invalid/saturated physical transform; retained without replacement"))
            end
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
function identifiability_report(problem::EstimationProblem,z;coordinates=:inference,step=cbrt(eps(Float64)),derivatives=problem.derivatives)
    coordinates in (:physical,:inference) || throw(ArgumentError("report coordinates must be :physical or :inference"))
    point=coordinates===:physical ? _fitted_values(problem,z) : z
    internal(v)=coordinates===:physical ? _inference_coordinates(problem,v) : v
    flatten(v)=vcat((vec(P) for blocks in predictions(problem,internal(v)) for P in blocks)...)
    J=_differentiate_jacobian(flatten,point;derivatives,step)
    decomposition=svd(J);s=decomposition.S
    numerical_rank=count(x->x>sqrt(eps())*maximum(s),s)
    likelihood_hessian=_differentiate_hessian(v->-loglikelihood(problem,v;coordinates),point;derivatives)
    prior_hessian=_differentiate_hessian(v->-logprior(problem,v;coordinates),point;derivatives)
    eig=eigen(Symmetric(likelihood_hessian))
    (;prediction_jacobian=J,prediction_singular_values=s,prediction_rank=numerical_rank,
      likelihood_hessian,prior_hessian,likelihood_eigenvalues=eig.values,weak_direction=eig.vectors[:,1],coordinates,
      prior_density_coordinates=:inference,convention=_MAP_CONVENTION)
end

"""Replace all scalar prior declarations, retaining transforms, data, histories and parameter sharing. Supply one physical distribution (or advanced Prior) per fitted parameter."""
function replace_priors(problem::EstimationProblem,priors)
    length(priors)==length(problem.parameters) || throw(DimensionMismatch("one replacement prior per parameter required"))
    ps=[ParameterSpec(p.name;initial=p.initial,prior=q,transform=p.transform,role=p.role,experiment=p.experiment) for (p,q) in zip(problem.parameters,priors)]
    EstimationProblem(ps,problem.experiments;fixed=problem.fixed,constraints=problem.constraints,derivatives=problem.derivatives)
end

"""Fit explicitly supplied alternative-prior problems; compare declared-coordinate MAP values, Laplace standard deviations, and shifts in reference posterior SD units. This reports prior dependence without treating it as data identifiability."""
function prior_sensitivity(problems::NamedTuple;shift_threshold=1.0,scale_ratio_threshold=2.0,kwargs...)
    _check_comparable_problems(problems;priors_may_change=true)
    fits=map(p->fit(p;uncertainty=:laplace,kwargs...),problems)
    reference=first(fits);sd=sqrt.(diag(reference.uncertainty.covariance))
    comparisons=map(problems,fits) do p,r
        shift=(r.minimizer-reference.uncertainty.center)./sd
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
        p.fixed==base.fixed && p.constraints===base.constraints ||
            throw(ArgumentError("comparison changes fixed parameters or constraints; reuse the same constraint callback"))
        length(p.parameters)==length(base.parameters) && length(p.experiments)==length(base.experiments) || throw(ArgumentError("comparison changes parameter or experiment count"))
        for (a,b) in zip(base.parameters,p.parameters)
            a.name==b.name && a.experiment==b.experiment && a.transform==b.transform || throw(ArgumentError("comparison changes parameter coordinates"))
            priors_may_change || (a.prior.coordinates==b.prior.coordinates && a.prior.distribution==b.prior.distribution) || throw(ArgumentError("numerical refinement must retain priors"))
        end
        for (a,b) in zip(base.experiments,p.experiments)
            a.name==b.name && length(a.measurements)==length(b.measurements) || throw(ArgumentError("comparison changes experiment layout"))
            a.fixed==b.fixed || throw(ArgumentError("comparison changes experiment-specific fixed parameters"))
            for (x,y) in zip(a.measurements,b.measurements)
                x.times==y.times && x.data==y.data && _same_likelihood(x.likelihood,y.likelihood) ||
                    throw(ArgumentError("comparison changes measurements or likelihood settings, including noise/covariance"))
                x.response===y.response || throw(ArgumentError("comparison changes the measurement response; reuse the same response callback"))
            end
        end
    end
end
_same_likelihood(a,b)=typeof(a)===typeof(b) &&
    all(isequal(getfield(a,k),getfield(b,k)) for k in fieldnames(typeof(a)))

"""Refit named numerical variants, comparing mode changes to reference Laplace SDs and predictions at a common parameter vector. Label spatial, domain, integration and closure variants separately; closure comparisons measure model approximation too."""
function refinement_study(problems::NamedTuple;kwargs...)
    _check_comparable_problems(problems)
    fits=NamedTuple{keys(problems)}(Tuple(i==1 ? fit(p;uncertainty=:laplace,kwargs...) : fit(p;kwargs...) for (i,p) in enumerate(values(problems))))
    reference=first(fits);sd=sqrt.(diag(reference.uncertainty.covariance))
    basepred=vcat((vec(P) for blocks in predictions(first(problems),reference.minimizer) for P in blocks)...)
    comparisons=map(problems,fits) do p,r
        common=reference.uncertainty.center
        P=vcat((vec(v) for blocks in predictions(p,common) for v in blocks)...)
        length(P)==length(basepred) || throw(DimensionMismatch("refinement must retain measurements and observation order"))
        shift=(r.minimizer-reference.uncertainty.center)./sd
        (;converged=r.converged,shift_in_reference_sd=r.converged ? shift : nothing,
          candidate_shift_in_reference_sd=shift,max_prediction_change=norm(P-basepred,Inf),reference_posterior_sd=sd)
    end
    (;fits,comparisons,coordinates=:inference,convention=_MAP_CONVENTION)
end
