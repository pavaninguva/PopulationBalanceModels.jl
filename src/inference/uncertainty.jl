# Statistical calculations use whitened residuals in optimization coordinates.
_estimate(z::AbstractVector)=float.(copy(z))
_estimate(result::FitResult)=_estimate(result.minimizer)

function _same_measurements(a,b)
    length(a.experiments)==length(b.experiments) || return false
    for (ea,eb) in zip(a.experiments,b.experiments)
        length(ea.observations)==length(eb.observations) || return false
        for (oa,ob) in zip(ea.observations,eb.observations)
            oa.times==ob.times && oa.data==ob.data && oa.sigma==ob.sigma || return false
            (oa.factor===nothing)==(ob.factor===nothing) || return false
            oa.factor===nothing || oa.factor.L==ob.factor.L || return false
        end
    end
    true
end

"""
    fit_diagnostics(calibration, estimate; variance=:known, refined=nothing, ...)

Assess a fit in its optimization coordinates. Returns whitened residuals,
Gauss–Newton covariance/correlation, rank, residual degrees of freedom,
stationarity, and agreement with an independent central-difference check. `variance=:known`
uses the supplied absolute observation errors; `:estimated` treats them as
relative weights with one unknown common variance, estimated as RSS/(n-rank).

For a stationary full-rank fit with known errors, the chi-square goodness-of-fit
p-value uses n-p degrees of freedom (a local nonlinear approximation).
`refined` may supply a finer forward calculation for the same measurements;
the report measures its changes in whitened predictions and derivatives.
These diagnostics do not establish model validity or global identifiability.
"""
function fit_diagnostics(cal::CalibrationProblem,estimate;variance=:known,
        relative_step=cbrt(eps(Float64)),rank_tolerance=sqrt(eps(Float64)),
        gradient_tolerance=1e-4,refined=nothing,derivatives=cal.derivatives)
    variance in (:known,:estimated) || throw(ArgumentError("variance must be :known or :estimated"))
    isfinite(gradient_tolerance) && gradient_tolerance>0 || throw(ArgumentError("invalid gradient tolerance"))
    z=_estimate(estimate); r=residuals(cal,z)
    coarse=sensitivity_report(cal,z;relative_step,rank_tolerance,derivatives=:finite_difference)
    fine=sensitivity_report(cal,z;relative_step=relative_step/2,rank_tolerance,derivatives)
    J=fine.jacobian; rss=sum(abs2,r); n=length(r); p=length(z); dof=n-fine.rank
    difference_error=norm(J-coarse.jacobian)/max(norm(J),eps(Float64))
    gradient=J'*r
    column_norms=vec(sqrt.(sum(abs2,J;dims=1)))
    score=maximum(abs.(gradient)./max.(column_norms,eps(Float64)))/max(norm(r),1.0)
    stationary=score<=gradient_tolerance
    variance_scale=variance===:known ? 1.0 : (dof>0 ? rss/dof : NaN)
    covariance=fine.covariance===nothing || !isfinite(variance_scale) ? nothing : fine.covariance.*variance_scale
    correlation=if covariance===nothing || any(diag(covariance).<=0)
        nothing
    else
        scale=sqrt.(diag(covariance)); covariance./(scale*scale')
    end
    pvalue=variance===:known && stationary && fine.rank==p && dof>0 ?
        Distributions.ccdf(Distributions.Chisq(dof),rss) : nothing
    numerical=if refined===nothing
        nothing
    else
        _same_measurements(cal,refined) || throw(ArgumentError("refinement must retain the same measurements and noise model"))
        rf=residuals(refined,z)
        sf=sensitivity_report(refined,z;relative_step=relative_step/2,rank_tolerance,derivatives)
        (;prediction_change=norm(rf-r,Inf),
          jacobian_change=norm(sf.jacobian-J)/max(norm(J),eps(Float64)),rank=sf.rank)
    end
    (;center=z,residuals=r,jacobian=J,singular_values=fine.singular_values,
      rank=fine.rank,condition_number=fine.condition_number,covariance,correlation,
      observations=n,parameters=p,degrees_of_freedom=dof,rss,variance,variance_scale,
      gradient,scaled_gradient=score,stationary,difference_error,numerical,
      goodness_of_fit_pvalue=pvalue,derivative_backend=fine.derivatives)
end

"""A local joint confidence ellipsoid in optimization coordinates; construct with `confidence_ellipsoid`."""
struct ConfidenceEllipsoid{Z,C,T,R}
    center::Z
    covariance::C
    cutoff::T
    level::T
    diagnostics::R
end

"""
    confidence_ellipsoid(calibration, estimate; level=0.95, variance=:known, ...)

Local Gauss–Newton joint region `(z-ẑ)' inv(C) (z-ẑ) <= cutoff`.
Known absolute errors use chi-square(p); relative weights with an estimated
common variance use p*F(p,n-p). Rejects rank deficiency, a nonstationary fit,
unverified derivatives, and a nonconverged FitResult. If a `refined`
calibration is supplied, also checks forward and derivative convergence.
A nonlinear parameter transform maps the region to a generally nonellipsoidal
physical region. The approximation is not a global coverage guarantee.
"""
function confidence_ellipsoid(cal::CalibrationProblem,estimate;level=0.95,
        variance=:known,relative_step=cbrt(eps(Float64)),rank_tolerance=sqrt(eps(Float64)),
        gradient_tolerance=1e-4,difference_tolerance=1e-3,refined=nothing,
        prediction_tolerance=0.01,jacobian_tolerance=0.01,derivatives=cal.derivatives)
    isfinite(level) && 0<level<1 || throw(ArgumentError("confidence level must lie in (0,1)"))
    all(x->isfinite(x) && x>0,(difference_tolerance,prediction_tolerance,jacobian_tolerance)) ||
        throw(ArgumentError("diagnostic tolerances must be positive and finite"))
    estimate isa FitResult && !estimate.converged && throw(ArgumentError("optimizer did not converge"))
    report=fit_diagnostics(cal,estimate;variance,relative_step,rank_tolerance,gradient_tolerance,refined,derivatives)
    report.rank==report.parameters || throw(ArgumentError("rank-deficient fit has no bounded joint ellipsoid"))
    report.stationary || throw(ArgumentError("estimate is not stationary; inspect scaled_gradient and refit"))
    report.difference_error<=difference_tolerance || throw(ArgumentError("selected derivatives disagree with the independent finite-difference check"))
    if report.numerical!==nothing
        report.numerical.prediction_change<=prediction_tolerance &&
        report.numerical.jacobian_change<=jacobian_tolerance &&
        report.numerical.rank==report.rank || throw(ArgumentError("forward discretization has not converged relative to the observation errors"))
    end
    if variance===:estimated
        report.degrees_of_freedom>0 && report.variance_scale>0 ||
            throw(ArgumentError("estimating a common variance requires positive residual degrees of freedom and RSS"))
    end
    C=report.covariance
    isposdef(Symmetric(C)) || throw(ArgumentError("covariance is numerically singular; inspect scaling and rank"))
    p=report.parameters
    cutoff=variance===:known ? Distributions.quantile(Distributions.Chisq(p),level) :
        p*Distributions.quantile(Distributions.FDist(p,report.degrees_of_freedom),level)
    cutoff,coverage=promote(float(cutoff),float(level))
    ConfidenceEllipsoid(report.center,C,cutoff,coverage,report)
end

function Base.in(z::AbstractVector,region::ConfidenceEllipsoid)
    length(z)==length(region.center) || throw(DimensionMismatch("parameter dimension mismatch"))
    all(isfinite,z) || return false
    delta=z-region.center
    dot(delta,Symmetric(region.covariance)\delta)<=region.cutoff*(1+1e-12)
end

"""
    ellipse_points(region; pair=(1,2), n=181)

2×n boundary points of the region's projection onto a coordinate pair. The
cutoff remains that of the full joint region; this is not a newly calculated
pairwise confidence level. Apply physical transforms explicitly to the points.
"""
function ellipse_points(region::ConfidenceEllipsoid;pair=(1,2),n=181)
    length(pair)==2 && pair[1]!=pair[2] && all(i->i isa Integer && 1<=i<=length(region.center),pair) ||
        throw(ArgumentError("select two distinct parameter coordinates"))
    n isa Integer && n>=8 || throw(ArgumentError("use at least eight boundary points"))
    indices=collect(pair); C=region.covariance[indices,indices]
    eig=eigen(Symmetric(C))
    angles=range(0,2π;length=n)
    circle=permutedims(hcat(cos.(angles),sin.(angles)))
    region.center[indices].+eig.vectors*Diagonal(sqrt.(region.cutoff.*eig.values))*circle
end

"""Fit from explicit starting vectors and select the lowest-objective converged result. Requires an installed fitting backend; reports failed starts."""
function fit_multistart(cal::CalibrationProblem,starts;kwargs...)
    starts=collect(starts); isempty(starts) && throw(ArgumentError("supply starting vectors"))
    results=Any[]; failures=String[]
    for start in starts
        result=try
            fit(cal,start;kwargs...)
        catch e
            e isa NumericalFailure || rethrow()
            push!(failures,sprint(showerror,e)); nothing
        end
        push!(results,result)
    end
    valid=[r for r in results if r!==nothing && r.converged && isfinite(r.minimum)]
    isempty(valid) && throw(NumericalFailure("no starting point produced a converged finite fit"))
    (;best=valid[argmin([r.minimum for r in valid])],results,failures)
end

"""Load Optim to evaluate `profile_likelihood(calibration, fitted, parameter_index, values; ...)`, reoptimizing nuisance parameters at each fixed value."""
function profile_likelihood end
