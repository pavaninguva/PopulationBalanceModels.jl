_plot_transform(t::IdentityTransform)=(;transform="identity",location=t.location,scale=t.scale)
_plot_transform(t::LogTransform)=(;transform="log",scale=t.scale)
_plot_transform(t::LogitTransform)=(;transform="logit",lower=t.lower,upper=t.upper)
_plot_transform(t::ShiftedLogTransform)=(;transform="shifted_log",bound=t.bound,upper_bound=t.upper,scale=t.scale)
_plot_transform(t::AbstractInferenceTransform)=throw(ArgumentError("plot_parameters needs explicit display maps for this custom transform; prepare values and use the shared Python renderer"))

_plot_endpoint(t::IdentityTransform,x)=(x-t.location)/t.scale
_plot_endpoint(t::LogTransform,x)=x<=0 ? -Inf : log(x)-log(t.scale)
_plot_endpoint(t::LogitTransform,x)=x<=t.lower ? -Inf : x>=t.upper ? Inf : _coordinate(t,x)
function _plot_endpoint(t::ShiftedLogTransform,x)
    delta=t.upper ? t.bound-x : x-t.bound
    delta<=0 ? -Inf : log(delta)-log(t.scale)
end
function _plot_prior_violation(spec,mean,variance)
    bounds=_prior_interval(spec.prior.distribution)
    bounds===nothing && return nothing # Unknown support, not zero violations.
    a,b=spec.prior.coordinates===:inference ? bounds :
        minmax(_plot_endpoint(spec.transform,bounds[1]),_plot_endpoint(spec.transform,bounds[2]))
    normal=Distributions.Normal(mean,sqrt(variance))
    Distributions.cdf(normal,a)+Distributions.ccdf(normal,b)
end

function _plot_display(spec,z,coordinates)
    coordinates===:inference && return z
    x=_physical(spec.transform,z)
    coordinates===:physical && return x
    x>0 ? log(x) : NaN
end

function _plot_prior(spec,x,coordinates)
    if coordinates===:inference
        z=x;theta=_physical(spec.transform,z)
        jac=spec.prior.coordinates===:physical ? _logjac(spec.transform,z) : zero(z)
    else
        theta=coordinates===:log ? exp(x) : x
        jac=coordinates===:log ? x : zero(x)
        if spec.prior.coordinates===:inference
            bounds=_transform_bounds(spec.transform)
            bounds===nothing || bounds[1]<theta<bounds[2] || return 0.
            z=_coordinate(spec.transform,theta)
            jac-=_logjac(spec.transform,z)
        else
            z=zero(theta) # unused for a physical prior
        end
    end
    value=spec.prior.coordinates===:physical ? theta : z
    density=exp(Distributions.logpdf(spec.prior.distribution,value)+jac)
    isnan(density) && throw(ArgumentError("undefined prior display density; select compatible coordinates"))
    density # An actual singular endpoint remains infinite, never replaced by zero.
end

function _parameter_data(problem::EstimationProblem,result;coordinates=:inference,indices=nothing,priors=true)
    coordinates in (:inference,:physical,:log) || throw(ArgumentError("coordinates must be :inference, :physical or :log (natural log of physical values)"))
    selected=indices===nothing ? collect(eachindex(problem.parameters)) : collect(indices)
    !isempty(selected) && all(i->1<=i<=length(problem.parameters),selected) && length(unique(selected))==length(selected) || throw(ArgumentError("select distinct valid parameter indices"))
    specs=problem.parameters[selected];n=length(specs)
    metadata=[merge(_plot_transform(s.transform),(;name=s.experiment===nothing ? string(s.name) : "$(s.experiment).$(s.name)",
        plotting_coordinate=string(coordinates),density_coordinates=string(s.prior.coordinates))) for s in specs]
    point=nothing;gaussian=nothing;samples=nothing;report=(;)
    if result isa Union{LaplaceApproximation,GaussNewtonApproximation}
        _check_approximation(problem,result) # parameter/convention check only, no model evaluations
        result.diagnostics.valid || throw(ArgumentError("approximation was rejected; plot its diagnostic report, not an accepted covariance"))
        center=copy(result.center[selected]);covariance=copy(result.covariance[selected,selected])
        gaussian=(;center,covariance,coordinates="inference")
        point=[_plot_display(spec,z,coordinates) for (spec,z) in zip(specs,center)]
        report=merge(result.diagnostics,(;marginal_prior_violation_probability=
            [_plot_prior_violation(specs[i],center[i],covariance[i,i]) for i in 1:n],
            support_probability_meaning="marginal probabilities, not the joint violation probability; no truncation"))
        raw=[range(center[i]-8sqrt(covariance[i,i]),center[i]+8sqrt(covariance[i,i]);length=1601) for i in 1:n]
    elseif result isa PosteriorSamples
        _check_parameterization(problem,result.coordinates)
        Z=reshape(result.z,size(result.z,1),:)[selected,:]
        # Renderer uses draws × parameters, while Z is parameter × draw.
        samples=hcat([[_plot_display(spec,z,coordinates) for z in Z[i,:]] for (i,spec) in enumerate(specs)]...)
        report=diagnostics(result)
        violations=[count(eachcol(Z)) do v
            value=spec.prior.coordinates===:inference ? v[i] : _physical(spec.transform,v[i])
            !isfinite(value) || !Distributions.insupport(spec.prior.distribution,value)
        end for (i,spec) in enumerate(specs)]
        report=merge(report,(;prior_support_violations=violations))
        raw=[copy(Z[i,:]) for i in 1:n]
    elseif result isa MAPResult
        result.convention===_MAP_CONVENTION || throw(ArgumentError("MAP convention differs; historical results require explicit recorded-coordinate rendering"))
        _check_parameterization(problem,result.coordinates)
        length(result.minimizer)==length(problem.parameters) || throw(DimensionMismatch("MAP dimension differs"))
        point=[_plot_display(spec,result.minimizer[i],coordinates) for (spec,i) in zip(specs,selected)]
        report=(;passed=result.converged,point_estimate_only=true,convention=result.convention)
        raw=[range(result.minimizer[i]-.5,result.minimizer[i]+.5;length=1001) for i in selected]
    else
        throw(ArgumentError("use retained PosteriorSamples, MAPResult or an accepted Laplace/GaussNewtonApproximation"))
    end
    any(v->any(!isfinite,v),raw) && throw(ArgumentError("nonfinite retained coordinates; no draw filtering is performed"))
    prior_curves=priors ? map(specs,raw) do spec,z
        lo,hi=extrema(z)
        # Nonuniform mapped grid resolves transformed narrow peaks. This does
        # not renormalize a broad prior to a posterior-focused display window.
        grid=[_plot_display(spec,v,coordinates) for v in range(lo,hi;length=2001)]
        all(isfinite,grid) || throw(ArgumentError("nonfinite prior display coordinates; no clipping"))
        order=sortperm(grid);x=grid[order]
        (;x,density=[_plot_prior(spec,v,coordinates) for v in x])
    end : nothing
    (;samples,metadata,gaussian,point,priors=prior_curves,diagnostics=report,
      point_only=result isa MAPResult)
end
