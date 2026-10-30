module IsothermalProteinSampling
using PopulationBalanceModels, ForwardDiff, LinearAlgebra
using Distributions: Normal, MixtureModel, cdf, pdf, logcdf, logccdf, logpdf, quantile

export bounded_rate_coordinates, transported_rate_coordinates, expected_information_tuning, curvature_tuning

# The inverse-CDF derivative follows the inverse-function theorem, including
# nested Duals. Differentiate the inverse, not the bracketing root-search path.
_mixture_quantile(d,p)=quantile(d,p)
function _mixture_quantile(d,p::ForwardDiff.Dual{Tag}) where Tag
    x=_mixture_quantile(d,ForwardDiff.value(p))
    density=pdf(d,x)
    isfinite(PopulationBalanceModels._primal(density)) && PopulationBalanceModels._primal(density)>0 ||
        throw(NumericalFailure("sampling transport inverse CDF has unresolved tail density"))
    ForwardDiff.Dual{Tag}(x,ForwardDiff.partials(p)/density)
end

"""
Fixed, invertible coordinate preconditioner for this case's five-dimensional
posterior. A Gaussian-mixture CDF rescales the logit(logAJ) coordinate. Smooth
conditional shifts and Cholesky factors straighten the remaining rate
coordinates, including a normal-CDF transform of bounded g. The mixture and
conditional matrices describe SAMPLER GEOMETRY, not physical populations,
priors, observation noise or posterior covariance estimates.

All coefficients are frozen from a calibration-only tuning pilot. The package
still evaluates the original posterior. Its inverse-map log determinant is
the base-rate determinant, the parent inverse-CDF factor, the conditional
triangular determinant and the bounded-g inverse factor, each exactly once.
Underflowed CDF tails raise NumericalFailure; no probabilities are floored or
clipped, and the package's inverse/Jacobian checks remain active.
"""
function transported_rate_coordinates(case,problem,reference,configuration;
        growth_reference_supersaturation=nothing)
    length(problem.parameters)==5 || throw(ArgumentError("this sampler preconditioner requires the five-parameter CNT + GRD model"))
    base=bounded_rate_coordinates(case,problem,reference;growth_reference_supersaturation)
    means=Float64.(configuration.mixture_means)
    widths=Float64.(configuration.mixture_scales)
    weights=Float64.(configuration.mixture_weights)
    !isempty(means) && length(means)==length(widths)==length(weights) &&
        all(isfinite,vcat(means,widths,weights)) && all(>(0),widths) && all(>(0),weights) &&
        isapprox(sum(weights),1.;atol=1e-12,rtol=0) ||
        throw(ArgumentError("sampling transport requires finite mixture means, positive widths/weights and weights summing to one"))
    coefficient=reshape(Float64.(configuration.polynomial_coefficients),4,4)
    low=reshape(Float64.(configuration.low_cholesky),4,4)
    high=reshape(Float64.(configuration.high_cholesky),4,4)
    lowmean=Float64.(configuration.low_mean);highmean=Float64.(configuration.high_mean)
    all(isfinite,vcat(vec(coefficient),vec(low),vec(high),lowmean,highmean)) &&
        length(lowmean)==length(highmean)==4 && istril(low) && istril(high) &&
        all(>(0),diag(low)) && all(>(0),diag(high)) ||
        throw(ArgumentError("sampling transport needs finite coefficients and lower triangular factors with positive diagonals"))
    all(isfinite,(configuration.parent_center,configuration.parent_scale,
        configuration.blend_center,configuration.blend_width)) &&
        configuration.parent_scale>0 && configuration.blend_width>0 ||
        throw(ArgumentError("sampling transport requires finite centers and positive finite coordinate scales"))
    mixture=MixtureModel([Normal(m,s) for (m,s) in zip(means,widths)],weights)
    reflected=MixtureModel([Normal(-m,s) for (m,s) in zip(means,widths)],weights)
    normal=Normal()
    bound=problem.parameters[1].transform
    sigmoid(v)=v>=0 ? inv(1+exp(-v)) : exp(v)/(1+exp(v))
    function probit(v)
        p=sigmoid(-abs(v))
        PopulationBalanceModels._primal(p)>0 ||
            throw(NumericalFailure("sampling transport bounded-coordinate inverse CDF underflow"))
        v>0 ? -quantile(normal,p) : quantile(normal,p)
    end
    function logitnormal(v)
        PopulationBalanceModels._primal(cdf(normal,-abs(v)))>0 ||
            throw(NumericalFailure("sampling transport bounded-coordinate CDF tail is not representable"))
        logcdf(normal,v)-logccdf(normal,v)
    end
    function parent_forward(v)
        p=cdf(mixture,v)
        tail=p<=.5 ? p : cdf(reflected,-v)
        PopulationBalanceModels._primal(tail)>0 ||
            throw(NumericalFailure("sampling transport marginal CDF tail is not representable"))
        p<=.5 ? quantile(normal,tail) : -quantile(normal,tail)
    end
    function parent_inverse(v)
        probability=cdf(normal,v<=0 ? v : -v)
        PopulationBalanceModels._primal(probability)>0 ||
            throw(NumericalFailure("sampling transport exceeds floating-point CDF tail support"))
        v<=0 ? _mixture_quantile(mixture,probability) : -_mixture_quantile(reflected,probability)
    end
    function conditional(parent)
        A=bound.lower+(bound.upper-bound.lower)*sigmoid(parent)
        x=(A-configuration.parent_center)/configuration.parent_scale
        w=sigmoid((A-configuration.blend_center)/configuration.blend_width)
        shift=vec([1,x,x^2,x^3]'*coefficient)+(1-w)*lowmean+w*highmean
        factor=LowerTriangular((1-w)*low+w*high)
        shift,factor
    end
    function from_sampler(u)
        a=parent_inverse(u[1]);shift,L=conditional(a)
        v=copy(u);v[1]=a;v[2:5]=shift+L*u[2:5]
        v[4]=logitnormal(v[4]);base.from_sampler(v)
    end
    function to_sampler(z)
        v=base.to_sampler(z);v[4]=probit(v[4]);a=v[1]
        shift,L=conditional(a);v[2:5]=L\(v[2:5]-shift)
        v[1]=parent_forward(a);v
    end
    function logabsdetjac(u)
        a=parent_inverse(u[1]);shift,L=conditional(a)
        v=copy(u);v[1]=a;v[2:5]=shift+L*u[2:5]
        g=v[4];v[4]=logitnormal(g)
        base.logabsdetjac(v)+logpdf(normal,u[1])-logpdf(mixture,a)+sum(log,diag(L))+
            logpdf(normal,g)-logcdf(normal,g)-logccdf(normal,g)
    end
    (;to_sampler,from_sampler,logabsdetjac,label=:CNT_transport_rates)
end

"""
Bound-preserving rate coordinates for the declared CNT law.
Keep logit(logAJ). Let b = B(gamma)/log(S0)^2 and v = inverse_softplus(b).
The second coordinate is v-logAJ, approximately minus the initial log rate
when b is large. The positive barrier b=softplus(y2+logAJ) avoids a curved
hard prior wall and the changing widths of a log-gamma shear.
The growth coordinate is log G at a fixed declared supersaturation (initial
supersaturation by default). Choosing a training-range reference changes only
sampler geometry; the physical growth law and all priors remain unchanged.

The inverse-map determinant is sigmoid(v)/(3*softplus(v)); it is included
once by the package sampler. This is a coordinate transformation, not a
changed prior or a physical approximation. The small exponential corrections
omitted below -36 are below Float64 relative precision.
"""
function bounded_rate_coordinates(case,problem,reference;growth_reference_supersaturation=nothing)
    names=[s.name for s in problem.parameters]
    length(names) in (4,5) && names[1:4]==[:logAJ,:gamma,:AG,:g] ||
        throw(ArgumentError("these case coordinates require logAJ, gamma, AG, g, and optional GRD"))
    a=problem.parameters[1].transform;b=problem.parameters[4].transform
    a isa LogitTransform && b isa LogitTransform ||
        throw(ArgumentError("logAJ and g require their declared logit coordinates"))
    all(problem.parameters[i].transform isa LogTransform &&
        problem.parameters[i].transform.scale==1 for i in (2,3)) ||
        throw(ArgumentError("gamma and AG require the case's unit-scale logarithmic coordinates"))
    p=physical_parameters(problem,reference,:batch)
    Sref=isnothing(growth_reference_supersaturation) ? case.C0/p.cstar : growth_reference_supersaturation
    isfinite(Sref) && Sref>1 || throw(ArgumentError("sampler growth reference supersaturation must exceed one"))
    logS2=log(case.C0/p.cstar)^2;logexcess=log(Sref-1)
    logK=log(case.barrier(1.0))
    sigmoid(v)=v>=0 ? inv(1+exp(-v)) : exp(v)/(1+exp(v))
    A(v)=a.lower+(a.upper-a.lower)*sigmoid(v)
    exponent(v)=b.lower+(b.upper-b.lower)*sigmoid(v)
    softplus(v)=v>0 ? v+log1p(exp(-v)) : log1p(exp(v))
    logsoftplus(v)=v < -36 ? v : log(softplus(v))
    function to_sampler(z)
        lb=logK+3z[2]-log(logS2)
        v=if lb < -36
            lb
        else
            barrier=exp(lb)
            isfinite(barrier) || throw(NumericalFailure("CNT sampling barrier overflow"))
            barrier+log(-expm1(-barrier))
        end
        y=copy(z);y[2]=v-A(z[1])
        y[3]+=log(.001)+exponent(z[4])*logexcess
        y
    end
    function from_sampler(y)
        v=y[2]+A(y[1])
        z=copy(y);z[2]=(logsoftplus(v)+log(logS2)-logK)/3
        z[3]-=log(.001)+exponent(y[4])*logexcess
        z
    end
    logabsdetjac(y)=let v=y[2]+A(y[1])
        -softplus(-v)-log(3)-logsoftplus(v)
    end
    (;to_sampler,from_sampler,logabsdetjac,label=:CNT_soft_barrier_rates)
end

"""Gaussian expected information plus actual prior curvature, for tuning only.
This omits residual curvature and is not a verified posterior covariance or
Laplace approximation. A nonpositive matrix fails; no diagonal ridge is added.
"""
function expected_information_tuning(problem,z,map)
    y=map.to_sampler(z)
    J=ForwardDiff.jacobian(v->vcat(vec.(only(predictions(problem,map.from_sampler(v))))...),y)
    W=copy(J);offset=0
    for measurement in only(problem.experiments).measurements
        sigma=measurement.likelihood.sigma
        sigma isa Number || sigma isa AbstractArray ||
            throw(ArgumentError("this case's tuning expects fixed Gaussian standard deviations"))
        rows=offset+1:offset+length(measurement.data)
        W[rows,:]./=sigma isa Number ? sigma : vec(sigma)
        offset+=length(measurement.data)
    end
    prior=-ForwardDiff.hessian(v->sampling_logprior(problem,map.from_sampler(v))+map.logabsdetjac(v),y)
    information=Symmetric(W'*W+prior)
    factor=cholesky(information)
    covariance=Matrix(Symmetric(factor\Matrix{Float64}(I,length(z),length(z))))
    (;covariance,information=Matrix(information),condition_number=cond(information),
      interpretation="Gaussian expected information plus prior curvature; sampler tuning only, not Laplace")
end

"""Initial sampler metric from local observed curvature in inference coordinates.
The inverse curvature is pushed into sampler coordinates by the local Jacobian.
This is tuning only: it is not a verified Laplace approximation, and no intervals
are obtained from this matrix. Nonfinite or nonpositive curvature fails without
a ridge. Distinct calibration basins may initialize distinct chain metrics.
"""
function curvature_tuning(problem,z,map)
    H=ForwardDiff.hessian(v->-sampling_logposterior(problem,v),z)
    all(isfinite,H) || throw(ArgumentError("nonfinite local posterior curvature; cannot initialize this sampler metric"))
    factor=cholesky(Symmetric(H))
    inverse=factor\Matrix{Float64}(I,length(z),length(z))
    J=ForwardDiff.jacobian(map.to_sampler,z)
    covariance=Matrix(Symmetric(J*inverse*J'))
    cholesky(Symmetric(covariance))
    (;covariance,information=Matrix(Symmetric(H)),condition_number=cond(covariance),
      interpretation="Local inference-coordinate observed curvature, transformed for sampler tuning only; not a verified Laplace covariance")
end
end
