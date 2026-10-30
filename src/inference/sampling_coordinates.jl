function _validate_sampling_map(map)
    all(k->hasproperty(map,k),(:to_sampler,:from_sampler,:logabsdetjac,:label)) ||
        throw(ArgumentError("sampling_map needs to_sampler, from_sampler, logabsdetjac (of from_sampler), and label"))
    nothing
end

# Check the change of variables where it is actually evaluated, including
# trajectories beyond chain initialization. This establishes local consistency,
# not a proof that an arbitrary callback is a global bijection of the support.
function _sampling_map_state(map,y)
    z=map.from_sampler(y)
    z isa AbstractVector && length(z)==length(y) || throw(ArgumentError("sampling_map must preserve parameter dimension"))
    all(isfinite,z) || throw(NumericalFailure("sampling_map produced nonfinite inference coordinates"))
    back=map.to_sampler(z)
    back isa AbstractVector && length(back)==length(y) || throw(ArgumentError("sampling_map inverse dimension mismatch"))
    all(isapprox.(back,y;rtol=1e-10,atol=1e-12)) ||
        throw(ArgumentError("sampling coordinate maps are not inverses at an evaluated point"))
    J=_ad_evaluate(()->ForwardDiff.jacobian(map.from_sampler,y))
    all(isfinite,J) || throw(NumericalFailure("nonfinite sampling_map Jacobian"))
    actual=first(logabsdet(J))
    isfinite(actual) || throw(NumericalFailure("singular sampling_map Jacobian"))
    declared=map.logabsdetjac(y)
    isfinite(declared) && isapprox(declared,actual;rtol=1e-9,atol=1e-10) ||
        throw(ArgumentError("sampling map Jacobian is inconsistent with from_sampler at an evaluated point"))
    (;z,jacobian=J,logjac=declared,actual_logjac=actual,
      inverse_error=maximum(abs.(back.-y)./max.(1,abs.(y))))
end

"""
    check_sampling_map(problem, map, z; derivatives=problem.derivatives,
                       step=cbrt(eps(Float64)), tolerance=1e-3)

Check a supplied sampling map at an inference-coordinate point `z`. The named
fields are `to_sampler`, `from_sampler`, `logabsdetjac` and `label`.
`logabsdetjac(y)` means log|det D(from_sampler)(y)|. The sampler density is
`sampling_logposterior(problem,from_sampler(y)) + logabsdetjac(y)`.

Checks both inverse directions, the Jacobian determinant, and the pulled-back
posterior gradient against independent step-halved differences. Repeat at
representative points and tails in the ORIGINAL posterior support. Callbacks
must be pure and define a bijection covering that entire support; finite tests
cannot certify global coverage or injectivity. The sampler also checks inverse
and determinant consistency at every mapped target evaluation. No priors or
physical parameter transforms are redefined by this auxiliary change.
"""
function check_sampling_map(problem::EstimationProblem,map,z;
        derivatives=problem.derivatives,step=cbrt(eps(Float64)),tolerance=1e-3,_counts=nothing)
    _validate_sampling_map(map)
    isfinite(tolerance) && tolerance>0 || throw(ArgumentError("positive map-check tolerance required"))
    length(z)==length(problem.parameters) && all(isfinite,z) || throw(ArgumentError("invalid inference-coordinate check point"))
    y=map.to_sampler(z)
    y isa AbstractVector && length(y)==length(z) && all(isfinite,y) || throw(ArgumentError("invalid sampler-coordinate check point"))
    state=_sampling_map_state(map,y)
    all(isapprox.(state.z,z;rtol=1e-10,atol=1e-12)) || throw(ArgumentError("sampling_map does not invert the supplied inference point"))
    density=sampling_logposterior(problem,state.z;_counts)
    isfinite(density) || throw(ArgumentError("map-check point is outside posterior support or has an invalid solve"))
    function target(v)
        at=_sampling_map_state(map,v)
        sampling_logposterior(problem,at.z;_counts)+at.logjac
    end
    gradient=state.jacobian'*sampling_gradient(problem,state.z;derivatives,_counts)+
        _ad_evaluate(()->ForwardDiff.gradient(map.logabsdetjac,y))
    coarse=_fd_gradient(target,y;step)
    fine=_fd_gradient(target,y;step=step/2)
    error=norm(gradient-fine)/max(1,norm(fine))
    halving=norm(coarse-fine)/max(1,norm(fine))
    (;valid=all(isfinite,gradient) && error<=tolerance && halving<=tolerance,
      y=copy(y),z=copy(state.z),inverse_error=state.inverse_error,
      logjac=state.logjac,actual_logjac=state.actual_logjac,
      logdensity_inference=density,logdensity_sampler=density+state.logjac,
      gradient,reference=fine,error,halving)
end
