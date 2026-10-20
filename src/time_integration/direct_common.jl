function _constant_transport(prob)
    op=prob.operator
    op isa CompiledPBM2D || throw(ArgumentError("this evolution method requires a 2D population"))
    length(op.model.terms)==1 && only(op.model.terms) isa Growth2D || throw(ArgumentError("this evolution method supports homogeneous growth only; source splitting is not silently applied"))
    rate=only(op.model.terms).rate
    rate isa Tuple || throw(ArgumentError("explicit constant velocity tuple is required"))
    all(_uniform,(op.grid.x,op.grid.y)) || throw(ArgumentError("uniform axes are required"))
    rate
end
function _evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    isfinite(dt) && dt>0 || throw(ArgumentError("dt must be finite and positive"))
    isfinite(negative_tolerance) && negative_tolerance>=0 || throw(ArgumentError("invalid negativity tolerance"))
    maxsteps>0 || throw(ArgumentError("maxsteps must be positive"))
    ts=float.(collect(saveat)); t0,tf=prob.tspan
    !isempty(ts) && all(isfinite,ts) && all(diff(ts).>0) && t0<=first(ts)<=last(ts)<=tf || throw(ArgumentError("saveat must be increasing inside tspan"))
    ts
end
