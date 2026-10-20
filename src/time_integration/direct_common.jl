function _constant_transport(prob)
    op=prob.operator
    op isa CompiledPBM2D || throw(ArgumentError("this evolution method requires a 2D population"))
    length(op.model.terms)==1 && only(op.model.terms) isa Growth2D || throw(ArgumentError("this evolution method supports homogeneous growth only; source splitting is not silently applied"))
    rate=only(op.model.terms).rate
    rate isa Tuple || throw(ArgumentError("explicit constant velocity tuple is required"))
    all(_uniform,(op.grid.x,op.grid.y)) || throw(ArgumentError("uniform axes are required"))
    rate
end
