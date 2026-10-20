function _local_rhs!(dN,N,term::Growth,op,p,t,ctx)
    F=_growth_fluxes(term,N,op,p,t,ctx)
    for i in eachindex(N)
        dN[i] += F[i]-F[i+1]
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
function _local_rhs!(dN,N,term::Birth,op,p,t,ctx)
    for i in eachindex(N)
        dN[i] += _nonnegative(term.source(op.grid.centers[i],p,t,ctx),"birth density")*op.grid.widths[i]
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
function _local_rhs!(dN,N,term::Death,op,p,t,ctx)
    for i in eachindex(N)
        dN[i] -= _nonnegative(term.rate(op.grid.centers[i],p,t,ctx),"death hazard")*N[i]
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
function _local_rhs!(dN,N,term::Dilution,op,p,t,ctx)
    for i in eachindex(N)
        dN[i]-=_nonnegative(term.rate(op.grid.centers[i],p,t,ctx),"dilution rate")*N[i]
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
function _local_rhs!(dN,N,term::DistributedNucleation,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"distributed nucleation rate")
    iszero(J) && return (zero(eltype(dN)),zero(eltype(dN)))
    weights=[_nonnegative(term.shape(x,p,t,ctx),"nucleation shape")*width
        for (x,width) in zip(op.grid.centers,op.grid.widths)]
    norm=term.normalization===:total_rate ? sum(weights) : one(J)
    norm>0 || throw(NumericalFailure("nucleation shape is unresolved on the grid"))
    dN .+= (J/norm).*weights
    (zero(eltype(dN)),zero(eltype(dN)))
end
function _local_rhs!(dN,N,term::Nucleation,op,p,t,ctx)
    rate=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
    _deposit!(dN,op.grid.centers,term.size,rate,:error)
end

function _local_rhs!(dN,N,term::CustomTerm,op,p,t,ctx)
    term.f!(dN,N,op.grid,p,t,ctx)
    (zero(eltype(dN)),zero(eltype(dN)))
end
