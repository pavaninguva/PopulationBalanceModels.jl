function _local_rhs!(dN,N,term::IntegratedBirth,op,p,t,ctx)
    for i in eachindex(N)
        dN[i]+=_nonnegative(term.integral(op.grid.edges[i],op.grid.edges[i+1],p,t,ctx),"integrated birth")
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
function _local_rhs!(dN,N,term::IntegratedNucleation,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
    iszero(J) && return (zero(eltype(dN)),zero(eltype(dN)))
    weights=[_nonnegative(term.fraction(op.grid.edges[i],op.grid.edges[i+1],p,t,ctx),"integrated nucleation fraction") for i in eachindex(N)]
    norm=term.normalization===:total_rate ? sum(weights) : one(J)
    isfinite(norm) && norm>0 || throw(NumericalFailure("nucleation has no resolved source weight"))
    dN .+= (J/norm).*weights
    (zero(eltype(dN)),zero(eltype(dN)))
end
