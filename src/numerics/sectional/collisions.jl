function _local_rhs!(dN,N,term::Aggregation,op,p,t,ctx)
    x=op.grid.centers; axis=_measureaxis(term,op.grid); out0=zero(eltype(dN)); out1=out0
    for j in eachindex(N), i in 1:j
        (iszero(N[i]) || iszero(N[j])) && continue
        K=_nonnegative(term.kernel(x[i],x[j],p,t,ctx),"aggregation kernel")
        # Symmetry is part of this term's contract; check it rather than hide a factor-of-two error.
        Kji=_nonnegative(term.kernel(x[j],x[i],p,t,ctx),"aggregation kernel")
        isapprox(K,Kji;rtol=1e-10,atol=0) || throw(ArgumentError("aggregation kernel must be symmetric"))
        event=K*N[i]*N[j]*(i==j ? 0.5 : 1)
        dN[i]-=event; dN[j]-=event
        a,b=_deposit!(dN,axis,axis[i]+axis[j],event,term.overflow)
        out0+=a; out1+=b
    end
    (out0,out1)
end
function _local_rhs!(dN,N,term::Breakage,op,p,t,ctx)
    x=op.grid.centers; axis=_measureaxis(term,op.grid); out0=zero(eltype(dN)); out1=out0
    for j in eachindex(N)
        iszero(N[j]) && continue
        event=_nonnegative(term.rate(x[j],p,t,ctx),"breakage hazard")*N[j]
        iszero(event) && continue
        daughters=collect(term.daughters(x[j],p,t,ctx))
        isempty(daughters) && throw(ArgumentError("breakage needs a daughter distribution"))
        mass=zero(eltype(x))
        for (v,count) in daughters
            _nonnegative(count,"daughter expected number")
            isfinite(v) && 0 < v <= x[j] || throw(ArgumentError("daughter sizes must be in (0,parent]"))
            mass += count*_measure(term,v)
        end
        isapprox(mass,axis[j];rtol=1e-8,atol=0) || throw(ArgumentError("daughter counts must conserve the declared additive measure"))
        dN[j]-=event
        for (v,count) in daughters
            a,b=_deposit!(dN,axis,_measure(term,v),event*count,term.overflow)
            out0+=a; out1+=b
        end
    end
    (out0,out1)
end
