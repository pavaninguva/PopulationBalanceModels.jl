function _tensor_flux(term,n,axis,other,dimension,bc,scheme,p,t,ctx)
    m=length(axis); F=similar(n,m+1); periodic=bc.left isa Periodic
    velocity(s)=dimension==1 ? _velocity(term,s,other,p,t,ctx)[1] : _velocity(term,other,s,p,t,ctx)[2]
    vl=velocity(first(axis.edges)); vr=velocity(last(axis.edges))
    if periodic
        isapprox(vl,vr;rtol=1e-10,atol=1e-14) || throw(ArgumentError("periodic velocity must match at endpoints"))
        F[1]=vl*_face_density(n,1,scheme,vl,true,axis); F[end]=F[1]
    else
        localctx=merge(ctx,(;tangential_coordinate=other,axis=dimension))
        F[1]=bc.left isa InflowFlux ? _nonnegative(bc.left.rate(p,t,localctx),"inward flux density") :
            (bc.left isa NoFlux || vl>=0 ? zero(vl) : vl*n[1])
        F[end]=bc.right isa InflowFlux ? -_nonnegative(bc.right.rate(p,t,localctx),"inward flux density") :
            (bc.right isa NoFlux || vr<=0 ? zero(vr) : vr*n[end])
    end
    for i in 2:m
        v=velocity(axis.edges[i]); F[i]=v*_face_density(n,i,scheme,v,periodic,axis)
    end
    F
end
function _tensor_term!(dN,N,term::Growth2D,op,p,t,ctx)
    g=op.grid; f=density(g,N)
    schemes=op.scheme isa NamedTuple ? op.scheme : (x=op.scheme,y=op.scheme)
    for j in axes(N,2)
        F=_tensor_flux(term,view(f,:,j),g.x,g.y.centers[j],1,op.bc.x,schemes.x,p,t,ctx)
        for i in axes(N,1)
            dN[i,j]+=(F[i]-F[i+1])*g.y.widths[j]
        end
    end
    for i in axes(N,1)
        F=_tensor_flux(term,view(f,i,:),g.y,g.x.centers[i],2,op.bc.y,schemes.y,p,t,ctx)
        for j in axes(N,2)
            dN[i,j]+=(F[j]-F[j+1])*g.x.widths[i]
        end
    end
end
function _tensor_term!(dN,N,term::Union{Birth,Death,Dilution},op,p,t,ctx)
    g=op.grid
    for j in axes(N,2),i in axes(N,1)
        xy=(g.x.centers[i],g.y.centers[j])
        if term isa Birth
            dN[i,j]+=_nonnegative(term.source(xy,p,t,ctx),"birth density")*g.x.widths[i]*g.y.widths[j]
        else
            dN[i,j]-=_nonnegative(term.rate(xy,p,t,ctx),"removal hazard")*N[i,j]
        end
    end
end
function _pivotweights(axis,x)
    j=searchsortedlast(axis.centers,x)
    j==length(axis) && return ((j,one(x)),)
    w=(axis.centers[j+1]-x)/(axis.centers[j+1]-axis.centers[j])
    ((j,w),(j+1,1-w))
end
function _tensor_term!(dN,N,term::Nucleation2D,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
    for (i,a) in _pivotweights(op.grid.x,term.size[1]),(j,b) in _pivotweights(op.grid.y,term.size[2])
        dN[i,j]+=J*a*b
    end
end
function _tensor_term!(dN,N,term::DistributedNucleation,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"nucleation rate"); iszero(J) && return
    g=op.grid
    weights=[_nonnegative(term.shape((x,y),p,t,ctx),"nucleation shape")*g.x.widths[i]*g.y.widths[j]
        for (i,x) in enumerate(g.x.centers),(j,y) in enumerate(g.y.centers)]
    norm=term.normalization===:total_rate ? sum(weights) : one(J)
    isfinite(norm) && norm>0 || throw(NumericalFailure("2D source shape is unresolved"))
    dN .+= (J/norm).*weights
end
