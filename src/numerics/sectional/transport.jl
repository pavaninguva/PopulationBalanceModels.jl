_minmod(a,b,c)=sign(a)==sign(b)==sign(c) ? sign(a)*min(abs(a),abs(b),abs(c)) : zero(a)
function _muscl(n,j,scheme::MUSCL,vel,periodic,g)
    m=length(n); i=vel>=0 ? j-1 : j
    ni=n[periodic ? mod1(i,m) : i]
    (!periodic && (i==1 || i==m)) && return ni
    at(k)=n[periodic ? mod1(k,m) : k]
    center(k)=periodic ? g.centers[mod1(k,m)]+fld(k-1,m)*(last(g.edges)-first(g.edges)) : g.centers[k]
    slope=_minmod(scheme.theta*(at(i)-at(i-1))/(center(i)-center(i-1)),
        (at(i+1)-at(i-1))/(center(i+1)-center(i-1)),
        scheme.theta*(at(i+1)-at(i))/(center(i+1)-center(i)))
    width=g.widths[periodic ? mod1(i,m) : i]
    # Rescale the slope, keeping the cell average fixed, so the reconstructed
    # endpoints of a nonnegative cell average remain nonnegative.
    maxslope=2*max(ni,zero(ni))/width
    slope=sign(slope)*min(abs(slope),maxslope)
    ni+slope*(g.edges[j]-center(i))
end
_reconstruct(n,j,op,vel,periodic)=op.scheme isa MUSCL ?
    _muscl(n,j,op.scheme,vel,periodic,op.grid) : _face_density(n,j,op.scheme,vel,periodic)

# Reconstruct a left state at i+1/2 from five cell averages around i.
function _weno(a,b,c,d,e,ε)
    q0 = (2a-7b+11c)/6
    q1 = (-b+5c+2d)/6
    q2 = (2c+5d-e)/6
    β0 = 13/12*(a-2b+c)^2 + 1/4*(a-4b+3c)^2
    β1 = 13/12*(b-2c+d)^2 + 1/4*(b-d)^2
    β2 = 13/12*(c-2d+e)^2 + 1/4*(3c-4d+e)^2
    a0 = (1/10)/(ε+β0)^2; a1 = (6/10)/(ε+β1)^2; a2 = (3/10)/(ε+β2)^2
    (a0*q0+a1*q1+a2*q2)/(a0+a1+a2)
end
function _face_density(n, j, scheme::Upwind, vel, periodic)
    m=length(n)
    n[periodic ? mod1(vel >= 0 ? j-1 : j,m) : (vel >= 0 ? j-1 : j)]
end
function _face_density(n, j, scheme::WENO5, vel, periodic)
    m = length(n)
    if !periodic && !(4 <= j <= m-2)
        return _face_density(n,j,Upwind(),vel,false)
    end
    at(i) = n[periodic ? mod1(i,m) : i]
    i = j-1
    if vel >= 0
        _weno(at(i-2),at(i-1),at(i),at(i+1),at(i+2),scheme.epsilon)
    else
        _weno(at(i+3),at(i+2),at(i+1),at(i),at(i-1),scheme.epsilon)
    end
end

function _growth_fluxes(term::Growth, N, op, p, t, ctx)
    g=op.grid; m=length(g); n=N ./ g.widths
    F=similar(N,m+1)
    periodic=op.bc.left isa Periodic
    if periodic
        vleft=term.rate(first(g.edges),p,t,ctx)
        vright=term.rate(last(g.edges),p,t,ctx)
        isapprox(vleft,vright;rtol=1e-10,atol=1e-14) ||
            throw(ArgumentError("growth rate must match at periodic boundaries"))
        F[1]=vleft*_reconstruct(n,1,op,vleft,true)
        F[end]=F[1]
    else
        vl=term.rate(first(g.edges),p,t,ctx); vr=term.rate(last(g.edges),p,t,ctx)
        F[1]=op.bc.left isa InflowFlux ? _nonnegative(op.bc.left.rate(p,t,ctx),"inward number flux") :
            (op.bc.left isa NoFlux || vl>=0 ? zero(vl) : vl*n[1])
        F[end]=op.bc.right isa InflowFlux ? -_nonnegative(op.bc.right.rate(p,t,ctx),"inward number flux") :
            (op.bc.right isa NoFlux || vr<=0 ? zero(vr) : vr*n[end])
    end
    for j in 2:m
        vel=term.rate(g.edges[j],p,t,ctx)
        F[j]=vel*_reconstruct(n,j,op,vel,periodic)
    end
    all(isfinite,F) || throw(NumericalFailure("nonfinite size-space flux"))
    F
end
