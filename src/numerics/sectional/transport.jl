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
_face_density(n,j,scheme,vel,periodic,g)=_face_density(n,j,scheme,vel,periodic)
_face_density(n,j,scheme::MUSCL,vel,periodic,g)=_muscl(n,j,scheme,vel,periodic,g)
_reconstruct(n,j,op,vel,periodic)=_face_density(n,j,op.scheme,vel,periodic,op.grid)

function _face_density(n,j,::VanLeer,vel,periodic,g)
    m=length(n);i=vel>=0 ? j-1 : j
    !periodic && (i==1 || i==m) && return n[i]
    at(k)=n[periodic ? mod1(k,m) : k]
    x(k)=periodic ? g.centers[mod1(k,m)]+fld(k-1,m)*(last(g.edges)-first(g.edges)) : g.centers[k]
    left=(at(i)-at(i-1))/(x(i)-x(i-1));right=(at(i+1)-at(i))/(x(i+1)-x(i))
    slope=left*right>0 ? 2left*right/(left+right) : zero(left)
    width=g.widths[periodic ? mod1(i,m) : i]
    slope=sign(slope)*min(abs(slope),2max(at(i),zero(at(i)))/width)
    at(i)+slope*(g.edges[j]-x(i))
end

function _face_density(n,j,scheme::WENO3,vel,periodic,g)
    m=length(n);i=vel>=0 ? j-1 : j;direction=vel>=0 ? 1 : -1
    !periodic && !(1<=i-direction<=m && 1<=i+direction<=m) && return n[i]
    index(k)=periodic ? mod1(k,m) : k
    # Orient and scale the local coordinate so the upwind cell is [-1,0].
    h=g.widths[index(i)];hu=g.widths[index(i-direction)]/h;hd=g.widths[index(i+direction)]/h
    n0=n[index(i)];nu=n[index(i-direction)];nd=n[index(i+direction)]
    su=(n0-nu)/((1+hu)/2);sd=(nd-n0)/((1+hd)/2)
    q0=n0+su/2;q1=n0+sd/2
    # Exact linear weights from reconstruction of the monomial x². This
    # preserves quadratic cell averages on nonuniform meshes as well.
    square_u=(1+hu/2)^2+hu^2/12;square_0=1/3;square_d=hd^2/3
    e0=square_0+(square_0-square_u)/(1+hu)
    e1=square_0+(square_d-square_0)/(1+hd)
    d0=e1/(e1-e0);d1=1-d0
    a0=d0/(scheme.epsilon+su^2)^2;a1=d1/(scheme.epsilon+sd^2)^2
    (a0*q0+a1*q1)/(a0+a1)
end

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
function _face_density(n,j,::Centered,vel,periodic)
    at(i)=n[periodic ? mod1(i,length(n)) : i]
    (at(j-1)+at(j))/2
end
function _face_density(n,j,::LinearUpwind,vel,periodic)
    i=vel>=0 ? j-1 : j; upstream=vel>=0 ? i-1 : i+1
    !periodic && !(1<=upstream<=length(n)) && return n[i]
    at(k)=n[periodic ? mod1(k,length(n)) : k]
    (3at(i)-at(upstream))/2
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
