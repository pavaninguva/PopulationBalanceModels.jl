"""
Compact fourth-order spatial derivative with alternating directional implicit
Cayley solves. time_order=2 is factorized Crank–Nicolson; time_order=4 composes
three symmetric steps (one negative substep). Restricted to homogeneous,
constant-velocity transport on periodic uniform axes. This is a fully specified
verified variant, NOT a transcription of Li et al. (2026), Eqs. 6–16.
"""
struct HOCADI
    time_order::Int
    function HOCADI(;time_order=4)
        time_order in (2,4) || throw(ArgumentError("time_order must be 2 or 4"))
        new(time_order)
    end
end

# Cyclic tridiagonal factorization: a tridiagonal LU plus a rank-two Woodbury
# correction. Setup O(n), each directional line solve O(n); no dense n×n solve.
struct _CyclicFactor{F,Z,S}
    factor::F
    Z::Z
    correction::S
end
function _cyclic_factor(n,lower,diagonal,upper)
    f=lu(Tridiagonal(fill(lower,n-1),fill(diagonal,n),fill(upper,n-1)))
    U=zeros(n,2); U[1,1]=lower; U[end,2]=upper
    Z=f\U; S=Matrix{Float64}(I,2,2)+Z[[n,1],:]
    _CyclicFactor(f,Z,lu(S))
end
function _cyclic_solve!(dest,f::_CyclicFactor,b)
    copyto!(dest,b); ldiv!(f.factor,dest)
    correction=f.correction\[dest[end],dest[1]]
    for i in eachindex(dest)
        dest[i]-=f.Z[i,1]*correction[1]+f.Z[i,2]*correction[2]
    end
    dest
end
function _compact_axis!(out,N,dimension,velocity,h,dt,factor,b)
    n=dimension==1 ? size(N,1) : size(N,2)
    lines=dimension==1 ? size(N,2) : size(N,1)
    a=3dt*velocity/(8h)
    for j in 1:lines
        src=dimension==1 ? view(N,:,j) : view(N,j,:)
        dest=dimension==1 ? view(out,:,j) : view(out,j,:)
        for i in 1:n
            b[i]=src[i]+(0.25+a)*src[mod1(i-1,n)]+(0.25-a)*src[mod1(i+1,n)]
        end
        _cyclic_solve!(dest,factor,b)
    end
end
function _simulate(prob::PBMProblem,method::HOCADI;dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    velocities=_constant_transport(prob); op=prob.operator; g=op.grid
    all(b->b.left isa Periodic,(op.bc.x,op.bc.y)) || throw(ArgumentError("HOCADI currently requires periodic axes; physical one-sided compact closures are not implemented"))
    min(size(g)...)>=5 || throw(ArgumentError("compact solver needs at least five bins per axis"))
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    γ=1/(2-2^(1/3)); composition=method.time_order==4 ? (γ,1-2γ,γ) : (1.0,)
    N=copy(prob.u0); tmp=similar(N); states=typeof(N)[]; cache=Dict{Tuple{Int,Float64},Any}()
    bx=zeros(size(N,1)); by=zeros(size(N,2)); t=first(prob.tspan); steps=0
    for target in ts
        while t<target
            h=min(dt,target-t); t+h>t || throw(NumericalFailure("time step cannot advance"))
            steps+=1; steps<=maxsteps || throw(NumericalFailure("maximum step count exceeded"))
            for c in composition
                δ=c*h
                fx=get!(cache,(1,Float64(δ))) do
                    a=3δ*velocities[1]/(8first(g.x.widths)); _cyclic_factor(size(N,1),0.25-a,1.0,0.25+a)
                end
                fy=get!(cache,(2,Float64(δ))) do
                    a=3δ*velocities[2]/(8first(g.y.widths)); _cyclic_factor(size(N,2),0.25-a,1.0,0.25+a)
                end
                _compact_axis!(tmp,N,1,velocities[1],first(g.x.widths),δ,fx,bx)
                _compact_axis!(N,tmp,2,velocities[2],first(g.y.widths),δ,fy,by)
            end
            _checkstate(N,check_nonnegative ? negative_tolerance : Inf)
            t=h==target-t ? target : t+h
        end
        push!(states,copy(N))
    end
    PBMSolution(ts,states,prob,steps,(policy=:HOCADI,accepted_steps=steps,rejected_steps=0))
end
