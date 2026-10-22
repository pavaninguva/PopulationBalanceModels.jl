"""Inguva–Braatz constant-growth CFL=1 dimensional shifts; integer cell travel only."""
struct ExactShift end


function _integer_cells(travel)
    isfinite(travel) && abs(travel-round(travel))<=1e-9 || throw(ArgumentError("ExactShift requires integer cell travel at every step/output; choose a compatible grid, dt and saveat"))
    round(Int,travel)
end
function _shift2d!(out,N,sx,sy,periodicx,periodicy)
    nx,ny=size(N)
    for j in 1:ny,i in 1:nx
        a=i-sx; b=j-sy
        out[i,j]=(!periodicx && !(1<=a<=nx)) || (!periodicy && !(1<=b<=ny)) ? zero(eltype(N)) : N[mod1(a,nx),mod1(b,ny)]
    end
end
function _simulate(prob::PBMProblem,method::ExactShift;dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    velocities=_constant_transport(prob); g=prob.operator.grid; bc=prob.operator.bc
    all(b->b.left isa Periodic || (b.left isa OpenBoundary && b.right isa OpenBoundary),(bc.x,bc.y)) ||
        throw(ArgumentError("ExactShift supports periodic or paired zero-inflow open boundaries"))
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    N=copy(prob.u0); tmp=similar(N); states=typeof(N)[]; t=first(prob.tspan); steps=0
    for target in ts
        while t<target
            h=min(dt,target-t); t+h>t || throw(NumericalFailure("time step cannot advance"))
            steps+=1; steps<=maxsteps || throw(NumericalFailure("maximum step count exceeded"))
            sx=_integer_cells(velocities[1]*h/first(g.x.widths)); sy=_integer_cells(velocities[2]*h/first(g.y.widths))
            _shift2d!(tmp,N,sx,sy,bc.x.left isa Periodic,bc.y.left isa Periodic); N,tmp=tmp,N
            _checkstate(N,check_nonnegative ? negative_tolerance : Inf)
            t=h==target-t ? target : t+h
        end
        push!(states,copy(N))
    end
    PBMSolution(ts,states,prob,steps,(policy=:ExactShift,accepted_steps=steps,rejected_steps=0))
end
