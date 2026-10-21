struct PBMProblem{O,U,T,P}
    operator::O
    u0::U
    tspan::T
    p::P
end
function PBMProblem(op::CompiledPBM,u0,tspan;p=NamedTuple())
    N=u0 isa AbstractVector ? reshape(float.(copy(u0)),:,1) : float.(copy(u0))
    N isa AbstractMatrix || throw(ArgumentError("u0 must be a vector or matrix of bin concentrations"))
    size(N,1)==length(op.grid) || throw(DimensionMismatch("u0 must have one row per bin"))
    size(N,2)==1 || throw(DimensionMismatch("BatchDomain requires one column"))
    all(isfinite,N) && all(N .>= 0) || throw(ArgumentError("initial concentrations must be finite and nonnegative"))
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] ||
        throw(ArgumentError("tspan must contain two finite increasing times"))
    ts=(float(tspan[1]),float(tspan[2]))
    PBMProblem(op,N,ts,p)
end

_check_problem_state(prob,N,tolerance)=_checkstate(N,tolerance)
remake(prob::PBMProblem;p=prob.p,u0=prob.u0,tspan=prob.tspan) = PBMProblem(prob.operator,u0,tspan;p)

struct PBMSolution{T,U,P}
    t::T
    u::U
    problem::P
    steps::Int
end
"""Explicit three-stage, third-order strong-stability-preserving Runge–Kutta."""
struct SSPRK33 end
_default_evolution(op)=SSPRK33()
_simulate(prob,method;kwargs...)=throw(ArgumentError("unsupported evolution method $(typeof(method)) for $(typeof(prob.operator))"))
Base.getindex(sol::PBMSolution,i::Integer)=sol.u[i]

function _checkstate(N,negative_tolerance)
    all(isfinite,N) || throw(NumericalFailure("nonfinite state during time integration"))
    minimum(N)>=-negative_tolerance ||
        throw(NumericalFailure("negative concentration during time integration; reduce dt, use Upwind, or add a proven positivity limiter"))
    nothing
end

# Shared by residual-based and direct evolution methods, including runs that
# only request the initial state and therefore never enter a stepping loop.
function _evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    isfinite(dt) && dt>0 || throw(ArgumentError("dt must be finite and positive"))
    isfinite(negative_tolerance) && negative_tolerance>=0 || throw(ArgumentError("invalid negativity tolerance"))
    maxsteps isa Integer && maxsteps>0 || throw(ArgumentError("maxsteps must be a positive integer"))
    ts=float.(collect(saveat)); t0,tf=prob.tspan
    !isempty(ts) && all(isfinite,ts) && all(diff(ts).>0) && t0<=first(ts)<=last(ts)<=tf ||
        throw(ArgumentError("saveat must be increasing inside tspan"))
    ts
end

"""
    simulate(problem; dt, saveat=[t0,tf], check_nonnegative=true,
             negative_tolerance=1e-12, maxsteps=10^7)

Explicit SSPRK(3,3) reference integrator. dt is a maximum fixed step; a step is
shortened to land on each output time. No clipping, automatic grid change, or
implicit stiffness treatment is performed. Choose dt using `suggest_dt` and
check convergence. For production adaptive/stiff integration use `odeproblem`.
"""
function simulate(prob::PBMProblem;dt,saveat=collect(prob.tspan),check_nonnegative=true,
                  negative_tolerance=1e-12,maxsteps=10^7,method=_default_evolution(prob.operator))
    method isa SSPRK33 || return _simulate(prob,method;dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    t0=first(prob.tspan)
    N=copy(prob.u0); U1=similar(N); U2=similar(N); k=similar(N)
    states=typeof(N)[]; t=t0; steps=0
    for target in ts
        while t<target
            h=min(dt,target-t)
            t+h>t || throw(NumericalFailure("time step is too small to advance"))
            steps+=1
            steps<=maxsteps || throw(NumericalFailure("maximum step count exceeded"))
            rhs!(k,N,prob.operator,prob.p,t)
            @. U1=N+h*k
            check_nonnegative && _check_problem_state(prob,U1,negative_tolerance)
            rhs!(k,U1,prob.operator,prob.p,t+h)
            @. U2=3/4*N+1/4*(U1+h*k)
            check_nonnegative && _check_problem_state(prob,U2,negative_tolerance)
            rhs!(k,U2,prob.operator,prob.p,t+h/2)
            @. N=1/3*N+2/3*(U2+h*k)
            _check_problem_state(prob,N,check_nonnegative ? negative_tolerance : Inf)
            t=h==target-t ? target : t+h
        end
        push!(states,copy(N))
    end
    PBMSolution(ts,states,prob,steps)
end
