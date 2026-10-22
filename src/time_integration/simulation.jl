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

struct PBMSolution{T,U,P,D}
    t::T
    u::U
    problem::P
    steps::Int
    integration_diagnostics::D
end
PBMSolution(t,u,p,steps)=PBMSolution(t,u,p,steps,(policy=:unspecified,accepted_steps=steps,))
"""Integration policy, accepted/rejected steps and admissibility diagnostics of a native solution."""
diagnostics(sol::PBMSolution)=sol.integration_diagnostics
abstract type AbstractExplicitRK end
"""One-stage, first-order forward Euler; positivity requires an appropriate step bound."""
struct ForwardEuler <: AbstractExplicitRK end
"""Two-stage, second-order strong-stability-preserving Runge–Kutta (Heun)."""
struct SSPRK22 <: AbstractExplicitRK end
"""Explicit three-stage, third-order strong-stability-preserving Runge–Kutta."""
struct SSPRK33 <: AbstractExplicitRK end
"""Classical four-stage, fourth-order Runge–Kutta; no general SSP guarantee."""
struct RK4 <: AbstractExplicitRK end
"""Five-stage, fourth-order SSP Runge–Kutta of Spiteri and Ruuth; SSP coefficient approximately 1.508. Uses the published nonautonomous stage times."""
struct SSPRK54 <: AbstractExplicitRK end
_default_evolution(op)=SSPRK33()
_simulate(prob,method;kwargs...)=throw(ArgumentError("unsupported evolution method $(typeof(method)) for $(typeof(prob.operator))"))
Base.getindex(sol::PBMSolution,i::Integer)=sol.u[i]

_checkstate(N,negative_tolerance)=_checkstate(N,negative_tolerance,"concentration",
    "reduce dt, use Upwind, or add a proven positivity limiter")
function _checkstate(N,negative_tolerance,description,remedy)
    all(isfinite,N) || throw(NumericalFailure("nonfinite $description during time integration; $remedy"))
    minimum(N)>=-negative_tolerance ||
        throw(NumericalFailure("negative $description during time integration; $remedy"))
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

Explicit fixed-step reference integrator (SSPRK33 by default). Select
ForwardEuler(), SSPRK22(), SSPRK33(), SSPRK54(), or RK4(). dt is a maximum step; a step is
shortened to land on each output time. No clipping, automatic grid change, or
implicit stiffness treatment is performed. Choose dt using `suggest_dt` and
check convergence. `odeproblem` connects compatible residuals to adaptive/stiff
SciML solvers; the native admissibility policy does not transfer automatically.
"""
function simulate(prob::PBMProblem;dt,saveat=collect(prob.tspan),check_nonnegative=true,
                  negative_tolerance=1e-12,maxsteps=10^7,method=_default_evolution(prob.operator))
    method isa AbstractExplicitRK || return _simulate(prob,method;dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    t0=first(prob.tspan)
    N=copy(prob.u0); cache=ntuple(_->similar(N),6)
    states=typeof(N)[]; t=t0; steps=0
    for target in ts
        while t<target
            h=min(dt,target-t)
            t+h>t || throw(NumericalFailure("time step is too small to advance"))
            steps+=1
            steps<=maxsteps || throw(NumericalFailure("maximum step count exceeded"))
            _rk_step!(N,cache,prob,t,h,method,check_nonnegative,negative_tolerance)
            _check_problem_state(prob,N,check_nonnegative ? negative_tolerance : Inf)
            # Necessary Hankel/component checks alone do not establish that the
            # selected closure can evaluate the final RK combination.
            _check_admissible_state(prob.operator,N)
            t=h==target-t ? target : t+h
        end
        push!(states,copy(N))
    end
    PBMSolution(ts,states,prob,steps,(policy=Symbol(nameof(typeof(method))),accepted_steps=steps,rejected_steps=0))
end

function _rk_step!(N,cache,prob,t,h,method::AbstractExplicitRK,check,tol)
    U1,U2,k1,k2,k3,k4=cache
    stage(U)=check && _check_problem_state(prob,U,tol)
    stage(N); rhs!(k1,N,prob.operator,prob.p,t)
    if method isa ForwardEuler
        @. N=N+h*k1
    elseif method isa SSPRK22
        @. U1=N+h*k1
        stage(U1); rhs!(k2,U1,prob.operator,prob.p,t+h)
        @. N=(N+U1+h*k2)/2
    elseif method isa SSPRK33
        @. U1=N+h*k1
        stage(U1); rhs!(k1,U1,prob.operator,prob.p,t+h)
        @. U2=3/4*N+1/4*(U1+h*k1)
        stage(U2); rhs!(k1,U2,prob.operator,prob.p,t+h/2)
        @. N=1/3*N+2/3*(U2+h*k1)
    elseif method isa RK4
        @. U1=N+h/2*k1
        stage(U1); rhs!(k2,U1,prob.operator,prob.p,t+h/2)
        @. U1=N+h/2*k2
        stage(U1); rhs!(k3,U1,prob.operator,prob.p,t+h/2)
        @. U1=N+h*k3
        stage(U1); rhs!(k4,U1,prob.operator,prob.p,t+h)
        @. N=N+h/6*(k1+2k2+2k3+k4)
    elseif method isa SSPRK54
        @. U1=N+0.391752226571890*h*k1
        stage(U1);rhs!(k1,U1,prob.operator,prob.p,t+0.391752226571890*h)
        @. U2=0.444370493651235*N+0.555629506348765*U1+0.368410593050371*h*k1
        stage(U2);rhs!(k1,U2,prob.operator,prob.p,t+0.586079689311540*h)
        @. k2=0.620101851488403*N+0.379898148511597*U2+0.251891774271694*h*k1
        stage(k2);rhs!(k3,k2,prob.operator,prob.p,t+0.474542363121400*h)
        @. U1=0.178079954393132*N+0.821920045606868*k2+0.544974750228521*h*k3
        stage(U1);rhs!(k4,U1,prob.operator,prob.p,t+0.935010630967653*h)
        # Increment form preserves a constant state despite the final published
        # decimal weights summing to 1+1e-15 in floating-point arithmetic.
        @. N=N+0.517231671970585*(U2-N)+0.096059710526147*(k2-N)+0.386708617503269*(U1-N)+h*(0.063692468666290*k3+0.226007483236906*k4)
    else
        throw(ArgumentError("unsupported native explicit integrator $(typeof(method))"))
    end
    nothing
end
