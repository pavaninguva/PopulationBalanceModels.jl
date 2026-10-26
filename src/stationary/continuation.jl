"""Accepted stationary solutions along a parameter path. `converged` means the
requested endpoint was reached. `attempts` includes rejected corrections and
their errors; partial branches are retained without claiming target completion.
"""
struct StationaryBranch{L,S,A}
    parameters::L
    solutions::S
    attempts::A
    converged::Bool
end

"""
    continue_stationary(problem, parameters, (λ0,λ1); algorithm=nothing,
                        initial_step=(λ1-λ0)/10, min_step=(λ1-λ0)*1e-6,
                        max_step=(λ1-λ0)/4, maxsteps=1000, solve_kwargs...)

Natural-parameter continuation. `parameters(λ)` returns the complete physical
parameter object for that solve. Solve the anchor, then warm-start from each
accepted state. Failed corrections halve the increment; successful corrections
increase it by 1.5 up to max_step. Each correction uses solve_stationary and
therefore the identical assembled physics, constraints and balance checks.

This is parameter continuation, not time integration or pseudo-arclength:
requires increasing finite λspan and cannot follow a fold that reverses λ.
No physical admissibility or root-existence guarantee is implied. Every callback
must be pure, including parameters(λ). The last accepted state is preserved on
failure. Mesh continuation instead requires recompilation and conservative state
transfer, and is not performed by this helper.
"""
function continue_stationary(prob::StationaryPBMProblem,parameters,span;
        algorithm=nothing,initial_step=(span[2]-span[1])/10,
        min_step=(span[2]-span[1])*1e-6,max_step=(span[2]-span[1])/4,
        maxsteps=1000,kwargs...)
    length(span)==2 && all(isfinite,span) && span[2]>span[1] || throw(ArgumentError("continuation requires a finite increasing span"))
    all(x->isfinite(x) && x>0,(initial_step,min_step,max_step)) && min_step<=initial_step<=max_step ||
        throw(ArgumentError("continuation requires 0 < min_step <= initial_step <= max_step"))
    maxsteps isa Integer && maxsteps>0 || throw(ArgumentError("maxsteps must be a positive integer"))
    a,b=promote(float(span[1]),float(span[2])); h=float(initial_step)
    path=typeof(a)[]; solutions=StationaryPBMSolution[]; attempts=NamedTuple[]
    current=a; guess=prob.guess
    for iteration in 1:maxsteps
        target=isempty(solutions) ? a : min(b,current+h)
        if !isempty(solutions) && target<=current
            break
        end
        sol=nothing; message=""
        try
            trial=remake(prob;p=parameters(target),guess)
            sol=solve_stationary(trial,algorithm;kwargs...)
        catch err
            err isa Union{NumericalFailure,DomainError} || rethrow()
            message=sprint(showerror,err)
        end
        accepted=sol!==nothing && sol.converged
        push!(attempts,(parameter=target,accepted=accepted,
            residual=sol===nothing ? Inf : diagnostics(sol).scaled_residual_norm,
            retcode=sol===nothing ? :InvalidEvaluation : Symbol(string(sol.retcode)),error=message))
        if accepted
            push!(path,target); push!(solutions,sol)
            target==b && return StationaryBranch(path,solutions,attempts,true)
            current=target; guess=copy(sol.u)
            h=min(max_step,h*1.5)
        else
            isempty(solutions) && break
            h/=2
            h>=min_step || break
        end
    end
    StationaryBranch(path,solutions,attempts,false)
end
