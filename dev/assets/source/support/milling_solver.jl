"""External BDF adapter for the milling example, using the package residual.
The external unknowns are per-pivot material inventories in kg; package states
remain scaled bin counts. Each speed interval is a separate initial-value problem. No stage straddles the
discontinuity, and the saved state at 30 s is continuous; outlet observations at
that time use the new speed. This adapter does not inherit native RK guarantees.
"""
module MillingSolver
using PopulationBalanceModels
import SciMLBase, Sundials

function solve(problem,times;reltol=1e-9,abstol=1e-14)
    times=collect(times)
    all(diff(times).>0) && first(times)>=first(problem.tspan) && last(times)<=last(problem.tspan) ||
        throw(ArgumentError("milling output times must increase inside tspan"))
    u=copy(problem.u0);states=Vector{typeof(u)}();steps=0
    n=length(problem.p.pivots)
    inactive=findall(==(0),problem.p.retained)
    all(==(0),u[inactive]) || throw(ArgumentError("the milling reduction requires zero inventory in instantaneous-discharge classes"))
    active=vcat(findall(>(0),problem.p.retained),n+1:n+2)
    units=vcat(problem.p.rho*problem.p.number_scale.*problem.p.pivots,1.,1.)
    scales=units[active]
    start=first(problem.tspan)
    boundaries=sort!(unique(vcat(start,
        start<problem.p.change_time<last(times) ? [problem.p.change_time] : Float64[],last(times))))
    for interval in 1:length(boundaries)-1
        a,b=boundaries[interval:interval+1]
        rpm=a<problem.p.change_time ? problem.p.speed_before : problem.p.speed_after
        local_problem=remake(problem;tspan=(a,b),p=merge(problem.p,(segment_speed=rpm,)))
        PopulationBalanceModels._check_problem_state(local_problem,u,1e-12)
        base=odeproblem(local_problem;u0=u)
        A,_=parentmodule(@__MODULE__).MillingModel.linear_system(local_problem)
        # Classes with fd=0 satisfy dN/dt=0 exactly and start at zero. Eliminate
        # these invariant zeros, rather than allowing Newton/interpolation
        # roundoff to create signed populations and then clipping them.
        full=zeros(length(u));full_du=similar(full)
        reduced! = function(dv,v,p,t)
            full[active].=v./scales
            base.f(full_du,full,p,t)
            dv.=scales.*view(full_du,active)
            nothing
        end
        # Per-pivot material coordinates remove the huge number/kg Jacobian
        # factors. This is an exact diagonal change of variables, not a new
        # physical balance or a repair of the population.
        jacobian=(A[active,active].*scales)./scales'
        f=SciMLBase.ODEFunction(reduced!;jac=(J,u,p,t)->copyto!(J,jacobian),jac_prototype=copy(jacobian))
        ode=SciMLBase.ODEProblem(f,scales.*base.u0[active],base.tspan,base.p)
        requested=filter(t->(interval==1 ? a<=t : a<t) && t<=b,times)
        # Always retain the segment endpoint for the next segment, even when it
        # is not requested as an observable time.
        save=sort!(unique(vcat(requested,b)))
        integrator=SciMLBase.init(ode,Sundials.CVODE_BDF(linear_solver=:Dense);
            reltol,abstol,saveat=save,tstops=save,save_everystep=false,dense=false,maxiters=10^6)
        # CVODE's inequality constraint rejects a step violating u >= 0; it is
        # not a projection/clipping operation. Landing on every output time also
        # avoids accepting a negative polynomial interpolation between steps.
        constraints=hasproperty(integrator,:ctx_handle) ?
            Sundials.NVector(ones(length(active)),integrator.ctx_handle.ctx) :
            Sundials.NVector(ones(length(active)))
        flag=Sundials.CVodeSetConstraints(integrator.mem,constraints)
        flag==0 || throw(NumericalFailure("CVODE could not install nonnegative milling constraints ($flag)"))
        result=SciMLBase.solve!(integrator)
        SciMLBase.successful_retcode(result) || throw(NumericalFailure("milling BDF failed: $(result.retcode)"))
        for t in requested
            index=findfirst(==(t),result.t)
            index===nothing && throw(NumericalFailure("BDF omitted requested milling output time $t"))
            state=zeros(length(u));state[active]=result.u[index]./scales
            PopulationBalanceModels._check_problem_state(problem,state,1e-12)
            push!(states,state)
        end
        stats=hasproperty(result,:stats) ? result.stats : result.destats
        u=zeros(length(u));u[active]=integrator.u./scales;steps+=stats.naccept
        PopulationBalanceModels._check_problem_state(problem,u,1e-12)
    end
    length(states)==length(times) || throw(DimensionMismatch("milling solver returned an incomplete time grid"))
    PBMSolution(times,states,problem,steps,(;policy=:CVODE_BDF,accepted_steps=steps,
        reltol,abstol,absolute_tolerance_units=:kg,segmented_at=problem.p.change_time,
        eliminated_invariant_zeros=length(inactive)))
end
end
