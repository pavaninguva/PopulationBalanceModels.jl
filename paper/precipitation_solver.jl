module PaperPrecipitationSolver
using PopulationBalanceModels, SciMLBase, Sundials, LinearAlgebra

"""Factored low-order transport/depletion preconditioner. Bidiagonal size
transport and tridiagonal axial transport only approximate the Newton matrix.
They change the Krylov solve, not the PBM residual or the physical method.
"""
function preconditioner(pb)
    grid = first(pb.operator.local_operators).system.populations.crystals.grid
    nx,nz = size(population(pb,pb.u0,:crystals))
    domain = pb.operator.domain
    dz = minimum(domain.grid.widths)
    domain.boundary === :danckwerts || error("paper preconditioner requires Danckwerts boundaries")
    advective,dispersive = domain.velocity/dz,domain.dispersion/dz^2
    previous_gamma = Ref(NaN)
    factor = Ref(lu(Tridiagonal(zeros(nz-1),ones(nz),zeros(nz-1))))
    p = pb.p
    function prec(z,r,parameters,t,y,fy,gamma,delta,side)
        state = reshape(y,size(pb.u0))
        right = reshape(r,size(pb.u0))
        answer = reshape(z,size(pb.u0))
        for j in 1:nz
            s = max(state[nx+1,j]/p.ceq-1,0.)
            growth = p.kg*s
            m2 = sum(grid.centers[i]^2*state[i,j] for i in 1:nx)
            solid = 1e12*p.rho*p.kv*sum(grid.centers[i]^3*state[i,j] for i in 1:nx)
            derivative = (p.kp*p.u*s^(p.u-1)+p.kb*p.b*s^(p.b-1)*max(solid,0.))/p.ceq
            depletion = p.rho*p.kv*first(grid.centers)^3*derivative+
                1e12*p.rho*p.kv*3p.kg/p.ceq*max(m2,0.)
            for i in 1:nx
                upstream = i==1 ? 0. : gamma*growth/grid.widths[i-1]*answer[i-1,j]
                outgoing = i==nx ? 0. : growth/grid.widths[i]
                answer[i,j] = (right[i,j]+upstream)/(1+gamma*outgoing)
            end
            answer[nx+1,j] = right[nx+1,j]/(1+gamma*depletion)
        end
        if gamma != previous_gamma[]
            diagonal = fill(1+gamma*(advective+2dispersive),nz)
            diagonal[1] -= gamma*dispersive
            diagonal[end] -= gamma*dispersive
            factor[] = lu(Tridiagonal(fill(-gamma*(advective+dispersive),nz-1),
                diagonal,fill(-gamma*dispersive,nz-1)))
            previous_gamma[] = gamma
        end
        for i in 1:nx+1
            ldiv!(factor[],view(answer,i,:))
        end
        nothing
    end
end

function step_counts(integrator)
    steps,failures = Ref{Clong}(0),Ref{Clong}(0)
    Sundials.CVodeGetNumSteps(integrator.mem,steps)
    Sundials.CVodeGetNumErrTestFails(integrator.mem,failures)
    Int(steps[]),Int(failures[])
end

"""Checked CVODE-BDF/GMRES integration through the package's existing ODE
adapter. Every returned internal step is checked using the native state checker.
Newton iterates are auxiliary states, not accepted physical RK stages. This is
not a positivity-preserving integrator and does not inherit a native guarantee.
An optional prescribed restart changes the maximum BDF order, copying the whole
accepted population/bulk state without projection or operator splitting.
"""
function advance(pb,times;reltol=1e-7,population_abstol=1e-12,bulk_abstol=1e-10,
        dtmax=.5,stability_limit_detect=true,max_order=3,late_max_order=0,restart_time=5.,progress=true)
    max_order in 1:5 || error("CVODE BDF maximum order must be between one and five")
    late_max_order in 0:5 || error("late maximum BDF order must be zero (disabled) or between one and five")
    times = collect(Float64,times)
    issorted(times) && all(diff(times).>0) || error("strictly increasing save times required")
    last(times)==last(pb.tspan) || error("the final save must be the requested horizon")
    restart = late_max_order!=0 && first(pb.tspan)<restart_time<last(pb.tspan)
    restart && !(restart_time in times) && error("the BDF restart must coincide with a saved time")
    tolerance = fill(population_abstol,size(pb.u0))
    tolerance[end,:] .= bulk_abstol
    algorithm(order) = CVODE_BDF(;linear_solver=:GMRES,krylov_dim=50,prec=preconditioner(pb),
        prec_side=1,stability_limit_detect,max_order=order)
    function initialize(problem,order,stops)
        SciMLBase.init(problem,algorithm(order);reltol,abstol=vec(tolerance),
            dt=1e-7,dtmax,tstops=stops,save_everystep=false,save_start=false,maxiters=2_000_000)
    end
    ode = odeproblem(pb)
    integrator = initialize(ode,max_order,times)
    states = typeof(pb.u0)[]
    phase_steps,phase_failures = Int[],Int[]
    phase_orders,phase_observed_orders = Int[],Int[]
    phase_end_times = Float64[]
    observed_order,last_order = 0,Ref{Cint}(0)
    current_maximum_order = max_order
    restarted = false
    checked_steps = 0
    last_report = time_ns()
    minimum_scaled_population = 0.0
    minimum_solute = 0.0
    for target in times
        while integrator.t < target
            previous = integrator.t
            SciMLBase.step!(integrator)
            integrator.t>previous || error("CVODE failed to advance: $(integrator.sol.retcode)")
            PopulationBalanceModels._check_problem_state(pb,integrator.u,1e-8)
            minimum_scaled_population = min(minimum_scaled_population,minimum(population(pb,integrator.u,:crystals)))
            minimum_solute = min(minimum_solute,minimum(bulk(pb,integrator.u,:c)))
            checked_steps += 1
            Sundials.CVodeGetLastOrder(integrator.mem,last_order)
            observed_order = max(observed_order,Int(last_order[]))
            if progress && (time_ns()-last_report)/1e9>30
                println("  advancing t = ",integrator.t," s; ",checked_steps," checked CVODE steps")
                flush(stdout)
                last_report = time_ns()
            end
        end
        isapprox(integrator.t,target;atol=1e-10,rtol=0) || error("missed output time $target")
        push!(states,copy(integrator.u))
        progress && (println("  t = ",target," s; ",checked_steps," checked CVODE steps");flush(stdout))
        if restart && !restarted && target==restart_time
            steps,failures = step_counts(integrator)
            push!(phase_steps,steps);push!(phase_failures,failures)
            push!(phase_orders,current_maximum_order);push!(phase_observed_orders,observed_order)
            push!(phase_end_times,target)
            # Only numerical history is discarded. All physical states and the
            # original RHS/parameters are retained through the standard adapter.
            next_problem = SciMLBase.remake(ode;u0=copy(integrator.u),tspan=(target,last(pb.tspan)))
            integrator = initialize(next_problem,late_max_order,filter(>(target),times))
            isequal(integrator.u,last(states)) || error("BDF restart changed the physical state")
            current_maximum_order = late_max_order
            observed_order = 0
            restarted = true
        end
    end
    steps,failures = step_counts(integrator)
    push!(phase_steps,steps);push!(phase_failures,failures)
    push!(phase_orders,current_maximum_order);push!(phase_observed_orders,observed_order)
    push!(phase_end_times,last(times))
    diagnostics = (;policy=:CVODE_BDF_GMRES,accepted_steps=sum(phase_steps),
        maximum_bdf_order=max_order,late_maximum_bdf_order=late_max_order,restarts=Int(restarted),
        phase_steps,phase_end_times,phase_maximum_orders=phase_orders,
        phase_maximum_observed_orders=phase_observed_orders,
        phase_error_test_failures=phase_failures,error_test_failures=sum(phase_failures),
        checked_steps,minimum_scaled_population,minimum_solute)
    PBMSolution(times,states,pb,sum(phase_steps),diagnostics)
end

export advance
end
