module PaperPrecipitationSolver
using PopulationBalanceModels, SciMLBase, Sundials, LinearAlgebra, ForwardDiff

struct _GrowthJacobianColor end

"""Case-7A Newton preconditioning, without changing the physical residual.

The native growth Jacobian has lower/upper bandwidths 3/2. Six-color
ForwardDiff differentiates that same term on its compiled nonuniform mesh.
A banded LU plus a rank-one correction includes nucleation and conservative
solute coupling: delta(c + q'N) = r_c + q'r_N. A separate low-order axial
factor approximates reactor transport. It is a linear-solve aid, not operator
splitting of the PBM. Jacobian data are lagged only when Sundials permits it;
all factors and rank-one terms consistently use their cached gamma.
"""
function preconditioner(pb)
    popop=first(pb.operator.local_operators).system.populations.crystals
    grid=popop.grid;nx,nz=size(population(pb,pb.u0,:crystals));p=pb.p
    pb.operator.domain.boundary===:danckwerts || error("The paper preconditioner requires Danckwerts reactor boundaries")
    p.kg>=0 || error("The paper growth preconditioner requires nonnegative growth")
    item=only(filter(x->x.term isa Growth,popop.prepared))
    lower,upper,colors=3,2,6
    # One row touches at most six adjacent populations. Distinct colors on
    # those columns recover the exact stencil derivatives without a dense J.
    D=ForwardDiff.Dual{_GrowthJacobianColor,Float64,6}
    duals=Vector{D}(undef,nx);dduals=similar(duals)
    seeds=[ForwardDiff.Partials(ntuple(k->Float64(k==mod1(i,colors)),6)) for i in 1:nx]
    jac=zeros(6,nx,nz);factors=zeros(9,nx,nz)
    pivots=[zeros(LinearAlgebra.BlasInt,nx) for _ in 1:nz]
    h=zeros(nx,nz);correction=zeros(nx,nz);denominators=ones(nz)
    q=1e12p.rho*p.kv.*grid.centers.^3;alpha=zeros(nz)
    has_growth=falses(nz);zero_local=trues(nz)
    ready=Ref(false);lastgamma=Ref(NaN);setups=Ref(0);updates=Ref(0)
    dz=first(pb.operator.domain.grid.widths)
    all(w->isapprox(w,dz;rtol=1e-10,atol=0),pb.operator.domain.grid.widths) ||
        error("The paper axial factor requires a uniform reactor mesh")
    adv=pb.operator.domain.velocity/dz;diffusion=pb.operator.domain.dispersion/dz^2
    axial=Ref(lu(Tridiagonal(zeros(nz-1),ones(nz),zeros(nz-1))))
    function setup(parameters,t,y,fy,jok,jcur,gamma)
        setups[]+=1
        refresh=!ready[] || !jok
        if refresh
            updates[]+=1;state=reshape(y,size(pb.u0))
            for j in 1:nz
                c=state[nx+1,j];s=max(c/p.ceq-1,0.);g=p.kg*s
                gc=s>0 ? p.kg/p.ceq : 0.
                fill!(view(jac,:,:,j),0.)
                fill!(view(h,:,j),0.)
                has_growth[j]=g>0
                # Exact zero kinetics needs no differentiation or local LU.
                # This is not a small-rate threshold or a change to the RHS.
                if has_growth[j]
                    for i in 1:nx
                        duals[i]=D(state[i,j],seeds[i])
                    end
                    fill!(dduals,zero(D))
                    ctx=(;grid,N=duals,bulk=(c=c,),fields=(z=pb.operator.domain.grid.centers[j],))
                    PopulationBalanceModels.add_term!(dduals,duals,item.term,item.method,item.cache,popop,p,t,ctx)
                    for i in 1:nx
                        h[i,j]=ForwardDiff.value(dduals[i])*(gc/g)
                        for col in max(1,i-lower):min(nx,i+upper)
                            jac[upper+1+i-col,col,j]=ForwardDiff.partials(dduals[i])[mod1(col,colors)]
                        end
                    end
                end
                solid=sum(q[i]*state[i,j] for i in 1:nx)
                h[1,j]+=(p.kp*p.u*s^(p.u-1)+p.kb*p.b*s^(p.b-1)*solid)/(1e12p.ceq)
                alpha[j]=p.kb*s^p.b/1e12
                zero_local[j]=!has_growth[j] && iszero(alpha[j]) && all(iszero,view(h,:,j))
            end
        end
        if refresh || gamma!=lastgamma[]
            for j in 1:nz
                if zero_local[j]
                    fill!(view(correction,:,j),0.)
                    denominators[j]=1.
                    continue
                end
                factor=view(factors,:,:,j);fill!(factor,0.)
                for col in 1:nx,k in 1:colors
                    # LAPACK band LU reserves the first `lower` rows for fill.
                    factor[lower+k,col]=-gamma*jac[k,col,j]
                end
                for i in 1:nx
                    factor[lower+upper+1,i]+=1
                end
                if has_growth[j]
                    _,piv=LinearAlgebra.LAPACK.gbtrf!(lower,upper,nx,factor)
                    copyto!(pivots[j],piv)
                end
                z=view(correction,:,j)
                for i in 1:nx;z[i]=gamma*(h[i,j]-(i==1 ? alpha[j] : 0.));end
                has_growth[j] && LinearAlgebra.LAPACK.gbtrs!('N',lower,upper,nx,factor,pivots[j],z)
                denominators[j]=1+sum(q[i]*z[i] for i in 1:nx)
                abs(denominators[j])>100eps(Float64)*(1+sum(abs(q[i]*z[i]) for i in 1:nx)) || error("Singular rank-one correction")
            end
            diagonal=fill(1+gamma*(adv+2diffusion),nz)
            diagonal[1]-=gamma*diffusion;diagonal[end]-=gamma*diffusion
            axial[]=lu(Tridiagonal(fill(-gamma*(adv+diffusion),nz-1),diagonal,fill(-gamma*diffusion,nz-1)))
            lastgamma[]=gamma
        end
        ready[]=true;jcur[1]=Int(refresh)
        nothing
    end
    function solve(z,r,parameters,t,y,fy,gamma,delta,side)
        ready[] || error("Setup required")
        answer=reshape(z,size(pb.u0));right=reshape(r,size(pb.u0))
        for j in 1:nz
            if zero_local[j]
                copyto!(view(answer,:,j),view(right,:,j))
                continue
            end
            material=right[nx+1,j]+sum(q[i]*right[i,j] for i in 1:nx)
            a=view(answer,1:nx,j)
            # Sundials may reuse a lagged preconditioner between setups. Use
            # its cached gamma coherently in the RHS, factors and correction.
            for i in 1:nx;a[i]=right[i,j]+lastgamma[]*h[i,j]*material;end
            has_growth[j] && LinearAlgebra.LAPACK.gbtrs!('N',lower,upper,nx,view(factors,:,:,j),pivots[j],a)
            coeff=sum(q[i]*a[i] for i in 1:nx)/denominators[j]
            for i in 1:nx;a[i]-=correction[i,j]*coeff;end
            answer[nx+1,j]=right[nx+1,j]+sum(q[i]*(right[i,j]-answer[i,j]) for i in 1:nx)
        end
        for i in 1:nx+1;ldiv!(axial[],view(answer,i,:));end
        nothing
    end
    (;setup,solve,setups,updates)
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
    preconditioners = Any[]
    function algorithm(order)
        hooks = preconditioner(pb)
        push!(preconditioners,hooks)
        CVODE_BDF(;linear_solver=:GMRES,krylov_dim=50,prec=hooks.solve,psetup=hooks.setup,
            prec_side=1,stability_limit_detect,max_order=order)
    end
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
        preconditioner=:native_growth_banded,
        preconditioner_setups=sum(hooks.setups[] for hooks in preconditioners),
        growth_jacobian_updates=sum(hooks.updates[] for hooks in preconditioners),
        maximum_bdf_order=max_order,late_maximum_bdf_order=late_max_order,restarts=Int(restarted),
        phase_steps,phase_end_times,phase_maximum_orders=phase_orders,
        phase_maximum_observed_orders=phase_observed_orders,
        phase_error_test_failures=phase_failures,error_test_failures=sum(phase_failures),
        checked_steps,minimum_scaled_population,minimum_solute)
    PBMSolution(times,states,pb,sum(phase_steps),diagnostics)
end

export advance
end
