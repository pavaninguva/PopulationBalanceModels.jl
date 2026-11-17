# Specialist interval solver and prescribed calibration starts. Included after
# C0, kinetic functions, parameter_specs and parameters are in scope. Functions
# resolve those declarations at call time. No solve/fit occurs on include.
function stage_concentration(p,ctx)
    c = ctx.bulk.c
    if hasproperty(p,:interval_cmax)
        PopulationBalanceModels._primal(c) <= p.interval_cmax ||
            throw(NumericalFailure("protein interval concentration bound exceeded; reduce coefficient_interval or check upper-size occupancy"))
    end
    c
end

"""Native SSPRK33 with the package's combined growth/diffusion/loss bound.
At most coefficient_interval minutes are advanced before recomputing the bound.
Monotone rates are bounded by the interval's starting c plus roundoff allowance;
all rate callbacks reject a violation at ANY RK stage. Power-of-two upward rate
buckets keep the schedule locally constant for ForwardDiff. Bounds are not
physical rate floors and no derivative across schedule changes is claimed.
"""
function cfl_solve(pb,times;maximum_dt=2.,cfl=.35,coefficient_interval=5.,maxsteps=200_000)
    maximum_dt>0 && isfinite(maximum_dt) && 0<cfl<=.35 && coefficient_interval>0 ||
        throw(ArgumentError("require positive step/interval and 0 < CFL ≤ 0.35"))
    p=pb.p
    p.AG>0 && p.g>0 && p.gamma>0 && p.grd_strength>=0 || throw(NumericalFailure("invalid CNT/GRD parameters"))
    hasproperty(p,:k_D) && !(isfinite(p.k_D) && p.k_D>=0) &&
        throw(NumericalFailure("invalid growth-dependent GRD coefficient k_D"))
    ts=collect(times)
    !isempty(ts) && all(isfinite,ts) && issorted(ts) &&
        first(pb.tspan)<=first(ts)<=last(ts)<=last(pb.tspan) ||
        throw(ArgumentError("save times must be sorted within the problem interval"))
    u=copy(pb.u0);states=typeof(u)[];t=first(pb.tspan);steps=0
    for target in ts
        while target>t
            endpoint=min(target,t+coefficient_interval)
            cmax=PopulationBalanceModels._primal(bulk(pb,u,:c))+128eps(Float64)*C0
            # The native recommendation evaluates the assembled growth+diffusion
            # operator and actual mesh geometry. Each component's monotone rate
            # is bounded at cmax, and stage_concentration enforces that bound.
            boundstate=copy(u);boundstate[pb.operator.bulk_indices.c]=cmax
            allowed=suggest_dt(pb.operator,boundstate,p,t;cfl=1.)
            rate=isfinite(allowed) ? PopulationBalanceModels._primal(inv(allowed)) : 0.
            bucket=rate>0 ? ldexp(1.,ceil(Int,log2(rate))) : 0.
            dt=bucket>0 ? min(maximum_dt,cfl/bucket) : maximum_dt
            steps+ceil((endpoint-t)/dt)<=maxsteps ||
                throw(NumericalFailure("trial exceeds $maxsteps explicit steps; inspect rate/dispersion scales or use a separately verified solver"))
            interval=remake(pb;u0=u,tspan=(t,endpoint),p=merge(p,(interval_cmax=cmax,)))
            sol=simulate(interval;dt,saveat=[endpoint],method=SSPRK33())
            u=last(sol.u);steps+=sol.steps;t=endpoint
        end
        push!(states,copy(u))
    end
    PBMSolution(ts,states,pb,steps)
end


"""Three calibration-only starts, identical common priors and optimizer budget."""
function fit_model(problem;maxiters=500,initial=nothing)
    model=any(s->s.name===:k_D,problem.parameters) ? :cnt_growth_grd :
        any(s->s.name===:grd_strength,problem.parameters) ? :cnt_grd : :cnt
    S0=C0/problem.fixed.cstar
    starts=map(((1.5,.001),(2.5,.01),(4.,.05))) do (g,D0)
        values=(logAJ=log(1e8)+barrier(FIXED_GAMMA)/log(S0)^2,
            AG=.15/(S0-1)^g,g=g,grd_strength=model===:cnt ? 0. : D0)
        model===:cnt_growth_grd && (values=merge(values,(grd_strength=0.,k_D=10D0,)))
        specs=parameter_specs(;p=parameters(;values...),model)
        initial_coordinates(EstimationProblem(specs,problem.experiments;fixed=problem.fixed))
    end
    starts=collect(starts)
    initial===nothing || (starts[1]=collect(initial))
    fit(problem;starts,maxiters,gradient_tolerance=1e-3)
end

