"""Independent spatial check through the closed raw-moment equations.
For size-independent growth, dM₀/dt=J, dMₖ/dt=k*G*Mₖ₋₁ and
dc/dt=-1e-6*rho*kv*dM₃/dt in the scaled count convention.
This compares moments/bulk only; quadrature nodes are not observed PSDs.
"""
function moment_reference_check(chosen)
    p=Case.decode(chosen.u;gated=chosen.settings.gated)
    model=PBMModel(growth=Growth(PolynomialRate(((p,t,ctx)->last(rates(p,t,ctx.bulk.c)),))),
        primary=Nucleation(rate=Case.nucleation,size=0.))
    op=compile(model,MomentSet(0:3;coordinate=:length);method=ExactClosure())
    system=PBMSystem((crystals=op,);nonnegative_bulk=(:c,),exchanges=(
        MomentExchange(:crystals,:c,3;factor=1e-6*p.rho*p.kv),))
    pb=PBMProblem(system,(crystals=zeros(4),),(0.,660.);bulk0=(c=p.c0,),p)
    ts=sort(unique(vcat(0.,[r.model_time_min for r in select_concentration(data)],74.,180.,420.,660.)))
    method=MeasureEuler(reconstruction=RankAdaptiveQMOM(2))
    coarse=simulate(pb;dt=.1,saveat=ts,method)
    fine=simulate(pb;dt=.05,saveat=ts,method)
    sectional=forward(p;bins=chosen.settings.bins,dt=chosen.settings.dt,times=ts)
    grid=sectional.problem.operator.system.populations.crystals.grid
    rows=[]
    for (t,a,b,c) in zip(ts,coarse.u,fine.u,sectional.u)
        for k in 0:3
            push!(rows,(;time_min=t,order=k,
                moment_dt0p1=moment_value(op.closure,op.grid,population(pb,a,:crystals),k),
                moment_dt0p05=moment_value(op.closure,op.grid,population(pb,b,:crystals),k),
                sectional_moment=moment(grid,population(sectional.problem,c,:crystals),k),
                moment_concentration_g_L=bulk(pb,b,:c),
                sectional_concentration_g_L=bulk(sectional.problem,c,:c)))
        end
    end
    csv(joinpath(OUTPUT,"moment_reference.csv"),rows)
    integration=max([abs(bulk(pb,a,:c)-bulk(pb,b,:c)) for (a,b) in zip(coarse.u,fine.u)]...)
    difference=maximum(abs(bulk(pb,b,:c)-bulk(sectional.problem,c,:c)) for (b,c) in zip(fine.u,sectional.u))
    drift=maximum(abs(bulk(pb,u,:c)+1e-6*p.rho*p.kv*moment_value(op.closure,op.grid,population(pb,u,:crystals),3)-p.c0) for u in fine.u)
    (;moment_steps=(coarse.steps,fine.steps),moment_timestep_halving_max_c_change_g_L=integration,
        sectional_vs_moment_max_c_change_g_L=difference,moment_material_error_g_L=drift,
        interpretation="Exactly closed growth/source moments, positive-measure integration; first-order coupled time update. No PSD reconstruction is claimed.")
end
