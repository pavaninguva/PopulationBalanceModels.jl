# Specialist scaled Rosenbrock execution; import SciMLBase, OrdinaryDiffEqRosenbrock
# and ADTypes explicitly before including. This file does not run a solve.
"""Scale particle counts and bulk concentrations before stiff integration; return physical states without clipping."""
function stiff_solve(problem,times;reltol=1e-7,max_evaluations=2_000_000)
    max_evaluations isa Integer && max_evaluations>0 || throw(ArgumentError("max_evaluations must be positive"))
    concentration=bulk(problem,problem.u0,:lipid)
    number_scale=concentration/(problem.p.rho*π/6*(100e-9)^3)
    scales=fill(concentration,length(problem.u0))
    scales[problem.operator.ranges.lnp].=number_scale
    evaluations=Ref(0)
    function scaled_rhs!(dv,v,p,t)
        rhs!(dv,v.*scales,problem.operator,p,t)
        dv./=scales
        evaluations[]+=1
        evaluations[]<=max_evaluations || throw(NumericalFailure("LNP RHS evaluation budget exceeded; inspect method, tolerances and state scales"))
        if evaluations[]%100000==0
            println("RHS evaluations=",evaluations[],"; t=",t," s");flush(stdout)
        end
        nothing
    end
    scaled=SciMLBase.ODEProblem(scaled_rhs!,problem.u0./scales,problem.tspan,problem.p)
    println("Starting scaled stiff integration");flush(stdout)
    result=SciMLBase.solve(scaled,Rodas5P(autodiff=AutoFiniteDiff());saveat=times,
        save_everystep=false,reltol,abstol=reltol*1e-3,maxiters=10^7)
    SciMLBase.successful_retcode(result) || error("LNP integration failed: $(result.retcode)")
    println("solver statistics: ",result.destats)
    (;t=result.t,u=[v.*scales for v in result.u],stats=result.destats)
end

