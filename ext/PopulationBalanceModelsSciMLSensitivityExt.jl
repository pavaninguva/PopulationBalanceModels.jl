module PopulationBalanceModelsSciMLSensitivityExt
using PopulationBalanceModels
import PopulationBalanceModels as PBM
import SciMLBase as SMB
import SciMLSensitivity as SMS
import ReverseDiff
import ForwardDiff

function checked_solve(prob,adapter;kwargs...)
    sol=SMB.solve(prob,adapter.algorithm;adapter.options...,kwargs...)
    SMB.successful_retcode(sol) || throw(PBM.NumericalFailure("sensitivity ODE solve failed: $(sol.retcode)"))
    all(u->all(isfinite,u),sol.u) || throw(PBM.NumericalFailure("nonfinite sensitivity ODE state"))
    sol
end
function PBM._sensitivity_solve(adapter::PBM.SciMLSensitivitySolver,pb::PBM.PBMProblem,times)
    sol=checked_solve(PBM.odeproblem(pb),adapter;saveat=times,save_everystep=false,dense=false)
    stats=hasproperty(sol,:stats) ? sol.stats : sol.destats
    PBM.PBMSolution(sol.t,sol.u,pb,stats.naccept)
end

# Inference coordinates are the ODE parameter vector. Reusing setup here keeps
# shared/local parameters, transformations and operating histories identical to
# the ordinary likelihood evaluation. Grid topology must remain fixed.
function PBM._experiment_gradient(adapter::PBM.SciMLSensitivitySolver,problem,z,ex,counts)
    pb=PBM._experiment_problem(problem,z,ex);ts=PBM._experiment_times(ex)
    PBM._moment_boundary_events(pb.operator) &&
        throw(ArgumentError("moment extinction/escape sensitivities require unimplemented event jump rules"))
    shape=size(pb.u0);nu=length(pb.u0);np=length(z)
    function f!(du,u,q,t)
        current=PBM._experiment_problem(problem,q,ex)
        size(current.u0)==shape || throw(ArgumentError("ODE sensitivities require a fixed state layout"))
        PBM.rhs!(reshape(du,shape),reshape(u,shape),current.operator,current.p,t)
        nothing
    end
    u0=vec(pb.u0)
    initial_jacobian=PBM._differentiate_jacobian(v->vec(PBM._experiment_problem(problem,v,ex).u0),z)
    method=adapter.sensitivity===:auto ? (np<=16 ? :forward : :adjoint) : adapter.sensitivity
    counts===nothing || (counts.solves+=1)
    if method===:forward
        augmented=SMS.ODEForwardSensitivityProblem(f!,u0,pb.tspan,collect(z);sensealg=SMS.ForwardSensitivity())
        # The default is zero initial sensitivities. Include fitted initial data.
        # Preserve the augmented initial state explicitly. The specialised
        # sensitivity-problem remake/solve path can reconstruct zero initial
        # sensitivities; use its generated residual in an ordinary ODEProblem.
        augmented=SMB.ODEProblem(augmented.f,vcat(u0,vec(initial_jacobian)),pb.tspan,collect(z))
        sol=checked_solve(augmented,adapter;saveat=ts,save_everystep=false,dense=false)
        states=hcat((v[1:nu] for v in sol.u)...)
    else
        ode=SMB.ODEProblem(f!,u0,pb.tspan,collect(z))
        # InterpolatingAdjoint needs a dense forward trajectory, not saveat-only
        # interpolation of sparse measurement snapshots.
        sol=checked_solve(ode,adapter;save_everystep=true,dense=true)
        states=hcat((sol(t) for t in ts)...)
    end
    function conditional(q,U)
        current=PBM._experiment_problem(problem,q,ex)
        snapshots=[reshape(view(U,:,j),shape) for j in axes(U,2)]
        PBM._experiment_loglikelihood(problem,q,ex,current,ts,snapshots)
    end
    direct=PBM._differentiate_gradient(q->conditional(q,states),z)
    # Reverse accumulation through the scalar measurement likelihood avoids
    # one forward pass per state coordinate. A fresh tape handles changing
    # observation branches; no branch-dependent tape is cached across solves.
    state_gradient=PBM._ad_evaluate(()->ReverseDiff.gradient(U->conditional(z,U),states))
    if method===:forward
        for (j,v) in enumerate(sol.u)
            S=reshape(view(v,nu+1:length(v)),nu,np)
            direct+=S'*view(state_gradient,:,j)
        end
        return direct
    end
    dg! = (out,u,q,t,j)->(out.=view(state_gradient,:,j);nothing)
    du0,dp=SMS.adjoint_sensitivities(sol,adapter.algorithm;t=ts,dgdu_discrete=dg!,
        sensealg=SMS.InterpolatingAdjoint(autojacvec=false),
        reltol=adapter.options.reltol,abstol=adapter.options.abstol)
    counts===nothing || (counts.adjoint_solves+=1)
    direct+vec(dp)+initial_jacobian'*du0
end
end
