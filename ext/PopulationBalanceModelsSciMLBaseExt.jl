module PopulationBalanceModelsSciMLBaseExt
using PopulationBalanceModels
import SciMLBase

"""
Create a SciML ODEProblem with the identical compiled sectional, tensor, axial or
moment residual. Supply a
compatible promoted u0 yourself for AD. This adapter does not add a positivity
limiter, CFL callback, snapshot tstops, or a sensitivity algorithm.
Direct evolution choices, including HOCFiniteVolume/HOCADI/ExactShift, belong to simulate and are not
converted into a SciML residual by this adapter.
"""
function PopulationBalanceModels.odeproblem(prob::PopulationBalanceModels.PBMProblem;u0=copy(prob.u0),p=prob.p)
    PopulationBalanceModels._unsegmented_history(prob) && throw(ArgumentError("odeproblem cannot represent OperatingHistory by one residual; use simulate with a configured external ForwardSolver for segmented execution"))
    prob.operator isa Union{PopulationBalanceModels.CompiledHOC,
        PopulationBalanceModels.CompiledCharacteristic,PopulationBalanceModels.CompiledConstantTransport} &&
        throw(ArgumentError("compiled direct transport has no ODE residual; use native simulate with its compiled evolution policy"))
    PopulationBalanceModels._moment_boundary_events(prob.operator) &&
        throw(ArgumentError("moment boundary events require native MeasureEuler; no continuum trace or external-solver event adapter is implemented"))
    # SciML's symbolic parameter-remake path can recurse for an empty NamedTuple.
    # Use its parameterless sentinel, while preserving the PBM callback contract.
    solver_parameters=p isa NamedTuple && isempty(p) ? SciMLBase.NullParameters() : p
    operator=PopulationBalanceModels.prepare_run(prob.operator,p)
    f! = (du,u,params,t)->PopulationBalanceModels.rhs!(du,u,operator,
        params isa SciMLBase.NullParameters ? p : params,t)
    SciMLBase.ODEProblem(f!,u0,prob.tspan,solver_parameters)
end

function PopulationBalanceModels._external_forward(pb::PopulationBalanceModels.PBMProblem,s::PopulationBalanceModels.ForwardSolver,ts)
    PBM=PopulationBalanceModels
    if last(ts)==first(pb.tspan)
        return PBM.PBMSolution(ts,[copy(pb.u0)],pb,0,
            (policy=:external,accepted_steps=0,rejected_steps=0,stopping_reason=:initial_output))
    end
    ode=PBM.odeproblem(PBM.PBMProblem(pb.operator,pb.u0,(first(pb.tspan),last(ts)),pb.p))
    step_options=s.dt===nothing ? (;) : (;dtmax=s.dt)
    haskey(s.options,:dtmax) && s.dt!==nothing && throw(ArgumentError("use ForwardSolver dt or external dtmax, not both"))
    result=SciMLBase.solve(ode,s.algorithm;saveat=ts,save_start=first(ts)==first(pb.tspan),save_end=true,
        save_everystep=false,dense=false,maxiters=s.maxsteps,step_options...,s.options...)
    SciMLBase.successful_retcode(result) || throw(PBM.NumericalFailure("external forward solve failed: $(result.retcode)"))
    length(result.t)==length(ts) && all(isapprox.(result.t,ts;rtol=1e-13,atol=1e-14)) ||
        throw(PBM.NumericalFailure("external solve stopped early or omitted requested outputs"))
    stats=hasproperty(result,:stats) ? result.stats : result.destats
    accepted=hasproperty(stats,:naccept) ? stats.naccept : 0
    rejected=hasproperty(stats,:nreject) ? stats.nreject : 0
    info=(policy=:external,algorithm=string(typeof(s.algorithm)),accepted_steps=accepted,
        rejected_steps=rejected,retcode=string(result.retcode),validation=:saved_states,
        stopping_reason=:requested_outputs_complete,options=s.options)
    PBM._checked_solution(pb,ts,[copy(u) for u in result.u],accepted,info,s,ts)
end
end
