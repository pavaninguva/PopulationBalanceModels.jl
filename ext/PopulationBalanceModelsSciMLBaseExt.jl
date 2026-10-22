module PopulationBalanceModelsSciMLBaseExt
using PopulationBalanceModels
import SciMLBase

"""
Create a SciML ODEProblem with the identical compiled sectional, tensor, axial or
moment residual. Supply a
compatible promoted u0 yourself for AD. This adapter does not add a positivity
limiter, CFL callback, snapshot tstops, or a sensitivity algorithm.
Direct evolution choices HOCADI/ExactShift belong to simulate and are not
converted into a SciML residual by this adapter.
"""
function PopulationBalanceModels.odeproblem(prob::PopulationBalanceModels.PBMProblem;u0=copy(prob.u0),p=prob.p)
    PopulationBalanceModels._moment_boundary_events(prob.operator) &&
        throw(ArgumentError("moment boundary events require native MeasureEuler; no continuum trace or external-solver event adapter is implemented"))
    # SciML's symbolic parameter-remake path can recurse for an empty NamedTuple.
    # Use its parameterless sentinel, while preserving the PBM callback contract.
    solver_parameters=p isa NamedTuple && isempty(p) ? SciMLBase.NullParameters() : p
    f! = (du,u,params,t)->PopulationBalanceModels.rhs!(du,u,prob.operator,
        params isa SciMLBase.NullParameters ? p : params,t)
    SciMLBase.ODEProblem(f!,u0,prob.tspan,solver_parameters)
end
end
