module PopulationBalanceModelsSciMLBaseExt
using PopulationBalanceModels
import SciMLBase

"""
Create a SciML ODEProblem with the identical compiled sectional, tensor or
moment residual. Supply a
compatible promoted u0 yourself for AD. This adapter does not add a positivity
limiter, CFL callback, snapshot tstops, or a sensitivity algorithm.
Direct evolution choices HOCADI/ExactShift belong to simulate and are not
converted into a SciML residual by this adapter.
"""
function PopulationBalanceModels.odeproblem(prob::PopulationBalanceModels.PBMProblem;u0=copy(prob.u0),p=prob.p)
    f! = (du,u,params,t)->PopulationBalanceModels.rhs!(du,u,prob.operator,params,t)
    SciMLBase.ODEProblem(f!,u0,prob.tspan,p)
end
end
