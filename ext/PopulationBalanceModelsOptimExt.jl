module PopulationBalanceModelsOptimExt
using PopulationBalanceModels
import Optim

"""
Finite-difference optimization of the declared known-variance Gaussian
likelihood. Choose parameter transforms in CalibrationProblem; no positivity
constraints, multistart, uncertainty intervals or convergence guarantees are
added implicitly. Rerun at multiple starts and refined dt/grid.
"""
function PopulationBalanceModels.fit(cal::PopulationBalanceModels.CalibrationProblem,z0::AbstractVector;
                 algorithm=Optim.BFGS(initial_stepnorm=0.5),maxiters=500,options=NamedTuple())
    failures=Ref(0)
    messages=String[]
    function loss(z)
        try
            PopulationBalanceModels.objective(cal,z)
        catch e
            if e isa PopulationBalanceModels.NumericalFailure
                failures[]+=1
                message=sprint(showerror,e)
                length(messages)<10 && !(message in messages) && push!(messages,message)
                return Inf
            end
            rethrow()
        end
    end
    isfinite(loss(z0)) || throw(PopulationBalanceModels.NumericalFailure("initial parameters do not give a valid forward solve; check the initial guess, grid and dt"))
    result=Optim.optimize(loss,float.(copy(z0)),algorithm,
        Optim.Options(;iterations=maxiters,options...))
    PopulationBalanceModels.FitResult(Optim.minimizer(result),Optim.minimum(result),Optim.converged(result),result,
        failures[],messages)
end
end
