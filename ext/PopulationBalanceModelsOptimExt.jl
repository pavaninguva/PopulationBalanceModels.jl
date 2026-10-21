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

function PopulationBalanceModels.profile_likelihood(cal::PopulationBalanceModels.CalibrationProblem,
        fitted::PopulationBalanceModels.FitResult,index::Integer,values;
        starts=(fitted.minimizer,),kwargs...)
    fitted.converged || throw(ArgumentError("profile reference fit did not converge"))
    center=copy(fitted.minimizer); count=length(center)
    1<=index<=count || throw(ArgumentError("invalid profiled coordinate"))
    fixed=float.(collect(values))
    !isempty(fixed) && all(isfinite,fixed) && all(diff(fixed).>0) ||
        throw(ArgumentError("profile values must be finite and strictly increasing"))
    seeds=collect(starts)
    !isempty(seeds) && all(s->length(s)==count && all(isfinite,s),seeds) ||
        throw(ArgumentError("supply full parameter vectors as profile starts"))
    nuisance=[i for i in 1:count if i!=index]
    base=PopulationBalanceModels.objective(cal,center)
    losses=Float64[]; converged=Bool[]; minimizers=Vector{Float64}[]
    for value in fixed
        function expand(v)
            full=copy(center); full[index]=value; full[nuisance]=v; full
        end
        if isempty(nuisance)
            full=expand(Float64[])
            loss=PopulationBalanceModels.objective(cal,full;on_failure=:infinity)
            push!(losses,loss);push!(converged,isfinite(loss));push!(minimizers,full)
        else
            conditional=PopulationBalanceModels.CalibrationProblem(cal.experiments;
                parameters=(v,ex)->cal.parameters(expand(v),ex),dt=cal.dt,solver=cal.solver)
            result=try
                PopulationBalanceModels.fit_multistart(conditional,[s[nuisance] for s in seeds];kwargs...).best
            catch e
                e isa PopulationBalanceModels.NumericalFailure || rethrow()
                nothing
            end
            push!(losses,result===nothing ? Inf : result.minimum)
            push!(converged,result!==nothing)
            push!(minimizers,result===nothing ? fill(NaN,count) : expand(result.minimizer))
        end
    end
    any(l->l<base-1e-6*max(1,abs(base)),losses) &&
        throw(ArgumentError("profile found a better fit than its reference; refit from that solution"))
    (;values=fixed,objectives=losses,twice_delta_nll=max.(0.0,2 .* (losses.-base)),
      converged,minimizers,reference=center)
end
end
