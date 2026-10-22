module PopulationBalanceModelsOptimExt
using PopulationBalanceModels
using LinearAlgebra, Random
import Optim

"""
Optimization of the declared known-variance Gaussian likelihood using the
CalibrationProblem's selected derivative backend (forward AD by default).
Choose parameter transforms in CalibrationProblem; no positivity
constraints, multistart, uncertainty intervals or convergence guarantees are
added implicitly. Rerun at multiple starts and refined dt/grid.
"""
function PopulationBalanceModels.fit(cal::PopulationBalanceModels.CalibrationProblem,z0::AbstractVector;
                 algorithm=Optim.BFGS(initial_stepnorm=0.5),maxiters=500,options=NamedTuple(),derivatives=cal.derivatives)
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
    gradient! = (g,z)->copyto!(g,PopulationBalanceModels._differentiate_gradient(loss,z;derivatives))
    result=Optim.optimize(loss,gradient!,float.(copy(z0)),algorithm,
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
            full=eltype(v).(center); full[index]=value; full[nuisance]=v; full
        end
        if isempty(nuisance)
            full=expand(Float64[])
            loss=PopulationBalanceModels.objective(cal,full;on_failure=:infinity)
            push!(losses,loss);push!(converged,isfinite(loss));push!(minimizers,full)
        else
            conditional=PopulationBalanceModels.CalibrationProblem(cal.experiments;
                parameters=(v,ex)->cal.parameters(expand(v),ex),dt=cal.dt,solver=cal.solver,derivatives=cal.derivatives)
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

"""
MAP fit of the shared EstimationProblem posterior. Starts are inference-coordinate
vectors; defaults come from physical ParameterSpec initial values. Additional
starts use a seeded Gaussian perturbation. Scaling changes optimizer coordinates
only and does not change the declared posterior density. Laplace is opt-in.
"""
function PopulationBalanceModels.fit(problem::EstimationProblem;
        method=:map,initial=initial_coordinates(problem),starts=nothing,nstarts=1,
        seed=2022,start_scale=0.3,scaling=ones(length(initial)),maxiters=500,
        algorithm=Optim.BFGS(linesearch=Optim.LineSearches.BackTracking()),gradient_tolerance=1e-3,
        derivative_step=cbrt(eps(Float64)),uncertainty=:none,derivatives=problem.derivatives)
    method===:map || throw(ArgumentError("fit defaults to MAP; request sampling separately with sample_posterior"))
    uncertainty in (:none,:laplace) || throw(ArgumentError("uncertainty must be :none or :laplace"))
    n=length(problem.parameters)
    isfinite(gradient_tolerance) && gradient_tolerance>0 || throw(ArgumentError("gradient_tolerance must be positive and finite"))
    length(scaling)==n && all(x->isfinite(x) && x>0,scaling) || throw(ArgumentError("one positive optimizer scale per coordinate required"))
    nstarts isa Integer && nstarts>0 && isfinite(start_scale) && start_scale>=0 || throw(ArgumentError("invalid multistart settings"))
    seeds=starts===nothing ? [Float64.(initial)] : [Float64.(s) for s in starts]
    !isempty(seeds) && all(s->length(s)==n && all(isfinite,s),seeds) || throw(ArgumentError("invalid initial coordinate vectors"))
    rng=MersenneTwister(seed)
    while length(seeds)<nstarts
        push!(seeds,seeds[1].+start_scale.*scaling.*randn(rng,n))
    end
    counts=PopulationBalanceModels._InferenceCounts();history=NamedTuple[];runs=Any[]
    started=time_ns()
    function loss(w)
        z=w.*scaling
        value=-logposterior(problem,z;on_failure=:reject,_counts=counts)
        length(history)<10000 && push!(history,(evaluation=counts.evaluations,objective=value,z=copy(z)))
        value
    end
    function gradient!(g,w)
        g.=-posterior_gradient(problem,w.*scaling;derivatives,step=derivative_step,_counts=counts).*scaling
    end
    for start in seeds
        check=try
            isfinite(loss(start./scaling)) || throw(NumericalFailure("initial posterior is not finite"))
            checked=check_gradient(problem,start;derivatives,step=derivative_step,_counts=counts)
            checked.valid || throw(ArgumentError("posterior gradient check failed at a start"))
            checked
        catch e
            e isa NumericalFailure || rethrow()
            PopulationBalanceModels._record_failure!(counts,e)
            push!(runs,(start=start,converged=false,result=nothing,message=sprint(showerror,e)))
            continue
        end
        raw=try
            Optim.optimize(loss,gradient!,start./scaling,algorithm,
                Optim.Options(iterations=maxiters,g_abstol=min(1e-7,gradient_tolerance)))
        catch e
            e isa NumericalFailure || rethrow()
            PopulationBalanceModels._record_failure!(counts,e)
            nothing
        end
        if raw===nothing
            push!(runs,(start=start,converged=false,result=nothing,message="failed derivative or forward solve"))
        else
            final_gradient=posterior_gradient(problem,Optim.minimizer(raw).*scaling;
                derivatives,step=derivative_step,_counts=counts).*scaling
            # Optim may stop because the objective rounded to the same value.
            # Filter starts using the same stationarity criterion as MAPResult
            # before selecting the best converged candidate.
            stationary=norm(final_gradient,Inf)<=gradient_tolerance
            push!(runs,(start=start,converged=Optim.converged(raw) && stationary,
                result=raw,gradient_check=check,scaled_gradient_norm=norm(final_gradient,Inf)))
        end
    end
    valid=[r for r in runs if r.result!==nothing && isfinite(Optim.minimum(r.result))]
    isempty(valid) && throw(NumericalFailure("no start produced a finite MAP result; $(join(counts.messages,"; "))"))
    converged=[r for r in valid if r.converged]
    candidates=isempty(converged) ? valid : converged
    best=candidates[argmin([Optim.minimum(r.result) for r in candidates])]
    z=Float64.(Optim.minimizer(best.result).*scaling)
    components=posterior_components(problem,z;_counts=counts)
    gradient=posterior_gradient(problem,z;derivatives,step=derivative_step,_counts=counts)
    scaled_gradient=gradient.*scaling
    diagnostics=(;gradient,gradient_norm=norm(gradient,Inf),scaled_gradient,
        scaled_gradient_norm=norm(scaled_gradient,Inf),gradient_tolerance,history,
        history_truncated=length(history)==10000,starts=runs,seed,scaling,derivative_step,
        objective_evaluations=counts.evaluations,forward_solves=counts.solves,
        gradient_evaluations=counts.gradient_evaluations,adjoint_solves=counts.adjoint_solves,
        derivative_backend=PopulationBalanceModels._gradient_backend(problem,derivatives),
        failed_evaluations=counts.failures,failure_messages=counts.messages,
        elapsed_seconds=(time_ns()-started)/1e9,optimizer=best.result)
    result=MAPResult(z,physical_parameters(problem,z),components.loglikelihood,
        components.logprior,components.logposterior,best.converged && norm(scaled_gradient,Inf)<=gradient_tolerance,
        diagnostics,ParameterSpec[problem.parameters...],nothing)
    if uncertainty===:laplace
        approximation=laplace_approximation(problem,result;derivatives)
        return MAPResult(result.minimizer,result.parameters,result.loglikelihood,result.logprior,
            result.logposterior,result.converged,result.diagnostics,result.coordinates,approximation)
    end
    result
end
PopulationBalanceModels.fit(p::EstimationProblem,z::AbstractVector;kwargs...)=fit(p;initial=z,kwargs...)
PopulationBalanceModels.fit_multistart(p::EstimationProblem,starts;kwargs...)=let r=fit(p;starts,kwargs...)
    (;best=r,results=r.diagnostics.starts)
end

function PopulationBalanceModels.estimation_profile(problem::EstimationProblem,result::MAPResult,index::Integer,values;
        target,maxiters=400,starts=(result.minimizer,),derivatives=problem.derivatives,gradient_tolerance=1e-3)
    target in (:likelihood,:posterior) || throw(ArgumentError("target must explicitly be :likelihood or :posterior"))
    result.converged || throw(ArgumentError("reference MAP did not converge"))
    isfinite(gradient_tolerance) && gradient_tolerance>0 || throw(ArgumentError("profile gradient tolerance must be positive"))
    n=length(result.minimizer)
    1<=index<=n || throw(ArgumentError("invalid profile coordinate"))
    grid=Float64.(collect(values))
    !isempty(grid) && all(isfinite,grid) && all(diff(grid).>0) || throw(ArgumentError("profile grid must increase"))
    seeds=[copy(s) for s in starts]
    !isempty(seeds) && all(s->length(s)==n && all(isfinite,s),seeds) || throw(ArgumentError("invalid profile starts"))
    f=z->target===:likelihood ? -loglikelihood(problem,z;on_failure=:reject) : -logposterior(problem,z;on_failure=:reject)
    # Likelihood profiles require their own unpenalized optimum, not the MAP baseline.
    full_gradient=z->target===:likelihood ? -likelihood_gradient(problem,z;derivatives) : -posterior_gradient(problem,z;derivatives)
    function optimize_profile(loss,start,gradient)
        isfinite(loss(start)) || return nothing
        grad! = (g,z)->copyto!(g,gradient(z))
        try
            Optim.optimize(loss,grad!,start,Optim.BFGS(linesearch=Optim.LineSearches.BackTracking()),Optim.Options(iterations=maxiters))
        catch e
            e isa NumericalFailure || rethrow()
            nothing
        end
    end
    function stationary(raw,gradient)
        raw===nothing && return false
        Optim.converged(raw) && isfinite(Optim.minimum(raw)) || return false
        try
            norm(gradient(Optim.minimizer(raw)),Inf)<=gradient_tolerance
        catch e
            e isa NumericalFailure || rethrow()
            false
        end
    end
    reference=optimize_profile(f,result.minimizer,full_gradient)
    reference===nothing && throw(NumericalFailure("profile reference has an invalid solve/derivative"))
    stationary(reference,full_gradient) || throw(NumericalFailure("profile reference optimization failed its stationarity check"))
    reference_z=Optim.minimizer(reference);minimum_value=Optim.minimum(reference)
    nuisance=[i for i in 1:n if i!=index]
    minimizers=Vector{Float64}[];losses=Float64[];flags=Bool[]
    for value in grid
        expand(v)=let z=copy(reference_z);z[index]=value;z[nuisance]=v;z;end
        if isempty(nuisance)
            z=expand(Float64[]);loss=f(z);flag=isfinite(loss)
        else
            results=[optimize_profile(v->f(expand(v)),s[nuisance],v->full_gradient(expand(v))[nuisance]) for s in seeds]
            conditional_gradient=v->full_gradient(expand(v))[nuisance]
            valid=[r for r in results if stationary(r,conditional_gradient)]
            flag=!isempty(valid)
            chosen=flag ? valid[argmin([Optim.minimum(r) for r in valid])] : nothing
            z=flag ? expand(Optim.minimizer(chosen)) : fill(NaN,n)
            loss=flag ? Optim.minimum(chosen) : Inf
        end
        push!(minimizers,z);push!(losses,loss);push!(flags,flag)
    end
    any(v->v<minimum_value-1e-6*max(1,abs(minimum_value)),losses) &&
        throw(NumericalFailure("a conditional profile improved its reference optimum; refit from that solution"))
    (;target,values=grid,negative_log_density=losses,delta=losses.-minimum_value,
      minimizers,converged=flags,reference=reference_z,gradient_tolerance,
      interpretation=target===:posterior ? :posterior_shape_no_likelihood_cutoff : :likelihood_profile)
end
end
