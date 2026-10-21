module PopulationBalanceModelsAdvancedHMCExt
using PopulationBalanceModels, LinearAlgebra, Random
import AdvancedHMC as HMC
import MCMCDiagnosticTools as MCD

function PopulationBalanceModels.sample_posterior(problem::EstimationProblem;
        initial=initial_coordinates(problem),starts=nothing,laplace=nothing,
        warmup=500,samples=1000,chains=4,seed=2022,target_acceptance=0.8,
        max_depth=8,start_scale=0.5,min_ess=400,rhat_limit=1.01,bfmi_limit=0.3,metric=:auto,derivatives=problem.derivatives)
    warmup isa Integer && warmup>=50 && samples isa Integer && samples>=8 &&
        chains isa Integer && chains>0 || throw(ArgumentError("require warmup>=50, samples>=8, chains>0"))
    0<target_acceptance<1 && max_depth isa Integer && max_depth>0 && isfinite(start_scale) && start_scale>0 &&
        isfinite(min_ess) && min_ess>0 && isfinite(rhat_limit) && rhat_limit>=1 &&
        isfinite(bfmi_limit) && bfmi_limit>0 || throw(ArgumentError("invalid sampler settings"))
    metric in (:auto,:diagonal,:dense) || throw(ArgumentError("metric must be :auto, :diagonal or :dense"))
    dense=metric===:dense || (metric===:auto && laplace!==nothing)
    center=initial isa MAPResult ? initial.minimizer : Float64.(initial)
    n=length(problem.parameters);rng=MersenneTwister(seed)
    length(center)==n && all(isfinite,center) || throw(ArgumentError("invalid initial coordinates"))
    laplace===nothing || (laplace isa LaplaceApproximation && length(laplace.center)==n) || throw(ArgumentError("Laplace initialization has incompatible dimensions"))
    seeds=if starts===nothing
        [center.+start_scale.*(laplace===nothing ? randn(rng,n) : laplace.factor.U\randn(rng,n)) for _ in 1:chains]
    else
        [Float64.(s) for s in starts]
    end
    length(seeds)==chains && all(s->length(s)==n && all(isfinite,s),seeds) || throw(ArgumentError("one initial inference-coordinate vector per chain required"))
    counts=PopulationBalanceModels._InferenceCounts();started=time_ns()
    logdensity=z->logposterior(problem,z;on_failure=:reject,_counts=counts)
    function value_gradient(z)
        value=logdensity(z)
        isfinite(value) || return (value,zeros(n))
        gradient=try
            posterior_gradient(problem,z;derivatives,_counts=counts)
        catch e
            e isa NumericalFailure || rethrow()
            PopulationBalanceModels._record_failure!(counts,e)
            return (-Inf,zeros(n))
        end
        (value,gradient)
    end
    Z=zeros(n,samples,chains);statistics=Any[];warmup_statistics=Any[];gradient_checks=Any[]
    warmup_divergences=0
    for c in 1:chains
        isfinite(logdensity(seeds[c])) || throw(ArgumentError("chain $c starts outside the posterior support"))
        check=check_gradient(problem,seeds[c];derivatives,_counts=counts)
        check.valid || throw(ArgumentError("gradient verification failed for chain $c"))
        push!(gradient_checks,check)
        mass=dense ? (laplace===nothing ? HMC.DenseEuclideanMetric(n) : HMC.DenseEuclideanMetric(copy(laplace.covariance))) :
            (laplace===nothing ? HMC.DiagEuclideanMetric(n) : HMC.DiagEuclideanMetric(diag(laplace.covariance)))
        hamiltonian=HMC.Hamiltonian(mass,logdensity,value_gradient)
        chain_rng=MersenneTwister(rand(rng,UInt))
        epsilon=HMC.find_good_stepsize(chain_rng,hamiltonian,seeds[c])
        integrator=HMC.Leapfrog(epsilon)
        kernel=HMC.HMCKernel(HMC.Trajectory{HMC.MultinomialTS}(integrator,HMC.GeneralisedNoUTurn(max_depth=max_depth)))
        adaptor=HMC.StanHMCAdaptor(HMC.MassMatrixAdaptor(mass),HMC.StepSizeAdaptor(target_acceptance,integrator))
        draws,stats=HMC.sample(chain_rng,hamiltonian,kernel,seeds[c],warmup+samples,adaptor,warmup;
            drop_warmup=false,verbose=false,progress=false)
        Z[:,:,c]=hcat(draws[warmup+1:end]...)
        warmup_divergences+=count(s->s.numerical_error,stats[1:warmup])
        push!(warmup_statistics,stats[1:warmup])
        push!(statistics,stats[warmup+1:end])
    end
    diagnostic_array=permutedims(Z,(2,3,1)) # MCD expects draws,chains,parameters.
    bulk=MCD.ess(diagnostic_array;kind=:bulk)
    tail=MCD.ess(diagnostic_array;kind=:tail)
    rhat=MCD.rhat(diagnostic_array;kind=:rank)
    mcse_mean=MCD.mcse(diagnostic_array)
    energy=hcat([[s.hamiltonian_energy for s in chain] for chain in statistics]...)
    ebfmi=MCD.bfmi(energy)
    per_chain=[(;initial=copy(seeds[c]),
        acceptance_rate=sum(s.acceptance_rate for s in statistics[c])/samples,
        step_size=last(statistics[c]).step_size,
        mean_leapfrog_steps=sum(s.n_steps for s in statistics[c])/samples,
        ebfmi=ebfmi[c],divergences=count(s->s.numerical_error,statistics[c]),
        warmup_divergences=count(s->s.numerical_error,warmup_statistics[c]),
        max_depth_hits=count(s->s.tree_depth>=max_depth,statistics[c])) for c in 1:chains]
    divergences=sum(count(s->s.numerical_error,chain) for chain in statistics)
    depth_hits=sum(count(s->s.tree_depth>=max_depth,chain) for chain in statistics)
    passed=chains>=2 && all(isfinite,bulk) && all(isfinite,tail) && all(isfinite,rhat) &&
        all(isfinite,mcse_mean) && all(isfinite,ebfmi) && minimum(ebfmi)>=bfmi_limit &&
        minimum(bulk)>=min_ess && minimum(tail)>=min_ess && maximum(rhat)<=rhat_limit && divergences==0 && depth_hits==0
    diagnostics=(;ess_bulk=bulk,ess_tail=tail,rhat,mcse_mean,mcse_coordinates=:inference,
        ebfmi,energy,per_chain,divergences,warmup_divergences,
        max_depth_hits=depth_hits,diagnostics_passed=passed,gradient_checks,statistics,warmup_statistics,
        seed,warmup,samples,chains,target_acceptance,max_depth,min_ess,rhat_limit,bfmi_limit,metric=dense ? :dense : :diagonal,
        forward_solves=counts.solves,posterior_evaluations=counts.evaluations,
        gradient_evaluations=counts.gradient_evaluations,adjoint_solves=counts.adjoint_solves,
        derivative_backend=PopulationBalanceModels._gradient_backend(problem,derivatives),
        failed_evaluations=counts.failures,failure_messages=counts.messages,
        elapsed_seconds=(time_ns()-started)/1e9)
    physical=[physical_parameters(problem,Z[:,i,c]) for i in 1:samples,c in 1:chains]
    PosteriorSamples(Z,physical,diagnostics,ParameterSpec[problem.parameters...])
end
end
