module PopulationBalanceModelsAdvancedHMCExt
using PopulationBalanceModels, LinearAlgebra, Random
import AdvancedHMC as HMC
import MCMCDiagnosticTools as MCD
import ForwardDiff

function PopulationBalanceModels.sample_posterior(problem::EstimationProblem;
        initial=initial_coordinates(problem),starts=nothing,laplace=nothing,
        warmup=500,samples=1000,chains=4,seed=2022,target_acceptance=0.8,
        max_depth=8,start_scale=0.5,min_ess=400,rhat_limit=1.01,bfmi_limit=0.3,metric=:auto,
        initial_covariance=nothing,sampling_map=nothing,verbose=false,derivatives=problem.derivatives)
    warmup isa Integer && warmup>=50 && samples isa Integer && samples>=8 &&
        chains isa Integer && chains>0 || throw(ArgumentError("require warmup>=50, samples>=8, chains>0"))
    0<target_acceptance<1 && max_depth isa Integer && max_depth>0 && isfinite(start_scale) && start_scale>0 &&
        isfinite(min_ess) && min_ess>0 && isfinite(rhat_limit) && rhat_limit>=1 &&
        isfinite(bfmi_limit) && bfmi_limit>0 || throw(ArgumentError("invalid sampler settings"))
    metric in (:auto,:diagonal,:dense) || throw(ArgumentError("metric must be :auto, :diagonal or :dense"))
    dense=metric===:dense || (metric===:auto && (laplace!==nothing || initial_covariance!==nothing))
    center=initial isa MAPResult ? initial.minimizer : Float64.(initial)
    n=length(problem.parameters);rng=MersenneTwister(seed)
    mapped=sampling_map!==nothing
    mapped && laplace!==nothing && throw(ArgumentError("Laplace metric initialization is in inference coordinates; supply an explicit sampler-coordinate tuning matrix with sampling_map"))
    mapped && PopulationBalanceModels._validate_sampling_map(sampling_map)
    to_sampler=z->mapped ? sampling_map.to_sampler(z) : z
    from_sampler=y->mapped ? sampling_map.from_sampler(y) : y
    logjac=y->mapped ? sampling_map.logabsdetjac(y) : 0.0
    initial_covariance!==nothing && laplace!==nothing && throw(ArgumentError("supply either Laplace initialization or an initial tuning covariance"))
    tuning=if initial_covariance===nothing
        nothing
    else
        C=Matrix{Float64}(initial_covariance)
        size(C)==(n,n) && all(isfinite,C) && issymmetric(C) || throw(ArgumentError("initial_covariance must be finite symmetric n×n"))
        isposdef(Symmetric(C)) || throw(ArgumentError("initial_covariance must be positive definite; it is not repaired"))
        C
    end
    length(center)==n && all(isfinite,center) || throw(ArgumentError("invalid initial coordinates"))
    laplace===nothing || (laplace isa LaplaceApproximation && length(laplace.center)==n) || throw(ArgumentError("Laplace initialization has incompatible dimensions"))
    seeds=if starts===nothing
        [center.+start_scale.*(laplace===nothing ? randn(rng,n) : laplace.factor.U\randn(rng,n)) for _ in 1:chains]
    else
        [Float64.(s) for s in starts]
    end
    length(seeds)==chains && all(s->length(s)==n && all(isfinite,s),seeds) || throw(ArgumentError("one initial inference-coordinate vector per chain required"))
    counts=PopulationBalanceModels._InferenceCounts();started=time_ns()
    function logdensity(y)
        state=try
            mapped ? PopulationBalanceModels._sampling_map_state(sampling_map,y) : nothing
        catch e
            e isa NumericalFailure || rethrow()
            PopulationBalanceModels._record_failure!(counts,e)
            return -Inf
        end
        logposterior(problem,mapped ? state.z : y;on_failure=:reject,_counts=counts)+
            (mapped ? state.logjac : 0.0)
    end
    function value_gradient(y)
        value=logdensity(y)
        isfinite(value) || return (value,zeros(n))
        gradient=try
            gz=posterior_gradient(problem,from_sampler(y);derivatives,_counts=counts)
            mapped ? ForwardDiff.jacobian(from_sampler,y)'*gz+ForwardDiff.gradient(logjac,y) : gz
        catch e
            e isa NumericalFailure || rethrow()
            PopulationBalanceModels._record_failure!(counts,e)
            return (-Inf,zeros(n))
        end
        (value,gradient)
    end
    Z=zeros(n,samples,chains);statistics=Any[];warmup_statistics=Any[];gradient_checks=Any[];map_checks=Any[]
    warmup_divergences=0
    for c in 1:chains
        y0=Float64.(to_sampler(seeds[c]))
        length(y0)==n && all(isfinite,y0) && isapprox(from_sampler(y0),seeds[c];rtol=1e-10,atol=1e-12) ||
            throw(ArgumentError("sampling coordinate maps must be finite inverses"))
        isfinite(logdensity(y0)) || throw(ArgumentError("chain $c starts outside the posterior support"))
        check=check_gradient(problem,seeds[c];derivatives,_counts=counts)
        check.valid || throw(ArgumentError("gradient verification failed for chain $c"))
        push!(gradient_checks,check)
        if mapped
            map_check=check_sampling_map(problem,sampling_map,seeds[c];derivatives,_counts=counts)
            map_check.valid || throw(ArgumentError("transformed sampler gradient verification failed"))
            push!(map_checks,map_check)
        end
        covariance=tuning===nothing ? (laplace===nothing ? nothing : laplace.covariance) : tuning
        mass=dense ? (covariance===nothing ? HMC.DenseEuclideanMetric(n) : HMC.DenseEuclideanMetric(copy(covariance))) :
            (covariance===nothing ? HMC.DiagEuclideanMetric(n) : HMC.DiagEuclideanMetric(diag(covariance)))
        hamiltonian=HMC.Hamiltonian(mass,logdensity,value_gradient)
        chain_rng=MersenneTwister(rand(rng,UInt))
        verbose && (println("Starting posterior chain ",c,"/",chains);flush(stdout))
        epsilon=HMC.find_good_stepsize(chain_rng,hamiltonian,y0)
        integrator=HMC.Leapfrog(epsilon)
        kernel=HMC.HMCKernel(HMC.Trajectory{HMC.MultinomialTS}(integrator,HMC.GeneralisedNoUTurn(max_depth=max_depth)))
        adaptor=HMC.StanHMCAdaptor(HMC.MassMatrixAdaptor(mass),HMC.StepSizeAdaptor(target_acceptance,integrator))
        draws,stats=HMC.sample(chain_rng,hamiltonian,kernel,y0,warmup+samples,adaptor,warmup;
            drop_warmup=false,verbose=false,progress=false)
        Z[:,:,c]=hcat((from_sampler(y) for y in draws[warmup+1:end])...)
        # Preserve the established meaning of statistics.log_density: p_z(z).
        # Hamiltonian energies remain those of the actual sampler coordinates.
        mapped && (stats=[merge(st,(log_density=st.log_density-logjac(y),)) for (st,y) in zip(stats,draws)])
        verbose && (println("Finished posterior chain ",c,"; retained mean leapfrog steps=",
            sum(s.n_steps for s in stats[warmup+1:end])/samples);flush(stdout))
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
        initial_covariance=tuning,metric_initialization=tuning===nothing ? (laplace===nothing ? :identity : :laplace) : :user_tuning,
        sampling_coordinates=mapped ? sampling_map.label : :inference,z_coordinates=:inference,
        map_checks,sampling_map_validation=mapped ? :every_evaluation : :none,
        forward_solves=counts.solves,posterior_evaluations=counts.evaluations,
        gradient_evaluations=counts.gradient_evaluations,adjoint_solves=counts.adjoint_solves,
        derivative_backend=PopulationBalanceModels._gradient_backend(problem,derivatives),
        failed_evaluations=counts.failures,failure_messages=counts.messages,
        elapsed_seconds=(time_ns()-started)/1e9)
    physical=[physical_parameters(problem,Z[:,i,c]) for i in 1:samples,c in 1:chains]
    PosteriorSamples(Z,physical,diagnostics,ParameterSpec[problem.parameters...])
end
end
