# Optional example runner. Each chain has its own Julia process and PBM caches.
# No inference target, adaptation rule or convergence threshold changes here.
function _combine_protein_chains(problem,runs;elapsed_seconds,allow_distinct_tuning=false)
    isempty(runs) && throw(ArgumentError("provide at least one isolated chain"))
    base=diagnostics(first(runs))
    for run in runs
        size(run.z,3)==1 || throw(ArgumentError("each isolated run must contain one chain"))
        length(run.coordinates)==length(problem.parameters) || throw(ArgumentError("parameter count differs"))
        for (a,b) in zip(run.coordinates,problem.parameters)
            (a.name,a.transform,a.prior.coordinates,a.prior.distribution,a.role,a.experiment)==
                (b.name,b.transform,b.prior.coordinates,b.prior.distribution,b.role,b.experiment) ||
                throw(ArgumentError("isolated chain has different parameter coordinates or priors"))
        end
        report=diagnostics(run)
        for key in (:samples,:warmup,:sampling_coordinates,:z_coordinates,:min_ess,
                :rhat_limit,:bfmi_limit,:target_acceptance,:max_depth,:metric,
                :derivative_backend)
            getproperty(report,key)==getproperty(base,key) ||
                throw(ArgumentError("isolated chain setting differs: $key"))
        end
        if !allow_distinct_tuning
            report.initial_covariance==base.initial_covariance ||
                throw(ArgumentError("isolated chain setting differs: initial_covariance; explicitly allow distinct sampler tuning to combine these chains"))
        elseif report.initial_covariance!==nothing
            C=report.initial_covariance
            size(C)==(length(problem.parameters),length(problem.parameters)) &&
                all(isfinite,C) && issymmetric(C) && isposdef(Symmetric(C)) ||
                throw(ArgumentError("each initial sampler tuning matrix must be finite symmetric positive definite"))
        end
    end
    chain_seeds=Tuple(diagnostics(r).seed for r in runs)
    length(unique(chain_seeds))==length(runs) || throw(ArgumentError("isolated chains require distinct random seeds"))
    Z=cat((r.z for r in runs)...;dims=3)
    diagnostic_array=permutedims(Z,(2,3,1))
    MCD=Base.get_extension(PopulationBalanceModels,:PopulationBalanceModelsAdvancedHMCExt).MCD
    bulk=MCD.ess(diagnostic_array;kind=:bulk)
    tail=MCD.ess(diagnostic_array;kind=:tail)
    rhat=MCD.rhat(diagnostic_array;kind=:rank)
    mcse=MCD.mcse(diagnostic_array)
    energy=hcat((diagnostics(r).energy for r in runs)...)
    bfmi=MCD.bfmi(energy)
    joined(key)=vcat((getproperty(diagnostics(r),key) for r in runs)...)
    total(key)=sum(getproperty(diagnostics(r),key) for r in runs)
    divergences=total(:divergences)
    depth_hits=total(:max_depth_hits)
    passed=length(runs)>=2 && all(isfinite,vcat(bulk,tail,rhat,mcse,bfmi)) &&
        minimum(bfmi)>=base.bfmi_limit && minimum(bulk)>=base.min_ess &&
        minimum(tail)>=base.min_ess && maximum(rhat)<=base.rhat_limit &&
        divergences==0 && depth_hits==0
    report=merge(base,(;chains=length(runs),chain_seeds,elapsed_seconds,
        initial_covariances=[diagnostics(r).initial_covariance for r in runs],
        distinct_initial_tuning=allow_distinct_tuning,
        sum_chain_elapsed_seconds=total(:elapsed_seconds),ess_bulk=bulk,ess_tail=tail,
        rhat,mcse_mean=mcse,energy,ebfmi=bfmi,divergences,max_depth_hits=depth_hits,
        warmup_divergences=total(:warmup_divergences),diagnostics_passed=passed,
        statistics=joined(:statistics),warmup_statistics=joined(:warmup_statistics),
        per_chain=joined(:per_chain),gradient_checks=joined(:gradient_checks),map_checks=joined(:map_checks),
        forward_solves=total(:forward_solves),posterior_evaluations=total(:posterior_evaluations),
        gradient_evaluations=total(:gradient_evaluations),adjoint_solves=total(:adjoint_solves),
        failed_evaluations=total(:failed_evaluations),failure_messages=joined(:failure_messages)))
    PosteriorSamples(Z,hcat((r.physical for r in runs)...),report,first(runs).coordinates)
end

