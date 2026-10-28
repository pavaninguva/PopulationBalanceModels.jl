module PopulationBalanceModelsAdvancedHMCExt
using PopulationBalanceModels, LinearAlgebra, Random
import AdvancedHMC as HMC
import MCMCDiagnosticTools as MCD
import ForwardDiff

# These are the same transitions and Stan adaptation used by HMC.sample in
# AdvancedHMC 0.8. Retaining the phase point (including momentum and caches),
# adapted Hamiltonian/kernel and private RNG avoids restarting a chain per batch.
mutable struct _PosteriorChain{R,H,K,Z}
    rng::R
    hamiltonian::H
    kernel::K
    point::Z
    draws::Vector{Vector{Float64}}
    statistics::Vector{NamedTuple}
    warmup_statistics::Vector{NamedTuple}
    quantities::Vector{Vector{Float64}}
end

function _stored_stat(stat,y,logjac)
    # Existing log_density means p_z(z); energies use the actual sampler target.
    merge(stat,(sampler_log_density=stat.log_density,log_density=stat.log_density-logjac(y)))
end

function _warmup(rng,hamiltonian,kernel,adaptor,initial,warmup,logjac)
    h,transition=HMC.sample_init(rng,hamiltonian,initial)
    point=transition.z
    statistics=NamedTuple[]
    for i in 1:warmup
        transition=HMC.transition(rng,h,kernel,point)
        stat=HMC.stat(transition)
        h,kernel,adapted=HMC.adapt!(h,kernel,adaptor,i,warmup,transition.z,stat.acceptance_rate)
        push!(statistics,_stored_stat(merge(stat,(is_adapt=adapted,)),transition.z.θ,logjac))
        point=transition.z
    end
    _PosteriorChain(rng,h,kernel,point,Vector{Float64}[],NamedTuple[],statistics,Vector{Float64}[])
end

function _advance!(chain,count,problem,from_sampler,logjac,quantities)
    for _ in 1:count
        transition=HMC.transition(chain.rng,chain.hamiltonian,chain.kernel,chain.point)
        chain.point=transition.z
        z=Float64.(from_sampler(chain.point.θ))
        length(z)==length(problem.parameters) && all(isfinite,z) ||
            throw(NumericalFailure("sampler returned invalid inference coordinates"))
        push!(chain.draws,copy(z))
        push!(chain.statistics,_stored_stat(merge(HMC.stat(transition),(is_adapt=false,)),chain.point.θ,logjac))
        if !isempty(quantities)
            theta=physical_parameters(problem,z)
            observed=map(values(quantities)) do f
                value=f(theta)
                value isa Real && isfinite(value) ||
                    throw(ArgumentError("diagnostic quantities must return finite real scalars"))
                Float64(value)
            end
            push!(chain.quantities,collect(observed))
        end
    end
end

function _mcse_settings(names,limits)
    limits isa NamedTuple || throw(ArgumentError("mcse_limits must be a named tuple"))
    all(name->name in names,keys(limits)) ||
        throw(ArgumentError("every mcse_limits name must have a diagnostic quantity"))
    map(names) do name
        spec=get(limits,name,NamedTuple())
        spec=spec isa Real ? (mean=spec,) : spec
        spec isa NamedTuple && all(k->k in (:mean,:quantiles),keys(spec)) ||
            throw(ArgumentError("MCSE limits use a positive mean limit or (mean=...,quantiles=(probability=>limit,...))"))
        mean_limit=get(spec,:mean,nothing)
        mean_limit===nothing || (mean_limit isa Real && isfinite(mean_limit) && mean_limit>0) ||
            throw(ArgumentError("mean MCSE limits must be finite and positive"))
        quantiles=get(spec,:quantiles,())
        quantiles isa Union{Tuple,AbstractVector,AbstractDict} ||
            throw(ArgumentError("quantile MCSE limits must contain probability=>limit pairs"))
        all(q->q isa Pair && first(q) isa Real && last(q) isa Real &&
            isfinite(first(q)) && 0<first(q)<1 && isfinite(last(q)) && last(q)>0,quantiles) ||
            throw(ArgumentError("quantile MCSE probabilities must lie inside (0,1), with positive finite limits"))
        probs=first.(collect(quantiles))
        length(unique(probs))==length(probs) || throw(ArgumentError("duplicate quantile MCSE probability"))
        (mean=mean_limit,quantiles=Tuple(sort!(collect(quantiles);by=first)))
    end
end

function _scalar_screen(bulk,tail,rhat,mcse,min_ess,rhat_limit)
    all(isfinite,bulk) && all(isfinite,tail) && all(isfinite,rhat) && all(isfinite,mcse) &&
        minimum(bulk)>=min_ess && minimum(tail)>=min_ess && maximum(rhat)<=rhat_limit
end

function PopulationBalanceModels.diagnostics(Z::AbstractArray{<:Real,3},statistics::AbstractVector;
        min_ess=400,rhat_limit=1.01,bfmi_limit=0.3,max_depth=8,
        quantity_values=NamedTuple(),mcse_limits=NamedTuple())
    n,samples,chains=size(Z)
    n>0 && samples>=8 && chains>0 && all(isfinite,Z) ||
        throw(ArgumentError("diagnostics require finite (parameters,draws,chains) values with at least eight draws"))
    isfinite(min_ess) && min_ess>0 && isfinite(rhat_limit) && rhat_limit>=1 &&
        isfinite(bfmi_limit) && bfmi_limit>0 && max_depth isa Integer && max_depth>0 ||
        throw(ArgumentError("invalid sampler diagnostic thresholds"))
    length(statistics)==chains && all(s->s isa AbstractVector && length(s)==samples,statistics) ||
        throw(DimensionMismatch("one retained-statistics vector per chain, matching the draw count, is required"))
    all(!get(s,:is_adapt,false) for chain in statistics for s in chain) ||
        throw(ArgumentError("exclude warmup from retained sampling diagnostics"))
    quantity_values isa NamedTuple || throw(ArgumentError("quantity_values must be a named tuple of draw×chain matrices"))
    limits=_mcse_settings(keys(quantity_values),mcse_limits)
    all(v->v isa AbstractMatrix{<:Real} && size(v)==(samples,chains) && all(isfinite,v),values(quantity_values)) ||
        throw(ArgumentError("quantity_values must be finite draw×chain matrices matching Z"))
    array=permutedims(Z,(2,3,1))
    bulk=MCD.ess(array;kind=:bulk)
    tail=MCD.ess(array;kind=:tail)
    rhat=MCD.rhat(array;kind=:rank)
    mcse=MCD.mcse(array)
    energy=hcat([[s.hamiltonian_energy for s in chain] for chain in statistics]...)
    ebfmi=MCD.bfmi(energy)
    per_chain=[(;acceptance_rate=sum(s.acceptance_rate for s in statistics[c])/samples,
        step_size=last(statistics[c]).step_size,
        mean_leapfrog_steps=sum(s.n_steps for s in statistics[c])/samples,
        ebfmi=ebfmi[c],divergences=count(s->s.numerical_error,statistics[c]),
        max_depth_hits=count(s->s.tree_depth>=max_depth,statistics[c])) for c in 1:chains]
    divergences=sum(c.divergences for c in per_chain)
    depth_hits=sum(c.max_depth_hits for c in per_chain)
    quantities=map(values(quantity_values),limits) do values,limit
        a=reshape(values,samples,chains,1)
        qb=MCD.ess(a;kind=:bulk);qt=MCD.ess(a;kind=:tail)
        qr=MCD.rhat(a;kind=:rank);qm=MCD.mcse(a)
        quantiles=map(limit.quantiles) do q
            # Use the backend's Statistics function identity so its dedicated
            # quantile-ESS estimator is selected; no new core dependency.
            error=MCD.mcse(values;kind=Base.Fix2(MCD.Statistics.quantile,first(q)))
            (;probability=first(q),mcse=error,limit=last(q),passed=isfinite(error) && error<=last(q))
        end
        passed=_scalar_screen(qb,qt,qr,qm,min_ess,rhat_limit) &&
            (limit.mean===nothing || only(qm)<=limit.mean) && all(q->q.passed,quantiles)
        (;ess_bulk=only(qb),ess_tail=only(qt),rhat=only(qr),mcse_mean=only(qm),
          mean_mcse_limit=limit.mean,quantile_mcse=quantiles,diagnostics_passed=passed,values=copy(values))
    end
    quantity_diagnostics=NamedTuple{keys(quantity_values)}(quantities)
    passed=chains>=2 && _scalar_screen(bulk,tail,rhat,mcse,min_ess,rhat_limit) &&
        all(isfinite,ebfmi) && minimum(ebfmi)>=bfmi_limit && divergences==0 && depth_hits==0 &&
        all(q->q.diagnostics_passed,values(quantity_diagnostics))
    (;ess_bulk=bulk,ess_tail=tail,rhat,mcse_mean=mcse,mcse_coordinates=:inference,
        ebfmi,energy,per_chain,divergences,max_depth_hits=depth_hits,diagnostics_passed=passed,
        screen_passed=passed,quantity_diagnostics,mcse_limits=deepcopy(mcse_limits),
        min_ess,rhat_limit,bfmi_limit,max_depth,samples,chains)
end

function _snapshot(report,eligible,consecutive,started)
    quantities=map(report.quantity_diagnostics) do q
        (;q.ess_bulk,q.ess_tail,q.rhat,q.mcse_mean,q.mean_mcse_limit,q.quantile_mcse,q.diagnostics_passed)
    end
    (;samples=report.samples,eligible,screen_passed=report.diagnostics_passed,
        diagnostics_passed=eligible && report.diagnostics_passed,consecutive_passes=consecutive,
        rhat=copy(report.rhat),ess_bulk=copy(report.ess_bulk),ess_tail=copy(report.ess_tail),
        mcse_mean=copy(report.mcse_mean),ebfmi=copy(report.ebfmi),divergences=report.divergences,
        max_depth_hits=report.max_depth_hits,quantity_diagnostics=quantities,
        elapsed_seconds=(time_ns()-started)/1e9)
end

function _draw_array(states,n)
    samples=length(first(states).draws)
    Z=Array{Float64}(undef,n,samples,length(states))
    for c in eachindex(states),i in 1:samples
        Z[:,i,c]=states[c].draws[i]
    end
    Z
end

function _quantity_arrays(states,names)
    samples=length(first(states).draws)
    NamedTuple{names}(Tuple([states[c].quantities[i][j] for i in 1:samples,c in eachindex(states)] for j in eachindex(names)))
end

function _tuning_matrix(input,n)
    input isa AbstractMatrix || throw(ArgumentError("each initial covariance must be an n×n matrix"))
    C=Matrix{Float64}(input)
    size(C)==(n,n) && all(isfinite,C) && issymmetric(C) ||
        throw(ArgumentError("initial_covariance must be finite symmetric n×n"))
    isposdef(Symmetric(C)) || throw(ArgumentError("initial_covariance must be positive definite; it is not repaired"))
    C
end

function PopulationBalanceModels.sample_posterior(problem::EstimationProblem;
        initial=initial_coordinates(problem),starts=nothing,laplace=nothing,
        warmup=500,samples=nothing,chains=4,seed=2022,target_acceptance=0.8,
        max_depth=8,start_scale=0.5,min_ess=400,rhat_limit=1.01,bfmi_limit=0.3,metric=:auto,
        initial_covariance=nothing,sampling_map=nothing,verbose=false,derivatives=problem.derivatives,
        stopping=:auto,min_samples=500,batch_samples=250,max_samples=10_000,consecutive_passes=2,
        diagnostic_quantities=NamedTuple(),mcse_limits=NamedTuple())
    stopping in (:auto,:diagnostics,:fixed) || throw(ArgumentError("stopping must be :auto, :diagnostics or :fixed"))
    policy=stopping===:auto ? (samples===nothing ? :diagnostics : :fixed) : stopping
    policy===:diagnostics && samples!==nothing &&
        throw(ArgumentError("diagnostic stopping uses min_samples/max_samples; omit samples or select stopping=:fixed"))
    policy===:fixed && !(samples isa Integer && samples>=8) &&
        throw(ArgumentError("fixed-length sampling requires samples>=8"))
    warmup isa Integer && warmup>=50 && chains isa Integer && chains>0 ||
        throw(ArgumentError("require warmup>=50 and chains>0"))
    if policy===:diagnostics
        chains>=2 || throw(ArgumentError("diagnostic stopping requires at least two chains; use explicit samples for a fixed single-chain run"))
        min_samples isa Integer && min_samples>=8 && batch_samples isa Integer && batch_samples>=8 &&
            max_samples isa Integer && max_samples>=min_samples &&
            consecutive_passes isa Integer && consecutive_passes>=1 ||
            throw(ArgumentError("require min_samples>=8, batch_samples>=8, max_samples>=min_samples and consecutive_passes>=1"))
    end
    0<target_acceptance<1 && max_depth isa Integer && max_depth>0 && isfinite(start_scale) && start_scale>0 &&
        isfinite(min_ess) && min_ess>0 && isfinite(rhat_limit) && rhat_limit>=1 &&
        isfinite(bfmi_limit) && bfmi_limit>0 || throw(ArgumentError("invalid sampler settings"))
    diagnostic_quantities isa NamedTuple || throw(ArgumentError("diagnostic_quantities must be named scalar functions of physical parameters"))
    _mcse_settings(keys(diagnostic_quantities),mcse_limits)
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
    elseif initial_covariance isa AbstractMatrix
        _tuning_matrix(initial_covariance,n)
    else
        initial_covariance isa Union{Tuple,AbstractVector} && length(initial_covariance)==chains ||
            throw(ArgumentError("supply a shared initial covariance matrix or one matrix per chain"))
        Tuple(_tuning_matrix(C,n) for C in initial_covariance)
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
    gradient_checks=Any[];map_checks=Any[];chain_seeds=UInt[]
    states=map(1:chains) do c
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
        covariance=tuning===nothing ? (laplace===nothing ? nothing : laplace.covariance) :
            tuning isa Tuple ? tuning[c] : tuning
        mass=dense ? (covariance===nothing ? HMC.DenseEuclideanMetric(n) : HMC.DenseEuclideanMetric(copy(covariance))) :
            (covariance===nothing ? HMC.DiagEuclideanMetric(n) : HMC.DiagEuclideanMetric(diag(covariance)))
        hamiltonian=HMC.Hamiltonian(mass,logdensity,value_gradient)
        chain_seed=rand(rng,UInt)
        push!(chain_seeds,chain_seed)
        chain_rng=MersenneTwister(chain_seed)
        verbose && (println("Warmup chain ",c,"/",chains);flush(stdout))
        epsilon=HMC.find_good_stepsize(chain_rng,hamiltonian,y0)
        integrator=HMC.Leapfrog(epsilon)
        kernel=HMC.HMCKernel(HMC.Trajectory{HMC.MultinomialTS}(integrator,HMC.GeneralisedNoUTurn(max_depth=max_depth)))
        adaptor=HMC.StanHMCAdaptor(HMC.MassMatrixAdaptor(mass),HMC.StepSizeAdaptor(target_acceptance,integrator))
        _warmup(chain_rng,hamiltonian,kernel,adaptor,y0,warmup,logjac)
    end
    history=NamedTuple[];passes=0;kept=0
    budget=policy===:fixed ? samples : max_samples
    batch=policy===:fixed ? samples : batch_samples
    reason=policy===:fixed ? :fixed_length : :max_samples
    report=nothing;Z=zeros(n,0,chains)
    while kept<budget
        count=min(batch,budget-kept)
        for state in states
            _advance!(state,count,problem,from_sampler,logjac,diagnostic_quantities)
        end
        kept+=count
        Z=_draw_array(states,n)
        qvalues=_quantity_arrays(states,keys(diagnostic_quantities))
        report=diagnostics(Z,[s.statistics for s in states];min_ess,rhat_limit,bfmi_limit,max_depth,
            quantity_values=qvalues,mcse_limits)
        eligible=policy===:fixed || kept>=min_samples
        passes=eligible && report.diagnostics_passed ? passes+1 : 0
        push!(history,_snapshot(report,eligible,passes,started))
        verbose && (println("Retained ",kept," draws/chain; max R-hat=",maximum(report.rhat),
            ", min bulk/tail ESS=",min(minimum(report.ess_bulk),minimum(report.ess_tail)),
            ", screen=",report.diagnostics_passed,", consecutive passes=",passes);flush(stdout))
        if policy===:diagnostics
            # These cumulative zero-tolerance conditions cannot recover merely
            # by appending draws. Preserve the draws and report the failure.
            if report.divergences>0 || report.max_depth_hits>0
                reason=:sampler_pathology
                break
            elseif passes>=consecutive_passes
                reason=:diagnostics_passed
                break
            end
        end
    end
    statistics=[s.statistics for s in states]
    warmup_statistics=[s.warmup_statistics for s in states]
    per_chain=[merge(report.per_chain[c],(;initial=copy(seeds[c]),
        warmup_divergences=count(s->s.numerical_error,warmup_statistics[c]),
        adapted_inverse_mass=copy(states[c].hamiltonian.metric.M⁻¹))) for c in 1:chains]
    diagnostics_passed=report.diagnostics_passed && (policy===:fixed || reason===:diagnostics_passed)
    final=merge(report,(;diagnostics_passed,per_chain,statistics,warmup_statistics,
        warmup_divergences=sum(c.warmup_divergences for c in per_chain),
        stopping_policy=policy,stopping_reason=reason,requested_stopping=stopping,
        requested_samples=samples,min_samples=policy===:fixed ? samples : min_samples,
        max_samples=budget,batch_samples=batch,consecutive_passes_required=policy===:fixed ? 1 : consecutive_passes,
        check_history=history,diagnostic_quantity_names=keys(diagnostic_quantities),
        seed,chain_seeds=Tuple(chain_seeds),warmup,target_acceptance,start_scale,metric=dense ? :dense : :diagonal,
        initial_covariance=tuning,metric_initialization=tuning===nothing ? (laplace===nothing ? :identity : :laplace) :
            tuning isa Tuple ? :user_tuning_per_chain : :user_tuning,
        sampling_coordinates=mapped ? sampling_map.label : :inference,z_coordinates=:inference,
        map_checks,sampling_map_validation=mapped ? :every_evaluation : :none,gradient_checks,
        forward_solves=counts.solves,posterior_evaluations=counts.evaluations,
        gradient_evaluations=counts.gradient_evaluations,adjoint_solves=counts.adjoint_solves,
        derivative_backend=PopulationBalanceModels._gradient_backend(problem,derivatives),
        failed_evaluations=counts.failures,failure_messages=counts.messages,
        julia_version=string(VERSION),
        backend_versions=(AdvancedHMC=string(Base.pkgversion(HMC)),MCMCDiagnosticTools=string(Base.pkgversion(MCD))),
        elapsed_seconds=(time_ns()-started)/1e9))
    physical=[physical_parameters(problem,Z[:,i,c]) for i in 1:kept,c in 1:chains]
    PosteriorSamples(Z,physical,final,ParameterSpec[problem.parameters...])
end
end
