# Reporting support, included only by run.jl. No alternative PBM or posterior.
include("protein_sampling_policy.jl")
# JSON null records undefined diagnostics explicitly (e.g. an 8-draw smoke
# chain); failed screens remain false and cannot produce credible bands.
json_safe(x::AbstractFloat)=isfinite(x) ? x : nothing
json_safe(x::NamedTuple)=map(json_safe,x)
json_safe(x::AbstractArray)=map(json_safe,x)
json_safe(x::AbstractDict)=Dict(k=>json_safe(v) for (k,v) in x)
json_safe(x)=x
json(path,value)=open(path,"w") do io;JSON3.pretty(io,json_safe(value));println(io);end
function csv(path,rows)
    isempty(rows) && return
    field(x)=occursin(r"[,\"\n]",string(x)) ? "\""*replace(string(x),"\""=>"\"\"")*"\"" : string(x)
    open(path,"w") do io
        println(io,join(keys(first(rows)),','))
        for row in rows;println(io,join(field.(values(row)),','));end
    end
end
hashfile(path)=bytes2hex(sha256(read(path)))
readjson(path)=JSON3.read(read(path,String))
function numerical_sources()
    paths=vcat(["examples/isothermal_protein_crystallization.jl"],
        [relpath(joinpath(dir,f),ROOT) for folder in ("src","ext") for (dir,_,files) in walkdir(joinpath(ROOT,folder)) for f in files if endswith(f,".jl")],
        [relpath(joinpath(C.DATA,f),ROOT) for f in ("concentration.csv","psd_cdf.csv","psd_summary.csv")])
    Dict(path=>hashfile(joinpath(ROOT,path)) for path in sort(paths))
end
function export_data(root)
    folder=joinpath(root,"shared");mkpath(folder)
    d=load_data()
    csv(joinpath(folder,"concentration.csv"),[(id=r.observation_id,time_min=r.model_time_min,
        observed_g_L=r.concentration_g_L,sigma_g_L=.05r.concentration_g_L,split=r.split) for r in d.concentration])
    rows=vcat(select_psds(d),select_psds(d,:validation))
    csv(joinpath(folder,"psd.csv"),[(time_min=r.time_min,split=r.split,size_um=r.size_um,
        conditional_density_per_um=r.conditional_density_per_um,conditional_cdf=r.conditional_cdf) for group in rows for r in group])
    quantiles=NamedTuple[]
    for group in rows
        r=first(group);s=only(filter(s->s.window=="isothermal_core" && s.time_min==r.time_min,d.summary))
        observed=[s.D10_um,s.D50_um,s.D90_um]
        fromcdf=cdf_quantiles([r.size_um for r in group],[r.conditional_cdf for r in group],[.1,.5,.9])
        maximum(abs.(observed-fromcdf))<1e-10 || error("supplied summary/CDF mismatch")
        for (p,x) in zip((.1,.5,.9),observed)
            push!(quantiles,(time_min=r.time_min,split=r.split,probability=p,observed_um=x))
        end
    end
    csv(joinpath(folder,"quantiles.csv"),quantiles)
    json(joinpath(folder,"observation_model.json"),(concentration_relative_sd=.05,CDF_absolute_sd=C.SIGMA_F,
        CDF_correlation_length_um=C.CORRELATION_LENGTH,CDF_ordinates_um=collect(3.:24.),window_um=C.WINDOW,
        initial_fixed_g_L=C.C0,calibration_concentrations=11,validation_concentrations=4,
        calibration_PSD_times=[74,180,420],validation_PSD_times=[660],quantiles_in_likelihood=false,
        uncertainties="fixed modeling assumptions, not measured analytical errors",
        data_hashes=Dict(f=>hashfile(joinpath(C.DATA,f)) for f in ("concentration.csv","psd_cdf.csv","psd_summary.csv"))))
    json(joinpath(folder,"priors.json"),C.prior_metadata(estimation_problem(model=:cnt_grd)))
    json(joinpath(folder,"fixed_parameters.json"),(gamma=C.FIXED_GAMMA,gamma_units="mJ/m²",
        rho=C.RHO,rho_units="g/μm³",population_units="particles/L",time_units="min",
        length_units="μm",D0_prior_mean=C.D0_PRIOR_MEAN,D0_units="μm²/min"))
end
function model_settings(config,model)
    (;model=string(model),formulation="fixed_gamma_constant_D0",population_units="particles/L",
      bins=config["bins"],Lmax_um=config["Lmax"],maximum_dt_min=config["maximum_dt"],
      coefficient_interval_min=config["coefficient_interval"],cfl=config["cfl"],
      time_method="SSPRK33",growth_reconstruction="MUSCL",diffusion_method=model===:cnt_grd ? "FiniteVolumeDiffusion (Ito)" : "none",
      constants=(c0=C.C0,cstar=2.16,T=C.TEMPERATURE,nu=C.MOLECULAR_VOLUME,kB=C.BOLTZMANN,rho_g_um3=C.RHO,gamma_mJ_m2=C.FIXED_GAMMA,kv=.81),
      observation_model=(concentration_relative_sd=.05,CDF_absolute_sd=C.SIGMA_F,CDF_correlation_length_um=C.CORRELATION_LENGTH))
end
function make_estimation(config,model;bins=config["bins"],Lmax=config["Lmax"],dt=config["maximum_dt"],cfl=config["cfl"])
    estimation_problem(;model,bins,Lmax,dt,cfl,coefficient_interval=config["coefficient_interval"])
end
function physical_from_z(problem,z)
    merge(parameters(),physical_parameters(problem,z,:batch))
end
function solve_physical(p,config,model,times;bins=config["bins"],Lmax=config["Lmax"],dt=config["maximum_dt"],cfl=config["cfl"])
    forward(p;model,bins,Lmax,dt,cfl,coefficient_interval=config["coefficient_interval"],times)
end
function map_export(problem,fitted)
    (;z=fitted.minimizer,parameters=physical_from_z(problem,fitted.minimizer),converged=fitted.converged,
      loglikelihood_whitened_CDF=fitted.loglikelihood,logprior=fitted.logprior,logposterior=fitted.logposterior,
      original_CDF_loglikelihood_offset=-sum(sum(log,diag(C.cdf_factor(m.observable.points))) for m in only(problem.experiments).measurements[2:end]),
      MAP_coordinates="z: logit(logAJ), log(AG), logit(g), optional log(D0); gamma fixed",
      gradient_norm=fitted.diagnostics.gradient_norm,gradient=fitted.diagnostics.gradient,
      elapsed_seconds=fitted.diagnostics.elapsed_seconds,forward_solves=fitted.diagnostics.forward_solves,
      failures=fitted.diagnostics.failed_evaluations,failure_messages=fitted.diagnostics.failure_messages,
      starts=[(start=r.start,converged=r.converged,objective=r.result===nothing ? nothing : Optim.minimum(r.result)) for r in fitted.diagnostics.starts])
end
function check_manifest(folder,hash)
    f=joinpath(folder,"manifest.json");isfile(f) || return false
    m=readjson(f);m.target_hash==hash || return false
    all(isfile(joinpath(folder,string(k))) && hashfile(joinpath(folder,string(k)))==v for (k,v) in pairs(m.files))
end
function write_manifest(folder,target_hash,sources;complete=false)
    files=Dict(f=>hashfile(joinpath(folder,f)) for f in readdir(folder) if
        isfile(joinpath(folder,f)) && f!="manifest.json")
    json(joinpath(folder,"manifest.json"),(mode="production",complete,target_hash,source_files=sources,files))
end
function same_environment(folder)
    path=joinpath(folder,"provenance.json")
    isfile(path) || return false
    previous=readjson(path)
    previous.julia==string(VERSION) || return false
    recorded=Dict(string(k)=>string(v) for (k,v) in pairs(previous.dependencies))
    current=Dict(x.name=>string(x.version) for x in values(Pkg.dependencies()) if x.version!==nothing)
    recorded==current
end
function quantiles_at(sol,u)
    g=sol.problem.operator.system.populations.crystals.grid
    N=population(sol.problem,u,:crystals)
    mass=sum(N[i]*max(0.,min(g.edges[i+1],C.WINDOW[2])-max(g.edges[i],C.WINDOW[1]))/g.widths[i] for i in eachindex(N))
    iszero(mass) ? fill(NaN,3) : conditional_quantiles(g,N,[.1,.5,.9];window=C.WINDOW)
end
function density_at(sol,u)
    g=sol.problem.operator.system.populations.crystals.grid;N=population(sol.problem,u,:crystals)
    edges=sort(unique(vcat(C.WINDOW[1],[x for x in g.edges if C.WINDOW[1]<x<C.WINDOW[2]],C.WINDOW[2])))
    F=conditional_cdf(g,N,edges;window=C.WINDOW)
    (;left=edges[1:end-1],right=edges[2:end],density=diff(F)./diff(edges))
end
function state_diagnostics(sol,u,p,t)
    pb=sol.problem;g=pb.operator.system.populations.crystals.grid
    N=population(pb,u,:crystals);c=bulk(pb,u,:c);q=quantiles_at(sol,u)
    J,G=rates(p,t,c);solid=p.rho*p.kv*moment(g,N,3)
    (;concentration_g_L=c,number_per_L=sum(N),number_per_m3=1000sum(N),
      J_per_L_min=J,G_um_min=G,D_G_um2_min=p.grd_strength,
      solid_concentration_g_L=solid,material_error_g_L=c+solid-p.c0,
      minimum_count_per_L=minimum(N),upper_bin_fraction=sum(N)>0 ? N[end]/sum(N) : 0.,
      D10_um=q[1],D50_um=q[2],D90_um=q[3])
end
function compare_forward(a,b)
    ga=a.problem.operator.system.populations.crystals.grid;gb=b.problem.operator.system.populations.crystals.grid
    dc=maximum(abs(bulk(a.problem,u,:c)-bulk(b.problem,v,:c)) for (u,v) in zip(a.u,b.u))
    dF=maximum(maximum(abs.(conditional_cdf(ga,population(a.problem,u,:crystals),3.:24.;window=C.WINDOW)-
        conditional_cdf(gb,population(b.problem,v,:crystals),3.:24.;window=C.WINDOW))) for (u,v) in zip(a.u,b.u))
    (;maximum_concentration_change_g_L=dc,maximum_CDF_change=dF)
end
function export_likelihood(folder,problem,z)
    values=only(predictions(problem,z));measurements=only(problem.experiments).measurements
    rows=NamedTuple[]
    for (block,(m,pred)) in enumerate(zip(measurements,values))
        for index in CartesianIndices(m.data)
            sigma=m.likelihood.sigma isa Number ? m.likelihood.sigma : m.likelihood.sigma[index]
            residual=(pred[index]-m.data[index])/sigma
            push!(rows,(block=block,component=index[1],time_min=m.times[index[2]],
                coordinate=block==1 ? "concentration_g_L" : "linearly_whitened_CDF",
                observed=m.data[index],predicted=pred[index],fixed_sigma=sigma,
                standardized_residual=residual,loglikelihood=-.5residual^2-log(sigma)-.5log(2pi)))
        end
    end
    csv(joinpath(folder,"likelihood_contributions.csv"),rows)
    abs(sum(r.loglikelihood for r in rows)-loglikelihood(problem,z))<1e-9 ||
        error("exported likelihood does not match the fitted target")
    C.original_predictions(problem,z)
end
function export_predictions(folder,problem,z,config,model)
    p=physical_from_z(problem,z);data=load_data()
    times=sort(unique(vcat(collect(0.:5.:660.),[r.model_time_min for r in data.concentration],[74.,180.,420.,660.])))
    solve_physical(p,config,model,[0.,1.]) # separate compilation warmup
    cost=@timed solve_physical(p,config,model,times)
    sol=cost.value;pb=sol.problem;g=pb.operator.system.populations.crystals.grid
    likelihood_predictions=export_likelihood(folder,problem,z)
    trajectory=map(zip(sol.t,sol.u)) do (t,u)
        merge((time_min=t,),state_diagnostics(sol,u,p,t))
    end
    csv(joinpath(folder,"trajectory.csv"),trajectory)
    at(t)=sol.u[searchsortedfirst(sol.t,t)]
    conc=[(id=r.observation_id,split=r.split,time_min=r.model_time_min,observed_g_L=r.concentration_g_L,
        sigma_g_L=.05r.concentration_g_L,predicted_g_L=bulk(pb,at(r.model_time_min),:c),
        residual_g_L=bulk(pb,at(r.model_time_min),:c)-r.concentration_g_L,
        standardized_residual=(bulk(pb,at(r.model_time_min),:c)-r.concentration_g_L)/(.05r.concentration_g_L)) for r in data.concentration]
    csv(joinpath(folder,"concentration_predictions.csv"),conc)
    psds=NamedTuple[];curves=NamedTuple[];qs=NamedTuple[];metrics=NamedTuple[]
    for rows in vcat(select_psds(data),select_psds(data,:validation))
        t=first(rows).time_min;u=at(t);N=population(pb,u,:crystals)
        x=[r.size_um for r in rows];F=[r.conditional_cdf for r in rows]
        pred=conditional_cdf(g,N,x;window=C.WINDOW)
        append!(psds,[(time_min=t,split=r.split,size_um=r.size_um,observed_cdf=r.conditional_cdf,
            predicted_cdf=pred[i],residual_cdf=pred[i]-r.conditional_cdf,
            observed_density_per_um=r.conditional_density_per_um) for (i,r) in enumerate(rows)])
        d=density_at(sol,u)
        append!(curves,[(time_min=t,left_um=a,right_um=b,conditional_density_per_um=f) for (a,b,f) in zip(d.left,d.right,d.density)])
        qo=cdf_quantiles(x,F,[.1,.5,.9]);qm=conditional_quantiles(g,N,[.1,.5,.9];window=C.WINDOW)
        append!(qs,[(time_min=t,split=first(rows).split,probability=v,observed_um=qo[i],predicted_um=qm[i],
            residual_um=qm[i]-qo[i]) for (i,v) in enumerate((.1,.5,.9))])
        push!(metrics,(time_min=t,split=first(rows).split,W2_um=C.tabulated_wasserstein(g,N,x,F;window=C.WINDOW),
            CDF_RMSE=sqrt(mean(abs2,pred-F)),CDF_max_abs=maximum(abs.(pred-F))))
    end
    csv(joinpath(folder,"psd_predictions.csv"),psds);csv(joinpath(folder,"distribution_curves.csv"),curves)
    csv(joinpath(folder,"quantile_predictions.csv"),qs);csv(joinpath(folder,"distribution_metrics.csv"),metrics)
    training=filter(r->r.split=="calibration",conc)
    concentration_change=maximum(abs.(likelihood_predictions[1][:]-[r.predicted_g_L for r in training]))
    cdf_change=maximum(maximum(abs.(conditional_cdf(g,population(pb,at(t),:crystals),3.:24.;window=C.WINDOW)-
        vec(likelihood_predictions[k+1]))) for (k,t) in enumerate((74.,180.,420.)))
    json(joinpath(folder,"observation_schedule_check.json"),(maximum_concentration_change_g_L=concentration_change,
        maximum_CDF_change=cdf_change,interpretation="dense figure saves versus the exact calibration solve schedule; same parameters and native controller"))
    json(joinpath(folder,"validation_metrics.json"),(evaluated_at=string(now(UTC)),
        calibration_concentration_RMSE_g_L=sqrt(mean(abs2,r.residual_g_L for r in conc if r.split=="calibration")),
        validation_concentration_RMSE_g_L=sqrt(mean(abs2,r.residual_g_L for r in conc if r.split=="validation")),
        maximum_material_error_g_L=maximum(abs(r.material_error_g_L) for r in trajectory),
        maximum_upper_bin_fraction=maximum(r.upper_bin_fraction for r in trajectory),
        solve_seconds=cost.time,allocated_bytes=cost.bytes,steps=sol.steps))
    sol
end
function moment_rate_checks(sol,p)
    pb=sol.problem;g=pb.operator.system.populations.crystals.grid
    map(zip(sol.t,sol.u)) do (t,u)
        du=zero(u);rhs!(du,u,pb.operator,p,t)
        N=population(pb,u,:crystals);dN=population(pb,du,:crystals)
        c=bulk(pb,u,:c);J,G=rates(p,t,c);D=p.grd_strength
        discrete=moment(g,dN,3)
        continuum=3G*moment(g,N,2)+6D*moment(g,N,1)
        (bins=length(g),time_min=t,discrete_M3_rate=discrete,continuum_M3_rate=continuum,
            relative_rate_difference=iszero(continuum) ? NaN : (discrete-continuum)/continuum,
            first_bin_birth_M3_rate=J*first(g.centers)^3,
            bulk_rate_g_L_min=bulk(pb,du,:c))
    end
end
function refinement(folder,problem,z,config,model)
    p=physical_from_z(problem,z);times=[74.,180.,420.,660.]
    base=solve_physical(p,config,model,times);rows=NamedTuple[]
    moments=moment_rate_checks(base,p)
    for (name,bins,Lmax,dt,cfl) in (("time",config["bins"],config["Lmax"],config["maximum_dt"]/2,config["cfl"]/2),
        ("size",2config["bins"],config["Lmax"],config["maximum_dt"],config["cfl"]),
        ("domain",3config["bins"]÷2,1.5config["Lmax"],config["maximum_dt"],config["cfl"]))
        fine=solve_physical(p,config,model,times;bins,Lmax,dt,cfl)
        push!(rows,merge((check=name,bins=bins,Lmax_um=Lmax,maximum_dt_min=dt,cfl=cfl),compare_forward(base,fine)))
        name=="size" && append!(moments,moment_rate_checks(fine,p))
    end
    csv(joinpath(folder,"refinement.csv"),rows)
    csv(joinpath(folder,"moment_rate_checks.csv"),moments)
    if model===:cnt_grd && config["bins"]==100
        medium=solve_physical(p,config,model,times;bins=200)
        fine=solve_physical(p,config,model,times;bins=400)
        csv(joinpath(folder,"additional_size_refinement.csv"),[
            merge((coarse_bins=200,fine_bins=400),compare_forward(medium,fine))])
    end
end
function refined_refit(folder,problem,z,config,model)
    fine=make_estimation(config,model;bins=2config["bins"])
    fitted=fit(fine;initial=z,maxiters=config["maxiters"],gradient_tolerance=1e-3,derivative_step=1e-7)
    base=C.original_predictions(problem,z);value=C.original_predictions(fine,fitted.minimizer)
    json(joinpath(folder,"refined_map.json"),(bins=2config["bins"],converged=fitted.converged,
        gradient_norm=fitted.diagnostics.gradient_norm,parameters=physical_from_z(fine,fitted.minimizer),
        calibration_concentration_prediction_change_g_L=maximum(abs.(value[1]-base[1])),
        calibration_CDF_prediction_change=maximum(maximum(abs.(value[k]-base[k])) for k in 2:4),
        interpretation="one-start calibration-only finer-grid refit; not the primary posterior or a validation refit"))
end
include(joinpath(ROOT,"examples","support","posterior_chain_aggregation.jl"))
include(joinpath(ROOT,"examples","support","isothermal_protein_sampling.jl"))
const COORDINATE_SOURCE_HASH=hashfile(joinpath(ROOT,"examples","support","isothermal_protein_sampling.jl"))
const DIAGNOSTIC_SOURCE_HASH=hashfile(joinpath(ROOT,"examples","support","posterior_chain_aggregation.jl"))
transport_path(config)=joinpath(CASE,get(config,"transport_file","sampling_transport.json"))
function sampling_map(problem,z,config)
    kind=get(config,"sampling_coordinates","initial_rates")
    kind=="initial_rates" && return C.sampling_coordinates(problem)
    kind=="soft_barrier_rates" &&
        return IsothermalProteinSampling.bounded_rate_coordinates(C,problem,z;
            growth_reference_supersaturation=get(config,"growth_reference_supersaturation",nothing))
    kind=="transport_rates" &&
        return IsothermalProteinSampling.transported_rate_coordinates(C,problem,z,readjson(transport_path(config));
            growth_reference_supersaturation=get(config,"growth_reference_supersaturation",nothing))
    error("unknown sampling coordinates: $kind")
end

function sample_chain(model,config,z,start,covariance,chain)
    problem=make_estimation(config,model)
    sample_posterior(problem;initial=z,starts=[start],sampling_map=sampling_map(problem,z,config),
        metric=:dense,initial_covariance=covariance,warmup=config["warmup"],samples=config["samples"],chains=1,
        seed=config["seed"]+100chain,target_acceptance=config["target_acceptance"],
        max_depth=config["max_depth"],verbose=true)
end
function sample_model(folder,problem,z,config)
    started=time_ns();model=any(s->s.name===:grd_strength,problem.parameters) ? :cnt_grd : :cnt
    n=length(z);map=sampling_map(problem,z,config)
    policy=get(config,"metric_initialization","expected_information")
    policy in ("expected_information","local_curvature","transport_empirical") || error("unknown sampler metric initialization: $policy")
    function tune(q)
        if policy=="transport_empirical"
            map.label===:CNT_transport_rates || error("transport tuning requires its matching coordinate map")
            settings=readjson(transport_path(config))
            covariance=reshape(Float64.(settings.initial_covariance),n,n)
            cholesky(Symmetric(covariance))
            information=cholesky(Symmetric(covariance))\Matrix{Float64}(I,n,n)
            return (;covariance,information,condition_number=cond(covariance),
                interpretation="Frozen empirical covariance in sampler coordinates from a failed calibration-only pilot; tuning only, never posterior uncertainty")
        end
        policy=="local_curvature" ? IsothermalProteinSampling.curvature_tuning(problem,q,map) :
        map.label===:CNT_soft_barrier_rates ?
            IsothermalProteinSampling.expected_information_tuning(problem,q,map) : C.sampler_tuning(problem,q,map)
    end
    tuning=tune(z)
    tunings=fill(tuning,config["chains"])
    centers=[copy(z) for _ in 1:config["chains"]]
    rng=MersenneTwister(config["seed"])
    if haskey(config,"secondary_start_inference")
        secondary=Float64.(config["secondary_start_inference"])
        length(secondary)==n || error("secondary calibration start dimension differs from ParameterSpecs")
        secondary_tuning=tune(secondary)
        for c in (config["chains"]÷2+1):config["chains"]
            centers[c]=secondary; tunings[c]=secondary_tuning
        end
    end
    covariances=[t.covariance for t in tunings]
    starts=[map.from_sampler(map.to_sampler(centers[c])+
        .3cholesky(Symmetric(covariances[c])).L*randn(rng,n)) for c in 1:config["chains"]]
    json(joinpath(folder,"sampler_tuning.json"),merge(tuning,
        (matrix_shape=(n,n),matrix_storage="column-major flat arrays",coordinates="sampler, not inference or physical",
         covariance_coordinates="sampler",information_coordinates=policy=="local_curvature" ? "inference" : "sampler",
         policy,chain_centers=centers,chain_tunings=tunings)))
    stopping_policy=get(config,"stopping","fixed")
    key=nothing
    chain_files=String[]
    post=if stopping_policy=="diagnostics"
        # Continue one set of chains jointly. No independent worker can decide
        # multi-chain convergence, and each chain keeps its adapted state/RNG.
        quantities=(
            log_J_at_S6=theta->theta.shared.logAJ-C.barrier(C.FIXED_GAMMA)/log(6.)^2,
            log_G_at_S6=theta->log(theta.shared.AG)+theta.shared.g*log(5.))
        sampled=sample_posterior(problem;initial=z,starts,chains=config["chains"],
            warmup=config["warmup"],seed=config["seed"],target_acceptance=config["target_acceptance"],
            max_depth=config["max_depth"],metric=:dense,initial_covariance=Tuple(covariances),
            sampling_map=map,stopping=:diagnostics,
            min_samples=config["min_samples"],batch_samples=config["batch_samples"],
            max_samples=config["max_samples"],consecutive_passes=config["consecutive_passes"],
            diagnostic_quantities=quantities,verbose=true)
        d=diagnostics(sampled)
        PosteriorSamples(sampled.z,sampled.physical,
            merge(d,(sum_chain_elapsed_seconds=d.elapsed_seconds,)),sampled.coordinates)
    elseif stopping_policy=="fixed"
        packets=Vector{Any}(undef,config["chains"]);pending=Int[]
        key=bytes2hex(sha256(JSON3.write((model_settings(config,model),string(VERSION),
            hashfile(joinpath(ROOT,"paper","Manifest.toml")),config["warmup"],config["samples"],config["chains"],config["seed"],
            config["target_acceptance"],config["max_depth"],map.label,
            get(config,"growth_reference_supersaturation",nothing),
            COORDINATE_SOURCE_HASH,get(config,"sampling_coordinates","")=="transport_rates" ? hashfile(transport_path(config)) : nothing,
            policy,covariances,starts,z,sort(collect(numerical_sources());by=first)))))[1:16]
        cache=joinpath(folder,"chains");mkpath(cache)
        paths=[joinpath(cache,"$(key)_$(c).bin") for c in 1:config["chains"]]
        for c in 1:config["chains"]
            stamp=joinpath(cache,"$(key)_$(c).json")
            if isfile(paths[c]) && isfile(stamp) && readjson(stamp).sha256==hashfile(paths[c])
                packets[c]=deserialize(paths[c])
            else
                push!(pending,c)
            end
        end
        input=joinpath(cache,"$(key)_input.bin")
        serialize(input,(model=model,config=config,z=z,starts=starts,covariances=covariances,paths=paths))
        # The established process-isolation protocol uses files, not local TCP
        # listeners. Every worker loads this same entry point and the same example.
        @sync for c in pending
            @async begin
                command=`$(Base.julia_cmd()) --threads=1 --project=$(joinpath(ROOT,"paper")) $(joinpath(CASE,"run.jl")) --chain-worker $input $c`
                logfile=joinpath(cache,"$(key)_$(c).log")
                println("Launching ",model," chain ",c,": warmup=",config["warmup"],
                    ", samples=",config["samples"],", target acceptance=",config["target_acceptance"]);flush(stdout)
                open(logfile,"w") do io
                    run(pipeline(command;stdout=io,stderr=io))
                end
                packets[c]=deserialize(paths[c])
                json(joinpath(cache,"$(key)_$(c).json"),(sha256=hashfile(paths[c]),chain=c,seed=config["seed"]+100c,
                    warmup=config["warmup"],samples=config["samples"],completed=true))
                println("Completed posterior chain ",model," ",c,"/",config["chains"]);flush(stdout)
            end
        end
        chain_files=["chains/$(key)_$(c).json" for c in 1:config["chains"]]
        _combine_protein_chains(problem,packets;elapsed_seconds=(time_ns()-started)/1e9,
            allow_distinct_tuning=true)
    else
        error("stopping must be fixed or diagnostics")
    end
    d=diagnostics(post)
    json(joinpath(folder,"sampling_diagnostics.json"),(passed=d.diagnostics_passed,ess_bulk=d.ess_bulk,
        ess_tail=d.ess_tail,rhat=d.rhat,mcse_mean=d.mcse_mean,mcse_coordinates="inference",ebfmi=d.ebfmi,divergences=d.divergences,max_depth_hits=d.max_depth_hits,
        per_chain=d.per_chain,warmup=d.warmup,samples=d.samples,chains=d.chains,seed=config["seed"],
        sampling_coordinates=d.sampling_coordinates,
        chain_seeds=d.chain_seeds,target_acceptance=d.target_acceptance,max_depth=d.max_depth,
        forward_solves=d.forward_solves,elapsed_seconds=(time_ns()-started)/1e9,summed_chain_seconds=d.sum_chain_elapsed_seconds,
        gradient_checks=d.gradient_checks,map_checks=d.map_checks,
        failed_evaluations=d.failed_evaluations,failure_messages=d.failure_messages,
        stopping_policy=d.stopping_policy,stopping_reason=d.stopping_reason,
        check_history=d.check_history,consecutive_passes_required=d.consecutive_passes_required,
        min_samples=d.min_samples,max_samples=d.max_samples,batch_samples=d.batch_samples,
        quantity_diagnostics=Base.map(q->Base.structdiff(q,(values=q.values,)),d.quantity_diagnostics)))
    csv(joinpath(folder,"posterior_draws.csv"),[merge((chain=c,draw=i),physical_parameters(problem,post.z[:,i,c],:batch),
        NamedTuple{Tuple(Symbol("z",k) for k in 1:n)}(Tuple(post.z[:,i,c]))) for c in 1:d.chains for i in 1:d.samples])
    csv(joinpath(folder,"sampler_statistics.csv"),[(chain=c,draw=i,logdensity_z=d.statistics[c][i].log_density,
        energy_sampler=d.statistics[c][i].hamiltonian_energy,leapfrog_steps=d.statistics[c][i].n_steps,
        divergence=d.statistics[c][i].numerical_error) for c in 1:d.chains for i in 1:d.samples])
    json(joinpath(folder,"sampling_provenance.json"),(producer_sha256=DRIVER_HASHES["report.jl"],recorded_at=string(now(UTC)),
        coordinate_source_sha256=COORDINATE_SOURCE_HASH,
        transport_configuration_sha256=get(config,"sampling_coordinates","")=="transport_rates" ? hashfile(transport_path(config)) : nothing,
        process_isolation=stopping_policy=="fixed",execution=stopping_policy=="fixed" ? "isolated fixed-length workers" : "serial joint diagnostic controller",
        stopping_policy,stopping_reason=d.stopping_reason,chain_seeds=d.chain_seeds,
        diagnostic_source_sha256=DIAGNOSTIC_SOURCE_HASH,chain_key=key,chain_files))
    post
end
function load_posterior(folder,problem)
    rows=C.table(joinpath(folder,"posterior_draws.csv"));d=readjson(joinpath(folder,"sampling_diagnostics.json"))
    d.passed || error("saved posterior did not pass diagnostics")
    n=length(problem.parameters);Z=zeros(n,d.samples,d.chains)
    for r in rows
        Z[:,Int(r.draw),Int(r.chain)]=[getproperty(r,Symbol("z",k)) for k in 1:n]
    end
    physical=[physical_parameters(problem,Z[:,i,c]) for i in 1:d.samples,c in 1:d.chains]
    PosteriorSamples(Z,physical,(diagnostics_passed=true,),ParameterSpec[problem.parameters...])
end
function verify_saved_target(root,model,config)
    folder=joinpath(root,string(model))
    manifest=readjson(joinpath(folder,"manifest.json"))
    manifest.complete && check_manifest(folder,manifest.target_hash) ||
        error("complete, hash-verified posterior required before --verify")
    all(hashfile(joinpath(ROOT,string(k)))==v for (k,v) in pairs(manifest.source_files)) ||
        error("numerical sources changed since the retained posterior")
    problem=make_estimation(config,model)
    expected=bytes2hex(sha256(JSON3.write((model_settings(config,model),sort(collect(numerical_sources());by=first)))))
    expected==manifest.target_hash ||
        error("verification settings differ from the retained target")
    raw=C.table(joinpath(folder,"posterior_draws.csv"))
    statistics=C.table(joinpath(folder,"sampler_statistics.csv"))
    n=length(problem.parameters);checks=NamedTuple[]
    reference=Float64.(readjson(joinpath(folder,"map.json")).z)
    map=sampling_map(problem,reference,config)
    for chain in sort(unique(r.chain for r in raw))
        indices=findall(r->r.chain==chain,raw)
        for j in unique([first(indices),last(indices)])
            r=raw[j];z=[getproperty(r,Symbol("z",i)) for i in 1:n]
            density=logposterior(problem,z)
            abs(density-statistics[j].logdensity_z)<1e-8+1e-10abs(density) ||
                error("saved inference-coordinate density does not match the active posterior")
            theta=physical_parameters(problem,z,:batch)
            all(isapprox(getproperty(theta,s.name),getproperty(r,s.name);rtol=1e-12) for s in problem.parameters) ||
                error("saved physical/inference coordinates disagree")
            derivative=check_gradient(problem,z)
            mapped=check_sampling_map(problem,map,z)
            derivative.valid && mapped.valid || error("posterior gradient verification failed at chain $chain, draw $(r.draw)")
            push!(checks,(chain=Int(chain),draw=Int(r.draw),logdensity_error=density-statistics[j].logdensity_z,
                gradient_relative_error=derivative.error,mapped_gradient_relative_error=mapped.error,
                map_inverse_error=mapped.inverse_error,log_jacobian=mapped.logjac))
        end
    end
    producer_receipt=joinpath(folder,"target_verification.json")
    first_verification=!isfile(producer_receipt)
    receipt=first_verification ? producer_receipt :
        joinpath(root,"analysis",string(model),"target_verification.json")
    mkpath(dirname(receipt))
    json(receipt,(passed=true,checked_at=string(now(UTC)),points=checks,
        target_hash=manifest.target_hash,posterior_draws_sha256=hashfile(joinpath(folder,"posterior_draws.csv")),
        sampler_statistics_sha256=hashfile(joinpath(folder,"sampler_statistics.csv")),
        interpretation="saved z densities and physical coordinates; AD and mapped gradients versus step-halved finite differences at retained draws"))
    # The first check completes the producer manifest. Later checks belong to
    # analysis and must preserve that manifest and its original receipt.
    first_verification && write_manifest(folder,manifest.target_hash,numerical_sources();complete=true)
    println("Verified retained posterior target ",model," at ",length(checks)," draw points")
end
function predict_draws(folder,problem,posterior,config,model)
    selected=parameter_draws(problem,posterior;n=config["predictive_draws"],seed=config["seed"]+1)
    csv(joinpath(folder,"latent_prediction_indices.csv"),[(prediction=i,raw_column=j) for (i,j) in enumerate(selected.indices)])
    times=sort(unique(vcat(collect(0.:5.:660.),[74.,180.,420.,660.])))
    predictions=NamedTuple[];densities=NamedTuple[];numerical=NamedTuple[];draw_diagnostics=NamedTuple[]
    for (i,physical) in enumerate(selected.physical)
        p=merge(parameters(),physical.shared)
        timing=@timed solve_physical(p,config,model,times)
        sol=timing.value;g=sol.problem.operator.system.populations.crystals.grid
        series=[state_diagnostics(sol,u,p,t) for (t,u) in zip(sol.t,sol.u)]
        push!(draw_diagnostics,(draw=i,steps=sol.steps,seconds=timing.time,
            maximum_material_error_g_L=maximum(abs(r.material_error_g_L) for r in series),
            minimum_count_per_L=minimum(r.minimum_count_per_L for r in series),
            maximum_upper_bin_fraction=maximum(r.upper_bin_fraction for r in series)))
        for (t,u,values) in zip(sol.t,sol.u,series)
            push!(predictions,merge((draw=i,time_min=t),values))
            if t in (74.,180.,420.,660.)
                d=density_at(sol,u)
                append!(densities,[(draw=i,time_min=t,left_um=a,right_um=b,conditional_density_per_um=f) for (a,b,f) in zip(d.left,d.right,d.density)])
            end
        end
        i%25==0 && (println("Latent propagation ",model," ",i,"/",length(selected.physical));flush(stdout))
    end
    csv(joinpath(folder,"latent_draw_diagnostics.csv"),draw_diagnostics)
    csv(joinpath(folder,"latent_predictions.csv"),predictions)
    csv(joinpath(folder,"latent_distribution_curves.csv"),densities)
    # Select posterior checks by rate/dispersion extremes, never held-out fit.
    p=[merge(parameters(),a.shared) for a in selected.physical]
    initialG=[last(rates(q,0.,C.C0)) for q in p]
    theta=[q.grd_strength for q in p]
    checks=unique([argmin(initialG),argmax(initialG),argmin(theta),argmax(theta)])
    ts=[74.,180.,420.,660.]
    for i in checks
        a=solve_physical(p[i],config,model,ts)
        for (name,bins,Lmax,dt,cfl) in (("size",2config["bins"],config["Lmax"],config["maximum_dt"],config["cfl"]),
            ("time",config["bins"],config["Lmax"],config["maximum_dt"]/2,config["cfl"]/2),
            ("domain",3config["bins"]÷2,1.5config["Lmax"],config["maximum_dt"],config["cfl"]))
            b=solve_physical(p[i],config,model,ts;bins,Lmax,dt,cfl)
            push!(numerical,merge((draw=i,check=name),compare_forward(a,b)))
        end
    end
    csv(joinpath(folder,"posterior_refinement.csv"),numerical)
end
function execute_model(root,model,config;smoke=false,sample=false,force=false)
    folder=joinpath(root,string(model));mkpath(folder)
    settings=model_settings(config,model);sources=numerical_sources()
    target_hash=bytes2hex(sha256(JSON3.write((settings,sort(collect(sources);by=first)))))
    valid=check_manifest(folder,target_hash) && same_environment(folder)
    sampling_settings=sample && !smoke ? protein_sampling_settings(config) : nothing
    if sampling_settings!==nothing
        sampling_settings["transport_sha256"]=get(config,"sampling_coordinates","")=="transport_rates" ? hashfile(transport_path(config)) : nothing
    end
    stored_path=joinpath(folder,"sampling_settings.json")
    stored_sampling=isfile(stored_path) ? readjson(stored_path) : nothing
    # This check precedes all writes and MAP/sampling work. Historical runs
    # require a fresh output directory or an explicit replacement request.
    protect_protein_posterior(folder;valid=valid && isfile(joinpath(folder,"map.json")),force,
        requested=sampling_settings,stored=stored_sampling)
    problem=make_estimation(config,model)
    json(joinpath(folder,"priors.json"),C.prior_metadata(problem))
    if smoke
        sol=solve_physical(parameters(grd_strength=model===:cnt ? 0. : C.D0_PRIOR_MEAN),config,model,[0.,74.])
        json(joinpath(folder,"smoke.json"),(model=model,calibrated=false,steps=sol.steps,minimum=minimum(sol.u[end])))
        return
    end
    if !valid || force || !isfile(joinpath(folder,"map.json"))
        println("MAP ",model," bins=",config["bins"]);flush(stdout)
        startp=parameters(logAJ=30.85+C.LOG_AJ_SHIFT,AG=.00048,g=2.89,grd_strength=model===:cnt ? 0. : C.D0_PRIOR_MEAN)
        start=initial_coordinates(EstimationProblem(C.parameter_specs(p=startp,model=model),problem.experiments;fixed=problem.fixed))
        fitted=C.fit_model(problem;maxiters=config["maxiters"],initial=start)
        json(joinpath(folder,"map.json"),map_export(problem,fitted))
        fitted.converged || error("MAP stationarity failed; diagnostics saved")
        # New target/MAP cannot inherit any old posterior files.
        for name in ("posterior_draws.csv","sampler_statistics.csv","sampling_diagnostics.json","latent_predictions.csv","latent_distribution_curves.csv","target_verification.json")
            path=joinpath(folder,name);isfile(path) && rm(path)
        end
    end
    maprecord=readjson(joinpath(folder,"map.json"));z=Float64.(maprecord.z)
    maprecord.converged || error("MAP did not pass stationarity")
    if !valid || force || !isfile(joinpath(folder,"frozen_model.json"))
        json(joinpath(folder,"frozen_model.json"),(model=model,parameters=physical_from_z(problem,z),
            frozen_at=string(now(UTC)),validation_used=false,target_hash,settings))
    end
    json(joinpath(folder,"run_settings.json"),settings)
    (!valid || force || !isfile(joinpath(folder,"provenance.json"))) && json(joinpath(folder,"provenance.json"),(revision=strip(read(`git -C $ROOT rev-parse HEAD`,String)),
        working_tree=read(`git -C $ROOT status --porcelain=v1 --untracked-files=all`,String),
        recorded_at=string(now(UTC)),julia=string(VERSION),cpu=Sys.cpu_info()[1].model,
        julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),target_hash,source_files=sources,
        dependencies=Dict(x.name=>string(x.version) for x in values(Pkg.dependencies()) if x.version!==nothing),
        driver_files=DRIVER_HASHES))
    json(joinpath(folder,"reporting_provenance.json"),(recorded_at=string(now(UTC)),driver_files=DRIVER_HASHES,
        target_hash,interpretation="current export/reporting pass; original solve/sampling provenance remains separate"))
    tuningfile=joinpath(folder,"sampler_tuning.json")
    if isfile(tuningfile) && !hasproperty(readjson(tuningfile),:matrix_shape)
        tuning=Dict(pairs(readjson(tuningfile)))
        tuning[:matrix_shape]=(length(z),length(z));tuning[:matrix_storage]="column-major flat arrays"
        tuning[:coordinates]="sampler, not inference or physical"
        json(tuningfile,tuning)
    end
    write_manifest(folder,target_hash,sources)
    if sample
        reuse=valid && isfile(joinpath(folder,"sampling_settings.json")) &&
            protein_sampling_matches(stored_sampling,sampling_settings) &&
            (get(config,"sampling_coordinates","initial_rates")=="initial_rates" ||
                (isfile(joinpath(folder,"sampling_provenance.json")) &&
                 readjson(joinpath(folder,"sampling_provenance.json")).coordinate_source_sha256==
                    hashfile(joinpath(ROOT,"examples","support","isothermal_protein_sampling.jl")))) &&
            isfile(joinpath(folder,"sampling_diagnostics.json")) &&
            readjson(joinpath(folder,"sampling_diagnostics.json")).passed
        posterior=if reuse
            load_posterior(folder,problem)
        else
            verification=joinpath(folder,"target_verification.json")
            isfile(verification) && rm(verification)
            json(joinpath(folder,"sampling_settings.json"),sampling_settings)
            println("NUTS ",model);flush(stdout)
            sampled=sample_model(folder,problem,z,config)
            write_manifest(folder,target_hash,sources)
            diagnostics(sampled).diagnostics_passed || error("NUTS screen failed; raw results retained, no credible bands")
            sampled
        end
        latent_settings=(draws=config["predictive_draws"],seed=config["seed"]+1,export_version=2)
        if !reuse || !isfile(joinpath(folder,"latent_settings.json")) ||
                readjson(joinpath(folder,"latent_settings.json")).draws!=config["predictive_draws"] ||
                get(readjson(joinpath(folder,"latent_settings.json")),:export_version,0)!=2 ||
                (!isfile(joinpath(folder,"latent_distribution_curves.csv")) || !isfile(joinpath(folder,"latent_draw_diagnostics.csv")))
            predict_draws(folder,problem,posterior,config,model)
            json(joinpath(folder,"latent_settings.json"),latent_settings)
        end
    end
    export_predictions(folder,problem,z,config,model)
    refinement(folder,problem,z,config,model)
    if !valid || force || !isfile(joinpath(folder,"refined_map.json"))
        refined_refit(folder,problem,z,config,model)
    end
    complete=isfile(joinpath(folder,"latent_predictions.csv")) && readjson(joinpath(folder,"sampling_diagnostics.json")).passed
    write_manifest(folder,target_hash,sources;complete)
    complete && verify_saved_target(root,model,config)
    println("Finished ",model," posterior_complete=",complete);flush(stdout)
end
