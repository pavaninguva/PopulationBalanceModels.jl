# One reproduction entry point. Model and inference definitions live in the example.
using PopulationBalanceModels, Optim, JSON3, SHA, Dates, LinearAlgebra, Statistics, Random
include(joinpath(@__DIR__,"..","..","..","examples","isothermal_protein_crystallization.jl"))
using .IsothermalProteinCrystallization
const ROOT=realpath(joinpath(@__DIR__,"..","..",".."))
const SMOKE="--smoke" in ARGS
const OUTPUT=joinpath(ROOT,"paper","results","04_protein_crystallization",SMOKE ? "smoke" : "production")
mkpath(OUTPUT)
# Posterior exports from another run must not survive a MAP-only execution.
for name in ("posterior_draws.csv","sampling_diagnostics.json","sampler_statistics.csv",
             "latent_predictions.csv","latent_prediction_indices.csv","posterior_refinement.csv","sampler_tuning.json","manifest.json")
    path=joinpath(OUTPUT,name)
    isfile(path) && rm(path)
end
normpath(dirname(dirname(pathof(PopulationBalanceModels))))==ROOT || error("load the checkout, not a registry release")

function json(path,value)
    open(path,"w") do io
        JSON3.pretty(io,value)
        println(io)
    end
end
function csv(path,rows)
    isempty(rows) && return
    quote_field(x)=occursin(r"[,\"\n]",string(x)) ? "\""*replace(string(x),"\""=>"\"\"")*"\"" : string(x)
    open(path,"w") do io
        println(io,join(keys(first(rows)),','))
        for row in rows
            println(io,join(quote_field.(values(row)),','))
        end
    end
end
hashfile(path)=bytes2hex(sha256(read(path)))
source_paths=vcat(["examples/isothermal_protein_crystallization.jl",
    "paper/case_studies/04_protein_crystallization/run.jl","paper/Project.toml","paper/Manifest.toml"],
    [relpath(joinpath(dir,f),ROOT) for folder in ("src","ext") for (dir,_,files) in walkdir(joinpath(ROOT,folder)) for f in files if endswith(f,".jl")],
    [relpath(joinpath(IsothermalProteinCrystallization.DATA,f),ROOT) for f in ("concentration.csv","psd_cdf.csv","psd_summary.csv")])
source_files=Dict(path=>hashfile(joinpath(ROOT,path)) for path in sort(source_paths))
target_hash=bytes2hex(sha256(JSON3.write(sort(collect(source_files);by=first))))
provenance=(revision=strip(read(`git -C $ROOT rev-parse HEAD`,String)),
    working_tree=read(`git -C $ROOT status --porcelain=v1 --untracked-files=all`,String),
    recorded_at=string(now(UTC)),julia=string(VERSION),cpu=Sys.cpu_info()[1].model,
    julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),target_hash,
    data="Original Tang 2018 Fig 11d/12a; unchanged supplied readings",source_files)
json(joinpath(OUTPUT,"provenance.json"),provenance)
if SMOKE
    sol=minimal_run()
    json(joinpath(OUTPUT,"smoke.json"),(model="CNT + power-law growth",calibrated=false,steps=sol.steps))
    exit()
end

# Freeze calibration definitions before inspecting any validation measurements.
const BINS=200
problem=estimation_problem(bins=BINS)
println("CNT Bayesian MAP: three starts, ",BINS," bins");flush(stdout)
fitted=IsothermalProteinCrystallization.fit_model(problem)
p=merge(parameters(),physical_parameters(problem,fitted.minimizer,:batch))
json(joinpath(OUTPUT,"map.json"),(z=fitted.minimizer,parameters=p,converged=fitted.converged,
    loglikelihood=fitted.loglikelihood,logprior=fitted.logprior,logposterior=fitted.logposterior,
    likelihood_coordinates="concentration and linearly whitened CDF ordinates",
    original_CDF_loglikelihood_offset=-sum(sum(log,diag(IsothermalProteinCrystallization.cdf_factor(m.observable.points))) for m in only(problem.experiments).measurements[2:end]),
    gradient=fitted.diagnostics.gradient,gradient_norm=fitted.diagnostics.gradient_norm,
    elapsed_seconds=fitted.diagnostics.elapsed_seconds,forward_solves=fitted.diagnostics.forward_solves,
    failures=fitted.diagnostics.failed_evaluations,failure_messages=fitted.diagnostics.failure_messages,
    starts=[(converged=r.converged,objective=r.result===nothing ? nothing : Optim.minimum(r.result)) for r in fitted.diagnostics.starts]))
fitted.converged || error("MAP stationarity failed; results retained, no final predictions")
println("MAP: ",p);flush(stdout)

# Verification is allowed to reject an attractive-looking uncertainty estimate.
laplace=nothing
try
    global laplace=laplace_approximation(problem,fitted)
    json(joinpath(OUTPUT,"laplace.json"),(accepted=true,diagnostics=laplace.diagnostics,covariance=laplace.covariance))
catch e
    e isa LaplaceFailure || rethrow()
    json(joinpath(OUTPUT,"laplace.json"),(accepted=false,diagnostics=e.report))
end

posterior=nothing
if "--sample" in ARGS
    using AdvancedHMC, MCMCDiagnosticTools
    sampler_map=IsothermalProteinCrystallization.sampling_coordinates(problem)
    rng=MersenneTwister(20261023)
    center=sampler_map.to_sampler(fitted.minimizer)
    tuning=IsothermalProteinCrystallization.sampler_tuning(problem,fitted.minimizer,sampler_map)
    json(joinpath(OUTPUT,"sampler_tuning.json"),tuning)
    factor=cholesky(Symmetric(tuning.covariance)).L
    starts=[sampler_map.from_sampler(center.+.3factor*randn(rng,6)) for _ in 1:4]
    println("NUTS: four dispersed chains in initial-rate coordinates");flush(stdout)
    global posterior=sample_posterior(problem;initial=fitted,starts,sampling_map=sampler_map,
        metric=:dense,initial_covariance=tuning.covariance,
        warmup=parse(Int,get(ENV,"PBM_PROTEIN_WARMUP","300")),
        samples=parse(Int,get(ENV,"PBM_PROTEIN_SAMPLES","500")),chains=4,seed=20261023,
        target_acceptance=.9,max_depth=8,verbose=true)
    d=diagnostics(posterior)
    json(joinpath(OUTPUT,"sampling_diagnostics.json"),(passed=d.diagnostics_passed,
        ess_bulk=d.ess_bulk,ess_tail=d.ess_tail,rhat=d.rhat,ebfmi=d.ebfmi,
        divergences=d.divergences,max_depth_hits=d.max_depth_hits,
        per_chain=d.per_chain,warmup=d.warmup,samples=d.samples,chains=d.chains,
        forward_solves=d.forward_solves,elapsed_seconds=d.elapsed_seconds,
        failed_evaluations=d.failed_evaluations,failure_messages=d.failure_messages,
        gradient_checks=d.gradient_checks,map_checks=d.map_checks,
        metric="adapted dense; expected-information initialization is tuning, not posterior covariance",
        coordinate_map="initial log J and log G; Jacobian of inverse map retained"))
    csv(joinpath(OUTPUT,"posterior_draws.csv"),[
        merge((chain=c,draw=i),physical_parameters(problem,posterior.z[:,i,c],:batch),
            NamedTuple{Tuple(Symbol("z",k) for k in 1:6)}(Tuple(posterior.z[:,i,c])))
        for c in axes(posterior.z,3) for i in axes(posterior.z,2)])
    csv(joinpath(OUTPUT,"sampler_statistics.csv"),[
        (chain=c,draw=i,logdensity_z=d.statistics[c][i].log_density,
         energy_sampler=d.statistics[c][i].hamiltonian_energy,
         leapfrog_steps=d.statistics[c][i].n_steps,divergence=d.statistics[c][i].numerical_error)
        for c in 1:d.chains for i in 1:d.samples])
end

# Forward prediction is frozen here; held-out data have never entered fitting.
json(joinpath(OUTPUT,"frozen_model.json"),(model="CNT + empirical growth (Mitchell Case 3)",
    parameters=p,bins=BINS,validation_used=false,frozen_at=string(now(UTC)),target_hash))
data=load_data()
allpsds=vcat(select_psds(data),select_psds(data,:validation))
times=sort(unique(vcat(collect(0.:10.:660.),[r.model_time_min for r in data.concentration],[74.,180.,420.,660.])))
forward(p;bins=BINS,times=[0.,74.]) # separate compilation warmup
cost=@timed forward(p;bins=BINS,times)
sol=cost.value
pb=sol.problem;grid=pb.operator.system.populations.crystals.grid
state(t)=sol.u[searchsortedfirst(sol.t,t)]
concentration(t)=bulk(pb,state(t),:c)
window=(2.5,25.)
function quantiles(u,probabilities)
    N=population(pb,u,:crystals)
    any(N.>0) || return fill(NaN,length(probabilities))
    try
        conditional_quantiles(grid,N,probabilities;window)
    catch e
        e isa NumericalFailure || rethrow()
        fill(NaN,length(probabilities)) # reporting only: empty windows are undefined
    end
end
trajectory=map(zip(sol.t,sol.u)) do (t,u)
    N=population(pb,u,:crystals);c=bulk(pb,u,:c);M3=moment(grid,N,3)
    q=quantiles(u,[.1,.5,.9])
    (time_min=t,concentration_g_L=c,number_per_m3=1e12sum(N),M3_scaled_um3=M3,
     material_error_g_L=c+1e-6*p.rho*p.kv*M3-p.c0,
     minimum_scaled_count=minimum(N),upper_bin_fraction=sum(N)>0 ? N[end]/sum(N) : 0.,
     D10_um=q[1],D50_um=q[2],D90_um=q[3])
end
csv(joinpath(OUTPUT,"trajectory.csv"),trajectory)
conc=[(id=r.observation_id,split=r.split,time_min=r.model_time_min,
    observed_g_L=r.concentration_g_L,predicted_g_L=concentration(r.model_time_min),
    residual_g_L=concentration(r.model_time_min)-r.concentration_g_L) for r in data.concentration]
csv(joinpath(OUTPUT,"concentration_predictions.csv"),conc)
psd_rows=NamedTuple[];curves=NamedTuple[];qs=NamedTuple[];distribution_metrics=NamedTuple[]
for rows in allpsds
    t=first(rows).time_min;N=population(pb,state(t),:crystals)
    xs=[r.size_um for r in rows];Fo=[r.conditional_cdf for r in rows]
    Fm=conditional_cdf(grid,N,xs;window)
    append!(psd_rows,[(time_min=t,split=r.split,size_um=r.size_um,observed_cdf=r.conditional_cdf,
        predicted_cdf=Fm[i],observed_density_per_um=r.conditional_density_per_um) for (i,r) in enumerate(rows)])
    norm=sum(N[i]*max(0.,min(grid.edges[i+1],window[2])-max(grid.edges[i],window[1]))/grid.widths[i] for i in eachindex(N))
    for i in eachindex(N)
        lo=max(grid.edges[i],window[1]);hi=min(grid.edges[i+1],window[2]);hi>lo || continue
        push!(curves,(time_min=t,left_um=lo,right_um=hi,conditional_density_per_um=N[i]/grid.widths[i]/norm))
    end
    probs=[.1,.5,.9];qm=conditional_quantiles(grid,N,probs;window);qo=cdf_quantiles(xs,Fo,probs)
    append!(qs,[(time_min=t,split=first(rows).split,probability=v,observed_um=qo[i],predicted_um=qm[i]) for (i,v) in enumerate(probs)])
    W2=IsothermalProteinCrystallization.tabulated_wasserstein(grid,N,xs,Fo;window)
    midpoint=map((2048,4096,8192)) do count
        u=((1:count).-0.5)./count
        sqrt(mean(abs2,conditional_quantiles(grid,N,u;window)-cdf_quantiles(xs,Fo,u)))
    end
    push!(distribution_metrics,(time_min=t,split=first(rows).split,W2_um=W2,
        CDF_RMSE=sqrt(mean(abs2,Fm-Fo)),W2_midpoint_2048_um=midpoint[1],
        W2_midpoint_4096_um=midpoint[2],W2_midpoint_8192_um=midpoint[3]))
end
csv(joinpath(OUTPUT,"psd_predictions.csv"),psd_rows)
csv(joinpath(OUTPUT,"distribution_curves.csv"),curves)
csv(joinpath(OUTPUT,"quantile_predictions.csv"),qs)
csv(joinpath(OUTPUT,"distribution_metrics.csv"),distribution_metrics)

# Independent time, size and domain checks; compare before any uncertainty claim.
base=IsothermalProteinCrystallization.original_predictions(problem,fitted.minimizer)
checks=NamedTuple[]
for (name,bins,Lmax,dt) in (("time",BINS,50.,.1),("size",2BINS,50.,2.),("domain",3BINS÷2,75.,2.))
    fine=estimation_problem(;bins,Lmax,dt)
    result=IsothermalProteinCrystallization.original_predictions(fine,fitted.minimizer)
    push!(checks,(check=name,bins=bins,Lmax_um=Lmax,maximum_dt_min=dt,
        maximum_concentration_change_g_L=maximum(abs.(result[1]-base[1])),
        maximum_CDF_change=maximum(maximum(abs.(result[k]-base[k])) for k in 2:4)))
end
csv(joinpath(OUTPUT,"refinement.csv"),checks)
println("Calibration-only size-refined MAP check");flush(stdout)
fine_problem=estimation_problem(bins=2BINS)
fine_map=fit(fine_problem;initial=fitted.minimizer.+[.001,.001,.001,.01,0.,0.],
    maxiters=300,derivative_step=1e-7)
fine_predictions=IsothermalProteinCrystallization.original_predictions(fine_problem,fine_map.minimizer)
json(joinpath(OUTPUT,"refined_map.json"),(bins=2BINS,converged=fine_map.converged,
    parameters=physical_parameters(fine_problem,fine_map.minimizer,:batch),
    logposterior=fine_map.logposterior,gradient_norm=fine_map.diagnostics.gradient_norm,
    concentration_prediction_change_g_L=maximum(abs.(fine_predictions[1]-base[1])),
    CDF_prediction_change=maximum(maximum(abs.(fine_predictions[k]-base[k])) for k in 2:4),
    used_to_refit_primary_validation=false))

metrics=(calibration_concentration_RMSE_g_L=sqrt(mean(abs2,r.residual_g_L for r in conc if r.split=="calibration")),
    validation_concentration_RMSE_g_L=sqrt(mean(abs2,r.residual_g_L for r in conc if r.split=="validation")),
    validation_W2_um=last(distribution_metrics).W2_um,
    maximum_material_error_g_L=maximum(abs(r.material_error_g_L) for r in trajectory),
    maximum_upper_bin_fraction=maximum(r.upper_bin_fraction for r in trajectory))
json(joinpath(OUTPUT,"run_settings.json"),(model="CNT + power-law growth",bins=BINS,Lmax_um=50.,
    cstar_g_L=p.cstar,temperature_K=IsothermalProteinCrystallization.TEMPERATURE,
    molecular_volume_m3=IsothermalProteinCrystallization.MOLECULAR_VOLUME,
    density_kg_m3=p.rho,shape_factor=p.kv,population_scale_per_m3=1e12,
    dt_cap_min=2.,CFL=.35,interval_step_budget=200000,
    integrator="SSPRK33",reconstruction="MUSCL",
    likelihood="Gaussian concentrations; correlated conditional number CDF blocks, 1 um ordinates, 2 um correlation length",
    uncertainty=posterior===nothing ? "MAP only; see Laplace verification" : "retained NUTS draws; inspect diagnostic screen",
    metrics,solve_seconds=cost.time,allocated_bytes=cost.bytes,steps=sol.steps))
json(joinpath(OUTPUT,"validation_metrics.json"),merge(metrics,(evaluated_at=string(now(UTC)),)))
csv(joinpath(OUTPUT,"fitted_parameters.csv"),[(name=s.name,MAP=getproperty(p,s.name),
    prior_coordinates=s.prior.coordinates,prior=string(s.prior.distribution)) for s in problem.parameters])
# Only verified samples may be used for posterior predictive intervals.
if posterior!==nothing && diagnostics(posterior).diagnostics_passed
    selected=parameter_draws(problem,posterior;n=100,seed=20261024)
    csv(joinpath(OUTPUT,"latent_prediction_indices.csv"),[(prediction=i,raw_column=j)
        for (i,j) in enumerate(selected.indices)])
    prediction_rows=NamedTuple[]
    for (i,physical) in enumerate(selected.physical)
        drawp=merge(parameters(),physical.shared)
        drawsol=forward(drawp;bins=BINS,times=times)
        for (t,u) in zip(drawsol.t,drawsol.u)
            push!(prediction_rows,(draw=i,time_min=t,concentration_g_L=bulk(drawsol.problem,u,:c)))
        end
    end
    csv(joinpath(OUTPUT,"latent_predictions.csv"),prediction_rows)
    # Extreme initial-growth and barrier draws also exercise the numerical domain.
    physical=[merge(parameters(),q.shared) for q in selected.physical]
    G=[last(rates(q,0.,q.c0)) for q in physical]
    gamma=[q.gamma for q in physical]
    indices=unique([argmin(G),argmax(G),argmin(gamma),argmax(gamma)])
    numerical=NamedTuple[]
    ts=[74.,180.,420.,660.]
    for i in indices
        a=forward(physical[i];bins=BINS,times=ts)
        b=forward(physical[i];bins=2BINS,times=ts)
        c=forward(physical[i];bins=3BINS÷2,Lmax=75.,times=ts)
        function differences(other)
            g=other.problem.operator.system.populations.crystals.grid
            delta_c=maximum(abs(bulk(a.problem,u,:c)-bulk(other.problem,v,:c)) for (u,v) in zip(a.u,other.u))
            delta_F=maximum(maximum(abs.(conditional_cdf(grid,population(a.problem,u,:crystals),3.:1.:24.;window)-
                conditional_cdf(g,population(other.problem,v,:crystals),3.:1.:24.;window))) for (u,v) in zip(a.u,other.u))
            (;delta_c,delta_F)
        end
        for (name,other) in (("size",b),("domain",c))
            delta=differences(other)
            push!(numerical,(predictive_draw=i,check=name,maximum_concentration_change_g_L=delta.delta_c,
                maximum_CDF_change=delta.delta_F,
                upper_bin_fraction=maximum(population(a.problem,u,:crystals)[end]/sum(population(a.problem,u,:crystals)) for u in a.u)))
        end
    end
    csv(joinpath(OUTPUT,"posterior_refinement.csv"),numerical)
end
files=Dict(f=>hashfile(joinpath(OUTPUT,f)) for f in readdir(OUTPUT) if any(ext->endswith(f,ext),(".json",".csv")) && f!="manifest.json")
json(joinpath(OUTPUT,"manifest.json"),(mode="production",complete=true,target_hash,source_files,files))
println("Completed CNT fits and frozen prediction: ",metrics);flush(stdout)
