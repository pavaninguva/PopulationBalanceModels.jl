# Read-only analysis of the retained fixed-error targets. Included by run.jl.
# All new records live outside production/; execute_model is never called.
const ANALYSIS_ROOT=joinpath(CASE,"data","analysis")
const REVIEWED_PRODUCER="80a7d7658a8e3babacdd57e7a1375d36e0622e79"
const REVIEWED_CURRENT="a38d582bfe2e7cca7eb6f40b57ecfb00a4957633"
const STATIONARY_REFACTORS=Dict(
    "src/PopulationBalanceModels.jl"=>"stationary exports/includes; transient definitions unchanged",
    "src/coupling/systems.jl"=>"factored identical population/bulk packing into _pack_system",
    "src/coupling/accounting.jl"=>"factored rate-tree formatting; exchange and loss fluxes unchanged",
    "src/compilation/axial_system.jl"=>"factored axial assembly; isothermal batch does not use it")
const STATIONARY_ADDITIONS=Set(("src/stationary/problems.jl","src/stationary/continuation.jl",
    "ext/PopulationBalanceModelsNonlinearSolveExt.jl"))

plain(x)=JSON3.read(JSON3.write(json_safe(x)),Dict{String,Any})
matrix_rows(x)=[collect(row) for row in eachrow(x)]
function retained_hashes(root)
    hashes=Dict{String,String}()
    for (dir,subdirectories,files) in walkdir(root)
        # Derived analyses now sit beside the frozen model inputs in data/.
        dir==root && filter!(!=("analysis"),subdirectories)
        for file in files
            path=joinpath(dir,file)
            hashes[relpath(path,ROOT)]=hashfile(path)
        end
    end
    hashes
end
function check_readonly(root,before)
    after=retained_hashes(root)
    before==after || error("Read-only protein analysis changed retained production files")
end

function source_review(manifest)
    current=numerical_sources();changes=NamedTuple[]
    extension_path=joinpath(@__DIR__,"uncertainty_compatibility.json")
    extension=isfile(extension_path) ? readjson(extension_path) : nothing
    extension_sources=extension===nothing ? Dict{String,Any}() : Dict(string(k)=>v for (k,v) in pairs(extension.sources))
    old=Dict(string(k)=>string(v) for (k,v) in pairs(manifest.source_files))
    for (path,digest) in old
        haskey(current,path) || error("Missing original target source $path")
        current[path]==digest && continue
        haskey(STATIONARY_REFACTORS,path) || haskey(extension_sources,path) || error("Unreviewed numerical source change: $path")
        original=read(`git -C $ROOT show $(REVIEWED_PRODUCER*":"*path)`)
        bytes2hex(sha256(original))==digest || error("Producer blob not verified for $path")
        if haskey(extension_sources,path)
            entry=extension_sources[path]
            current[path]==entry.sha256 || error("Uncertainty source changed after review: $path")
            baseline=read(`git -C $ROOT show $(REVIEWED_CURRENT*":"*path)`)
            bytes2hex(sha256(baseline))==entry.baseline_sha256 || error("Uncertainty review baseline differs for $path")
        else
            reviewed=read(`git -C $ROOT show $(REVIEWED_CURRENT*":"*path)`)
            bytes2hex(sha256(reviewed))==current[path] || error("Source changed after the stationary refactor review: $path")
        end
        push!(changes,(path,original_sha256=digest,current_sha256=current[path],
            review=haskey(extension_sources,path) ? extension_sources[path].review : STATIONARY_REFACTORS[path]))
    end
    added=setdiff(Set(keys(current)),Set(keys(old)))
    allowed=union(STATIONARY_ADDITIONS,Set(keys(extension_sources)))
    added ⊆ allowed || error("Unreviewed added numerical sources: $(setdiff(added,allowed))")
    for path in added
        if haskey(extension_sources,path)
            current[path]==extension_sources[path].sha256 || error("Added uncertainty source changed after review: $path")
        else
            reviewed=read(`git -C $ROOT show $(REVIEWED_CURRENT*":"*path)`)
            bytes2hex(sha256(reviewed))==current[path] || error("Added stationary source changed after review: $path")
        end
    end
    original_project=TOML.parse(read(`git -C $ROOT show $(REVIEWED_PRODUCER*":Project.toml")`,String))
    project=TOML.parsefile(joinpath(ROOT,"Project.toml"))
    project["deps"]==original_project["deps"] || error("Required dependencies changed; review before reusing draws")
    # The core-CI correction adds Statistics to tests only. Check the complete
    # expected change, retaining all runtime dependencies/compatibility exactly.
    baseline_project=read(`git -C $ROOT show $(REVIEWED_CURRENT*":Project.toml")`)
    expected=TOML.parse(String(copy(baseline_project)))
    test_project_review=if project==expected
        nothing
    else
        expected["extras"]["Statistics"]="10745b16-79ce-11e8-11f9-7d13ad32a3b2"
        expected["compat"]["Statistics"]="1"
        push!(expected["targets"]["test"],"Statistics")
        project==expected || error("Unreviewed Project.toml change beyond the test-only Statistics declaration")
        (baseline_sha256=bytes2hex(sha256(baseline_project)),
         current_sha256=hashfile(joinpath(ROOT,"Project.toml")),
         change="Statistics added to test extras/target and test compatibility only; complete remaining project unchanged")
    end
    (;original_sources=old,current_sources=current,changes,additions=sort!(collect(added)),
      reviewed_source_revision=REVIEWED_PRODUCER,reviewed_current_revision=REVIEWED_CURRENT,
      test_project_review,
      uncertainty_review_sha256=extension===nothing ? nothing : hashfile(extension_path))
end

function compatible_target(root,model,config)
    folder=joinpath(root,string(model));out=joinpath(ANALYSIS_ROOT,string(model));mkpath(out)
    manifest=readjson(joinpath(folder,"manifest.json"))
    manifest.complete && check_manifest(folder,manifest.target_hash) || error("Retained posterior/file hashes failed")
    current=numerical_sources()
    original=Dict(string(k)=>string(v) for (k,v) in pairs(manifest.source_files))
    review=original==current ? (original_sources=original,current_sources=current,
        changes=NamedTuple[],additions=String[],interpretation="exact current-source producer match") : source_review(manifest)
    settings=model_settings(config,model)
    plain(settings)==JSON3.read(read(joinpath(folder,"run_settings.json"),String),Dict{String,Any}) ||
        error("Numerical/physical settings differ from the retained target")
    problem=make_estimation(config,model)
    JSON3.read(JSON3.write(C.prior_metadata(problem)))==readjson(joinpath(folder,"priors.json")) ||
        error("Parameter order, priors or transformations differ")
    shared=readjson(joinpath(root,"shared","observation_model.json"))
    shared.concentration_relative_sd==.05 && shared.CDF_absolute_sd==C.SIGMA_F &&
        shared.CDF_correlation_length_um==C.CORRELATION_LENGTH &&
        collect(shared.window_um)==collect(C.WINDOW) && !shared.quantiles_in_likelihood || error("Observation model changed")
    for (f,h) in pairs(shared.data_hashes)
        hashfile(joinpath(C.DATA,string(f)))==h || error("Changed observations: $f")
    end
    diag=readjson(joinpath(folder,"sampling_diagnostics.json"))
    diag.passed || error("Retained MCMC did not pass its screen")
    saved=readjson(joinpath(folder,"map.json"));z=Float64.(saved.z)
    parts=posterior_components(problem,z);g=posterior_gradient(problem,z)
    abs(parts.logposterior-saved.logposterior)<1e-6 &&
        abs(parts.loglikelihood-saved.loglikelihood_whitened_CDF)<1e-6 &&
        abs(parts.logprior-saved.logprior)<1e-9 || error("Saved MAP posterior components changed")
    norm(g-Float64.(saved.gradient),Inf)<1e-4 || error("Saved MAP gradient changed")
    checks=NamedTuple[]
    push!(checks,(kind="MAP",chain=0,draw=0,logdensity_error=parts.logposterior-saved.logposterior,
        gradient_error=norm(g-Float64.(saved.gradient),Inf)))
    rows=C.table(joinpath(folder,"posterior_draws.csv"));stats=C.table(joinpath(folder,"sampler_statistics.csv"))
    n=length(problem.parameters)
    coordinates(r)=[Float64(getproperty(r,Symbol("z",k))) for k in 1:n]
    selected=Int[]
    for chain in sort(unique(r.chain for r in rows))
        indices=findall(r->r.chain==chain,rows);append!(selected,(first(indices),last(indices)))
    end
    for k in 1:n
        values=[coordinates(r)[k] for r in rows]
        append!(selected,(argmin(values),argmax(values)))
    end
    for index in unique(selected)
        r=rows[index];zz=coordinates(r)
        stat=only(filter(s->s.chain==r.chain && s.draw==r.draw,stats))
        density=logposterior(problem,zz)
        abs(density-stat.logdensity_z)<1e-6 || error("Saved posterior density changed at draw $index")
        theta=physical_parameters(problem,zz,:batch)
        all(isapprox(getproperty(theta,s.name),getproperty(r,s.name);rtol=1e-12) for s in problem.parameters) ||
            error("Physical/inference coordinates disagree")
        gradient=check_gradient(problem,zz;step=1e-6)
        gradient.valid || error("Gradient verification failed at chain $(r.chain), draw $(r.draw): $(gradient.error), $(gradient.halving)")
        push!(checks,(kind="retained",chain=Int(r.chain),draw=Int(r.draw),
            logdensity_error=density-stat.logdensity_z,gradient_error=gradient.error))
    end
    inputs=Dict(f=>hashfile(joinpath(folder,f)) for f in ("manifest.json","map.json",
        "posterior_draws.csv","sampler_statistics.csv","priors.json","run_settings.json","target_verification.json"))
    projects=Dict(path=>hashfile(joinpath(ROOT,path)) for path in ("Project.toml","paper/Project.toml","paper/Manifest.toml"))
    producer=readjson(joinpath(folder,"provenance.json"))
    dependencies=Dict(x.name=>string(x.version) for x in values(Pkg.dependencies()) if x.version!==nothing)
    differences=Dict(k=>(producer=get(producer.dependencies,Symbol(k),nothing),current=v)
        for (k,v) in dependencies if get(producer.dependencies,Symbol(k),nothing)!=v)
    receipt=(schema=1,passed=true,model=string(model),checked_at=string(now(UTC)),
        original_target_hash=manifest.target_hash,inputs,projects,
        original_producer_revision=producer.revision,current_revision=strip(read(`git -C $ROOT rev-parse HEAD`,String)),
        working_tree=read(`git -C $ROOT status --porcelain=v1 --untracked-files=all`,String),
        source_review=review,settings,checks,julia=string(VERSION),dependencies,dependency_differences=differences,
        parameter_names=string.(getproperty.(problem.parameters,:name)),
        likelihood="unchanged calibration-only 11 concentration + 3 correlated CDF blocks; full normalized posterior in original z, with prior Jacobians",
        interpretation="compatibility evidence for these retained targets and checked points, not replacement producer provenance or proof for arbitrary source changes")
    receipt_path=joinpath(out,"compatibility.json")
    if isfile(receipt_path)
        previous=readjson(receipt_path)
        same=previous.passed && previous.original_target_hash==receipt.original_target_hash &&
            plain(previous.source_review)==plain(receipt.source_review) &&
            plain(previous.inputs)==plain(receipt.inputs) && plain(previous.projects)==plain(receipt.projects) &&
            plain(previous.settings)==plain(receipt.settings) && plain(previous.dependencies)==plain(receipt.dependencies)
        if same
            # A repeated check of the identical target must not invalidate an
            # existing approximation solely because its timestamp changed.
            receipt=previous
        else
            history=joinpath(out,"history");mkpath(history)
            cp(receipt_path,joinpath(history,"compatibility_$(hashfile(receipt_path)).json");force=true)
            json(receipt_path,receipt)
        end
    else
        json(receipt_path,receipt)
    end
    println("Verified ",model," at MAP and ",length(checks)-1," retained tail/chain points");flush(stdout)
    problem,saved,receipt
end

function retained_map_result(problem,saved)
    z=Float64.(saved.z);g=posterior_gradient(problem,z);parts=posterior_components(problem,z)
    original_convergence=saved.converged && any(s->s.converged,saved.starts)
    valid=original_convergence && norm(g,Inf)<=1e-3
    # Rehydrate genuine saved optimizer evidence; do not invent an Optim result.
    info=(gradient=g,gradient_norm=norm(g,Inf),gradient_tolerance=1e-3,
        origin=:verified_retained_map,original_optimizer_starts=saved.starts,
        original_converged=saved.converged)
    MAPResult(z,physical_parameters(problem,z),parts.loglikelihood,parts.logprior,
        parts.logposterior,valid,info,ParameterSpec[problem.parameters...],nothing)
end

function analyze_uncertainty(root,model,config;compatibility_only=false,method=:laplace)
    method in (:laplace,:gauss_newton) || error("Unknown uncertainty method")
    before=retained_hashes(root)
    try
        problem,saved,receipt=compatible_target(root,model,config)
        compatibility_only && return
        out=joinpath(ANALYSIS_ROOT,string(model));result=retained_map_result(problem,saved)
        refined=false
        if !result.converged
            result=fit(problem;initial=Float64.(saved.z),starts=(Float64.(saved.z),),
                maxiters=200,derivative_step=1e-6)
            refined=true
        end
        json(joinpath(out,"map_analysis.json"),(original=saved,refined,
            selected_z=result.minimizer,selected_parameters=physical_parameters(problem,result.minimizer,:batch),
            converged=result.converged,gradient=posterior_gradient(problem,result.minimizer),
            inference_coordinates="logit(logAJ), log(AG), logit(g), optional log(D0); gamma fixed",
            evidence=refined ? "new local Optim MAP run from retained coordinates" : "original optimizer evidence plus independently re-evaluated stationarity"))
        attempts=Any[];approximation=nothing
        initial_step=method===:laplace ? eps(Float64)^(1/4) : cbrt(eps(Float64))
        for step in (initial_step,initial_step/4,initial_step/16)
            println(method," ",model," verification step=",step);flush(stdout)
            try
                approximation=method===:laplace ? laplace_approximation(problem,result;step) :
                    gauss_newton_approximation(problem,result;step)
                push!(attempts,(step,status="accepted",report=approximation.diagnostics))
                break
            catch failure
                failure isa Union{LaplaceFailure,GaussNewtonFailure} || rethrow()
                push!(attempts,(step,status="rejected",report=failure.report))
            end
        end
        common=(schema=1,model=string(model),approximation=string(method),
            coordinates=method===:laplace ? "original inference z; full negative log posterior, including all prior Jacobians" :
                "original inference z; whitened prediction sensitivities plus actual prior curvature, including all prior Jacobians",
            parameter_names=string.(getproperty.(problem.parameters,:name)),center=result.minimizer,
            physical_center=physical_parameters(problem,result.minimizer,:batch),
            priors=C.prior_metadata(problem),attempts,refined,
            compatibility_sha256=hashfile(joinpath(out,"compatibility.json")),
            analysis_source_sha256=hashfile(@__FILE__),
            nonfinite_diagnostics="JSON null denotes a nonfinite value, not regularization",
            no_sampling_or_latent_propagation=true)
        report=if approximation===nothing
            merge(common,(status="rejected",diagnostics=last(attempts).report))
        else
            curvature=method===:laplace ? (hessian=matrix_rows(approximation.hessian),) :
                (precision=matrix_rows(approximation.precision),jacobian=matrix_rows(approximation.jacobian),
                 likelihood_information=matrix_rows(approximation.likelihood_information),
                 prior_hessian=matrix_rows(approximation.prior_hessian))
            merge(common,curvature,(status="accepted",covariance=matrix_rows(approximation.covariance),
                diagnostics=approximation.diagnostics))
        end
        json(joinpath(out,string(method)*".json"),report)
        println(method," ",model,": ",report.status);flush(stdout)
    finally
        check_readonly(root,before)
    end
end
