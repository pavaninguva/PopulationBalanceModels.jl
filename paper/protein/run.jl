# Thin entry point: one shared model/likelihood, optional verified checkpoints.
module ProteinCaseReproduction
using PopulationBalanceModels, Optim, JSON3, SHA, Dates, LinearAlgebra, Statistics, Random, TOML, Pkg, AdvancedHMC, MCMCDiagnosticTools, Serialization
include(joinpath(@__DIR__,"..","..","examples","isothermal_protein_crystallization.jl"))
using .IsothermalProteinCrystallization
const C=IsothermalProteinCrystallization
const ROOT=realpath(joinpath(@__DIR__,"..",".."))
const CASE=@__DIR__
include("../_support/protein_report.jl")
const DRIVER_HASHES=Dict("run.jl"=>hashfile(joinpath(CASE,"run.jl")), "report.jl"=>hashfile(joinpath(ROOT,"paper","_support","protein_report.jl")), "config.toml"=>hashfile(joinpath(CASE,"config.toml")))
include("../_support/protein_analysis.jl")

function main(args=ARGS)
    if !isempty(args) && first(args)=="--chain-worker"
        input=deserialize(args[2]);chain=parse(Int,args[3])
        post=sample_chain(input.model,input.config,input.z,input.starts[chain],input.covariances[chain],chain)
        serialize(input.paths[chain],post)
        return
    end
    allowed=("--model","--bins","--warmup","--samples","--chains","--seed","--predictive-draws","--maxiters","--target-acceptance",
        "--stopping","--min-samples","--batch-samples","--max-samples","--consecutive-passes","--output")
    flags=("--smoke","--map","--full","--sample","--data-only","--verify","--force","--laplace","--gauss-newton","--compatibility-only")
    all(a->a in flags || first(split(a,'=')) in allowed,args) || error("Unknown flag; see README.md")
    option(name,default)=let found=filter(a->startswith(a,name*"="),args)
        isempty(found) ? default : split(last(found),'=';limit=2)[2]
    end
    smoke="--smoke" in args
    analysis=any(a->a in ("--laplace","--gauss-newton","--compatibility-only"),args)
    count(a->a in ("--laplace","--gauss-newton","--compatibility-only"),args)<=1 ||
        error("Select one uncertainty/compatibility analysis per invocation")
    analysis && any(a->a in ("--smoke","--map","--full","--sample","--data-only","--verify","--force"),args) &&
        error("Uncertainty/compatibility modes are read-only analyses; do not combine them with reproduction flags")
    config=TOML.parsefile(joinpath(@__DIR__,"config.toml"))[smoke ? "smoke" : "production"]
    for (flag,key) in (("--bins","bins"),("--warmup","warmup"),("--samples","samples"),
                       ("--chains","chains"),("--seed","seed"),("--predictive-draws","predictive_draws"),("--maxiters","maxiters"))
        haskey(config,key) && (config[key]=parse(Int,option(flag,string(config[key]))))
    end
    haskey(config,"target_acceptance") &&
        (config["target_acceptance"]=parse(Float64,option("--target-acceptance",string(config["target_acceptance"]))))
    explicit_samples=any(a->startswith(a,"--samples="),args)
    config["stopping"]=option("--stopping",explicit_samples ? "fixed" : "diagnostics")
    config["stopping"] in ("fixed","diagnostics") || error("--stopping=fixed or diagnostics")
    config["stopping"]=="diagnostics" && explicit_samples &&
        error("diagnostic stopping uses --min-samples/--max-samples; omit --samples or select --stopping=fixed")
    for (flag,key,default) in (("--min-samples","min_samples",500),
            ("--batch-samples","batch_samples",250),("--max-samples","max_samples",10000),
            ("--consecutive-passes","consecutive_passes",2))
        config[key]=parse(Int,option(flag,string(default)))
    end
    model=option("--model","both")
    model in ("cnt","cnt_grd","both") || error("--model=cnt, cnt_grd or both")
    mode=smoke ? "smoke" : "production"
    default_root=smoke ? joinpath(ROOT,".cache","paper","checks","protein") : joinpath(CASE,"data")
    output=option("--output",nothing)
    analysis && output!==nothing && error("--output is for new calculations, not retained-result analyses")
    root=output===nothing ? default_root : abspath(output)
    normpath(dirname(dirname(pathof(PopulationBalanceModels))))==ROOT || error("develop this checkout in the paper environment; see paper/README.md")
    analysis || export_data(root)
    "--data-only" in args && return
    for name in (model=="both" ? (:cnt,:cnt_grd) : (Symbol(model),))
        active=copy(config)
        if !any(a->startswith(a,"--bins="),args) && haskey(config,"bins_by_model")
            active["bins"]=config["bins_by_model"][string(name)]
        end
        if haskey(config,"sampler_by_model") && haskey(config["sampler_by_model"],string(name))
            for (key,value) in config["sampler_by_model"][string(name)]
                flag="--"*replace(key,'_'=>'-')*"="
                any(a->startswith(a,flag),args) || (active[key]=value)
            end
        end
        if analysis
            analyze_uncertainty(root,name,active;compatibility_only="--compatibility-only" in args,
                method="--gauss-newton" in args ? :gauss_newton : :laplace)
        elseif "--verify" in args
            smoke && error("--verify checks completed production posteriors; use --smoke for forward checks")
            manifest=readjson(joinpath(root,string(name),"manifest.json"))
            if Dict(string(k)=>string(v) for (k,v) in pairs(manifest.source_files))==numerical_sources()
                verify_saved_target(root,name,active)
            else
                analyze_uncertainty(root,name,active;compatibility_only=true)
            end
        else
            execute_model(root,name,active;smoke,
                sample=("--full" in args || "--sample" in args),force="--force" in args)
        end
    end
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    ProteinCaseReproduction.main()
end
