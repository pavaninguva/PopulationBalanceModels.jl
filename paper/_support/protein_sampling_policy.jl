# Sampling configuration and output protection; no PBM or sampler implementation.
function protein_sampling_settings(config)
    policy=get(config,"stopping","fixed")
    policy in ("fixed","diagnostics") || error("stopping must be fixed or diagnostics")
    settings=Dict{String,Any}(k=>config[k] for k in
        ("warmup","chains","seed","target_acceptance","max_depth"))
    settings["stopping"]=policy
    if policy=="fixed"
        settings["samples"]=config["samples"]
    else
        for (key,default) in (("min_samples",500),("batch_samples",250),
                ("max_samples",10000),("consecutive_passes",2))
            settings[key]=get(config,key,default)
        end
        settings["diagnostic_quantities"]=["log_J_at_S6","log_G_at_S6"]
    end
    settings["sampling_coordinates"]=get(config,"sampling_coordinates","initial_rates")
    settings["secondary_start_inference"]=get(config,"secondary_start_inference",nothing)
    settings["metric_initialization"]=get(config,"metric_initialization","expected_information")
    settings["growth_reference_supersaturation"]=get(config,"growth_reference_supersaturation",nothing)
    settings
end

function protein_sampling_matches(stored,requested)
    stored===nothing && return false
    all(requested) do (key,value)
        default=key=="stopping" ? "fixed" : key=="sampling_coordinates" ? "initial_rates" :
            key=="metric_initialization" ? "expected_information" : nothing
        get(stored,Symbol(key),get(stored,key,default))==value
    end
end

function protect_protein_posterior(folder;valid,force=false,requested=nothing,stored=nothing)
    if isfile(joinpath(folder,"posterior_draws.csv")) && !force &&
            (!valid || (requested!==nothing && !protein_sampling_matches(stored,requested)))
        throw(ArgumentError("Retained posterior would be replaced. Use --output=/path/to/a/new/run; " *
            "--force explicitly authorizes replacing this run. Original results have not been changed."))
    end
end
