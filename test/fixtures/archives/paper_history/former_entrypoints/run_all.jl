"""Run selected paper cases sequentially. Full reproduction includes protein MCMC."""
choices=[split(a,'=';limit=2)[2] for a in ARGS if startswith(a,"--case=")]
length(choices)<=1 || error("Supply --case=1,2,3,4 or all once")
selected=isempty(choices) ? "all" : only(choices)
selected in ("all","1","2","3","4") || error("Unknown paper case")
all(a->a=="--smoke" || startswith(a,"--case="),ARGS) || error("Use a case's run.jl for model-specific options")
failed=String[]
for (number,directory) in enumerate(("01_growth_time_methods","02_aggregation_fragmentation","03_tubular_precipitation","04_protein_crystallization"))
    selected in ("all",string(number)) || continue
    script=joinpath(@__DIR__,"case_studies",directory,"run.jl")
    arguments="--smoke" in ARGS ? ["--smoke"] : number==4 ? ["--full"] : String[]
    try
        run(`$(Base.julia_cmd()) --project=$(@__DIR__) $script $arguments`)
    catch failure
        push!(failed,directory)
        @error "Case failed; continuing independent cases" directory exception=failure
    end
end
isempty(failed) || error("Incomplete cases: $(join(failed,", "))")
