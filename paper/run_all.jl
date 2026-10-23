"""Run each case in a fresh Julia process, sequentially, using this environment."""
failed = String[]
for directory in ("01_growth_time_methods","02_aggregation_fragmentation","03_tubular_precipitation","04_protein_crystallization")
    script = joinpath(@__DIR__,"case_studies",directory,"run.jl")
    arguments = "--smoke" in ARGS ? ["--smoke"] : String[]
    try
        run(`$(Base.julia_cmd()) --project=$(@__DIR__) $script $arguments`)
    catch error
        push!(failed,directory)
        @error "Case failed; continuing independent cases" directory exception=error
    end
end
isempty(failed) || error("Incomplete cases: $(join(failed,", "))")
