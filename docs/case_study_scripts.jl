"""Read authored tutorials; manual pages and downloads use the same source."""
module CaseStudyScripts
const ROOT = normpath(joinpath(@__DIR__, ".."))
const CASES = (:constant_growth, :crystallization, :moment_crystallization,
    :continuous_precipitation, :protein_crystallization, :stem_cells,
    :transport, :estimating_growth, :milling_uncertainty)
function source_path(name::Symbol; root=ROOT)
    name in CASES || throw(ArgumentError("Unknown tutorial: $name"))
    joinpath(root, "examples", "tutorials", "$name.jl")
end
script(name::Symbol; root=ROOT) = read(source_path(name; root), String)
function write_scripts(destination; root=ROOT)
    mkpath(destination)
    for file in readdir(joinpath(root,"examples","tutorials"))
        endswith(file,".jl") || continue
        cp(joinpath(root,"examples","tutorials",file),joinpath(destination,file);force=true)
    end
    # Keep the original download addresses and relative ../support and ../data
    # includes usable. Canonical downloads also live under source/examples/.
    for directory in ("support", "data")
        source = joinpath(root, "examples", directory)
        target = joinpath(dirname(destination), directory)
        for (path, dirs, files) in walkdir(source)
            filter!(!=("__pycache__"), dirs)
            relative = relpath(path, source)
            mkpath(joinpath(target, relative))
            for file in files
                splitext(file)[2] in (".jl", ".csv", ".txt", ".json", ".toml") || continue
                cp(joinpath(path, file), joinpath(target, relative, file); force=true)
            end
        end
    end
    destination
end
end
