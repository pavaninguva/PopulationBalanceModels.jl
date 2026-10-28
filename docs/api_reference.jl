"""Source ownership for the topic-based API reference; no simulation dependencies."""
module APIReference

# A directory rule includes future files in that subsystem. Exact file rules
# take precedence over a containing directory, e.g. observations vs inference.
const SECTIONS = [
    (id=:models, title="Models and representations", page="api/models.md",
        sources=("src/modeling/", "src/representations/", "src/core/")),
    (id=:numerics, title="Numerical methods and compilation", page="api/numerics.md",
        sources=("src/numerics/", "src/compilation/")),
    (id=:problems, title="Problems, solvers and integrations", page="api/problems.md",
        sources=("src/time_integration/", "src/stationary/", "src/PopulationBalanceModels.jl")),
    (id=:coupling, title="Coupling and accounting", page="api/coupling.md",
        sources=("src/coupling/",)),
    (id=:observations, title="Observations and discrepancies", page="api/observations.md",
        sources=("src/inference/observations.jl", "src/inference/distribution_discrepancies.jl")),
    (id=:estimation, title="Parameter estimation", page="api/estimation.md",
        sources=("src/inference/",)),
    (id=:uq, title="Forward uncertainty quantification", page="api/uncertainty.md",
        sources=("src/uq/",)),
]

pages() = [section.page for section in SECTIONS]
navigation() = ["Overview and symbol index" => "api.md";
    [section.title => section.page for section in SECTIONS]]

_normalized(path) = replace(normpath(replace(path, '\\' => '/')), '\\' => '/')

function source_owner(path; sections=SECTIONS)
    path = _normalized(path)
    matches = Tuple{Int,Symbol}[]
    for section in sections, rule in section.sources
        prefix = endswith(rule, "/") || endswith(rule, "\\")
        normalized = prefix ? rstrip(_normalized(rule), '/') : _normalized(rule)
        if prefix ? startswith(path, normalized * "/") : path == normalized
            push!(matches, (length(normalized), section.id))
        end
    end
    isempty(matches) && throw(ArgumentError(
        "API reference has no source rule for $path. Add its subsystem to docs/api_reference.jl."))
    specificity = maximum(first, matches)
    owners = unique(last(match) for match in matches if first(match) == specificity)
    length(owners) == 1 || throw(ArgumentError(
        "Ambiguous API reference ownership for $path: $(join(string.(owners), ", "))."))
    only(owners)
end

"""Partition all checkout source files once; fail before rendering if routing is incomplete.

Documenter selects individual docstrings by these source paths. Its strict
checkdocs and duplicate checks remain responsible for canonical API coverage.
"""
function source_index(root; sections=SECTIONS)
    ids = [section.id for section in sections]
    length(unique(ids)) == length(ids) || throw(ArgumentError("duplicate API reference section id"))
    pages = [section.page for section in sections]
    length(unique(pages)) == length(pages) || throw(ArgumentError("duplicate API reference page"))
    for page in pages
        isfile(joinpath(root, "docs/src", page)) ||
            throw(ArgumentError("missing API reference page docs/src/$page"))
    end
    index = Dict(id => String[] for id in ids)
    for (directory, _, files) in walkdir(joinpath(root, "src")), file in sort(files)
        endswith(file, ".jl") || continue
        path = relpath(joinpath(directory, file), root)
        push!(index[source_owner(path; sections)], path)
    end
    for section in sections
        # An empty Pages list means ALL docstrings to Documenter, not none.
        isempty(index[section.id]) && throw(ArgumentError(
            "API reference section $(section.id) has no source files; fix its rules before rendering."))
        sort!(index[section.id])
    end
    index
end
end
