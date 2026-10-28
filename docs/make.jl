using Documenter
using PopulationBalanceModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
include("api_reference.jl")
const API_REFERENCE_SOURCES = APIReference.source_index(ROOT)
# Downloadable sources preserve the reproduction drivers without embedding them
# in the walkthroughs. The generated copies are ignored by Git.
source_assets = joinpath(@__DIR__, "src", "assets", "source")
# This directory is generated entirely below; remove obsolete downloadable paths.
isdir(source_assets) && rm(source_assets; recursive=true)
mkpath(source_assets)
for (directory, subdirectories, files) in walkdir(joinpath(ROOT, "examples"))
    filter!(!=("__pycache__"), subdirectories)
    target = joinpath(source_assets, "examples", relpath(directory, joinpath(ROOT, "examples")))
    mkpath(target)
    for file in files
        file == "Manifest.toml" && continue
        splitext(file)[2] in (".jl", ".py", ".csv", ".txt", ".md", ".toml") || continue
        name = endswith(file, ".md") ? file[1:end-3] * ".txt" : file
        cp(joinpath(directory, file), joinpath(target, name); force=true)
    end
end
plot_sources = joinpath(source_assets, "plotting")
mkpath(plot_sources)
for file in readdir(joinpath(ROOT, "docs", "benchmarks"))
    endswith(file, ".py") || file == "README.md" || continue
    cp(joinpath(ROOT, "docs", "benchmarks", file), joinpath(plot_sources, file == "README.md" ? "README.txt" : file); force=true)
end

# GitHub provides the actual owner; local archives need no guessed remote.
slug = get(ENV, "GITHUB_REPOSITORY", "")
remote = if isempty(slug)
    nothing
else
    owner, repository = split(slug, '/'; limit=2)
    Documenter.Remotes.GitHub(owner, repository)
end
remote_options = isnothing(remote) ? (remotes=nothing,) : (repo=remote,)

makedocs(;
    root = @__DIR__,
    modules = [PopulationBalanceModels],
    sitename = "PopulationBalanceModels.jl",
    authors = "PBM.jl contributors",
    remote_options...,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = isempty(slug) ? nothing : "main",
        repolink = isempty(slug) ? nothing : "https://github.com/$(slug)",
        size_threshold_warn = 100 * 2^10,
        size_threshold = 200 * 2^10,
    ),
    checkdocs = :exports,
    warnonly = false,
    workdir = ROOT,
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Defining models" => "defining_models.md",
        "Coupled systems" => "coupled_systems.md",
        "Axially dispersed reactors" => "axial_reactors.md",
        "Stationary population balances" => "steady_state.md",
        "Numerical methods" => [
            "Overview and method selection" => "numerical_methods.md",
            "Growth and dissolution" => "methods/growth.md",
            "Internal-coordinate diffusion and GRD" => "methods/dispersion.md",
            "Transport along growth trajectories" => "methods/characteristics.md",
            "Nucleation" => "methods/nucleation.md",
            "Birth, death, and dilution" => "methods/sources.md",
            "Aggregation and breakage" => "methods/collisions.md",
            "Time integration" => "methods/time.md",
            "Compact finite-volume growth" => "methods/compact.md",
            "Moment methods" => "moment_methods.md",
            "Realizable evolution and particle removal" => "methods/realizable_evolution.md",
        ],
        "Parameter estimation" => "inference.md",
        "Forward uncertainty quantification" => "uncertainty_quantification.md",
        "Case studies" => [
            "Overview" => "case_studies/index.md",
            "Comparing growth and time methods" => "case_studies/constant_growth.md",
            "Batch cooling crystallization" => "case_studies/crystallization.md",
            "Aggregation and nucleation with moments" => "case_studies/moment_crystallization.md",
            "Continuous tubular precipitation" => "case_studies/continuous_precipitation.md",
            "Isothermal protein crystallization" => "case_studies/protein_crystallization.md",
            "Stem-cell self-renewal and differentiation" => "case_studies/stem_cells.md",
            "Lipid nanoparticle formation" => "case_studies/nanoparticles.md",
            "Growth in two particle dimensions" => "case_studies/transport.md",
            "Estimating size-dependent growth" => "case_studies/estimating_growth.md",
            "Uncertainty in continuous milling" => "case_studies/milling_uncertainty.md",
        ],
        "API reference" => APIReference.navigation(),
        "Contributing and internals" => "development.md",
        "Verification and reproduction" => "verification.md",
        "Compact verification notes" => "methods/compact_verification.md",
        hide("audit.md"), # Preserve existing /audit/ links without a navigation entry.
        "Publishing" => "publishing.md",
    ],
)

if get(ENV, "POPULATIONBALANCEMODELS_DOCS_DEPLOY", "false") == "true"
    isempty(slug) && error("GITHUB_REPOSITORY is required for documentation deployment")
    isempty(get(ENV, "DOCUMENTER_KEY", "")) &&
        error("Add the DOCUMENTER_KEY repository secret; see docs/src/publishing.md")
    deploydocs(
        repo = "github.com/$(slug).git",
        devbranch = "main",
        push_preview = false,
        versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
    )
end
