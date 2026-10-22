using Documenter
using PopulationBalanceModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
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
        "Numerical methods" => [
            "Overview and method selection" => "numerical_methods.md",
            "Growth and dissolution" => "methods/growth.md",
            "Transport along growth trajectories" => "methods/characteristics.md",
            "Nucleation" => "methods/nucleation.md",
            "Birth, death, and dilution" => "methods/sources.md",
            "Aggregation and breakage" => "methods/collisions.md",
            "Time integration" => "methods/time.md",
            "Compact-method verification" => "methods/compact_verification.md",
            "Moment methods" => "moment_methods.md",
            "Realizable evolution and particle removal" => "methods/realizable_evolution.md",
        ],
        "Parameter estimation" => "inference.md",
        "Case studies" => [
            "Overview" => "case_studies/index.md",
            "Comparing growth and time methods" => "case_studies/constant_growth.md",
            "Batch cooling crystallization" => "case_studies/crystallization.md",
            "Dissolving a finite population" => "case_studies/dissolution.md",
            "Aggregation and nucleation with moments" => "case_studies/moment_crystallization.md",
            "Continuous tubular precipitation" => "case_studies/continuous_precipitation.md",
            "Inferring protein crystallization" => "case_studies/protein_crystallization.md",
            "Lipid nanoparticle formation" => "case_studies/nanoparticles.md",
            "Growth in two particle dimensions" => "case_studies/transport.md",
            "Estimating size-dependent growth" => "case_studies/estimating_growth.md",
        ],
        "API reference" => "api.md",
        "Contributing and internals" => "development.md",
        "Audit and verification" => "audit.md",
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
