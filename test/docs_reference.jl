using Test
include("../docs/api_reference.jl")

const ROOT = normpath(joinpath(@__DIR__, ".."))

@testset "API reference ownership and growth" begin
    index = APIReference.source_index(ROOT)
    files = vcat(values(index)...)
    expected = [relpath(joinpath(directory, file), ROOT)
        for (directory, _, entries) in walkdir(joinpath(ROOT, "src"))
        for file in entries if endswith(file, ".jl")]
    @test sort(files) == sort(expected)
    @test length(files) == length(unique(files))
    @test all(!isempty, values(index))
    @test first(APIReference.navigation()) == ("Overview and symbol index" => "api.md")
    @test length(APIReference.navigation()) == length(APIReference.SECTIONS)+1
    @test APIReference.source_owner("src/inference/observations.jl") == :observations
    @test APIReference.source_owner("src/inference/distribution_discrepancies.jl") == :observations
    @test APIReference.source_owner("src/inference/future_likelihood.jl") == :estimation
    @test APIReference.source_owner("src\\uq\\future_backend.jl") == :uq
    @test APIReference.source_owner("src/uq/new/basis.jl") == :uq
    @test APIReference.source_owner("src/inference/observations.jl";
        sections=reverse(APIReference.SECTIONS)) == :observations
    @test_throws ArgumentError APIReference.source_owner("src/future_subsystem/method.jl")
    @test_throws ArgumentError APIReference.source_owner("src/uq_lookalike/method.jl")
    @test_throws ArgumentError APIReference.source_owner("src/inference/prefix_observations.jl";
        sections=[only(filter(s -> s.id == :observations, APIReference.SECTIONS))])
    ambiguous = [APIReference.SECTIONS;
        (id=:another_uq, title="Ambiguous", page="api/other.md", sources=("src/uq/",))]
    @test_throws ArgumentError APIReference.source_owner("src/uq/new.jl"; sections=ambiguous)

    # A new source file joins an existing topic without listing its symbols or
    # editing Markdown. A new subsystem fails until explicitly assigned.
    mktempdir() do root
        for section in APIReference.SECTIONS
            page = joinpath(root, "docs/src", section.page)
            mkpath(dirname(page))
            write(page, "# Reference\n")
            rule = first(section.sources)
            source = joinpath(root, endswith(rule, "/") ? rule * "fixture.jl" : rule)
            mkpath(dirname(source))
            write(source, "# synthetic source\n")
        end
        before = APIReference.source_index(root)
        path = joinpath("src", "uq", "new", "basis.jl")
        mkpath(dirname(joinpath(root, path)))
        write(joinpath(root, path), "# a future feature\n")
        after = APIReference.source_index(root)
        @test after[:uq] == sort([before[:uq]; path])
        @test all(after[id] == before[id] for id in keys(before) if id != :uq)
        unknown = joinpath(root, "src", "new_subsystem", "new.jl")
        mkpath(dirname(unknown))
        write(unknown, "# no reference page yet\n")
        @test_throws ArgumentError APIReference.source_index(root)
        rm(unknown)
        rm(joinpath(root, "docs/src", first(APIReference.pages())))
        @test_throws ArgumentError APIReference.source_index(root)
    end

    duplicate = [APIReference.SECTIONS; first(APIReference.SECTIONS)]
    @test_throws ArgumentError APIReference.source_index(ROOT; sections=duplicate)
    empty_section = [APIReference.SECTIONS;
        (id=:empty, title="Empty", page="api.md", sources=("src/absent/",))]
    @test_throws ArgumentError APIReference.source_index(ROOT; sections=empty_section)
end
