# Guard the read-only dispatch independently of the production cache validity.
using Test
include("../paper/protein/run.jl")
const Reproduction = ProteinCaseReproduction
@testset "Protein analysis modes protect production files" begin
    root = joinpath(Reproduction.ROOT,"paper","protein","data")
    before = Reproduction.retained_hashes(root)
    # Both mistakes fail before exporting data or entering execute_model.
    @test_throws ErrorException Reproduction.main(["--laplace","--force"])
    @test_throws ErrorException Reproduction.main(["--compatibility-only","--full"])
    @test_throws ErrorException Reproduction.main(["--gauss-newton","--force"])
    @test_throws ErrorException Reproduction.main(["--gauss-newton","--laplace"])
    # A changed mesh cannot be mistaken for retained-target compatibility.
    @test_throws ErrorException Reproduction.main(["--laplace","--model=cnt","--bins=17"])
    @test Reproduction.retained_hashes(root) == before
    # Relocated derived outputs may change; posterior input protection must not.
    mktempdir() do temporary
        mkpath(joinpath(temporary,"cnt"))
        mkpath(joinpath(temporary,"analysis"))
        posterior=joinpath(temporary,"cnt","posterior_draws.csv")
        report=joinpath(temporary,"analysis","report.json")
        write(posterior,"original draws\n")
        write(report,"old analysis\n")
        saved=Reproduction.retained_hashes(temporary)
        write(report,"new analysis\n")
        @test Reproduction.retained_hashes(temporary)==saved
        write(posterior,"changed draws\n")
        @test_throws ErrorException Reproduction.check_readonly(temporary,saved)
    end
end
