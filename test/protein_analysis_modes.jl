# Guard the read-only dispatch independently of the production cache validity.
using Test
include("../paper/case_studies/04_protein_crystallization/run.jl")
const Reproduction = ProteinCaseReproduction
@testset "Protein analysis modes protect production files" begin
    root = joinpath(Reproduction.ROOT,"paper","results","04_protein_crystallization","production")
    before = Reproduction.retained_hashes(root)
    # Both mistakes fail before exporting data or entering execute_model.
    @test_throws ErrorException Reproduction.main(["--laplace","--force"])
    @test_throws ErrorException Reproduction.main(["--compatibility-only","--full"])
    @test_throws ErrorException Reproduction.main(["--gauss-newton","--force"])
    @test_throws ErrorException Reproduction.main(["--gauss-newton","--laplace"])
    # A changed mesh cannot be mistaken for retained-target compatibility.
    @test_throws ErrorException Reproduction.main(["--laplace","--model=cnt","--bins=17"])
    @test Reproduction.retained_hashes(root) == before
end
