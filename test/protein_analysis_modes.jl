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
    @test_throws ErrorException Reproduction.main(["--compatibility-only","--output=/tmp/unused-protein-run"])
    @test_throws ErrorException Reproduction.main(["--stopping=diagnostics","--samples=100"])
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
        # A changed/invalid production cache must fail before fitting, writing
        # model reports, or removing old draws, not just in the pure guard.
        config=Reproduction.TOML.parsefile(joinpath(Reproduction.CASE,"config.toml"))["production"]
        config["stopping"]="diagnostics"
        @test_throws ArgumentError Reproduction.execute_model(temporary,:cnt,config;sample=true)
        @test Reproduction.retained_hashes(temporary)==saved
        write(posterior,"changed draws\n")
        @test_throws ErrorException Reproduction.check_readonly(temporary,saved)
    end
end

@testset "Retained sampler diagnostics match exported coordinates" begin
    # Reassemble by the saved chain/draw identities, independently of the
    # sampler's internal array and the production reporting code.
    MCD=Reproduction.MCMCDiagnosticTools
    for model in ("cnt","cnt_grd")
        folder=joinpath(Reproduction.CASE,"data",model)
        d=Reproduction.readjson(joinpath(folder,"sampling_diagnostics.json"))
        n=length(Reproduction.readjson(joinpath(folder,"priors.json")))
        Z=zeros(d.samples,d.chains,n);energy=zeros(d.samples,d.chains)
        for r in Reproduction.C.table(joinpath(folder,"posterior_draws.csv"))
            Z[Int(r.draw),Int(r.chain),:]=[getproperty(r,Symbol("z",j)) for j in 1:n]
        end
        stats=Reproduction.C.table(joinpath(folder,"sampler_statistics.csv"))
        for r in stats
            energy[Int(r.draw),Int(r.chain)]=r.energy_sampler
        end
        @test MCD.rhat(Z;kind=:rank)≈Float64.(d.rhat) rtol=1e-10
        @test MCD.ess(Z;kind=:bulk)≈Float64.(d.ess_bulk) rtol=1e-10
        @test MCD.ess(Z;kind=:tail)≈Float64.(d.ess_tail) rtol=1e-10
        @test MCD.bfmi(energy)≈Float64.(d.ebfmi) rtol=1e-10
        @test count(r->string(r.divergence)=="true",stats)==d.divergences
    end
end
