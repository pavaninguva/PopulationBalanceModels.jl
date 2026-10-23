# Independent checks for the example's isolated-chain aggregation (no worker PBMs).
using Test, PopulationBalanceModels, AdvancedHMC, MCMCDiagnosticTools, Serialization
using Statistics, LinearAlgebra
using Distributions: Normal
isdefined(@__MODULE__,:bayesian_linear_fixture) || include("fixtures/bayesian_linear.jl")
isdefined(@__MODULE__,:ProteinCrystallization) || include("../examples/protein_crystallization.jl")
const PC=ProteinCrystallization
@testset "Independent chains retain the statistical target" begin
    f=bayesian_linear_fixture();p=f.problem
    runs=[sample_posterior(p;initial=f.mode,chains=1,seed, samples=250,warmup=150)
        for seed in (71,72)]
    buffer=IOBuffer();serialize(buffer,runs);seekstart(buffer);runs=deserialize(buffer)
    combined=PC._combine_protein_chains(p,runs;elapsed_seconds=1.)
    array=permutedims(combined.z,(2,3,1))
    @test size(combined.z)==(2,250,2)
    @test combined.z[:,:,1]==runs[1].z[:,:,1]
    @test combined.z[:,:,2]==runs[2].z[:,:,1]
    @test diagnostics(combined).rhat==MCMCDiagnosticTools.rhat(array;kind=:rank)
    @test diagnostics(combined).ess_tail==MCMCDiagnosticTools.ess(array;kind=:tail)
    @test diagnostics(combined).forward_solves==sum(diagnostics(r).forward_solves for r in runs)
    @test diagnostics(combined).chain_seeds==(71,72)
    values=reshape(combined.z,2,:)
    @test all(abs.(vec(mean(values;dims=2))-f.mode).<.3sqrt.(diag(f.covariance)))
    @test norm(cov(permutedims(values))-f.covariance)/norm(f.covariance)<.3
    @test_throws ArgumentError PC._combine_protein_chains(p,[runs[1],runs[1]];elapsed_seconds=1.)
    changed=replace_priors(p,[Prior(Normal(0,9);coordinates=:physical),p.parameters[2].prior])
    @test_throws ArgumentError PC._combine_protein_chains(changed,runs;elapsed_seconds=1.)
end
