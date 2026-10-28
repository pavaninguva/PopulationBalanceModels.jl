include("fixtures/example_modules.jl")
# Independent checks for the example's isolated-chain aggregation (no worker PBMs).
using Test, PopulationBalanceModels, AdvancedHMC, MCMCDiagnosticTools, Serialization
using Statistics, LinearAlgebra
using Distributions: Normal
import Distributions
include("../paper/_support/protein_sampling_policy.jl")
@testset "Protein sampling policies protect retained results" begin
    base=Dict{String,Any}("warmup"=>50,"samples"=>100,"chains"=>4,"seed"=>92,
        "target_acceptance"=>.9,"max_depth"=>10)
    legacy=protein_sampling_settings(base)
    @test legacy["stopping"]=="fixed" && legacy["samples"]==100
    automatic=protein_sampling_settings(merge(base,Dict("stopping"=>"diagnostics")))
    @test automatic["min_samples"]==500 && automatic["max_samples"]==10000
    @test !haskey(automatic,"samples")
    @test automatic["diagnostic_quantities"]==["log_J_at_S6","log_G_at_S6"]
    @test protein_sampling_matches(legacy,legacy)
    old=copy(legacy);delete!(old,"stopping")
    @test protein_sampling_matches(old,legacy) # historical fixed-length runs
    @test protein_sampling_matches(Dict(Symbol(k)=>v for (k,v) in old),legacy)
    @test !protein_sampling_matches(old,automatic)
    @test !protein_sampling_matches(nothing,automatic)
    @test_throws ErrorException protein_sampling_settings(merge(base,Dict("stopping"=>"unknown")))
    mktempdir() do folder
        posterior=joinpath(folder,"posterior_draws.csv")
        write(posterior,"retained posterior bytes\n")
        before=read(posterior)
        @test_throws ArgumentError protect_protein_posterior(folder;valid=false)
        @test_throws ArgumentError protect_protein_posterior(folder;valid=true,requested=automatic,stored=old)
        @test protect_protein_posterior(folder;valid=true,requested=legacy,stored=old)===nothing
        @test protect_protein_posterior(folder;valid=false,force=true)===nothing
        @test read(posterior)==before
    end
end
isdefined(@__MODULE__,:bayesian_linear_fixture) || include("fixtures/bayesian_linear.jl")
_load_example(:ProteinCrystallization, "protein_crystallization.jl")
const PC=ProteinCrystallization
@testset "Independent chains retain the statistical target" begin
    f=bayesian_linear_fixture();p=f.problem
    runs=[sample_posterior(p;initial=f.mode,chains=1,seed,metric=:dense,samples=250,warmup=150)
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
    initial_metric=Matrix(Symmetric(2f.covariance))
    tuned=sample_posterior(p;initial=f.mode,chains=1,seed=73,samples=250,warmup=150,
        metric=:dense,initial_covariance=initial_metric)
    @test_throws ArgumentError PC._combine_protein_chains(p,[runs[1],tuned];elapsed_seconds=1.)
    mixed=PC._combine_protein_chains(p,[runs[1],tuned];elapsed_seconds=1.,allow_distinct_tuning=true)
    @test mixed.z[:,:,2]==tuned.z[:,:,1]
    @test diagnostics(mixed).distinct_initial_tuning
    @test diagnostics(mixed).initial_covariances==[nothing,initial_metric]
    @test diagnostics(mixed).rhat==MCMCDiagnosticTools.rhat(permutedims(mixed.z,(2,3,1));kind=:rank)
    @test norm(cov(permutedims(reshape(mixed.z,2,:)))-f.covariance)/norm(f.covariance)<.3
end

isdefined(@__MODULE__,:IsothermalProteinCrystallization) ||
    include("../examples/isothermal_protein_crystallization.jl")
isdefined(@__MODULE__,:IsothermalProteinSampling) ||
    include("../examples/support/isothermal_protein_sampling.jl")
isdefined(@__MODULE__,:protein_transport_fixture) || include("fixtures/protein_transport.jl")
using ForwardDiff
@testset "Nonlinear transport sampling has an independent Gaussian reference" begin
    C=LegacyProteinCoordinates;S=IsothermalProteinSampling
    template=C.estimation_problem(model=:cnt_grd,bins=20)
    mapping=S.transported_rate_coordinates(C,template,initial_coordinates(template),protein_transport_fixture();
        growth_reference_supersaturation=6.)
    mu=mapping.from_sampler(zeros(5));sd=[.2,.01,.1,.3,.3]
    physical=physical_parameters(template,mu,:batch)
    specs=[ParameterSpec(s.name;initial=getproperty(physical,s.name),transform=s.transform,
        prior=Prior(Normal(mu[i],sd[i]);coordinates=:inference)) for (i,s) in enumerate(template.parameters)]
    pb=PBMProblem(compile(PBMModel(),Grid1D([0.,.5,1.])),[0.,0.],(0.,1.))
    measurement=Measurement([1.],[0.],FunctionObservable((u,op,p,t)->zero(p.gamma));
        likelihood=GaussianLikelihood(sigma=1.))
    experiment=InferenceExperiment(:batch,pb,measurement;
        solver=(p,ts)->PBMSolution(ts,[copy(p.u0) for _ in ts],p,0))
    problem=EstimationProblem(specs,(experiment,);fixed=(cstar=2.16,))
    J=ForwardDiff.jacobian(mapping.to_sampler,mu)
    covariance=Matrix(Symmetric(J*Diagonal(sd.^2)*J'))
    post=sample_posterior(problem;initial=mu,starts=[mu.+.1sd,mu.-.1sd],chains=2,
        sampling_map=mapping,initial_covariance=covariance,metric=:dense,
        samples=500,warmup=250,target_acceptance=.95,max_depth=10,seed=12458)
    values=reshape(post.z,5,:)
    @test all(abs.(vec(mean(values;dims=2))-mu).<.25sd)
    standardized=(values.-mu)./sd
    @test norm(cov(permutedims(standardized))-Matrix{Float64}(I,5,5))<.4
    @test diagnostics(post).divergences==0
    for c in 1:2,i in (1,500)
        independent=sum(Distributions.logpdf(Normal(mu[k],sd[k]),post.z[k,i,c]) for k in 1:5)-.5log(2pi)
        @test diagnostics(post).statistics[c][i].log_density≈independent atol=2e-10
    end
end
