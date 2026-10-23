# Invoked by run_posterior(parallel=true), using the parent's Julia environment.
using PopulationBalanceModels, AdvancedHMC, MCMCDiagnosticTools, Serialization
include(joinpath(@__DIR__,"..","protein_crystallization.jl"))
const PC=ProteinCrystallization
directory,chain_text=ARGS
chain=parse(Int,chain_text)
s=deserialize(joinpath(directory,"settings.bin"))
problem=PC.estimation_problem(bins=s.bins,priors=s.priors,
    solver=(pb,ts)->PC.scheduled_solver(pb,ts;scale=s.time_scale))
draws=sample_posterior(problem;initial=s.center,starts=[s.starts[chain]],
    initial_covariance=s.tuning,samples=s.samples,warmup=s.warmup,chains=1,
    seed=s.seed+chain-1,target_acceptance=s.target_acceptance,max_depth=s.max_depth,
    min_ess=s.min_ess,metric=:dense,sampling_map=PC.reference_sampling_map(problem),verbose=true)
serialize(joinpath(directory,"chain_$chain.bin"),draws)
