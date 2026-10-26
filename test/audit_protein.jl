include("fixtures/example_modules.jl")
# Revalidate intentional retained results without generating new chains or
# changing observations, priors, likelihoods, or noise assumptions.
# julia --project=/path/to/optional-env test/audit_protein.jl
using Test, PopulationBalanceModels, ForwardDiff, MCMCDiagnosticTools
using Random, Statistics, LinearAlgebra
_load_example(:ProteinCrystallization, "protein_crystallization.jl")
const PC=ProteinCrystallization
const ASSETS=joinpath(@__DIR__,"..","docs","src","assets","protein_crystallization")
const PBM=PopulationBalanceModels
table(name)=[split(row,',') for row in readlines(joinpath(ASSETS,name))[2:end]]
number(row,i)=parse(Float64,row[i])

function audit_protein()
    problem=PC.estimation_problem(bins=80,solver=PC.scheduled_solver)
    center=PC.saved_map(ASSETS,problem)
    rows=table("posterior_draws.csv")
    chains=maximum(parse(Int,r[1]) for r in rows)
    samples=maximum(parse(Int,r[2]) for r in rows)
    Z=zeros(4,samples,chains)
    for row in rows
        c=parse(Int,row[1]);i=parse(Int,row[2])
        Z[:,i,c]=[PBM._coordinate(s.transform,number(row,k+2)) for (k,s) in enumerate(problem.parameters)]
    end
    available=reshape(Z,4,:)
    array=permutedims(Z,(2,3,1))
    rhat=MCMCDiagnosticTools.rhat(array;kind=:rank)
    essbulk=MCMCDiagnosticTools.ess(array;kind=:bulk)
    esstail=MCMCDiagnosticTools.ess(array;kind=:tail)
    saved=table("sampler_diagnostics.csv")
    checks=Pair{String,Float64}[]
    @testset "Retained protein posterior: coordinates and diagnostics" begin
        @test (samples,chains)==(1500,4)
        # Only physical draws were retained. Mapping exp(z) back through log
        # changes some z values by a few ulps; folded ranks can break ties.
        # Measure that sensitivity rather than imposing bitwise rank identity.
        rng=MersenneTwister(23)
        rank_perturbations=[begin
            perturbed=copy(Z)
            for i in eachindex(perturbed);perturbed[i]+=rand(rng,-2:2)*eps(perturbed[i]);end
            MCMCDiagnosticTools.rhat(permutedims(perturbed,(2,3,1)))
        end for _ in 1:12]
        changes=maximum(abs.(hcat(rank_perturbations...).-rhat);dims=2)[:]
        differences=abs.(rhat-[number(r,2) for r in saved])
        @test all(differences.<=changes.+1e-10)
        @test maximum(changes)<1e-4 # far below the 0.01 convergence-screen margin
        push!(checks,"max_rhat_roundtrip_change"=>maximum(differences),
            "max_rhat_two_ulp_sensitivity"=>maximum(changes))
        @test essbulk≈[number(r,3) for r in saved] rtol=1e-10
        @test esstail≈[number(r,4) for r in saved] rtol=1e-10
        @test maximum(rhat)<1.01 && minimum(essbulk)>400 && minimum(esstail)>400
        push!(checks,"max_rhat"=>maximum(rhat),"min_ess_bulk"=>minimum(essbulk),"min_ess_tail"=>minimum(esstail))
        energies=zeros(samples,chains)
        statistics=table("chains.csv")
        for row in statistics
            energies[parse(Int,row[2]),parse(Int,row[1])]=number(row,4)
        end
        bfmi=MCMCDiagnosticTools.bfmi(energies)
        @test bfmi≈[number(r,3) for r in table("chain_diagnostics.csv")] rtol=1e-10
        @test count(r->r[8]=="true",statistics)==0
        push!(checks,"min_bfmi"=>minimum(bfmi),"retained_divergences"=>0.)
        map=PC.reference_sampling_map(problem)
        for z in eachcol(available)
            state=PBM._sampling_map_state(map,map.to_sampler(z))
            @test state.z≈z rtol=1e-12 atol=1e-12
            @test abs(state.actual_logjac)<1e-12
        end
        # Test gradients in the original posterior support, including observed tails.
        selected=unique(vcat([argmin(available[i,:]) for i in 1:4],
                             [argmax(available[i,:]) for i in 1:4]))
        maxerror=0.;maxdensity=0.
        for z in eachcol(hcat(center,available[:,selected]))
            # The default relative stencil straddles limiter switches near the
            # saved MAP. A measured 1e-6 stencil and its half agree with AD;
            # smaller 2e-7 probes independently confirm the reference.
            report=check_sampling_map(problem,map,z;step=1e-6)
            @test report.valid
            maxerror=max(maxerror,report.error)
        end
        stats=Dict((parse(Int,r[2]),parse(Int,r[1]))=>number(r,3) for r in statistics)
        for index in selected
            draw,chain=Tuple(CartesianIndices((samples,chains))[index])
            difference=abs(logposterior(problem,available[:,index])-stats[(draw,chain)])
            @test difference<1e-7
            maxdensity=max(maxdensity,difference)
        end
        push!(checks,"max_map_gradient_error"=>maxerror,"max_saved_logdensity_change"=>maxdensity,
            "corr_logAJ_j"=>cor(vec(Z[1,:,:]),vec([PBM._physical(problem.parameters[2].transform,z) for z in Z[2,:,:]])))
    end
    flattened(p,z)=vcat((vec(v) for v in first(predictions(p,z)))...)
    data=PC.observations()
    scale=vcat(fill(data.concentration.sigma,length(data.concentration.t)),fill(data.D50.sigma,length(data.D50.t)))
    @testset "Retained protein predictions and temporal checks" begin
        fine=PC.estimation_problem(bins=80,solver=(pb,ts)->PC.scheduled_solver(pb,ts;scale=.5))
        uniform=PC.estimation_problem(bins=80,dt=.5)
        # Retain the original draw identities; randperm is not stable across
        # Julia releases even when MersenneTwister has the same seed.
        selection=table("posterior_predictive_indices.csv")
        indices=[(parse(Int,r[2])-1)*samples+parse(Int,r[3]) for r in selection]
        @test [parse(Int,r[1]) for r in selection]==collect(1:100)
        predicted=table("posterior_predictive.csv")
        maxprediction=0.
        for (i,index) in enumerate(indices)
            latent=flattened(problem,available[:,index])
            saved=[number(r,5) for r in predicted if parse(Int,r[1])==i]
            @test length(saved)==length(latent)
            difference=maximum(abs.(latent-saved)./scale)
            @test difference<1e-7
            maxprediction=max(maxprediction,difference)
        end
        push!(checks,"max_saved_latent_change_sigma"=>maxprediction)
        selected=randperm(MersenneTwister(2028),size(available,2))[1:12]
        halferror=0.;uniformerror=0.
        for z in eachcol(hcat(center,available[:,selected]))
            base=flattened(problem,z)
            halferror=max(halferror,maximum(abs.(flattened(fine,z)-base)./scale))
            uniformerror=max(uniformerror,maximum(abs.(flattened(uniform,z)-base)./scale))
        end
        @test halferror<.01 && uniformerror<.01
        push!(checks,"max_half_step_change_sigma"=>halferror,"max_uniform_step_change_sigma"=>uniformerror)
        residual=(flattened(problem,center)-vcat(data.concentration.y,data.D50.y))./scale
        @test residual≈[number(r,5) for r in table("fit_residuals.csv")] rtol=1e-8 atol=1e-8
        nc=length(data.concentration.y)
        push!(checks,"concentration_residual_rms"=>sqrt(mean(abs2,residual[1:nc])),
            "D50_residual_rms"=>sqrt(mean(abs2,residual[nc+1:end])),
            "max_absolute_standardized_residual"=>maximum(abs,residual))
        pb=PC.build(bins=20,tf=1.)
        for times in ([.8,.2],[NaN],[1.1],[-.1])
            @test_throws ArgumentError PC.scheduled_solver(pb,times)
        end
    end
    for (name,value) in checks;println(name," = ",value);end
    if !isempty(ARGS)
        open(first(ARGS),"w") do io
            println(io,"quantity,value")
            for (name,value) in checks;println(io,name,',',value);end
        end
    end
    checks
end
audit_protein()
