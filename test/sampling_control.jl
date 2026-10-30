using Test, LinearAlgebra, Random, Statistics, PopulationBalanceModels
using AdvancedHMC, MCMCDiagnosticTools
using Distributions: Normal, LogNormal
import Distributions
isdefined(@__MODULE__, :bayesian_linear_fixture) || include("fixtures/bayesian_linear.jl")

# A synthetic Gaussian location likelihood has a known conjugate posterior.
# The native variant also integrates a positive, uniformly decaying population;
# its additive signal has only the independently bounded RK discretization error.
function sampling_location_fixture(; native=false)
    grid=Grid1D([0.,.5,1.])
    model=PBMModel(removal=Death((x,p,t,ctx)->1.))
    pb=PBMProblem(compile(model,grid),[1.,0.],(0.,1.))
    observable=FunctionObservable((u,op,p,t)->p.location+(native ? sum(u) : 0.))
    measurement=Measurement([1.],[2. + (native ? exp(-1.) : 0.)],observable;
        likelihood=GaussianLikelihood(sigma=1.))
    spec=ParameterSpec(:location;initial=.5,role=:nuisance,
        prior=Prior(Normal(0.,1.);coordinates=:physical),transform=IdentityTransform())
    experiment=native ? InferenceExperiment(:batch,pb,measurement;dt=.02) :
        InferenceExperiment(:batch,pb,measurement;
            solver=(p,ts)->PBMSolution(ts,[copy(p.u0) for _ in ts],p,0))
    EstimationProblem((spec,),(experiment,))
end

@testset "Diagnostic stopping against an exact Gaussian posterior" begin
    p=sampling_location_fixture()
    draws=sample_posterior(p;initial=[1.],warmup=250,seed=2580,
        target_acceptance=.9,max_depth=10,max_samples=2500,
        diagnostic_quantities=(location=theta->theta.shared.location,
            square=theta->theta.shared.location^2),
        mcse_limits=(location=(mean=.04,quantiles=(.025=>.08,.975=>.08)),))
    d=diagnostics(draws)
    @test d.stopping_policy===:diagnostics && d.stopping_reason===:diagnostics_passed
    @test d.diagnostics_passed && d.screen_passed
    @test 500<size(draws.z,2)<2500
    @test size(draws.z,3)==4 && d.samples==size(draws.z,2)
    @test d.check_history[end].consecutive_passes==2
    @test all(!h.diagnostics_passed for h in d.check_history if !h.eligible)
    @test all(h.screen_passed && h.eligible for h in d.check_history[end-1:end])
    @test all(length(s)==250 && all(t->t.is_adapt,s) for s in d.warmup_statistics)
    @test all(length(s)==d.samples && all(t->!t.is_adapt,s) for s in d.statistics)
    @test all(c->all(isfinite,c.adapted_inverse_mass),d.per_chain)
    @test length(unique(d.chain_seeds))==4
    @test d.backend_versions.AdvancedHMC==string(Base.pkgversion(AdvancedHMC))
    @test d.backend_versions.MCMCDiagnosticTools==string(Base.pkgversion(MCMCDiagnosticTools))
    values=vec(draws.z)
    @test abs(mean(values)-1.)<6only(d.mcse_mean)
    @test abs(var(values)-.5)<.1
    @test d.divergences==0 && d.max_depth_hits==0
    q=d.quantity_diagnostics.location
    @test q.values≈reshape([theta.shared.location for theta in draws.physical],size(q.values))
    @test q.mcse_mean<=.04
    @test length(q.quantile_mcse)==2 && all(x->x.passed,q.quantile_mcse)
    for estimate in q.quantile_mcse
        reference=Distributions.quantile(Normal(1.,sqrt(.5)),estimate.probability)
        @test abs(quantile(values,estimate.probability)-reference)<max(.1,6estimate.mcse)
        @test estimate.mcse==MCMCDiagnosticTools.mcse(q.values;
            kind=Base.Fix2(quantile,estimate.probability))
    end
    @test d.quantity_diagnostics.square.diagnostics_passed
    @test abs(mean(d.quantity_diagnostics.square.values)-1.5)<
        6d.quantity_diagnostics.square.mcse_mean

    # Same seeds, starts and adapted transitions: stopping changes length, not
    # the prefix of a chain. No checkpoint can consume a chain's RNG stream.
    fixed=sample_posterior(p;initial=[1.],warmup=250,samples=d.samples,chains=4,seed=2580,
        target_acceptance=.9,max_depth=10)
    @test fixed.z==draws.z
    @test fixed.diagnostics.chain_seeds==d.chain_seeds
    @test fixed.diagnostics.forward_solves==d.forward_solves
    @test fixed.diagnostics.gradient_evaluations==d.gradient_evaluations
end

@testset "Batch continuation, caps and fixed-length compatibility" begin
    f=bayesian_linear_fixture();p=f.problem;C=Matrix(Symmetric(f.covariance))
    settings=(initial=f.mode,initial_covariance=C,warmup=200,chains=2,seed=2701,
        target_acceptance=.9,max_depth=10)
    a=sample_posterior(p;settings...,min_samples=64,batch_samples=64,max_samples=257,min_ess=1e9)
    b=sample_posterior(p;settings...,min_samples=64,batch_samples=73,max_samples=257,min_ess=1e9)
    fixed=sample_posterior(p;settings...,samples=257)
    explicit=sample_posterior(p;settings...,stopping=:fixed,samples=257)
    @test a.z==b.z==fixed.z==explicit.z
    @test a.diagnostics.stopping_reason===:max_samples
    @test !a.diagnostics.diagnostics_passed
    @test size(a.z)==(2,257,2)
    @test [h.samples for h in a.diagnostics.check_history]==[64,128,192,256,257]
    @test fixed.diagnostics.stopping_policy===:fixed && fixed.diagnostics.stopping_reason===:fixed_length
    @test fixed.diagnostics.requested_samples==257
    @test all(x->x.is_adapt,Iterators.flatten(fixed.diagnostics.warmup_statistics))
    @test !any(x->x.is_adapt,Iterators.flatten(fixed.diagnostics.statistics))

    # Independent upstream sample() reference: same low-level transition
    # sequence, including exactly one warmup and its final tuned metric.
    rng=MersenneTwister(2701)
    for chain in 1:2
        crng=MersenneTwister(rand(rng,UInt))
        # Explicit starts avoid drawing randomized starts from this reference
        # seed. Use a second package run with those starts for an exact replay.
        start=f.mode.+[.03,-.02].*(chain==1 ? 1 : -1)
        h=AdvancedHMC.Hamiltonian(AdvancedHMC.DenseEuclideanMetric(copy(C)),
            z->logposterior(p,z),z->(logposterior(p,z),posterior_gradient(p,z)))
        epsilon=AdvancedHMC.find_good_stepsize(crng,h,start)
        integrator=AdvancedHMC.Leapfrog(epsilon)
        kernel=AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integrator,AdvancedHMC.GeneralisedNoUTurn(max_depth=10)))
        adaptor=AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(h.metric),
            AdvancedHMC.StepSizeAdaptor(.9,integrator))
        positions,stats=AdvancedHMC.sample(crng,h,kernel,start,328,adaptor,200;
            drop_warmup=false,verbose=false,progress=false)
        package=sample_posterior(p;initial=f.mode,starts=[f.mode.+[.03,-.02],f.mode.-[.03,-.02]],
            initial_covariance=C,warmup=200,samples=128,chains=2,seed=2701,
            target_acceptance=.9,max_depth=10)
        @test hcat(positions[201:end]...)==package.z[:,:,chain]
        @test [s.hamiltonian_energy for s in stats[201:end]]==package.diagnostics.energy[:,chain]
    end

    values=permutedims(reshape(fixed.z,2,:))
    @test all(abs.(vec(mean(values;dims=1))-f.mode).<.4sqrt.(diag(f.covariance)))
    @test norm(cov(values)-f.covariance)/norm(f.covariance)<.3
    per_chain=sample_posterior(p;settings...,initial_covariance=(C,copy(C)),samples=257)
    @test per_chain.z==fixed.z
    @test per_chain.diagnostics.metric_initialization===:user_tuning_per_chain
    @test_throws ArgumentError sample_posterior(p;settings...,initial_covariance=(C,),samples=8)
end

@testset "Precision failure, minimum length and hard sampler failures" begin
    p=sampling_location_fixture()
    common=(initial=[1.],warmup=200,chains=2,target_acceptance=.95,max_depth=10,seed=2711)
    precise=sample_posterior(p;common...,min_samples=64,batch_samples=64,max_samples=130,
        min_ess=1.,rhat_limit=2.,bfmi_limit=.01,
        diagnostic_quantities=(location=theta->theta.shared.location,),mcse_limits=(location=1e-12,))
    @test precise.diagnostics.stopping_reason===:max_samples && !precise.diagnostics.diagnostics_passed
    @test size(precise.z,2)==130
    @test !precise.diagnostics.quantity_diagnostics.location.diagnostics_passed
    @test [h.samples for h in precise.diagnostics.check_history]==[64,128,130]

    streak=sample_posterior(p;common...,min_samples=64,batch_samples=64,max_samples=128,
        consecutive_passes=3,min_ess=1.,rhat_limit=2.,bfmi_limit=.01)
    @test streak.diagnostics.stopping_reason===:max_samples
    @test streak.diagnostics.screen_passed && !streak.diagnostics.diagnostics_passed
    @test streak.diagnostics.check_history[end].consecutive_passes==2

    constant=sample_posterior(p;common...,min_samples=64,batch_samples=64,max_samples=128,
        min_ess=1.,rhat_limit=2.,bfmi_limit=.01,diagnostic_quantities=(constant=theta->1.,))
    @test constant.diagnostics.stopping_reason===:max_samples
    @test !constant.diagnostics.quantity_diagnostics.constant.diagnostics_passed

    bad=sample_posterior(p;initial=[1.],warmup=150,chains=2,seed=2722,target_acceptance=.9,
        max_depth=1,min_samples=64,batch_samples=32,max_samples=512)
    @test bad.diagnostics.stopping_reason===:sampler_pathology
    @test !bad.diagnostics.diagnostics_passed && bad.diagnostics.max_depth_hits>0
    @test size(bad.z,2)==32 # failure may stop before the minimum SUCCESS length
    @test all(length(s)==32 for s in bad.diagnostics.statistics)

    single=sample_posterior(p;warmup=50,samples=16,chains=1,seed=2731)
    @test size(single.z)==(1,16,1) && single.diagnostics.stopping_reason===:fixed_length
    @test !single.diagnostics.diagnostics_passed
end

@testset "Controller input contracts and independent reassessment" begin
    p=sampling_location_fixture()
    @test_throws ArgumentError sample_posterior(p;stopping=:unknown)
    @test_throws ArgumentError sample_posterior(p;stopping=:diagnostics,samples=100)
    @test_throws ArgumentError sample_posterior(p;stopping=:fixed)
    @test_throws ArgumentError sample_posterior(p;chains=1)
    @test_throws ArgumentError sample_posterior(p;min_samples=4)
    @test_throws ArgumentError sample_posterior(p;batch_samples=0)
    @test_throws ArgumentError sample_posterior(p;max_samples=100,min_samples=200)
    @test_throws ArgumentError sample_posterior(p;consecutive_passes=0)
    @test_throws ArgumentError sample_posterior(p;mcse_limits=(unknown=.1,))
    @test_throws ArgumentError sample_posterior(p;diagnostic_quantities=(a=identity,),mcse_limits=(a=-.1,))
    @test_throws ArgumentError sample_posterior(p;diagnostic_quantities=(a=identity,),
        mcse_limits=(a=(quantiles=(1.0=>.1,),),))
    @test_throws ArgumentError sample_posterior(p;diagnostic_quantities=(a=identity,),
        mcse_limits=(a=(quantiles=(.5=>.1,.5=>.2),),))
    @test_throws ArgumentError sample_posterior(p;warmup=50,samples=8,chains=1,
        diagnostic_quantities=(bad=theta->NaN,))
    @test_throws ErrorException sample_posterior(p;warmup=50,samples=8,chains=1,
        diagnostic_quantities=(bad=theta->error("callback programming error"),))
    s=sample_posterior(p;warmup=150,samples=100,chains=2,seed=2741)
    report=diagnostics(s.z,s.diagnostics.statistics;max_depth=s.diagnostics.max_depth)
    @test report.rhat==MCMCDiagnosticTools.rhat(permutedims(s.z,(2,3,1));kind=:rank)
    @test report.ess_tail==MCMCDiagnosticTools.ess(permutedims(s.z,(2,3,1));kind=:tail)
    @test report.energy==s.diagnostics.energy
    @test_throws DimensionMismatch diagnostics(s.z,[s.diagnostics.statistics[1]])
    @test_throws ArgumentError diagnostics(s.z,[s.diagnostics.warmup_statistics[1][1:100],
        s.diagnostics.warmup_statistics[2][1:100]])
    @test_throws ArgumentError diagnostics(s.z,s.diagnostics.statistics;
        quantity_values=(bad=zeros(99,2),))
end

@testset "Mapped posterior and a native PBM use the controller" begin
    p=sampling_location_fixture(native=true)
    s=sample_posterior(p;initial=[1.],warmup=150,chains=2,seed=2751,target_acceptance=.9,
        min_samples=200,batch_samples=100,max_samples=1500,min_ess=100,rhat_limit=1.03,max_depth=10)
    @test s.diagnostics.diagnostics_passed && s.diagnostics.forward_solves>0
    @test abs(mean(s.z)-1.)<6only(s.diagnostics.mcse_mean)

    # A physical lognormal prior plus its transform Jacobian gives an exact
    # normal posterior in z. An auxiliary sinh map has a NONconstant Jacobian.
    pb=PBMProblem(compile(PBMModel(),Grid1D([0.,.5,1.])),[0.,0.],(0.,1.))
    m=Measurement([1.],[0.],FunctionObservable((u,op,p,t)->zero(p.rate));
        likelihood=GaussianLikelihood(sigma=1.))
    ex=InferenceExperiment(:batch,pb,m;solver=(p,ts)->PBMSolution(ts,[copy(p.u0) for _ in ts],p,0))
    spec=ParameterSpec(:rate;initial=1.,transform=LogTransform(),
        prior=Prior(LogNormal(0.,.5);coordinates=:physical))
    problem=EstimationProblem((spec,),(ex,))
    mapping=(to_sampler=z->asinh.(z),from_sampler=y->sinh.(y),
        logabsdetjac=y->sum(log.(cosh.(y))),label=:controlled_sinh)
    post=sample_posterior(problem;initial=[0.],sampling_map=mapping,
        initial_covariance=reshape([.25],1,1),warmup=250,chains=2,seed=2752,
        target_acceptance=.95,max_depth=10,min_samples=300,batch_samples=100,max_samples=2000,
        min_ess=150,rhat_limit=1.03,diagnostic_quantities=(rate=theta->theta.shared.rate,))
    d=diagnostics(post)
    @test d.diagnostics_passed
    @test abs(mean(post.z))<6only(d.mcse_mean)
    @test abs(var(vec(post.z))-.25)<.06
    @test abs(mean(d.quantity_diagnostics.rate.values)-exp(.5^2/2))<
        6d.quantity_diagnostics.rate.mcse_mean
    for c in 1:2,i in (1,d.samples÷2,d.samples)
        z=post.z[:,i,c];stat=d.statistics[c][i]
        @test stat.log_density≈sampling_logposterior(problem,z) atol=1e-10
        @test stat.sampler_log_density-stat.log_density≈mapping.logabsdetjac(mapping.to_sampler(z)) atol=1e-10
        @test stat.hamiltonian_energy==d.energy[i,c]
    end
end
