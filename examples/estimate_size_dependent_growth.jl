module EstimateSizeDependentGrowth
using PopulationBalanceModels, Distributions, Random, LinearAlgebra

export synthetic_measurements, estimation_problem, run_example, refit_numerics, exact_counts, exact_moment

const TRUTH=(g0=0.434,g1=0.2604,initial_scale=1.0,noise=0.003)
const TIMES=[0.1,0.25,0.4,0.6,0.8,1.0]
const EDGES=collect(0.0:0.05:1.4)
const CENTER=0.2
const WIDTH=sqrt(0.0005/2)
const INITIAL_NUMBER=50sqrt(π*0.0005)
const C0=2.0
const MATERIAL=0.2

function initial_integral(lo,hi)
    hi<=lo && return 0.0
    distribution=Normal(CENTER,WIDTH)
    probability=lo>CENTER ? ccdf(distribution,lo)-ccdf(distribution,hi) : cdf(distribution,hi)-cdf(distribution,lo)
    INITIAL_NUMBER*probability
end
function initial_moment(k)
    a=-CENTER/WIDTH;phi=pdf(Normal(),a)
    integrals=[ccdf(Normal(),a),phi]
    for j in 2:k
        push!(integrals,a^(j-1)*phi+(j-1)*integrals[j-1])
    end
    INITIAL_NUMBER*sum(binomial(k,j)*CENTER^(k-j)*WIDTH^j*integrals[j+1] for j in 0:k)
end
"""Characteristic solution integrated over bins, with zero inflow at size zero."""
function exact_counts(edges,t,p)
    inverse(a)=max(0.0,a*exp(-p.g1*t)+p.g0*expm1(-p.g1*t)/p.g1)
    [p.initial_scale*initial_integral(inverse(a),inverse(b)) for (a,b) in zip(edges[1:end-1],edges[2:end])]
end
"""Exactly closed affine-growth moments, including the truncated initial Gaussian."""
function exact_moment(k,t,p)
    growth=exp(p.g1*t);shift=p.g0*expm1(p.g1*t)/p.g1
    p.initial_scale*sum(binomial(k,j)*growth^j*shift^(k-j)*initial_moment(j) for j in 0:k)
end
concentration(t,p)=C0-MATERIAL*(exact_moment(3,t,p)-exact_moment(3,0.0,p))

"""Seeded, explicitly synthetic snapshots: actual sampled particle counts and independent bulk measurements."""
function synthetic_measurements(;seed=2022,sampled_particles=400)
    rng=MersenneTwister(seed)
    counts=hcat([let n=exact_counts(EDGES,t,TRUTH);rand(rng,Multinomial(sampled_particles,n/sum(n)));end for t in TIMES]...)
    bulk=concentration.(TIMES,Ref(TRUTH))+TRUTH.noise*randn(rng,length(TIMES))
    (;counts,bulk,times=copy(TIMES),edges=copy(EDGES),seed,sampled_particles)
end

function physical_problem(;bins=280,upper=1.4,scheme=MUSCL(),edges=nothing)
    grid=Grid1D(edges===nothing ? range(0.0,upper;length=bins+1) : edges;coordinate=:length)
    model=PBMModel(growth=Growth(rate=(a,p,t,ctx)->p.g0+p.g1*a))
    operator=compile(model,grid;methods=TermMethods(growth=scheme),bc=SizeBC(NoFlux(),NoFlux()))
    system=PBMSystem((crystals=operator,);nonnegative_bulk=(:c,),
        exchanges=(ConservativeExchange(:crystals,:c,(a,p,t,ctx)->MATERIAL*a^3;mechanisms=(:growth,)),))
    initial=[initial_integral(a,b) for (a,b) in zip(grid.edges[1:end-1],grid.edges[2:end])]
    PBMProblem(system,(crystals=initial,),(0.0,1.0);bulk0=(c=C0,),p=TRUTH)
end
function analytical_solver(pb,times)
    grid=pb.operator.system.populations.crystals.grid
    states=[vcat(exact_counts(grid.edges,t,pb.p),concentration(t,pb.p)) for t in times]
    all(u->all(isfinite,u) && minimum(u)>=0,states) || throw(NumericalFailure("invalid characteristic state"))
    all(u->sum(u[1:end-1])>=(1-1e-8)*exact_moment(0,0.0,pb.p),states) || throw(NumericalFailure("characteristic population leaves the finite closed size domain; enlarge it"))
    PBMSolution(times,states,pb,0)
end

"""
Estimate shared growth coefficients and nuisance initial amplitude/noise from
one time series. The default analytical solver is a case-specific reference;
it is not a new general ExactShift capability. Sectional variants use the same
physical model, data, likelihoods and priors. Measurement edges must align with
solver edges in the analytical variant, whose stored values are exact bin integrals.
"""
function estimation_problem(data=synthetic_measurements();forward=:analytic,bins=280,upper=1.4,dt=0.002,scheme=MUSCL())
    forward in (:analytic,:sectional) || throw(ArgumentError("unknown forward calculation"))
    pb=physical_problem(;bins,upper,scheme,edges=forward===:analytic ? data.edges : nothing)
    if forward===:analytic
        grid=pb.operator.system.populations.crystals.grid
        all(e->any(x->isapprox(x,e;atol=1e-12),grid.edges),data.edges) || throw(ArgumentError("analytical histogram reference requires aligned measurement edges"))
    end
    histogram=DistributionObservable(data.edges;coordinate=:length,population=:crystals)
    sizes=Measurement(data.times,data.counts,histogram;likelihood=MultinomialLikelihood(),source=:particle_counter)
    bulk=Measurement(data.times,data.bulk,BulkObservable(:c);
        likelihood=GaussianLikelihood(sigma=(theta,experiment)->theta.noise),source=:independent_concentration_instrument)
    function setup(theta,base)
        initial=copy(base.u0)
        population(base,initial,:crystals).*=theta.initial_scale
        remake(base;u0=initial,p=theta)
    end
    ex=InferenceExperiment(:batch,pb,sizes,bulk;setup,
        solver=forward===:analytic ? analytical_solver : nothing,dt=forward===:sectional ? dt : nothing)
    specs=[ParameterSpec(:g0;initial=0.40,transform=LogTransform(),prior=Prior(LogNormal(log(0.4),0.4);coordinates=:physical)),
        ParameterSpec(:g1;initial=0.23,transform=LogTransform(),prior=Prior(LogNormal(log(0.25),0.6);coordinates=:physical)),
        ParameterSpec(:initial_scale;initial=1.0,transform=LogTransform(),role=:nuisance,prior=Prior(LogNormal(0,0.2);coordinates=:physical)),
        ParameterSpec(:noise;initial=0.003,transform=LogTransform(),role=:nuisance,prior=Prior(LogNormal(log(0.003),0.4);coordinates=:physical))]
    EstimationProblem(specs,(ex,))
end

function write_table(path,header,rows)
    open(path,"w") do io
        println(io,header)
        for row in rows;println(io,join(row,','));end
    end
end

"""Numerical refits with a finer derivative stencil, retaining failed strict stationarity checks. Reference uncertainty comes from the full joint analytical posterior."""
function refit_numerics(problem,fitted,data;output)
    variants=(space_280=(bins=280,upper=1.4,dt=0.002),space_560=(bins=560,upper=1.4,dt=0.002),
        time_half=(bins=560,upper=1.4,dt=0.001),domain_larger=(bins=800,upper=2.0,dt=0.001))
    numerical=NamedTuple[];reference=predictions(problem,fitted.minimizer)
    sd=sqrt.(diag(fitted.uncertainty.covariance))
    for (name,options) in pairs(variants)
        ep=estimation_problem(data;forward=:sectional,options...)
        r=fit(ep;initial=fitted.minimizer,maxiters=200,derivative_step=1e-6,scaling=sd)
        P=predictions(ep,fitted.minimizer)
        histogram_distance=maximum(sqrt(sum(data.counts[:,j]))*
            norm(sqrt.(P[1][1][:,j]/sum(P[1][1][:,j]))-sqrt.(reference[1][1][:,j]/sum(reference[1][1][:,j]))) for j in axes(data.counts,2))
        bulk_noise_change=maximum(abs.(P[1][2]-reference[1][2]))/fitted.parameters.shared.noise
        push!(numerical,(name=name,converged=r.converged,z=r.minimizer,
            shift=(r.minimizer-fitted.minimizer)./sd,histogram_distance,bulk_noise_change,
            gradient_norm=r.diagnostics.gradient_norm,scaled_gradient_norm=r.diagnostics.scaled_gradient_norm,
            derivative_step=r.diagnostics.derivative_step,
            solves=r.diagnostics.forward_solves,elapsed=r.diagnostics.elapsed_seconds))
        println("Numerical refit ",name,": converged=",r.converged,
            ", scaled gradient=",r.diagnostics.scaled_gradient_norm)
        flush(stdout)
    end
    write_table(joinpath(output,"refinement.csv"),"variant,converged,g0_shift_sd,g1_shift_sd,initial_shift_sd,noise_shift_sd,histogram_sqrtN_hellinger,bulk_change_noise_sd,gradient_norm,scaled_gradient_norm,derivative_step,solves,seconds",
        [(r.name,r.converged,r.shift...,r.histogram_distance,r.bulk_noise_change,r.gradient_norm,r.scaled_gradient_norm,r.derivative_step,r.solves,r.elapsed) for r in numerical])
    numerical
end

"""
Reproducible example budget: MAP with three seeded starts; 4 NUTS chains of
500 warmup + 1000 retained draws by default; 40 predictive reference solves.
Spatial, time and domain refits are optional and explicitly recorded. Load
Optim, AdvancedHMC and MCMCDiagnosticTools in the working environment first.
"""
function run_example(;output=joinpath(@__DIR__,"..","docs","src","assets","bayesian_growth"),
        samples=1000,warmup=500,chains=4,refine=true)
    data=synthetic_measurements();problem=estimation_problem(data)
    fitted=fit(problem;nstarts=3,seed=22,uncertainty=:laplace)
    fitted.converged || error("MAP failed to converge")
    println("MAP physical parameters: ",fitted.parameters.shared)
    flush(stdout)
    lap=fitted.uncertainty
    draws=parameter_draws(problem,lap;n=1000,seed=23)
    mcmc=sample_posterior(problem;initial=fitted,laplace=lap,samples,warmup,chains,seed=24)
    println("NUTS diagnostic screen passed: ",mcmc.diagnostics.diagnostics_passed)
    flush(stdout)
    predictive=posterior_predictive(problem,parameter_draws(problem,mcmc;n=40,seed=25);
        max_solves=40,measurement_noise=true,seed=26)
    predictive.failed_draws==0 || error("predictive solves failed")
    profiles=(likelihood=estimation_profile(problem,fitted,2,range(fitted.minimizer[2]-0.4,fitted.minimizer[2]+0.4;length=15);target=:likelihood),
        posterior=estimation_profile(problem,fitted,2,range(fitted.minimizer[2]-0.4,fitted.minimizer[2]+0.4;length=15);target=:posterior))
    broader=replace_priors(problem,[Prior(LogNormal(log(0.4),0.8);coordinates=:physical),
        Prior(LogNormal(log(0.25),1.2);coordinates=:physical),problem.parameters[3].prior,problem.parameters[4].prior])
    alternative=fit(broader;initial=fitted.minimizer,uncertainty=:laplace)
    mkpath(output)
    write_table(joinpath(output,"parameters.csv"),"name,true,map,laplace_sd_z,alternative_map,alternative_sd_z",
        [(p.name,getproperty(TRUTH,p.name),getproperty(fitted.parameters.shared,p.name),sqrt(lap.covariance[i,i]),
          getproperty(alternative.parameters.shared,p.name),sqrt(alternative.uncertainty.covariance[i,i])) for (i,p) in enumerate(problem.parameters)])
    write_table(joinpath(output,"covariance.csv"),"row,column,covariance_z",
        [(i,j,lap.covariance[i,j]) for i in axes(lap.covariance,1),j in axes(lap.covariance,2)])
    write_table(joinpath(output,"draws.csv"),"method,draw,g0,g1,initial_scale,noise",
        Iterators.flatten(([("laplace",i,values(p.shared)...) for (i,p) in enumerate(draws.physical)],
            [("mcmc",i,values(p.shared)...) for (i,p) in enumerate(vec(mcmc.physical))])))
    write_table(joinpath(output,"measurements.csv"),"time,lower,upper,count,model_probability",
        [(t,data.edges[i],data.edges[i+1],data.counts[i,j],exact_counts(data.edges,t,fitted.parameters.shared)[i]/sum(exact_counts(data.edges,t,fitted.parameters.shared)))
          for (j,t) in enumerate(data.times) for i in axes(data.counts,1)])
    write_table(joinpath(output,"bulk.csv"),"time,data,map,truth",
        [(t,data.bulk[i],concentration(t,fitted.parameters.shared),concentration(t,TRUTH)) for (i,t) in enumerate(data.times)])
    write_table(joinpath(output,"predictive.csv"),"draw,time,latent_concentration,noisy_concentration",
        [(i,t,r.measurements[1][2].latent[1,j],r.measurements[1][2].observed[1,j]) for (i,r) in enumerate(predictive.results) for (j,t) in enumerate(data.times)])
    write_table(joinpath(output,"profiles.csv"),"target,log_g1,delta,converged",
        [(name,v,d,flag) for (name,p) in pairs(profiles) for (v,d,flag) in zip(p.values,p.delta,p.converged)])
    numerical=NamedTuple[]
    if refine
        numerical=refit_numerics(problem,fitted,data;output)
    end
    open(joinpath(output,"diagnostics.txt"),"w") do io
        println(io,"SYNTHETIC DATA: affine growth adapted from Inguva, Schickel & Braatz (2022), Case 2. doi:10.1016/j.compchemeng.2022.107808")
        println(io,"Paper growth g0=.434, g1=.2604; f0=50 exp(-(a-.2)^2/.0005), a>=0. Size/time units follow scaled benchmark coordinates.")
        println(io,"Added demonstration bulk balance c'= -0.2 mu3', c0=2; independent Gaussian sigma=.003. Not experimental paper data.")
        println(io,"400 sampled particles at each of six times, conditional on [0,1.4]; joint unknowns log(g0), log(g1), log(initial_scale), log(noise).")
        println(io,"MAP coordinates define the posterior density. Physical LogNormal priors include exactly one transform Jacobian.")
        println(io,"MAP converged=",fitted.converged,"; gradient infinity norm=",fitted.diagnostics.gradient_norm,"; solves=",fitted.diagnostics.forward_solves)
        println(io,"MAP scaled gradient=",fitted.diagnostics.scaled_gradient_norm,
            "; tolerance=",fitted.diagnostics.gradient_tolerance,"; derivative step=",fitted.diagnostics.derivative_step,
            "; optimizer scales=",fitted.diagnostics.scaling,"; elapsed seconds=",fitted.diagnostics.elapsed_seconds)
        println(io,"Laplace Hessian agreement=",lap.diagnostics.hessian_agreement,"; condition=",lap.diagnostics.condition_number)
        for k in (:ess_bulk,:ess_tail,:rhat,:divergences,:warmup_divergences,:max_depth_hits,:diagnostics_passed,
                :min_ess,:rhat_limit,:target_acceptance,:metric,:max_depth,:warmup,:samples,:chains,
                :posterior_evaluations,:forward_solves,:failed_evaluations,:failure_messages,:elapsed_seconds)
            println(io,"MCMC ",k," = ",getproperty(mcmc.diagnostics,k))
        end
        println(io,"Predictive propagation uses 40 additional reference solves, returning latent and noisy values separately.")
        println(io,"Julia ",VERSION,"; RNG seeds data=2022, MAP=22, Laplace=23, NUTS=24, predictive=25/26.")
        sampler_extension=Base.get_extension(PopulationBalanceModels,:PopulationBalanceModelsAdvancedHMCExt)
        optimizer_extension=Base.get_extension(PopulationBalanceModels,:PopulationBalanceModelsOptimExt)
        println(io,"Versions: Distributions=",Base.pkgversion(Distributions),
            "; Optim=",Base.pkgversion(optimizer_extension.Optim),
            "; AdvancedHMC=",Base.pkgversion(sampler_extension.HMC),
            "; MCMCDiagnosticTools=",Base.pkgversion(sampler_extension.MCD))
    end
    println("MAP physical parameters: ",fitted.parameters.shared)
    println("Sampler diagnostics passed: ",mcmc.diagnostics.diagnostics_passed,"; Rhat=",mcmc.diagnostics.rhat)
    (;problem,fitted,mcmc,predictive,numerical,profiles,alternative)
end

if abspath(PROGRAM_FILE)==@__FILE__
    @eval using Optim, AdvancedHMC, MCMCDiagnosticTools
    run_example()
end
end
