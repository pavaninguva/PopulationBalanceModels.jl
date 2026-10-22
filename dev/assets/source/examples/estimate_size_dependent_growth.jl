module EstimateSizeDependentGrowth
using PopulationBalanceModels, Random, LinearAlgebra
using Distributions: Normal, LogNormal, Uniform, pdf, cdf, ccdf, quantile, truncated

export synthetic_measurements, estimation_problem, run_example, refit_numerics,
       exact_density, exact_counts, exact_moment, physical_problem

const TRUTH=(g0=0.434,g1=0.2604)
const TIMES=[0.1,0.25,0.4,0.6,0.8,1.0]
const SIZES=collect(range(0.0,1.4;length=281))
const CENTER=0.2
const WIDTH=sqrt(0.0005/2)
const INITIAL_NUMBER=50sqrt(π*0.0005)
const UNIFORM_BOUNDS=((0.05,1.0),(0.02,1.0))
const LAW=AffineGrowth(:g0,:g1)
initial_density(a)=50exp(-(a-CENTER)^2/0.0005)
# Independent closed-form reference for data generation, not a discretized solve.
shift(t,p)=iszero(p.g1) ? p.g0*t : p.g0*expm1(p.g1*t)/p.g1
function exact_density(a,t,p=TRUTH)
    foot=(a-shift(t,p))*exp(-p.g1*t)
    foot<0 ? 0.0 : initial_density(foot)*exp(-p.g1*t)
end
function initial_integral(lo,hi)
    hi<=lo && return 0.0
    d=Normal(CENTER,WIDTH)
    INITIAL_NUMBER*(lo>CENTER ? ccdf(d,lo)-ccdf(d,hi) : cdf(d,hi)-cdf(d,lo))
end
function initial_moment(k)
    a=-CENTER/WIDTH;phi=pdf(Normal(),a);integrals=[ccdf(Normal(),a),phi]
    for j in 2:k;push!(integrals,a^(j-1)*phi+(j-1)*integrals[j-1]);end
    INITIAL_NUMBER*sum(binomial(k,j)*CENTER^(k-j)*WIDTH^j*integrals[j+1] for j in 0:k)
end
function exact_counts(edges,t,p=TRUTH)
    inverse(a)=max(0.0,(a-shift(t,p))*exp(-p.g1*t))
    [initial_integral(inverse(a),inverse(b)) for (a,b) in zip(edges[1:end-1],edges[2:end])]
end
function exact_moment(k,t,p=TRUTH)
    growth=exp(p.g1*t);translation=shift(t,p)
    sum(binomial(k,j)*growth^j*translation^(k-j)*initial_moment(j) for j in 0:k)
end

"""Evaluate analytical densities and sample independent Gaussian observations conditioned to be nonnegative. The fitted likelihood includes this truncation; no clipping, counts or renormalization are used."""
function synthetic_measurements(;seed=2022,noise=1.0,sizes=SIZES,times=TIMES)
    noise>0 && isfinite(noise) || throw(ArgumentError("noise must be positive"))
    truth=[exact_density(a,t) for a in sizes,t in times]
    rng=MersenneTwister(seed)
    values=[rand(rng,truncated(Normal(truth[i,j],noise),0.0,Inf)) for i in axes(truth,1),j in axes(truth,2)]
    (;values,truth,noise,sizes=collect(sizes),times=collect(times),seed,noise_model=:truncated_gaussian)
end

function physical_problem(;p=TRUTH,forward=:analytic,bins=280,upper=1.4,dt=0.005,scheme=MUSCL())
    boundary=SizeBC(OpenBoundary(),OpenBoundary())
    model=PBMModel(growth=Growth(LAW))
    op=if forward in (:analytic,:inguva)
        grid=characteristic_grid(LAW,(0.0,upper);dt,p)
        compile(model,grid;evolution=Inguva2022(),p,bc=boundary)
    elseif forward===:sectional
        grid=Grid1D(range(0.0,upper;length=bins+1);coordinate=:length)
        compile(model,grid;methods=TermMethods(growth=scheme),bc=boundary)
    else
        throw(ArgumentError("forward must be :analytic, :inguva or :sectional"))
    end
    PBMProblem(op,exact_counts(op.grid.edges,0.0,p),(0.0,1.0);p)
end

function analytical_solver(pb,times)
    states=[reshape(exact_counts(pb.operator.grid.edges,t,pb.p),:,1) for t in times]
    PBMSolution(times,states,pb,0)
end
# Point observations from a bin solver require an explicit reconstruction. This
# linear interpolation of bin-average densities is NOT an exact continuous PSD.
function sample_sectional(N,op,points)
    g=op.grid;f=vec(density(g,N))
    map(points) do x
        i=searchsortedlast(g.centers,x)
        i==0 && return first(f)
        i>=length(f) && return last(f)
        w=(x-g.centers[i])/(g.centers[i+1]-g.centers[i])
        (1-w)*f[i]+w*f[i+1]
    end
end

"""
Two fitted kinetic parameters; the initial density and truncated Gaussian error scale are known.
`prior=:lognormal` retains the previous growth priors. `prior=:uniform` uses
physical bounds from stated growth/stretching timescales. Both use log inference
coordinates, so uniform physical priors still receive their Jacobians.
The default forward calculation evaluates the analytical continuous density.
The optional bin solvers rebuild the characteristic mesh when parameters change.
"""
function estimation_problem(data=synthetic_measurements();prior=:lognormal,forward=:analytic,bins=280,upper=1.4,dt=0.005,scheme=MUSCL())
    prior in (:lognormal,:uniform) || throw(ArgumentError("unknown prior scenario"))
    base=physical_problem(;forward,bins,upper,dt,scheme)
    observable=forward===:analytic ?
        FunctionObservable((N,op,p,t)->exact_density.(data.sizes,t,Ref(p))) :
        FunctionObservable((N,op,p,t)->sample_sectional(N,op,data.sizes))
    measurement=Measurement(data.times,data.values,observable;
        likelihood=TruncatedGaussianLikelihood(sigma=data.noise),source=:density_measurements)
    setup=(theta,pb)->forward===:inguva ? physical_problem(;p=theta,forward,bins,upper,dt,scheme) : remake(pb;p=theta)
    solver=forward===:analytic ? analytical_solver : forward===:inguva ?
        ((pb,ts)->simulate(pb;dt=pb.operator.dt,saveat=ts)) : nothing
    ex=InferenceExperiment(:growth,base,measurement;setup,solver,dt=forward===:sectional ? dt : nothing)
    distributions=prior===:lognormal ? (LogNormal(log(0.4),0.4),LogNormal(log(0.25),0.6)) :
        (Uniform(UNIFORM_BOUNDS[1]...),Uniform(UNIFORM_BOUNDS[2]...))
    specs=[ParameterSpec(name;initial,transform=LogTransform(),prior=Prior(d;coordinates=:physical))
        for (name,initial,d) in zip((:g0,:g1),(0.40,0.23),distributions)]
    EstimationProblem(specs,(ex,);derivatives=forward===:inguva ? :finite_difference : :auto)
end

function write_table(path,header,rows)
    open(path,"w") do io
        println(io,header)
        for row in rows;println(io,join(row,','));end
    end
end

"""Refit the same continuous measurements with numerical solvers and compare against the analytical posterior. The Inguva mesh and time step are refined TOGETHER to retain CFL=1."""
function refit_numerics(problem,fitted,data;output)
    variants=(inguva_010=(forward=:inguva,dt=0.01),inguva_005=(forward=:inguva,dt=0.005),
        space_280=(forward=:sectional,bins=280,dt=0.001),
        space_560=(forward=:sectional,bins=560,dt=0.001),
        time_half=(forward=:sectional,bins=560,dt=0.0005),
        domain_larger=(forward=:sectional,bins=800,upper=2.0,dt=0.0005))
    rows=NamedTuple[];reference=predictions(problem,fitted.minimizer)[1][1]
    sd=sqrt.(diag(fitted.uncertainty.covariance))
    for (name,options) in pairs(variants)
        ep=estimation_problem(data;options...)
        r=fit(ep;initial=fitted.minimizer,scaling=sd,maxiters=200,derivative_step=1e-6)
        P=predictions(ep,fitted.minimizer)[1][1]
        push!(rows,(;name,converged=r.converged,shift=(r.minimizer-fitted.minimizer)./sd,
            max_error_sigma=maximum(abs,P-reference)/data.noise,
            gradient=r.diagnostics.scaled_gradient_norm,solves=r.diagnostics.forward_solves))
        println("Numerical refit ",name,": converged=",r.converged,"; shift/SD=",last(rows).shift);flush(stdout)
    end
    write_table(joinpath(output,"refinement.csv"),"variant,converged,g0_shift_sd,g1_shift_sd,max_density_error_sigma,scaled_gradient,solves",
        [(r.name,r.converged,r.shift...,r.max_error_sigma,r.gradient,r.solves) for r in rows])
    rows
end

"""
Run both prior scenarios, with MAP, Laplace and four NUTS chains each. The runner
requires the diagnostic screen to pass before exporting posterior summaries.
It records per-chain traces/energies and R-hat, ESS, MCSE, BFMI, divergences and
adaptation statistics. These checks are evidence, not a convergence guarantee.
"""
function run_example(;output=joinpath(@__DIR__,"..","docs","src","assets","bayesian_growth"),
        samples=1500,warmup=1000,chains=4,refine=true)
    mkpath(output);data=synthetic_measurements();runs=Dict{Symbol,Any}()
    for (i,scenario) in enumerate((:lognormal,:uniform))
        problem=estimation_problem(data;prior=scenario)
        fitted=fit(problem;nstarts=3,seed=22,uncertainty=:laplace)
        fitted.converged || error("MAP failed for $scenario")
        println(scenario," MAP: ",fitted.parameters.shared);flush(stdout)
        # Disperse starts in BOTH principal directions of the local posterior.
        Z=fitted.minimizer.+3.0.*(fitted.uncertainty.factor.U\randn(MersenneTwister(80),2,chains))
        mcmc=sample_posterior(problem;initial=fitted,laplace=fitted.uncertainty,starts=collect(eachcol(Z)),
            samples,warmup,chains,seed=24+i,target_acceptance=0.9,max_depth=9)
        report=diagnostics(mcmc)
        println(scenario," sampler screen=",report.diagnostics_passed,"; Rhat=",report.rhat,
            "; ESS=",report.ess_bulk,"; BFMI=",report.ebfmi);flush(stdout)
        report.diagnostics_passed || error("Sampler diagnostics failed for $scenario; inspect the returned chain settings before interpreting uncertainty")
        laplace=parameter_draws(problem,fitted.uncertainty;n=1500,seed=23)
        profile=estimation_profile(problem,fitted,2,range(fitted.minimizer[2]-0.06,fitted.minimizer[2]+0.06;length=17);target=:likelihood)
        runs[scenario]=(;problem,fitted,mcmc,laplace,profile)
    end
    write_table(joinpath(output,"measurements.csv"),"time,size,data,truth",
        [(t,a,data.values[i,j],data.truth[i,j]) for (j,t) in enumerate(data.times) for (i,a) in enumerate(data.sizes)])
    write_table(joinpath(output,"residuals.csv"),"time,size,quantile_residual",
        [(t,a,quantile(Normal(),cdf(truncated(Normal(exact_density(a,t,runs[:lognormal].fitted.parameters.shared),data.noise),0.0,Inf),data.values[i,j])))
            for (j,t) in enumerate(data.times) for (i,a) in enumerate(data.sizes)])
    x=range(0.0,1.4;length=1401)
    write_table(joinpath(output,"curves.csv"),"prior,time,size,truth,map",
        [(s,t,a,exact_density(a,t),exact_density(a,t,r.fitted.parameters.shared))
            for (s,r) in sort!(collect(runs);by=first) for t in data.times for a in x])
    write_table(joinpath(output,"parameters.csv"),"prior,name,true,map,laplace_sd_z",
        [(s,p.name,getproperty(TRUTH,p.name),getproperty(r.fitted.parameters.shared,p.name),sqrt(r.fitted.uncertainty.covariance[i,i]))
            for (s,r) in sort!(collect(runs);by=first) for (i,p) in enumerate(r.problem.parameters)])
    write_table(joinpath(output,"draws.csv"),"prior,method,chain,draw,g0,g1",
        Iterators.flatten([Iterators.flatten(([(s,"laplace",0,i,p.shared.g0,p.shared.g1) for (i,p) in enumerate(r.laplace.physical)],
            [(s,"mcmc",c,i,p.shared.g0,p.shared.g1) for c in 1:chains for (i,p) in enumerate(r.mcmc.physical[:,c])]))
            for (s,r) in sort!(collect(runs);by=first)]))
    write_table(joinpath(output,"chains.csv"),"prior,chain,draw,g0,g1,log_posterior,energy,acceptance,step_size,tree_depth,leapfrog_steps,divergent",
        [(s,c,i,p.shared.g0,p.shared.g1,st.log_density,st.hamiltonian_energy,st.acceptance_rate,st.step_size,st.tree_depth,st.n_steps,st.numerical_error)
            for (s,r) in sort!(collect(runs);by=first) for c in 1:chains for i in 1:samples
            for p in (r.mcmc.physical[i,c],) for st in (r.mcmc.diagnostics.statistics[c][i],)])
    write_table(joinpath(output,"sampler_diagnostics.csv"),"prior,parameter,rhat,ess_bulk,ess_tail,mcse_mean_z,divergences,warmup_divergences,min_bfmi,passed",
        [(s,p.name,d.rhat[i],d.ess_bulk[i],d.ess_tail[i],d.mcse_mean[i],d.divergences,d.warmup_divergences,minimum(d.ebfmi),d.diagnostics_passed)
            for (s,r) in sort!(collect(runs);by=first) for d in (r.mcmc.diagnostics,) for (i,p) in enumerate(r.problem.parameters)])
    write_table(joinpath(output,"chain_diagnostics.csv"),"prior,chain,acceptance_rate,step_size,mean_leapfrog_steps,bfmi,divergences,warmup_divergences,max_depth_hits",
        [(s,c,d.acceptance_rate,d.step_size,d.mean_leapfrog_steps,d.ebfmi,d.divergences,d.warmup_divergences,d.max_depth_hits)
            for (s,r) in sort!(collect(runs);by=first) for (c,d) in enumerate(r.mcmc.diagnostics.per_chain)])
    write_table(joinpath(output,"profiles.csv"),"prior,log_g1,delta,converged",
        [(s,v,d,flag) for (s,r) in sort!(collect(runs);by=first) for (v,d,flag) in zip(r.profile.values,r.profile.delta,r.profile.converged)])
    write_table(joinpath(output,"priors.csv"),"prior,parameter,value,density",
        [(s,p.name,v,pdf(p.prior.distribution,v)) for (s,r) in sort!(collect(runs);by=first) for p in r.problem.parameters for v in range(0.001,1.2;length=601)])
    reference=runs[:lognormal]
    numerical=refine ? refit_numerics(reference.problem,reference.fitted,data;output) : nothing
    open(joinpath(output,"diagnostics.txt"),"w") do io
        println(io,"SYNTHETIC DATA: nonnegative truncated-Gaussian observations of analytical affine-growth density, underlying sigma=",data.noise)
        println(io,"Inguva, Schickel & Braatz (2022), Case 2; doi:10.1016/j.compchemeng.2022.107808")
        println(io,"f0=50 exp(-(a-.2)^2/.0005); g0=.434; g1=.2604. Scaled size/time units from the benchmark.")
        println(io,length(data.sizes)," density observation positions at each of six times; no sampled counts or binning.")
        println(io,"Initial density and noise are fixed. Only g0 and g1 are fitted. All priors are physical-coordinate densities, with log-coordinate Jacobians exactly once.")
        println(io,"Uniform bounds: g0 in [0.05,1], g1 in [0.02,1], corresponding at reference size 1 to offset times 1–20 and stretching times 1–50.")
        for (s,r) in sort!(collect(runs);by=first)
            println(io,s," MAP=",r.fitted.parameters.shared,"; converged=",r.fitted.converged,
                "; derivative backend=",r.fitted.diagnostics.derivative_backend,"; gradient=",r.fitted.diagnostics.gradient_norm,"; Hessian agreement=",r.fitted.uncertainty.diagnostics.hessian_agreement)
            d=r.mcmc.diagnostics
            for k in (:rhat,:ess_bulk,:ess_tail,:mcse_mean,:ebfmi,:divergences,:warmup_divergences,:max_depth_hits,
                    :diagnostics_passed,:min_ess,:rhat_limit,:bfmi_limit,:target_acceptance,:warmup,:samples,:chains,:seed,:forward_solves,:elapsed_seconds)
                println(io,s," ",k," = ",getproperty(d,k))
            end
        end
        println(io,"Julia ",VERSION,"; data seed=2022, MAP seed=22, dispersed-start seed=80, Laplace seed=23.")
        ext=Base.get_extension(PopulationBalanceModels,:PopulationBalanceModelsAdvancedHMCExt)
        opt=Base.get_extension(PopulationBalanceModels,:PopulationBalanceModelsOptimExt)
        println(io,"Distributions=",Base.pkgversion(Distributions),"; Optim=",Base.pkgversion(opt.Optim),
            "; AdvancedHMC=",Base.pkgversion(ext.HMC),"; MCMCDiagnosticTools=",Base.pkgversion(ext.MCD))
    end
    (;data,runs,numerical)
end
if abspath(PROGRAM_FILE)==@__FILE__
    @eval using Optim, AdvancedHMC, MCMCDiagnosticTools
    run_example()
end
end
