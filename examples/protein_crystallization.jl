module ProteinCrystallization
using PopulationBalanceModels
using Distributions: Uniform, Normal, rand, pdf, logpdf
using Random, LinearAlgebra
import ForwardDiff
export parameters,build,volume_median,observations,estimation_problem,run_study,run_posterior,scheduled_solver,physical_prior_density

# Time is minutes, length µm, c is g/L (=kg/m³); N is in 10^12 particles/m³.
# Table 7 Case 2: AJ has units m^-3 s^-1; AG is printed in nm/min.
parameters()=(logAJ=18.9,j=2.14,AG=.367,g=2.45,ceq=2.16,rho=1240.,kv=.81)
saturation(p,ctx)=max(ctx.bulk.c/p.ceq-1,zero(ctx.bulk.c))
growth(L,p,t,ctx)=1e-3*p.AG*saturation(p,ctx)^p.g
nucleation(p,t,ctx)=60/1e12*exp(p.logAJ)*saturation(p,ctx)^p.j

"""Unseeded isothermal protein crystallization, power-law kinetics (paper Case 2).
Rates are size-independent; no aggregation, breakage, dissolution or shear model.
"""
function build(;bins=100,Lmax=40.,p=parameters(),c0=19.9883,tf=700.,scheme=MUSCL())
    grid=Grid1D(range(0.,Lmax;length=bins+1);coordinate=:length)
    model=PBMModel(growth=Growth(rate=growth),primary=Nucleation(rate=nucleation,size=0.))
    op=compile(model,grid;methods=TermMethods(growth=scheme,by_term=(primary=BoundaryInjection(:left),)))
    system=PBMSystem((crystals=op,);nonnegative_bulk=(:c,),exchanges=(
        ConservativeExchange(:crystals,:c,(L,p,t,ctx)->1e-6*p.rho*p.kv*L^3),))
    PBMProblem(system,(crystals=zeros(bins),),(0.,tf);bulk0=(c=c0,),p)
end

"""Volume-weighted D50 of the declared piecewise-constant sectional density.
Integrate L³ over each bin, then invert its cumulative integral. This is a median,
not a ratio of moments. A raw-moment closure needs a separate reconstruction.
"""
function volume_median(u,op,p,t)
    g=op.system.populations.crystals.grid;N=population(op,u,:crystals)
    mass=N.*(g.edges[2:end].^4-g.edges[1:end-1].^4)./(4 .*g.widths)
    total=sum(mass)
    total>0 || throw(NumericalFailure("D50 is undefined for an empty population"))
    all(>=(-1e-10),mass) || throw(NumericalFailure("invalid negative particle volume"))
    target=total/2;running=zero(total)
    for i in eachindex(mass)
        if running+mass[i]>=target
            fraction=(target-running)/mass[i]
            return (g.edges[i]^4+fraction*(g.edges[i+1]^4-g.edges[i]^4))^(1/4)
        end
        running+=mass[i]
    end
    throw(NumericalFailure("D50 inversion failed"))
end

function observations()
    lines=readlines(joinpath(@__DIR__,"data","protein_crystallization","figure4_observations.csv"))[2:end]
    rows=[split(line,',') for line in lines]
    function selected(name)
        r=filter(x->x[1]==name,rows)
        (;t=[parse(Float64,x[2]) for x in r],y=[parse(Float64,x[3]) for x in r],sigma=parse(Float64,first(r)[4]))
    end
    (;concentration=selected("concentration"),D50=selected("D50"))
end

"""One posterior for graph-read concentration and volume-median observations.
Table 3 supplies starts/bounds, not priors. Here logAJ∈[0,100], j∈[1,10], g∈[1,3]
are explicit uniform-prior assumptions; log(AG) has a weak normal prior centred
on Table 7's Case 2 estimate. Change growth_prior_sd to examine that assumption.
The two measurements have independent Gaussian errors, interpreting the paper's
0.1 g/L and 0.5 µm as standard deviations. Raw instrument covariance is absent.
"""
function estimation_problem(;bins=100,solver=nothing,dt=.5,growth_prior_sd=1.5)
    p=parameters();problem=build(;bins,p);data=observations()
    concentration=Measurement(data.concentration.t,data.concentration.y,BulkObservable(:c);
        likelihood=GaussianLikelihood(sigma=data.concentration.sigma),source=:figure4_concentration)
    sizes=Measurement(data.D50.t,data.D50.y,FunctionObservable(volume_median);
        likelihood=GaussianLikelihood(sigma=data.D50.sigma),source=:figure4_D50)
    specs=(
        ParameterSpec(:logAJ;initial=p.logAJ,prior=Prior(Uniform(0,100);coordinates=:physical)),
        ParameterSpec(:j;initial=p.j,transform=LogitTransform(1,10),prior=Prior(Uniform(1,10);coordinates=:physical)),
        ParameterSpec(:AG;initial=p.AG,transform=LogTransform(),prior=Prior(Normal(log(p.AG),growth_prior_sd);coordinates=:inference)),
        ParameterSpec(:g;initial=p.g,transform=LogitTransform(1,3),prior=Prior(Uniform(1,3);coordinates=:physical)))
    ex=InferenceExperiment(:batch,problem,concentration,sizes;dt=solver===nothing ? dt : nothing,solver)
    EstimationProblem(specs,(ex,))
end

function run_study(;output=joinpath(@__DIR__,"..","docs","src","assets","protein_crystallization"),bins=80,
        solver=nothing,dt=.5,maxiters=200,posterior=false,posterior_options=NamedTuple())
    # Optional dependency loaded by the caller, preserving the package extension.
    Base.get_extension(PopulationBalanceModels,:PopulationBalanceModelsOptimExt)!==nothing ||
        error("Load Optim before run_study (see the complete-script invocation).")
    mkpath(output);problem=estimation_problem(;bins,solver,dt)
    fitresult=fit(problem;maxiters,gradient_tolerance=.02)
    theta=physical_parameters(problem,fitresult.minimizer,:batch)
    open(joinpath(output,"fit.txt"),"w") do io
        println(io,"Digitized experimental markers: Mitchell et al. (2023), Figure 4; no raw laboratory data.")
        println(io,"MAP converged: ",fitresult.converged,"; gradient norm: ",fitresult.diagnostics.gradient_norm)
        println(io,"Coordinates: identity(logAJ), logit(j), log(AG), logit(g); theta=",theta)
        println(io,"Posterior: ",posterior_components(problem,fitresult.minimizer))
        println(io,"Volume-weighted D50 is an explicit observation assumption; verify against the original instrument convention.")
    end
    times=collect(0.:5.:700.)
    open(joinpath(output,"predictions.csv"),"w") do io
        println(io,"method,bins,time_min,concentration_g_L,D50_um,material_drift")
        for (label,n,p) in (("reported_parameters",bins,parameters()),("MAP",bins,merge(parameters(),theta)),("MAP_refined",2bins,merge(parameters(),theta)))
            pb=build(bins=n,p=p)
            solution=solver===nothing ? simulate(pb;dt=dt/2,saveat=times) : solver(pb,times)
            grid=pb.operator.system.populations.crystals.grid
            for (t,u) in zip(solution.t,solution.u)
                D50=t==0 ? NaN : volume_median(u,pb.operator,p,t)
                drift=bulk(pb,u,:c)+1e-6*p.rho*p.kv*moment(grid,population(pb,u,:crystals),3)-pb.u0[end]
                println(io,join((label,n,t,bulk(pb,u,:c),D50,drift),','))
            end
        end
    end
    open(joinpath(output,"map_parameters.csv"),"w") do io
        println(io,"parameter,inference_coordinate,physical_value")
        for (i,spec) in enumerate(problem.parameters)
            println(io,join((spec.name,fitresult.minimizer[i],getproperty(theta,spec.name)),','))
        end
    end
    # Weak posterior curvature is reported, never repaired into confidence.
    try
        lap=laplace_approximation(problem,fitresult)
        open(joinpath(output,"fit.txt"),"a") do io;println(io,"Laplace: ",lap);end
    catch e
        e isa LaplaceFailure || rethrow()
        open(joinpath(output,"fit.txt"),"a") do io;println(io,"Laplace unavailable: ",sprint(showerror,e));end
    end
    posterior && run_posterior(;output,bins,initial=fitresult.minimizer,posterior_options...)
    fitresult
end

"""The same sectional PBM and SSPRK33 stages with a fixed time schedule.
Use 1-min steps to 50 min, 3-min steps to 150 min, then 10-min steps while
rates decline. Output times and schedule boundaries are reached exactly.
scale halves every scheduled step for a separate temporal accuracy check.
The schedule is parameter independent, so AD includes no hidden step controller.
Every native RK stage retains its population and bulk checks.
"""
function scheduled_solver(pb,times;scale=1.)
    isfinite(scale) && scale>0 || throw(ArgumentError("positive time scale required"))
    u=copy(pb.u0);t=first(pb.tspan);states=typeof(u)[];steps=0
    ts=collect(times)
    for target in ts
        while t<target
            finish=t<50 ? min(target,50.) : t<150 ? min(target,150.) : target
            dt=scale*(t<50 ? 1. : t<150 ? 3. : 10.)
            localpb=remake(pb;u0=u,tspan=(t,finish))
            sol=simulate(localpb;dt,saveat=[finish])
            u=last(sol.u);steps+=sol.steps;t=finish
        end
        push!(states,copy(u))
    end
    PBMSolution(ts,states,pb,steps)
end

"""Prior density in PHYSICAL coordinates, including the inverse Jacobian when
its declared density is in inference coordinates. In particular the Normal
prior on log(AG) becomes a lognormal density on AG. Plotting is not a second
application of the inference posterior Jacobian.
"""
function physical_prior_density(spec,x)
    if spec.prior.coordinates===:physical
        return pdf(spec.prior.distribution,x)
    end
    z=PopulationBalanceModels._coordinate(spec.transform,x)
    exp(logpdf(spec.prior.distribution,z)-PopulationBalanceModels._logjac(spec.transform,z))
end
"""Straighten kinetic coefficient/exponent correlations at a fixed reference
supersaturation. This is a sampler-only bijection with unit determinant;
the EstimationProblem and its declared inference-coordinate posterior are unchanged.
"""
function reference_sampling_map(problem;supersaturation=8.)
    supersaturation>0 || throw(ArgumentError("positive reference supersaturation required"))
    logref=log(supersaturation)
    function shifted(v,sign)
        out=copy(v)
        j=PopulationBalanceModels._physical(problem.parameters[2].transform,v[2])
        g=PopulationBalanceModels._physical(problem.parameters[4].transform,v[4])
        out[1]+=sign*j*logref;out[3]+=sign*g*logref
        out
    end
    (to_sampler=z->shifted(z,1),from_sampler=y->shifted(y,-1),
     logabsdetjac=y->zero(first(y)),label=:reference_supersaturation)
end

function prior_draw(rng,spec)
    v=rand(rng,spec.prior.distribution)
    spec.prior.coordinates===:physical ? v : PopulationBalanceModels._physical(spec.transform,v)
end
function saved_map(output,problem)
    file=joinpath(output,"map_parameters.csv")
    if isfile(file)
        return [parse(Float64,split(row,',')[2]) for row in readlines(file)[2:end]]
    end
    # Compatibility with earlier study exports. Read named numbers; never eval.
    content=read(joinpath(output,"fit.txt"),String)
    physical=[parse(Float64,match(Regex(string(spec.name)*raw" = ([+\-0-9.eE]+)"),content).captures[1]) for spec in problem.parameters]
    [PopulationBalanceModels._coordinate(spec.transform,v) for (spec,v) in zip(problem.parameters,physical)]
end
function _table(path,header,rows)
    open(path,"w") do io
        println(io,header)
        for row in rows;println(io,join(row,','));end
    end
end

"""Sample the same sectional likelihood and declared priors with optional NUTS.
The initial covariance is only a sampler-tuning matrix from likelihood
sensitivities plus prior curvature; it is NOT a verified Laplace approximation.
Chains, draws, proper physical-coordinate priors and diagnostic failures are
exported even when the convergence screen fails. Such draws are provisional.
"""
function run_posterior(;output=joinpath(@__DIR__,"..","docs","src","assets","protein_crystallization"),
        bins=80,initial=nothing,samples=750,warmup=500,chains=4,seed=2027,
        time_scale=1.,predictive_draws=100,min_ess=400,target_acceptance=.9,max_depth=9)
    Base.get_extension(PopulationBalanceModels,:PopulationBalanceModelsAdvancedHMCExt)!==nothing ||
        error("Load AdvancedHMC and MCMCDiagnosticTools before run_posterior")
    mkpath(output)
    solver=(pb,ts)->scheduled_solver(pb,ts;scale=time_scale)
    problem=estimation_problem(;bins,solver)
    center=initial===nothing ? saved_map(output,problem) : collect(initial)
    data=observations();scales=vcat(fill(data.concentration.sigma,length(data.concentration.t)),fill(data.D50.sigma,length(data.D50.t)))
    flattened=z->vcat((vec(v) for v in first(predictions(problem,z)))...)
    J=ForwardDiff.jacobian(flattened,center)./scales
    prior_curvature=ForwardDiff.hessian(z->-logprior(problem,z),center)
    precision=Symmetric(J'*J+prior_curvature)
    factor=cholesky(precision) # no ridge, no uncertainty assertion
    tuning_z=Matrix(Symmetric(factor\Matrix{Float64}(I,length(center),length(center))))
    sampling_map=reference_sampling_map(problem)
    B=ForwardDiff.jacobian(sampling_map.to_sampler,center)
    tuning=Matrix(Symmetric(B*tuning_z*B'))
    metric_factor=cholesky(Symmetric(tuning)).L
    rng=MersenneTwister(seed)
    starts=[sampling_map.from_sampler(sampling_map.to_sampler(center) .+ 0.4 .* (metric_factor*randn(rng,length(center)))) for _ in 1:chains]
    println("Protein posterior: ",chains," chains, ",warmup," warmup, ",samples," retained per chain");flush(stdout)
    draws=sample_posterior(problem;initial=center,starts,initial_covariance=tuning,
        samples,warmup,chains,seed,target_acceptance,max_depth,min_ess,metric=:dense,sampling_map,verbose=true)
    report=diagnostics(draws)
    names=Tuple(spec.name for spec in problem.parameters)
    _table(joinpath(output,"posterior_draws.csv"),"chain,draw,"*join(names,','),
        [(c,i,(getproperty(draws.physical[i,c].shared,name) for name in names)...) for c in 1:chains,i in 1:samples])
    _table(joinpath(output,"sampler_diagnostics.csv"),"parameter,rhat,ess_bulk,ess_tail,mcse_mean_z,passed",
        [(spec.name,report.rhat[i],report.ess_bulk[i],report.ess_tail[i],report.mcse_mean[i],report.diagnostics_passed) for (i,spec) in enumerate(problem.parameters)])
    _table(joinpath(output,"chain_diagnostics.csv"),"chain,acceptance,bfmi,divergences,warmup_divergences,max_depth_hits,step_size,mean_leapfrog_steps",
        [(c,d.acceptance_rate,d.ebfmi,d.divergences,d.warmup_divergences,d.max_depth_hits,d.step_size,d.mean_leapfrog_steps) for (c,d) in enumerate(report.per_chain)])
    _table(joinpath(output,"chains.csv"),"chain,draw,log_posterior,energy,acceptance,leapfrog_steps,tree_depth,divergent",
        [(c,i,d.log_density,d.hamiltonian_energy,d.acceptance_rate,d.n_steps,d.tree_depth,d.numerical_error) for c in 1:chains,i in 1:samples for d in (report.statistics[c][i],)])
    prior_values=[[prior_draw(rng,spec) for spec in problem.parameters] for _ in 1:10000]
    _table(joinpath(output,"prior_draws.csv"),join(names,','),prior_values)
    prior_grids=(logAJ=range(0.,100.;length=1001),j=range(1.,10.;length=1001),
        AG=exp.(range(log(.0005),log(200.);length=2001)),g=range(1.,3.;length=1001))
    _table(joinpath(output,"prior_density.csv"),"parameter,value,density",
        [(spec.name,x,physical_prior_density(spec,x)) for spec in problem.parameters for x in getproperty(prior_grids,spec.name)])
    center_prediction=flattened(center)
    observed=vcat(data.concentration.y,data.D50.y)
    _table(joinpath(output,"fit_residuals.csv"),"observable,time_min,data,prediction,standardized_residual",
        [(label,t,y,center_prediction[i],(center_prediction[i]-y)/scales[i])
         for (i,(label,t,y)) in enumerate(vcat([("concentration",t,y) for (t,y) in zip(data.concentration.t,data.concentration.y)],
                                              [("D50",t,y) for (t,y) in zip(data.D50.t,data.D50.y)]))])
    residual_rms=sqrt(sum(abs2,(center_prediction-observed)./scales)/length(observed))
    # Numerical differences in units of each measurement standard deviation.
    fine=estimation_problem(;bins,solver=(pb,ts)->scheduled_solver(pb,ts;scale=time_scale/2))
    reference=estimation_problem(;bins,dt=.5)
    selected=parameter_draws(problem,draws;n=min(12,samples*chains),seed=seed+1)
    checks=NamedTuple[]
    for (i,z) in enumerate(eachcol(hcat(center,selected.z)))
        base=flattened(z)
        F=vcat((vec(v) for v in first(predictions(fine,z)))...)
        R=vcat((vec(v) for v in first(predictions(reference,z)))...)
        push!(checks,(draw=i-1,half_step_error_sigma=maximum(abs.((F-base)./scales)),
            uniform_step_error_sigma=maximum(abs.((R-base)./scales)),
            logposterior_difference=logposterior(fine,z)-logposterior(problem,z)))
    end
    _table(joinpath(output,"posterior_time_refinement.csv"),"draw,half_step_error_sigma,uniform_step_error_sigma,logposterior_difference",[Tuple(x) for x in checks])
    count=min(predictive_draws,samples*chains)
    if count>0
        selected=parameter_draws(problem,draws;n=count,seed=seed+2)
        predictive=posterior_predictive(problem,selected;max_solves=count,measurement_noise=true,seed=seed+3)
        rows=Tuple[]
        for i in 1:count
            predictive.results[i]===nothing && continue
            blocks=first(predictive.results[i].measurements)
            for (block,datum,label) in zip(blocks,(data.concentration,data.D50),("concentration","D50"))
                for j in eachindex(datum.t)
                    push!(rows,(i,label,datum.t[j],datum.y[j],block.latent[j],block.observed[j]))
                end
            end
        end
        _table(joinpath(output,"posterior_predictive.csv"),"draw,observable,time_min,data,latent,observed",rows)
    end
    open(joinpath(output,"posterior.txt"),"w") do io
        println(io,"Same 80-bin (or configured bins=$bins) sectional physical model and priors as the MAP example; digitized data, not raw laboratory records.")
        println(io,"Native SSPRK33 scheduled steps=(1,3,10) min, breaks=(50,150) min, time scale=$time_scale. No gradient through a hidden adaptive controller.")
        println(io,"Sampler coordinates use logAJ+j*log(8), logit(j), logAG+g*log(8), logit(g). Unit determinant verified; stored z and prior definitions remain in declared inference coordinates.")
        println(io,"Initial inverse mass matrix from positive likelihood-sensitivity/prior curvature in sampler coordinates: tuning ONLY, not a Laplace covariance.")
        for key in (:diagnostics_passed,:rhat,:ess_bulk,:ess_tail,:mcse_mean,:ebfmi,:divergences,:warmup_divergences,:max_depth_hits,
                    :warmup,:samples,:chains,:seed,:min_ess,:target_acceptance,:max_depth,:elapsed_seconds,:forward_solves,:failed_evaluations)
            println(io,key," = ",getproperty(report,key))
        end
        println(io,"RMS standardized residual at initialization/MAP = ",residual_rms)
        println(io,"Largest half-step prediction change / sigma = ",maximum(c.half_step_error_sigma for c in checks))
        println(io,"Largest difference from uniform dt=.5 / sigma = ",maximum(c.uniform_step_error_sigma for c in checks))
        println(io,"A passed screen is evidence, not proof, of convergence; posterior intervals remain conditional on noise, priors, discretization and observation assumptions.")
    end
    println("Protein sampler passed=",report.diagnostics_passed,"; Rhat=",report.rhat,"; ESS=",report.ess_bulk);flush(stdout)
    (;problem,draws,checks)
end

if abspath(PROGRAM_FILE)==@__FILE__
    @eval using Optim
    if "--posterior-only" in ARGS || "--posterior" in ARGS
        @eval using AdvancedHMC, MCMCDiagnosticTools
    end
    "--posterior-only" in ARGS ? run_posterior() : run_study(posterior="--posterior" in ARGS)
end
end
