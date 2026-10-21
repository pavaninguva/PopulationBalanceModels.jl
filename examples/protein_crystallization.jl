module ProteinCrystallization
using PopulationBalanceModels
using Distributions: Uniform, Normal
export parameters,build,volume_median,observations,estimation_problem,run_study

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
        solver=nothing,dt=.5,maxiters=200)
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
    # Weak posterior curvature is reported, never repaired into confidence.
    try
        lap=laplace_approximation(problem,fitresult)
        open(joinpath(output,"fit.txt"),"a") do io;println(io,"Laplace: ",lap);end
    catch e
        e isa LaplaceFailure || rethrow()
        open(joinpath(output,"fit.txt"),"a") do io;println(io,"Laplace unavailable: ",sprint(showerror,e));end
    end
    fitresult
end
if abspath(PROGRAM_FILE)==@__FILE__
    @eval using Optim
    run_study()
end
end
