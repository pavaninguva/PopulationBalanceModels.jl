module CoolingCrystallizationBenchmark
using PopulationBalanceModels
include("papers/gunawan2004_case2.jl")
using .Gunawan2004Case2

benchmark_grid(n)=Grid1D(sort!(unique(vcat(collect(range(0,1600;length=n+1)),180.5,210.5)));coordinate=:length)
function continuum_moments(initial_density)
    grid=benchmark_grid(3200)
    [sum(initial_counts(grid,L->L^k*initial_density(L))) for k in 0:3]
end
# Compatibility for scripts using the earlier runner's helper.
seed_moments(f)=continuum_moments(f)

# Independent integration of the explicitly coded continuum moment equations.
function reference(initial,p,times;dt=0.05)
    u=copy(initial);t=0.0;states=Vector{Float64}[]
    k1=similar(u);k2=similar(u);k3=similar(u);k4=similar(u)
    for target in times
        while t<target
            h=min(dt,target-t)
            moment_rhs!(k1,u,p,t);moment_rhs!(k2,u.+h/2 .*k1,p,t+h/2)
            moment_rhs!(k3,u.+h/2 .*k2,p,t+h/2);moment_rhs!(k4,u.+h.*k3,p,t+h)
            u .+= h/6 .* (k1.+2k2.+2k3.+k4)
            t=h==target-t ? target : t+h
        end
        push!(states,copy(u))
    end
    states
end

"""Conservatively integrate a piecewise-constant fine-grid density into target bins."""
function rebin(source,N,target)
    out=zeros(length(target));j=1
    for i in eachindex(out)
        a,b=target.edges[i],target.edges[i+1]
        while j<length(source) && source.edges[j+1]<=a;j+=1;end
        k=j
        while k<=length(source) && source.edges[k]<b
            overlap=max(0.0,min(b,source.edges[k+1])-max(a,source.edges[k]))
            out[i]+=N[k]*overlap/source.widths[k];k+=1
        end
    end
    out
end

function sectional(n,scheme,f0,c0,tf,dt,times)
    grid=benchmark_grid(n)
    prob=build(;grid,initial_density=f0,c0,scheme,tspan=(0.0,tf))
    timed=@timed simulate(prob;dt,saveat=times)
    (;grid,problem=prob,solution=timed.value,seconds=timed.time)
end
final_numbers(run)=population(run.problem,last(run.solution.u),:crystals)
relative_difference(run,truth)=sum(abs.(final_numbers(run).-rebin(truth.grid,final_numbers(truth),run.grid)))/sum(final_numbers(truth))

function run_benchmark(;initial_density=benchmark_initial_density,c0=solubility(temperature(0.0)),
        provenance="Chosen initial condition: 0.08*(1-((L-195.5)/15)^2) on [180.5,210.5] micrometers; saturated solution at 32 C. Not a repair of Eq.34.",
        initial_moments=continuum_moments(initial_density),grids=(400,800,1600,3200),
        reference_bins=6400,check_bins=12800,dt=0.1,tf=1000.0,
        output=joinpath(@__DIR__,"..","docs","src","assets","gunawan_example2"))
    isfinite(c0) && c0>=0 || throw(ArgumentError("invalid initial concentration"))
    !isempty(strip(provenance)) || throw(ArgumentError("record the initial-condition provenance"))
    all(n->n isa Integer && n>=2,grids) && reference_bins>maximum(grids) && check_bins>reference_bins ||
        throw(ArgumentError("reference grids must be successively finer than the comparison grids"))
    all(x->isfinite(x) && x>=0,initial_density.(benchmark_grid(check_bins).centers)) ||
        throw(ArgumentError("initial density must be finite and nonnegative; the literal printed Eq.34 is retained separately"))
    mkpath(output);times=collect(range(0,tf;length=101));params=parameters()
    initial=vcat(initial_moments,c0)
    moments=reference(initial,params,times);moment_check=reference(initial,params,times;dt=0.025)
    moment_step_difference=maximum(abs.(last(moments).-last(moment_check))./max.(abs.(last(moment_check)),eps()))
    open(joinpath(output,"reference.csv"),"w") do io
        println(io,"time,mu0,mu1,mu2,mu3,concentration")
        for (t,u) in zip(times,moments);println(io,join((t,u...),','));end
    end
    println("Computing MUSCL reference and refinement checks...")
    coarse_ref=sectional(reference_bins,MUSCL(),initial_density,c0,tf,dt,times)
    truth=sectional(check_bins,MUSCL(),initial_density,c0,tf,dt,times)
    time_check=sectional(reference_bins,MUSCL(),initial_density,c0,tf,dt/2,times)
    space_error=relative_difference(coarse_ref,truth)
    time_error=relative_difference(coarse_ref,time_check)
    open(joinpath(output,"reference_checks.csv"),"w") do io
        println(io,"check,coarse_bins,fine_bins,relative_l1,relative_mu3_error")
        println(io,join(("spatial",reference_bins,check_bins,space_error,abs(moment(truth.grid,final_numbers(truth),3)/last(moments)[4]-1)),','))
        println(io,join(("temporal",reference_bins,reference_bins,time_error,abs(moment(coarse_ref.grid,final_numbers(coarse_ref),3)/moment(time_check.grid,final_numbers(time_check),3)-1)),','))
    end
    open(joinpath(output,"reference_psd.csv"),"w") do io
        println(io,"length,density")
        for (L,f) in zip(truth.grid.centers,density(truth.grid,final_numbers(truth)));println(io,L,',',f);end
    end
    allruns=Tuple{String,Int,Any}[("reference",check_bins,truth)]
    for n in grids,(label,scheme) in (("upwind",Upwind()),("muscl",MUSCL()))
        result=sectional(n,scheme,initial_density,c0,tf,dt,times)
        push!(allruns,(label,n,result))
        println((method=label,bins=n,relative_l1=relative_difference(result,truth),seconds=result.seconds))
    end
    open(joinpath(output,"errors.csv"),"w") do io
        println(io,"method,bins,mean_absolute_error,relative_l1,rmse,max_error,relative_mu3_error,concentration_error,material_drift,seconds")
        for (label,n,run) in allruns
            N=final_numbers(run);target=rebin(truth.grid,final_numbers(truth),run.grid)
            err=(N.-target)./run.grid.widths
            total0=c0+params.rho*moment(run.grid,population(run.problem,run.problem.u0,:crystals),3)
            c=bulk(run.problem,last(run.solution.u),:c)
            drift=maximum(abs(bulk(run.problem,u,:c)+params.rho*moment(run.grid,population(run.problem,u,:crystals),3)-total0) for u in run.solution.u)
            println(io,join((label,n,sum(abs.(err).*run.grid.widths)/1600,sum(abs.(N.-target))/sum(final_numbers(truth)),sqrt(sum(err.^2 .* run.grid.widths)/1600),maximum(abs.(err)),abs(moment(run.grid,N,3)/last(moments)[4]-1),abs(c-bulk(truth.problem,last(truth.solution.u),:c)),drift,run.seconds),','))
            open(joinpath(output,"$(label)_$(n)_psd.csv"),"w") do psd
                println(psd,"length,density")
                for (L,f) in zip(run.grid.centers,density(run.grid,N));println(psd,L,',',f);end
            end
        end
    end
    open(joinpath(output,"moments.csv"),"w") do io
        println(io,"method,bins,time,mu0,mu1,mu2,mu3,concentration,material_drift")
        for (label,n,run) in allruns
            total0=c0+params.rho*moment(run.grid,population(run.problem,run.problem.u0,:crystals),3)
            for (t,u) in zip(run.solution.t,run.solution.u)
                N=population(run.problem,u,:crystals);mu=[moment(run.grid,N,k) for k in 0:3];c=bulk(run.problem,u,:c)
                println(io,join((label,n,t,mu...,c,c+params.rho*mu[4]-total0),','))
            end
        end
    end
    open(joinpath(output,"inputs.txt"),"w") do io
        println(io,"Kinetics and cooling: Gunawan et al. (2004), Example 2, Eqs.28–34. Explicit alternative initial condition below.")
        println(io,provenance);println(io,"c0=",c0,"; continuum initial moments=",initial_moments)
        println(io,"MUSCL is the package's limited FV method, not a claim of implementing the paper's HR1.")
        println(io,"Final PSD reference: ",check_bins," nominal bins, dt=",dt,"; second reference: ",reference_bins," bins; temporal check at ",reference_bins," bins and dt=",dt/2)
        println(io,"Relative L1 reference changes: spatial=",space_error,"; temporal=",time_error)
        println(io,"Independent moment RK4 dt=0.05 vs 0.025 final relative change: ",moment_step_difference)
        println(io,"Julia ",VERSION,"; tf=",tf,"; zero upper size flux, domain [0,1600] micrometers.")
    end
    (;spatial_reference_change=space_error,temporal_reference_change=time_error,moment_step_difference)
end
if abspath(PROGRAM_FILE)==@__FILE__
    config=isempty(ARGS) ? (;) : Base.include(@__MODULE__,abspath(first(ARGS)))
    println(run_benchmark(;config...))
end
end
