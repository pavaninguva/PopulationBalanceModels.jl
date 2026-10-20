module CoolingCrystallizationBenchmark
using PopulationBalanceModels
include("papers/gunawan2004_case2.jl")
using .Gunawan2004Case2

# Equation 34 is kept verbatim in reported_seed_density. Inputs are explicit;
# this driver does not replace a negative seed with a demonstration polynomial.
function seed_moments(initial_density)
    nodes=(-sqrt(3/5),0.0,sqrt(3/5)); weights=(5/9,8/9,5/9)
    [15sum(w*(195.5+15z)^k*initial_density(195.5+15z) for (z,w) in zip(nodes,weights)) for k in 0:3]
end
# Independent classical RK4 integration of the explicitly coded continuum ODEs.
function reference(initial,p,times;dt=0.01)
    u=copy(initial); t=0.0; states=Vector{Float64}[]
    k1=similar(u);k2=similar(u);k3=similar(u);k4=similar(u)
    for target in times
        while t<target
            h=min(dt,target-t)
            moment_rhs!(k1,u,p,t); moment_rhs!(k2,u.+h/2 .*k1,p,t+h/2)
            moment_rhs!(k3,u.+h/2 .*k2,p,t+h/2); moment_rhs!(k4,u.+h.*k3,p,t+h)
            u .+= h/6 .* (k1.+2k2.+2k3.+k4)
            t=h==target-t ? target : t+h
        end
        push!(states,copy(u))
    end
    states
end
function run_benchmark(;initial_density=reported_seed_density,c0,provenance,
        initial_moments=seed_moments(initial_density),grids=(3200,),dt=0.01,tf=1000.0,
        output=joinpath(@__DIR__,"..","docs","src","assets","gunawan_example2"))
    values=initial_density.(range(180.5,210.5;length=301))
    all(x->isfinite(x) && x>=0,values) || throw(ArgumentError(
        "Seed density is negative on its support. Verbatim Eq.34 has maximum -0.01264367816; no clipping or replacement is applied."))
    isfinite(c0) && c0>=0 || throw(ArgumentError("initial dissolved concentration must be finite and nonnegative"))
    isempty(strip(provenance)) && throw(ArgumentError("record the source of the initial conditions"))
    mkpath(output); times=collect(range(0,tf;length=101))
    initial=vcat(initial_moments,c0)
    params=parameters()
    truth=reference(initial,params,times)
    check=reference(initial,params,times;dt=0.005)
    reference_error=maximum(abs.(last(truth).-last(check))./max.(abs.(last(check)),eps()))
    open(joinpath(output,"reference.csv"),"w") do io
        println(io,"time,mu0,mu1,mu2,mu3,concentration")
        for (t,u) in zip(times,truth);println(io,join((t,u...),','));end
    end
    open(joinpath(output,"moments.csv"),"w") do io
        println(io,"method,bins,time,mu0,mu1,mu2,mu3,concentration,material_drift")
        for n in grids,(label,scheme) in (("upwind",Upwind()),("muscl",MUSCL()))
            # Split seed-support discontinuities exactly, even on coarse grids.
            edges=sort!(unique(vcat(collect(range(0,1600;length=n+1)),180.5,210.5)))
            grid=Grid1D(edges;coordinate=:length)
            problem=build(;grid,initial_density,c0,scheme,tspan=(0.0,tf))
            solution=simulate(problem;dt,saveat=times)
            total0=c0+params.rho*moment(grid,population(problem,problem.u0,:crystals),3)
            for (t,u) in zip(solution.t,solution.u)
                N=population(problem,u,:crystals); mu=[moment(grid,N,k) for k in 0:3]; c=bulk(problem,u,:c)
                println(io,join((label,length(grid),t,mu...,c,c+params.rho*mu[4]-total0),','))
            end
            open(joinpath(output,"$(label)_$(n)_psd.csv"),"w") do psd
                println(psd,"length,density")
                f=density(grid,population(problem,last(solution.u),:crystals))
                for (x,y) in zip(grid.centers,f);println(psd,x,',',y);end
            end
            error=abs(moment(grid,population(problem,last(solution.u),:crystals),3)-last(truth)[4])/last(truth)[4]
            println((method=label,bins=length(grid),relative_mu3_error=error))
        end
    end
    open(joinpath(output,"inputs.txt"),"w") do io
        println(io,"Gunawan et al. (2004), Example 2, equations 28–34; target Figures 2 and 3.")
        println(io,provenance)
        println(io,"c0=",c0,"; initial continuum moments=",collect(initial_moments))
        println(io,"HR1 and Lax–Wendroff are not implemented here. Comparing equation solutions does not reproduce those algorithms.")
        println(io,"Julia ",VERSION,"; dt=",dt,"; tf=",tf,"; reference RK4 dt=0.01 checked at 0.005.")
        println(io,"Maximum final relative reference step-halving difference: ",reference_error)
    end
    reference_error
end
if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) && error("Supply a .jl input file returning (initial_density,c0,provenance). Eq.34 is not silently repaired; see the case study.")
    config=Base.include(@__MODULE__,abspath(first(ARGS)))
    run_benchmark(;config...)
end
end
