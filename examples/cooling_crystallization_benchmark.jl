module CoolingCrystallizationBenchmark
using PopulationBalanceModels
include("papers/gunawan2004_case2.jl")
using .Gunawan2004Case2

# Explicit substitute for the negative rounded polynomial printed in Eq. 34.
const DEMONSTRATION_SEED=L->180.5<=L<=210.5 ? 0.08*(1-((L-195.5)/15)^2) : 0.0
function seed_moments()
    nodes=(-sqrt(3/5),0.0,sqrt(3/5)); weights=(5/9,8/9,5/9)
    [15sum(w*(195.5+15z)^k*DEMONSTRATION_SEED(195.5+15z) for (z,w) in zip(nodes,weights)) for k in 0:3]
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
function run_benchmark(;grids=(800,1600),dt=0.05,tf=1000.0,
        output=joinpath(@__DIR__,"..","docs","src","assets","cooling_benchmark"))
    mkpath(output); times=collect(range(0,tf;length=101)); c0=solubility(temperature(0.0))
    initial=vcat(seed_moments(),c0)
    params=(kb=4.64e-7,b=1.78,kg=116.0,g=1.32,rho=2.11e-12)
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
            problem=build(;grid,initial_density=DEMONSTRATION_SEED,c0,scheme,tspan=(0.0,tf))
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
        println(io,"DEMONSTRATION INPUTS: seed=0.08*(1-((L-195.5)/15)^2) on [180.5,210.5]; c0=csat(32).")
        println(io,"These are NOT author-confirmed paper inputs. HR1 and Lax–Wendroff are not implemented here.")
        println(io,"Julia ",VERSION,"; dt=",dt,"; tf=",tf,"; reference RK4 dt=0.01 checked at 0.005.")
        println(io,"Maximum final relative reference step-halving difference: ",reference_error)
    end
    reference_error
end
if abspath(PROGRAM_FILE)==@__FILE__;run_benchmark();end
end
