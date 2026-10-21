module ConstantGrowthMethods
using PopulationBalanceModels
using Statistics

const growth=0.1
initial_density(a)=a>=0 ? 100exp(-a/0.01) : 0.0
primitive(a)=a<=0 ? 0.0 : -expm1(-a/0.01)
# Difference of two nearly saturated CDF values loses the exponential tail.
function exact_counts(grid,t)
    [b<=growth*t ? 0.0 : exp(-max(a-growth*t,0)/0.01)*
        (-expm1(-(b-max(a,growth*t))/0.01))
        for (a,b) in zip(grid.edges[1:end-1],grid.edges[2:end])]
end
function smooth_counts(grid,t)
    a=grid.edges[1:end-1].-growth*t;b=grid.edges[2:end].-growth*t
    grid.widths.+0.2.*(cos.(2pi.*a).-cos.(2pi.*b))./(2pi).+
        0.1.*(sin.(4pi.*b).-sin.(4pi.*a))./(4pi)
end

function combinations()
    [("upwind","euler",Upwind(),ForwardEuler(),1,1),
     ("upwind","ssprk22",Upwind(),SSPRK22(),1,2),
     ("upwind","ssprk33",Upwind(),SSPRK33(),1,3),
     ("upwind","rk4",Upwind(),RK4(),1,4),
     ("muscl","ssprk33",MUSCL(),SSPRK33(),2,3),
     ("muscl","rk4",MUSCL(),RK4(),2,4),
     ("vanleer","rk4",VanLeer(),RK4(),2,4),
     ("weno3","rk4",WENO3(),RK4(),3,4),
     ("weno5","ssprk33",WENO5(),SSPRK33(),5,3),
     ("weno5","rk4",WENO5(),RK4(),5,4),
     ("weno5","ssprk54",WENO5(),SSPRK54(),5,4),
     ("centered","rk4",Centered(),RK4(),2,4),
     ("linear_upwind","rk4",LinearUpwind(),RK4(),2,4),
     ("lax_wendroff","coupled",nothing,LaxWendroff(),2,2),
     ("beam_warming","coupled",nothing,BeamWarming(),2,2)]
end

function measure(grid,initial,truth,scheme,time,tf,dt,bc;repeats=3)
    model=PBMModel(growth=Growth(AffineGrowth(growth,0.0)))
    operator=scheme===nothing ? compile(model,grid;evolution=time,bc) : compile(model,grid;growth_scheme=scheme,bc)
    problem=PBMProblem(operator,initial,(0.0,tf))
    simulate(problem;method=time,dt,check_nonnegative=false) # compilation/warmup
    elapsed=Float64[];solution=nothing
    for _ in 1:repeats
        timed=@timed simulate(problem;method=time,dt,check_nonnegative=false)
        push!(elapsed,timed.time);solution=timed.value
    end
    N=vec(last(solution.u));f=density(grid,N);error=density(grid,N.-truth)
    (;N,f,rmse=sqrt(sum(error.^2 .*grid.widths)/sum(grid.widths)),
        l1=sum(abs.(error).*grid.widths),maximum_error=maximum(abs.(error)),
        minimum_density=minimum(f),negative_number=-sum(min.(N,0)),
        number_drift=sum(N)-sum(initial),seconds=median(elapsed),
        timing_minimum=minimum(elapsed),timing_maximum=maximum(elapsed),steps=solution.steps)
end

function run(;grids=(50,100,200,400,800,1600),smooth_grids=(40,80,160,320,640),
        courants=(0.5,1.0),tf=4.0,repeats=3,
        output=joinpath(@__DIR__,"..","docs","src","assets","constant_growth"))
    mkpath(output);choices=combinations()
    open(joinpath(output,"errors.csv"),"w") do io
        println(io,"scheme,time_method,courant,bins,rmse,mean_absolute_error,max_error,minimum_density,negative_number,number_drift,seconds,steps")
        for n in grids,C in courants,(name,tname,scheme,time,_,_) in choices
            # LinearUpwind/RK4 at CFL=1 is outside its linear stability interval.
            name=="linear_upwind" && C>0.65 && continue
            grid=Grid1D(range(0,1;length=n+1);coordinate=:length)
            truth=exact_counts(grid,tf)
            result=measure(grid,exact_counts(grid,0),truth,scheme,time,tf,C/(n*growth),
                SizeBC(OpenBoundary(),NoFlux());repeats)
            println(io,join((name,tname,C,n,result.rmse,result.l1,result.maximum_error,
                result.minimum_density,result.negative_number,result.number_drift,result.seconds,result.steps),','))
            if n==200
                open(joinpath(output,"$(name)_$(tname)_$(C)_$(n).csv"),"w") do data
                    println(data,"size,density,exact")
                    for i in eachindex(result.f);println(data,join((grid.centers[i],result.f[i],truth[i]/grid.widths[i]),','));end
                end
            end
        end
    end
    open(joinpath(output,"smooth_errors.csv"),"w") do io
        println(io,"scheme,time_method,step_policy,bins,dt,courant,rmse,l1,observed_rmse_order,seconds,timing_minimum,timing_maximum,steps,minimum_density,number_drift")
        for (name,tname,scheme,time,space_order,time_order) in choices,policy in ("fixed_cfl","spatial")
            # Avoid presenting a prohibitively small RK3 step as a timing comparison.
            policy=="spatial" && !(tname in ("euler","rk4","ssprk54","coupled")) && continue
            previous=NaN;previous_n=0
            for n in smooth_grids
                grid=Grid1D(range(0,1;length=n+1);coordinate=:length);h=1/n
                exponent=policy=="fixed_cfl" ? 1.0 : max(1.0,space_order/time_order)
                dt=0.5*h^exponent/growth
                result=measure(grid,smooth_counts(grid,0),smooth_counts(grid,tf),scheme,time,tf,dt,
                    SizeBC(Periodic(),Periodic());repeats)
                order=isfinite(previous) ? log(previous/result.rmse)/log(n/previous_n) : NaN
                println(io,join((name,tname,policy,n,dt,growth*dt/h,result.rmse,result.l1,order,
                    result.seconds,result.timing_minimum,result.timing_maximum,result.steps,result.minimum_density,result.number_drift),','))
                previous=result.rmse;previous_n=n
            end
        end
    end
    open(joinpath(output,"inputs.txt"),"w") do io
        println(io,"Paper problem: Inguva et al. (2022), Case 1 Eq.56: g=0.1, f0=100exp(-a/0.01), domain [0,1], tf=",tf)
        println(io,"Zero inflow and closed upper boundary; front is discontinuous. Errors compare analytical bin integrals; no formal smooth order is claimed for this front.")
        println(io,"Smooth order check: f0=1+0.2sin(2πa)+0.1cos(4πa), periodic domain [0,1].")
        println(io,"Fixed-CFL runs use C=0.5. Spatial checks use dt=0.5*h^max(1,p/q)/g for spatial order p and time order q; actual dt/CFL are exported.")
        println(io,"Timings are medians of ",repeats," warmed integrations, excluding compilation/plotting; minima/maxima exported.")
        println(io,"No clipping; signed density minima and number drift are retained. Julia ",VERSION,"; CPU ",Sys.CPU_NAME)
    end
    println("Growth comparison written to ",output)
end
if abspath(PROGRAM_FILE)==@__FILE__;run();end
end
