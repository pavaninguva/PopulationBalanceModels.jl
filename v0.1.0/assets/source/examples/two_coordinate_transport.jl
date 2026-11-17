module PBM2DBenchmark
using PopulationBalanceModels

export problem,reference,run_benchmark
const VELOCITY=(0.5,0.25)
const FINAL_TIME=0.25
const PERIODIC=SizeBC(Periodic(),Periodic())

function _smooth_average(axis,t,velocity)
    L=last(axis.edges)-first(axis.edges)
    x=(axis.centers.-first(axis.edges).-velocity*t)./L
    n=length(axis)
    1 .+ 0.7*sinc(1/n).*cos.(2π.*(x.-0.2)) .+ 0.1*sinc(2/n).*cos.(4π.*(x.-0.2))
end
function _rectangle_average(axis,t,velocity)
    L=last(axis.edges)-first(axis.edges); a=first(axis.edges)+L/8+velocity*t; b=a+L/4
    [sum(max(0,min(axis.edges[i+1],b+k*L)-max(axis.edges[i],a+k*L)) for k in -2:2)/axis.widths[i] for i in 1:length(axis)]
end
function reference(g::Grid2D,t;profile=:smooth)
    average=profile===:smooth ? _smooth_average : profile===:rectangle ? _rectangle_average : throw(ArgumentError("unknown profile"))
    x=average(g.x,t,VELOCITY[1]); y=average(g.y,t,VELOCITY[2])
    (x*y').*(g.x.widths*g.y.widths')
end
function problem(n;profile=:smooth)
    # Unequal physical speeds and axis lengths prevent accidental cancellation
    # from assigning one speed (or their product) to both coordinate directions.
    g=Grid2D(range(0,1;length=n+1),range(0,0.5;length=n+1))
    op=compile(PBMModel(Growth2D(VELOCITY)),g;
        growth_scheme=WENO5(epsilon=1e-12),bc=SizeBC2D(x=PERIODIC,y=PERIODIC))
    PBMProblem(op,reference(g,0.0;profile),(0.0,FINAL_TIME))
end
function _metrics(prob,sol,truth)
    g=prob.operator.grid; area=g.x.widths*g.y.widths'; e=(sol.u[end].-truth)./area
    f=density(g,sol.u[end]); initial_number=sum(prob.u0)
    (;L1=sum(abs.(e).*area),L2=sqrt(sum(e.^2 .*area)),Linf=maximum(abs,e),
        relative_number_error=abs(sum(sol.u[end])-initial_number)/initial_number,
        minimum_density=minimum(f),negative_number=-sum(min.(sol.u[end],0.0)))
end
function run_benchmark(;grids=(32,64,128),profiles=(:smooth,:rectangle),repetitions=3,
        output=joinpath(@__DIR__,"..","docs","src","assets","results_2d.csv"))
    repetitions>=1 || throw(ArgumentError("repetitions must be positive"))
    rows=NamedTuple[]
    for profile in profiles,n in grids
        n%8==0 || throw(ArgumentError("use multiples of 8 for aligned exact output times"))
        prob=problem(n;profile); truth=reference(prob.operator.grid,FINAL_TIME;profile)
        profiles_at_final=Dict{String,Any}()
        # WENO uses an unsplit explicit CFL sum of 0.4. ADI/ExactShift use
        # directional CFL=1, which is not stable for the unsplit explicit method.
        for (label,method,dt) in (("WENO5_SSPRK33",SSPRK33(),0.4/n),
                ("HOCADI4_variant",HOCADI(time_order=4),2.0/n),
                ("InguvaBraatz_constant_shift",ExactShift(),2.0/n))
            solveit()=simulate(prob;dt,method,saveat=[FINAL_TIME],check_nonnegative=false)
            solveit() # JIT warmup, including the actual forward path
            times=Float64[]; bytes=Int[]; solution=nothing
            for _ in 1:repetitions
                sample=@timed solveit()
                push!(times,sample.time); push!(bytes,sample.bytes); solution=sample.value
            end
            metrics=_metrics(prob,solution,truth)
            profiles_at_final[label]=solution.u[end]
            row=merge((;profile=String(profile),n,method=label,dt,steps=solution.steps,
                seconds=minimum(times),allocated_bytes=minimum(bytes)),metrics)
            push!(rows,row)
            println(row)
        end
        if n==maximum(grids)
            g=prob.operator.grid
            # Cut through the pulse's interior at the final time.
            j=argmin(abs.(g.y.centers.-0.1875))
            mkpath(dirname(output))
            labels=("WENO5_SSPRK33","HOCADI4_variant","InguvaBraatz_constant_shift")
            open(joinpath(dirname(output),String(profile)*"_cut.csv"),"w") do io
                println(io,"x,exact,",join(labels,","))
                for i in 1:length(g.x)
                    area=g.x.widths[i]*g.y.widths[j]
                    println(io,join((g.x.centers[i],truth[i,j]/area,(profiles_at_final[label][i,j]/area for label in labels)...),","))
                end
            end
        end
    end
    mkpath(dirname(output))
    open(output,"w") do io
        println(io,join(string.(keys(first(rows))),","))
        for row in rows
            println(io,join(string.(values(row)),","))
        end
    end
    open(joinpath(dirname(output),"environment.txt"),"w") do io
        println(io,"PopulationBalanceModels ",Base.pkgversion(PopulationBalanceModels))
        println(io,"Julia ",VERSION,"; ",Sys.KERNEL," ",Sys.ARCH,"; threads=",Threads.nthreads())
        println(io,"CPU: ",Sys.cpu_info()[1].model)
        println(io,"Minimum of ",repetitions," warmed solves; numerical setup/factorization and output included; JIT/model construction excluded.")
    end
    rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_benchmark()
end
end
