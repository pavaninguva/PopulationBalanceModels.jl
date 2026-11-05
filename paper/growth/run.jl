module PaperGrowth
using PopulationBalanceModels, TOML
include("../_support/common.jl")
using .PaperWorkflow
include(joinpath(ROOT,"examples","constant_growth_methods.jl"))
const C = ConstantGrowthMethods

choices() = [(C.comparison_id(name,time),scheme,method)
    for (name,time,scheme,method) in C.combinations()]
courant_tag(c) = replace(string(Float64(c)),"."=>"p")
gaussian_key(id,n,c) = "gaussian_$(id)_C$(courant_tag(c))_N$(n)"

function run(mode="production";output=nothing)
    ctx = context(@__DIR__,mode)
    if output===nothing && mode=="production"
        output=joinpath(ROOT,".cache","paper","calculations","growth")
    end
    if output!==nothing
        ctx=merge(ctx,(output=abspath(output),))
        mkpath(ctx.output)
    end
    cfg = ctx.config
    cfg["profile_family"]=="gaussian" || error("This driver requires the Gaussian benchmark")
    runs = Dict{String,String}()
    boundary = SizeBC(OpenBoundary(),OpenBoundary())
    for (id,scheme,time) in choices()
        grid = Grid1D(range(0,1;length=21))
        courant = id=="inguva2022" ? 1. : cfg["courant"]
        dt=courant/(20C.growth)
        C.measure(grid,C.gaussian_counts(grid,0.),C.gaussian_counts(grid,dt),
            scheme,time,dt,dt,boundary)
    end
    for (id,scheme,time) in choices()
        courant = id=="inguva2022" ? 1. : cfg["courant"]
        jobs = [(n,courant) for n in cfg["bins"]]
        append!(jobs,[(cfg["profile_bins"],c) for c in cfg["profile_courants"]
            if id!="inguva2022" || c==1.])
        for (n,c) in unique(jobs)
            settings = Dict("profile"=>"gaussian","bins"=>n,"method"=>id,
                "api_time_method"=>string(typeof(time)),"courant"=>c,
                "tf"=>cfg["final_time"],"growth"=>C.growth,
                "initialization"=>"exact bin integrals","state_checks"=>"signed diagnostic; no clipping",
                "boundary"=>"zero inflow / open outflow","height"=>C.gaussian_height,
                "center"=>C.gaussian_center,"standard_deviation"=>C.gaussian_width)
            name = gaussian_key(id,n,c)
            folder = cached(ctx,name,settings) do destination
                grid = Grid1D(range(0,1;length=n+1);coordinate=:length)
                initial,truth = C.gaussian_counts(grid,0.),C.gaussian_counts(grid,settings["tf"])
                result = C.measure(grid,initial,truth,scheme,time,settings["tf"],c/(n*C.growth),boundary)
                csv(joinpath(destination,"profile.csv"),[(;left=grid.edges[i],right=grid.edges[i+1],
                    number=result.N[i],density=result.f[i],exact_number=truth[i],exact_density=truth[i]/grid.widths[i]) for i in 1:n])
                row = (;method=id,profile="gaussian",bins=n,courant=c,dt=c/(n*C.growth),tf=settings["tf"],
                    rmse=result.rmse,l1=result.l1,maximum_error=result.maximum_error,
                    minimum_density=result.minimum_density,negative_number=result.negative_number,
                    number_drift=result.number_drift,seconds=result.seconds,steps=result.steps,
                    bytes=result.bytes,gc_seconds=result.gc_seconds)
                csv(joinpath(destination,"metrics.csv"),[row])
                (["profile.csv","metrics.csv"],Dict("status"=>"completed","seconds"=>result.seconds,"steps"=>result.steps))
            end
            runs[name] = folder
        end
    end
    orders = NamedTuple[]
    for (id,_,_) in choices()
        old = nothing
        courant = id=="inguva2022" ? 1. : cfg["courant"]
        for n in cfg["bins"]
            path = runs[gaussian_key(id,n,courant)]
            header,values = split.(readlines(joinpath(path,"metrics.csv")),',')
            row = Dict(zip(header,values))
            rms,l1 = parse(Float64,row["rmse"]),parse(Float64,row["l1"])
            # Analysis screen in density/number units; not a solver tolerance.
            rms_floor = cfg["roundoff_factor"]*eps(Float64)*C.gaussian_height
            l1_floor = rms_floor # represented coordinate interval has length one
            pr = old===nothing || min(old.rms,rms)<=rms_floor ? NaN : log(old.rms/rms)/log(n/old.n)
            pl = old===nothing || min(old.l1,l1)<=l1_floor ? NaN : log(old.l1/l1)/log(n/old.n)
            push!(orders,(;method=id,bins=n,rmse=rms,l1,observed_rmse_order=pr,observed_l1_order=pl,
                roundoff_limit_rmse=rms_floor,roundoff_limit_l1=l1_floor))
            old = (;n,rms,l1)
        end
    end
    csv(joinpath(ctx.output,"gaussian_orders.csv"),orders)
    index(ctx,runs)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    output_arg=findfirst(arg->startswith(arg,"--output="),ARGS)
    output=output_arg===nothing ? nothing : split(ARGS[output_arg],'=';limit=2)[2]
    PaperGrowth.run("--smoke" in ARGS ? "smoke" : "production";output)
end
