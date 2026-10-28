module PaperGrowth
using PopulationBalanceModels, TOML
include("../_support/common.jl")
using .PaperWorkflow
include(joinpath(ROOT,"examples","constant_growth_methods.jl"))
const C = ConstantGrowthMethods

# Use the same seven pairings as the manual, including both WENO5 time methods.
choices() = [(name=="weno5" && time=="ssprk33" ? "weno5_ssprk33" : name,scheme,method)
    for (name,time,scheme,method) in C.combinations()]
courant_tag(c) = replace(string(Float64(c)),"."=>"p")
gaussian_key(id,n,c) = "gaussian_$(id)_C$(courant_tag(c))_N$(n)"

function run(mode="production")
    ctx = context(@__DIR__,mode)
    cfg = ctx.config
    cfg["profile_family"]=="gaussian" || error("This driver requires the Gaussian benchmark")
    runs = Dict{String,String}()
    for (id,scheme,time) in choices(), periodic in (false,true)
        grid = Grid1D(range(0,1;length=21))
        courant = id=="inguva2022" ? 1. : cfg["courant"]
        exact = periodic ? C.smooth_counts : C.gaussian_counts
        bc = periodic ? SizeBC(Periodic(),Periodic()) : SizeBC(OpenBoundary(),OpenBoundary())
        C.measure(grid,exact(grid,0.),exact(grid,cfg["final_time"]),scheme,time,
            cfg["final_time"],courant/(20C.growth),bc)
    end
    for (id,scheme,time) in choices(), profile in ("gaussian","smooth")
        courant = id=="inguva2022" ? 1. : cfg["courant"]
        jobs = [(n,courant) for n in cfg["bins"]]
        if profile=="gaussian"
            append!(jobs,[(cfg["profile_bins"],c) for c in cfg["profile_courants"]
                if id!="inguva2022" || c==1.])
        end
        for (n,c) in unique(jobs)
            settings = Dict("profile"=>profile,"bins"=>n,"method"=>id,
                "api_time_method"=>string(typeof(time)),"courant"=>c,
                "tf"=>cfg["final_time"],"growth"=>C.growth,
                "initialization"=>"exact bin integrals","state_checks"=>"signed diagnostic; no clipping",
                "boundary"=>profile=="gaussian" ? "zero inflow / open outflow" : "periodic")
            if profile=="gaussian"
                merge!(settings,Dict("height"=>C.gaussian_height,
                    "center"=>C.gaussian_center,"standard_deviation"=>C.gaussian_width))
            end
            name = profile=="gaussian" ? gaussian_key(id,n,c) : "smooth_$(id)_N$(n)"
            folder = cached(ctx,name,settings) do output
                grid = Grid1D(range(0,1;length=n+1);coordinate=:length)
                exact = profile=="gaussian" ? C.gaussian_counts : C.smooth_counts
                initial,truth = exact(grid,0.),exact(grid,settings["tf"])
                bc = profile=="gaussian" ? SizeBC(OpenBoundary(),OpenBoundary()) : SizeBC(Periodic(),Periodic())
                result = C.measure(grid,initial,truth,scheme,time,settings["tf"],c/(n*C.growth),bc)
                csv(joinpath(output,"profile.csv"),[(;left=grid.edges[i],right=grid.edges[i+1],
                    number=result.N[i],density=result.f[i],exact_number=truth[i],exact_density=truth[i]/grid.widths[i]) for i in 1:n])
                row = (;method=id,profile,bins=n,courant=c,dt=c/(n*C.growth),tf=settings["tf"],
                    rmse=result.rmse,l1=result.l1,maximum_error=result.maximum_error,
                    minimum_density=result.minimum_density,negative_number=result.negative_number,
                    number_drift=result.number_drift,seconds=result.seconds,steps=result.steps,
                    bytes=result.bytes,gc_seconds=result.gc_seconds)
                csv(joinpath(output,"metrics.csv"),[row])
                (["profile.csv","metrics.csv"],Dict("status"=>"completed","seconds"=>result.seconds,"steps"=>result.steps))
            end
            runs[name] = folder
        end
    end
    orders = NamedTuple[]
    for (id,_,_) in choices()
        old = nothing
        for n in cfg["bins"]
            path = runs["smooth_$(id)_N$(n)"]
            header,values = split.(readlines(joinpath(path,"metrics.csv")),',')
            row = Dict(zip(header,values))
            rms,l1 = parse(Float64,row["rmse"]),parse(Float64,row["l1"])
            floor = cfg["roundoff_factor"]*eps(Float64)
            pr = old===nothing || min(old.rms,rms)<=floor ? NaN : log(old.rms/rms)/log(n/old.n)
            pl = old===nothing || min(old.l1,l1)<=floor ? NaN : log(old.l1/l1)/log(n/old.n)
            push!(orders,(;method=id,bins=n,rmse=rms,l1,observed_rmse_order=pr,observed_l1_order=pl,roundoff_limit=floor))
            old = (;n,rms,l1)
        end
    end
    csv(joinpath(ctx.output,"observed_orders.csv"),orders)
    index(ctx,runs)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    PaperGrowth.run("--smoke" in ARGS ? "smoke" : "production")
end
