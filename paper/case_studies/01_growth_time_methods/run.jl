module PaperGrowth
using PopulationBalanceModels, TOML
include("../../common.jl")
using .PaperWorkflow
include(joinpath(ROOT,"examples","constant_growth_methods.jl"))
const C = ConstantGrowthMethods
choices() = [("upwind",Upwind(),ForwardEuler()),("muscl",MUSCL(),SSPRK33()),
    ("weno3",WENO3(),SSPRK33()),("weno5",WENO5(),RK4()),
    ("inguva2022",nothing,Inguva2022()),("hoc",nothing,HOCFiniteVolume())]

function run(mode="production")
    ctx = context(@__DIR__,mode)
    runs = Dict{String,String}()
    for (id,scheme,time) in choices(), periodic in (false,true)
        grid = Grid1D(range(0,1;length=21))
        courant = id=="inguva2022" ? 1. : ctx.config["courant"]
        dt = courant/(20C.growth)
        bc = periodic ? SizeBC(Periodic(),Periodic()) : SizeBC(OpenBoundary(),OpenBoundary())
        simulate(C.make_problem(grid,C.exact_counts(grid,0.),scheme,time,dt,bc);
            method=time,dt,check_nonnegative=false)
    end
    for profile in ("front","smooth"), (id,scheme,time) in choices()
        bins = profile=="front" ? [ctx.config["profile_bins"]] : ctx.config["smooth_bins"]
        for n in bins
            courant = id=="inguva2022" ? 1. : ctx.config["courant"]
            settings = Dict("profile"=>profile,"bins"=>n,"method"=>id,
                "api_time_method"=>string(typeof(time)),"courant"=>courant,
                "tf"=>ctx.config["final_time"],"growth"=>C.growth,
                "initialization"=>"exact bin integrals","state_checks"=>"signed diagnostic; no clipping")
            name = "$(profile)_$(id)_N$(n)"
            folder = cached(ctx,name,settings) do output
                grid = Grid1D(range(0,1;length=n+1);coordinate=:length)
                exact = profile=="front" ? C.exact_counts : C.smooth_counts
                initial,truth = exact(grid,0.),exact(grid,settings["tf"])
                bc = profile=="front" ? SizeBC(OpenBoundary(),OpenBoundary()) : SizeBC(Periodic(),Periodic())
                result = C.measure(grid,initial,truth,scheme,time,settings["tf"],courant/(n*C.growth),bc)
                csv(joinpath(output,"profile.csv"),[(;left=grid.edges[i],right=grid.edges[i+1],
                    number=result.N[i],density=result.f[i],exact_number=truth[i],exact_density=truth[i]/grid.widths[i]) for i in 1:n])
                row = (;method=id,profile,bins=n,courant,dt=courant/(n*C.growth),tf=settings["tf"],
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
    rows = NamedTuple[]
    for (id,_,_) in choices()
        old = nothing
        for n in ctx.config["smooth_bins"]
            path = runs["smooth_$(id)_N$(n)"]
            cells = split(readlines(joinpath(path,"metrics.csv"))[2],',')
            rms,l1 = parse(Float64,cells[7]),parse(Float64,cells[8])
            floor = ctx.config["roundoff_factor"]*eps(Float64)
            pr = old===nothing || min(old.rms,rms)<=floor ? NaN : log(old.rms/rms)/log(n/old.n)
            pl = old===nothing || min(old.l1,l1)<=floor ? NaN : log(old.l1/l1)/log(n/old.n)
            push!(rows,(;method=id,bins=n,rmse=rms,l1,observed_rmse_order=pr,observed_l1_order=pl,roundoff_limit=floor))
            old = (;n,rms,l1)
        end
    end
    csv(joinpath(ctx.output,"observed_orders.csv"),rows)
    index(ctx,runs)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    PaperGrowth.run("--smoke" in ARGS ? "smoke" : "production")
end
