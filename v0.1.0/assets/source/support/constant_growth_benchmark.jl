# Included inside ConstantGrowthMethods. Every recorded run is the same
# translating Gaussian with zero inflow/open outflow and exact bin integrals.
function run(;grids=refinement_bins,courants=profile_courants,tf=comparison_time,
        profile_resolution=profile_bins,
        output=joinpath(@__DIR__,"..","..",".cache","benchmarks","constant_growth"))
    mkpath(output)
    choices=combinations()
    boundary=SizeBC(OpenBoundary(),OpenBoundary())
    for (_,_,scheme,time) in choices
        grid=Grid1D(range(0.,1.;length=21))
        dt=(time isa Inguva2022 ? 1. : refinement_courant)/(20growth)
        # One tiny, separate compilation warmup with the actual boundary and
        # time method. The measured integration is not repeated for timing.
        simulate(make_problem(grid,gaussian_counts(grid,0.),scheme,time,dt,boundary);
            method=time,dt,check_nonnegative=false)
    end
    metrics=joinpath(output,"gaussian_metrics.csv")
    profiles=joinpath(output,"gaussian_profiles.csv")
    open(metrics,"w") do summary
        println(summary,"method,profile,bins,courant,dt,tf,rmse,l1,maximum_error,minimum_density,negative_number,number_drift,seconds,steps,bytes,gc_seconds")
        open(profiles,"w") do curves
            println(curves,"method,courant,bins,left,right,number,density,exact_number,exact_density")
            for (name,tname,scheme,time) in choices
                id=comparison_id(name,tname)
                c=name=="inguva2022" ? 1. : refinement_courant
                jobs=[(n,c) for n in grids]
                append!(jobs,[(profile_resolution,C) for C in courants if name!="inguva2022" || C==1.])
                for (n,C) in unique(jobs)
                    grid=Grid1D(range(0.,1.;length=n+1);coordinate=:length)
                    truth=gaussian_counts(grid,tf)
                    dt=C/(n*growth)
                    result=measure(grid,gaussian_counts(grid,0.),truth,scheme,time,tf,dt,boundary)
                    println(summary,join((id,"gaussian",n,C,dt,tf,result.rmse,result.l1,
                        result.maximum_error,result.minimum_density,result.negative_number,
                        result.number_drift,result.seconds,result.steps,result.bytes,result.gc_seconds),','))
                    if n==profile_resolution
                        for i in eachindex(result.f)
                            println(curves,join((id,C,n,grid.edges[i],grid.edges[i+1],
                                result.N[i],result.f[i],truth[i],truth[i]/grid.widths[i]),','))
                        end
                    end
                end
            end
        end
    end
    root=normpath(joinpath(@__DIR__,"..",".."))
    numerical_files=[joinpath(root,"Project.toml")]
    for folder in ("src","ext"), (directory,_,names) in walkdir(joinpath(root,folder)), name in names
        endswith(name,".jl") && push!(numerical_files,joinpath(directory,name))
    end
    digest(files)=bytes2hex(sha256(join((relpath(p,root)*"\0"*bytes2hex(sha256(read(p)))*"\n" for p in sort(files)))))
    source=digest(numerical_files)
    workflow=digest([joinpath(root,"examples","constant_growth_methods.jl"),@__FILE__])
    config=Dict("bins"=>collect(grids),"profile_bins"=>profile_resolution,
        "profile_courants"=>collect(courants),"final_time"=>tf,"courant"=>refinement_courant,
        "profile_family"=>"gaussian","roundoff_factor"=>roundoff_factor)
    benchmark=Dict("profile"=>"gaussian","height"=>gaussian_height,"center"=>gaussian_center,
        "standard_deviation"=>gaussian_width,"growth"=>growth,"final_time"=>tf,
        "domain"=>[0.,1.],"boundary"=>"zero inflow / open outflow")
    provenance=Dict("kind"=>"new Gaussian benchmark calculation","source_digest"=>source,
        "workflow_digest"=>workflow,"environment"=>Dict("julia"=>string(VERSION),
            "cpu"=>first(Sys.cpu_info()).model,"cpu_target"=>Sys.CPU_NAME,
            "julia_threads"=>Threads.nthreads(),"blas_threads"=>PopulationBalanceModels.LinearAlgebra.BLAS.get_num_threads()),
        "timing_protocol"=>"one measured solve after separate tiny compilation warmups; setup, explicit GC, export and plotting excluded")
    open(joinpath(output,"gaussian_source.toml"),"w") do io
        TOML.print(io,Dict("schema"=>1,"benchmark"=>benchmark,"configuration"=>config,
            "provenance"=>provenance,"files"=>Dict(basename(path)=>bytes2hex(sha256(read(path))) for path in (metrics,profiles)));sorted=true)
    end
    println("Gaussian comparison written to ",output)
end
