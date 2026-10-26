# Reporting only. The model and all solves use the public package interfaces.
using Printf, TOML, SHA, Dates, Pkg

function write_csv(path, rows)
    isempty(rows) && throw(ArgumentError("cannot export an empty table"))
    open(path, "w") do io
        println(io, join(keys(first(rows)), ','))
        for row in rows
            println(io, join(values(row), ','))
        end
    end
end

function source_hashes()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = ["Project.toml", "examples/stem_cell_population.jl", "examples/support/stem_cell_study.jl"]
    for directory in ("src", "ext")
        for (folder, _, files) in walkdir(joinpath(root, directory)), file in files
            endswith(file, ".jl") && push!(paths, relpath(joinpath(folder, file), root))
        end
    end
    Dict(path => bytes2hex(sha256(read(joinpath(root, path)))) for path in paths)
end

function compare_solutions(a, b)
    a.t == b.t || error("comparison needs identical output times")
    ga = a.problem.operator.system.populations.stem.grid
    gb = b.problem.operator.system.populations.stem.grid
    edges = sort(unique(vcat(ga.edges, gb.edges)))
    largest_cdf = 0.0
    largest_number = 0.0
    largest_mean = 0.0
    for (ua, ub) in zip(a.u, b.u)
        na, nb = population(a.problem, ua, :stem), population(b.problem, ub, :stem)
        # Outside each finite represented domain its density is zero and CDF one.
        fa = conditional_cdf(ga, na, min.(edges, last(ga.edges)); window=(0.0, last(ga.edges)))
        fb = conditional_cdf(gb, nb, min.(edges, last(gb.edges)); window=(0.0, last(gb.edges)))
        largest_cdf = max(largest_cdf, maximum(abs, fa-fb))
        largest_number = max(largest_number, abs(sum(na)-sum(nb))/sum(nb))
        largest_mean = max(largest_mean, abs(moment(ga,na,1)/sum(na)-moment(gb,nb,1)/sum(nb)))
    end
    (maximum_CDF_difference=largest_cdf, maximum_relative_number_difference=largest_number,
     maximum_mean_cargo_difference=largest_mean)
end

function export_case(folder, name, solution)
    problem = solution.problem
    grid = problem.operator.system.populations.stem.grid
    summary = [merge((scenario=name,), state_summary(problem,u,t)) for (t,u) in zip(solution.t,solution.u)]
    write_csv(joinpath(folder, "trajectory_$name.csv"), summary)
    counts = NamedTuple[]
    for (t,u) in zip(solution.t, solution.u)
        cells = population(problem,u,:stem)
        for i in eachindex(cells)
            push!(counts, (time=t, left=grid.edges[i], right=grid.edges[i+1],
                count=cells[i], normalized_density=cells[i]/sum(cells)/grid.widths[i]))
        end
    end
    write_csv(joinpath(folder, "distribution_$name.csv"), counts)
    state, time = last(solution.u), last(solution.t)
    event = event_distributions(problem,state,time)
    recovered = rate_recovery(problem,state,time)
    write_csv(joinpath(folder,"events_$name.csv"), [
        (left=grid.edges[i],right=grid.edges[i+1],x=x,
         total=event.total[i]/grid.widths[i],dividing=event.dividing[i]/grid.widths[i],
         newborn=event.newborn[i]/grid.widths[i],
         differentiating=isnothing(event.differentiating) ? NaN : event.differentiating[i]/grid.widths[i],
         R=cargo_rate(x,problem.p,time,nothing),gamma=division_rate(x,problem.p,time,nothing),
         kappa=differentiation_rate(x,problem.p,time,nothing),
         R_recovered=recovered.R[i],gamma_recovered=recovered.gamma[i],
         kappa_recovered=recovered.kappa[i],central=recovered.active[i])
        for (i,x) in enumerate(grid.centers)])
    active = recovered.active
    expected_R = cargo_rate.(grid.centers, Ref(problem.p), time, nothing)
    expected_gamma = division_rate.(grid.centers, Ref(problem.p), time, nothing)
    expected_kappa = differentiation_rate.(grid.centers, Ref(problem.p), time, nothing)
    initial_cargo = moment(grid, population(problem,problem.u0,:stem), 1)
    checks = (
        scenario=name, steps=solution.steps,
        minimum_count=minimum(r.minimum_count for r in summary),
        maximum_number_balance=maximum(abs(r.number_balance) for r in summary),
        maximum_relative_cargo_balance=maximum(abs(r.cargo_balance)/initial_cargo for r in summary),
        maximum_lost_number=maximum(r.lost_number for r in summary),
        maximum_lost_cargo=maximum(r.lost_cargo for r in summary),
        maximum_upper_bin_fraction=maximum(r.upper_bin_fraction for r in summary),
        final_stem=last(summary).stem, final_differentiated=last(summary).differentiated,
        final_mean_cargo=last(summary).mean_cargo,
        final_division_average=event.division_average,
        final_differentiation_average=event.differentiation_average,
        final_net_rate=event.division_average-event.differentiation_average,
        normalized_shape_L1_rate=recovered.normalized_shape_L1_rate,
        maximum_CDF_rate=recovered.maximum_CDF_rate,
        R_recovery_RMSE=sqrt(sum(abs2,recovered.R[active]-expected_R[active])/sum(active)),
        gamma_recovery_RMSE=sqrt(sum(abs2,recovered.gamma[active]-expected_gamma[active])/sum(active)),
        kappa_recovery_RMSE=sqrt(sum(abs2,recovered.kappa[active]-expected_kappa[active])/sum(active)),
        unresolved_newborn_fraction=event.unresolved_newborn_fraction,
    )
    checks.minimum_count >= 0 || error("negative population")
    checks.maximum_number_balance < 1e-9 * max(1.0,checks.final_stem+checks.final_differentiated) || error("number/event balance failed")
    checks.maximum_relative_cargo_balance < 1e-9 || error("cargo balance failed")
    checks.maximum_upper_bin_fraction < 1e-8 || error("increase xmax: upper boundary is occupied")
    checks
end

"""Generate documentation data and independent refinement comparisons.
Production and smoke folders are distinct; plotters reject smoke data.
This is a forward PBM study, without fitted parameters or experimental data.
"""
function run_study(; output=joinpath(@__DIR__,"..","..","docs","src","assets","stem_cells"),
                   smoke=false, bins=smoke ? 100 : 600, tf=smoke ? 10.0 : 200.0)
    root = normpath(joinpath(@__DIR__,"..",".."))
    mkpath(output)
    settings = (bins=bins,xmax=120.0,tf=tf,daughter_order=12,maximum_dt=0.05,cfl=0.35)
    times = collect(range(0.0,tf;length=smoke ? 11 : 101))
    sources = source_hashes()
    started = string(now(UTC))
    cpu = Sys.cpu_info()[1].model
    summaries = NamedTuple[]
    refinements = NamedTuple[]
    for (name, different) in (("self_renewal",false),("differentiation",true))
        println("Stem cells: ",name,"; ",bins," bins; horizon ",tf); flush(stdout)
        # Separate compilation warmup; no plotting/export inside measured solve.
        solve_case(;bins=40,xmax=120.0,tf=0.1,differentiation=different,
                   daughter_order=settings.daughter_order,saveat=[0.0,0.1])
        GC.gc()
        timing = @timed solve_case(;settings...,differentiation=different,saveat=times)
        solution = timing.value
        checks = export_case(output,name,solution)
        push!(summaries,merge(checks,(seconds=timing.time,allocated_bytes=timing.bytes,)))
        @printf("  %d steps; %.3f s; final stem %.6g; exits %.6g; cargo defect %.3g\n",
                solution.steps,timing.time,checks.final_stem,checks.final_differentiated,checks.maximum_relative_cargo_balance)
        flush(stdout)
        smoke && continue
        configs = (
            ("coarse_grid",bins÷2,120.0,12,0.05,0.35),
            ("fine_grid",2bins,120.0,12,0.05,0.35),
            ("time",bins,120.0,12,0.025,0.175),
            ("daughter_quadrature",bins,120.0,24,0.05,0.35),
            ("domain",round(Int,bins*160/120),160.0,12,0.05,0.35),
        )
        for (check,n,xmax,q,dt,cfl) in configs
            println("  refinement: ",check,"; bins=",n,"; quadrature=",q); flush(stdout)
            refined = @timed solve_case(;bins=n,xmax,tf,daughter_order=q,
                maximum_dt=dt,cfl,differentiation=different,saveat=times)
            comparison = compare_solutions(solution,refined.value)
            push!(refinements,merge((scenario=name,check=check,bins=n,xmax=xmax,
                daughter_order=q,maximum_dt=dt,cfl=cfl,steps=refined.value.steps,
                seconds=refined.time),comparison))
        end
    end
    write_csv(joinpath(output,"summary.csv"),summaries)
    isempty(refinements) || write_csv(joinpath(output,"refinement.csv"),refinements)
    dependencies = Dict(info.name => string(info.version) for info in values(Pkg.dependencies()) if info.version !== nothing)
    manifest = Dict(
        "mode" => smoke ? "smoke" : "production", "recorded_at_UTC" => started,
        "git_revision" => readchomp(`git -C $root rev-parse HEAD`),
        "working_tree" => readchomp(`git -C $root status --short`),
        "julia" => string(VERSION), "cpu" => cpu, "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(), "dependencies" => dependencies,
        "reference_DOI" => REFERENCE_DOI, "authors_code_revision" => AUTHOR_REVISION,
        "settings" => Dict(string(k)=>v for (k,v) in pairs(settings)),
        "initial_condition" => "Normal(50,0.5), truncated to [0,100]; exact bin integrals; M0(0)=1",
        "units" => "synthetic cargo and model time; concentrations relative to initial culture",
        "methods" => "MUSCL; fixed-pivot binary division; SSPRK33; native stage checks",
        "timing_scope" => "one solve after separate warmup, including setup/checks/saves; excludes export/plotting",
        "rate_recovery" => "balanced-growth consistency; Gamma/kappa identities are not independent inference",
        "source_hashes" => sources,
        "files" => Dict(file=>bytes2hex(sha256(read(joinpath(output,file)))) for file in readdir(output) if endswith(file,".csv")),
    )
    open(joinpath(output,"run.toml"),"w") do io
        TOML.print(io,manifest;sorted=true)
    end
    println("Documentation data: ",normpath(output))
    (summary=summaries,refinement=refinements)
end

function main(args=ARGS)
    all(a->a=="--smoke" || startswith(a,"--output=") || startswith(a,"--bins="),args) ||
        throw(ArgumentError("use --smoke, --output=PATH or --bins=N"))
    smoke = "--smoke" in args
    directory = smoke ? joinpath(tempdir(),"pbm-stem-cell-smoke") : joinpath(@__DIR__,"..","..","docs","src","assets","stem_cells")
    bins = smoke ? 100 : 600
    for arg in args
        startswith(arg,"--output=") && (directory=split(arg,'=';limit=2)[2])
        startswith(arg,"--bins=") && (bins=parse(Int,split(arg,'=';limit=2)[2]))
    end
    run_study(;output=directory,smoke,bins)
end
