# Included inside ContinuousPrecipitation. Timings exclude assembly and export.
# The full study enumerates the paper's Cartesian product, not diagonal pairs.
function benchmark_configurations(;full=false)
    configurations = NamedTuple[]
    if full
        for mesh in (:log,:uniform), name in ("upwind","muscl","weno5")
            mesh === :log && name == "weno5" && continue
            for nx in (12,25,50,100,200), nz in (37,75,150,300,600)
                push!(configurations,(;mesh,size_method=name,axial_method="muscl",nx,nz,
                    refinement="cartesian"))
            end
        end
    else
        # One coordinate at a time: four points on each curve. The shared
        # (72,96) endpoint is measured once and used in both refinement panels.
        for name in ("upwind","muscl","weno5")
            for nx in (12,24,48,72)
                push!(configurations,(mesh=:uniform,size_method=name,axial_method="muscl",
                    nx,nz=96,refinement="size"))
            end
            for nz in (16,32,64)
                push!(configurations,(mesh=:uniform,size_method=name,axial_method="muscl",
                    nx=72,nz,refinement="axial"))
            end
        end
        # Retain a logarithmic-size sensitivity check, against the SAME reference.
        for nx in (12,24,36,48)
            push!(configurations,(mesh=:log,size_method="muscl",axial_method="muscl",
                nx,nz=64,refinement="log_size"))
        end
    end
    configurations
end
reconstruction(label) = label == "upwind" ? Upwind() : label == "muscl" ? MUSCL() :
    label == "weno5" ? WENO5() : throw(ArgumentError("unknown reconstruction: $label"))

"""Export cell averages, optionally cropping the display without changing the solve."""
function export_field(path,problem,solution;max_size=Inf)
    grid = first(problem.operator.local_operators).system.populations.crystals.grid
    axial = problem.operator.domain.grid
    open(path,"w") do io
        println(io,"time,z_m,size_m,number_density,concentration_kg_m3,z_left,z_right,x_left,x_right")
        for (t,u) in zip(solution.t,solution.u), j in axes(u,2), i in eachindex(grid.centers)
            grid.edges[i] < max_size || continue
            row = (t,axial.centers[j],grid.centers[i],1e12*u[i,j]/grid.widths[i],
                bulk(problem,u,:c)[j],axial.edges[j],axial.edges[j+1],grid.edges[i],grid.edges[i+1])
            println(io,join(row,','))
        end
    end
end

function export_reference_outlets(path,problem,solution)
    grid = first(problem.operator.local_operators).system.populations.crystals.grid
    N = population(problem,last(solution.u),:crystals)[:,end]
    open(path,"w") do io
        println(io,"mesh,x_left,x_right,number_density")
        for i in eachindex(N)
            println(io,join((:uniform,grid.edges[i],grid.edges[i+1],1e12*N[i]/grid.widths[i]),','))
        end
    end
end

"""Compare time steps on the same mesh, before and after reactor startup.
Bin numbers already include size widths; axial widths complete the inventory norm.
"""
function export_temporal_checks(path,problem,base,fine)
    base.t == fine.t || throw(ArgumentError("temporal checks require matching observation times"))
    widths = transpose(problem.operator.domain.grid.widths)
    open(path,"w") do io
        println(io,"time_s,normalized_field_L1,normalized_outlet_L1,max_bulk_difference_kg_m3,nx,nz")
        for (t,u,v) in zip(base.t,base.u,fine.u)
            iszero(t) && continue  # normalized number error is undefined for the empty state
            N = population(problem,u,:crystals)
            R = population(problem,v,:crystals)
            field = sum(abs.(N.-R).*widths)/sum(N.*widths)
            outlet = outlet_error(problem,v,problem,u)
            bulk_error = maximum(abs.(bulk(problem,v,:c).-bulk(problem,u,:c)))
            println(io,join((t,field,outlet,bulk_error,size(N)...),','))
        end
    end
end

"""Independent size/axial refinement against uniform-size WENO5 + axial MUSCL/RK4.
The reference is numerical, not an analytical solution. Full mode enumerates
SI mesh pairs with the supported fixed axial MUSCL treatment explicitly labelled.
"""
function run_benchmark(;output=joinpath(@__DIR__,"..","..","docs","src","assets","continuous_precipitation"),
        full=false,quick=nothing)
    quick === nothing || (full = !quick)
    mkpath(output)
    # This is the affordable comparison/check stage. The CLI follows it with
    # run_size_resolution(), which replaces the displayed reference and rebases
    # comparison errors without repeating expensive fine-grid time/axial checks.
    refsize = full ? (250,750) : (144,128)
    spatial_checks = full ? ((200,750),(250,600)) : ((108,128),(144,96))
    configurations = benchmark_configurations(;full)
    timing_rows,check_rows,failure_rows = [],[],[]
    for (mesh,name) in unique([(c.mesh,c.size_method) for c in configurations])
        advance(build(;nx=6,nz=6,mesh,scheme=reconstruction(name),axial_scheme=MUSCL(),tf=.01),[.01])
    end
    # Compile the reference too, before measuring any production solve.
    advance(build(;nx=6,nz=6,mesh=:uniform,scheme=WENO5(),axial_scheme=MUSCL(),tf=.01),[.01])
    reference = build(;nx=refsize[1],nz=refsize[2],mesh=:uniform,scheme=WENO5(),axial_scheme=MUSCL())
    println("Reference WENO5/MUSCL + RK4 ",refsize); flush(stdout)
    measured = @timed advance(reference,[0.,40.,80.,200.])
    solution = measured.value
    println("  reference: ",solution.steps," steps, ",measured.time," s"); flush(stdout)
    push!(timing_rows,("reference",:uniform,"weno5","muscl",refsize...,measured.time,solution.steps))
    export_field(joinpath(output,"field.csv"),reference,solution;max_size=300e-6)
    export_reference_outlets(joinpath(output,"reference_outlets.csv"),reference,solution)
    half = @timed advance(reference,[0.,40.,80.,200.];fraction=.15)
    push!(check_rows,(:uniform,"time_half",refsize...,
        outlet_error(reference,last(half.value.u),reference,last(solution.u)),half.time,half.value.steps))
    export_temporal_checks(joinpath(output,"temporal_checks.csv"),reference,solution,half.value)
    println("  temporal check: ",half.time," s"); flush(stdout)
    for (nx,nz) in spatial_checks
        coarser = build(;nx,nz,mesh=:uniform,scheme=WENO5(),axial_scheme=MUSCL())
        measured_coarse = @timed advance(coarser,[200.])
        difference = outlet_error(coarser,last(measured_coarse.value.u),reference,last(solution.u))
        push!(check_rows,(:uniform,nx==refsize[1] ? "axial_grid" : "size_grid",nx,nz,
            difference,measured_coarse.time,measured_coarse.value.steps))
        println("  spatial check ",(nx,nz),": ",difference,"; ",measured_coarse.time," s"); flush(stdout)
    end
    open(joinpath(output,"outlets.csv"),"w") do outlets
        println(outlets,"mesh,size_method,axial_method,nx,nz,size_m,number_density,volume_density")
        open(joinpath(output,"performance.csv"),"w") do performance
            println(performance,"mesh,size_method,axial_method,nx,nz,seconds,normalized_L1,steps,minimum_state,upper_bin_number_fraction,refinement")
            for config in configurations
                (;mesh,size_method,axial_method,nx,nz,refinement) = config
                println(config); flush(stdout)
                problem = build(;nx,nz,mesh,scheme=reconstruction(size_method),axial_scheme=MUSCL())
                measured_run = try
                    @timed advance(problem,[200.])
                catch error
                    error isa NumericalFailure || rethrow()
                    push!(failure_rows,(mesh,size_method,axial_method,nx,nz,replace(sprint(showerror,error),','=>';','\n'=>' ')))
                    continue
                end
                sol = measured_run.value; u = last(sol.u)
                error = outlet_error(problem,u,reference,last(solution.u))
                grid = first(problem.operator.local_operators).system.populations.crystals.grid
                N = population(problem,u,:crystals)[:,end]
                volume = sum(N .* grid.centers.^3)
                println(performance,join((mesh,size_method,axial_method,nx,nz,measured_run.time,error,sol.steps,
                    minimum(u),N[end]/sum(N),refinement),',')); flush(performance)
                push!(timing_rows,("comparison",mesh,size_method,axial_method,nx,nz,measured_run.time,sol.steps))
                for i in eachindex(N)
                    println(outlets,join((mesh,size_method,axial_method,nx,nz,grid.centers[i],
                        1e12*N[i]/grid.widths[i],N[i]*grid.centers[i]^3/(grid.widths[i]*volume)),','))
                end
            end
        end
    end
    # Revalidate higher-order axial startup independently of production timings.
    for nz in (32,64), fraction in (.3,.03)
        try
            probe = build(;nx=48,nz,mesh=:uniform,scheme=WENO5(),axial_scheme=WENO5(),tf=1.)
            advance(probe,[1.];fraction)
            message = "startup completed to 1 s only"
            push!(failure_rows,(:uniform,"weno5","weno5",48,nz,"fraction=$fraction: $message"))
        catch error
            error isa Union{ArgumentError,NumericalFailure} || rethrow()
            message = replace(sprint(showerror,error),','=>';','\n'=>' ')
            push!(failure_rows,(:uniform,"weno5","weno5",48,nz,"fraction=$fraction: $message"))
        end
    end
    for (name,header,rows) in (
        ("reference_checks.csv","mesh,check,nx,nz,normalized_L1_difference,seconds,steps,reference_nx,reference_nz",
            [tuple(row...,refsize...) for row in check_rows]),
        ("runtimes.csv","role,mesh,size_method,axial_method,nx,nz,seconds,steps",timing_rows),
        ("failures.csv","mesh,size_method,axial_method,nx,nz,message",failure_rows))
        open(joinpath(output,name),"w") do io
            println(io,header)
            for row in rows
                println(io,join(row,','))
            end
        end
    end
    write(joinpath(output,"verification.txt"),
        "Zhang et al. (2024) Case 7A, SI S8. Outlet at 200 s; fields at 0; 40; 80; 200 s.\n"*
        "Executed mode: $(full ? "full SI mesh arrangement with axial MUSCL" : "independent-coordinate documentation study").\n"*
        "Common reference: uniform-size WENO5; axial MUSCL; RK4; mesh=$refsize. Not WENO5 in both directions.\n"*
        "RK4: fraction=.3 divided by summed size/axial advection and axial diffusion rates; local solute-depletion bound; maximum step .25 s.\n"*
        "All native stage checks retained at 1e-8; no clipping. Axial WENO5 startup failures retained.\n"*
        "Error: common-partition piecewise-constant outlet L1 difference divided by reference number; not spline interpolation.\n"*
        "reference_checks.csv isolates time; size; axial refinement. Finite numerical references are not exact.\n"*
        "Uniform/reference work plots use only their dedicated measured solves. Log-mesh sensitivity costs are not used for efficiency rankings.\n"*
        "Single measured solve after separate small compilation warmup. Timing includes stepping; step control; checks; saved states. Excludes assembly; export; plotting.\n"*
        "Julia $(VERSION); CPU $(first(Sys.cpu_info()).model); Julia threads $(Threads.nthreads()); BLAS threads $(PopulationBalanceModels.LinearAlgebra.BLAS.get_num_threads()).\n"*
        "Full option: Nx=(12;25;50;100;200) x Ncol=(37;75;150;300;600); reference=(250;750); not executed by default. WENO5 only on uniform sizes; axial MUSCL in all configurations.\n"*
        "Discrete exchange conserves centre-weighted x^3; closed upper size boundary. Inspect final-bin occupancy; no domain convergence claimed.\n"*
        "No analytical coupled PSD solution for Case 7. SI analytical reaction solution belongs to Case 6.\n")
    (;reference,solution,check_rows,failure_rows)
end

"""Replace reference-dependent errors without rerunning unchanged comparisons.
Solve timings and states remain those of the original exported configurations.
"""
function rebase_comparison_errors(output,reference,reference_state)
    path = joinpath(output,"performance.csv")
    isfile(path) || return
    outlets = split.(readlines(joinpath(output,"outlets.csv"))[2:end],',')
    lines = readlines(path)
    rows = split.(lines[2:end],',')
    for row in rows
        mesh,name,axial = Symbol(row[1]),row[2],row[3]
        nx,nz = parse(Int,row[4]),parse(Int,row[5])
        problem = build(;nx,nz,mesh,scheme=reconstruction(name),axial_scheme=reconstruction(axial))
        grid = first(problem.operator.local_operators).system.populations.crystals.grid
        saved = filter(r->r[1:5]==row[1:5],outlets)
        length(saved)==nx || throw(ArgumentError("missing saved outlet for $(row[1:5])"))
        state = copy(problem.u0)
        population(problem,state,:crystals)[:,end] .= parse.(Float64,getindex.(saved,7)) .* grid.widths ./ 1e12
        row[7] = string(outlet_error(problem,state,reference,reference_state))
    end
    open(path,"w") do io
        println(io,first(lines))
        for row in rows
            println(io,join(row,','))
        end
    end
end

"""Size-resolution study for the plotted number-density peak.

All runs use uniform sizes, axial MUSCL and size WENO5 with RK4. No smoothing
or interpolation is applied to the exported cell averages. Existing comparison
timings are retained; their errors are recomputed against the new fine outlet.
"""
function run_size_resolution(;sizes=(144,288,576,1152),nz=128,
        output=joinpath(@__DIR__,"..","..","docs","src","assets","continuous_precipitation"))
    length(sizes)>=2 && all(n->n isa Integer && n>=6,sizes) &&
        issorted(sizes) && all(>(0),diff(collect(sizes))) ||
        throw(ArgumentError("sizes must contain at least two strictly increasing integer bin counts, each at least 6"))
    mkpath(output)
    advance(build(;nx=6,nz=6,mesh=:uniform,scheme=WENO5(),axial_scheme=MUSCL(),tf=.01),[.01])
    runs = map(sizes) do nx
        problem = build(;nx,nz,mesh=:uniform,scheme=WENO5(),axial_scheme=MUSCL())
        GC.gc()  # excluded from the single measured solve
        measured = @timed advance(problem,[0.,40.,80.,200.];progress=true)
        println("Size resolution ",nx," × ",nz,": ",measured.time," s; ",measured.value.steps," steps")
        flush(stdout)
        (;problem,solution=measured.value,seconds=measured.time)
    end
    export_size_resolution(output,runs)
end

# Shared by the reproduction driver and imports of already executed solves.
function export_size_resolution(output,runs)
    reference = last(runs).problem
    solution = last(runs).solution
    reference_state = last(solution.u)
    open(joinpath(output,"size_resolution.csv"),"w") do summary
        println(summary,"nx,nz,dx_um,seconds,steps,minimum_final_density_m4,normalized_L1_to_finest,number_per_m3,M3,solute_outlet_kg_m3,number_below_10um_per_m3,upper_bin_fraction,minimum_final_solute_kg_m3")
        open(joinpath(output,"size_resolution_outlets.csv"),"w") do outlets
            println(outlets,"nx,nz,x_left,x_right,number_density")
            for run in runs
                (;problem,solution,seconds) = run
                grid = first(problem.operator.local_operators).system.populations.crystals.grid
                state = last(solution.u)
                N = population(problem,state,:crystals)[:,end]
                minimum_density = 1e12*minimum(population(problem,state,:crystals)./grid.widths)
                nx,nz = length(grid),length(problem.operator.domain.grid)
                fine_fraction = clamp.((10e-6 .- grid.edges[1:end-1])./grid.widths,0,1)
                error = outlet_error(problem,state,reference,reference_state)
                println(summary,join((nx,nz,first(grid.widths)*1e6,seconds,solution.steps,
                    minimum_density,error,1e12*sum(N),1e12*moment(grid,N,3),
                    bulk(problem,state,:c)[end],1e12*sum(N.*fine_fraction),N[end]/sum(N),
                    minimum(bulk(problem,state,:c))),','))
                for i in eachindex(N)
                    println(outlets,join((nx,nz,grid.edges[i],grid.edges[i+1],1e12*N[i]/grid.widths[i]),','))
                end
            end
        end
    end
    export_field(joinpath(output,"field.csv"),reference,solution;max_size=300e-6)
    export_reference_outlets(joinpath(output,"reference_outlets.csv"),reference,solution)
    rebase_comparison_errors(output,reference,reference_state)
    # Historical checks name their own baseline mesh; replacing the reference
    # must not silently relabel a coarse-grid time check as a fine-grid check.
    timing_path = joinpath(output,"runtimes.csv")
    nx,nz = size(population(reference,reference_state,:crystals))
    rows = isfile(timing_path) ? split.(readlines(timing_path)[2:end],',') : []
    for row in rows
        row[1]=="reference" && (row[1]="historical_reference")
    end
    filter!(r->r[1]!="size_resolution",rows)
    open(timing_path,"w") do io
        println(io,"role,mesh,size_method,axial_method,nx,nz,seconds,steps")
        for row in rows
            println(io,join(row,','))
        end
        for (i,run) in enumerate(runs)
            n = size(population(run.problem,last(run.solution.u),:crystals),1)
            role = i==length(runs) ? "reference" : "size_resolution"
            println(io,join((role,"uniform","weno5","muscl",n,nz,run.seconds,run.solution.steps),','))
        end
    end
    settings = joinpath(output,"verification.txt")
    text = isfile(settings) ? first(split(read(settings,String),"Size-resolution follow-up:")) : ""
    isempty(text) && (text = "Zhang et al. (2024), Case 7A. Physical parameters in SI: $(parameters()).\n"*
        "Common reference: uniform-size WENO5; axial MUSCL; RK4; mesh=($nx, $nz).\n"*
        "RK4 fraction=.3; summed transport and local solute-depletion bounds; step cap .25 s; native stage checks at 1e-8.\n")
    text = replace(text,r"Common reference: [^\n]*"=>
        "Common reference: uniform-size WENO5; axial MUSCL; RK4; mesh=($nx, $nz). Not WENO5 in both directions.")
    write(settings,text*"Size-resolution follow-up:\n"*
        "size_resolution.csv and size_resolution_outlets.csv hold independently solved size meshes at fixed axial resolution. All save 0; 40; 80; 200 s; same RK4 controller and physical parameters.\n"*
        "New field.csv and reference_outlets.csv use the finest size mesh ($nx; $nz). Field display cropped to 300 micrometres; outlet and inventories retain the entire 900-micrometre domain. No interpolation; no clipping.\n"*
        "performance.csv errors recomputed from unchanged saved comparison outlets. Their solve timings are historical; not remeasured during the size study.\n"*
        "reference_checks.csv and temporal_checks.csv retain explicit mesh dimensions: checks on an older mesh do not certify the finer reference.\n"*
        "Size-study timings: one measured solve after separate compilation warmup; GC before timing; no concurrent substantial compute. Julia $(VERSION); CPU $(first(Sys.cpu_info()).model); Julia threads $(Threads.nthreads()); BLAS threads $(PopulationBalanceModels.LinearAlgebra.BLAS.get_num_threads()).\n")
    (;reference,solution)
end
