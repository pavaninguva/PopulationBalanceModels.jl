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

function export_field(path,problem,solution)
    grid = first(problem.operator.local_operators).system.populations.crystals.grid
    axial = problem.operator.domain.grid
    open(path,"w") do io
        println(io,"time,z_m,size_m,number_density,concentration_kg_m3,z_left,z_right,x_left,x_right")
        for (t,u) in zip(solution.t,solution.u), j in axes(u,2), i in eachindex(grid.centers)
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
        println(io,"time_s,normalized_field_L1,normalized_outlet_L1,max_bulk_difference_kg_m3")
        for (t,u,v) in zip(base.t,base.u,fine.u)
            iszero(t) && continue  # normalized number error is undefined for the empty state
            N = population(problem,u,:crystals)
            R = population(problem,v,:crystals)
            field = sum(abs.(N.-R).*widths)/sum(N.*widths)
            outlet = outlet_error(problem,v,problem,u)
            bulk_error = maximum(abs.(bulk(problem,v,:c).-bulk(problem,u,:c)))
            println(io,join((t,field,outlet,bulk_error),','))
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
    export_field(joinpath(output,"field.csv"),reference,solution)
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
        ("reference_checks.csv","mesh,check,nx,nz,normalized_L1_difference,seconds,steps",check_rows),
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
        "Starting revision afda48e603a466b7b981f63d0d4d34ce8960c18a; initially clean working tree.\n"*
        "Full option: Nx=(12;25;50;100;200) x Ncol=(37;75;150;300;600); reference=(250;750); not executed by default. WENO5 only on uniform sizes; axial MUSCL in all configurations.\n"*
        "Discrete exchange conserves centre-weighted x^3; closed upper size boundary. Inspect final-bin occupancy; no domain convergence claimed.\n"*
        "No analytical coupled PSD solution for Case 7. SI analytical reaction solution belongs to Case 6.\n")
    (;reference,solution,check_rows,failure_rows)
end
