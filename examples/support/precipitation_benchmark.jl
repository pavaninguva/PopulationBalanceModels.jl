# Included inside ContinuousPrecipitation. Timings exclude assembly and export.
# The full study enumerates the paper's Cartesian product, not diagonal pairs.
function benchmark_configurations(;full=false)
    sizes = full ? (12,25,50,100,200) : (12,24)
    axial = full ? (37,75,150,300,600) : (16,32,48)
    configurations = NamedTuple[]
    for mesh in (:log,:uniform), scheme in (Upwind(),MUSCL())
        label = scheme isa Upwind ? "upwind" : "muscl"
        for nx in sizes, nz in axial
            push!(configurations,(;mesh,size_method=label,axial_method=label,nx,nz))
        end
    end
    # Isolate size reconstruction: all three use the same axial MUSCL residual.
    for label in ("upwind","weno5"), nx in sizes, nz in axial
        push!(configurations,(mesh=:uniform,size_method=label,axial_method="muscl",nx,nz))
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

function export_reference_outlets(path,references)
    open(path,"w") do io
        println(io,"mesh,x_left,x_right,number_density")
        for mesh in (:log,:uniform)
            problem,solution = references[mesh]
            grid = first(problem.operator.local_operators).system.populations.crystals.grid
            N = population(problem,last(solution.u),:crystals)[:,end]
            for i in eachindex(N)
                println(io,join((mesh,grid.edges[i],grid.edges[i+1],1e12*N[i]/grid.widths[i]),','))
            end
        end
    end
end

"""Documentation-budget two-coordinate study, or the expensive SI mesh product.
The reference is a MUSCL/MUSCL numerical solution on each size-grid family.
`full=true` is an unexecuted study option unless its exports are generated.
"""
function run_benchmark(;output=joinpath(@__DIR__,"..","..","docs","src","assets","continuous_precipitation"),
        full=false,quick=nothing)
    # Keep the earlier keyword working; full mode now uses all 25 mesh pairs.
    quick === nothing || (full = !quick)
    mkpath(output)
    refsize = full ? (250,750) : (48,64)
    spatial_checks = full ? ((200,750),(250,600)) : ((36,64),(48,48))
    configurations = benchmark_configurations(;full)
    references = Dict()
    timing_rows = []
    check_rows = []
    failure_rows = []

    # Compile each size/axial pairing separately before any measured solves.
    for (mesh,size_method,axial_method) in unique([(c.mesh,c.size_method,c.axial_method) for c in configurations])
        advance(build(;nx=6,nz=6,mesh,scheme=reconstruction(size_method),
            axial_scheme=reconstruction(axial_method),tf=.01),[.01])
    end
    for mesh in (:log,:uniform)
        pb = build(;nx=refsize[1],nz=refsize[2],mesh,scheme=MUSCL(),axial_scheme=MUSCL())
        println("Reference ",mesh," ",refsize); flush(stdout)
        measured = @timed advance(pb,[80.,200.])
        sol = measured.value
        println("  reference: ",sol.steps," steps, ",measured.time," s"); flush(stdout)
        references[mesh] = (pb,sol)
        push!(timing_rows,("reference",mesh,"muscl","muscl",refsize...,measured.time,sol.steps))
        mesh === :log && export_field(joinpath(output,"field.csv"),pb,sol)
        # Time refinement keeps the spatial grid fixed.
        measured_half = @timed advance(pb,[80.,200.];fraction=.15)
        println("  half-fraction check: ",measured_half.value.steps," steps, ",measured_half.time," s"); flush(stdout)
        push!(check_rows,(mesh,"time_half",refsize...,outlet_error(pb,last(measured_half.value.u),pb,last(sol.u)),
            measured_half.time,measured_half.value.steps))
        for (nx,nz) in spatial_checks
            coarser = build(;nx,nz,mesh,scheme=MUSCL(),axial_scheme=MUSCL())
            measured_coarse = @timed advance(coarser,[200.])
            println("  spatial check ",(nx,nz),": ",measured_coarse.time," s"); flush(stdout)
            difference = outlet_error(coarser,last(measured_coarse.value.u),pb,last(sol.u))
            push!(check_rows,(mesh,nx==refsize[1] ? "axial_grid" : "size_grid",nx,nz,difference,
                measured_coarse.time,measured_coarse.value.steps))
        end
    end
    logpb,logsol = references[:log]
    uniformpb,uniformsol = references[:uniform]
    # Algebraic comparison of existing states: no additional timed solve.
    family_difference = outlet_error(uniformpb,last(uniformsol.u),logpb,last(logsol.u))
    push!(check_rows,("uniform/log","mesh_family",refsize...,family_difference,NaN,0))
    export_reference_outlets(joinpath(output,"reference_outlets.csv"),references)
    open(joinpath(output,"outlets.csv"),"w") do outlets
        println(outlets,"mesh,size_method,axial_method,nx,nz,size_m,number_density,volume_density")
        open(joinpath(output,"performance.csv"),"w") do performance
            println(performance,"mesh,size_method,axial_method,nx,nz,seconds,normalized_L1,steps,minimum_state,upper_bin_number_fraction")
            for config in configurations
                (;mesh,size_method,axial_method,nx,nz) = config
                println(config); flush(stdout)
                pb = build(;nx,nz,mesh,scheme=reconstruction(size_method),axial_scheme=reconstruction(axial_method))
                measured = try
                    @timed advance(pb,[200.])
                catch error
                    error isa NumericalFailure || rethrow()
                    push!(failure_rows,(mesh,size_method,axial_method,nx,nz,replace(sprint(showerror,error),','=>';','\n'=>' ')))
                    continue
                end
                sol = measured.value; u = last(sol.u)
                refpb,refsol = references[mesh]
                error = outlet_error(pb,u,refpb,last(refsol.u))
                grid = first(pb.operator.local_operators).system.populations.crystals.grid
                N = population(pb,u,:crystals)[:,end]
                volume = sum(N .* grid.centers.^3)
                println(performance,join((mesh,size_method,axial_method,nx,nz,measured.time,error,sol.steps,
                    minimum(u),N[end]/sum(N)),',')); flush(performance)
                push!(timing_rows,("comparison",mesh,size_method,axial_method,nx,nz,measured.time,sol.steps))
                for i in eachindex(N)
                    println(outlets,join((mesh,size_method,axial_method,nx,nz,grid.centers[i],
                        1e12*N[i]/grid.widths[i],N[i]*grid.centers[i]^3/(grid.widths[i]*volume)),','))
                end
            end
        end
    end
    # Preserve the failed higher-order startup check, without relaxing state checks.
    try
        probe = build(;nx=24,nz=32,mesh=:uniform,scheme=WENO5(),axial_scheme=WENO5(),tf=1.)
        advance(probe,[1.])
        push!(failure_rows,(:uniform,"weno5","weno5",24,32,"startup completed to 1 s only; not included in benchmark"))
    catch error
        error isa Union{ArgumentError,NumericalFailure} || rethrow()
        push!(failure_rows,(:uniform,"weno5","weno5",24,32,replace(sprint(showerror,error),','=>';','\n'=>' ')))
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
        "Zhang et al. (2024) Case 7A, SI S8. Observation at 200 s; fields at 80 and 200 s.\n"*
        "Executed mode: $(full ? "full SI mesh arrangement" : "documentation budget").\n"*
        "References: size/axial MUSCL/MUSCL, $(refsize), separate logarithmic/uniform size meshes; uniform axial mesh.\n"*
        "SSPRK33: fraction=.3 divided by axial advection+diffusion+maximum size advection rates; local solute depletion bound; maximum step .25 s.\n"*
        "No clipping; all native stage checks retained at their existing tolerance.\n"*
        "Error: common-partition piecewise-constant outlet L1 difference divided by reference number; not spline interpolation.\n"*
        "See reference_checks.csv for temporal and separate size/axial differences; finite numerical references are not exact.\n"*
        "One measured solve after separate compilation warmup; excludes assembly, file export and plotting; includes step control, state checks and saved states.\n"*
        "Julia $(VERSION); CPU $(first(Sys.cpu_info()).model); Julia threads $(Threads.nthreads()); BLAS threads $(PopulationBalanceModels.LinearAlgebra.BLAS.get_num_threads()).\n"*
        "Run starting from 84b0860; solve timings newly measured for this documentation revision.\n"*
        "Full SI option: Cartesian product Nx=(12,25,50,100,200), Ncol=(37,75,150,300,600), reference=(250,750); not executed in documentation-budget mode.\n"*
        "Discrete solute exchange conserves centre-weighted x^3. Closed size boundary; inspect upper-bin occupancy.\n"*
        "Case 6 SI analytical reaction solution does not apply to this coupled precipitation model. Case 7B not claimed.\n")
    (;references,check_rows,failure_rows)
end
