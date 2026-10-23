module PaperPrecipitation
using PopulationBalanceModels, TOML
include("../../common.jl")
using .PaperWorkflow
include(joinpath(ROOT,"examples","continuous_precipitation.jl"))
include(joinpath(PAPER,"precipitation_solver.jl"))
const C = ContinuousPrecipitation
const S = PaperPrecipitationSolver
grid(pb) = first(pb.operator.local_operators).system.populations.crystals.grid

function export_run(output,pb,solution)
    g,z,p = grid(pb),pb.operator.domain.grid,pb.p
    files = String["size_edges.csv","axial_edges.csv","accounting.csv","outlet.csv"]
    csv(joinpath(output,"size_edges.csv"),[(;edge=i,size_m=x) for (i,x) in enumerate(g.edges)])
    csv(joinpath(output,"axial_edges.csv"),[(;edge=i,z_m=x) for (i,x) in enumerate(z.edges)])
    accounting = NamedTuple[]
    for (frame,(t,u)) in enumerate(zip(solution.t,solution.u))
        N = population(pb,u,:crystals)
        c = bulk(pb,u,:c)
        name = "population_$(frame).csv.gz"
        open(pipeline(`gzip -c`;stdout=joinpath(output,name)),"w") do io
            println(io,"number_per_m3")
            for value in N
                println(io,1e12value)
            end
        end
        push!(files,name)
        name = "bulk_$(frame).csv"
        csv(joinpath(output,name),[(;cell=j,z_m=z.centers[j],solute_kg_m3=c[j]) for j in eachindex(c)])
        push!(files,name)
        du = zero(u);rhs!(du,u,pb.operator,p,t)
        F = axial_fluxes(pb.operator,u,p,t)
        weights = vcat(1e12*p.rho*p.kv.*g.centers.^3,1.)
        material_rate = sum(weights.*du.*z.widths')
        inlet = sum(weights.*F[:,1])
        outlet = sum(weights.*F[:,end])
        source = 0.0
        for j in eachindex(c)
            # Same physical nucleation law; no solver implementation here.
            localop = pb.operator.local_operators[j].system.populations.crystals
            ctx = (;bulk=(c=c[j],),grid=g,N=view(N,:,j))
            source += 1e12*C.nucleation(p,t,ctx)*z.widths[j]
        end
        M3 = 1e12*sum(g.centers.^3 .* N .* z.widths')
        solid = p.rho*p.kv*M3
        solute = sum(c.*z.widths)
        push!(accounting,(;time_s=t,number_per_m2=1e12sum(N.*z.widths'),integrated_M3_m=M3,
            dissolved_inventory_kg_m2=solute,solid_inventory_kg_m2=solid,total_inventory_kg_m2=solute+solid,
            column_dissolved_kg=p.area*p.porosity*solute,column_solid_kg=p.area*p.porosity*solid,
            column_total_kg=p.area*p.porosity*(solute+solid),
            material_inlet_kg_m2_s=inlet,material_outlet_kg_m2_s=outlet,material_derivative_kg_m2_s=material_rate,
            inlet_kg_s=p.area*p.porosity*inlet,outlet_kg_s=p.area*p.porosity*outlet,
            material_rate_defect_kg_m2_s=material_rate-(inlet-outlet),
            nucleation_number_per_m2_s=source,minimum_density_m4=1e12minimum(N./g.widths),
            negative_number_per_m2=-1e12sum(min.(N,0.).*z.widths'),minimum_solute_kg_m3=minimum(c),
            upper_bin_fraction=sum(N[end,:].*z.widths)/sum(N.*z.widths'),
            outlet_number_per_m3=1e12sum(N[:,end]),outlet_solute_kg_m3=c[end]))
    end
    csv(joinpath(output,"accounting.csv"),accounting)
    N = 1e12population(pb,last(solution.u),:crystals)[:,end]
    V = sum(g.centers.^3 .* N)
    csv(joinpath(output,"outlet.csv"),[(;left_m=g.edges[i],right_m=g.edges[i+1],
        number_per_m3=N[i],number_density_m4=N[i]/g.widths[i],
        volume_density_per_m=N[i]*g.centers[i]^3/(g.widths[i]*V)) for i in eachindex(N)])
    files
end

function load_run(folder)
    info = TOML.parsefile(joinpath(folder,"provenance.toml"))
    s = info["settings"]
    pb = C.build(nx=s["nx"],nz=s["nz"],scheme=WENO5(),axial_scheme=MUSCL(),mesh=:log,tf=s["tf"])
    states = typeof(pb.u0)[]
    for frame in eachindex(s["times"])
        u = copy(pb.u0)
        values = open(`gzip -dc $(joinpath(folder,"population_$(frame).csv.gz"))`) do io
            readline(io)
            parse.(Float64,readlines(io))
        end
        population(pb,u,:crystals) .= reshape(values,size(population(pb,u,:crystals)))./1e12
        bulkdata = split.(readlines(joinpath(folder,"bulk_$(frame).csv"))[2:end],',')
        bulk(pb,u,:c) .= parse.(Float64,getindex.(bulkdata,3))
        push!(states,u)
    end
    pb,PBMSolution(s["times"],states,pb,info["summary"]["steps"])
end

function field_error(pb,u,reference,v)
    a,b = grid(pb),grid(reference)
    z,w = pb.operator.domain.grid,reference.operator.domain.grid
    xe,ze = sort!(unique(vcat(a.edges,b.edges))),sort!(unique(vcat(z.edges,w.edges)))
    A = population(pb,u,:crystals)./a.widths
    B = population(reference,v,:crystals)./b.widths
    total,normalization = 0.0,0.0
    for j in 1:length(ze)-1
        zj = (ze[j]+ze[j+1])/2
        ja,jb = min(searchsortedlast(z.edges,zj),length(z)),min(searchsortedlast(w.edges,zj),length(w))
        for i in 1:length(xe)-1
            x = (xe[i]+xe[i+1])/2
            ia,ib = min(searchsortedlast(a.edges,x),length(a)),min(searchsortedlast(b.edges,x),length(b))
            measure = (xe[i+1]-xe[i])*(ze[j+1]-ze[j])
            total += abs(A[ia,ja]-B[ib,jb])*measure
            normalization += B[ib,jb]*measure
        end
    end
    total/normalization
end

function run(mode="production")
    ctx = context(@__DIR__,mode;support=[joinpath(PAPER,"precipitation_solver.jl")])
    cfg = ctx.config
    runs = Dict{String,String}()
    runs["solver_verification"] = cached(ctx,"solver_verification",Dict(
            "nx"=>12,"nz"=>12,"tf"=>10.,"maximum_bdf_order"=>cfg["maximum_bdf_order"],
            "late_maximum_bdf_order"=>cfg["late_maximum_bdf_order"],"bdf_restart_time"=>cfg["bdf_restart_time"],
            "role"=>"independent native/implicit verification")) do output
        pb = C.build(nx=12,nz=12,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=10.)
        ts = [1.,5.,10.]
        implicit = S.advance(pb,ts;reltol=1e-8,population_abstol=1e-16,bulk_abstol=1e-12,dtmax=.1,
            max_order=cfg["maximum_bdf_order"],late_max_order=cfg["late_maximum_bdf_order"],
            restart_time=cfg["bdf_restart_time"],progress=false)
        native = C.advance(pb,ts;fraction=.075)
        rows = map(eachindex(ts)) do i
            reference = population(pb,native.u[i],:crystals)
            difference = sum(abs.(population(pb,implicit.u[i],:crystals).-reference))/sum(abs,reference)
            bulk_difference = maximum(abs.(bulk(pb,implicit.u[i],:c).-bulk(pb,native.u[i],:c)))
            difference<2e-3 && bulk_difference<1e-4 || error("CVODE/native verification failed at $(ts[i]) s")
            (;time_s=ts[i],relative_population_difference=difference,maximum_bulk_difference=bulk_difference,
                implicit_steps=implicit.steps,native_steps=native.steps)
        end
        csv(joinpath(output,"verification.csv"),rows)
        (["verification.csv"],Dict("role"=>"verification","status"=>"passed"))
    end
    warm = C.build(nx=8,nz=6,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=.02)
    bulk(warm,warm.u0,:c) .= 2warm.p.ceq
    population(warm,warm.u0,:crystals) .= 1e-12
    S.advance(warm,[.01,.02];max_order=cfg["maximum_bdf_order"],
        late_max_order=cfg["late_maximum_bdf_order"],restart_time=.01,progress=false)
    configurations = [("paper_field",cfg["paper_field_nx"],cfg["paper_field_nz"],1.)]
    append!(configurations,[("outlet_Nz$(nz)",cfg["outlet_nx"],nz,1.) for nz in cfg["outlet_nz"]])
    append!(configurations,[("size_refinement",cfg["size_check_nx"],cfg["reference_nz"],1.),
        ("axial_refinement",cfg["reference_nx"],cfg["axial_check_nz"],1.),
        ("reference",cfg["reference_nx"],cfg["reference_nz"],1.),
        ("time_refinement",cfg["reference_nx"],cfg["reference_nz"],cfg["temporal_refinement_factor"])])
    for (id,nx,nz,factor) in configurations
        settings = Dict("nx"=>nx,"nz"=>nz,"mesh"=>"logarithmic sizes; uniform axial",
            "size_reconstruction"=>"geometry-aware WENO5","axial_reconstruction"=>"MUSCL",
            "time_solver"=>"CVODE_BDF/left-preconditioned GMRES(50)",
            "maximum_bdf_order"=>cfg["maximum_bdf_order"],
            "late_maximum_bdf_order"=>cfg["late_maximum_bdf_order"],"bdf_restart_time"=>cfg["bdf_restart_time"],
            "stability_limit_detection"=>cfg["stability_limit_detection"],
            "preconditioner"=>"factored bidiagonal Upwind size / tridiagonal axial transport, with diagonal solute depletion; original WENO5/MUSCL residual unchanged",
            "relative_tolerance"=>cfg["relative_tolerance"]/factor,
            "population_absolute_tolerance"=>cfg["population_absolute_tolerance"]/factor,
            "bulk_absolute_tolerance"=>cfg["bulk_absolute_tolerance"]/factor,
            "maximum_step"=>cfg["maximum_step"]/(factor>1 ? 2 : 1),
            "times"=>cfg["times"],"tf"=>cfg["final_time"],"physical_parameters"=>Dict(string(k)=>v for (k,v) in pairs(C.parameters())),
            "checks"=>"native checker after every accepted CVODE step at existing 1e-8; no clipping; Newton iterates are auxiliary",
            "population_scaling"=>1e12,"literature"=>"Zhang2024 SI Table S8; rounded Q=1.67e-7; public CADET uses 1.6666666667e-7")
        folder = cached(ctx,id,settings) do output
            setup = @timed C.build(;nx,nz,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=cfg["final_time"])
            pb = setup.value
            # Enforce use of the actual geometry-aware prepared reconstruction.
            popop = first(pb.operator.local_operators).system.populations.crystals
            only(filter(x->x.term isa Growth,popop.prepared)).cache isa PopulationBalanceModels._NonuniformWENO5 ||
                error("nonuniform WENO5 geometry was not prepared")
            GC.gc()
            measured = @timed S.advance(pb,cfg["times"];
                reltol=settings["relative_tolerance"],population_abstol=settings["population_absolute_tolerance"],
                bulk_abstol=settings["bulk_absolute_tolerance"],dtmax=settings["maximum_step"],
                max_order=settings["maximum_bdf_order"],
                late_max_order=settings["late_maximum_bdf_order"],restart_time=settings["bdf_restart_time"],
                stability_limit_detect=settings["stability_limit_detection"])
            solution = measured.value
            files = export_run(output,pb,solution)
            diag = diagnostics(solution)
            summary = Dict("status"=>"completed","seconds"=>measured.time,"setup_seconds"=>setup.time,
                "steps"=>solution.steps,"bytes"=>measured.bytes,"gc_seconds"=>measured.gctime,
                "error_test_failures"=>diag.error_test_failures,"checked_steps"=>diag.checked_steps,
                "restarts"=>diag.restarts,"phase_steps"=>diag.phase_steps,
                "phase_end_times"=>diag.phase_end_times,"phase_maximum_orders"=>diag.phase_maximum_orders,
                "phase_maximum_observed_orders"=>diag.phase_maximum_observed_orders,
                "phase_error_test_failures"=>diag.phase_error_test_failures,
                "minimum_scaled_population"=>diag.minimum_scaled_population,"minimum_solute"=>diag.minimum_solute,
                "first_bin_width_m"=>first(grid(pb).widths),
                "nucleus_discrete_to_physical_material_ratio"=>(first(grid(pb).centers)/pb.p.xc)^3,
                "completed_time"=>last(solution.t))
            println("Completed ",id," ",(nx,nz)," in ",measured.time," s; ",solution.steps," steps");flush(stdout)
            files,summary
        end
        runs[id] = folder
        index(ctx,runs) # completed matching runs remain recoverable after interruption
    end
    reference,refsol = load_run(runs["reference"])
    rows = NamedTuple[]
    for id in ("time_refinement","size_refinement","axial_refinement","paper_field")
        pb,sol = load_run(runs[id])
        for (i,t) in enumerate(sol.t)
            reference_outlet_number = sum(population(reference,refsol.u[i],:crystals)[:,end])
            # An effectively empty outlet cannot define a normalized PSD.
            # Keep a raw ratio only when its denominator is positive; the
            # analysis table separately flags signals below absolute tolerances.
            outlet_difference = reference_outlet_number>0 ? C.outlet_error(pb,sol.u[i],reference,refsol.u[i]) : NaN
            push!(rows,(;comparison=id,time_s=t,nx=length(grid(pb)),nz=length(pb.operator.domain.grid),
                reference_nx=length(grid(reference)),reference_nz=length(reference.operator.domain.grid),
                normalized_field_L1=field_error(pb,sol.u[i],reference,refsol.u[i]),
                normalized_outlet_L1=outlet_difference))
        end
    end
    csv(joinpath(ctx.output,"refinement.csv"),rows)
    index(ctx,runs)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    PaperPrecipitation.run("--smoke" in ARGS ? "smoke" : "production")
end
