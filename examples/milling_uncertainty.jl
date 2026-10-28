module MillingUncertainty
using PopulationBalanceModels, LinearAlgebra, Random, SHA, TOML, Printf
import Distributions, JSON3, Pkg

include("support/milling_model.jl")
include("support/milling_solver.jl")

const ROOT = normpath(joinpath(@__DIR__,".."))
const CONFIG = joinpath(@__DIR__,"data","milling_uncertainty","reconstruction.toml")
const ASSETS = joinpath(ROOT,"docs","src","assets","milling_uncertainty")

function output_times()
    sort!(unique(vcat(.001,.01,collect(.1:.1:2),collect(3.:10.),collect(12.:2:28.),
        29.,29.9,29.999,30.,30.001,30.1,30.5,31.,collect(32.:2:60.),
        collect(65.:5:100.),collect(110.:10:200.))))
end

physical(z) = NamedTuple{MillingModel.PARAMETER_NAMES}(Tuple(
    getproperty(MillingModel.MEAN,k)+getproperty(MillingModel.SD,k)*getproperty(z,k)
    for k in MillingModel.PARAMETER_NAMES))

function inputs()
    marginals = NamedTuple{MillingModel.PARAMETER_NAMES}(ntuple(_->Distributions.Normal(),4))
    UncertainInputs(marginals;map=physical,valid=MillingModel.admissible,
        coordinates=:standard_normal,metadata=(transformation=:table1_affine,
        joint_law=:independent_gaussian,physical_units=(:dimensionless,:inverse_RPM_seconds,
        :dimensionless,:dimensionless)))
end

function state_diagnostics(pb,sol)
    p=pb.p
    material=[p.rho*p.number_scale*dot(p.pivots,population(pb,u,:mill)) for u in sol.u]
    errors=[material[i]+bulk(pb,sol.u[i],:discharged_mass)-p.mass_feed*t for (i,t) in enumerate(sol.t)]
    occupancy=[p.rho*p.number_scale*p.pivots[end]*last(population(pb,u,:mill)) for u in sol.u]
    (;steps=sol.steps,minimum_inventory=minimum(minimum.(sol.u)),
        maximum_mass_error_kg=maximum(abs,errors),
        maximum_upper_pivot_mass_kg=maximum(occupancy),
        final_fines_mass_kg=bulk(pb,last(sol.u),:fines_mass))
end

"""Build a data-free forward UQ problem. Rebuild daughter projections when sigma
changes. All requested times share one segmented forward solve per realization.
"""
function problem(;bins=100,times=output_times(),reltol=1e-9,abstol=1e-14,
        quadrature_order=8,records=nothing,progress=0)
    base=MillingModel.build(;bins,quadrature_order)
    setup=(theta,base)->MillingModel.build(theta;bins,quadrature_order)
    calls=Ref(0)
    solver=function(pb,ts)
        calls[]+=1
        if progress>0 && (calls[]==1 || calls[]%progress==0)
            println("  forward solve ",calls[]);flush(stdout)
        end
        sol=MillingSolver.solve(pb,ts;reltol,abstol)
        records===nothing || push!(records,state_diagnostics(pb,sol))
        sol
    end
    UQProblem(base,inputs();setup,solver,times,outputs=(d50=MillingModel.median_observable(),),
        output_coordinates=(d50=(units=:micrometre,weighting=:instantaneous_outlet_volume,
            interpolation=:linear_diameter_CDF),))
end

function write_csv(path,header,rows)
    mkpath(dirname(path))
    writer=io->begin
        println(io,join(header,','))
        for row in rows
            println(io,join(row,','))
        end
    end
    if endswith(path,".gz")
        open(pipeline(`gzip -n -c`,stdout=path),"w") do io
            writer(io)
        end
    else
        open(writer,path,"w")
    end
end

write_json(path,object)=open(io->JSON3.pretty(io,object),path,"w")

function export_ensemble(folder,name,result)
    write_csv(joinpath(folder,name*"_inputs.csv"),
        vcat(["sample_id","success"],"z_".*string.(collect(MillingModel.PARAMETER_NAMES)),
             string.(collect(MillingModel.PARAMETER_NAMES))),
        (vcat(Any[result.ids[i],result.success[i]],collect(values(result.inputs[i])),
            result.physical[i]===nothing ? fill(NaN,4) : collect(values(result.physical[i])))
            for i in eachindex(result.ids)))
    write_csv(joinpath(folder,name*"_outputs.csv.gz"),vcat("sample_id","t".*string.(eachindex(result.problem.times))),
        ([result.ids[i],result.values[:,i]...] for i in eachindex(result.ids)))
    write_json(joinpath(folder,name*"_diagnostics.json"),diagnostics(result))
end

function summary_rows(result,pce=nothing)
    s=uq_summary(result;covariance=false)
    analytic=pce===nothing ? nothing : uq_summary(pce;covariance=false)
    [begin
        mean=analytic===nothing ? s.mean.d50[1,i] : analytic.mean.d50[1,i]
        sd=sqrt(analytic===nothing ? s.variance.d50[1,i] : analytic.variance.d50[1,i])
        mcse=s.mean_mcse===nothing ? NaN : s.mean_mcse.d50[1,i]
        (t,mean,sd,s.quantiles.d50[1,i,1],s.quantiles.d50[1,i,2],s.quantiles.d50[1,i,3],
            mean-1.96sd,mean+1.96sd,mcse)
    end for (i,t) in enumerate(result.problem.times)]
end

function export_pce(folder,name,pce,cheap,common,reference)
    export_ensemble(folder,name*"_training",pce.training)
    write_csv(joinpath(folder,name*"_coefficients.csv"),
        vcat("basis_index","order_".*string.(collect(MillingModel.PARAMETER_NAMES)),
             "t".*string.(eachindex(pce.problem.times))),
        ([i,pce.multi_indices[i,:]...,pce.coefficients[i,:]...] for i in axes(pce.coefficients,1)))
    write_csv(joinpath(folder,name*"_summary.csv"),
        ["time_s","mean_um","sd_um","q025_um","median_um","q975_um","gaussian_low_um","gaussian_high_um","surrogate_mean_mcse_um"],
        summary_rows(cheap,pce))
    write_json(joinpath(folder,name*"_diagnostics.json"),diagnostics(pce))
    write_csv(joinpath(folder,name*"_surrogate_checks.csv"),
        ["time_s","minimum_um","maximum_um","negative_fraction"],
        [(t,minimum(cheap.values[i,:]),maximum(cheap.values[i,:]),
            count(<(0),cheap.values[i,:])/length(cheap.ids)) for (i,t) in enumerate(pce.problem.times)])
    v=pce.validation
    write_csv(joinpath(folder,name*"_validation.csv"),
        ["time_s","rmse_um","rmse_over_output_sd","max_abs_error_um","signed_mean_error_um"],
        [(t,v.rmse[i],v.normalized_rmse[i],v.maximum_absolute_error[i],v.signed_mean_error[i]) for (i,t) in enumerate(pce.problem.times)])
    write_csv(joinpath(folder,name*"_common_inputs_error.csv"),
        ["time_s","rmse_um","signed_mean_error_um"],
        [(t,sqrt(sum(abs2,common.values[i,:]-reference.values[i,:])/size(common.values,2)),
            sum(common.values[i,:]-reference.values[i,:])/size(common.values,2)) for (i,t) in enumerate(pce.problem.times)])
end

function source_identity(settings)
    paths=String[]
    for directory in ("src","ext")
        for (root,_,files) in walkdir(joinpath(ROOT,directory)), file in sort(files)
            endswith(file,".jl") && push!(paths,relpath(joinpath(root,file),ROOT))
        end
    end
    append!(paths,["Project.toml","examples/environments/milling/Project.toml","examples/milling_uncertainty.jl",
        "examples/support/milling_model.jl","examples/support/milling_solver.jl",
        "examples/data/milling_uncertainty/reconstruction.toml"])
    hashes=Dict(path=>bytes2hex(sha256(read(joinpath(ROOT,path)))) for path in sort!(paths))
    key=bytes2hex(sha256(JSON3.write((settings,hashes))))
    (;key,hashes)
end

"""Independent affine matrix exponential, evaluated separately on both speed
segments. It verifies time integration of this semi-discrete model, not the
accuracy of the fixed-pivot approximation to the continuum PBM.
"""
function exponential_reference(pb,times)
    u=copy(pb.u0);lasttime=0.;out=Vector{typeof(u)}()
    # Independently exponentiate in kilograms per pivot. Directly exponentiating
    # the mixed scaled-number/kg system is badly conditioned by unit factors.
    units=vcat(pb.p.rho*pb.p.number_scale.*pb.p.pivots,1.,1.)
    for t in times
        stops=lasttime<pb.p.change_time<t ? [pb.p.change_time,t] : [t]
        for stop in stops
            rpm=lasttime<pb.p.change_time ? pb.p.speed_before : pb.p.speed_after
            localpb=remake(pb;u0=pb.u0,p=merge(pb.p,(segment_speed=rpm,)))
            A,b=MillingModel.linear_system(localpb)
            mass_operator=(A.*units)./units'
            augmented=[mass_operator units.*b;zeros(1,length(b)+1)]
            mass=(exp((stop-lasttime)*augmented)*vcat(units.*u,1.))[1:end-1]
            u=mass./units
            lasttime=stop
        end
        push!(out,copy(u))
    end
    PBMSolution(collect(times),out,pb,0)
end

function numerical_checks(folder;smoke=false)
    ts=[.01,5.,29.999,30.,30.001,31.,100.,200.]
    configurations=[("base",100,1e-9,1e-14,8,1e-3),
        ("time",100,1e-10,1e-15,8,1e-3),
        ("quadrature",100,1e-9,1e-14,16,1e-3),
        ("size200",200,1e-9,1e-14,8,1e-3)]
    if !smoke
        push!(configurations,("size400",400,1e-9,1e-14,8,1e-3))
        ratio=(1e-3/MillingModel.parameters().first_pivot)^(1/99)
        push!(configurations,("domain",104,1e-9,1e-14,8,1e-3*ratio^4))
    end
    rows=Tuple[];reference=nothing;refmedian=nothing
    for (name,bins,rt,at,order,last) in configurations
        pb=MillingModel.build(;bins,last_pivot=last,quadrature_order=order)
        started=time_ns();sol=MillingSolver.solve(pb,ts;reltol=rt,abstol=at)
        elapsed=(time_ns()-started)/1e9
        medians=[only(observe(MillingModel.median_observable(),sol;index=i)) for i in eachindex(ts)]
        name=="base" && (reference=sol;refmedian=medians)
        d=state_diagnostics(pb,sol)
        for (i,t) in enumerate(ts)
            push!(rows,(name,bins,order,rt,at,t,medians[i],medians[i]-refmedian[i],
                d.maximum_mass_error_kg,d.minimum_inventory,sol.steps,elapsed))
        end
    end
    exact=exponential_reference(reference.problem,ts)
    differences=[maximum(abs,reference.u[i]-exact.u[i]) for i in eachindex(ts)]
    median_difference=[only(observe(MillingModel.median_observable(),reference;index=i))-
        only(observe(MillingModel.median_observable(),exact;index=i)) for i in eachindex(ts)]
    write_csv(joinpath(folder,"numerical_refinement.csv"),
        ["check","bins","daughter_quadrature","reltol","abstol","time_s","d50_um","difference_from_base_um",
         "max_material_error_kg","minimum_scaled_inventory","steps","solve_seconds"],rows)
    write_csv(joinpath(folder,"matrix_exponential.csv"),
        ["time_s","maximum_absolute_state_difference","d50_difference_um"],zip(ts,differences,median_difference))
    maximum(abs,median_difference)<1e-3 || error("BDF median is not resolved against matrix exponential")
    rows
end

function run(;full=false,force=false)
    config=TOML.parsefile(CONFIG)
    settings=merge(config,Dict("mode"=>full ? "production" : "smoke",
        "mc_samples"=>full ? config["mc_samples"] : 64,
        "validation_samples"=>full ? config["validation_samples"] : 32,
        "pce_quantile_samples"=>full ? config["pce_quantile_samples"] : 2048))
    identity=source_identity(settings)
    folder=full ? joinpath(ASSETS,"data") : joinpath(ROOT,".cache","milling_uncertainty","smoke")
    mkpath(folder)
    receipt=joinpath(folder,"provenance.json")
    if isfile(receipt) && !force
        saved=JSON3.read(read(receipt,String))
        saved.configuration_key==identity.key && saved.complete && return println("Reusing matching completed milling results: ",folder)
        error("existing milling outputs have different/incomplete provenance; use --force explicitly to replace this case's results")
    end
    write_json(joinpath(folder,"settings.json"),(;configuration_key=identity.key,settings,
        source_hashes=identity.hashes,git_revision=strip(read(`git -C $ROOT rev-parse HEAD`,String)),
        initial_population=:empty,initial_outlet_median=:undefined,
        output_start_s=first(output_times()),solver_unknowns=:per_pivot_kg,
        full_forward_solve=:two_constant_speed_BDF_segments))
    println("Milling ",settings["mode"],": ",settings["mc_samples"]," MC realizations; independent validation ",settings["validation_samples"])
    flush(stdout)
    # Warmup is separate and uses the actual residual, boundary, observation,
    # regression and surrogate types. Every warmup solve is counted separately.
    pb=MillingModel.build();warm_started=time_ns()
    MillingSolver.solve(pb,[.01,30.,31.,200.])
    box=hcat(fill(-2.,4),fill(2.,4))
    warmuq=problem()
    d1=pce_design(warmuq.inputs;n=12,seed=config["pce1_seed"],bounds=box)
    axial=hcat(zeros(4),sqrt(3.)*Matrix{Float64}(I,4,4),-sqrt(3.)*Matrix{Float64}(I,4,4))
    extra=pce_design(warmuq.inputs;n=23,seed=config["pce2_seed"],bounds=box)
    d2=merge(extra,(samples=hcat(axial,extra.samples),method=:axial_augmented_lhs))
    warmvalidation=propagate(warmuq,MonteCarlo(2;seed=18);max_solves=2)
    for (degree,design) in ((1,d1),(2,d2),(1,axial))
        fit=fit_pce(warmuq,PolynomialChaos(degree);design,validation=warmvalidation,
            max_solves=size(design isa NamedTuple ? design.samples : design,2))
        uq_summary(evaluate_pce(fit,MonteCarlo(8));covariance=false)
    end
    warmup_seconds=(time_ns()-warm_started)/1e9
    numerical_checks(folder;smoke=!full)
    records=NamedTuple[]
    uq=problem(;records,progress=500)
    ts=uq.times
    write_csv(joinpath(folder,"times.csv"),["time_s"],((t,) for t in ts))
    write_csv(joinpath(folder,"grid.csv"),["edge_index","volume_m3"],enumerate(pb.p.edges))
    tick=time_ns();nominal=MillingSolver.solve(pb,ts);nominal_seconds=(time_ns()-tick)/1e9
    write_csv(joinpath(folder,"nominal.csv"),["time_s","d50_um","mill_mass_kg","discharged_mass_kg","fines_mass_kg","material_error_kg"],
        ((t,only(observe(MillingModel.median_observable(),nominal;index=i)),
          pb.p.rho*pb.p.number_scale*dot(pb.p.pivots,population(pb,u,:mill)),bulk(pb,u,:discharged_mass),
          bulk(pb,u,:fines_mass),pb.p.rho*pb.p.number_scale*dot(pb.p.pivots,population(pb,u,:mill))+
          bulk(pb,u,:discharged_mass)-pb.p.mass_feed*t) for (i,(t,u)) in enumerate(zip(ts,nominal.u))))
    mc=propagate(uq,MonteCarlo(settings["mc_samples"];seed=config["mc_seed"]);max_solves=settings["mc_samples"])
    export_ensemble(folder,"mc",mc)
    all(mc.success) || error("MC contains failures; exports retained, no success-conditioned production figure")
    write_csv(joinpath(folder,"mc_summary.csv"),
        ["time_s","mean_um","sd_um","q025_um","median_um","q975_um","gaussian_low_um","gaussian_high_um","mean_mcse_um"],summary_rows(mc))
    validation=propagate(uq,MonteCarlo(settings["validation_samples"];seed=config["validation_seed"]);max_solves=settings["validation_samples"])
    export_ensemble(folder,"validation",validation)
    all(validation.success) || error("independent validation contains failures")
    surrogates=Dict{String,Any}();costs=NamedTuple[]
    push!(costs,(method="MC",full_solves=mc.metadata.solves,training_points=0,basis_terms=0,
        setup_seconds=mc.metadata.setup_seconds,solve_seconds=mc.metadata.solve_seconds,
        regression_seconds=0.,validation_seconds=0.,surrogate_seconds=0.,maximum_validation_rmse_um=0.))
    for (name,degree,design) in (("pce1",1,d1),("pce2",2,d2),("pce1_nine",1,axial))
        pce=fit_pce(uq,PolynomialChaos(degree);design,validation,max_solves=size(design isa NamedTuple ? design.samples : design,2))
        surrogates[name]=pce
        started=time_ns()
        cheap=evaluate_pce(pce,MonteCarlo(settings["pce_quantile_samples"];seed=config["surrogate_seed"]))
        elapsed=(time_ns()-started)/1e9
        common=evaluate_pce(pce,mc.inputs)
        export_pce(folder,name,pce,cheap,common,mc)
        if name=="pce1"
            write_csv(joinpath(folder,"pce_propagation_inputs.csv.gz"),
                vcat("sample_id","z_".*string.(collect(MillingModel.PARAMETER_NAMES))),
                ([cheap.ids[i],values(cheap.inputs[i])...] for i in eachindex(cheap.ids)))
        end
        tr=pce.training.metadata
        push!(costs,(method=name,full_solves=tr.solves,training_points=length(pce.training.ids),basis_terms=size(pce.coefficients,1),
            setup_seconds=tr.setup_seconds,solve_seconds=tr.solve_seconds,
            regression_seconds=pce.report.preparation_seconds+pce.report.regression_seconds,
            validation_seconds=validation.metadata.solve_seconds,surrogate_seconds=elapsed,
            maximum_validation_rmse_um=maximum(pce.validation.rmse)))
    end
    write_csv(joinpath(folder,"costs.csv"),string.(collect(keys(first(costs)))),values.(costs))
    roles=vcat(fill("MC",length(mc.ids)),fill("validation",length(validation.ids)),
        fill("pce1",12),fill("pce2",32),fill("pce1_nine",9))
    write_csv(joinpath(folder,"solve_checks.csv"),vcat("full_solve_id","role",string.(collect(keys(first(records))))),
        ([i,roles[i],values(record)...] for (i,record) in enumerate(records)))
    j=only(findall(==(31.),ts));convergence=Tuple[]
    for n in filter(<=(length(mc.ids)),full ? [100,300,1000,3000,6000,12000] : [16,32,64])
        y=mc.values[j,1:n];mu=sum(y)/n;sd=sqrt(sum(abs2,y.-mu)/(n-1))
        push!(convergence,(n,mu,sd,sd/sqrt(n)))
    end
    write_csv(joinpath(folder,"mc_convergence_31s.csv"),["samples","mean_um","sd_um","mean_mcse_um"],convergence)
    write_csv(joinpath(folder,"published_comparison.csv"),["quantity","publication_um","reconstruction_um","difference_um"],
        [("mean_31s",208.366,sum(mc.values[j,:])/length(mc.ids),sum(mc.values[j,:])/length(mc.ids)-208.366),
         ("sd_31s",41.7409,uq_summary(mc;covariance=false).variance.d50[1,j]^0.5,uq_summary(mc;covariance=false).variance.d50[1,j]^0.5-41.7409)])
    accuracy=Tuple[]
    mcs=uq_summary(mc;covariance=false)
    push!(accuracy,("MC",length(mc.ids),mcs.mean.d50[1,j],sqrt(mcs.variance.d50[1,j]),NaN,mc.metadata.solve_seconds))
    for name in ("pce1","pce2","pce1_nine")
        fit=surrogates[name];s=uq_summary(fit;covariance=false)
        push!(accuracy,(name,length(fit.training.ids),s.mean.d50[1,j],sqrt(s.variance.d50[1,j]),
            fit.validation.rmse[j],fit.training.metadata.solve_seconds))
    end
    write_csv(joinpath(folder,"accuracy_31s.csv"),
        ["method","production_solves","mean_um","sd_um","validation_rmse_um","integration_seconds"],accuracy)
    open(joinpath(folder,"accuracy.md"),"w") do io
        println(io,"| Method | Production solves | Mean at 31 s (µm) | SD (µm) | Validation RMSE (µm) | Integration (s) |")
        println(io,"|:--|--:|--:|--:|--:|--:|")
        for (name,n,mean,sd,error,seconds) in accuracy
            @printf(io,"| %s | %d | %.2f | %.2f | %s | %.2f |\n",name,n,mean,sd,isnan(error) ? "—" : @sprintf("%.2f",error),seconds)
        end
    end
    cp(CONFIG,joinpath(folder,"reconstruction.toml");force=true)
    packages=Dict(string(info.name)=>string(info.version) for (_,info) in Pkg.dependencies() if info.is_direct_dep)
    files=Dict(name=>bytes2hex(sha256(read(joinpath(folder,name)))) for name in readdir(folder) if name!="provenance.json")
    status=read(`git -C $ROOT status --porcelain`,String)
    write_json(receipt,(configuration_key=identity.key,complete=true,mode=settings["mode"],settings,
        source_hashes=identity.hashes,git_revision=strip(read(`git -C $ROOT rev-parse HEAD`,String)),dirty=!isempty(status),
        working_tree=status,julia=string(VERSION),packages,cpu=Sys.CPU_NAME,cpu_model=Sys.cpu_info()[1].model,
        julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),warmup_seconds,warmup_full_solves=56,nominal_seconds,
        full_solves=length(records)+1,nominal_steps=nominal.steps,
        invalid_gaussian_probability_upper_bound=Distributions.cdf(Distributions.Normal(.44,.07),0.)+
            Distributions.ccdf(Distributions.Normal(.44,.07),1.)+
            Distributions.cdf(Distributions.Normal(8.82e-6,1.01e-6),0.)+
            Distributions.cdf(Distributions.Normal(2.1,.12),0.),
        literature_pdf_hash=isfile(joinpath(ROOT,"local_papers","Bhonsale2018.pdf")) ? bytes2hex(sha256(read(joinpath(ROOT,"local_papers","Bhonsale2018.pdf")))) : "not_available",
        numerical_files=files,interval=:pointwise_output_quantiles,gaussian_reference_multiplier=1.96,
        source_band_degrees_of_freedom=:unrecovered))
    println("Completed ",settings["mode"]," results: ",folder)
    folder
end

if abspath(PROGRAM_FILE)==@__FILE__
    all(x->x in ("--full","--smoke","--force"),ARGS) || error("usage: --smoke or --full [--force]")
    "--full" in ARGS && "--smoke" in ARGS && error("choose one mode")
    run(;full="--full" in ARGS,force="--force" in ARGS)
end
end
