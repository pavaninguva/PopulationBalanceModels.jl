# Reporting and supporting studies; the physical PBM remains in the example.
const Case=IsothermalProteinCrystallization
const data=load_data()
const study_configuration=JSON3.read(read(joinpath(Case.DATA,"config","case.json"),String))

function refinement(fitresult; bins=fitresult.settings.bins,dt=fitresult.settings.dt,
        Lmax=fitresult.settings.Lmax,points=fitresult.settings.points,cfl=fitresult.settings.cfl)
    s=fitresult.settings
    cal=calibration(;bins,dt,Lmax,points,cfl,cstar=s.cstar,window=s.window,sc=s.sc,
        weight=s.weight,gated=s.gated)
    a=predictions(fitresult.cal,fitresult.u)[1]
    b=predictions(cal,fitresult.u)[1]
    dc=maximum(abs.(a[1]-b[1]))/s.sc
    dq=maximum(maximum(abs.(a[i]-b[i]))/.5 for i in 2:4)
    dq_rms=maximum(sqrt(sum(abs2,a[i]-b[i])/length(a[i]))/.5 for i in 2:4)
    (;bins,dt,cfl,Lmax,points,concentration_scaled_change=dc,quantile_scaled_change=dq,
        quantile_RMS_scaled_change=dq_rms,
        loss=objective(cal,fitresult.u),material=max(dc,dq_rms)>.25)
end

# Select/refine using calibration only. Nothing below this block has read a
# validation value to choose a model, parameter set or numerical setting.
spatial=refinement(chosen;bins=2chosen.settings.bins,dt=chosen.settings.dt/2)
if spatial.material
    chosen=fit_case(OUTPUT;tag="selected_spatial_refit",bins=2chosen.settings.bins,
        gated=selected_gate,initial=chosen.theta,nstarts=3)
end
spatial_final=refinement(chosen;bins=2chosen.settings.bins,dt=chosen.settings.dt/2)
if spatial_final.material && chosen.settings.bins<400
    chosen=fit_case(OUTPUT;tag="selected_further_spatial_refit",bins=2chosen.settings.bins,
        gated=selected_gate,initial=chosen.theta,nstarts=3)
    spatial_final=refinement(chosen;bins=2chosen.settings.bins,dt=chosen.settings.dt/2)
end
temporal=refinement(chosen;dt=chosen.settings.dt/2,cfl=chosen.settings.cfl/2)
const report_signature=bytes2hex(sha256(read(@__FILE__,String)*basename(chosen.folder)*JSON3.write(chosen.u)))[1:12]
domain=refinement(chosen;Lmax=75.,bins=round(Int,1.5chosen.settings.bins))
quadrature=[(;points=n,loss=objective(calibration(bins=chosen.settings.bins,
    dt=chosen.settings.dt,gated=selected_gate,points=n),chosen.u)) for n in (256,512,1024)]
fixed_dt=.35*(chosen.settings.Lmax/chosen.settings.bins)/2
fixed_cal=CalibrationProblem(chosen.cal.experiments;parameters=chosen.cal.parameters,
    solver=(pb,ts)->simulate(pb;dt=fixed_dt,saveat=ts,method=SSPRK33()))
fixed_outputs=predictions(fixed_cal,chosen.u)[1]
native_outputs=predictions(chosen.cal,chosen.u)[1]
fixed_schedule_check=(;dt=fixed_dt,loss=objective(fixed_cal,chosen.u),
    concentration_max_difference_g_L=maximum(abs.(fixed_outputs[1]-native_outputs[1])),
    quantile_RMS_difference_um=maximum(sqrt(sum(abs2,fixed_outputs[i]-native_outputs[i])/length(fixed_outputs[i])) for i in 2:4))
json(joinpath(OUTPUT,"frozen_model.json"),(;frozen_at=string(now(UTC)),
    family=selected_gate ? "logistic_activation" : "baseline_power_laws",theta=chosen.theta,
    fit_cache=relpath(chosen.folder,ROOT),baseline_cache=relpath(baseline.folder,ROOT),
    candidate_cache=relpath(gated.folder,ROOT),
    settings=chosen.settings,calibration_loss=chosen.best.loss,
    calibration_groups=(chosen.best.concentration_loss,chosen.best.distribution_loss),
    selection_file="calibration_selection.json",validation_used=false,
    numerical_checks=(;spatial,spatial_final,temporal,domain,quadrature,fixed_schedule_check),START_STATE))

function evaluate(fitresult;export_tables=false,observations=data)
    s=fitresult.settings
    p=Case.decode(fitresult.u;cstar=s.cstar,gated=s.gated)
    cr=observations.concentration
    psds=vcat(select_psds(observations;window=s.window),select_psds(observations,:validation;window=s.window))
    ts=sort(unique(vcat(collect(0.:5.:660.),[r.model_time_min for r in cr],
                        [first(r).time_min for r in psds])))
    # Compilation warmup then one timed solve. Excludes export and observation work.
    warm_problem=build(p=p,bins=s.bins,Lmax=s.Lmax)
    cfl_solve(warm_problem,[0.,1.];maximum_dt=s.dt,cfl=s.cfl)
    setup=@timed build(p=p,bins=s.bins,Lmax=s.Lmax)
    elapsed=@timed cfl_solve(setup.value,ts;maximum_dt=s.dt,cfl=s.cfl)
    sol=elapsed.value; pb=sol.problem; grid=pb.operator.system.populations.crystals.grid
    index(t)=searchsortedfirst(ts,t)
    concentration=[(;observation_id=r.observation_id,time_min=r.model_time_min,split=r.split,
        observed_g_L=r.concentration_g_L,predicted_g_L=bulk(pb,sol.u[index(r.model_time_min)],:c),
        residual_g_L=bulk(pb,sol.u[index(r.model_time_min)],:c)-r.concentration_g_L) for r in cr]
    psdrows=[];quantiles=[];curves=[];w2rows=[]
    window=(first(first(psds)).normalization_min_um,first(first(psds)).normalization_max_um)
    for rows in psds
        r=first(rows); t=r.time_min; N=population(pb,sol.u[index(t)],:crystals)
        x=[a.size_um for a in rows]; F=[a.conditional_cdf for a in rows]
        Fc=conditional_cdf(grid,N,x;window)
        total=sum(N[i]*max(0.,min(window[2],grid.edges[i+1])-max(window[1],grid.edges[i]))/grid.widths[i] for i in eachindex(N))
        for (k,a) in enumerate(rows)
            cell=clamp(searchsortedlast(grid.edges,a.size_um),1,length(grid))
            push!(psdrows,(;time_min=t,split=r.split,window=r.window,size_um=a.size_um,
                observed_cdf=a.conditional_cdf,predicted_cdf=Fc[k],
                observed_density_per_um=a.conditional_density_per_um,
                predicted_density_per_um=N[cell]/grid.widths[cell]/total))
        end
        for i in eachindex(N)
            push!(curves,(;time_min=t,split=r.split,left_um=grid.edges[i],right_um=grid.edges[i+1],
                conditional_density_per_um=grid.edges[i+1]<=window[1] || grid.edges[i]>=window[2] ? 0. : N[i]/grid.widths[i]/total,
                physical_density_m3_um=N[i]*1e12/grid.widths[i]))
        end
        observed=cdf_quantiles(x,F,[.1,.5,.9])
        if observations===data
            summary=only(filter(a->a.time_min==t && a.window==r.window,data.summary))
            supplied=[summary.D10_um,summary.D50_um,summary.D90_um]
            maximum(abs.(supplied-observed))<1e-8 || error("summary quantiles disagree with the active CDF")
            observed=supplied # diagnostic only; never another calibration block
        end
        predicted=conditional_quantiles(grid,N,[.1,.5,.9];window)
        for (u,qo,qp) in zip((.1,.5,.9),observed,predicted)
            push!(quantiles,(;time_min=t,split=r.split,window=r.window,probability=u,
                observed_um=qo,predicted_um=qp,error_um=qp-qo))
        end
        u=((1:1024).-0.5)./1024
        w2= sqrt(sum(abs2,conditional_quantiles(grid,N,u;window)-cdf_quantiles(x,F,u))/1024)
        push!(w2rows,(;time_min=t,split=r.split,W2_um=w2,quantile_points=1024))
    end
    mass=[]; trajectory=[]
    for (t,state) in zip(ts,sol.u)
        N=population(pb,state,:crystals);c=bulk(pb,state,:c)
        escaped=cumulative(pb,state,:crystals,:upper_escape,:material)*1e-6*p.rho*p.kv
        solid=1e-6*p.rho*p.kv*moment(grid,N,3)
        all(x->isfinite(x) && x>=0,N) || throw(NumericalFailure("invalid population in reported trajectory"))
        measured_number=sum(N[i]*max(0.,min(window[2],grid.edges[i+1])-max(window[1],grid.edges[i]))/grid.widths[i] for i in eachindex(N))
        qs=measured_number==0 ? [NaN,NaN,NaN] : conditional_quantiles(grid,N,[.1,.5,.9];window)
        J,G=rates(p,t,c)
        push!(trajectory,(;time_min=t,concentration_g_L=c,D10_um=qs[1],D50_um=qs[2],D90_um=qs[3],
            J_m3_min=J,G_um_min=G,total_number_m3=1e12*sum(N)))
        born=sum(N)+cumulative(pb,state,:crystals,:upper_escape)
        push!(mass,(;time_min=t,solute_g_L=c,solid_g_L=solid,escaped_g_L=escaped,
            material_error_g_L=c+solid+escaped-p.c0,minimum_scaled_count=minimum(N),
            last_bin_number_m3=N[end]*1e12,
            last_bin_material_g_L=N[end]*1e-6*p.rho*p.kv*last(grid.centers)^3,
            cumulative_first_bin_birth_material_g_L=born*1e-6*p.rho*p.kv*first(grid.centers)^3))
    end
    calr=filter(r->r.split=="calibration",concentration)
    valr=filter(r->r.split=="validation",concentration)
    rmse(rs)=sqrt(sum(r.residual_g_L^2 for r in rs)/length(rs))
    metrics=(;evaluated_at=string(now(UTC)),W2_reporting_quadrature_points=1024,
        calibration_concentration_RMSE_g_L=rmse(calr),
        validation_concentration_RMSE_g_L=rmse(valr),
        validation_concentration_MAE_g_L=sum(abs(r.residual_g_L) for r in valr)/length(valr),
        calibration_W2_um=[r.W2_um for r in w2rows if r.split=="calibration"],
        validation_W2_um=only(r.W2_um for r in w2rows if r.split=="validation"),
        validation_quantile_errors_um=[r.error_um for r in quantiles if r.split=="validation"],
        max_material_error_g_L=maximum(abs(r.material_error_g_L) for r in mass),
        max_escaped_g_L=maximum(r.escaped_g_L for r in mass),
        minimum_scaled_count=minimum(r.minimum_scaled_count for r in mass),
        max_last_bin_material_g_L=maximum(r.last_bin_material_g_L for r in mass),
        first_bin_birth_material_g_L=last(mass).cumulative_first_bin_birth_material_g_L,
        steps=sol.steps,setup_seconds=setup.time,solve_seconds=elapsed.time,solve_allocated_bytes=elapsed.bytes)
    if export_tables
        csv(joinpath(OUTPUT,"concentration_predictions.csv"),concentration)
        csv(joinpath(OUTPUT,"psd_predictions.csv"),psdrows)
        csv(joinpath(OUTPUT,"distribution_curves.csv"),curves)
        csv(joinpath(OUTPUT,"quantile_predictions.csv"),quantiles)
        csv(joinpath(OUTPUT,"trajectory.csv"),trajectory)
        csv(joinpath(OUTPUT,"material_balance.csv"),mass)
        csv(joinpath(OUTPUT,"distribution_metrics.csv"),w2rows)
    end
    metrics
end

metrics=evaluate(chosen;export_tables=true)
include("moment_check.jl")
moment_check=moment_reference_check(chosen)
json(joinpath(OUTPUT,"validation_metrics.json"),metrics)
json(joinpath(OUTPUT,"conservation_refinement.json"),(;metrics,moment_check,spatial,spatial_final,temporal,domain,quadrature,fixed_schedule_check,
    refitted=spatial.material || baseline_change>.25,remaining_refinement_warning=spatial_final.material,
    invariant="c + 1e-6*rho*kv*(M3_scaled + escaped_M3_scaled) = c0",
    first_bin_note="Boundary births acquire the first pivot's discrete mass via the same conservative exchange; vanishes on refinement"))

labels=["log_J_ref","log_G_ref","j","g","t_on"]
units=["log(particles m^-3 min^-1)","log(um min^-1)","1","1","min"]
function bound_hits(theta)
    lower,upper=length(theta)==5 ? (vcat(Case.BOUNDS[1],0.),vcat(Case.BOUNDS[2],150.)) : Case.BOUNDS
    join([labels[i] for i in eachindex(theta) if min(theta[i]-lower[i],upper[i]-theta[i])<.001*(upper[i]-lower[i])],';')
end
lo,hi=selected_gate ? (vcat(Case.BOUNDS[1],0.),vcat(Case.BOUNDS[2],150.)) : Case.BOUNDS
parameter_rows=[(;parameter=labels[i],value=x,unit=units[i],
    lower_bound=lo[i],upper_bound=hi[i],near_bound=min(x-lo[i],hi[i]-x)<.001*(hi[i]-lo[i]),
    optimizer_converged=chosen.best.converged) for (i,x) in enumerate(chosen.theta)]
for (i,label,unit) in ((1,"J_ref","particles m^-3 min^-1"),(2,"G_ref","um min^-1"))
    push!(parameter_rows,(;parameter=label,value=exp(chosen.theta[i]),unit,
        lower_bound=exp(lo[i]),upper_bound=exp(hi[i]),near_bound=parameter_rows[i].near_bound,
        optimizer_converged=chosen.best.converged))
end
csv(joinpath(OUTPUT,"fitted_parameters.csv"),parameter_rows)
csv(joinpath(OUTPUT,"multistart_summary.csv"),[(;family=name,start=r.start,
    initial_theta=join(r.initial_theta,';'),theta=join(r.theta,';'),loss=r.loss,
    concentration_loss=r.concentration_loss,distribution_loss=r.distribution_loss,
    converged=r.converged,iterations=r.iterations,invalid_trials=r.invalid_trials,seconds=r.seconds,
    bound_hits=bound_hits(r.theta))
    for (name,f) in (("baseline100",baseline100),("baseline",baseline),("onset_candidate",gated),("selected",chosen)) for r in f.rows])
csv(joinpath(OUTPUT,"fit_contributions.csv"),loss_contributions(chosen.cal,chosen.u))
ad=sensitivity_report(chosen.cal,chosen.u)
fd=sensitivity_report(chosen.cal,chosen.u;derivatives=:finite_difference,relative_step=1e-5)
res=residuals(chosen.cal,chosen.u)
gradient=2ad.jacobian'*res
json(joinpath(OUTPUT,"optimizer_diagnostics.json"),(;converged=chosen.best.converged,
    gradient=collect(gradient),gradient_norm=norm(gradient),rank=ad.rank,
    singular_values=ad.singular_values,condition_number=ad.condition_number,
    AD_central_difference_relative_error=norm(ad.jacobian-fd.jacobian)/max(norm(ad.jacobian),eps()),
    interpretation="local geometry of the chosen comparison loss, not Fisher information or confidence intervals"))

# All sensitivity scenarios are predeclared and use calibration only. Their
# validation metrics are reported after fitting, never used for selection.
scenarios=[("cstar_1p0",(cstar=1.,)),("cstar_3p5",(cstar=3.5,)),
    ("cstar_4p5",(cstar=4.5,)),("psd_weight_0p25",(weight=.25,)),
    ("psd_weight_4p0",(weight=4.,)),("concentration_scale_0p3",(sc=.3,)),
    ("wider_window",(window="visible_window",))]
sensitivity=[]
for (name,change) in scenarios
    settings=merge((bins=chosen.settings.bins,gated=selected_gate,initial=chosen.theta,
        nstarts=3,tag=name),change)
    f=fit_case(OUTPUT;settings...)
    m=evaluate(f)
    push!(sensitivity,(;scenario=name,theta=join(f.theta,';'),cstar=f.settings.cstar,
        window=f.settings.window,weight=f.settings.weight,concentration_scale=f.settings.sc,
        calibration_loss=f.best.loss,concentration_loss=f.best.concentration_loss,
        distribution_loss=f.best.distribution_loss,converged=f.best.converged,
        calibration_RMSE_g_L=m.calibration_concentration_RMSE_g_L,
        calibration_W2_um=join(m.calibration_W2_um,';'),
        validation_RMSE_g_L=m.validation_concentration_RMSE_g_L,validation_W2_um=m.validation_W2_um))
    csv(joinpath(OUTPUT,"sensitivities.csv"),sensitivity)
end

# The extra baseline-ordinate check was supplied with the handover. Keep its
# reconstructed observations separate; never overwrite the authoritative CSV.
tailrows=copy(data.psd)
for time in (74.,180.,420.,660.)
    ids=findall(r->r.time_min==time && r.window=="isothermal_core",tailrows)
    sort!(ids;by=i->tailrows[i].size_um)
    x=[tailrows[i].size_um for i in ids]
    y=[tailrows[i].plotted_relative_frequency+.005 for i in ids]
    cumulative=vcat(0.,cumsum(diff(x).*(y[1:end-1]+y[2:end])./2))
    area=last(cumulative);F=cumulative./area
    for (k,i) in enumerate(ids)
        tailrows[i]=merge(tailrows[i],(conditional_cdf=F[k],conditional_density_per_um=y[k]/area))
    end
end
taildata=merge(data,(psd=tailrows,))
tailcal=calibration(data=taildata,bins=chosen.settings.bins,gated=selected_gate)
tailfits=[]
for (k,start) in enumerate((chosen.u,chosen.u .+ .15,chosen.u .- .15))
    recordfile=joinpath(OUTPUT,"fits","ordinate_offset_$(report_signature)_start_$(k).json")
    if isfile(recordfile)
        saved=JSON3.read(read(recordfile,String))
        basename(saved.reference_folder)==basename(chosen.folder) || error("ordinate sensitivity cache belongs to another reference fit")
        push!(tailfits,saved)
        continue
    end
    println("Fit ordinate offset +0.005, start ",k,"/3");flush(stdout)
    elapsed=@timed fit(tailcal,start;maxiters=400,
        algorithm=Optim.BFGS(linesearch=Optim.LineSearches.BackTracking()),
        options=(g_abstol=1e-4,f_reltol=1e-10,))
    result=elapsed.value
    saved=(;u=result.minimizer,loss=result.minimum,converged=result.converged,
        seconds=elapsed.time,reference_folder=chosen.folder,raw_ordinate_offset=.005)
    json(recordfile,saved);push!(tailfits,saved)
end
tailbest=tailfits[argmin([r.loss for r in tailfits])]
tailu=collect(tailbest.u)
tailfit=(;settings=chosen.settings,u=tailu)
tailmetrics=evaluate(tailfit;observations=taildata)
tailcontributions=loss_contributions(tailcal,tailu)
push!(sensitivity,(;scenario="ordinate_baseline_plus_0p005",theta=join(Case.physical_coordinates(tailu;gated=selected_gate),';'),
    cstar=2.16,window="isothermal_core",weight=1.,concentration_scale=.1,
    calibration_loss=tailbest.loss,concentration_loss=first(tailcontributions).value,
    distribution_loss=sum(r.value for r in tailcontributions[2:end]),converged=tailbest.converged,
    calibration_RMSE_g_L=tailmetrics.calibration_concentration_RMSE_g_L,
    calibration_W2_um=join(tailmetrics.calibration_W2_um,';'),
    validation_RMSE_g_L=tailmetrics.validation_concentration_RMSE_g_L,
    validation_W2_um=tailmetrics.validation_W2_um))
csv(joinpath(OUTPUT,"sensitivities.csv"),sensitivity)

# Conditional comparison-loss profiles: all remaining coordinates are refitted.
# They are not confidence bounds or likelihood-ratio tests.
profiles=[]
for k in (3,4), delta in (-1.,-.5,0.,.5,1.)
    value=chosen.theta[k]+delta
    lo[k]<value<hi[k] || continue
    println("Conditional loss profile ",labels[k]," = ",value);flush(stdout)
    profile_path=joinpath(OUTPUT,"fits","profile_$(report_signature)_$(k)_$(delta).json")
    if isfile(profile_path)
        row=JSON3.read(read(profile_path,String))
        push!(profiles,row)
        csv(joinpath(OUTPUT,"rate_tradeoffs.csv"),profiles)
        continue
    end
    indices=setdiff(1:length(chosen.u),[k]);fixed=copy(chosen.theta);fixed[k]=value
    fixed_u=Case.coordinates(fixed;gated=selected_gate)[k]
    function expand(v)
        u=eltype(v).(chosen.u);u[k]=fixed_u;u[indices]=v;u
    end
    cal=CalibrationProblem(chosen.cal.experiments;
        parameters=(v,e)->chosen.cal.parameters(expand(v),e),solver=chosen.cal.solver)
    result=fit(cal,chosen.u[indices];maxiters=200,
        algorithm=Optim.BFGS(linesearch=Optim.LineSearches.BackTracking()),
        options=(g_abstol=1e-4,f_reltol=1e-10,))
    row=(;parameter=labels[k],fixed_value=value,loss=result.minimum,
        delta_loss=result.minimum-chosen.best.loss,converged=result.converged,
        bound_hits=bound_hits(Case.physical_coordinates(expand(result.minimizer);gated=selected_gate)),
        theta=join(Case.physical_coordinates(expand(result.minimizer);gated=selected_gate),';'))
    json(profile_path,row);push!(profiles,row)
    csv(joinpath(OUTPUT,"rate_tradeoffs.csv"),profiles)
end

versions=Dict(v.name=>string(v.version) for v in values(Pkg.dependencies()) if v.version!==nothing)
settings=(;mode=MODE,completed=true,model=selected_gate ? "logistic_activation" : "baseline_power_laws",
    fit_cache=relpath(chosen.folder,ROOT),
    experiment=study_configuration.experiment,size_scale_um=.5,
    theta_names=labels[1:length(chosen.theta)],
    optimizer_coordinates="unconstrained box logits of the declared theta coordinates; no prior or density Jacobian",
    selected=chosen.settings,theta=chosen.theta,c0=Case.C0,rho_kg_m3=1240.,kv=.81,
    cstar_status="nominal conditional assumption, not measured Tang equilibrium",
    number_reference_m3=1e12,rate_units=(J="m^-3 min^-1",G="um min^-1"),
    scheme="MUSCL",integrator="SSPRK33",growth_CFL_upper_bound=.35,
    actual_maximum_dt=min(chosen.settings.dt,.35*(chosen.settings.Lmax/chosen.settings.bins)/ldexp(1.,ceil(Int,chosen.theta[2]/log(2)))),
    schedule="fixed per trial, power-of-two upper bound on G_ref; native steps shorten at requested outputs",
    comparison_scales_are_sensor_standard_deviations=false,seed=2027,
    julia=string(VERSION),packages=versions,cpu=first(Sys.cpu_info()).model,
    julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),START_STATE,
    calibration_loss=chosen.best.loss,metrics,
    literature="Tang et al. 2018, doi:10.1016/j.jcrysgro.2018.06.023, original Figs 11d and 12a",
    data_origin="examples/data/protein_crystallization/tang2018_isothermal/provenance/integration.json")
json(joinpath(OUTPUT,"run_settings.json"),settings)
files=filter(f->(endswith(f,".csv") || endswith(f,".json")) && !(f in ("manifest.json","verification.json")),readdir(OUTPUT))
fit_provenance=JSON3.read(read(joinpath(chosen.folder,"fit.json"),String))
source_hashes=Dict(string(k)=>string(v) for (k,v) in pairs(fit_provenance.source_hashes))
for f in [joinpath(@__DIR__,"run.jl"),@__FILE__,joinpath(Case.DATA,"psd_summary.csv"),
          joinpath(@__DIR__,"moment_check.jl"),
          joinpath(Case.DATA,"config","case.json"),joinpath(Case.DATA,"provenance","integration.json")]
    source_hashes[relpath(f,ROOT)]=bytes2hex(sha256(read(f)))
end
json(joinpath(OUTPUT,"manifest.json"),(;mode=MODE,complete=true,
    files=Dict(f=>bytes2hex(sha256(read(joinpath(OUTPUT,f)))) for f in files),
    source_files=source_hashes))
open(joinpath(OUTPUT,"implementation_result.txt"),"w") do io
    println(io,"Tang 100 mL isothermal batch; original Figure 11d and Figure 12a number-CDF data.")
    println(io,"Selected family: ",settings.model,"; fitted coordinates: ",chosen.theta)
    println(io,"Baseline/candidate calibration losses: ",baseline.best.loss," / ",gated.best.loss)
    println(io,"Selection used calibration only; see calibration_selection.json and frozen_model.json.")
    println(io,"Calibration concentration RMSE (g/L): ",metrics.calibration_concentration_RMSE_g_L)
    println(io,"Calibration W2 (um): ",metrics.calibration_W2_um)
    println(io,"Held-out concentration RMSE/MAE (g/L): ",metrics.validation_concentration_RMSE_g_L," / ",metrics.validation_concentration_MAE_g_L)
    println(io,"Held-out 660-min W2 (um): ",metrics.validation_W2_um)
    println(io,"Maximum material error (g/L): ",metrics.max_material_error_g_L)
    println(io,"Forward solve: ",metrics.solve_seconds," s, ",metrics.steps," steps; one measured solve after warmup.")
    println(io,"Settings, multistarts, predictions, sensitivities, profiles and numerical checks are retained beside this file.")
    println(io,"Comparison scales are not measured sensor errors. No confidence interval or universal kinetics claim.")
    println(io,"Nominal solubility, shape/density and digitization remain assumptions. Preserve calibration and validation mismatch.")
    println(io,"Reproduce: julia --project=paper paper/case_studies/04_protein_crystallization/run.jl")
    println(io,"Plot: python paper/plot_all.py --case 4")
    println(io,"Verify exports: python paper/check_protein.py")
end
println("Completed frozen prediction and supporting studies: ",metrics);flush(stdout)
