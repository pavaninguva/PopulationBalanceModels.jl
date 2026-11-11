# Optional: develop this checkout into an environment containing PythonPlot.
# For batch use set MPLBACKEND=Agg BEFORE importing PythonPlot; no renderer
# forces a headless backend in interactive Julia sessions.
include("plotting.jl")
import PythonPlot
const PyCallPlot=PythonPlot.PythonCall

@testset "PythonPlot extension uses the package-owned shared renderer" begin
    @test Base.get_extension(PBMPlot,:PopulationBalanceModelsPythonPlotExt)!==nothing
    grid,pb,sol=plotting_fixture();original=deepcopy(sol.u)
    ep,map,calls=plotting_estimation_fixture()
    prepared=predictions(ep,map.minimizer)
    approximation=laplace_approximation(ep,map)
    before=calls[]
    figures=Any[]
    distribution=plot_distribution(sol;times=[0.,1.],xlabel="Internal coordinate",ylabel="Number density")
    push!(figures,distribution)
    @test PyCallPlot.pyconvert(Int,distribution.axes.patches.__len__())==2
    measured=Observation([0.,1.],[19.,19.],MomentObservable(0);covariance=[1. .5;.5 2.])
    push!(figures,plot_observable(sol,MomentObservable(0);measurement=measured))
    fitfigure=plot_fit(ep,prepared);push!(figures,fitfigure)
    @test calls[]==before
    push!(figures,plot_parameters(ep,map;coordinates=:physical)) # no posterior required
    push!(figures,plot_parameters(ep,approximation;coordinates=:physical))
    push!(figures,plot_parameters(ep,approximation;coordinates=:inference))
    @test calls[]==before
    # Compose into the existing axis without changing its identity.
    comparison=plot_distribution(sol;times=[0.],ax=distribution.axes,curve="centers",labels=["Reference"])
    @test PyCallPlot.pyconvert(Bool,comparison.axes.__eq__(distribution.axes))
    @test sol.u==original
    g2=Grid2D([0.,1.,3.],[0.,2.,5.]);op=compile(PBMModel(),g2)
    p2=PBMProblem(op,[2. 3.;4. 5.],(0.,1.));s2=PBMSolution([0.],[p2.u0],p2,0)
    field=plot_field(s2;orientation=:internal_x,xlabel="x",ylabel="y");push!(figures,field)
    @test PyCallPlot.pyconvert(Int,field.figure.axes.__len__())==2 # one shared colorbar
    mcproblem=UQProblem(pb,UncertainInputs((k=Uniform(.1,.3),));
        outputs=(number=MomentObservable(0),),times=[0.,1.],dt=.1)
    mc=propagate(mcproblem,MonteCarlo(5;seed=7);max_solves=5)
    original_values=copy(mc.values)
    push!(figures,plot_uq(mc;output=:number))
    @test mc.values==original_values && calls[]==before
    mktempdir() do output
        for (i,figure) in enumerate(figures)
            save_figure(figure,joinpath(output,"figure_$i.png"))
        end
        @test all(endswith(".png"),readdir(output))
        save_figure(distribution,joinpath(output,"wide.png");width=7.,height=3.2,font_size=11.,layout="constrained")
        save_figure(distribution,joinpath(output,"small.png");width=3.35,height=2.8,font_size=8.,dpi=300)
        image=PyCallPlot.pyimport("PIL.Image").open(joinpath(output,"wide.png"))
        @test PyCallPlot.pyconvert(Tuple{Int,Int},image.size)==(2100,960)
        image.close()
    end
    for figure in figures
        PythonPlot.plotclose(figure.figure)
    end
end

# Include at top level so subsequent test code sees the newly defined tutorial
# bindings and methods on Julia versions with global-binding world ages.
include("case_study_scripts.jl")

@testset "Direct tutorial plotting recipes use prepared numerical objects" begin
    for name in (:constant_growth,:crystallization,:moment_crystallization,
                 :continuous_precipitation,:stem_cells,:transport)
        s=TUTORIALS[name]
        solution = if name===:constant_growth
            s.solution
        elseif name===:continuous_precipitation
            latest(simulate,remake(s.problem;tspan=(0.,.01));solver=s.solver,saveat=[0.,.01])
        elseif name===:stem_cells
            latest(simulate,remake(s.problem;tspan=(0.,.05));method=SSPRK33(),dt=s.dt,
                negative_tolerance=0.,saveat=[0.,.05])
        elseif name===:transport
            latest(simulate,remake(s.problem;tspan=(0.,.02));method=SSPRK33(),dt=s.dt,
                check_nonnegative=false,saveat=[0.,.02])
        else
            last_time=name===:crystallization ? .05 : .02
            latest(simulate,remake(s.problem;tspan=(0.,last_time));method=SSPRK33(),
                dt=last_time,saveat=[0.,last_time])
        end
        original=deepcopy(solution.u)
        figures = if name===:continuous_precipitation
            [plot_field(solution;population=:crystals,orientation=:axial_x,
                xlabel="Axial position (m)",ylabel="Size (m)",colorbar_label="Scaled number density"),
             plot_distribution(solution;population=:crystals,cell=s.nz,times=[last(solution.t)],
                xlabel="Size (m)",ylabel="Scaled number density")]
        elseif name===:transport
            [plot_field(solution;orientation=:internal_x,xlabel="First coordinate",ylabel="Second coordinate")]
        elseif name===:moment_crystallization
            [plot_observable(solution,MomentObservable(0);xlabel="Time (s)",ylabel="Total number")]
        elseif name===:crystallization
            [plot_distribution(solution;population=:crystals,xlabel="Length (micrometres)"),
             plot_observable(solution,BulkObservable(:c);ylabel="Solute mass / water mass")]
        else
            [plot_distribution(solution;xlabel=name===:stem_cells ? "Cargo (arbitrary units)" : "Size coordinate",
                ylabel="Number density")]
        end
        @test solution.u==original
        for figure in figures
            @test haskey(figure,:figure) && haskey(figure,:axes)
            PythonPlot.plotclose(figure.figure)
        end
    end
    for name in (:protein_crystallization,:estimating_growth)
        s=TUTORIALS[name]
        ep=if name===:protein_crystallization
            pb=latest(s.batch_problem,s.cnt_model;bins=8,tf=420.)
            EstimationProblem(s.parameter_specs(),
                (InferenceExperiment(:batch,pb,s.measurements...;solver=s.solver),);fixed=s.estimation.fixed)
        else
            s.estimation
        end
        z=initial_coordinates(ep)
        point=MAPResult(z,physical_parameters(ep,z),NaN,NaN,NaN,false,
            (;display_fixture=true),ParameterSpec[ep.parameters...],nothing;convention=:inference_map_v2)
        prepared=predictions(ep,z) # Expensive work is explicit, before rendering.
        snapshot=deepcopy(prepared)
        figures=[plot_parameters(ep,point;coordinates=:physical)]
        if name===:estimating_growth
            push!(figures,plot_fit(ep,prepared;abscissa=s.data.sizes,xlabel="Size coordinate",ylabel="Number density"))
        else
            ex=only(ep.experiments)
            sol=latest(ex.solver,ex.problem,[0.,1.])
            push!(figures,plot_observable(sol,BulkObservable(:c);xlabel="Time (min)",ylabel="Concentration (g/L)"))
        end
        @test prepared==snapshot
        @test !PyCallPlot.pyconvert(Bool,first(figures).figure._pbm_parameter_diagnostics["passed"])
        for figure in figures;PythonPlot.plotclose(figure.figure);end
    end
    if haskey(TUTORIALS,:milling_uncertainty)
        s=TUTORIALS[:milling_uncertainty];pb=latest(s.build_mill;bins=12)
        uq=UQProblem(pb,s.inputs;setup=s.setup,solver=s.solver,times=s.times,outputs=(d50=s.d50,),
            output_coordinates=s.uq.output_coordinates)
        mc=propagate(uq,MonteCarlo(2;seed=2018255);max_solves=2)
        design=pce_design(uq.inputs;n=6,seed=2018257,bounds=hcat(fill(-2.,4),fill(2.,4)))
        pce=fit_pce(uq,PolynomialChaos(1);design,validation=mc,max_solves=6)
        count=mc.metadata.solves+pce.training.metadata.solves
        figures=[plot_uq(mc;output=:d50),plot_uq(pce;output=:d50,band=:sd)]
        @test mc.metadata.solves+pce.training.metadata.solves==count
        for figure in figures;PythonPlot.plotclose(figure.figure);end
    end
end
