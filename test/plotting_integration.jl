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

@testset "Tutorial optional plotting runs on existing small results" begin
    include("case_study_scripts.jl")
    for name in (:constant_growth,:crystallization,:moment_crystallization,:continuous_precipitation,
                 :stem_cells,:nanoparticles,:transport)
        scope=load_tutorial(name)
        # Reuse already supported reduced settings; no production campaign.
        kwargs=name===:constant_growth ? (;bins=40) : name===:crystallization ? (;times=[0.,.05]) :
            name===:moment_crystallization ? (;times=[0.,.02]) :
            name===:continuous_precipitation ? (;nx=8,nz=6,times=[0.,.5]) :
            name===:stem_cells ? (;bins=40,times=[0.,.05]) :
            name===:nanoparticles ? (;bins=16,times=[0.,1e-8]) : (;bins=12,times=[0.,.02])
        result=Base.invokelatest(getfield(scope,:main);kwargs...)
        plots=Base.invokelatest(getfield(scope,:plot_results),result)
        @test plots isa NamedTuple
        if haskey(plots,:figure)
            PythonPlot.plotclose(plots.figure)
        else
            for figure in values(plots);PythonPlot.plotclose(figure.figure);end
        end
    end
    for name in (:protein_crystallization,:estimating_growth)
        scope=load_tutorial(name)
        ep=name===:protein_crystallization ? construct(scope;bins=8) : construct(scope)
        z=initial_coordinates(ep)
        # An explicitly UNCONVERGED display fixture tests plumbing without any
        # optimizer call. Its diagnostic failure must remain visible.
        point=MAPResult(z,physical_parameters(ep,z),NaN,NaN,NaN,false,
            (;display_fixture=true),ParameterSpec[ep.parameters...],nothing;convention=:inference_map_v2)
        if name===:protein_crystallization
            ex=only(ep.experiments)
            sol=Base.invokelatest(ex.solver,ex.problem,[0.,1.])
            result=(;estimation=ep,fitted=point,solution=sol)
        else
            result=(;estimation=ep,fitted=point,predicted=predictions(ep,z))
        end
        plots=Base.invokelatest(getfield(scope,:plot_results),result)
        @test !PyCallPlot.pyconvert(Bool,plots.parameters.figure._pbm_parameter_diagnostics["passed"])
        for figure in values(plots);PythonPlot.plotclose(figure.figure);end
    end
    if Base.find_package("Sundials")!==nothing
        scope=load_tutorial(:milling_uncertainty)
        result=Base.invokelatest(getfield(scope,:main);bins=12,mc_samples=2,
            validation_samples=2,training_samples=6,surrogate_draws=50)
        count=result.mc.metadata.solves+result.pce.training.metadata.solves
        plots=Base.invokelatest(getfield(scope,:plot_results),result)
        @test result.mc.metadata.solves+result.pce.training.metadata.solves==count
        for figure in values(plots);PythonPlot.plotclose(figure.figure);end
    end
end
