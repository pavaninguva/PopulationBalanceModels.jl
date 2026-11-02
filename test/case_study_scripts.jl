using Test, PopulationBalanceModels, LinearAlgebra
include("../docs/case_study_scripts.jl")
include("fixtures/example_modules.jl")

function load_tutorial(name)
    scope = Module(gensym(:Tutorial))
    # Match an ordinary Julia module's include binding in this isolated scope.
    Core.eval(scope, :(include(path) = Base.include(@__MODULE__,path)))
    Base.include_string(scope, CaseStudyScripts.script(name), "$(name)_tutorial.jl")
    scope
end
construct(scope; kwargs...) = Base.invokelatest(getfield(scope,:tutorial_problem); kwargs...)

function same_initial_rates(a,b; time=first(a.tspan))
    @test a.u0 == b.u0
    @test a.tspan == b.tspan
    da=similar(a.u0);db=similar(b.u0)
    Base.invokelatest(rhs!,da,a.u0,a.operator,a.p,time)
    Base.invokelatest(rhs!,db,b.u0,b.operator,b.p,time)
    @test all(isfinite,da)
    @test da≈db rtol=5e-13 atol=1e-15
end

@testset "Tutorial source extraction and standalone downloads" begin
    parse_error(ex)=ex isa Expr && (ex.head in (:error,:incomplete) || any(parse_error,ex.args))
    for name in CaseStudyScripts.CASES
        @test !parse_error(Meta.parseall(CaseStudyScripts.script(name)))
        page=read(joinpath(CaseStudyScripts.ROOT,"docs/src/case_studies","$name.md"),String)
        @test occursin("CaseStudyScripts.script(:$name)",page)
        @test occursin("assets/source/tutorials/$name.jl",page)
    end
    mktempdir() do folder
        CaseStudyScripts.write_scripts(folder)
        @test sort(readdir(folder))==sort(string.(collect(CaseStudyScripts.CASES)).*".jl")
        for name in CaseStudyScripts.CASES
            @test read(joinpath(folder,"$name.jl"),String)==CaseStudyScripts.script(name)
        end
        path=joinpath(folder,"source.jl")
        write(path,"# tutorial:start model\nx = 1\n# tutorial:end model\n")
        @test CaseStudyScripts.sections(path)["model"]=="x = 1\n"
        write(path,"# tutorial:start model\nmodule Fixture\nconst scale = 2\n\"\"\"A reference docstring.\"\"\"\nfunction rate(x)\n    x^2\nend\nunused() = error(\"must not run\")\nend\n# tutorial:end model\n")
        selected=CaseStudyScripts.definitions("source.jl","model",(:scale,:rate);root=folder)
        @test occursin("const scale = 2",selected)
        @test occursin("function rate(x)\n    x^2\nend",selected)
        @test !occursin("unused",selected)
        @test !occursin("reference docstring",selected)
        @test_throws ErrorException CaseStudyScripts.definitions("source.jl","model",(:missing,);root=folder)
        for content in ("# tutorial:end model\n", "# tutorial:start model\n",
                "# tutorial:start a\n# tutorial:start b\n",
                "# tutorial:start a\n# tutorial:end b\n",
                "# tutorial:start a\n# tutorial:end a\n# tutorial:start a\n")
            write(path,content)
            @test_throws ErrorException CaseStudyScripts.sections(path)
        end
    end
    @test_throws ArgumentError CaseStudyScripts.script(:unknown_case)
end

@testset "Tutorial models match maintained initial states and rates" begin
    # No tutorial main(), fit(), simulate() or production exporter is invoked.
    _load_example(:ConstantGrowthMethods,"constant_growth_methods.jl")
    scope=load_tutorial(:constant_growth);case=construct(scope)
    @test case.initial==ConstantGrowthMethods.gaussian_counts(case.grid,0.)
    @test case.reference==ConstantGrowthMethods.gaussian_counts(case.grid,4.)
    @test case.grid.coordinate===:length

    _load_example(:Gunawan2004Case2,"papers/gunawan2004_case2.jl")
    scope=load_tutorial(:crystallization);pb=construct(scope)
    ref=Gunawan2004Case2.build(grid=pb.operator.system.populations.crystals.grid,
        initial_density=Gunawan2004Case2.benchmark_initial_density,
        c0=Gunawan2004Case2.solubility(Gunawan2004Case2.temperature(0.)),tspan=pb.tspan)
    same_initial_rates(pb,ref;time=5.)

    _load_example(:QamarMomentBenchmarks,"qamar_moment_benchmarks.jl")
    scope=load_tutorial(:moment_crystallization)
    pb=construct(scope)
    same_initial_rates(pb,QamarMomentBenchmarks.problem4(;method=QMOM()))
    @test pb.operator.grid.orders==collect(0:5)
    @test only(observe(MomentObservable(3),PBMSolution([0.],[pb.u0],pb,0)))≈0.001

    _load_example(:ContinuousPrecipitation,"continuous_precipitation.jl")
    scope=load_tutorial(:continuous_precipitation);pb=construct(scope;nx=8,nz=6)
    same_initial_rates(pb,ContinuousPrecipitation.build(nx=8,nz=6,scheme=WENO5(),
        axial_scheme=MUSCL(),tf=5.))

    _load_example(:StemCellPopulation,"stem_cell_population.jl")
    scope=load_tutorial(:stem_cells)
    for differentiation in (false,true)
        same_initial_rates(construct(scope;bins=40,differentiation),
            StemCellPopulation.build_problem(bins=40,tf=20.;differentiation))
    end

    _load_example(:InguvaBraatzCase2,"inguva_braatz_case2.jl")
    scope=load_tutorial(:transport)
    same_initial_rates(construct(scope;bins=12),InguvaBraatzCase2.problem(12))

    # The LNP startup assembly uses only core dependencies. Its historical
    # full-horizon Rosenbrock driver is deliberately not imported or executed.
    _load_example(:ShinLNPReproduction,"shin_lnp_reproduction.jl")
    scope=load_tutorial(:nanoparticles);pb=construct(scope;bins=16)
    material=ShinLNPReproduction.CEJ2025LNP
    grid=pb.operator.system.populations.lnp.grid;w=3/4;T=298.15
    permittivity=24.3*(1-w)+78.4w;viscosity=1.2e-3*(1-w)+0.89e-3w
    attachment=(L,M,w,T)->material.dlvo_attachment(L/2,M/2;
        temperature=T,ionic_strength=100w,relative_permittivity=permittivity)
    c3=(7/4)/240;c5=(10/6)/600
    solubility=(w,T)->exp(log(c3)+(w-3/4)/(5/6-3/4)*log(c5/c3))
    ref=ShinLNPReproduction.figure4_problem(;grid,FRR=3.,temperature=t->T,
        viscosity=(w,T)->viscosity,solubility,
        equilibrium_ratio=(L,w,T)->material.kelvin_ratio(L,T,.010,5e-28),
        attachment_without_peg=attachment,molecular_volume=5e-28,particle_density=1000.,
        nucleation_width=1e-9,nucleation_normalization=:total_rate,overflow=:discard,
        track_consumption=true,scheme=Upwind(),tspan=(0.,1e-5))
    same_initial_rates(pb,ref)
end

if Base.find_package("Optim")!==nothing
    @testset "Tutorial inference assembly without fitting or sampling" begin
        _load_example(:EstimateSizeDependentGrowth,"estimate_size_dependent_growth.jl")
        scope=load_tutorial(:estimating_growth)
        for prior in (:lognormal,:uniform)
            ep=construct(scope;prior)
            ref=EstimateSizeDependentGrowth.estimation_problem(;prior)
            z=initial_coordinates(ep)
            @test z==initial_coordinates(ref)
            @test logposterior(ep,z)≈logposterior(ref,z) atol=1e-9
        end
        _load_example(:IsothermalProteinCrystallization,"isothermal_protein_crystallization.jl")
        scope=load_tutorial(:protein_crystallization)
        for model in (:cnt,:cnt_grd)
            ep=construct(scope;model,bins=8)
            ref=IsothermalProteinCrystallization.estimation_problem(;model,bins=8)
            z=initial_coordinates(ep)
            @test z==initial_coordinates(ref)
            @test logprior(ep,z)==logprior(ref,z)
            @test length(only(ep.experiments).measurements)==4
            tutorial=only(ep.experiments).measurements
            retained=only(ref.experiments).measurements
            @test tutorial[1].data==retained[1].data
            logdet=0.0
            for (a,b) in zip(tutorial[2:end],retained[2:end])
                L=IsothermalProteinCrystallization.cdf_factor(b.observable.points)
                @test a.data≈L*b.data atol=2e-14 rtol=0
                @test a.likelihood.covariance≈0.03^2*(L*L') atol=1e-18
                @test a.times==b.times && a.observable.window==b.observable.window
                logdet+=sum(log,diag(L))
            end
            same_initial_rates(only(ep.experiments).problem,only(ref.experiments).problem)
            # A full covariance and externally whitened residuals describe the
            # same posterior law. Restore the KNOWN determinant constant when
            # comparing their normalized log densities, never refit a noise scale.
            @test logposterior(ep,z)≈logposterior(ref,z)-logdet atol=2e-7 rtol=1e-10
            @test posterior_gradient(ep,z)≈posterior_gradient(ref,z) rtol=2e-7 atol=1e-6
            pb=only(ep.experiments).problem
            short=Base.invokelatest(IsothermalProteinCrystallization.cfl_solve,pb,[0.,1.])
            g=pb.operator.system.populations.crystals.grid
            inventory(u)=bulk(pb,u,:c)+pb.p.rho*pb.p.kv*moment(g,population(pb,u,:crystals),3)
            @test maximum(abs(inventory(u)-inventory(pb.u0)) for u in short.u)<1e-12
        end
    end
else
    @info "Inference tutorial constructor checks need Optim; use the docs environment"
end

if Base.find_package("Sundials")!==nothing && Base.find_package("SciMLBase")!==nothing
    @testset "Milling tutorial physical model and UQ assembly without ensembles" begin
        _load_example(:MillingModel,"support/milling_model.jl")
        scope=load_tutorial(:milling_uncertainty)
        uq=construct(scope;bins=40)
        ref=MillingModel.build(;bins=40)
        same_initial_rates(uq.base,ref;time=31.)
        @test uq.times==[1.,5.,29.9,30.,31.,60.,200.]
        @test keys(uq.outputs)==(:d50,)
        @test Base.invokelatest(getfield(scope,:physical),
            (delta=0.,alpha=0.,gamma=0.,sigma=0.))==MillingModel.MEAN
    end
else
    @info "Milling tutorial constructor check needs the documented milling environment (Sundials/SciMLBase)"
end
