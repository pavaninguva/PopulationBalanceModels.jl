using Test, PopulationBalanceModels, LinearAlgebra
include("../docs/case_study_scripts.jl")
include("fixtures/example_modules.jl")

function load_tutorial(name; path=CaseStudyScripts.source_path(name))
    scope = Module(gensym(:Tutorial))
    Core.eval(scope, :(include(path) = Base.include(@__MODULE__,path)))
    redirect_stdout(devnull) do
        Base.include(scope,path)
    end
    scope
end
latest(f,args...;kwargs...) = Base.invokelatest(f,args...;kwargs...)
function same_initial_rates(a,b;time=first(a.tspan))
    @test a.u0≈b.u0 rtol=5e-13 atol=1e-15
    @test a.tspan==b.tspan
    da=similar(a.u0); db=similar(b.u0)
    latest(rhs!,da,a.u0,a.operator,a.p,time)
    latest(rhs!,db,b.u0,b.operator,b.p,time)
    @test all(isfinite,da)
    @test da≈db rtol=5e-13 atol=1e-15
end

@testset "Authored source, displayed code and portable downloads" begin
    for name in CaseStudyScripts.CASES
        code=CaseStudyScripts.script(name)
        @test code==read(CaseStudyScripts.source_path(name),String)
        @test !occursin(r"PopulationBalanceModels\._|\btutorial_problem\b|\bplot_results\b",code)
        @test !occursin(r"\bwhile\s",code) # specialized execution stays in named companions
        @test Meta.parseall(code) isa Expr
        page=read(joinpath(CaseStudyScripts.ROOT,"docs/src/case_studies","$name.md"),String)
        @test occursin("CaseStudyScripts.script(:$name)",page)
        @test occursin("assets/source/tutorials/$name.jl",page)
    end
    mktempdir() do folder
        target=joinpath(folder,"tutorials")
        CaseStudyScripts.write_scripts(target)
        for name in CaseStudyScripts.CASES
            @test read(joinpath(target,"$name.jl"),String)==CaseStudyScripts.script(name)
        end
        # Actually include downloaded files outside the checkout, including
        # data, companion includes and the separate accounting/propagation stage.
        for name in (:continuous_precipitation,:protein_crystallization,:estimating_growth)
            scope=load_tutorial(name;path=joinpath(target,"$name.jl"))
            # Julia 1.13 versions global bindings as well as methods. This
            # closure predates the module just included above.
            @test latest(getfield,scope,:problem) isa PBMProblem
        end
        @test isfile(joinpath(target,"stem_cell_accounting.jl"))
        stem=load_tutorial(:stem_cells;path=joinpath(target,"stem_cells.jl"))
        Base.include(stem,joinpath(target,"stem_cell_accounting.jl"))
        @test latest(getfield,stem,:accounted_problem) isa PBMProblem
        @test isfile(joinpath(target,"milling_propagation.jl"))
        if Base.find_package("Sundials")!==nothing && Base.find_package("SciMLBase")!==nothing
            mill=load_tutorial(:milling_uncertainty;path=joinpath(target,"milling_uncertainty.jl"))
            @test latest(getfield,mill,:problem) isa PBMProblem && !latest(isdefined,mill,:mc)
        end
    end
    @test_throws ArgumentError CaseStudyScripts.script(:unknown_case)
end

const TUTORIALS=Dict(name=>load_tutorial(name) for name in CaseStudyScripts.CASES
    if name!==:milling_uncertainty && (Base.find_package("Optim")!==nothing ||
        !(name in (:protein_crystallization,:estimating_growth))))

const TUTORIAL_REFERENCES=Dict{Symbol,Any}()

@testset "Visible tutorial models preserve scientific inputs and initial rates" begin
    _load_example(:ConstantGrowthMethods,"constant_growth_methods.jl")
    s=TUTORIALS[:constant_growth]; pb=s.problem
    @test pb.initial_description isa InitialDensity
    @test vec(pb.u0)≈ConstantGrowthMethods.gaussian_counts(s.grid,0.) atol=3e-15 rtol=0
    @test initialization_report(pb).policy===:physical
    @test s.solution.t==[0.,2.,4.] # this short standalone script really executes

    _load_example(:Gunawan2004Case2,"papers/gunawan2004_case2.jl")
    s=TUTORIALS[:crystallization]; pb=s.problem
    ref=Gunawan2004Case2.density_problem(grid=s.grid,initial_density=Gunawan2004Case2.benchmark_initial_density,
        support=(180.5,210.5),c0=Gunawan2004Case2.solubility(Gunawan2004Case2.temperature(0.)),tspan=pb.tspan)
    same_initial_rates(pb,ref;time=5.)
    @test all(x->x in s.grid.edges,(180.5,210.5))
    @test sum(population(pb,pb.u0,:crystals))≈1.6 atol=3e-15

    _load_example(:QamarMomentBenchmarks,"qamar_moment_benchmarks.jl")
    s=TUTORIALS[:moment_crystallization]
    same_initial_rates(s.problem,QamarMomentBenchmarks.problem4(;method=QMOM()))
    @test s.basis.orders==collect(0:5)

    _load_example(:ContinuousPrecipitation,"continuous_precipitation.jl")
    s=TUTORIALS[:continuous_precipitation]
    same_initial_rates(s.problem,ContinuousPrecipitation.build(nx=50,nz=50,scheme=WENO5(),axial_scheme=MUSCL(),tf=5.))
    @test all(iszero,s.problem.u0)

    _load_example(:StemCellPopulation,"stem_cell_population.jl")
    s=TUTORIALS[:stem_cells]; pb=s.problem
    @test vec(pb.u0)==StemCellPopulation.initial_population(s.grid)
    ref=latest(StemCellPopulation.continuous_problem;bins=600,tf=20.,differentiation=true)
    a=similar(pb.u0);b=similar(ref.u0)
    latest(rhs!,a,pb.u0,pb.operator,pb.p,0.)
    latest(rhs!,b,ref.u0,ref.operator,ref.p,0.)
    @test vec(a)≈population(ref,b,:stem) rtol=5e-13 atol=1e-15
    @test pb.p.differentiation # omitting counters must NOT omit this sink

    _load_example(:InguvaBraatzCase2,"inguva_braatz_case2.jl")
    s=TUTORIALS[:transport];pb=s.problem
    @test pb.u0≈InguvaBraatzCase2.reference(s.grid,0.,pb.p) rtol=3e-13 atol=2e-15
    ref=InguvaBraatzCase2.problem(50;initial=s.initial)
    same_initial_rates(pb,ref)

end

if Base.find_package("Optim")!==nothing
    @testset "Fixed-parameter inference targets, splits, priors and gradients" begin
        _load_example(:EstimateSizeDependentGrowth,"estimate_size_dependent_growth.jl")
        s=TUTORIALS[:estimating_growth];ep=s.estimation
        ref=EstimateSizeDependentGrowth.estimation_problem()
        @test s.data.values==EstimateSizeDependentGrowth.synthetic_measurements().values
        for z in (initial_coordinates(ep),initial_coordinates(ep).+[.02,-.01])
            @test logprior(ep,z)==logprior(ref,z)
            @test loglikelihood(ep,z)≈loglikelihood(ref,z) atol=1e-9
            @test posterior_gradient(ep,z)≈posterior_gradient(ref,z) rtol=2e-12 atol=1e-8
        end
        @test only(ep.experiments).measurements[1].likelihood isa TruncatedGaussianLikelihood
        @test only(ep.experiments).problem.initial_description isa InitialDistribution

        _load_example(:IsothermalProteinCrystallization,"isothermal_protein_crystallization.jl")
        s=TUTORIALS[:protein_crystallization]
        @test length(s.calibration_concentration)==11 && length(s.test_concentration)==4
        @test [first(r).time_min for r in s.calibration_psds]==[74.,180.,420.]
        @test [first(r).time_min for r in s.test_psds]==[660.]
        @test length(s.estimation.parameters)==3 && length(s.grd_estimation.parameters)==4
        for (model,terms,p) in ((:cnt,s.cnt_model,s.parameters()),(:cnt_grd,s.grd_model,s.grd_parameters),
                                  (:cnt_growth_grd,s.growth_grd_model,s.growth_grd_parameters))
            pb=latest(s.batch_problem,terms;p,bins=8,tf=420.)
            ex=InferenceExperiment(:batch,pb,s.measurements...;solver=s.solver)
            fixed=model===:cnt ? (cstar=p.cstar,gamma=s.FIXED_GAMMA,grd_strength=0.) :
                (cstar=p.cstar,gamma=s.FIXED_GAMMA)
            ep=EstimationProblem(latest(s.parameter_specs;p,model),(ex,);fixed)
            ref=IsothermalProteinCrystallization.estimation_problem(;model,bins=8)
            z=initial_coordinates(ep)
            @test z==initial_coordinates(ref) && logprior(ep,z)==logprior(ref,z)
            @test length(ex.measurements)==4 # no derived quantiles or held-out blocks
            @test ex.measurements[1].data==only(ref.experiments).measurements[1].data
            logdet=0.0
            for (a,b) in zip(ex.measurements[2:end],only(ref.experiments).measurements[2:end])
                L=IsothermalProteinCrystallization.cdf_factor(b.observable.points)
                @test a.data≈L*b.data atol=2e-14 rtol=0
                @test a.likelihood.covariance≈.03^2*(L*L') atol=1e-18
                @test a.times==b.times && a.observable.window==b.observable.window
                logdet+=sum(log,diag(L))
            end
            same_initial_rates(pb,only(ref.experiments).problem)
            # Original CDF and whitened residual likelihoods differ by this
            # known, parameter-independent normalization constant, and only it.
            @test logposterior(ep,z)≈logposterior(ref,z)-logdet atol=2e-7 rtol=1e-10
            @test posterior_gradient(ep,z)≈posterior_gradient(ref,z) rtol=2e-7 atol=1e-6
        end
    end
end

if Base.find_package("Sundials")!==nothing && Base.find_package("SciMLBase")!==nothing
    @testset "Milling setup rebuilds parameter-dependent physics without an ensemble" begin
        s=load_tutorial(:milling_uncertainty);TUTORIALS[:milling_uncertainty]=s
        _load_example(:MillingModel,"support/milling_model.jl")
        pb=latest(s.build_mill;bins=12);ref=MillingModel.build(;bins=12)
        same_initial_rates(pb,ref;time=31.)
        @test !isdefined(s,:nominal) && !isdefined(s,:mc) && !isdefined(s,:pce)
        @test s.times==[1.,5.,29.9,30.,31.,60.,200.]
        @test latest(s.physical,(delta=0.,alpha=0.,gamma=0.,sigma=0.))==MillingModel.MEAN
        theta=merge(s.MEAN,(sigma=s.MEAN.sigma+.1,))
        rebuilt=latest(s.setup,theta,pb)
        @test rebuilt.p.sigma==theta.sigma && rebuilt.p.Q!=pb.p.Q
        @test pb.p.sigma==s.MEAN.sigma
    end
end
