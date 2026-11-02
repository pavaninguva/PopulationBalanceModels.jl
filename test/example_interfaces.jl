using Test, PopulationBalanceModels
using Distributions: Uniform, Exponential
include("fixtures/example_modules.jl")

@testset "Example and paper drivers parse without executing" begin
    has_parse_error(ex)=ex isa Expr && (ex.head in (:error,:incomplete) || any(has_parse_error,ex.args))
    root=normpath(joinpath(@__DIR__,".."))
    for tree in ("examples","paper"), (folder,_,files) in walkdir(joinpath(root,tree)), file in files
        endswith(file,".jl") || continue
        @test !has_parse_error(Meta.parseall(read(joinpath(folder,file),String)))
    end
end

# Constructor/RHS checks only: never run a benchmark, fit, chain or exporter.
_load_example(:QamarMomentBenchmarks, "qamar_moment_benchmarks.jl")
_load_example(:TemperatureProgrammedCrystallization, "temperature_programmed_crystallization.jl")
_load_example(:EstimateSizeDependentGrowth, "estimate_size_dependent_growth.jl")
_load_example(:ProteinCrystallization, "protein_crystallization.jl")
_load_example(:IsothermalProteinCrystallization, "isothermal_protein_crystallization.jl")
_load_example(:AxiallyDispersedGrowth, "axially_dispersed_growth.jl")
_load_example(:ContinuousPrecipitation, "continuous_precipitation.jl")

module ExampleBoundaryBuilders
include("../examples/realizable_dissolution.jl")
include("../examples/realizable_startup.jl")
include("../examples/realizable_axial_transport.jl")
end

@testset "Moment recipe constructors resolve selected families" begin
    study=QamarMomentBenchmarks
    families=((QMOM(),QMOM(3)),(RankAdaptiveQMOM(),RankAdaptiveQMOM(3)),
        (EQMOM(secondary_nodes=16),EQMOM(3;secondary_nodes=16)),
        (DQMOM(condition_limit=1e11),DQMOM(3;condition_limit=1e11)))
    for (inferred,explicit) in families, pulse in (false,true)
        builder=pulse ? study.problem5 : study.problem4
        a=builder(;method=inferred,nodes=3)
        b=builder(;method=explicit)
        @test a.operator.grid.orders==b.operator.grid.orders
        @test a.operator.closure==b.operator.closure
        @test a.u0≈b.u0
        da=similar(a.u0);db=similar(b.u0)
        rhs!(da,a.u0,a.operator,a.p,0.)
        rhs!(db,b.u0,b.operator,b.p,0.)
        @test da≈db
        if pulse
            legacy=compile(a.operator.model,a.operator.grid;moment_method=a.operator.closure,
                methods=TermMethods(by_term=(growth=ExactMomentRule(),primary=ExactMomentRule())))
            old=similar(a.u0);rhs!(old,a.u0,legacy,a.p,0.)
            @test da≈old
            source=study.problem5(;method=inferred,collisions=false)
            ds=similar(source.u0);rhs!(ds,source.u0,source.operator,source.p,0.)
            rates=[moment_rate(source.operator.closure,source.operator.grid,source.u0,ds,k)
                for k in source.operator.grid.orders]
            @test rates≈[k==0 ? 100. : k*study.initial_moment5(k-1)
                for k in source.operator.grid.orders] rtol=1e-10
        end
        # DQMOM entries are weights/weighted positions, not raw moments.
        raw=pulse ? [study.initial_moment5(k) for k in a.operator.grid.orders] :
            study.exact4(0.,a.operator.grid.orders)
        @test [moment_value(a.operator.closure,a.operator.grid,a.u0,k)
            for k in a.operator.grid.orders]≈raw
    end
    @test study.method_basis(QMOM(),4).orders==collect(0:7)
    @test study.method_basis(EQMOM(),4).orders==collect(0:8)
    @test study.method_basis(QMOM(2),4).orders==collect(0:3)
    @test study.method_basis(LogNormalClosure(),4).orders==collect(0:2)

    cooling=TemperatureProgrammedCrystallization
    p=cooling.parameters(density=1.293)
    pb=cooling.build_moments(;p,c0=0.3,temperature_profile=t->303.15,tspan=(0.,1.))
    @test pb.operator.system.populations.crystals.closure isa ExactClosure
    du=similar(pb.u0);rhs!(du,pb.u0,pb.operator,pb.p,0.)
    @test all(isfinite,du)
    dissolving=ExampleBoundaryBuilders.dissolution_problem(:moments;resolution=2)
    @test dissolving.operator.system.populations.crystals.closure isa RankAdaptiveQMOM
    @test dissolving.operator.system.populations.crystals.boundary.left===:absorb
end

@testset "Example parameter declarations preserve physical laws and starts" begin
    growth=EstimateSizeDependentGrowth
    data=growth.synthetic_measurements(sizes=[0.1,0.3],times=[0.1,0.5])
    for prior in (:lognormal,:uniform)
        ep=growth.estimation_problem(data;prior)
        z=initial_coordinates(ep)
        @test physical_parameters(ep,z,:growth).g0≈0.4
        @test physical_parameters(ep,z,:growth).g1≈0.23
        @test only(ep.experiments).problem.p==growth.TRUTH
        @test all(s->s.transform isa LogTransform,ep.parameters)
        @test all(s->s.prior.coordinates===:physical,ep.parameters)
        # Compatibility declarations denote the very same prior probability law.
        legacy=[ParameterSpec(s.name;initial=s.initial,transform=s.transform,
            prior=Prior(s.prior.distribution;coordinates=:physical)) for s in ep.parameters]
        old=EstimationProblem(legacy,ep.experiments)
        @test [logprior(ep,z.+d) for d in (-0.1,0.,0.1)]≈
            [logprior(old,z.+d) for d in (-0.1,0.,0.1)]
    end
    historical=ProteinCrystallization.estimation_problem(bins=8)
    @test length(historical.parameters)==4
    @test all(s->s.prior.coordinates===:physical,historical.parameters)

    protein=IsothermalProteinCrystallization
    for model in (:cnt,:cnt_grd)
        ep=protein.estimation_problem(bins=8,model=model)
        z=initial_coordinates(ep)
        @test length(z)==(model===:cnt ? 3 : 4)
        @test ep.parameters[1].prior.distribution==Uniform(protein.AJ_BOUNDS...)
        @test ep.parameters[3].prior.distribution==Uniform(1.,5.)
        # The retained log-uniform AG law has a constant density in log(AG).
        @test ep.parameters[2].prior.coordinates===:inference
        @test ep.parameters[2].prior.distribution==Uniform(log(1e-5),log(1.5))
        logistic_logdensity(v)=-log1p(exp(v))-log1p(exp(-v))
        expected=logistic_logdensity(z[1])+logistic_logdensity(z[3])-log(log(1.5/1e-5))
        if model===:cnt_grd
            D=exp(z[4]);scale=protein.D0_PRIOR_MEAN
            @test ep.parameters[4].prior.distribution==Exponential(scale)
            expected+=-log(scale)-D/scale+log(D)
        end
        @test logprior(ep,z)≈expected atol=1e-12
        @test protein.prior_metadata(ep)[2].density_coordinates===:inference
    end
end

@testset "Axial example builders select independent reconstructions" begin
    standalone=AxiallyDispersedGrowth.build_problem(bins=8,axial_cells=6)
    @test standalone.operator.scheme isa MUSCL
    @test term_methods(standalone.operator).growth isa Upwind
    @test standalone.operator.domain.inlet isa DensityInlet
    @test all(iszero,standalone.u0)
    F=axial_fluxes(standalone.operator,standalone.u0,standalone.p,0.)
    @test sum(F[:,1])>0
    coupled=ContinuousPrecipitation.build(nx=8,nz=6,scheme=WENO5(),axial_scheme=MUSCL())
    @test coupled.operator.scheme isa MUSCL
    @test term_methods(first(coupled.operator.local_operators).system.populations.crystals).growth isa WENO5
    @test all(iszero,coupled.u0)
end

if Base.find_package("Optim")!==nothing && Base.find_package("NonlinearSolve")!==nothing
    _load_example(:EstimateStationaryCrystallizer, "estimate_stationary_crystallizer.jl")
    @testset "Stationary estimation example builds without solving" begin
        ep,truth=EstimateStationaryCrystallizer.estimation_problem()
        @test only(ep.experiments).problem isa StationaryPBMProblem
        @test only(ep.experiments).problem.operator.closure isa ExactClosure
        @test all(s->s.transform isa LogTransform,ep.parameters)
        @test physical_parameters(ep,initial_coordinates(ep),:crystallizer).G≈0.3
        @test truth.G==0.4
    end
end
