# Construction tests regenerate every download and verify maintained-source,
# measurement and posterior consistency. The tests below execute REDUCED paths.
# No production fit, posterior sampling, retained export or plotting is invoked.
include("case_study_scripts.jl")
using ForwardDiff
using Distributions: Normal, Beta, cdf, ccdf

latest(f,args...;kwargs...)=Base.invokelatest(f,args...;kwargs...)
tutorial_main(scope;kwargs...)=latest(getfield(scope,:main);kwargs...)

@testset "Tutorial numerical infrastructure is public and explicit" begin
    for name in CaseStudyScripts.CASES
        code=CaseStudyScripts.script(name)
        @test !occursin(r"PopulationBalanceModels\._",code)
        # Physical rate/source loops are allowed; no native stepping controller
        # or hand-built spectral daughter rule belongs in the displayed scripts.
        @test !occursin(r"\bwhile\s",code)
    end
end

@testset "Gaussian and cooling walkthrough initialization and actual solves" begin
    scope=load_tutorial(:constant_growth)
    case=construct(scope;bins=40)
    sol=tutorial_main(scope;bins=40)
    @test sol.t==[4.]
    @test diagnostics(sol).settings.algorithm==string(typeof(SSPRK33()))
    # Independent CDF integrals from the retained analytical reference. Moving
    # the Gaussian mean or its integration limits is equivalent up to roundoff.
    old=ConstantGrowthMethods.gaussian_counts(case.grid,0.)
    @test maximum(abs,vec(case.problem.u0)-old)<3e-15
    @test maximum(abs,case.reference-ConstantGrowthMethods.gaussian_counts(case.grid,4.))<3e-15
    @test initialization_report(case.problem).policy===:physical
    legacy=remake(case.problem;u0=old)
    dt=.5first(case.grid.widths)/ConstantGrowthMethods.growth
    before=simulate(legacy;dt,method=SSPRK33(),saveat=[4.])
    @test last(sol.u)≈last(before.u) rtol=2e-14 atol=3e-15

    scope=load_tutorial(:crystallization)
    times=[0.,.05,.1]
    sol=tutorial_main(scope;times)
    pb=sol.problem;grid=pb.operator.system.populations.crystals.grid
    initial=population(pb,pb.u0,:crystals)
    expected=map(grid.edges[1:end-1],grid.edges[2:end]) do left,right
        a=max(left,180.5);b=min(right,210.5)
        b<=a ? 0. : .08*((b-a)-((b-195.5)^3-(a-195.5)^3)/(3*15^2))
    end
    @test initial≈expected rtol=3e-14 atol=2e-16
    @test sum(initial)≈1.6 atol=3e-15
    old=Gunawan2004Case2.initial_counts(grid,Gunawan2004Case2.benchmark_initial_density)
    @test initial≈old rtol=3e-14 atol=2e-16
    u0=copy(pb.u0);population(pb,u0,:crystals).=old
    before=simulate(remake(pb;u0);dt=.05,method=SSPRK33(),saveat=times)
    @test last(sol.u)≈last(before.u) rtol=5e-13 atol=1e-15
    inventory(u)=bulk(pb,u,:c)+pb.p.rho*moment(grid,population(pb,u,:crystals),3)
    @test maximum(abs(inventory(u)-inventory(pb.u0)) for u in sol.u)<3e-14
end

@testset "Moment and tubular walkthroughs retain their method and bounds" begin
    scope=load_tutorial(:moment_crystallization)
    times=[0.,.02,.04]
    sol=tutorial_main(scope;times)
    @test vec(sol.problem.u0)==QamarMomentBenchmarks.exact4(0.,0:5)
    @test abs(only(observe(MomentObservable(0),sol))-QamarMomentBenchmarks.exact4(.04)[1])<3e-8
    @test maximum(abs(only(observe(MomentObservable(3),sol;index=i))-.001) for i in eachindex(times))<3e-15

    scope=load_tutorial(:continuous_precipitation)
    times=[0.,.5,2.]
    sol=tutorial_main(scope;nx=8,nz=6,times)
    pb=sol.problem
    @test all(iszero,pb.u0) # intentional packed axial arrays
    @test sol.t==times && sol.steps>0
    @test diagnostics(sol).control===:stability_and_admissibility
    @test minimum(last(sol.u))>=-1e-8
    before=latest(ContinuousPrecipitation.advance,pb,times;method=RK4())
    @test sol.u==before.u
    @test sol.steps==before.steps
    # Independent reactor inventory equation: actual discrete uptake cancels
    # between particles and bulk; only the axial boundary flux remains.
    op=pb.operator;localop=first(op.local_operators)
    g=localop.system.populations.crystals.grid;p=pb.p
    for (u,t) in zip(sol.u,sol.t)
        du=similar(u);rhs!(du,u,op,p,t)
        crystal_rate=vec(sum((1e12p.rho*p.kv .*g.centers.^3).*population(pb,du,:crystals);dims=1))
        actual=sum(op.domain.grid.widths.*(bulk(pb,du,:c)+crystal_rate))
        flux=axial_fluxes(op,u,p,t)
        c=localop.bulk_indices.c;r=localop.ranges.crystals
        boundary=flux[c,1]-flux[c,end]+sum(1e12p.rho*p.kv .*g.centers.^3 .*(flux[r,1]-flux[r,end]))
        @test actual≈boundary atol=1e-14 rtol=1e-12
    end
end

function beta_interval(d,a,b)
    a>=b ? 0. : a>.5 ? ccdf(d,a)-ccdf(d,b) : cdf(d,b)-cdf(d,a)
end
@testset "Stem-cell continuous law, independent pivot hats and historical distinction" begin
    scope=load_tutorial(:stem_cells)
    sol=tutorial_main(scope;bins=40,times=[0.,.05,.1])
    pb=sol.problem;op=pb.operator.system.populations.stem;g=op.grid
    @test population(pb,pb.u0,:stem)==StemCellPopulation.initial_population(g)
    for (u,t) in zip(sol.u,sol.t)
        report=latest(StemCellPopulation.state_summary,pb,u,t)
        @test report.minimum_count>=0
        @test abs(report.number_balance)<3e-13
        @test abs(report.cargo_balance)<3e-11
    end
    # Expected daughter count integral: 2 Beta(40,40); its first partial raw
    # moment is a Beta(41,40) CDF times the parent. These independent integrals
    # determine every linear pivot-hat coefficient without spectral nodes.
    parent=g.centers[13];N=zeros(length(g),1);N[13]=1.
    law=latest(StemCellPopulation.division_daughters)
    term=Breakage((x,p,t,c)->1.,law;overflow=:discard,coordinate_map=AdditiveCoordinate(identity,identity))
    direct=compile(PBMModel(term),g);du=zero(N);rhs!(du,N,direct,(;),0.)
    births=vec(du+N);expected=zeros(length(g))
    for i in 1:12
        a,b=g.centers[i:i+1]
        m0=2beta_interval(Beta(40.,40.),a/parent,b/parent)
        m1=parent*beta_interval(Beta(41.,40.),a/parent,b/parent)
        expected[i]+=(b*m0-m1)/(b-a)
        expected[i+1]+=(m1-a*m0)/(b-a)
    end
    @test births≈expected rtol=2e-12 atol=2e-14
    # Reuse the existing normalized-event tolerances from stem_cells.jl.
    # Beta's log-normalizer and the summed hat integrals incur roundoff; no
    # computed daughter count is rescaled to enforce a balance.
    @test sum(births)/2≈1. atol=2e-13
    @test sum(g.centers.*births)/2≈parent/2 rtol=1e-12
    oldlaw=latest(StemCellPopulation.beta_daughters;order=12)
    oldop=compile(PBMModel(Breakage((x,p,t,c)->1.,oldlaw;overflow=:discard,
        coordinate_map=AdditiveCoordinate(identity,identity))),g)
    olddu=zero(N);rhs!(olddu,N,oldop,(;),0.)
    @test sum(olddu+N)/2≈sum(births)/2 atol=2e-13
    @test sum(g.centers.*vec(olddu+N))≈parent atol=3e-13
    @test norm(vec(olddu)-vec(du),1)>1e-6 # no false bitwise-equivalence claim
    before=latest(StemCellPopulation.solve_case;bins=40,tf=.1,differentiation=true,saveat=sol.t)
    for (u,t) in zip(before.u,before.t)
        report=latest(StemCellPopulation.state_summary,before.problem,u,t)
        @test abs(report.number_balance)<3e-13 && abs(report.cargo_balance)<3e-11
    end
    println("Stem migration short-run difference: ",(
        population_l1=norm(population(pb,last(sol.u),:stem)-population(before.problem,last(before.u),:stem),1),
        number=sum(population(pb,last(sol.u),:stem))-sum(population(before.problem,last(before.u),:stem)),
        cargo=moment(g,population(pb,last(sol.u),:stem)-population(before.problem,last(before.u),:stem),1),
        exits=bulk(pb,last(sol.u),:differentiated_cells)-bulk(before.problem,last(before.u),:differentiated_cells)))
end

@testset "Joint initial density remains parameter dependent; diagnostic transport and LNP" begin
    scope=load_tutorial(:transport)
    pb=construct(scope;bins=12);g=pb.operator.grid
    reference=InguvaBraatzCase2.reference(g,0.,pb.p)
    @test pb.u0≈reference rtol=3e-13 atol=2e-15
    p=merge(pb.p,(amplitude=30.,center=(.5,.45),width=(.06,.04)))
    changed=remake(pb;p)
    fresh=latest(getfield(scope,:problem),12;p,initial=pb.initial_description)
    @test changed.u0==fresh.u0
    @test changed.u0≈InguvaBraatzCase2.reference(g,0.,p) rtol=3e-13 atol=2e-15
    physical_area=2pi*prod(pb.p.width)*prod(cdf(Normal(pb.p.center[i],pb.p.width[i]),2.)-
        cdf(Normal(pb.p.center[i],pb.p.width[i]),0.) for i in 1:2)
    f=A->sum(remake(pb;p=merge(pb.p,(amplitude=A,))).u0)
    @test ForwardDiff.derivative(f,50.)≈physical_area atol=3e-15
    @test pb.p.amplitude==50.
    sol=tutorial_main(scope;bins=12,times=[0.,.02])
    @test diagnostics(sol).settings.check_nonnegative===false
    @test only(observe(MomentObservable((0,0)),sol))≈sum(sol.problem.u0) atol=3e-14

    scope=load_tutorial(:nanoparticles)
    sol=tutorial_main(scope;bins=16,times=[0.,1e-8,2e-8])
    @test sol.steps==2 && all(isfinite,last(sol.u))
    @test sum(population(sol.problem,last(sol.u),:lnp))>0
end

if Base.find_package("Optim")!==nothing
    @testset "Retained analytical and bucketed inference callables execute" begin
        scope=load_tutorial(:estimating_growth)
        ep=construct(scope);ex=only(ep.experiments)
        sol=latest(ex.solver,ex.problem,[0.,.1])
        @test sol.problem===ex.problem && all(isfinite,last(sol.u))
        @test ex.problem.initial_description isa InitialDistribution
        @test all(isfinite,predictions(ep,initial_coordinates(ep))[1][1])
        scope=load_tutorial(:protein_crystallization)
        for model in (:cnt,:cnt_grd)
            ep=construct(scope;model,bins=8);ex=only(ep.experiments)
            @test ex.solver===getfield(scope,:cfl_solve)
            modern=latest(ex.solver,ex.problem,[0.,1.])
            before=latest(IsothermalProteinCrystallization.cfl_solve,ex.problem,[0.,1.])
            @test modern.u==before.u && modern.steps==before.steps
        end
    end
end

if Base.find_package("Sundials")!==nothing && Base.find_package("SciMLBase")!==nothing
    @testset "Reduced milling main: direct, MC and PCE share the specialized solver" begin
        scope=load_tutorial(:milling_uncertainty)
        result=tutorial_main(scope;bins=12,mc_samples=2,validation_samples=2,training_samples=6,surrogate_draws=50)
        @test result.mc.metadata.solves==2 && all(result.mc.success)
        @test result.pce.training.metadata.solves==6 && all(result.pce.training.success)
        @test result.nominal.t==result.mc.problem.times
        pb=result.nominal.problem
        @test all(iszero,pb.u0)
        u=result.nominal.u[4] # exactly 30 s, new speed for the outlet observation
        before=latest(scope.MillingModel.outlet,merge(pb.p,(segment_speed=4923.,)),population(pb,u,:mill),30.)
        after=latest(scope.MillingModel.outlet,pb.p,population(pb,u,:mill),30.)
        @test after.resolved≈(1500/4923).*before.resolved.+(1-1500/4923).*(1 .-pb.p.retained).*pb.p.feed rtol=3e-15
        _load_example(:MillingUncertainty,"milling_uncertainty.jl")
        exact=latest(MillingUncertainty.exponential_reference,pb,result.nominal.t)
        @test maximum(maximum(abs,result.nominal.u[i]-exact.u[i]) for i in eachindex(exact.t))<5e-8
        @test latest(MillingUncertainty.state_diagnostics,pb,result.nominal).maximum_mass_error_kg<1e-11
        # sigma changes the cached daughter projection: each scenario rebuilds
        # the SAME model, rather than silently reusing the nominal operator.
        theta=merge(scope.MEAN,(sigma=scope.MEAN.sigma+.1,))
        rebuilt=latest(result.mc.problem.setup,theta,pb)
        @test rebuilt.p.sigma==theta.sigma && rebuilt.p.Q!=pb.p.Q
        @test pb.p.sigma==scope.MEAN.sigma
    end
end
