# Reduced executions of authored workflows; no fitting, MCMC or retained export.
include("case_study_scripts.jl")
using ForwardDiff
using Distributions: Normal, Beta, cdf, ccdf

@testset "Initial integrals and short growth/cooling/moment trajectories" begin
    s=TUTORIALS[:constant_growth];pb=s.problem
    before=simulate(remake(pb;u0=ConstantGrowthMethods.gaussian_counts(s.grid,0.));
        method=SSPRK33(),dt=s.dt,saveat=s.solution.t)
    @test last(s.solution.u)≈last(before.u) rtol=2e-14 atol=3e-15
    @test vec(pb.u0)≈ConstantGrowthMethods.gaussian_counts(s.grid,0.) atol=3e-15 rtol=0
    area=sum(pb.u0)/pb.p.height
    @test ForwardDiff.derivative(A->sum(remake(pb;p=merge(pb.p,(height=A,))).u0),100.)≈area atol=3e-15
    @test pb.p.height==100. # parameter-dependent initialization did not mutate base

    s=TUTORIALS[:crystallization];pb=s.problem;grid=s.grid
    expected=map(grid.edges[1:end-1],grid.edges[2:end]) do left,right
        a=max(left,180.5);b=min(right,210.5)
        b<=a ? 0. : .08*((b-a)-((b-195.5)^3-(a-195.5)^3)/(3*15^2))
    end
    @test population(pb,pb.u0,:crystals)≈expected rtol=3e-14 atol=2e-16
    times=[0.,.05,.1]
    solution=simulate(remake(pb;tspan=(0.,.1));method=SSPRK33(),dt=.05,saveat=times)
    old=Gunawan2004Case2.build(grid=grid,initial_density=Gunawan2004Case2.benchmark_initial_density,
        c0=s.solubility(s.temperature(0.)),tspan=(0.,.1))
    previous=simulate(old;method=SSPRK33(),dt=.05,saveat=times)
    @test last(solution.u)≈last(previous.u) rtol=5e-13 atol=1e-15
    inventory(u)=bulk(pb,u,:c)+pb.p.rho*moment(grid,population(pb,u,:crystals),3)
    @test maximum(abs(inventory(u)-inventory(pb.u0)) for u in solution.u)<3e-14

    s=TUTORIALS[:moment_crystallization]
    solution=simulate(remake(s.problem;tspan=(0.,.04));method=SSPRK33(),dt=.02,saveat=[0.,.02,.04])
    @test vec(solution.problem.u0)==QamarMomentBenchmarks.exact4(0.,0:5)
    @test abs(only(observe(MomentObservable(0),solution))-QamarMomentBenchmarks.exact4(.04)[1])<3e-8
    @test maximum(abs(only(observe(MomentObservable(3),solution;index=i))-.001) for i in eachindex(solution.t))<3e-15
end

# Explicitly reduced NUMERICAL settings for the same tutorial physical model.
# Keeping these fixtures in tests avoids adding tutorial factories/CLI switches.
function small_precipitation(s)
    p=s.p;nx,nz=8,6
    grid=Grid1D(exp.(range(log(p.xc),log(p.xmax);length=nx+1));coordinate=:length)
    op=compile(s.model,grid;methods=TermMethods(growth=WENO5(),nucleation=BoundaryInjection(:left)))
    exchange=ConservativeExchange(:crystals,:c,(x,p,t,ctx)->1e12*p.rho*p.kv*x^3)
    system=PBMSystem((crystals=op,);exchanges=(exchange,),nonnegative_bulk=(:c,))
    local_pb=PBMProblem(system,(crystals=zeros(nx),),(0.,2.);p,bulk0=(c=0.,))
    reactor=AxialDispersion(AxialGrid(range(0.,p.L;length=nz+1));
        velocity=p.Q/(p.area*p.porosity),dispersion=p.Dax,
        inlet=(p,t)->(populations=(crystals=zeros(nx),),bulk=(c=p.cfeed,)))
    axial_problem(local_pb,reactor;axial_scheme=MUSCL())
end

@testset "Preserved precipitation bound, execution and axial material balance" begin
    s=TUTORIALS[:continuous_precipitation];pb=small_precipitation(s);times=[0.,.5,2.]
    solution=latest(simulate,pb;solver=s.solver,saveat=times)
    previous=latest(ContinuousPrecipitation.advance,pb,times;method=RK4())
    @test solution.u==previous.u && solution.steps==previous.steps
    @test diagnostics(solution).control===:stability_and_admissibility
    @test minimum(last(solution.u))>=-1e-8
    @test latest(s.precipitation_step_bound,pb,last(solution.u),2.)==
        latest(ContinuousPrecipitation.precipitation_step_bound,pb,last(solution.u),2.)
    op=pb.operator;localop=first(op.local_operators);g=localop.system.populations.crystals.grid;p=pb.p
    for (u,t) in zip(solution.u,solution.t)
        du=similar(u);latest(rhs!,du,u,op,p,t)
        crystal_rate=vec(sum((1e12p.rho*p.kv .*g.centers.^3).*population(pb,du,:crystals);dims=1))
        actual=sum(op.domain.grid.widths.*(bulk(pb,du,:c)+crystal_rate))
        flux=latest(axial_fluxes,op,u,p,t);c=localop.bulk_indices.c;r=localop.ranges.crystals
        boundary=flux[c,1]-flux[c,end]+sum(1e12p.rho*p.kv .*g.centers.^3 .*(flux[r,1]-flux[r,end]))
        @test actual≈boundary atol=1e-14 rtol=1e-12
    end
end

beta_interval(d,a,b)=a>=b ? 0. : a>.5 ? ccdf(d,a)-ccdf(d,b) : cdf(d,b)-cdf(d,a)
@testset "Stem dynamics without counters equal the complete accounting model" begin
    s=TUTORIALS[:stem_cells]
    g=Grid1D(range(0.,120.;length=41);coordinate=:oct4)
    op=compile(s.model,g;methods=TermMethods(growth=MUSCL(),breakage=FixedPivot(daughter_quadrature=GaussLegendre(8))))
    pb=PBMProblem(op,s.initial,(0.,.1);p=s.p)
    dt=min(.05,suggest_dt(op,pb.u0,pb.p,0.;cfl=.35))
    solution=latest(simulate,pb;method=SSPRK33(),dt,negative_tolerance=0.,saveat=[0.,.05,.1])
    # Run the authored diagnostic extension with the same reduced geometry.
    for (key,value) in ((:grid,g),(:operator,op),(:problem,pb))
        Core.eval(s,Expr(:(=),key,QuoteNode(value)))
    end
    Base.include(s,joinpath(dirname(CaseStudyScripts.source_path(:stem_cells)),"stem_cell_accounting.jl"))
    accounted=latest(simulate,s.accounted_problem;method=SSPRK33(),dt,negative_tolerance=0.,saveat=solution.t)
    ref=latest(StemCellPopulation.continuous_problem;bins=40,tf=.1,differentiation=true)
    reference=latest(simulate,ref;method=SSPRK33(),dt,negative_tolerance=0.,saveat=solution.t)
    @test pb.u0==reshape(StemCellPopulation.initial_population(g),:,1)
    for i in eachindex(solution.t)
        @test vec(solution.u[i])≈population(accounted.problem,accounted.u[i],:stem) rtol=2e-14 atol=3e-15
        @test accounted.u[i]≈reference.u[i] rtol=5e-13 atol=1e-15
        report=latest(StemCellPopulation.state_summary,accounted.problem,accounted.u[i],accounted.t[i])
        @test report.minimum_count>=0
        @test abs(report.number_balance)<3e-13 && abs(report.cargo_balance)<3e-11
    end
    # Independent incomplete-Beta integrals of the linear pivot hat functions.
    parent=g.centers[13];N=zeros(length(g),1);N[13]=1.
    term=Breakage((x,p,t,c)->1.,s.daughters;overflow=:discard,
        coordinate_map=AdditiveCoordinate(identity,identity))
    dop=compile(PBMModel(term),g);du=zero(N);rhs!(du,N,dop,(;),0.)
    births=vec(du+N);expected=zeros(length(g))
    for i in 1:12
        a,b=g.centers[i:i+1]
        m0=2beta_interval(Beta(40.,40.),a/parent,b/parent)
        m1=parent*beta_interval(Beta(41.,40.),a/parent,b/parent)
        expected[i]+=(b*m0-m1)/(b-a);expected[i+1]+=(m1-a*m0)/(b-a)
    end
    @test births≈expected rtol=2e-12 atol=2e-14
    @test sum(births)/2≈1. atol=2e-13
    @test sum(g.centers.*births)≈parent rtol=1e-12
end

@testset "Joint initialization, signed transport and checked LNP startup" begin
    s=TUTORIALS[:transport]
    g=Grid2D(range(0.,2.;length=13),range(0.,2.;length=13))
    boundary=SizeBC(OpenBoundary(),NoFlux())
    op=compile(s.model,g;methods=TermMethods(growth=WENO5(epsilon=1e-12)),bc=SizeBC2D(x=boundary,y=boundary))
    pb=PBMProblem(op,s.initial,(0.,.02);p=s.p)
    @test pb.u0≈InguvaBraatzCase2.reference(g,0.,pb.p) rtol=3e-13 atol=2e-15
    p=merge(pb.p,(amplitude=30.,center=(.5,.45),width=(.06,.04)))
    changed=remake(pb;p)
    @test changed.u0≈InguvaBraatzCase2.reference(g,0.,p) rtol=3e-13 atol=2e-15
    area=2pi*prod(pb.p.width)*prod(cdf(Normal(pb.p.center[i],pb.p.width[i]),2.)-
        cdf(Normal(pb.p.center[i],pb.p.width[i]),0.) for i in 1:2)
    @test ForwardDiff.derivative(A->sum(remake(pb;p=merge(pb.p,(amplitude=A,))).u0),50.)≈area atol=3e-15
    dt=suggest_dt(op,pb.u0,pb.p,0.;cfl=.4)
    solution=latest(simulate,pb;method=SSPRK33(),dt,saveat=[0.,.02],check_nonnegative=false)
    @test diagnostics(solution).settings.check_nonnegative===false
    @test only(observe(MomentObservable((0,0)),solution))≈sum(pb.u0) atol=3e-14

    s=TUTORIALS[:nanoparticles]
    solution=latest(simulate,remake(s.problem;tspan=(0.,2e-8));method=SSPRK33(),dt=1e-8,saveat=[0.,1e-8,2e-8])
    @test solution.steps==2 && all(isfinite,last(solution.u))
    @test sum(population(solution.problem,last(solution.u),:lnp))>0
    @test bulk(solution.problem,last(solution.u),:lipid)>0
    reference=latest(simulate,remake(TUTORIAL_REFERENCES[:nanoparticles];tspan=(0.,2e-8));
        method=SSPRK33(),dt=1e-8,saveat=solution.t)
    @test last(solution.u)≈last(reference.u) rtol=5e-13 atol=1e-15
end

if Base.find_package("Optim")!==nothing
    @testset "Analytical and bucketed inference solvers without calibration" begin
        s=TUTORIALS[:estimating_growth];ex=only(s.estimation.experiments)
        solution=latest(ex.solver,ex.problem,[0.,.1])
        @test solution.problem===ex.problem && all(isfinite,last(solution.u))
        @test all(isfinite,predictions(s.estimation,initial_coordinates(s.estimation))[1][1])
        @test !isdefined(s,:fitted)
        s=TUTORIALS[:protein_crystallization]
        for terms in (s.cnt_model,s.grd_model)
            pb=latest(s.batch_problem,terms;bins=8)
            solution=latest(s.solver,pb,[0.,1.])
            previous=latest(IsothermalProteinCrystallization.cfl_solve,pb,[0.,1.])
            @test solution.u==previous.u && solution.steps==previous.steps
            g=pb.operator.system.populations.crystals.grid
            inventory(u)=bulk(pb,u,:c)+pb.p.rho*pb.p.kv*moment(g,population(pb,u,:crystals),3)
            @test maximum(abs(inventory(u)-inventory(pb.u0)) for u in solution.u)<1e-12
        end
        @test !isdefined(s,:fitted)
    end
end

if haskey(TUTORIALS,:milling_uncertainty)
    @testset "Separate nominal, MC, PCE and independent validation stages" begin
        s=TUTORIALS[:milling_uncertainty];pb=latest(s.build_mill;bins=12)
        nominal=latest(s.solver,pb,s.times)
        @test all(iszero,pb.u0) && all(isfinite,last(nominal.u))
        # Instantaneous observations at the transition use the right-hand speed.
        u=nominal.u[4];N=population(pb,u,:mill)
        before=latest(s.MillingModel.outlet,merge(pb.p,(segment_speed=4923.,)),N,30.)
        after=latest(s.MillingModel.outlet,pb.p,N,30.)
        @test after.resolved≈(1500/4923).*before.resolved.+(1-1500/4923).*(1 .-pb.p.retained).*pb.p.feed rtol=3e-15
        # Independent augmented affine-system exponential in material coordinates.
        units=vcat(pb.p.rho*pb.p.number_scale.*pb.p.pivots,1.,1.)
        state=copy(pb.u0);lasttime=0.;exact=typeof(state)[]
        for t in s.times
            stops=lasttime<pb.p.change_time<t ? [pb.p.change_time,t] : [t]
            for endpoint in stops
                rpm=lasttime<pb.p.change_time ? pb.p.speed_before : pb.p.speed_after
                local_pb=remake(pb;p=merge(pb.p,(segment_speed=rpm,)))
                A,b=latest(s.MillingModel.linear_system,local_pb)
                augmented=[(A.*units)./units' units.*b;zeros(1,length(b)+1)]
                mass=exp((endpoint-lasttime)*augmented)*vcat(units.*state,1.)
                state=mass[1:end-1]./units
                lasttime=endpoint
            end
            push!(exact,copy(state))
        end
        @test maximum(maximum(abs,nominal.u[i]-exact[i]) for i in eachindex(exact))<5e-8
        mass_errors=[pb.p.rho*pb.p.number_scale*dot(pb.p.pivots,population(pb,u,:mill))+
            bulk(pb,u,:discharged_mass)-pb.p.mass_feed*t for (u,t) in zip(nominal.u,nominal.t)]
        @test maximum(abs,mass_errors)<1e-11
        uq=UQProblem(pb,s.inputs;setup=s.setup,solver=s.solver,times=s.times,outputs=(d50=s.d50,),
            output_coordinates=s.uq.output_coordinates)
        Core.eval(s,:(uq=$uq))
        Base.include(s,joinpath(dirname(CaseStudyScripts.source_path(:milling_uncertainty)),"milling_propagation.jl"))
        @test s.mc.metadata.solves==16 && all(s.mc.success)
        @test s.pce.training.metadata.solves==12 && all(s.pce.training.success)
        @test s.pce.validation!==nothing # its failed/pass diagnostic is retained
        @test s.pce.validation.passed isa Bool
        @test !isdefined(s,:cheap) # no automatic polynomial-draw campaign
    end
end
