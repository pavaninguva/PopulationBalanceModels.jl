using Test, PopulationBalanceModels, LinearAlgebra, Random
using Distributions: Normal, LogNormal, Uniform, Gamma, pdf
const PBMPlot=PopulationBalanceModels

function plotting_fixture()
    grid=Grid1D([0.,1.,3.,6.]);op=compile(PBMModel(),grid)
    pb=PBMProblem(op,[2.,8.,9.],(0.,1.);p=(k=.2,offset=1.))
    sol=simulate(pb;dt=.1,saveat=[0.,.5,1.])
    grid,pb,sol
end

@testset "Plot extraction uses public state/observation meaning without plotting dependencies" begin
    grid,pb,sol=plotting_fixture();original=deepcopy(sol.u)
    data=PBMPlot._distribution_data(sol;times=[0.,1.])
    @test data.edges==[0.,1.,3.,6.]
    @test data.values==[2. 4. 3.;2. 4. 3.]
    @test data.values[1,:]'*diff(data.edges)==19
    @test_throws ArgumentError PBMPlot._distribution_data(sol;times=[.25])
    @test sol.u==original
    stationary=StationaryPBMProblem(pb.operator,pb.u0;p=pb.p,autonomous=true)
    root=StationaryPBMSolution(copy(pb.u0),stationary,zero(pb.u0),zero(pb.u0),true,:analytical,
        nothing,(;analytical_fixture=true))
    @test PBMPlot._distribution_data(root).values==[2. 4. 3.]
    @test PBMPlot._observable_data(root,MomentObservable(0)).values==reshape([19.],1,1)
    @test_throws ArgumentError PBMPlot._distribution_data(root;times=[0.])
    o=MomentObservable(0)
    m=Measurement([0.,1.],[19.,19.],o;likelihood=GaussianLikelihood(sigma=[.2 .4]))
    d=PBMPlot._observable_data(sol,o;measurement=m)
    @test d.times==m.times && d.values==m.data && d.sigma==[.2 .4]
    C=[1. .7;.7 4.]
    measured=Observation([0.,1.],[19.,19.],o;covariance=C)
    @test PBMPlot._observable_data(measured).covariance≈C
    @test PBMPlot._observable_data(measured).data==measured.data

    basis=MomentSet(0:3);mop=compile(PBMModel(),basis;moment_method=QMOM())
    mp=PBMProblem(mop,[1.,1.,2.,6.],(0.,.1));ms=simulate(mp;dt=.01,saveat=[0.,.1])
    @test_throws ArgumentError PBMPlot._distribution_data(ms)
    @test PBMPlot._observable_data(ms,MomentObservable(2)).values==[2. 2.]
    reconstruction=(u,op,p,t)->(edges=[0.,1.],density=[u[1]])
    @test PBMPlot._distribution_data(ms;reconstruction).values==ones(2,1)
    direct=compile(PBMModel(),basis;moment_method=DQMOM())
    dp=PBMProblem(direct,InitialDistribution(Gamma(2.,.5);total=3.),(0.,.1))
    ds=simulate(dp;dt=.01,saveat=[0.,.1])
    @test PBMPlot._observable_data(ds,MomentObservable(2)).values≈[4.5 4.5]
    @test_throws ArgumentError PBMPlot._distribution_data(ds)
    if Base.get_extension(PBMPlot,:PopulationBalanceModelsPythonPlotExt)===nothing
        @test_throws ArgumentError plot_distribution(sol)
    end
end

@testset "Nonuniform tensor marginals, named populations and explicit field orientation" begin
    grid=Grid2D([0.,1.,3.],[0.,2.,5.]);N=[2. 3.;4. 5.]
    op=compile(PBMModel(),grid);pb=PBMProblem(op,N,(0.,1.));sol=simulate(pb;dt=.1,saveat=[0.,1.])
    @test_throws ArgumentError PBMPlot._distribution_data(sol)
    @test PBMPlot._distribution_data(sol;marginal=:x).values==[5. 4.5;5. 4.5]
    @test PBMPlot._distribution_data(sol;marginal=:y).values==[3. 8/3;3. 8/3]
    d=PBMPlot._field_data(sol;orientation=:internal_x)
    @test d.xedges==grid.x.edges && d.yedges==grid.y.edges
    @test d.values==N./([1.,2.]*[2.,3.]') && d.orientation=="rows_x"
    system=PBMSystem((cells=op,));sp=PBMProblem(system,(cells=N,),(0.,1.));ss=simulate(sp;dt=.1,saveat=[0.,1.])
    @test PBMPlot._field_data(ss;population=:cells,orientation=:internal_x).values==d.values
    @test PBMPlot._distribution_data(ss;population=:cells,marginal=:x).values==[5. 4.5;5. 4.5]
    @test_throws ArgumentError PBMPlot._distribution_data(ss;marginal=:x)

    g=Grid1D([0.,1.,3.]);localop=compile(PBMModel(),g)
    domain=AxialDispersion(AxialGrid([0.,.2,.7,1.]);velocity=.1,dispersion=0.)
    # Constructors/accessors preserve concentrations, not axial-cell inventories.
    axop=compile(PBMModel(),g;domain)
    initial=[2. 3. 4.;6. 8. 10.];ap=PBMProblem(axop,initial,(0.,1.))
    a=PBMSolution([0.],[copy(ap.u0)],ap,0)
    field=PBMPlot._field_data(a;orientation=:axial_x)
    @test field.xedges==domain.grid.edges
    @test field.values==initial./[1.,2.]
    @test_throws ArgumentError PBMPlot._field_data(a;orientation=:internal_x)
    @test PBMPlot._distribution_data(a;cell=2).values==[3. 4.]
end

function plotting_estimation_fixture()
    g,pb,_=plotting_fixture();calls=Ref(0)
    solver=(pb,ts)->begin calls[]+=1;simulate(pb;solver=ForwardSolver(SSPRK33();dt=.1),saveat=ts) end
    measurement=Measurement([0.,1.],[19.,19.],MomentObservable(0);likelihood=GaussianLikelihood(sigma=.5))
    ex=InferenceExperiment(:one,pb,measurement;solver)
    parameters=[ParameterSpec(:k;initial=.2,prior=LogNormal(log(.2),.5),transform=LogTransform(scale=2.)),
        ParameterSpec(:offset;initial=1.,prior=Normal(1.,.1),transform=IdentityTransform(location=.2,scale=2.),role=:nuisance)]
    ep=EstimationProblem(parameters,(ex,));z=initial_coordinates(ep)
    # Analytical mode of these independent inference-coordinate Normal laws.
    map=MAPResult(z,physical_parameters(ep,z),loglikelihood(ep,z),logprior(ep,z),logposterior(ep,z),true,
        (;analytical_fixture=true),ParameterSpec[parameters...],nothing;convention=:inference_map_v2)
    ep,map,calls
end

@testset "Prepared fit blocks and recorded parameter coordinates" begin
    ep,map,calls=plotting_estimation_fixture();P=predictions(ep,map.minimizer);before=calls[]
    data=PBMPlot._fit_data(ep,P)
    @test calls[]==before && length(data)==1
    @test data[1].times==[0.,1.] && data[1].values==[19. 19.]
    @test data[1].sigma==.5
    profile=FunctionObservable((N,op,p,t)->[sum(N),2sum(N)])
    C=diagm([1.,4.,9.,16.])
    m=Measurement([0.,1.],[19. 19.;38. 38.],profile;likelihood=GaussianLikelihood(covariance=C))
    ex=InferenceExperiment(:profile,only(ep.experiments).problem,m;dt=.1)
    profile_problem=EstimationProblem(ep.parameters,(ex,))
    plotted=only(PBMPlot._fit_data(profile_problem,([copy(m.data)],);abscissa=[.1,.2]))
    @test plotted.values==permutedims(m.data)
    @test diag(plotted.covariance)==[1.,9.,4.,16.]
    @test plotted.times==[.1,.2]
    # Expected count conversion must not confuse a population measure with
    # observed counts/exposure. These values use the likelihood's own contract.
    for (likelihood,expected) in ((PoissonLikelihood(exposure=2.),[4.,6.]),
                                  (MultinomialLikelihood(),[4.,6.]))
        obs=HistogramObservable([0.,1.,3.])
        m=Measurement([0.],[4.,6.].*ones(1,1),obs;likelihood)
        ex=InferenceExperiment(:counts,only(ep.experiments).problem,m;dt=.1)
        countproblem=EstimationProblem(ep.parameters,(ex,))
        block=only(PBMPlot._fit_data(countproblem,([reshape([2.,3.],:,1)],)))
        @test vec(block.values)≈expected
    end
    @test_throws DimensionMismatch PBMPlot._fit_data(ep,())
    @test_throws ArgumentError PBMPlot._fit_data(ep,P;bands=([([18. 18.],[20. 20.])],))
    approximation=laplace_approximation(ep,map)
    @test approximation.covariance≈diagm([.25,.0025]) rtol=1e-10
    before=calls[]
    d=PBMPlot._parameter_data(ep,approximation;coordinates=:physical)
    @test d.gaussian.coordinates=="inference"
    @test d.point≈[.2,1.] && d.metadata[1].scale==2.
    @test d.priors[1].density≈pdf.(LogNormal(log(.2),.5),d.priors[1].x)
    zdata=PBMPlot._parameter_data(ep,approximation;coordinates=:inference)
    @test zdata.priors[1].density≈pdf.(Normal(log(.1),.5),zdata.priors[1].x)
    draws=repeat(reshape(map.minimizer,:,1,1),1,5,2)
    draws[1,:,:].+=reshape(range(-.2,.2;length=10),5,2)
    samples=PosteriorSamples(draws,nothing,(;passed=false),ep.parameters)
    result=PBMPlot._parameter_data(ep,samples;coordinates=:physical)
    @test size(result.samples)==(10,2)
    @test result.samples[:,1]≈2exp.(vec(draws[1,:,:]))
    @test result.diagnostics.passed===false && calls[]==before
    @test PBMPlot._parameter_data(ep,map).point_only
end

@testset "UQ extraction preserves probability meaning and failed validation" begin
    g,pb,sol=plotting_fixture()
    inputs=UncertainInputs((k=Uniform(.1,.3),))
    uq=UQProblem(pb,inputs;outputs=(number=MomentObservable(0),),times=[0.,1.],dt=.1)
    mc=propagate(uq,MonteCarlo(5;seed=3);max_solves=5)
    original=copy(mc.values);d=PBMPlot._uq_data(mc;output=:number)
    @test d.mean==[19. 19.]
    @test size(d.quantiles)==(1,2,2) && all(==(19.),d.quantiles)
    @test mc.values==original && d.diagnostics.failed==0
    @test_throws ArgumentError PBMPlot._uq_data(mc;output=:missing)
    summary=uq_summary(mc;covariance=false)
    @test PBMPlot._uq_data(summary;output=:number).mean==d.mean
end
