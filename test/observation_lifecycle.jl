@testset "Timeless measurements and common observation lifecycle" begin
    g=Grid1D([0.,.2,.6,1.])
    op=compile(PBMModel(Birth((x,p,t,c)->2.)+Death((x,p,t,c)->.5)),g)
    steady=StationaryPBMProblem(op,ones(3);autonomous=true)
    state=reshape(4g.widths,:,1) # analytical birth/loss equilibrium
    @test observe(MomentObservable(0),steady,state)≈[4.]
    @test observe(HistogramObservable([0.,.3,1.]),steady,state)≈[1.2,2.8]
    residual=zero(state);stationary_residual!(residual,state,steady)
    @test maximum(abs,residual)<1e-14
    @test_throws NumericalFailure observe(MomentObservable(0),steady,-state)
    transient=PBMProblem(op,state,(0.,.1))
    solution=simulate(transient;dt=.01)
    @test observe(MomentObservable(0),solution)≈[4.]
    @test observe(MomentObservable(0),transient,state;time=.03)≈[4.]
    @test_throws BoundsError observe(MomentObservable(0),solution;index=100000)

    o=Observation([4.],MomentObservable(0);sigma=.1)
    m=Measurement(o)
    @test o.times===nothing && m.times===nothing
    @test size(o.data)==size(m.data)==(1,1)
    vector=Measurement([1.2,2.8],HistogramObservable([0.,.3,1.]);likelihood=GaussianLikelihood(sigma=.2))
    @test size(vector.data)==(2,1)
    replicates=Measurement([1.2 1.3;2.8 2.9],vector.observable;
        likelihood=GaussianLikelihood(covariance=Matrix{Float64}(I,4,4)))
    @test replicates.times===nothing && size(replicates.data)==(2,2)
    @test_throws DimensionMismatch Measurement([1.,2.],MomentObservable(0);likelihood=GaussianLikelihood(sigma=.1))
    @test_throws DimensionMismatch Measurement(zeros(2,0),vector.observable;likelihood=GaussianLikelihood(sigma=.1))
    @test_throws ArgumentError Measurement([.3],MomentObservable(0);likelihood=PoissonLikelihood(exposure=1.))
    @test_throws ArgumentError InferenceExperiment(:batch,transient,m;dt=.01)
    timed=Measurement([0.],[4.],MomentObservable(0);likelihood=GaussianLikelihood(sigma=.1))
    @test_throws ArgumentError InferenceExperiment(:equilibrium,steady,timed)
    @test_throws ArgumentError InferenceExperiment(:equilibrium,steady,m;dt=.01)
    @test_throws ArgumentError Experiment(transient,o)
    @test_throws ArgumentError Experiment(steady,Observation([0.],[4.],MomentObservable(0)))
    @test_throws ArgumentError StationarySolver(condition_limit=.5)
    experiment=InferenceExperiment(:equilibrium,steady,m)
    @test experiment.solver isa StationarySolver
    @test experiment.dt===nothing
    @test CalibrationProblem((Experiment(steady,o),)).dt===nothing
    @test_throws ArgumentError CalibrationProblem((Experiment(steady,o),);dt=.01)
    discrepancy=ResidualObservation([4.],MomentObservable(0);scale=.2,weight=2.)
    @test discrepancy.times===nothing && size(discrepancy.data)==(1,1)
    wasserstein=WassersteinObservation([0.,.3,1.],[0.,.3,1.],:crystals;
        window=(0.,1.),points=16)
    @test wasserstein.times===nothing
    @test size(wasserstein.data)==(16,1)

    for method in (QMOM(1),DQMOM(1))
        mo=compile(PBMModel(),MomentSet(0:1);method)
        values=method isa DQMOM ? quadrature_state(method,[3.],[2.]) : [2.,6.]
        mp=StationaryPBMProblem(mo,values;autonomous=true)
        @test observe(MomentObservable(0),mp,mp.guess)≈[2.]
        @test observe(MomentObservable(1),mp,mp.guess)≈[6.]
        @test_throws ArgumentError observe(HistogramObservable([0.,1.]),mp,mp.guess)
    end
end

@testset "Axial local and tensor observations share measurement operators" begin
    g=Grid1D([0.,.2,.6,1.]);z=AxialGrid([0.,.3,1.])
    domain=AxialDispersion(z;velocity=.5,dispersion=0.,inlet=(p,t)->2g.widths)
    model=PBMModel(Death((x,p,t,c)->.2))
    axial=compile(model,g;domain)
    values=hcat(2g.widths,3g.widths)
    problem=PBMProblem(axial,values,(0.,1.))
    @test observe(CellObservable(MomentObservable(0),2),problem,values;time=0.)≈[3.]
    @test observe(CellObservable(HistogramObservable([0.,.3,1.]),2),problem,values;time=0.)≈[.9,2.1]
    @test_throws ArgumentError CellObservable(MomentObservable(0),0)
    @test_throws ArgumentError observe(CellObservable(MomentObservable(0),3),problem,values;time=0.)
    batch=compile(model,g)
    @test_throws ArgumentError observe(CellObservable(MomentObservable(0),1),PBMProblem(batch,2g.widths,(0.,1.)),2g.widths;time=0.)
    sys=PBMSystem((crystals=batch,);bulk_rhs=(ctx,p,t)->(c=-.2ctx.bulk.c,))
    bp=PBMProblem(sys,(crystals=2g.widths,),(0.,1.);bulk0=(c=4.,))
    coupled=axial_problem(bp,AxialDispersion(z;velocity=.5,dispersion=0.,
        inlet=(p,t)->(populations=(crystals=2g.widths,),bulk=(c=4.,))))
    @test observe(CellObservable(PopulationMoment(:crystals,0),1),coupled,coupled.u0;time=0.)≈[2.]
    @test observe(CellObservable(BulkObservable(:c),2),coupled,coupled.u0;time=0.)≈[4.]
    @test observe(CellObservable(PopulationCDF(:crystals,[0.,.3,1.];window=(0.,1.)),2),coupled,coupled.u0;time=0.)≈[0.,.3,1.]
    grid=Grid2D(Grid1D([0.,.3,1.]),Grid1D([0.,.5,2.]))
    tensor=compile(PBMModel(),grid)
    initial=2grid.x.widths*grid.y.widths'
    tp=PBMProblem(tensor,initial,(0.,1.))
    @test observe(MomentObservable((0,0)),tp,initial;time=0.)≈[4.]
    @test observe(MomentObservable((1,0)),tp,initial;time=0.)≈[2.]
    @test_throws ArgumentError observe(HistogramObservable([0.,1.]),tp,initial;time=0.)

    uniform=Grid1D(range(0,1;length=9))
    hoc=compile(PBMModel(Growth(AffineGrowth(.2,0))),uniform;
        evolution=HOCFiniteVolume(),bc=SizeBC(Periodic(),Periodic()))
    hp=PBMProblem(hoc,2uniform.widths,(0.,.1))
    histogram=DistributionObservable([0.,.3,1.];coordinate=:volume)
    @test observe(histogram,hp,hp.u0;time=0.)≈[.6,1.4]
end
